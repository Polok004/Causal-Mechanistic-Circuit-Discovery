"""Causal path patching over SAE features in ESM-2.

The intervention
----------------
Given a wild-type sequence (*clean*) and a point mutant (*corrupted*), we ask:
which SAE features at which layer and which residue positions, if restored to
their clean values inside the corrupted forward pass, recover the model's
wild-type behaviour? A feature that recovers a lot is causally responsible for
the model's response to that mutation; a feature that correlates with the
mutation but recovers nothing is a bystander. This is the distinction that
attention maps and post-hoc saliency cannot make.

Error-preserving splicing (the part that is easy to get wrong)
--------------------------------------------------------------
The naive patch replaces the layer's activation with the SAE's decode of the
edited latents::

    a_new = decode(z_corrupt with feature f restored)

That is wrong, and wrong in a way that inflates every number. The SAE does not
reconstruct ``a`` perfectly, so this substitution changes the activation by

    (patch effect) + (SAE reconstruction error)

and the reconstruction error term is typically *larger* than the patch effect
for a single feature. What we do instead is add only the difference::

    a_new = a_corrupt + (decode(z_patched) - decode(z_corrupt))

The reconstruction error is identical in both decodes and cancels exactly. The
measured effect is then attributable to feature ``f`` alone. Set
``splice_mode="replace"`` to reproduce the naive behaviour for comparison — the
benchmark script reports both, because the gap is worth showing.

Baseline ordering
-----------------
The corrupted baseline must be measured with **no hook attached**. Computing it
inside the patched context — a common bug, since the hook is registered on the
module and fires for every forward pass until removed — yields a baseline equal
to the patched run and therefore a direct effect of exactly zero. Every hook
here is scoped by a context manager and the baselines are cached before any
hook is registered.

Metric sign convention
----------------------
All metrics are defined so that **higher means more wild-type-like**. The direct
causal effect is then ``metric(patched) - metric(corrupted)``, positive when the
patch restores wild-type behaviour. ``normalized_dce`` divides by the full
clean-minus-corrupted gap, so 1.0 means the single feature recovered the whole
effect of the mutation and 0.0 means it recovered none of it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from tqdm.auto import tqdm

from models.esm_hooks import (
    ESMActivationExtractor,
    ESMWrapper,
    TokenBatch,
    make_patch_hook,
    resolve_encoder_layers,
)
from models.sparse_autoencoder import TopKSparseAutoencoder
from utils.protein import Mutation, parse_mutation, seq_to_token_pos, validate_sequence

__all__ = [
    "CausalPatcher",
    "LogitDiffMetric",
    "Metric",
    "PatchResult",
    "PatchingSetup",
    "attribution_scores",
]

SpliceMode = Literal["error_preserving", "replace"]


# --------------------------------------------------------------------------- #
# Metrics                                                                       #
# --------------------------------------------------------------------------- #


class Metric:
    """Maps masked-LM logits to a scalar, higher = more wild-type-like."""

    name: str = "metric"

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class LogitDiffMetric(Metric):
    """``log p(wt_aa) - log p(mut_aa)`` at the mutated position.

    This is the standard logit-difference readout adapted to variant effects. It
    is high when the model still "believes" the wild-type residue belongs at
    that site and low when the mutant context has convinced it otherwise, so it
    tracks exactly the quantity a loss-of-function mutation is supposed to move.

    Using a *difference* of two logits rather than a single probability matters:
    it is invariant to the softmax normaliser, so a patch that uniformly raises
    or lowers confidence at that position does not register as an effect.
    """

    token_pos: int
    wt_token_id: int
    mut_token_id: int
    batch_index: int = 0
    name: str = "logit_diff"

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log_softmax(logits[self.batch_index, self.token_pos], dim=-1)
        return log_probs[self.wt_token_id] - log_probs[self.mut_token_id]


@dataclass
class SequenceLogLikelihoodMetric(Metric):
    """Mean log-likelihood the model assigns to the residues actually present.

    A coarser, position-agnostic readout. Useful as a robustness check: a
    circuit that only moves :class:`LogitDiffMetric` at the mutated site but
    never shifts the model's overall view of the sequence is a narrower claim
    than one that moves both.
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    batch_index: int = 0
    name: str = "seq_loglik"

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        lp = torch.log_softmax(logits[self.batch_index], dim=-1)
        ids = self.input_ids[self.batch_index]
        mask = self.attention_mask[self.batch_index].bool()
        token_lp = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return token_lp[mask].mean()


# --------------------------------------------------------------------------- #
# Setup and results                                                             #
# --------------------------------------------------------------------------- #


@dataclass
class PatchingSetup:
    """A clean/corrupted pair plus the readout, built once and reused.

    Building this is the only place mutation strings are converted to token
    positions, so the 1-indexed/0-indexed/token-offset conversions happen once
    and are unit-tested in one place.
    """

    clean: TokenBatch
    corrupted: TokenBatch
    metric: Metric
    mutation: Mutation
    token_pos: int

    @classmethod
    def from_mutation(
        cls,
        wrapper: ESMWrapper,
        wt_sequence: str,
        mutation: str | Mutation,
        *,
        metric: Metric | None = None,
    ) -> PatchingSetup:
        wt = validate_sequence(wt_sequence)
        mut = parse_mutation(mutation) if isinstance(mutation, str) else mutation
        mutant_seq = mut.apply(wt)

        clean = wrapper.tokenize(wt)
        corrupted = wrapper.tokenize(mutant_seq)
        if clean.input_ids.shape != corrupted.input_ids.shape:
            raise ValueError(
                "clean and corrupted tokenisations differ in shape; a single "
                "substitution must not change sequence length"
            )

        token_pos = seq_to_token_pos(mut.seq_pos)
        if token_pos >= clean.input_ids.shape[1]:
            raise IndexError(
                f"mutation {mut.raw} is at token {token_pos} but the tokenised sequence "
                f"has only {clean.input_ids.shape[1]} positions — it was truncated at "
                f"max_seq_len={wrapper.max_seq_len}"
            )

        if metric is None:
            metric = LogitDiffMetric(
                token_pos=token_pos,
                wt_token_id=wrapper.aa_token_id(mut.wt_aa),
                mut_token_id=wrapper.aa_token_id(mut.mut_aa),
            )
        return cls(
            clean=clean, corrupted=corrupted, metric=metric, mutation=mut, token_pos=token_pos
        )

    @property
    def seq_len(self) -> int:
        return int(self.clean.input_ids.shape[1])


@dataclass
class PatchResult:
    """Outcome of one intervention."""

    layer_idx: int
    feature_idx: int
    positions: tuple[int, ...]
    clean_metric: float
    corrupted_metric: float
    patched_metric: float

    @property
    def dce(self) -> float:
        """Direct causal effect: how far the patch moved the metric."""
        return self.patched_metric - self.corrupted_metric

    @property
    def normalized_dce(self) -> float:
        """DCE as a fraction of the full clean-corrupted gap.

        1.0 = this one feature explains the entire behavioural difference.
        Returns 0.0 when the mutation had no measurable effect to begin with,
        since the ratio is undefined there and reporting a large number from a
        near-zero denominator would be worse than reporting nothing.
        """
        gap = self.clean_metric - self.corrupted_metric
        if abs(gap) < 1e-8:
            return 0.0
        return self.dce / gap

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer_idx,
            "feature": self.feature_idx,
            "positions": list(self.positions),
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "patched_metric": self.patched_metric,
            "dce": self.dce,
            "normalized_dce": self.normalized_dce,
        }


# --------------------------------------------------------------------------- #
# The patcher                                                                   #
# --------------------------------------------------------------------------- #


class CausalPatcher:
    """Runs SAE-feature interventions on an ESM-2 forward pass.

    Args:
        wrapper: the model under study.
        sae: an SAE trained on ``layer_idx`` of that model.
        layer_idx: residual-stream layer to intervene on.
        splice_mode: ``"error_preserving"`` (default, correct) or ``"replace"``
            (naive, for comparison).
    """

    def __init__(
        self,
        wrapper: ESMWrapper,
        sae: TopKSparseAutoencoder,
        layer_idx: int,
        *,
        splice_mode: SpliceMode = "error_preserving",
    ) -> None:
        self.wrapper = wrapper
        self.model = wrapper.model
        self.sae = sae.to(wrapper.device).eval()
        self.layer_idx = layer_idx
        self.splice_mode = splice_mode

        n = len(resolve_encoder_layers(self.model))
        if not 0 <= layer_idx < n:
            raise IndexError(f"layer {layer_idx} outside range 0..{n - 1}")
        if sae.d_in != wrapper.d_model:
            raise ValueError(
                f"SAE expects d_in={sae.d_in} but the model's residual width is "
                f"{wrapper.d_model}. This SAE was trained on a different checkpoint."
            )
        if sae.cfg.layer_idx >= 0 and sae.cfg.layer_idx != layer_idx:
            raise ValueError(
                f"SAE was trained on layer {sae.cfg.layer_idx}, cannot patch layer {layer_idx}"
            )

    # -- plain forward passes --------------------------------------------------

    @torch.no_grad()
    def run_metric(self, batch: TokenBatch, metric: Metric) -> float:
        """Forward pass with no hooks, returning the scalar metric."""
        logits = self.model(**batch.as_model_kwargs()).logits
        return float(metric(logits))

    @torch.no_grad()
    def sae_latents(self, batch: TokenBatch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(activations, latents)`` at the patched layer.

        ``activations`` is ``[batch, seq, d_model]``; ``latents`` is
        ``[batch, seq, d_sae]``.
        """
        extractor = ESMActivationExtractor(self.model, [self.layer_idx])
        acts = extractor.capture(batch)[self.layer_idx]
        acts = self._scale(acts)
        b, s, d = acts.shape
        out = self.sae(acts.reshape(-1, d))
        return acts, out.latents.reshape(b, s, -1)

    def _scale(self, acts: torch.Tensor) -> torch.Tensor:
        """Apply the activation normalisation the SAE was trained with."""
        scale = getattr(self.sae.cfg, "activation_scale", 1.0) or 1.0
        return acts * scale if scale != 1.0 else acts

    def _unscale(self, acts: torch.Tensor) -> torch.Tensor:
        scale = getattr(self.sae.cfg, "activation_scale", 1.0) or 1.0
        return acts / scale if scale != 1.0 else acts

    # -- the intervention ------------------------------------------------------

    def _build_transform(
        self,
        clean_latents: torch.Tensor,
        feature_idx: int | Sequence[int],
        positions: Sequence[int] | None,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Make the residual-stream transform implementing one patch."""
        features = (
            [int(feature_idx)] if isinstance(feature_idx, int) else [int(f) for f in feature_idx]
        )
        feat_index = torch.as_tensor(features, dtype=torch.long, device=clean_latents.device)

        def transform(hidden: torch.Tensor) -> torch.Tensor:
            scaled = self._scale(hidden)
            b, s, d = scaled.shape
            flat = scaled.reshape(-1, d)
            out = self.sae(flat)
            z = out.latents.reshape(b, s, -1)

            z_patched = z.clone()
            pos = (
                torch.arange(s, device=z.device)
                if positions is None
                else torch.as_tensor(list(positions), dtype=torch.long, device=z.device)
            )
            # index_put over (positions x features) — the clean values overwrite
            # the corrupted ones only at the chosen sites.
            z_patched[:, pos[:, None], feat_index[None, :]] = clean_latents[
                :, pos[:, None], feat_index[None, :]
            ].to(z_patched.dtype)

            recon_patched = self.sae.decode(z_patched.reshape(-1, z.shape[-1])).reshape(b, s, d)

            if self.splice_mode == "replace":
                return self._unscale(recon_patched)

            # Error-preserving: add only the change, so SAE reconstruction error
            # cancels between the two decodes.
            recon_original = out.recon.reshape(b, s, d)
            return self._unscale(scaled + (recon_patched - recon_original))

        return transform

    @contextlib.contextmanager
    def _patched(
        self,
        clean_latents: torch.Tensor,
        feature_idx: int | Sequence[int],
        positions: Sequence[int] | None,
    ) -> Iterator[None]:
        layers = resolve_encoder_layers(self.model)
        transform = self._build_transform(clean_latents, feature_idx, positions)
        handle = layers[self.layer_idx].register_forward_hook(make_patch_hook(transform))
        try:
            yield
        finally:
            handle.remove()

    @torch.no_grad()
    def direct_causal_effect(
        self,
        setup: PatchingSetup,
        feature_idx: int | Sequence[int],
        *,
        positions: Sequence[int] | None = None,
        cache: dict[str, Any] | None = None,
    ) -> PatchResult:
        """Measure the effect of restoring clean feature values in the mutant run.

        Args:
            setup: the clean/corrupted pair and metric.
            feature_idx: one SAE feature, or several patched jointly.
            positions: token positions to patch. ``None`` patches every
                position, which answers "does this feature matter anywhere?";
                passing the mutated site alone answers the sharper question of
                whether it matters *there*.
            cache: optional dict reused across calls to avoid recomputing the
                clean latents and the two baselines. :meth:`sweep_features`
                builds one for you.
        """
        cache = self.baselines(setup) if cache is None else cache
        clean_latents = cache["clean_latents"]

        with self._patched(clean_latents, feature_idx, positions):
            patched_metric = float(
                setup.metric(self.model(**setup.corrupted.as_model_kwargs()).logits)
            )

        pos_tuple = (
            tuple(range(setup.seq_len)) if positions is None else tuple(int(p) for p in positions)
        )
        if isinstance(feature_idx, int):
            first_feature = feature_idx
        else:
            # An empty feature set is a legitimate call: it is the no-op control
            # that verifies error-preserving splicing cancels exactly. Report -1
            # rather than indexing into an empty list.
            features = list(feature_idx)
            first_feature = int(features[0]) if features else -1
        return PatchResult(
            layer_idx=self.layer_idx,
            feature_idx=first_feature,
            positions=pos_tuple,
            clean_metric=cache["clean_metric"],
            corrupted_metric=cache["corrupted_metric"],
            patched_metric=patched_metric,
        )

    @torch.no_grad()
    def baselines(self, setup: PatchingSetup) -> dict[str, Any]:
        """Compute clean/corrupted metrics and clean latents, with no hooks live.

        Ordering is the point: these run before any hook is registered, so the
        corrupted baseline is a genuinely unpatched forward pass.
        """
        clean_metric = self.run_metric(setup.clean, setup.metric)
        corrupted_metric = self.run_metric(setup.corrupted, setup.metric)
        _, clean_latents = self.sae_latents(setup.clean)
        _, corrupted_latents = self.sae_latents(setup.corrupted)
        return {
            "clean_metric": clean_metric,
            "corrupted_metric": corrupted_metric,
            "clean_latents": clean_latents,
            "corrupted_latents": corrupted_latents,
        }

    # -- sweeps ----------------------------------------------------------------

    @torch.no_grad()
    def sweep_features(
        self,
        setup: PatchingSetup,
        features: Iterable[int],
        *,
        positions: Sequence[int] | None = None,
        progress: bool = True,
    ) -> list[PatchResult]:
        """Exact DCE for each feature in turn, sorted by descending |DCE|.

        One forward pass per feature. On the 8M model with a few hundred
        candidate features this is seconds; over a full 5120-feature dictionary
        it is not, which is what :func:`attribution_scores` is for.
        """
        cache = self.baselines(setup)
        feats = list(features)
        results = []
        for f in tqdm(feats, disable=not progress, desc=f"DCE sweep L{self.layer_idx}"):
            results.append(
                self.direct_causal_effect(setup, f, positions=positions, cache=cache)
            )
        results.sort(key=lambda r: abs(r.dce), reverse=True)
        return results

    @torch.no_grad()
    def active_features(
        self, setup: PatchingSetup, *, positions: Sequence[int] | None = None
    ) -> list[int]:
        """Features that fire in either run — the only ones worth sweeping.

        A feature inactive in both the clean and corrupted pass has identical
        (zero) values in both, so patching it is a no-op by construction. With
        k=32 and d_sae=5120 this prunes the candidate set by two orders of
        magnitude before any exact patching happens.
        """
        cache = self.baselines(setup)
        clean, corrupt = cache["clean_latents"], cache["corrupted_latents"]
        if positions is not None:
            idx = torch.as_tensor(list(positions), dtype=torch.long, device=clean.device)
            clean, corrupt = clean[:, idx], corrupt[:, idx]
        # amax over (batch, seq) rather than .any(dim=(0, 1)): tuple dims are
        # accepted by amax across torch versions, by any() only in recent ones.
        active = (clean.abs().amax(dim=(0, 1)) > 0) | (corrupt.abs().amax(dim=(0, 1)) > 0)
        return active.nonzero(as_tuple=True)[0].tolist()

    # -- ablation (for faithfulness) -------------------------------------------

    @torch.no_grad()
    def ablate_features(
        self,
        batch: TokenBatch,
        metric: Metric,
        features: Sequence[int],
        *,
        positions: Sequence[int] | None = None,
        mode: Literal["zero", "mean"] = "zero",
        mean_latents: torch.Tensor | None = None,
    ) -> float:
        """Zero (or mean-) ablate features and return the resulting metric.

        Mean ablation is the better control of the two: zeroing a feature moves
        the activation off the data manifold, so some of the measured damage is
        the model reacting to an impossible input rather than to the missing
        feature. Mean ablation replaces it with its corpus average instead.
        """
        if mode == "mean" and mean_latents is None:
            raise ValueError("mode='mean' requires mean_latents")

        feat_index = torch.as_tensor(list(features), dtype=torch.long, device=self.wrapper.device)

        def transform(hidden: torch.Tensor) -> torch.Tensor:
            scaled = self._scale(hidden)
            b, s, d = scaled.shape
            out = self.sae(scaled.reshape(-1, d))
            z = out.latents.reshape(b, s, -1)
            z_abl = z.clone()
            pos = (
                torch.arange(s, device=z.device)
                if positions is None
                else torch.as_tensor(list(positions), dtype=torch.long, device=z.device)
            )
            if mode == "zero":
                z_abl[:, pos[:, None], feat_index[None, :]] = 0.0
            else:
                repl = mean_latents.to(z.device)[feat_index]  # [n_feat]
                z_abl[:, pos[:, None], feat_index[None, :]] = repl[None, None, :]

            recon_abl = self.sae.decode(z_abl.reshape(-1, z.shape[-1])).reshape(b, s, d)
            if self.splice_mode == "replace":
                return self._unscale(recon_abl)
            return self._unscale(scaled + (recon_abl - out.recon.reshape(b, s, d)))

        layers = resolve_encoder_layers(self.model)
        handle = layers[self.layer_idx].register_forward_hook(make_patch_hook(transform))
        try:
            return float(metric(self.model(**batch.as_model_kwargs()).logits))
        finally:
            handle.remove()


# --------------------------------------------------------------------------- #
# Attribution patching (linear approximation, for prescreening)                 #
# --------------------------------------------------------------------------- #


def attribution_scores(
    patcher: CausalPatcher,
    setup: PatchingSetup,
    *,
    positions: Sequence[int] | None = None,
) -> torch.Tensor:
    """First-order estimate of every feature's DCE in **one** backward pass.

    Exact patching costs one forward pass per feature, which is fine for
    hundreds of features and hopeless for a 5120-element dictionary across 6
    layers and hundreds of variants. The standard remedy is attribution
    patching: approximate the effect of moving latent ``z_f`` from its corrupted
    to its clean value by the first-order term

        DCE_f  ~  (z_clean,f - z_corrupt,f) . dmetric/dz_f

    evaluated at the corrupted activations. One backward pass scores the whole
    dictionary at once.

    This is an approximation and it is *not* the number to report. It is
    reliable for ranking and unreliable in magnitude — in particular it is known
    to be poor exactly where the model's response saturates, which is where
    interesting circuit behaviour often lives. Use it to shortlist, then confirm
    the shortlist with :meth:`CausalPatcher.sweep_features`. The pipeline in
    ``scripts/discover_circuits.py`` does exactly that.

    Returns:
        ``[d_sae]`` tensor of attribution scores, on CPU.
    """
    model, sae = patcher.model, patcher.sae
    layers = resolve_encoder_layers(model)

    with torch.no_grad():
        _, clean_latents = patcher.sae_latents(setup.clean)

    # Capture corrupted latents as a differentiable leaf, then splice with the
    # same error-preserving rule so the gradient corresponds to the intervention
    # we would actually perform.
    holder: dict[str, torch.Tensor] = {}

    def transform(hidden: torch.Tensor) -> torch.Tensor:
        scaled = patcher._scale(hidden)
        b, s, d = scaled.shape
        out = sae(scaled.reshape(-1, d))
        z = out.latents.reshape(b, s, -1).detach().requires_grad_(True)
        holder["z"] = z
        recon = sae.decode(z.reshape(-1, z.shape[-1])).reshape(b, s, d)
        recon_ref = out.recon.reshape(b, s, d).detach()
        if patcher.splice_mode == "replace":
            return patcher._unscale(recon)
        return patcher._unscale(scaled.detach() + (recon - recon_ref))

    was_training = model.training
    model.eval()
    handle = layers[patcher.layer_idx].register_forward_hook(make_patch_hook(transform))
    try:
        with torch.enable_grad():
            logits = model(**setup.corrupted.as_model_kwargs()).logits
            value = setup.metric(logits)
            (grad,) = torch.autograd.grad(value, holder["z"])
    finally:
        handle.remove()
        if was_training:
            model.train()

    delta = (clean_latents - holder["z"].detach()).to(grad.dtype)
    contrib = delta * grad  # [batch, seq, d_sae]

    if positions is not None:
        idx = torch.as_tensor(list(positions), dtype=torch.long, device=contrib.device)
        contrib = contrib[:, idx]

    return contrib.sum(dim=(0, 1)).detach().to("cpu")


@torch.no_grad()
def positionwise_effect(
    patcher: CausalPatcher,
    setup: PatchingSetup,
    feature_idx: int,
    *,
    cache: dict[str, Any] | None = None,
    progress: bool = False,
) -> torch.Tensor:
    """DCE of one feature patched at each position separately.

    This is what turns a feature into a set of *residues*: a feature can matter
    a great deal in aggregate while the effect is concentrated at three sites.
    Those sites are what the biophysical validator checks against the structure.

    Returns:
        ``[seq_len]`` tensor of per-position DCE.
    """
    cache = patcher.baselines(setup) if cache is None else cache
    seq_len = setup.seq_len
    effects = torch.zeros(seq_len)
    for pos in tqdm(range(seq_len), disable=not progress, desc=f"positions f{feature_idx}"):
        res = patcher.direct_causal_effect(setup, feature_idx, positions=[pos], cache=cache)
        effects[pos] = res.dce
    return effects
