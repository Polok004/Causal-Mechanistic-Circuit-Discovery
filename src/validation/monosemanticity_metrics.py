"""Measuring whether SAE features are worth interpreting.

Two separate questions get conflated in SAE papers and are kept apart here.

**Is the dictionary any good?** :func:`evaluate_reconstruction` reports fraction
of variance unexplained, L0, the dead-feature fraction, and — the one that
actually matters downstream — *cross-entropy recovered*: how much of the model's
language-modelling performance survives when its residual stream is replaced by
the SAE's reconstruction. A dictionary can have excellent MSE and still destroy
the model's behaviour, because MSE weights every direction equally while the
model does not. If you report one number, report this one.

**Is a given feature interpretable?** :func:`feature_purity` scores how
concentrated a feature's top-activating tokens are over some labelling —
amino-acid identity, or DSSP secondary structure when available. A feature that
fires on 90% histidines is monosemantic with respect to residue identity; one
whose top tokens are uniformly spread is not, at least not along that axis. The
"at least not along that axis" caveat is load-bearing: low purity over amino
acids is evidence of nothing on its own, since the interesting features are
precisely the ones that track structure or function rather than identity.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from models.sparse_autoencoder import TopKSparseAutoencoder
from utils.protein import AA_ALPHABET

__all__ = [
    "FeatureProfile",
    "ReconstructionMetrics",
    "cross_entropy_recovered",
    "evaluate_reconstruction",
    "feature_purity",
    "normalized_entropy",
    "profile_features",
]


@dataclass
class ReconstructionMetrics:
    """Dictionary-level quality numbers. This is Table 1 of the paper."""

    fvu: float
    mse: float
    l0: float
    dead_fraction: float
    explained_variance: float
    cosine_similarity: float
    n_tokens: int
    ce_recovered: float | None = None
    layer_idx: int = -1

    def as_dict(self) -> dict[str, Any]:
        d = {
            "layer": self.layer_idx,
            "fvu": self.fvu,
            "mse": self.mse,
            "l0": self.l0,
            "dead_fraction": self.dead_fraction,
            "explained_variance": self.explained_variance,
            "cosine_similarity": self.cosine_similarity,
            "n_tokens": self.n_tokens,
        }
        if self.ce_recovered is not None:
            d["ce_recovered"] = self.ce_recovered
        return d

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        ce = f"  ce_recovered={self.ce_recovered:.3f}" if self.ce_recovered is not None else ""
        return (
            f"L{self.layer_idx}  FVU={self.fvu:.4f}  EV={self.explained_variance:.4f}  "
            f"L0={self.l0:.1f}  dead={self.dead_fraction:.1%}  cos={self.cosine_similarity:.4f}{ce}"
        )


@torch.no_grad()
def evaluate_reconstruction(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    *,
    layer_idx: int = -1,
    batch_size: int = 8192,
) -> ReconstructionMetrics:
    """Reconstruction quality over held-out activations.

    Args:
        activations: ``[n_tokens, d_in]``, already scaled the way the SAE was
            trained (``SAEConfig.activation_scale``).
    """
    if activations.dim() != 2:
        raise ValueError(f"expected [n_tokens, d_in], got {tuple(activations.shape)}")

    sae.eval()
    device = sae.device
    n = activations.shape[0]

    sq_err = 0.0
    cos_sum = 0.0
    l0_sum = 0.0
    seen = 0
    # Two-pass variance would need the whole tensor resident; accumulate the
    # sums instead so this works on a corpus larger than memory.
    sum_x = torch.zeros(sae.d_in, dtype=torch.float64)
    sum_x2 = 0.0

    for start in range(0, n, batch_size):
        x = activations[start : start + batch_size].to(device, dtype=sae.dtype)
        out = sae(x)
        err = x - out.recon
        sq_err += float(err.pow(2).sum())
        cos_sum += float(
            torch.nn.functional.cosine_similarity(x, out.recon, dim=-1).sum()
        )
        l0_sum += float((out.latents != 0).sum(dim=-1).float().sum())
        sum_x += x.sum(dim=0).double().cpu()
        sum_x2 += float(x.double().pow(2).sum())
        seen += x.shape[0]

    mean_x = sum_x / seen
    total_var = sum_x2 - seen * float((mean_x**2).sum())
    fvu = sq_err / max(total_var, 1e-12)

    return ReconstructionMetrics(
        fvu=fvu,
        mse=sq_err / seen,
        l0=l0_sum / seen,
        dead_fraction=sae.dead_fraction(),
        explained_variance=1.0 - fvu,
        cosine_similarity=cos_sum / seen,
        n_tokens=seen,
        layer_idx=layer_idx,
    )


@torch.no_grad()
def cross_entropy_recovered(
    wrapper: Any,
    sae: TopKSparseAutoencoder,
    layer_idx: int,
    batches: Iterable[Any],
    *,
    max_batches: int = 16,
) -> float:
    """Fraction of the model's masked-LM performance that survives the SAE.

    Defined as ``(CE_ablated - CE_sae) / (CE_ablated - CE_clean)``, where
    ``CE_ablated`` replaces the layer's activation with its batch mean. 1.0 means
    the SAE reconstruction is behaviourally free; 0.0 means it is as damaging as
    deleting the layer's contribution entirely.

    This is the metric that catches a dictionary with good MSE and bad
    behaviour, which is a real and common failure: the residual stream has a few
    very high-variance directions that dominate MSE, and an SAE can spend its
    whole capacity on them while discarding the low-variance directions the
    model's computation actually reads.
    """
    from models.esm_hooks import make_patch_hook, resolve_encoder_layers

    layers = resolve_encoder_layers(wrapper.model)
    scale = getattr(sae.cfg, "activation_scale", 1.0) or 1.0

    def ce_of(batch: Any) -> float:
        logits = wrapper.model(**batch.as_model_kwargs()).logits
        lp = torch.log_softmax(logits, dim=-1)
        ids = batch.input_ids
        mask = batch.attention_mask.bool()
        token_lp = lp.gather(-1, ids.unsqueeze(-1)).squeeze(-1)
        return float(-token_lp[mask].mean())

    def run_with(transform: Any, batch: Any) -> float:
        handle = layers[layer_idx].register_forward_hook(make_patch_hook(transform))
        try:
            return ce_of(batch)
        finally:
            handle.remove()

    def sae_transform(hidden: torch.Tensor) -> torch.Tensor:
        b, s, d = hidden.shape
        recon = sae(hidden.reshape(-1, d) * scale).recon.reshape(b, s, d)
        return recon / scale

    def mean_transform(hidden: torch.Tensor) -> torch.Tensor:
        return hidden.mean(dim=(0, 1), keepdim=True).expand_as(hidden)

    clean_ce, sae_ce, abl_ce, n = 0.0, 0.0, 0.0, 0
    for i, batch in enumerate(batches):
        if i >= max_batches:
            break
        clean_ce += ce_of(batch)
        sae_ce += run_with(sae_transform, batch)
        abl_ce += run_with(mean_transform, batch)
        n += 1

    if n == 0:
        return float("nan")
    clean_ce, sae_ce, abl_ce = clean_ce / n, sae_ce / n, abl_ce / n
    denom = abl_ce - clean_ce
    if abs(denom) < 1e-8:
        return float("nan")
    return (abl_ce - sae_ce) / denom


# --------------------------------------------------------------------------- #
# Feature-level interpretability                                                #
# --------------------------------------------------------------------------- #


@dataclass
class FeatureProfile:
    """What one SAE feature fires on."""

    feature_idx: int
    layer_idx: int
    firing_rate: float
    max_activation: float
    mean_activation: float
    # Top-activating tokens, as (sequence_index, position, residue, activation).
    top_tokens: list[tuple[int, int, str, float]] = field(default_factory=list)
    aa_distribution: dict[str, float] = field(default_factory=dict)
    purity: float = 0.0
    dominant_label: str = ""
    entropy: float = 0.0

    @property
    def is_dead(self) -> bool:
        return self.firing_rate == 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature_idx,
            "layer": self.layer_idx,
            "firing_rate": self.firing_rate,
            "max_activation": self.max_activation,
            "mean_activation": self.mean_activation,
            "purity": self.purity,
            "dominant_label": self.dominant_label,
            "entropy": self.entropy,
            "aa_distribution": self.aa_distribution,
            "top_tokens": [list(t) for t in self.top_tokens],
        }


def normalized_entropy(counts: Sequence[float] | dict[Any, float]) -> float:
    """Shannon entropy scaled to [0, 1] by the maximum for that alphabet size.

    0 = every top-activating token carries the same label (maximally
    monosemantic along this axis); 1 = uniform.
    """
    values = list(counts.values()) if isinstance(counts, dict) else list(counts)
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    probs = [v / total for v in values if v > 0]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(values))


def feature_purity(labels: Sequence[str], alphabet: Sequence[str] = AA_ALPHABET) -> tuple[float, str, float]:
    """Concentration of a feature's top-activating tokens over ``labels``.

    Returns ``(purity, dominant_label, normalized_entropy)`` where purity is the
    fraction carrying the single most common label.

    Purity is reported alongside entropy because they fail differently: a
    feature split evenly between exactly two residues has high entropy over a
    20-letter alphabet yet is arguably interpretable ("aromatic"), and purity
    alone would call it noise while entropy alone would miss how concentrated it
    is.
    """
    if not labels:
        return 0.0, "", 0.0
    counts = Counter(labels)
    dominant, top_count = counts.most_common(1)[0]
    purity = top_count / len(labels)
    full = {a: counts.get(a, 0) for a in alphabet}
    return purity, dominant, normalized_entropy(full)


@torch.no_grad()
def profile_features(
    sae: TopKSparseAutoencoder,
    activations: torch.Tensor,
    token_residues: Sequence[str],
    *,
    features: Sequence[int] | None = None,
    layer_idx: int = -1,
    top_n: int = 32,
    sequence_ids: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    batch_size: int = 8192,
) -> dict[int, FeatureProfile]:
    """Build per-feature profiles over a corpus of activations.

    Args:
        activations: ``[n_tokens, d_in]``.
        token_residues: parallel list of one-letter residues, one per token.
        features: which features to profile. ``None`` profiles all of them,
            which at d_sae=5120 needs the full ``[n_tokens, d_sae]`` matrix
            resident — pass an explicit list on a memory-constrained machine.

    Only the top ``top_n`` activations per feature are retained, via a running
    top-k merge across batches, so corpus size is not bounded by memory.
    """
    if len(token_residues) != activations.shape[0]:
        raise ValueError(
            f"{len(token_residues)} residue labels for {activations.shape[0]} tokens"
        )

    feat_list = list(range(sae.d_sae)) if features is None else [int(f) for f in features]
    feat_index = torch.as_tensor(feat_list, dtype=torch.long, device=sae.device)

    n_tokens = activations.shape[0]
    fired = torch.zeros(len(feat_list), dtype=torch.long)
    act_sum = torch.zeros(len(feat_list), dtype=torch.float64)
    best_vals = torch.full((len(feat_list), top_n), float("-inf"))
    best_idx = torch.full((len(feat_list), top_n), -1, dtype=torch.long)

    for start in range(0, n_tokens, batch_size):
        x = activations[start : start + batch_size].to(sae.device, dtype=sae.dtype)
        z = sae(x).latents[:, feat_index]  # [b, n_feat]
        zc = z.detach().float().cpu()

        fired += (zc > 0).sum(dim=0).long()
        act_sum += zc.sum(dim=0).double()

        # Merge this batch's candidates into the running top-n.
        take = min(top_n, zc.shape[0])
        vals, rows = torch.topk(zc.t(), take, dim=-1)  # [n_feat, take]
        global_rows = rows + start
        merged_vals = torch.cat([best_vals, vals], dim=1)
        merged_idx = torch.cat([best_idx, global_rows], dim=1)
        best_vals, order = torch.topk(merged_vals, top_n, dim=1)
        best_idx = merged_idx.gather(1, order)

    profiles: dict[int, FeatureProfile] = {}
    for j, feature in enumerate(feat_list):
        rate = float(fired[j]) / max(n_tokens, 1)
        top: list[tuple[int, int, str, float]] = []
        labels: list[str] = []
        for slot in range(top_n):
            idx = int(best_idx[j, slot])
            val = float(best_vals[j, slot])
            if idx < 0 or not math.isfinite(val) or val <= 0:
                continue
            residue = token_residues[idx]
            seq_id = int(sequence_ids[idx]) if sequence_ids is not None else -1
            pos = int(positions[idx]) if positions is not None else idx
            top.append((seq_id, pos, residue, val))
            labels.append(residue)

        purity, dominant, entropy = feature_purity(labels)
        counts = Counter(labels)
        total = max(len(labels), 1)
        profiles[feature] = FeatureProfile(
            feature_idx=feature,
            layer_idx=layer_idx,
            firing_rate=rate,
            max_activation=float(best_vals[j, 0]) if math.isfinite(float(best_vals[j, 0])) else 0.0,
            mean_activation=float(act_sum[j]) / max(int(fired[j]), 1),
            top_tokens=top,
            aa_distribution={a: counts.get(a, 0) / total for a in AA_ALPHABET},
            purity=purity,
            dominant_label=dominant,
            entropy=entropy,
        )
    return profiles


def summarise_profiles(profiles: dict[int, FeatureProfile]) -> dict[str, float]:
    """Corpus-level summary of a set of feature profiles."""
    live = [p for p in profiles.values() if not p.is_dead]
    if not live:
        return {"n_features": len(profiles), "n_live": 0}
    purities = [p.purity for p in live]
    entropies = [p.entropy for p in live]
    return {
        "n_features": float(len(profiles)),
        "n_live": float(len(live)),
        "dead_fraction": 1.0 - len(live) / len(profiles),
        "mean_purity": sum(purities) / len(purities),
        "mean_entropy": sum(entropies) / len(entropies),
        # Features that are highly concentrated on one residue type. Reported as
        # a diagnostic, not a target: a dictionary of nothing but residue-identity
        # detectors has learned the input alphabet, not the model's computation.
        "frac_purity_above_0.5": sum(p > 0.5 for p in purities) / len(purities),
    }
