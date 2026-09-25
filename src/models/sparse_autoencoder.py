"""Top-K sparse autoencoder over ESM-2 residual activations.

Why Top-K rather than an L1 penalty
-----------------------------------
An L1 SAE has a sparsity coefficient that trades reconstruction against L0, and
the right value differs per layer and per model width. Every comparison across
layers then needs a sweep, and the resulting dictionaries are not directly
comparable because they sit at different points on that trade-off curve. Top-K
fixes L0 = k by construction, so "layer 2 vs layer 4" is a clean comparison and
there is no coefficient to tune. It also avoids activation shrinkage: L1 pulls
every active latent toward zero, biasing reconstructions in a way that shows up
as systematic error in downstream patching.

The three details that decide whether this actually works
---------------------------------------------------------
1. **Unit-norm decoder columns.** Without the constraint the model trivially
   reduces the loss by scaling decoder directions up and latents down, and
   feature magnitudes stop meaning anything. We renormalise after every step
   *and* project the parallel component out of the gradient first — renormalising
   alone fights the optimiser, because Adam keeps re-accumulating a radial
   component that we then throw away, which distorts its second-moment estimate.

2. **Dead latents.** A Top-K SAE will happily let most of the dictionary die:
   a latent that never enters the top-k gets no gradient and never will. The
   AuxK loss fixes this by asking the top ``aux_k`` *dead* latents to reconstruct
   the residual error of the main reconstruction. Dead latents therefore always
   receive gradient, and revive. Turning this off (``aux_k=0``) typically leaves
   50-80% of the dictionary dead by the end of training.

3. **Pre-centering.** Subtracting ``b_dec`` before encoding means the encoder
   sees a roughly zero-mean input, and ``b_dec`` can absorb the large constant
   offset that transformer residual streams carry. Initialising ``b_dec`` to the
   data mean (rather than zeros) removes a long transient at the start of
   training where the SAE is mostly learning that offset.

Decoder convention
------------------
``W_dec`` has shape ``[d_sae, d_in]``: row *j* is dictionary element *j*, and the
unit-norm constraint is over ``dim=1``. ``W_enc`` is ``[d_in, d_sae]``.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["SAEConfig", "SAEOutput", "TopKSparseAutoencoder"]


@dataclass
class SAEConfig:
    """Architecture and provenance of a trained SAE.

    Provenance fields (``model_name``, ``layer_idx``) are not decoration: an SAE
    is only meaningful for the exact layer of the exact model it was trained on,
    and loading a layer-3 dictionary against layer-6 activations produces
    plausible-looking nonsense. :meth:`TopKSparseAutoencoder.load` checks them.
    """

    d_in: int
    dict_mult: int = 16
    k: int = 32
    aux_k: int = 256
    aux_alpha: float = 0.03125
    dead_after_tokens: int = 1_000_000
    center_input: bool = True

    # Provenance.
    model_name: str = "unknown"
    layer_idx: int = -1
    # Set at training time when `normalize_activations` is on; applied on load so
    # inference sees the same input scale training did.
    activation_scale: float = 1.0

    @property
    def d_sae(self) -> int:
        return self.d_in * self.dict_mult

    def __post_init__(self) -> None:
        if self.k <= 0:
            raise ValueError(f"k must be positive, got {self.k}")
        if self.k > self.d_sae:
            raise ValueError(
                f"k={self.k} exceeds the dictionary size d_sae={self.d_sae}; "
                "either lower k or raise dict_mult"
            )
        if self.aux_k < 0:
            raise ValueError(f"aux_k must be non-negative, got {self.aux_k}")


@dataclass
class SAEOutput:
    """Everything a caller might want from one SAE forward pass.

    ``latents`` is the dense ``[n, d_sae]`` tensor with non-selected entries
    zeroed. ``indices``/``values`` are the sparse ``[n, k]`` view, which is what
    you want when scanning for max-activating tokens over a large corpus — the
    dense tensor at d_sae=5120 is 16x the size of the activations it came from.
    """

    recon: torch.Tensor
    latents: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    pre_acts: torch.Tensor | None = None


class TopKSparseAutoencoder(nn.Module):
    """A Top-K SAE.

    Args:
        config: architecture. Alternatively pass ``d_in``/``dict_mult``/``k``
            directly and a config is built for you.
    """

    def __init__(
        self,
        config: SAEConfig | None = None,
        *,
        d_in: int | None = None,
        dict_mult: int = 16,
        k: int = 32,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is None:
            if d_in is None:
                raise ValueError("pass either a SAEConfig or d_in=")
            config = SAEConfig(d_in=d_in, dict_mult=dict_mult, k=k, **kwargs)
        self.cfg = config

        d_in_, d_sae = config.d_in, config.d_sae

        # Encoder initialised as the decoder's transpose (a standard trick: the
        # encoder starts as a matched detector for each dictionary direction,
        # which gives useful gradients from step one).
        w_dec = F.normalize(torch.randn(d_sae, d_in_), dim=1)
        self.W_dec = nn.Parameter(w_dec.clone())
        self.W_enc = nn.Parameter(w_dec.t().clone())
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in_))

        # Dead-latent bookkeeping. Buffers so they survive save/load and move
        # with .to(device) — a dead-latent counter that silently stayed on CPU
        # while training ran on MPS would quietly disable AuxK.
        self.register_buffer("tokens_since_fired", torch.zeros(d_sae, dtype=torch.long))
        self.register_buffer("num_tokens_seen", torch.zeros((), dtype=torch.long))

    # -- properties ------------------------------------------------------------

    @property
    def d_in(self) -> int:
        return self.cfg.d_in

    @property
    def d_sae(self) -> int:
        return self.cfg.d_sae

    @property
    def k(self) -> int:
        return self.cfg.k

    @property
    def dtype(self) -> torch.dtype:
        return self.W_dec.dtype

    @property
    def device(self) -> torch.device:
        return self.W_dec.device

    def dead_mask(self) -> torch.Tensor:
        """Boolean ``[d_sae]``: latents that have not fired recently."""
        return self.tokens_since_fired > self.cfg.dead_after_tokens

    def dead_fraction(self) -> float:
        return float(self.dead_mask().float().mean())

    # -- core computation ------------------------------------------------------

    def preactivations(self, x: torch.Tensor) -> torch.Tensor:
        """``[n, d_in] -> [n, d_sae]`` encoder output before sparsification."""
        x_centered = x - self.b_dec if self.cfg.center_input else x
        return x_centered @ self.W_enc + self.b_enc

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sparsify to exactly ``k`` active latents.

        Returns ``(latents, indices, values)`` where ``latents`` is dense.

        ReLU is applied *after* the top-k selection rather than before. Selecting
        on pre-activations and then clamping means the k slots are spent on the
        k most strongly-driven directions; applying ReLU first and then taking
        top-k gives the same result whenever at least k pre-activations are
        positive, and differs only in the degenerate early-training regime where
        fewer than k are. Clamping after selection keeps latents non-negative in
        that regime instead of admitting negative activations into the code.
        """
        pre_acts = self.preactivations(x)
        values, indices = torch.topk(pre_acts, self.cfg.k, dim=-1, sorted=False)
        values = F.relu(values)
        latents = torch.zeros_like(pre_acts).scatter_(-1, indices, values)
        return latents, indices, values

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """``[n, d_sae] -> [n, d_in]``."""
        return latents @ self.W_dec + self.b_dec

    def decode_sparse(self, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Decode from the ``[n, k]`` sparse view without materialising ``[n, d_sae]``.

        Equivalent to ``decode(scatter(indices, values))`` but allocates
        ``n * k * d_in`` instead of ``n * d_sae``, which is the difference
        between fitting and not fitting when scanning a corpus on 8 GB.
        """
        # W_dec[indices] -> [n, k, d_in]; weight by values and sum over k.
        return torch.einsum("nk,nkd->nd", values, self.W_dec[indices]) + self.b_dec

    def forward(self, x: torch.Tensor, *, return_pre_acts: bool = False) -> SAEOutput:
        """Encode then decode. ``x`` is ``[n_tokens, d_in]``.

        Callers holding ``[batch, seq, d_in]`` should flatten first; keeping the
        SAE strictly 2-D avoids silent broadcasting mistakes when the same
        latents tensor is later scattered back into a sequence.
        """
        if x.dim() != 2:
            raise ValueError(
                f"expected [n_tokens, d_in], got {tuple(x.shape)}. "
                "Flatten the batch/sequence dims before calling the SAE."
            )
        if x.shape[-1] != self.d_in:
            raise ValueError(f"expected d_in={self.d_in}, got {x.shape[-1]}")

        pre_acts = self.preactivations(x) if return_pre_acts else None
        latents, indices, values = self.encode(x)
        recon = self.decode(latents)
        if self.training:
            self._update_firing_stats(indices, n_tokens=x.shape[0])
        return SAEOutput(
            recon=recon, latents=latents, indices=indices, values=values, pre_acts=pre_acts
        )

    # -- losses ----------------------------------------------------------------

    def loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """Reconstruction MSE plus the AuxK dead-latent revival term.

        The main loss is mean squared error summed over the feature dimension
        and averaged over tokens, which keeps the scale independent of ``d_in``
        so the same learning rate transfers between the 8M and 35M models.
        """
        out = self(x)
        err = x - out.recon
        mse = err.pow(2).sum(dim=-1).mean()

        metrics: dict[str, float] = {}
        total = mse

        aux_loss = self._auxk_loss(x, err)
        if aux_loss is not None:
            total = total + self.cfg.aux_alpha * aux_loss
            metrics["aux_loss"] = float(aux_loss.detach())

        with torch.no_grad():
            # Fraction of variance unexplained: the scale-free reconstruction
            # number to report, since raw MSE is meaningless without knowing the
            # activation scale of the layer.
            var = (x - x.mean(dim=0, keepdim=True)).pow(2).sum(dim=-1).mean()
            metrics.update(
                mse=float(mse.detach()),
                fvu=float((mse / var.clamp_min(1e-8)).detach()),
                l0=float(self.cfg.k),
                dead_frac=self.dead_fraction(),
                total=float(total.detach()),
            )
        return total, metrics

    def _auxk_loss(self, x: torch.Tensor, err: torch.Tensor) -> torch.Tensor | None:
        """Ask the top dead latents to reconstruct the main model's error."""
        if self.cfg.aux_k <= 0:
            return None
        dead = self.dead_mask()
        n_dead = int(dead.sum())
        if n_dead == 0:
            return None

        aux_k = min(self.cfg.aux_k, n_dead)
        pre_acts = self.preactivations(x)
        # Restrict the top-k to dead latents by sending live ones to -inf.
        masked = pre_acts.masked_fill(~dead.unsqueeze(0), float("-inf"))
        values, indices = torch.topk(masked, aux_k, dim=-1, sorted=False)
        values = F.relu(values)
        # Decode without b_dec: we are reconstructing a residual, not the signal.
        err_hat = torch.einsum("nk,nkd->nd", values, self.W_dec[indices])
        return (err.detach() - err_hat).pow(2).sum(dim=-1).mean()

    # -- constraint maintenance ------------------------------------------------

    @torch.no_grad()
    def _update_firing_stats(self, indices: torch.Tensor, n_tokens: int) -> None:
        fired = torch.zeros(self.d_sae, dtype=torch.bool, device=indices.device)
        fired.scatter_(0, indices.reshape(-1), True)
        self.tokens_since_fired += n_tokens
        self.tokens_since_fired[fired] = 0
        self.num_tokens_seen += n_tokens

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Project decoder rows back onto the unit sphere. Call after each step."""
        self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @torch.no_grad()
    def remove_parallel_gradient(self) -> None:
        """Strip the radial component of the decoder gradient.

        Call between ``loss.backward()`` and ``optimizer.step()``. Only the
        tangential component can change a unit-norm direction; leaving the radial
        part in means Adam's second-moment estimate is inflated by a component
        that :meth:`normalize_decoder` immediately discards, which in practice
        shows up as an effective learning rate that drifts during training.
        """
        if self.W_dec.grad is None:
            return
        w = self.W_dec.data
        g = self.W_dec.grad
        radial = (g * w).sum(dim=1, keepdim=True) * w
        self.W_dec.grad -= radial

    @torch.no_grad()
    def init_b_dec_from_data(self, x: torch.Tensor) -> None:
        """Initialise the decoder bias to the data mean.

        The geometric median is the theoretically better choice, but on residual
        streams the two are close and the mean costs one pass instead of an
        iterative solve. Call once, before training.
        """
        self.b_dec.data = x.mean(dim=0).to(self.b_dec.dtype)

    # -- persistence -----------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write weights and config side by side.

        Weights go to ``<path>`` and config to ``<path>.json``. The config is
        plain JSON on purpose: you can read what a checkpoint is without loading
        torch.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path)
        path.with_suffix(path.suffix + ".json").write_text(
            json.dumps(asdict(self.cfg), indent=2)
        )
        return path

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        device: torch.device | str = "cpu",
        expect_model: str | None = None,
        expect_layer: int | None = None,
    ) -> TopKSparseAutoencoder:
        """Load a checkpoint, refusing obvious provenance mismatches."""
        path = Path(path)
        cfg_path = path.with_suffix(path.suffix + ".json")
        if not cfg_path.exists():
            raise FileNotFoundError(
                f"{cfg_path} is missing; an SAE checkpoint needs its config sidecar "
                "to know d_in, k and which layer it belongs to"
            )
        cfg = SAEConfig(**json.loads(cfg_path.read_text()))

        if expect_model is not None and cfg.model_name not in ("unknown", expect_model):
            raise ValueError(
                f"SAE at {path} was trained on {cfg.model_name!r} but you are running "
                f"{expect_model!r}. Dictionaries do not transfer between checkpoints."
            )
        if expect_layer is not None and cfg.layer_idx >= 0 and cfg.layer_idx != expect_layer:
            raise ValueError(
                f"SAE at {path} was trained on layer {cfg.layer_idx} but you asked for "
                f"layer {expect_layer}. Features are layer-specific."
            )

        sae = cls(cfg)
        state = torch.load(path, map_location="cpu", weights_only=True)
        sae.load_state_dict(state)
        return sae.to(device)

    def extra_repr(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"d_in={self.d_in}, d_sae={self.d_sae}, k={self.k}, "
            f"aux_k={self.cfg.aux_k}, layer={self.cfg.layer_idx}, model={self.cfg.model_name}"
        )


@dataclass
class ActivationNormalizer:
    """Rescale activations to unit mean L2 norm.

    ESM-2 residual norms grow substantially with depth. Without this, a learning
    rate tuned on layer 2 is wrong on layer 8, and reconstruction MSE is not
    comparable across layers. The scale is a single scalar, stored on the SAE
    config so inference reproduces training exactly.
    """

    scale: float = 1.0
    fitted: bool = False
    target_norm: float = field(default=1.0)

    @classmethod
    def fit(cls, x: torch.Tensor, target_norm: float = 1.0) -> ActivationNormalizer:
        mean_norm = float(x.norm(dim=-1).mean())
        if mean_norm <= 0:
            raise ValueError("activations have zero mean norm; nothing to normalise")
        return cls(scale=target_norm / mean_norm, fitted=True, target_norm=target_norm)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scale

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale
