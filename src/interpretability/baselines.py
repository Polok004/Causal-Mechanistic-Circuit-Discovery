"""Post-hoc saliency baselines that causal patching is compared against.

Each baseline returns a per-residue importance vector for one variant, so they
plug into the same evaluation as a discovered circuit. The point of the
comparison is not that these methods are bad — they are cheap and often useful —
but that they answer a different question. Attention and gradients report what
the model *looked at*; patching reports what changing the value actually
*does*. Those come apart exactly where it matters: a residue can receive heavy
attention while contributing nothing to the output, and a residue in a saturated
regime can be decisive while having a near-zero gradient.

Every method here produces scores over residue positions (0-indexed into the
sequence), with the ``<cls>``/``<eos>`` tokens excluded, so the evaluation never
has to reason about token offsets.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from models.esm_hooks import ESMWrapper, TokenBatch
from utils.protein import N_PREFIX_TOKENS

__all__ = [
    "attention_saliency",
    "integrated_gradients_saliency",
    "random_saliency",
    "sae_circuit_saliency",
]


def _strip_special(scores: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Drop <cls>/<eos> and truncate to the residue count."""
    return scores[N_PREFIX_TOKENS : N_PREFIX_TOKENS + seq_len]


@torch.no_grad()
def attention_saliency(
    wrapper: ESMWrapper,
    batch: TokenBatch,
    *,
    query_token_pos: int,
    layers: Sequence[int] | None = None,
    head_reduction: str = "mean",
) -> torch.Tensor:
    """How much attention the mutated position pays to each residue.

    This is the "raw attention head" baseline: take the attention weights from
    the mutated position as query, average over heads and the chosen layers, and
    read the result as importance.

    Its well-known failure is that attention weights are a *convex combination*,
    so they always sum to one and always look like an explanation, whether or
    not the attended values influence the output. Attention rollout and
    attention-times-gradient variants exist; raw attention is used here because
    it is what the plan names and what most protein-LM papers actually plot.
    """
    seq_len = len(batch.sequences[0]) if batch.sequences else int(batch.attention_mask[0].sum()) - 2
    out = wrapper.model(**batch.as_model_kwargs(), output_attentions=True)
    attentions = out.attentions  # tuple of [batch, heads, seq, seq]
    if attentions is None:
        raise RuntimeError(
            "the model did not return attentions; it may have been loaded with an "
            "attention implementation that does not expose weights (e.g. SDPA). "
            "Reload with attn_implementation='eager'."
        )

    chosen = range(len(attentions)) if layers is None else layers
    acc = None
    for layer in chosen:
        a = attentions[layer][0]  # [heads, seq, seq]
        row = a[:, query_token_pos, :]  # [heads, seq]
        reduced = row.max(dim=0).values if head_reduction == "max" else row.mean(dim=0)
        acc = reduced if acc is None else acc + reduced
    assert acc is not None
    return _strip_special(acc / len(list(chosen)), seq_len).detach().cpu()


def integrated_gradients_saliency(
    wrapper: ESMWrapper,
    batch: TokenBatch,
    metric_fn,
    *,
    steps: int = 32,
    baseline: str = "zero",
) -> torch.Tensor:
    """Integrated gradients over the input embeddings.

    Interpolates the token embeddings from a baseline to the actual input and
    integrates the gradient of the metric along that path, then contracts the
    embedding dimension with an L2 norm to get one score per residue.

    Two choices worth naming, because they change the numbers:

    * **Baseline.** A zero embedding is the conventional choice but is not a
      meaningful "absence of a residue" for a protein LM. ``baseline="mask"``
      uses the ``<mask>`` embedding instead, which is the model's own
      representation of an unknown residue and is the more defensible reference
      point. Both are offered; the benchmark reports the zero baseline so the
      comparison matches published practice.
    * **Contraction.** L2 over embedding dimensions discards sign, so the result
      is an unsigned magnitude. A signed dot product with the input is
      available in some implementations but mixes attribution with input scale.
    """
    embeddings = wrapper.model.get_input_embeddings()
    input_ids = batch.input_ids
    seq_len = len(batch.sequences[0]) if batch.sequences else int(batch.attention_mask[0].sum()) - 2

    with torch.no_grad():
        actual = embeddings(input_ids)  # [1, seq, d]
        if baseline == "mask":
            mask_id = getattr(wrapper.tokenizer, "mask_token_id", None)
            if mask_id is None:
                raise ValueError("tokenizer has no mask token; use baseline='zero'")
            base = embeddings(torch.full_like(input_ids, int(mask_id)))
        else:
            base = torch.zeros_like(actual)

    total = torch.zeros_like(actual)
    for step in range(steps):
        # Midpoint (Riemann trapezoid) sampling: strictly better than left
        # endpoints at the same cost, and the completeness error is what people
        # most often blame on "IG being noisy".
        alpha = (step + 0.5) / steps
        point = (base + alpha * (actual - base)).detach().requires_grad_(True)
        out = wrapper.model(
            inputs_embeds=point, attention_mask=batch.attention_mask
        )
        value = metric_fn(out.logits)
        (grad,) = torch.autograd.grad(value, point)
        total = total + grad

    avg_grad = total / steps
    attributions = (actual - base) * avg_grad  # [1, seq, d]
    scores = attributions[0].norm(dim=-1)
    return _strip_special(scores, seq_len).detach().cpu()


def random_saliency(seq_len: int, *, seed: int = 0) -> torch.Tensor:
    """Uniform random scores — the floor every method must clear.

    Included because a surprising number of saliency comparisons omit it, and
    on faithfulness curves in particular a random baseline is a stronger
    competitor than people expect.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(seq_len, generator=g)


def sae_circuit_saliency(
    circuit_residues_1indexed: Sequence[int],
    residue_scores: Sequence[float],
    seq_len: int,
) -> torch.Tensor:
    """Turn a discovered circuit into a dense per-residue score vector.

    Residues outside the circuit get zero. This makes the circuit directly
    comparable to the dense baselines, at the cost of being unable to rank
    beyond its own size — which is honest, since the method's claim is about the
    residues it selects, not about everything else.
    """
    scores = torch.zeros(seq_len)
    for residue, score in zip(circuit_residues_1indexed, residue_scores, strict=False):
        idx = residue - 1
        if 0 <= idx < seq_len:
            scores[idx] = abs(float(score))
    return scores
