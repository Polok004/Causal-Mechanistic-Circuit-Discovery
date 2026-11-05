"""Residual-stream access for HuggingFace ESM-2.

What "the residual stream" means here
-------------------------------------
ESM-2 is a pre-LayerNorm transformer. Each ``EsmLayer`` computes
``x <- x + attn(ln(x))`` then ``x <- x + mlp(ln(x))`` and returns the updated
``x`` as element 0 of its output tuple. So a forward hook on
``encoder.layer[i]`` observes the residual stream *after* block *i* — which is
exactly the object an SAE should be trained on, and exactly the object a patch
should be written back into. We hook the block rather than an internal
sub-module precisely so that reading and writing are the same tensor.

Why the layer path is resolved rather than hard-coded
-----------------------------------------------------
``EsmForMaskedLM.encoder`` does not exist; the blocks live at
``model.esm.encoder.layer``. ``EsmModel.encoder.layer`` does. Plenty of
published patching code hard-codes one of the two and breaks on the other, so
:func:`resolve_encoder_layers` walks the known paths and raises a message that
says what it actually found.

Reading vs writing
------------------
:class:`ESMActivationExtractor` only reads. Writing is the job of
``interpretability.path_patching``, which uses :func:`make_patch_hook` from this
module so that the read and write paths cannot drift apart.
"""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn

from utils.protein import ESM_VOCAB, seq_to_token_pos, validate_sequence

__all__ = [
    "ESMActivationExtractor",
    "ESMWrapper",
    "HookHandleSet",
    "TokenBatch",
    "make_patch_hook",
    "resolve_encoder_layers",
]

# Candidate attribute paths to the ModuleList of transformer blocks, in the
# order we try them. Covers EsmModel, EsmForMaskedLM, EsmForSequenceClassification
# and models wrapped one level deep.
_LAYER_PATHS: tuple[tuple[str, ...], ...] = (
    ("esm", "encoder", "layer"),
    ("encoder", "layer"),
    ("model", "esm", "encoder", "layer"),
    ("bert", "encoder", "layer"),
)


def _getattr_path(obj: Any, path: Sequence[str]) -> Any | None:
    for name in path:
        obj = getattr(obj, name, None)
        if obj is None:
            return None
    return obj


def resolve_encoder_layers(model: nn.Module) -> nn.ModuleList:
    """Find the ``ModuleList`` of transformer blocks inside an ESM model.

    Raises a diagnostic error rather than an ``AttributeError`` so that a new
    transformers layout is obvious from the traceback.
    """
    for path in _LAYER_PATHS:
        candidate = _getattr_path(model, path)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    top_level = [n for n, _ in model.named_children()]
    raise AttributeError(
        f"could not locate the transformer block list on {type(model).__name__}. "
        f"Tried {['.'.join(p) for p in _LAYER_PATHS]}; top-level children are {top_level}. "
        "If transformers changed the ESM layout, add the new path to _LAYER_PATHS."
    )


def num_layers(model: nn.Module) -> int:
    return len(resolve_encoder_layers(model))


def hidden_size(model: nn.Module) -> int:
    """Residual-stream width, read from the model config."""
    cfg = getattr(model, "config", None)
    if cfg is not None and getattr(cfg, "hidden_size", None):
        return int(cfg.hidden_size)
    raise AttributeError("model has no config.hidden_size; cannot infer residual width")


def _normalise_layer_output(output: Any) -> tuple[torch.Tensor, tuple[Any, ...]]:
    """Split an ``EsmLayer`` output into (hidden_states, rest_of_tuple)."""
    if torch.is_tensor(output):
        return output, ()
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0], tuple(output[1:])
    raise TypeError(
        f"unexpected layer output of type {type(output)!r}; expected a tensor or a "
        "tuple whose first element is the hidden states"
    )


def _rebuild_layer_output(original: Any, new_hidden: torch.Tensor) -> Any:
    """Put ``new_hidden`` back into the same container shape the layer returned."""
    if torch.is_tensor(original):
        return new_hidden
    rest = tuple(original[1:])
    return (new_hidden,) + rest


class HookHandleSet:
    """A set of hook handles that always gets removed.

    Forgetting to remove a hook silently corrupts every later forward pass in
    the process, which in an interpretability sweep shows up as impossible
    numbers many minutes later. Every hook registration in this codebase goes
    through a context manager for that reason.
    """

    def __init__(self) -> None:
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def add(self, handle: torch.utils.hooks.RemovableHandle) -> None:
        self._handles.append(handle)

    def remove(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __len__(self) -> int:
        return len(self._handles)

    def __enter__(self) -> HookHandleSet:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.remove()


@dataclass
class TokenBatch:
    """A tokenised batch plus the bookkeeping patching needs.

    ``attention_mask`` is kept because mean-ablation and activation statistics
    must ignore padding; ``lengths`` is derived from it so callers do not
    recompute it.
    """

    input_ids: torch.Tensor          # [batch, seq]
    attention_mask: torch.Tensor     # [batch, seq]
    sequences: tuple[str, ...] = ()

    @property
    def lengths(self) -> torch.Tensor:
        return self.attention_mask.sum(dim=-1)

    def to(self, device: torch.device | str) -> TokenBatch:
        return TokenBatch(
            input_ids=self.input_ids.to(device),
            attention_mask=self.attention_mask.to(device),
            sequences=self.sequences,
        )

    def as_model_kwargs(self) -> dict[str, torch.Tensor]:
        return {"input_ids": self.input_ids, "attention_mask": self.attention_mask}

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


class ESMActivationExtractor:
    """Capture residual-stream activations from chosen ESM-2 layers.

    Example::

        with ESMActivationExtractor(model, [3, 4]) as ex:
            model(**batch.as_model_kwargs())
            acts = ex.activations[3]        # [batch, seq, d_model]

    The extractor detaches what it captures. It never keeps a graph alive, so it
    is safe to use inside ``torch.no_grad``; for gradient-based attribution use
    ``detach=False`` and keep the forward pass inside a grad-enabled context.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layers: Iterable[int],
        *,
        detach: bool = True,
        store_on_cpu: bool = False,
        clone: bool = True,
    ) -> None:
        self.model = model
        self.layers = resolve_encoder_layers(model)
        self.target_layers: list[int] = sorted({int(i) for i in target_layers})
        self.detach = detach
        self.store_on_cpu = store_on_cpu
        self.clone = clone
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks = HookHandleSet()

        n = len(self.layers)
        bad = [i for i in self.target_layers if not 0 <= i < n]
        if bad:
            raise IndexError(
                f"layer indices {bad} are outside this model's range 0..{n - 1}. "
                "Layer indices are 0-based and count transformer blocks, not embeddings."
            )

    # -- hook plumbing ---------------------------------------------------------

    def _make_hook(self, layer_idx: int) -> Callable[..., None]:
        def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
            hidden, _ = _normalise_layer_output(output)
            if self.detach:
                hidden = hidden.detach()
            if self.clone:
                # Without the clone, a later in-place op on the residual stream
                # would retroactively change what we "captured".
                hidden = hidden.clone()
            if self.store_on_cpu:
                hidden = hidden.to("cpu", non_blocking=True)
            self.activations[layer_idx] = hidden

        return hook

    def register(self) -> ESMActivationExtractor:
        if len(self._hooks):
            return self
        for idx in self.target_layers:
            self._hooks.add(self.layers[idx].register_forward_hook(self._make_hook(idx)))
        return self

    def clear(self) -> None:
        self.activations.clear()

    def remove(self) -> None:
        self._hooks.remove()

    def __enter__(self) -> ESMActivationExtractor:
        self.clear()
        return self.register()

    def __exit__(self, *exc: Any) -> None:
        self.remove()

    # -- convenience -----------------------------------------------------------

    @torch.no_grad()
    def capture(self, batch: TokenBatch) -> dict[int, torch.Tensor]:
        """Run one forward pass and return a copy of the captured activations."""
        with self:
            self.model(**batch.as_model_kwargs())
            return dict(self.activations)

    def flat(self, layer_idx: int, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Return layer activations as ``[n_tokens, d_model]``, padding dropped.

        This is the shape the SAE trains on. Dropping padding matters: ``<pad>``
        positions are a large fraction of a batched protein corpus and they
        would otherwise dominate the dictionary with one trivial feature.
        """
        acts = self.activations[layer_idx]
        if attention_mask is None:
            return acts.reshape(-1, acts.shape[-1])
        mask = attention_mask.to(acts.device).bool().reshape(-1)
        return acts.reshape(-1, acts.shape[-1])[mask]


def make_patch_hook(
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[..., Any]:
    """Build a forward hook that rewrites the residual stream.

    ``transform`` maps ``[batch, seq, d_model] -> [batch, seq, d_model]``. The
    hook preserves whatever container the layer returned, so downstream code
    (attention outputs, ``output_hidden_states``) keeps working.
    """

    def hook(_module: nn.Module, _inputs: Any, output: Any) -> Any:
        hidden, _ = _normalise_layer_output(output)
        new_hidden = transform(hidden)
        if new_hidden.shape != hidden.shape:
            raise ValueError(
                f"patch transform changed the activation shape from {tuple(hidden.shape)} "
                f"to {tuple(new_hidden.shape)}; a residual-stream patch must be shape-preserving"
            )
        return _rebuild_layer_output(output, new_hidden)

    return hook


@contextlib.contextmanager
def patched_layer(
    model: nn.Module,
    layer_idx: int,
    transform: Callable[[torch.Tensor], torch.Tensor],
) -> Iterator[None]:
    """Temporarily rewrite the residual stream at ``layer_idx``."""
    layers = resolve_encoder_layers(model)
    handle = layers[layer_idx].register_forward_hook(make_patch_hook(transform))
    try:
        yield
    finally:
        handle.remove()


@dataclass
class ESMWrapper:
    """Model + tokenizer + the few derived facts the rest of the code needs.

    Construct with :meth:`from_pretrained` for real runs, or directly from an
    already-built model and tokenizer (which is how the tests build an
    architecture-identical model with random weights and no network access).
    """

    model: nn.Module
    tokenizer: Any
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    max_seq_len: int = 1022

    def __post_init__(self) -> None:
        self.model.eval()
        self.model.to(self.device)

    # -- construction ----------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        hf_id: str,
        *,
        device: torch.device | str = "cpu",
        cache_dir: str | None = None,
        max_seq_len: int = 1022,
        dtype: torch.dtype | None = None,
    ) -> ESMWrapper:
        """Load a pretrained ESM-2 checkpoint.

        Requires the weights to be present locally or downloadable. Run
        ``python scripts/fetch_assets.py --models`` once to populate the cache,
        after which this works offline.
        """
        from transformers import AutoTokenizer, EsmForMaskedLM

        tokenizer = AutoTokenizer.from_pretrained(hf_id, cache_dir=cache_dir)
        model = EsmForMaskedLM.from_pretrained(hf_id, cache_dir=cache_dir)
        if dtype is not None:
            model = model.to(dtype)
        return cls(
            model=model,
            tokenizer=tokenizer,
            device=torch.device(device),
            max_seq_len=max_seq_len,
        )

    # -- derived properties ----------------------------------------------------

    @property
    def n_layers(self) -> int:
        return num_layers(self.model)

    @property
    def d_model(self) -> int:
        return hidden_size(self.model)

    @property
    def layers(self) -> nn.ModuleList:
        return resolve_encoder_layers(self.model)

    def aa_token_id(self, aa: str) -> int:
        """Vocabulary id of a single amino-acid letter.

        Falls back to the canonical ESM-2 alphabet if the tokenizer does not
        expose ``convert_tokens_to_ids`` (it always does for EsmTokenizer, but
        the fallback keeps stub tokenizers in tests honest).
        """
        convert = getattr(self.tokenizer, "convert_tokens_to_ids", None)
        if convert is not None:
            tid = convert(aa)
            if tid is not None and tid != getattr(self.tokenizer, "unk_token_id", -1):
                return int(tid)
        return ESM_VOCAB.index(aa)

    # -- tokenisation ----------------------------------------------------------

    def tokenize(self, sequences: str | Sequence[str]) -> TokenBatch:
        """Tokenise one or more sequences into a padded :class:`TokenBatch`."""
        if isinstance(sequences, str):
            sequences = [sequences]
        cleaned = [validate_sequence(s)[: self.max_seq_len] for s in sequences]
        enc = self.tokenizer(
            list(cleaned),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len + 2,  # room for <cls> and <eos>
        )
        return TokenBatch(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
            sequences=tuple(cleaned),
        ).to(self.device)

    # -- forward passes --------------------------------------------------------

    @torch.no_grad()
    def logits(self, batch: TokenBatch) -> torch.Tensor:
        """Masked-LM logits, ``[batch, seq, vocab]``."""
        out = self.model(**batch.as_model_kwargs())
        return out.logits

    @torch.no_grad()
    def log_probs(self, batch: TokenBatch) -> torch.Tensor:
        return torch.log_softmax(self.logits(batch), dim=-1)

    def residual_stream(
        self, batch: TokenBatch, layers: Iterable[int]
    ) -> dict[int, torch.Tensor]:
        """Capture the residual stream at the requested layers."""
        extractor = ESMActivationExtractor(self.model, layers)
        return extractor.capture(batch)

    # -- variant scoring -------------------------------------------------------

    @torch.no_grad()
    def masked_marginal_score(
        self, sequence: str, seq_pos: int, wt_aa: str, mut_aa: str
    ) -> float:
        """ESM-2's standard masked-marginal variant score.

        ``log p(mut | context with position masked) - log p(wt | same context)``.
        Negative means the model considers the mutation worse than wild type,
        which is the direction that correlates with loss of function.

        This is the quantity path patching perturbs, so it is defined here once
        and imported everywhere rather than being reimplemented per script.
        """
        seq = validate_sequence(sequence)
        batch = self.tokenize(seq)
        tok_pos = seq_to_token_pos(seq_pos)
        if tok_pos >= batch.input_ids.shape[1]:
            raise IndexError(
                f"position {seq_pos} (token {tok_pos}) is beyond the tokenised length "
                f"{batch.input_ids.shape[1]}; the sequence was probably truncated at "
                f"max_seq_len={self.max_seq_len}"
            )
        mask_id = getattr(self.tokenizer, "mask_token_id", None)
        if mask_id is None:
            mask_id = ESM_VOCAB.index("<mask>")
        masked = batch.input_ids.clone()
        masked[0, tok_pos] = int(mask_id)
        masked_batch = TokenBatch(masked, batch.attention_mask, batch.sequences)

        lp = torch.log_softmax(self.model(**masked_batch.as_model_kwargs()).logits, dim=-1)
        return float(lp[0, tok_pos, self.aa_token_id(mut_aa)] - lp[0, tok_pos, self.aa_token_id(wt_aa)])
