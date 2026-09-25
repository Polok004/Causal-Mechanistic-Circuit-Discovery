"""Loading variant-effect data and streaming activations.

Two jobs live here.

**Variant pairs.** ProteinGym and ClinVar both give (wild-type sequence,
substitution, label) triples but disagree about column names, numbering and how
multi-substitution variants are encoded. :func:`load_variants` normalises all of
that into :class:`VariantPair`, and — importantly — *verifies* that the
wild-type residue named in each mutation string is the residue actually present
at that position. Numbering offsets between a mutation table and its FASTA are
the single most common silent corruption in variant-effect pipelines, and every
downstream conclusion inherits it. Mismatches are dropped and counted rather
than being allowed through.

**Activation streaming.** Training an SAE wants shuffled activations, but
materialising them is not an option: 10,000 sequences at ~300 tokens and
d=320 in float32 is roughly 4 GB, on a machine with 8 GB total that is also
holding the model. :class:`ActivationBuffer` keeps a fixed-size shuffled pool,
refilling from fresh forward passes as it drains. Memory is bounded by
``buffer_tokens`` regardless of corpus size.

Shuffling is not cosmetic: consecutive tokens from one protein are strongly
correlated, and an SAE trained on unshuffled batches learns per-protein
idiosyncrasies that look like features and do not generalise.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch

from utils.protein import AA_ALPHABET, Mutation, parse_mutation, validate_sequence

__all__ = [
    "ActivationBuffer",
    "VariantPair",
    "iter_fasta",
    "load_clinvar",
    "load_proteingym",
    "load_variants",
    "synthetic_corpus",
    "synthetic_variants",
]


@dataclass
class VariantPair:
    """A wild-type / mutant pair with its phenotype label."""

    wt_sequence: str
    mutation: Mutation
    label: float | None = None
    binary_label: int | None = None
    assay: str = ""
    protein_id: str = ""
    pdb_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def mut_sequence(self) -> str:
        return self.mutation.apply(self.wt_sequence)

    @property
    def name(self) -> str:
        return f"{self.protein_id or self.assay or 'protein'}:{self.mutation.raw}"

    def __len__(self) -> int:
        return len(self.wt_sequence)


# --------------------------------------------------------------------------- #
# FASTA                                                                         #
# --------------------------------------------------------------------------- #


def iter_fasta(path: str | Path) -> Iterator[tuple[str, str]]:
    """Yield ``(header, sequence)`` from a FASTA file, streaming.

    Written by hand rather than via Biopython's SeqIO so that SAE-corpus loading
    does not pull in Biopython's parsing stack, and so a multi-GB FASTA never
    lands in memory.
    """
    path = Path(path)
    header: str | None = None
    chunks: list[str] = []
    with path.open("r") as fh:
        for line in fh:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(chunks)
                header, chunks = line[1:], []
            else:
                chunks.append(line)
    if header is not None:
        yield header, "".join(chunks)


def load_sequence_corpus(
    fasta: str | Path | None,
    *,
    min_len: int = 40,
    max_len: int = 512,
    limit: int | None = None,
    allow_synthetic_fallback: bool = True,
    seed: int = 0,
) -> list[str]:
    """Sequences for SAE training.

    Falls back to a synthetic corpus when the FASTA is absent, so the training
    script runs end to end before any download has happened. The fallback is
    announced loudly — an SAE trained on synthetic sequences is a smoke test,
    not a result.
    """
    if fasta is not None and Path(fasta).exists():
        out: list[str] = []
        for _, seq in iter_fasta(fasta):
            try:
                cleaned = validate_sequence(seq)
            except ValueError:
                continue
            if min_len <= len(cleaned) <= max_len:
                out.append(cleaned)
            if limit is not None and len(out) >= limit:
                break
        if out:
            return out

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"no usable sequences at {fasta}; run `python scripts/fetch_assets.py --corpus`"
        )
    print(
        f"[dataloaders] WARNING: no corpus at {fasta}; falling back to synthetic sequences. "
        "Results from this run are a smoke test only — run "
        "`python scripts/fetch_assets.py --corpus` for a real dictionary."
    )
    return synthetic_corpus(
        n=limit or 1000, min_len=min_len, max_len=min(max_len, 200), seed=seed
    )


# --------------------------------------------------------------------------- #
# ProteinGym                                                                    #
# --------------------------------------------------------------------------- #


def load_proteingym(
    dms_dir: str | Path,
    *,
    assays: Sequence[str] | None = None,
    mutant_col: str = "mutant",
    score_col: str = "DMS_score",
    binarised_col: str = "DMS_score_bin",
    max_wt_len: int = 500,
    max_variants_per_assay: int | None = 200,
    skip_multi_mutants: bool = True,
    seed: int = 0,
    verbose: bool = True,
) -> list[VariantPair]:
    """Load ProteinGym DMS substitution assays.

    Each assay CSV carries a ``mutated_sequence`` column; the wild-type is
    recovered by reverting the substitution, which is more reliable than
    matching the assay to an external reference FASTA and is self-consistent by
    construction.
    """
    import pandas as pd

    dms_dir = Path(dms_dir)
    if not dms_dir.exists():
        raise FileNotFoundError(
            f"{dms_dir} does not exist. Run `python scripts/fetch_assets.py --proteingym` "
            "or switch to data.source=synthetic for a dry run."
        )

    files = sorted(dms_dir.glob("*.csv"))
    if assays:
        wanted = {a if a.endswith(".csv") else f"{a}.csv" for a in assays}
        files = [f for f in files if f.name in wanted]
    if not files:
        raise FileNotFoundError(f"no assay CSVs found in {dms_dir}")

    rng = random.Random(seed)
    out: list[VariantPair] = []
    n_dropped = 0

    for path in files:
        df = pd.read_csv(path)
        if mutant_col not in df.columns:
            if verbose:
                print(f"[dataloaders] skipping {path.name}: no '{mutant_col}' column")
            continue

        seq_col = next(
            (c for c in ("mutated_sequence", "sequence", "mutant_sequence") if c in df.columns),
            None,
        )
        if seq_col is None:
            if verbose:
                print(f"[dataloaders] skipping {path.name}: no mutated-sequence column")
            continue

        rows = df.to_dict("records")
        if max_variants_per_assay is not None and len(rows) > max_variants_per_assay:
            rows = rng.sample(rows, max_variants_per_assay)

        for row in rows:
            token = str(row[mutant_col])
            if ":" in token:
                if skip_multi_mutants:
                    continue
                token = token.split(":")[0]
            try:
                mutation = parse_mutation(token)
                mutant_seq = validate_sequence(str(row[seq_col]))
                # Revert the substitution to recover the wild type.
                if not 0 <= mutation.seq_pos < len(mutant_seq):
                    n_dropped += 1
                    continue
                if mutant_seq[mutation.seq_pos] != mutation.mut_aa:
                    # The mutation string and the sequence disagree: a numbering
                    # offset. Dropping is the only safe response.
                    n_dropped += 1
                    continue
                wt_seq = (
                    mutant_seq[: mutation.seq_pos]
                    + mutation.wt_aa
                    + mutant_seq[mutation.seq_pos + 1 :]
                )
            except (ValueError, IndexError):
                n_dropped += 1
                continue

            if len(wt_seq) > max_wt_len:
                continue

            out.append(
                VariantPair(
                    wt_sequence=wt_seq,
                    mutation=mutation,
                    label=float(row[score_col]) if score_col in row and row[score_col] == row[score_col] else None,
                    binary_label=(
                        int(row[binarised_col])
                        if binarised_col in row and row[binarised_col] == row[binarised_col]
                        else None
                    ),
                    assay=path.stem,
                    protein_id=path.stem.split("_")[0],
                )
            )

    if verbose:
        print(
            f"[dataloaders] ProteinGym: {len(out)} variants from {len(files)} assays"
            + (f" ({n_dropped} dropped on numbering mismatch)" if n_dropped else "")
        )
    return out


def load_clinvar(
    table: str | Path,
    *,
    seq_col: str = "wt_sequence",
    mutant_col: str = "mutant",
    label_col: str = "clinical_significance",
    pathogenic_labels: Sequence[str] = ("Pathogenic", "Likely_pathogenic"),
    benign_labels: Sequence[str] = ("Benign", "Likely_benign"),
    verbose: bool = True,
) -> list[VariantPair]:
    """Load a ClinVar missense table.

    Expects one row per variant with a wild-type sequence, a mutation string and
    a clinical significance label. Variants of uncertain significance are
    dropped: the point of using ClinVar here is a clean binary contrast, and VUS
    rows would add label noise of exactly the kind that makes a circuit look
    less faithful than it is.
    """
    import pandas as pd

    table = Path(table)
    if not table.exists():
        raise FileNotFoundError(
            f"{table} does not exist. See scripts/fetch_assets.py --clinvar for the "
            "expected format."
        )

    sep = "\t" if table.suffix in (".tsv", ".txt") else ","
    df = pd.read_csv(table, sep=sep)
    pathogenic, benign = set(pathogenic_labels), set(benign_labels)

    out: list[VariantPair] = []
    n_dropped = 0
    for row in df.to_dict("records"):
        label = str(row.get(label_col, "")).strip()
        if label in pathogenic:
            binary = 0  # deleterious
        elif label in benign:
            binary = 1  # functional
        else:
            continue

        try:
            mutation = parse_mutation(str(row[mutant_col]))
            wt = validate_sequence(str(row[seq_col]))
            mutation.apply(wt)  # validates the wild-type residue matches
        except (ValueError, IndexError, KeyError):
            n_dropped += 1
            continue

        out.append(
            VariantPair(
                wt_sequence=wt,
                mutation=mutation,
                binary_label=binary,
                assay="clinvar",
                protein_id=str(row.get("gene", "") or row.get("protein_id", "")),
                pdb_id=str(row["pdb_id"]) if row.get("pdb_id") else None,
                metadata={"clinical_significance": label},
            )
        )

    if verbose:
        print(
            f"[dataloaders] ClinVar: {len(out)} variants"
            + (f" ({n_dropped} dropped on validation)" if n_dropped else "")
        )
    return out


# --------------------------------------------------------------------------- #
# Synthetic data                                                                #
# --------------------------------------------------------------------------- #


def synthetic_corpus(
    n: int = 1000, *, min_len: int = 60, max_len: int = 200, seed: int = 0
) -> list[str]:
    """Random sequences with realistic amino-acid frequencies.

    Frequencies are roughly SwissProt's, so a dictionary trained on this does
    not learn a uniform-alphabet prior that real proteins immediately violate.
    Still synthetic: there is no structure here to find.
    """
    rng = random.Random(seed)
    weights = [
        8.3, 1.4, 5.4, 6.7, 3.9, 7.1, 2.3, 5.9, 5.8, 9.7,
        2.4, 4.1, 4.7, 3.9, 5.5, 6.6, 5.3, 6.9, 1.1, 2.9,
    ]
    return [
        "".join(rng.choices(AA_ALPHABET, weights=weights, k=rng.randint(min_len, max_len)))
        for _ in range(n)
    ]


def synthetic_variants(
    n_proteins: int = 64,
    *,
    min_len: int = 60,
    max_len: int = 180,
    variants_per_protein: int = 8,
    motif: str = "HExxH",
    seed: int = 0,
) -> list[VariantPair]:
    """Variants with a planted, known-important motif.

    Each protein gets one copy of ``motif`` (``x`` = any residue) inserted at a
    random position. Variants inside the motif are labelled deleterious, those
    outside it functional. That gives circuit discovery a ground truth to be
    scored against before any real data is involved — if the pipeline cannot
    recover a motif it was handed, it will not recover an active site.
    """
    rng = random.Random(seed)
    corpus = synthetic_corpus(n_proteins, min_len=min_len, max_len=max_len, seed=seed)
    out: list[VariantPair] = []

    for i, base in enumerate(corpus):
        motif_len = len(motif)
        start = rng.randint(5, max(6, len(base) - motif_len - 5))
        realised = "".join(
            rng.choice(AA_ALPHABET) if c == "x" else c for c in motif
        )
        seq = base[:start] + realised + base[start + motif_len :]
        motif_positions = set(range(start, start + motif_len))
        # Only the fixed (non-x) motif residues are functionally constrained.
        critical = {start + j for j, c in enumerate(motif) if c != "x"}

        for _ in range(variants_per_protein):
            in_motif = rng.random() < 0.5
            pool = sorted(critical) if in_motif and critical else [
                p for p in range(len(seq)) if p not in motif_positions
            ]
            pos = rng.choice(pool)
            wt_aa = seq[pos]
            mut_aa = rng.choice([a for a in AA_ALPHABET if a != wt_aa])
            mutation = parse_mutation(f"{wt_aa}{pos + 1}{mut_aa}")
            is_critical = pos in critical
            out.append(
                VariantPair(
                    wt_sequence=seq,
                    mutation=mutation,
                    label=-2.0 if is_critical else 0.0,
                    binary_label=0 if is_critical else 1,
                    assay="synthetic",
                    protein_id=f"SYN{i:04d}",
                    metadata={
                        "motif": realised,
                        "motif_start_1indexed": start + 1,
                        "critical_residues_1indexed": sorted(p + 1 for p in critical),
                        "is_critical": is_critical,
                    },
                )
            )
    return out


def load_variants(cfg: Any, *, seed: int = 0, verbose: bool = True) -> list[VariantPair]:
    """Dispatch to the loader named by ``cfg.source``."""
    source: Literal["proteingym", "clinvar", "synthetic"] = cfg.source
    if source == "proteingym":
        pg = cfg.proteingym
        return load_proteingym(
            pg.dms_dir,
            assays=pg.assays,
            mutant_col=pg.mutant_col,
            score_col=pg.score_col,
            binarised_col=pg.binarised_col,
            max_wt_len=pg.max_wt_len,
            max_variants_per_assay=pg.max_variants_per_assay,
            seed=seed,
            verbose=verbose,
        )
    if source == "clinvar":
        cv = cfg.clinvar
        return load_clinvar(
            cv.table,
            seq_col=cv.seq_col,
            mutant_col=cv.mutant_col,
            label_col=cv.label_col,
            pathogenic_labels=list(cv.pathogenic_labels),
            benign_labels=list(cv.benign_labels),
            verbose=verbose,
        )
    if source == "synthetic":
        sy = cfg.synthetic
        return synthetic_variants(
            n_proteins=sy.num_proteins,
            min_len=sy.min_len,
            max_len=sy.max_len,
            variants_per_protein=sy.variants_per_protein,
            motif=sy.motif,
            seed=sy.seed,
        )
    raise ValueError(f"unknown data source {source!r}; expected proteingym, clinvar or synthetic")


# --------------------------------------------------------------------------- #
# Activation streaming                                                          #
# --------------------------------------------------------------------------- #


class ActivationBuffer:
    """A bounded, shuffled pool of residual-stream activations.

    Usage::

        buffer = ActivationBuffer(extract_fn, sequences, d_model=320,
                                  buffer_tokens=262_144, batch_size=4096)
        for batch in buffer:            # [batch_size, d_model]
            ...

    The pool is refilled whenever it drops below ``refill_at`` of capacity, and
    is reshuffled on every refill. Peak memory is ``buffer_tokens * d_model *
    4`` bytes plus whatever the model needs for one extraction batch — about
    320 MB at the defaults, which is what makes this run alongside ESM-2 on an
    8 GB machine.

    Exhausting the sequence list ends iteration; set ``loop=True`` to cycle,
    which is what SAE training wants when ``total_steps`` exceeds one pass over
    the corpus.
    """

    def __init__(
        self,
        extract_fn: Callable[[Sequence[str]], torch.Tensor],
        sequences: Sequence[str],
        *,
        d_model: int,
        buffer_tokens: int = 262_144,
        batch_size: int = 4096,
        extract_batch_size: int = 8,
        refill_at: float = 0.5,
        loop: bool = True,
        seed: int = 0,
        device: torch.device | str = "cpu",
    ) -> None:
        self.extract_fn = extract_fn
        self.sequences = list(sequences)
        self.d_model = d_model
        self.buffer_tokens = buffer_tokens
        self.batch_size = batch_size
        self.extract_batch_size = extract_batch_size
        self.refill_at = refill_at
        self.loop = loop
        self.device = torch.device(device)
        self._rng = random.Random(seed)
        self._gen = torch.Generator().manual_seed(seed)

        # Buffer lives on CPU; batches are moved to the compute device on
        # demand. Keeping 262k x 320 floats on MPS would compete with the model
        # for the same unified memory pool for no gain.
        self._buffer = torch.empty(0, d_model)
        self._cursor = 0
        self._exhausted = False
        self._rng.shuffle(self.sequences)

    def _next_sequences(self) -> list[str]:
        if self._cursor >= len(self.sequences):
            if not self.loop:
                self._exhausted = True
                return []
            self._cursor = 0
            self._rng.shuffle(self.sequences)
        chunk = self.sequences[self._cursor : self._cursor + self.extract_batch_size]
        self._cursor += self.extract_batch_size
        return chunk

    def _refill(self) -> None:
        collected = [self._buffer] if self._buffer.numel() else []
        have = self._buffer.shape[0]
        while have < self.buffer_tokens and not self._exhausted:
            chunk = self._next_sequences()
            if not chunk:
                break
            acts = self.extract_fn(chunk).detach().to("cpu", dtype=torch.float32)
            if acts.numel() == 0:
                continue
            collected.append(acts)
            have += acts.shape[0]

        if not collected:
            self._buffer = torch.empty(0, self.d_model)
            return

        pool = torch.cat(collected, dim=0)
        # Reshuffle the whole pool, not just the new part: otherwise recently
        # added tokens stay clumped by protein of origin.
        perm = torch.randperm(pool.shape[0], generator=self._gen)
        self._buffer = pool[perm]

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        low_water = int(self.refill_at * self.buffer_tokens)
        if self._buffer.shape[0] < max(low_water, self.batch_size):
            self._refill()
        if self._buffer.shape[0] < self.batch_size:
            raise StopIteration
        batch, self._buffer = (
            self._buffer[: self.batch_size],
            self._buffer[self.batch_size :],
        )
        return batch.to(self.device)

    def peek(self, n: int) -> torch.Tensor:
        """Return ``n`` tokens without consuming them.

        Used to fit the activation normaliser and initialise ``b_dec`` before
        the first training step.
        """
        if self._buffer.shape[0] < n:
            self._refill()
        return self._buffer[:n].clone().to(self.device)
