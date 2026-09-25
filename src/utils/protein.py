"""Protein sequence constants and mutation parsing.

The indexing conventions here are the single most common source of off-by-one
bugs in this kind of pipeline, so they are stated once and enforced everywhere:

* **Mutation strings** ("A45T") are 1-indexed into the wild-type sequence, which
  is what ProteinGym, ClinVar and the PDB all use.
* **Sequence positions** (``seq_pos``) are 0-indexed into the plain amino-acid
  string.
* **Token positions** (``tok_pos``) are 0-indexed into the tokenised batch, and
  for ESM-2 are shifted by one because of the leading ``<cls>`` token.

:func:`parse_mutation` returns a 0-indexed ``seq_pos``; converting to a token
position is :func:`seq_to_token_pos`, and nothing in the codebase is allowed to
do that conversion inline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "AA_ALPHABET",
    "ESM_VOCAB",
    "Mutation",
    "is_canonical_aa",
    "parse_mutation",
    "seq_to_token_pos",
    "three_to_one",
    "token_to_seq_pos",
    "validate_sequence",
]

#: The 20 canonical amino acids, in the conventional single-letter order.
AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"

#: The ESM-2 token vocabulary, in checkpoint order. Index 0 is ``<cls>``, and
#: every ESM-2 checkpoint from 8M to 15B shares this 33-token alphabet.
ESM_VOCAB: tuple[str, ...] = (
    "<cls>", "<pad>", "<eos>", "<unk>",
    "L", "A", "G", "V", "S", "E", "R", "T", "I", "D", "P", "K",
    "Q", "N", "F", "Y", "M", "H", "W", "C", "X", "B", "U", "Z", "O",
    ".", "-", "<null_1>", "<mask>",
)

#: Number of special tokens prepended by the ESM-2 tokenizer (just ``<cls>``).
N_PREFIX_TOKENS = 1

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    # Common modified / non-standard residues seen in PDB entries.
    "MSE": "M", "SEC": "U", "PYL": "O", "HYP": "P", "SEP": "S",
    "TPO": "T", "PTR": "Y", "CSO": "C", "CME": "C", "MLY": "K",
}

_MUTATION_RE = re.compile(r"^(?P<wt>[A-Z])(?P<pos>\d+)(?P<mut>[A-Z])$")


def three_to_one(three: str, default: str = "X") -> str:
    """Convert a PDB three-letter residue code to one letter.

    Unknown codes map to ``default`` ("X") rather than raising, because real PDB
    files contain ligands and modified residues that we want to skip, not crash
    on.
    """
    return _THREE_TO_ONE.get(three.strip().upper(), default)


def is_canonical_aa(aa: str) -> bool:
    return len(aa) == 1 and aa.upper() in AA_ALPHABET


@dataclass(frozen=True)
class Mutation:
    """A single amino-acid substitution.

    Attributes:
        wt_aa: Wild-type residue, single letter.
        seq_pos: **0-indexed** position in the amino-acid string.
        mut_aa: Mutant residue, single letter.
        raw: The original mutation string, kept for reporting.
    """

    wt_aa: str
    seq_pos: int
    mut_aa: str
    raw: str

    @property
    def one_indexed_pos(self) -> int:
        """1-indexed position, i.e. the number that appears in the raw string."""
        return self.seq_pos + 1

    def apply(self, sequence: str) -> str:
        """Return ``sequence`` with this substitution applied, checking the wt."""
        if not 0 <= self.seq_pos < len(sequence):
            raise IndexError(
                f"mutation {self.raw} is at 1-indexed position {self.one_indexed_pos}, "
                f"outside a sequence of length {len(sequence)}"
            )
        observed = sequence[self.seq_pos]
        if observed != self.wt_aa:
            raise ValueError(
                f"mutation {self.raw} expects wild-type '{self.wt_aa}' at 1-indexed "
                f"position {self.one_indexed_pos}, but the sequence has '{observed}'. "
                "This usually means the mutation table and the FASTA disagree on "
                "numbering (a common ProteinGym/UniProt offset issue)."
            )
        return sequence[: self.seq_pos] + self.mut_aa + sequence[self.seq_pos + 1 :]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw


def parse_mutation(mutation: str) -> Mutation:
    """Parse a ``"A45T"``-style substitution string.

    The position in the string is 1-indexed; the returned ``seq_pos`` is
    0-indexed. Multi-substitution strings (ProteinGym joins them with ":") are
    rejected here — callers that support them should split first and decide
    explicitly how to handle the multi-mutant case.
    """
    token = mutation.strip().upper()
    if ":" in token:
        raise ValueError(
            f"{mutation!r} looks like a multi-substitution variant. Split on ':' and "
            "handle each substitution explicitly; single-site causal attribution is "
            "not well defined for a joint mutant."
        )
    m = _MUTATION_RE.match(token)
    if not m:
        raise ValueError(
            f"could not parse mutation {mutation!r}; expected '<wt><1-indexed pos><mut>', e.g. 'A45T'"
        )
    pos = int(m.group("pos"))
    if pos < 1:
        raise ValueError(f"mutation {mutation!r} has a non-positive position; positions are 1-indexed")
    wt, mut = m.group("wt"), m.group("mut")
    for aa, role in ((wt, "wild-type"), (mut, "mutant")):
        if not is_canonical_aa(aa):
            raise ValueError(f"{role} residue {aa!r} in {mutation!r} is not a canonical amino acid")
    return Mutation(wt_aa=wt, seq_pos=pos - 1, mut_aa=mut, raw=token)


def seq_to_token_pos(seq_pos: int) -> int:
    """0-indexed sequence position -> 0-indexed ESM token position.

    ESM-2 prepends ``<cls>``, so residue *i* lives at token *i + 1*.
    """
    if seq_pos < 0:
        raise ValueError(f"seq_pos must be non-negative, got {seq_pos}")
    return seq_pos + N_PREFIX_TOKENS


def token_to_seq_pos(tok_pos: int) -> int:
    """0-indexed ESM token position -> 0-indexed sequence position."""
    seq_pos = tok_pos - N_PREFIX_TOKENS
    if seq_pos < 0:
        raise ValueError(
            f"token position {tok_pos} is a special token (<cls>), not a residue"
        )
    return seq_pos


def validate_sequence(sequence: str, allow_noncanonical: bool = True) -> str:
    """Normalise and check a protein sequence.

    Uppercases, strips whitespace, and (unless ``allow_noncanonical``) rejects
    anything outside the 20 canonical residues. Non-canonical characters that
    ESM-2 understands (X, B, U, Z, O) are kept as-is when allowed.
    """
    seq = "".join(sequence.split()).upper()
    if not seq:
        raise ValueError("empty sequence")
    extra = set("XBUZO") if allow_noncanonical else set()
    allowed = set(AA_ALPHABET) | extra
    bad = sorted(set(seq) - allowed)
    if bad:
        raise ValueError(
            f"sequence contains unsupported characters {bad}; "
            f"{'set allow_noncanonical=True' if not allow_noncanonical else 'clean the input'}"
        )
    return seq
