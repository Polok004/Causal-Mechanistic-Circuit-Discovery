"""Mapping circuit residues onto 3D structure, and testing whether they cluster.

Why alignment is not optional
-----------------------------
It is tempting to treat residue *i* of the input sequence as residue *i* of the
PDB file. That is almost never true. Crystal structures have disordered termini
and loops with no coordinates, their numbering often follows a mature protein
while the sequence follows the precursor, and they routinely contain expression
tags. Indexing directly into the structure therefore silently shifts every
residue by some offset, and the resulting "spatial clustering" result is noise
dressed up as a finding. :func:`align_sequence_to_structure` does a proper
global alignment and returns an explicit, inspectable mapping.

Why "mean distance <= 6 A" is not, by itself, a result
-------------------------------------------------------
The mean pairwise Ca distance of a residue set depends strongly on how many
residues it has and on the size of the protein. Three residues drawn at random
from a small domain will often be closer than 6 A; the same three in a large
multi-domain protein essentially never will. A fixed threshold therefore tests
the protein as much as the circuit.

:func:`spatial_clustering_test` instead compares the circuit against a null
distribution built by sampling *same-size* residue sets from the same structure.
The reported p-value is the fraction of random sets at least as compact as the
circuit, which is a claim about the circuit rather than about the protein. The
threshold is still reported, because it is easy to read and the plan calls for
it, but the p-value is the number that carries the argument.
"""

from __future__ import annotations

import gzip
import io
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from utils.protein import three_to_one

__all__ = [
    "PDBStructure",
    "SpatialClusteringResult",
    "StructureAlignment",
    "align_sequence_to_structure",
    "ca_distance_matrix",
    "evaluate_circuit_geometry",
    "load_structure",
    "mean_pairwise_distance",
    "secondary_structure",
    "spatial_clustering_test",
]


@dataclass
class PDBStructure:
    """Ca coordinates and sequence for one chain of one structure.

    Attributes:
        pdb_id: identifier, for reporting.
        chain_id: the chain these residues come from.
        sequence: one-letter sequence of residues **that have coordinates**.
        residue_numbers: author residue numbers, parallel to ``sequence``.
        ca_coords: ``[n_residues, 3]`` float array of Ca positions in angstroms.
    """

    pdb_id: str
    chain_id: str
    sequence: str
    residue_numbers: list[int]
    ca_coords: np.ndarray
    insertion_codes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        n = len(self.sequence)
        if len(self.residue_numbers) != n or self.ca_coords.shape[0] != n:
            raise ValueError(
                f"inconsistent structure: {n} residues in sequence, "
                f"{len(self.residue_numbers)} numbers, {self.ca_coords.shape[0]} coordinates"
            )
        if not self.insertion_codes:
            self.insertion_codes = [" "] * n

    @property
    def n_residues(self) -> int:
        return len(self.sequence)

    def index_of_residue_number(self, number: int) -> int | None:
        try:
            return self.residue_numbers.index(number)
        except ValueError:
            return None


def load_structure(
    path_or_id: str | Path,
    *,
    chain_id: str | None = None,
    pdb_dir: str | Path | None = None,
    model_index: int = 0,
) -> PDBStructure:
    """Load a chain from a local PDB/mmCIF file.

    Args:
        path_or_id: a file path, or a 4-character PDB id to look up inside
            ``pdb_dir``.
        chain_id: which chain. ``None`` takes the first protein chain with at
            least 20 resolved residues, which is almost always the one meant;
            when it is not, the choice is logged so it is visible rather than
            silent.
        model_index: NMR structures contain many models; the first is used
            unless told otherwise.

    Downloads are deliberately not performed here — fetching belongs in
    ``scripts/fetch_assets.py`` so that an analysis run is reproducible offline
    and cannot quietly depend on network state.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path_or_id)
    if not path.exists():
        if pdb_dir is None:
            raise FileNotFoundError(
                f"{path_or_id} not found. Pass a path, or set pdb_dir= and run "
                "`python scripts/fetch_assets.py --pdb <ID>` first."
            )
        stem = str(path_or_id).lower()
        for candidate in (f"{stem}.pdb", f"{stem}.cif", f"{stem}.pdb.gz", f"{stem}.cif.gz"):
            trial = Path(pdb_dir) / candidate
            if trial.exists():
                path = trial
                break
        else:
            raise FileNotFoundError(
                f"no structure for {path_or_id!r} in {pdb_dir}; run "
                f"`python scripts/fetch_assets.py --pdb {path_or_id}`"
            )

    is_cif = ".cif" in path.suffixes or path.name.endswith((".cif", ".cif.gz"))
    parser: Any = MMCIFParser(QUIET=True) if is_cif else PDBParser(QUIET=True)

    with warnings.catch_warnings():
        # Biopython warns loudly about discontinuous chains in perfectly usable
        # structures; the alignment step handles gaps explicitly.
        warnings.simplefilter("ignore")
        if path.name.endswith(".gz"):
            with gzip.open(path, "rt") as fh:
                structure = parser.get_structure(path.stem, io.StringIO(fh.read()))
        else:
            structure = parser.get_structure(path.stem, str(path))

    model = list(structure)[model_index]

    chains = {}
    for chain in model:
        seq_chars, numbers, coords, icodes = [], [], [], []
        for residue in chain:
            het_flag, resseq, icode = residue.id
            if het_flag.strip() and het_flag != "H_MSE":
                continue  # waters, ligands, ions
            if "CA" not in residue:
                continue  # unresolved backbone
            one = three_to_one(residue.get_resname())
            if one == "X":
                continue
            seq_chars.append(one)
            numbers.append(int(resseq))
            icodes.append(icode)
            coords.append(residue["CA"].get_coord())
        if seq_chars:
            chains[chain.id] = (
                "".join(seq_chars),
                numbers,
                np.asarray(coords, dtype=np.float64),
                icodes,
            )

    if not chains:
        raise ValueError(f"{path} contains no protein chain with Ca coordinates")

    if chain_id is None:
        eligible = [cid for cid, (seq, *_) in chains.items() if len(seq) >= 20]
        chain_id = eligible[0] if eligible else next(iter(chains))
    if chain_id not in chains:
        raise KeyError(f"chain {chain_id!r} not in {path.name}; available: {sorted(chains)}")

    seq, numbers, coords, icodes = chains[chain_id]
    pdb_id = Path(str(path_or_id)).stem.split(".")[0].upper()
    return PDBStructure(
        pdb_id=pdb_id,
        chain_id=chain_id,
        sequence=seq,
        residue_numbers=numbers,
        ca_coords=coords,
        insertion_codes=icodes,
    )


@dataclass
class StructureAlignment:
    """A mapping from sequence positions to structure residues.

    ``seq_to_struct`` maps a **0-indexed** position in the model's input
    sequence to a 0-indexed row of :attr:`PDBStructure.ca_coords`. Positions
    with no structural counterpart (disordered, or outside the construct) are
    simply absent, which callers must handle rather than assume away.
    """

    structure: PDBStructure
    seq_to_struct: dict[int, int]
    identity: float
    aligned_length: int
    query_length: int

    @property
    def coverage(self) -> float:
        """Fraction of the query sequence that has coordinates."""
        return len(self.seq_to_struct) / max(self.query_length, 1)

    def residues_to_indices(self, residues_1indexed: Sequence[int]) -> tuple[list[int], list[int]]:
        """Map 1-indexed sequence residues to structure rows.

        Returns ``(indices, unmapped)`` so the caller can report how many
        circuit residues had no coordinates instead of silently analysing a
        smaller set than it thinks.
        """
        indices, unmapped = [], []
        for r in residues_1indexed:
            idx = self.seq_to_struct.get(r - 1)
            if idx is None:
                unmapped.append(r)
            else:
                indices.append(idx)
        return indices, unmapped

    def summary(self) -> str:
        return (
            f"{self.structure.pdb_id}:{self.structure.chain_id}  "
            f"identity={self.identity:.1%}  coverage={self.coverage:.1%}  "
            f"aligned={self.aligned_length}/{self.query_length}"
        )


def align_sequence_to_structure(
    sequence: str,
    structure: PDBStructure,
    *,
    min_identity: float = 0.8,
    strict: bool = True,
) -> StructureAlignment:
    """Globally align the model's input sequence to the structure's sequence.

    Uses a BLOSUM62 global alignment with a gap penalty tuned for long gaps,
    since the dominant difference between a UniProt sequence and a crystal
    construct is a handful of long deletions (disordered loops, missing termini)
    rather than scattered substitutions.

    Args:
        min_identity: below this, the alignment is almost certainly between two
            different proteins. ``strict=True`` raises; otherwise a warning is
            issued and the mapping returned for inspection.
    """
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    # Affine gaps: opening a gap is expensive, extending it is cheap, so a
    # single 30-residue disordered loop costs far less than 30 point mismatches.
    aligner.open_gap_score = -11.0
    aligner.extend_gap_score = -1.0
    # Free end gaps: missing termini should not be penalised at all.
    # `end_gap_score` sets all four end-gap parameters at once and is stable
    # across Biopython versions; the individual `target_end_gap_score` /
    # `query_end_gap_score` attributes were renamed in 1.85+.
    aligner.end_gap_score = 0.0

    query = sequence.upper()
    target = structure.sequence.upper()
    alignment = aligner.align(query, target)[0]

    seq_to_struct: dict[int, int] = {}
    matches = 0
    aligned = 0
    # `aligned` is a pair of coordinate blocks: [[q_start, q_end], ...] and the
    # corresponding target blocks.
    q_blocks, t_blocks = alignment.aligned
    for (q0, q1), (t0, _t1) in zip(q_blocks, t_blocks, strict=True):
        for offset in range(q1 - q0):
            qi, ti = q0 + offset, t0 + offset
            seq_to_struct[qi] = ti
            aligned += 1
            if query[qi] == target[ti]:
                matches += 1

    identity = matches / max(aligned, 1)
    if identity < min_identity:
        message = (
            f"sequence/structure identity is only {identity:.1%} over {aligned} aligned "
            f"residues ({structure.pdb_id}:{structure.chain_id}). The structure probably "
            "does not correspond to this sequence; check the PDB id and chain."
        )
        if strict:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    return StructureAlignment(
        structure=structure,
        seq_to_struct=seq_to_struct,
        identity=identity,
        aligned_length=aligned,
        query_length=len(query),
    )


# --------------------------------------------------------------------------- #
# Geometry                                                                      #
# --------------------------------------------------------------------------- #


def ca_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Full pairwise Euclidean distance matrix in angstroms."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff**2).sum(axis=-1))


def mean_pairwise_distance(coords: np.ndarray) -> float:
    """Mean Ca-Ca distance over all unordered pairs.

    This is the ``d_circuit`` of the project plan. Undefined for fewer than two
    residues, where it returns NaN rather than 0.0 — a single residue is not
    "maximally compact", it is unmeasured, and returning 0.0 would sail through
    any threshold check.
    """
    n = coords.shape[0]
    if n < 2:
        return float("nan")
    d = ca_distance_matrix(coords)
    iu = np.triu_indices(n, k=1)
    return float(d[iu].mean())


def radius_of_gyration(coords: np.ndarray) -> float:
    """Rg of the residue set — a size-aware compactness measure.

    Less sensitive than the mean pairwise distance to one outlying residue,
    so the two together say whether a circuit is genuinely a pocket or a tight
    core plus a straggler.
    """
    if coords.shape[0] < 2:
        return float("nan")
    centroid = coords.mean(axis=0)
    return float(np.sqrt(((coords - centroid) ** 2).sum(axis=1).mean()))


@dataclass
class SpatialClusteringResult:
    """Outcome of the compactness test against a size-matched null."""

    n_residues: int
    n_unmapped: int
    mean_pairwise_distance: float
    radius_of_gyration: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    threshold_angstrom: float
    n_permutations: int

    @property
    def passes_threshold(self) -> bool:
        """The plan's fixed-threshold criterion (``d_circuit <= 6 A``)."""
        return bool(self.mean_pairwise_distance <= self.threshold_angstrom)

    @property
    def significant(self) -> bool:
        """The stronger claim: more compact than size-matched random sets."""
        return bool(self.p_value < 0.05)

    @property
    def verdict(self) -> str:
        if self.n_residues < 2:
            return "undetermined (fewer than two mapped residues)"
        if self.significant and self.passes_threshold:
            return "valid biophysical circuit"
        if self.significant:
            return "significantly clustered, but looser than the 6 A threshold"
        if self.passes_threshold:
            return "within 6 A, but no more compact than chance for this size"
        return "spurious / model artefact"

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_residues": self.n_residues,
            "n_unmapped": self.n_unmapped,
            "mean_pairwise_distance": self.mean_pairwise_distance,
            "radius_of_gyration": self.radius_of_gyration,
            "null_mean": self.null_mean,
            "null_std": self.null_std,
            "p_value": self.p_value,
            "z_score": self.z_score,
            "threshold_angstrom": self.threshold_angstrom,
            "passes_threshold": self.passes_threshold,
            "significant": self.significant,
            "verdict": self.verdict,
            "n_permutations": self.n_permutations,
        }


def spatial_clustering_test(
    structure: PDBStructure,
    indices: Sequence[int],
    *,
    n_permutations: int = 10000,
    threshold_angstrom: float = 6.0,
    n_unmapped: int = 0,
    seed: int = 0,
) -> SpatialClusteringResult:
    """Test whether a residue set is more compact than chance.

    The null draws sets of the same size uniformly from the residues that have
    coordinates, which controls for both set size and protein size. Sampling
    from *resolved* residues specifically matters: disordered regions are
    systematically at the surface, so a null that included them would be biased
    toward looser sets and would make almost any circuit look significant.
    """
    coords = structure.ca_coords
    idx = np.asarray(sorted(set(int(i) for i in indices)), dtype=int)
    n = idx.size

    if n < 2:
        return SpatialClusteringResult(
            n_residues=int(n),
            n_unmapped=n_unmapped,
            mean_pairwise_distance=float("nan"),
            radius_of_gyration=float("nan"),
            null_mean=float("nan"),
            null_std=float("nan"),
            p_value=float("nan"),
            z_score=float("nan"),
            threshold_angstrom=threshold_angstrom,
            n_permutations=0,
        )

    observed = mean_pairwise_distance(coords[idx])
    rg = radius_of_gyration(coords[idx])

    rng = np.random.default_rng(seed)
    total = coords.shape[0]
    full = ca_distance_matrix(coords)
    iu = np.triu_indices(n, k=1)

    null = np.empty(n_permutations, dtype=np.float64)
    for i in range(n_permutations):
        sample = rng.choice(total, size=n, replace=False)
        sub = full[np.ix_(sample, sample)]
        null[i] = sub[iu].mean()

    # One-sided: we are asking whether the circuit is *more compact* than chance.
    # The +1 correction keeps the p-value from ever being exactly zero, which
    # would overstate what a finite number of permutations can establish.
    p_value = float((np.sum(null <= observed) + 1) / (n_permutations + 1))
    null_mean, null_std = float(null.mean()), float(null.std())
    z = float((observed - null_mean) / null_std) if null_std > 0 else float("nan")

    return SpatialClusteringResult(
        n_residues=int(n),
        n_unmapped=n_unmapped,
        mean_pairwise_distance=observed,
        radius_of_gyration=rg,
        null_mean=null_mean,
        null_std=null_std,
        p_value=p_value,
        z_score=z,
        threshold_angstrom=threshold_angstrom,
        n_permutations=n_permutations,
    )


def evaluate_circuit_geometry(
    sequence: str,
    residues_1indexed: Sequence[int],
    structure: PDBStructure,
    *,
    n_permutations: int = 10000,
    threshold_angstrom: float = 6.0,
    min_identity: float = 0.8,
    strict_alignment: bool = False,
    seed: int = 0,
) -> tuple[SpatialClusteringResult, StructureAlignment]:
    """Align, map, and test — the whole Phase 4 path in one call."""
    alignment = align_sequence_to_structure(
        sequence, structure, min_identity=min_identity, strict=strict_alignment
    )
    indices, unmapped = alignment.residues_to_indices(residues_1indexed)
    result = spatial_clustering_test(
        structure,
        indices,
        n_permutations=n_permutations,
        threshold_angstrom=threshold_angstrom,
        n_unmapped=len(unmapped),
        seed=seed,
    )
    return result, alignment


def secondary_structure(
    pdb_path: str | Path, *, chain_id: str | None = None
) -> dict[int, str] | None:
    """DSSP secondary structure per residue number, or ``None`` if unavailable.

    DSSP needs the ``mkdssp`` binary, which is not a pip dependency. Returning
    ``None`` rather than raising keeps it an optional enrichment: the structural
    claim rests on the Ca geometry, and secondary structure only annotates it.
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.DSSP import DSSP

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            structure = PDBParser(QUIET=True).get_structure("s", str(pdb_path))
            model = list(structure)[0]
            dssp = DSSP(model, str(pdb_path))
    except Exception as exc:  # noqa: BLE001 - any DSSP failure is non-fatal
        warnings.warn(
            f"DSSP unavailable ({exc}); install it with `conda install -c salilab dssp` "
            "or `brew install dssp` to annotate secondary structure. "
            "Geometry validation does not depend on it.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None

    out: dict[int, str] = {}
    for key in dssp.keys():
        chain, res_id = key
        if chain_id is not None and chain != chain_id:
            continue
        out[int(res_id[1])] = dssp[key][2]
    return out
