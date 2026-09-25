"""Tests for structural alignment, geometry and the data loaders.

The alignment tests are the important ones. Treating sequence position *i* as
structure residue *i* is the default mistake in this kind of analysis, and it
produces a result that is wrong by a constant offset while looking entirely
normal. These tests construct structures with known gaps and known offsets and
check that the mapping recovers them.
"""

from __future__ import annotations

import numpy as np
import pytest

from utils.dataloaders import synthetic_corpus, synthetic_variants
from utils.protein import three_to_one, validate_sequence
from validation.pdb_aligner import (
    align_sequence_to_structure,
    ca_distance_matrix,
    mean_pairwise_distance,
    radius_of_gyration,
    spatial_clustering_test,
)

# --------------------------------------------------------------------------- #
# Parsing                                                                       #
# --------------------------------------------------------------------------- #


def test_load_structure_reads_coordinates(mini_structure):
    assert mini_structure.n_residues == 40
    assert mini_structure.ca_coords.shape == (40, 3)
    assert len(mini_structure.sequence) == 40
    assert mini_structure.residue_numbers[0] == 1
    assert mini_structure.residue_numbers[-1] == 40


def test_three_to_one_handles_modified_residues():
    assert three_to_one("ALA") == "A"
    assert three_to_one("MSE") == "M"      # selenomethionine
    assert three_to_one("HOH") == "X"      # water -> unknown, not a crash
    assert three_to_one("ala") == "A"      # case insensitive


# --------------------------------------------------------------------------- #
# Alignment                                                                     #
# --------------------------------------------------------------------------- #


def test_identical_sequences_align_one_to_one(mini_structure):
    alignment = align_sequence_to_structure(mini_structure.sequence, mini_structure)
    assert alignment.identity == pytest.approx(1.0)
    assert alignment.coverage == pytest.approx(1.0)
    assert all(k == v for k, v in alignment.seq_to_struct.items())


def test_alignment_recovers_an_n_terminal_offset(mini_structure):
    """The construct has 10 extra residues the crystal does not resolve.

    Naive indexing would map every circuit residue 10 positions too early. The
    alignment must shift the mapping by exactly 10.
    """
    prefix = "GGGGGGGGGG"
    query = prefix + mini_structure.sequence
    alignment = align_sequence_to_structure(query, mini_structure)

    for struct_idx in range(mini_structure.n_residues):
        assert alignment.seq_to_struct[struct_idx + len(prefix)] == struct_idx
    # The unresolved tag maps nowhere.
    assert all(i not in alignment.seq_to_struct for i in range(len(prefix)))


def test_alignment_handles_an_internal_gap(mini_structure):
    """A disordered loop present in the sequence but missing from the structure."""
    seq = mini_structure.sequence
    query = seq[:20] + "WWWWWWWW" + seq[20:]
    alignment = align_sequence_to_structure(query, mini_structure, min_identity=0.5)

    assert alignment.seq_to_struct[0] == 0
    # Residues after the insertion are shifted by its length in query space.
    assert alignment.seq_to_struct[20 + 8] == 20
    assert alignment.coverage < 1.0


def test_alignment_rejects_an_unrelated_sequence(mini_structure):
    unrelated = "W" * mini_structure.n_residues
    with pytest.raises(ValueError, match="identity"):
        align_sequence_to_structure(unrelated, mini_structure, min_identity=0.8, strict=True)


def test_alignment_warns_instead_of_raising_when_not_strict(mini_structure):
    unrelated = "W" * mini_structure.n_residues
    with pytest.warns(RuntimeWarning, match="identity"):
        align_sequence_to_structure(unrelated, mini_structure, min_identity=0.8, strict=False)


def test_residues_to_indices_reports_unmapped(mini_structure):
    prefix = "GGGGGGGGGG"
    alignment = align_sequence_to_structure(prefix + mini_structure.sequence, mini_structure)
    indices, unmapped = alignment.residues_to_indices([1, 2, 15, 20])
    assert unmapped == [1, 2]          # inside the unresolved tag
    assert indices == [4, 9]           # 15 -> struct 4, 20 -> struct 9


# --------------------------------------------------------------------------- #
# Geometry                                                                      #
# --------------------------------------------------------------------------- #


def test_distance_matrix_is_symmetric_with_zero_diagonal():
    coords = np.random.default_rng(0).normal(size=(10, 3)) * 5
    d = ca_distance_matrix(coords)
    assert np.allclose(d, d.T)
    assert np.allclose(np.diag(d), 0.0)


def test_mean_pairwise_distance_on_a_known_triangle():
    coords = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])
    # sides 3, 4, 5 -> mean 4
    assert mean_pairwise_distance(coords) == pytest.approx(4.0)


def test_mean_pairwise_distance_is_nan_for_a_single_residue():
    """Not 0.0 — a single residue is unmeasured, and 0.0 would pass any threshold."""
    assert np.isnan(mean_pairwise_distance(np.zeros((1, 3))))
    assert np.isnan(radius_of_gyration(np.zeros((1, 3))))


def test_planted_pocket_is_significantly_clustered(mini_structure):
    """Residues 10, 12, 15 were constructed to sit within a few angstroms."""
    indices = [9, 11, 14]  # 0-indexed
    result = spatial_clustering_test(
        mini_structure, indices, n_permutations=2000, seed=0
    )
    assert result.n_residues == 3
    assert result.mean_pairwise_distance < 6.0
    assert result.passes_threshold
    assert result.significant
    assert result.p_value < 0.05
    assert result.z_score < 0
    assert result.verdict == "valid biophysical circuit"


def test_spread_out_residues_are_not_significant(mini_structure):
    """Residues along the extended helix must not read as a pocket."""
    indices = [0, 13, 26, 39]
    result = spatial_clustering_test(
        mini_structure, indices, n_permutations=2000, seed=0
    )
    assert not result.significant or result.mean_pairwise_distance > 6.0
    assert result.verdict != "valid biophysical circuit"


def test_p_value_is_never_exactly_zero(mini_structure):
    """A finite permutation test cannot establish p = 0."""
    result = spatial_clustering_test(mini_structure, [9, 11, 14], n_permutations=100, seed=0)
    assert result.p_value > 0.0
    assert result.p_value >= 1.0 / 101


def test_clustering_test_degrades_gracefully_below_two_residues(mini_structure):
    result = spatial_clustering_test(mini_structure, [5], n_permutations=100)
    assert result.n_residues == 1
    assert np.isnan(result.mean_pairwise_distance)
    assert "undetermined" in result.verdict


def test_evaluate_circuit_geometry_end_to_end(mini_structure):
    from validation.pdb_aligner import evaluate_circuit_geometry

    result, alignment = evaluate_circuit_geometry(
        mini_structure.sequence,
        [10, 12, 15],  # 1-indexed
        mini_structure,
        n_permutations=1000,
        seed=0,
    )
    assert alignment.identity == pytest.approx(1.0)
    assert result.n_unmapped == 0
    assert result.significant


# --------------------------------------------------------------------------- #
# Data loaders                                                                  #
# --------------------------------------------------------------------------- #


def test_synthetic_corpus_is_valid_and_reproducible():
    a = synthetic_corpus(20, seed=7)
    b = synthetic_corpus(20, seed=7)
    assert a == b
    for seq in a:
        validate_sequence(seq, allow_noncanonical=False)


def test_synthetic_variants_are_internally_consistent():
    """Every mutation must apply cleanly to its own wild type.

    This is the invariant that catches numbering bugs in the generator, which
    would otherwise present as unexplainable patching results later.
    """
    variants = synthetic_variants(n_proteins=5, variants_per_protein=4, seed=0)
    assert len(variants) == 20
    for variant in variants:
        mutant = variant.mut_sequence
        assert len(mutant) == len(variant.wt_sequence)
        pos = variant.mutation.seq_pos
        assert mutant[pos] == variant.mutation.mut_aa
        assert variant.wt_sequence[pos] == variant.mutation.wt_aa
        assert variant.binary_label in (0, 1)


def test_synthetic_variants_label_motif_residues_as_critical():
    variants = synthetic_variants(n_proteins=8, variants_per_protein=6, motif="HExxH", seed=1)
    critical = [v for v in variants if v.metadata["is_critical"]]
    assert critical, "the generator should produce some critical-site variants"
    for variant in critical:
        assert variant.binary_label == 0
        assert variant.mutation.one_indexed_pos in variant.metadata["critical_residues_1indexed"]


def test_activation_buffer_shuffles_and_bounds_memory():
    import torch

    from utils.dataloaders import ActivationBuffer

    calls = {"n": 0}

    def extract(seqs):
        calls["n"] += 1
        # One distinguishable row per sequence.
        return torch.stack([torch.full((4,), float(len(s))) for s in seqs])

    sequences = [f"{'A' * (i + 10)}" for i in range(200)]
    buffer = ActivationBuffer(
        extract,
        sequences,
        d_model=4,
        buffer_tokens=64,
        batch_size=8,
        extract_batch_size=8,
        loop=True,
        seed=0,
    )
    batch = next(buffer)
    assert batch.shape == (8, 4)
    # Shuffled: consecutive rows should not be in the order the sequences were.
    values = batch[:, 0].tolist()
    assert values != sorted(values)
    assert calls["n"] > 0


def test_activation_buffer_stops_when_not_looping():
    import torch

    from utils.dataloaders import ActivationBuffer

    buffer = ActivationBuffer(
        lambda seqs: torch.ones(len(seqs), 4),
        ["AAAA"] * 4,
        d_model=4,
        buffer_tokens=16,
        batch_size=8,
        extract_batch_size=2,
        loop=False,
        seed=0,
    )
    batches = list(buffer)
    assert len(batches) == 0 or all(b.shape == (8, 4) for b in batches)


def test_iter_fasta_roundtrip(tmp_path):
    from utils.dataloaders import iter_fasta

    path = tmp_path / "x.fasta"
    path.write_text(">one desc\nMKTA\nYIAK\n>two\nMVLS\n")
    records = list(iter_fasta(path))
    assert records == [("one desc", "MKTAYIAK"), ("two", "MVLS")]
