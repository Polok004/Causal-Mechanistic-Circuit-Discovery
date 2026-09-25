"""Shared fixtures.

The whole suite runs with **no network access and no pretrained weights**. That
is a deliberate design constraint, not a convenience: correctness of the hooks,
the SAE, the patching algebra and the structural alignment does not depend on
the weights being good, and a test suite that needs a 30 MB download is a test
suite nobody runs before committing.

The trick is that ``transformers`` ships the ESM-2 *architecture* as code. We
build an ``EsmForMaskedLM`` from a config with the same structure as
``esm2_t6_8M_UR50D`` — rotary position embeddings, token dropout, pre-LayerNorm
blocks, the 33-token alphabet — only much narrower, with random weights. Every
code path under test (layer resolution, hook placement, output-tuple handling,
residual-stream shapes, token offsets) exercises the real module, so a
transformers layout change breaks these tests exactly as it would break a real
run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.protein import ESM_VOCAB  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

# Small enough to be fast, wide enough that top-k over a dictionary is not
# degenerate.
TINY_HIDDEN = 32
TINY_LAYERS = 4
TINY_HEADS = 2


@pytest.fixture(autouse=True)
def _deterministic():
    seed_everything(1234)
    yield


@pytest.fixture(scope="session")
def vocab_file(tmp_path_factory) -> Path:
    """The real ESM-2 33-token alphabet, written to disk for EsmTokenizer."""
    path = tmp_path_factory.mktemp("esm") / "vocab.txt"
    path.write_text("\n".join(ESM_VOCAB) + "\n")
    return path


@pytest.fixture(scope="session")
def tokenizer(vocab_file):
    from transformers import EsmTokenizer

    return EsmTokenizer(vocab_file=str(vocab_file))


@pytest.fixture(scope="session")
def tiny_config():
    """An ESM-2-shaped config: same architecture family, tiny dimensions."""
    from transformers import EsmConfig

    return EsmConfig(
        vocab_size=len(ESM_VOCAB),
        hidden_size=TINY_HIDDEN,
        num_hidden_layers=TINY_LAYERS,
        num_attention_heads=TINY_HEADS,
        intermediate_size=TINY_HIDDEN * 2,
        max_position_embeddings=1026,
        position_embedding_type="rotary",  # ESM-2 uses rotary, not learned
        token_dropout=True,
        emb_layer_norm_before=False,
        pad_token_id=ESM_VOCAB.index("<pad>"),
        mask_token_id=ESM_VOCAB.index("<mask>"),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )


@pytest.fixture(scope="session")
def tiny_model(tiny_config):
    from transformers import EsmForMaskedLM

    torch.manual_seed(0)
    model = EsmForMaskedLM(tiny_config)
    model.eval()
    return model


@pytest.fixture
def wrapper(tiny_model, tokenizer):
    from models.esm_hooks import ESMWrapper

    return ESMWrapper(
        model=tiny_model, tokenizer=tokenizer, device=torch.device("cpu"), max_seq_len=256
    )


@pytest.fixture
def sequences() -> list[str]:
    """A few fixed sequences. Fixed, not random, so failures are reproducible."""
    return [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKR",
        "MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPN",
        "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
    ]


@pytest.fixture
def wt_sequence(sequences) -> str:
    return sequences[0]


@pytest.fixture
def sae(wrapper):
    """An untrained SAE of the right shape for the tiny model.

    Untrained is fine for the properties under test: sparsity, shape,
    invertibility of the splice algebra and gradient flow are all structural,
    not learned. Tests that genuinely need a fit dictionary train one inline.
    """
    from models.sparse_autoencoder import SAEConfig, TopKSparseAutoencoder

    cfg = SAEConfig(
        d_in=wrapper.d_model,
        dict_mult=4,
        k=8,
        aux_k=0,
        model_name="tiny-test",
        layer_idx=1,
    )
    return TopKSparseAutoencoder(cfg)


@pytest.fixture
def trained_sae(sae, activations):
    """The same SAE, briefly fit to the tiny model's own activations.

    A handful of tests make *directional* claims — patching the clean code into
    the corrupted run should move the metric toward the clean value — and those
    claims only hold once the dictionary spans the activations reasonably well.
    With random decoder directions the splice adds a vector unrelated to
    ``a_clean - a_corrupt``, and the sign of the result is a coin flip. Rather
    than weaken those tests to "something changed", we spend a second fitting
    the dictionary so the property under test is actually present.
    """
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
    sae.init_b_dec_from_data(activations)
    sae.train()
    g = torch.Generator().manual_seed(0)
    for _ in range(400):
        idx = torch.randint(0, activations.shape[0], (128,), generator=g)
        loss, _ = sae.loss(activations[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sae.remove_parallel_gradient()
        opt.step()
        sae.normalize_decoder()
    sae.eval()
    return sae


@pytest.fixture
def activations(wrapper, sequences) -> torch.Tensor:
    """Real residual-stream activations from the tiny model, ``[n_tokens, d]``."""
    from models.esm_hooks import ESMActivationExtractor

    batch = wrapper.tokenize(sequences)
    extractor = ESMActivationExtractor(wrapper.model, [1])
    with torch.no_grad(), extractor:
        wrapper.model(**batch.as_model_kwargs())
        return extractor.flat(1, batch.attention_mask).clone()


# --------------------------------------------------------------------------- #
# Structure fixtures                                                            #
# --------------------------------------------------------------------------- #


def _pdb_line(serial: int, resname: str, resseq: int, xyz, chain: str = "A") -> str:
    """One ATOM record in fixed-column PDB format.

    PDB is a fixed-width format that Biopython parses by character offset, so a
    "close enough" line yields wrong values rather than a parse error. The
    columns (1-indexed, as the spec numbers them) are::

        1-6    record name "ATOM  "      18-20  residue name
        7-11   serial number            22     chain id
        13-16  atom name                23-26  residue sequence number
        17     altLoc                   27     insertion code
                                        31-38  x   39-46  y   47-54  z
        55-60  occupancy                61-66  temperature factor
        77-78  element

    The altLoc column at 17 is the one that is easy to drop, and dropping it
    shifts the residue name and everything after it left by one.
    """
    x, y, z = xyz
    return (
        "ATOM  "                 # cols 1-6
        f"{serial:>5}"           # 7-11
        " "                      # 12
        f"{' CA ':<4}"           # 13-16 atom name
        " "                      # 17 altLoc
        f"{resname:>3}"          # 18-20
        " "                      # 21
        f"{chain}"               # 22
        f"{resseq:>4}"           # 23-26
        " "                      # 27 insertion code
        "   "                    # 28-30
        f"{x:>8.3f}{y:>8.3f}{z:>8.3f}"   # 31-54
        f"{1.00:>6.2f}{20.00:>6.2f}"     # 55-66
        "          "             # 67-76
        f"{'C':>2}"              # 77-78 element
        "\n"
    )


@pytest.fixture
def mini_pdb(tmp_path) -> Path:
    """A synthetic 40-residue structure with a known compact cluster.

    Geometry is constructed rather than downloaded so the expected answer is
    known exactly: residues 10, 12 and 15 are placed within ~4 A of each other
    (a "binding pocket"), while the rest of the chain runs along an extended
    helix. The clustering test must call the pocket significant and a spread-out
    residue set not significant.
    """
    import numpy as np

    n = 40
    # An extended alpha helix: rise 1.5 A per residue, radius 2.3 A, 100 deg turn.
    coords = []
    for i in range(n):
        angle = np.deg2rad(100.0 * i)
        coords.append([2.3 * np.cos(angle), 2.3 * np.sin(angle), 1.5 * i])
    coords = np.asarray(coords)

    # Pull residues 10, 12, 15 (1-indexed) into a tight pocket.
    pocket_center = np.array([8.0, 8.0, 20.0])
    for resseq, offset in ((10, [0.0, 0.0, 0.0]), (12, [2.4, 0.9, 0.5]), (15, [1.1, 2.6, -0.8])):
        coords[resseq - 1] = pocket_center + np.asarray(offset)

    residues = ["ALA", "GLY", "SER", "VAL", "LEU", "THR", "ILE", "PRO", "PHE", "HIS"]
    lines = ["HEADER    TEST STRUCTURE                          01-JAN-00   MINI\n"]
    for i in range(n):
        lines.append(_pdb_line(i + 1, residues[i % len(residues)], i + 1, coords[i]))
    lines.append("TER\nEND\n")

    path = tmp_path / "mini.pdb"
    path.write_text("".join(lines))
    return path


@pytest.fixture
def mini_structure(mini_pdb):
    from validation.pdb_aligner import load_structure

    return load_structure(mini_pdb)
