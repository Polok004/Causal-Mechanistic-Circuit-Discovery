#!/usr/bin/env python
"""Run the whole pipeline end to end with no downloads and no pretrained weights.

    python scripts/smoke_test.py

Builds an architecture-identical but tiny ESM-2 from ``EsmConfig`` (random
weights, no network), trains a small Top-K SAE on its residual stream, discovers
a circuit for a synthetic variant, and validates the result against a generated
structure. Takes well under a minute on an M2.

What this does and does not tell you: it proves the install works, every module
imports, the shapes line up and the stages compose. It says nothing about
whether the circuits are biologically meaningful — the weights are random, so
there is nothing real to find. Use it after ``pip install``, before spending an
hour on ``fetch_assets.py``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from interpretability.circuit_extraction import discover_circuit  # noqa: E402
from interpretability.path_patching import CausalPatcher, PatchingSetup  # noqa: E402
from models.esm_hooks import ESMActivationExtractor, ESMWrapper  # noqa: E402
from models.sparse_autoencoder import SAEConfig, TopKSparseAutoencoder  # noqa: E402
from utils.dataloaders import synthetic_variants  # noqa: E402
from utils.device import device_report, resolve_device  # noqa: E402
from utils.protein import ESM_VOCAB  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402
from validation.monosemanticity_metrics import evaluate_reconstruction  # noqa: E402
from validation.pdb_aligner import PDBStructure, spatial_clustering_test  # noqa: E402

LAYER = 2


def build_tiny_model(tmp: Path):
    """A real ESM-2 module graph at toy dimensions, built offline."""
    from transformers import EsmConfig, EsmForMaskedLM, EsmTokenizer

    vocab = tmp / "vocab.txt"
    vocab.write_text("\n".join(ESM_VOCAB) + "\n")
    tokenizer = EsmTokenizer(vocab_file=str(vocab))

    config = EsmConfig(
        vocab_size=len(ESM_VOCAB),
        hidden_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        intermediate_size=128,
        max_position_embeddings=1026,
        position_embedding_type="rotary",
        token_dropout=True,
        emb_layer_norm_before=False,
        pad_token_id=ESM_VOCAB.index("<pad>"),
        mask_token_id=ESM_VOCAB.index("<mask>"),
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0,
    )
    return EsmForMaskedLM(config), tokenizer


def main() -> int:
    import tempfile

    seed_everything(0)
    device = resolve_device("auto")
    print(f"[smoke] {device_report(device)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # --- 1. model ---------------------------------------------------------
        model, tokenizer = build_tiny_model(tmp)
        wrapper = ESMWrapper(model=model, tokenizer=tokenizer, device=device, max_seq_len=256)
        print(f"[smoke] 1/5 model: {wrapper.n_layers} layers, d_model={wrapper.d_model}")

        # --- 2. activations ---------------------------------------------------
        variants = synthetic_variants(n_proteins=12, variants_per_protein=2, seed=0)
        sequences = sorted({v.wt_sequence for v in variants})
        extractor = ESMActivationExtractor(wrapper.model, [LAYER])
        chunks = []
        for i in range(0, len(sequences), 4):
            batch = wrapper.tokenize(sequences[i : i + 4])
            with torch.no_grad(), extractor:
                wrapper.model(**batch.as_model_kwargs())
                chunks.append(extractor.flat(LAYER, batch.attention_mask).cpu())
        acts = torch.cat(chunks)
        print(f"[smoke] 2/5 activations: {tuple(acts.shape)}")

        # --- 3. SAE -----------------------------------------------------------
        sae = TopKSparseAutoencoder(
            SAEConfig(
                d_in=wrapper.d_model,
                dict_mult=4,
                k=8,
                aux_k=32,
                model_name="smoke-test",
                layer_idx=LAYER,
            )
        ).to(device)
        sae.init_b_dec_from_data(acts.to(device))

        opt = torch.optim.Adam(sae.parameters(), lr=3e-3)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 500)
        sae.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(500):
            idx = torch.randint(0, acts.shape[0], (256,), generator=g)
            loss, metrics = sae.loss(acts[idx].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            sae.remove_parallel_gradient()
            opt.step()
            sched.step()
            sae.normalize_decoder()
        sae.eval()

        report = evaluate_reconstruction(sae, acts.to(device), layer_idx=LAYER)
        print(f"[smoke] 3/5 SAE: {report}")

        checkpoint = sae.save(tmp / f"sae_layer{LAYER}.pt")
        reloaded = TopKSparseAutoencoder.load(checkpoint, device=device, expect_layer=LAYER)
        assert torch.allclose(
            sae(acts[:64].to(device)).recon, reloaded(acts[:64].to(device)).recon, atol=1e-6
        )

        # --- 4. circuit -------------------------------------------------------
        variant = max(variants, key=lambda v: v.metadata["is_critical"])
        patcher = CausalPatcher(wrapper, reloaded, LAYER)
        setup = PatchingSetup.from_mutation(wrapper, variant.wt_sequence, variant.mutation)
        base = patcher.baselines(setup)
        print(
            f"[smoke] 4/5 variant {variant.mutation.raw}: clean={base['clean_metric']:.4f} "
            f"mutant={base['corrupted_metric']:.4f}"
        )

        # The no-op control: patching zero features must leave the metric exactly
        # where it was. If error-preserving splicing is broken this is the first
        # thing that moves.
        control = patcher.direct_causal_effect(setup, [], cache=base)
        assert abs(control.dce) < 1e-4, f"empty patch was not a no-op (dce={control.dce})"

        circuit = discover_circuit(
            {LAYER: patcher},
            setup,
            top_k_attribution=16,
            max_features_per_layer=3,
            find_edges=False,
            progress=False,
        )
        circuit.sequence = variant.wt_sequence
        print(
            f"[smoke]     {len(circuit.nodes)} nodes, {len(circuit.residues)} residues, "
            f"joint recovery {circuit.recovered_fraction:.1%}"
        )

        out = ROOT / "outputs" / "circuits" / "smoke_test.json"
        circuit.save(out)

        # --- 5. geometry ------------------------------------------------------
        n = len(variant.wt_sequence)
        rng = np.random.default_rng(0)
        coords = np.cumsum(rng.normal(scale=3.0, size=(n, 3)), axis=0)
        structure = PDBStructure(
            pdb_id="SMOKE",
            chain_id="A",
            sequence=variant.wt_sequence,
            residue_numbers=list(range(1, n + 1)),
            ca_coords=coords,
        )
        indices = [r - 1 for r in circuit.residues if 0 < r <= n]
        if len(indices) >= 2:
            result = spatial_clustering_test(structure, indices, n_permutations=2000, seed=0)
            print(
                f"[smoke] 5/5 geometry: mean Ca {result.mean_pairwise_distance:.2f} A, "
                f"p={result.p_value:.3g} -> {result.verdict}"
            )
        else:
            print("[smoke] 5/5 geometry: too few residues for a clustering test (fine here)")

    print(f"\n[smoke] PASS — pipeline is wired correctly. Circuit written to {out}")
    print("[smoke] Weights were random, so the circuit itself means nothing. Next:")
    print("[smoke]   python scripts/fetch_assets.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
