#!/usr/bin/env python
"""Discover the causal circuit behind one mutation, and check it against structure.

    python scripts/discover_circuits.py --pdb 1A2Y --mutation A45T --layer 3
    python scripts/discover_circuits.py --sequence MKTAY... --mutation H64A --layers 2 3 4
    python scripts/discover_circuits.py --pdb 1UBQ --mutation L67A --model esm2_35m

This is the single-command demo. It runs prune -> attribution rank -> exact
patching -> greedy selection -> positional localisation -> edge discovery, then
maps the resulting residues onto the structure and tests whether they are more
spatially clustered than size-matched random residue sets.

argparse rather than Hydra here on purpose: this is the entry point people run
by hand with a PDB id and a mutation, and ``--mutation A45T`` reads better than
``+mutation=A45T``. The training and benchmarking scripts, which are swept, use
Hydra.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from omegaconf import OmegaConf  # noqa: E402

from interpretability.circuit_extraction import discover_circuit  # noqa: E402
from interpretability.path_patching import CausalPatcher, PatchingSetup  # noqa: E402
from models.esm_hooks import ESMWrapper  # noqa: E402
from models.sparse_autoencoder import TopKSparseAutoencoder  # noqa: E402
from utils.device import device_report, resolve_device  # noqa: E402
from utils.protein import parse_mutation  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402
from validation.pdb_aligner import evaluate_circuit_geometry, load_structure  # noqa: E402


def load_model_cfg(name: str):
    path = ROOT / "configs" / "model" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (ROOT / "configs" / "model").glob("*.yaml"))
        raise SystemExit(f"unknown model config {name!r}; available: {available}")
    return OmegaConf.load(path)


def find_sae(sae_dir: Path | None, layer: int) -> Path:
    """Locate a trained SAE checkpoint for ``layer``.

    Searches the explicit directory if given, otherwise the most recent Hydra
    run under ``outputs/``. Falling back to the newest run is convenient and
    also a little dangerous, so the chosen path is always printed.
    """
    if sae_dir is not None:
        candidate = Path(sae_dir) / f"sae_layer{layer}.pt"
        if candidate.exists():
            return candidate
        raise SystemExit(f"no SAE for layer {layer} in {sae_dir}")

    matches = sorted(
        (ROOT / "outputs").rglob(f"sae_layer{layer}.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SystemExit(
            f"no trained SAE found for layer {layer}. Train one first:\n"
            f"    python scripts/train_sae.py model.target_layers=[{layer}]\n"
            "or point --sae-dir at a directory containing sae_layer*.pt"
        )
    return matches[0]


def sequence_from_structure(pdb_id: str, pdb_dir: Path, chain: str | None) -> tuple[str, object]:
    structure = load_structure(pdb_id, chain_id=chain, pdb_dir=pdb_dir)
    return structure.sequence, structure


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdb", help="PDB id; the chain sequence is used as the wild type")
    src.add_argument("--sequence", help="wild-type sequence given directly")
    src.add_argument("--fasta", help="FASTA file whose first record is the wild type")

    p.add_argument("--mutation", required=True, help="substitution, 1-indexed, e.g. A45T")
    p.add_argument("--chain", default=None, help="PDB chain id (default: first protein chain)")
    p.add_argument("--model", default="esm2_8m", help="model config name")
    p.add_argument("--layer", type=int, default=None, help="single layer to search")
    p.add_argument("--layers", type=int, nargs="*", default=None, help="several layers to search")
    p.add_argument("--sae-dir", default=None, help="directory holding sae_layer*.pt")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--top-k-attribution", type=int, default=64)
    p.add_argument("--max-features-per-layer", type=int, default=4)
    p.add_argument("--min-gain", type=float, default=0.01)
    p.add_argument("--position-threshold", type=float, default=0.1)
    p.add_argument("--no-edges", action="store_true", help="skip edge discovery")
    p.add_argument("--permutations", type=int, default=10000)
    p.add_argument("--distance-threshold", type=float, default=6.0)
    p.add_argument(
        "--splice-mode",
        choices=["error_preserving", "replace"],
        default="error_preserving",
        help="error_preserving cancels SAE reconstruction error; replace is the naive variant",
    )
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    seed_everything(args.seed)
    device = resolve_device(args.device)
    model_cfg = load_model_cfg(args.model)
    layers = args.layers or ([args.layer] if args.layer is not None else list(model_cfg.target_layers))

    print(f"[circuits] {device_report(device)}")
    print(f"[circuits] model={model_cfg.hf_id}  layers={layers}")

    # --- wild-type sequence ---------------------------------------------------
    structure = None
    if args.pdb:
        sequence, structure = sequence_from_structure(
            args.pdb, ROOT / "assets" / "pdb", args.chain
        )
        print(
            f"[circuits] {structure.pdb_id} chain {structure.chain_id}: "
            f"{structure.n_residues} resolved residues"
        )
    elif args.fasta:
        from utils.dataloaders import iter_fasta

        header, sequence = next(iter_fasta(args.fasta))
        print(f"[circuits] wild type from {args.fasta}: {header[:60]}")
    else:
        sequence = args.sequence

    mutation = parse_mutation(args.mutation)
    if not 0 <= mutation.seq_pos < len(sequence):
        raise SystemExit(
            f"mutation {mutation.raw} is outside the {len(sequence)}-residue sequence"
        )
    observed = sequence[mutation.seq_pos]
    if observed != mutation.wt_aa:
        hint = ""
        if args.pdb:
            hint = (
                "\nNote: with --pdb the sequence is the *resolved* chain, so its numbering "
                "starts at the first residue with coordinates and may not match the "
                "author numbering in the PDB file. Pass --sequence explicitly if your "
                "mutation uses author numbering."
            )
        raise SystemExit(
            f"mutation {mutation.raw} expects {mutation.wt_aa} at position "
            f"{mutation.one_indexed_pos} but the sequence has {observed}.{hint}"
        )

    # --- model and SAEs -------------------------------------------------------
    wrapper = ESMWrapper.from_pretrained(
        model_cfg.hf_id,
        device=device,
        cache_dir=str(ROOT / "assets" / "models"),
        max_seq_len=int(model_cfg.max_seq_len),
    )

    patchers = {}
    for layer in layers:
        path = find_sae(Path(args.sae_dir) if args.sae_dir else None, layer)
        print(f"[circuits] layer {layer}: {path}")
        sae = TopKSparseAutoencoder.load(
            path, device=device, expect_model=str(model_cfg.hf_id), expect_layer=layer
        )
        patchers[layer] = CausalPatcher(wrapper, sae, layer, splice_mode=args.splice_mode)

    # --- discovery ------------------------------------------------------------
    setup = PatchingSetup.from_mutation(wrapper, sequence, mutation)
    base = patchers[layers[0]].baselines(setup)
    print(
        f"[circuits] metric: clean={base['clean_metric']:.4f}  "
        f"mutant={base['corrupted_metric']:.4f}  "
        f"gap={base['clean_metric'] - base['corrupted_metric']:.4f}"
    )
    if abs(base["clean_metric"] - base["corrupted_metric"]) < 1e-3:
        print(
            "[circuits] WARNING: the mutation barely moves the metric, so there is no "
            "causal effect to decompose. Normalised DCE values will be unstable. "
            "Pick a variant the model actually responds to."
        )

    circuit = discover_circuit(
        patchers,
        setup,
        top_k_attribution=args.top_k_attribution,
        max_features_per_layer=args.max_features_per_layer,
        min_gain=args.min_gain,
        position_threshold=args.position_threshold,
        find_edges=not args.no_edges,
        progress=not args.quiet,
    )
    circuit.pdb_id = structure.pdb_id if structure else None

    print(f"\n[circuits] {len(circuit.nodes)} nodes, {len(circuit.edges)} edges")
    print(f"[circuits] joint recovery: {circuit.recovered_fraction:.1%} of the mutation's effect")
    for node in sorted(circuit.nodes, key=lambda n: abs(n.dce), reverse=True):
        print(
            f"  L{node.layer} f{node.feature:<6}  DCE={node.dce:+.4f}  "
            f"({node.normalized_dce:+.1%})  residues={node.residues[:10]}"
        )
    print(f"[circuits] circuit residues (1-indexed): {circuit.residues}")

    # --- biophysical validation ----------------------------------------------
    geometry = None
    if structure is not None and len(circuit.residues) >= 2:
        result, alignment = evaluate_circuit_geometry(
            sequence,
            circuit.residues,
            structure,
            n_permutations=args.permutations,
            threshold_angstrom=args.distance_threshold,
            strict_alignment=False,
            seed=args.seed,
        )
        geometry = result.as_dict()
        print(f"\n[circuits] structure alignment: {alignment.summary()}")
        print(
            f"[circuits] mean pairwise Ca distance: {result.mean_pairwise_distance:.2f} A "
            f"(random same-size sets: {result.null_mean:.2f} +/- {result.null_std:.2f} A)"
        )
        print(
            f"[circuits] p={result.p_value:.4g}  z={result.z_score:+.2f}  "
            f"Rg={result.radius_of_gyration:.2f} A"
        )
        if result.n_unmapped:
            print(
                f"[circuits] NOTE: {result.n_unmapped} circuit residue(s) had no coordinates "
                "and were excluded from the geometry test"
            )
        print(f"[circuits] verdict: {result.verdict}")
    elif structure is not None:
        print("\n[circuits] too few circuit residues for a geometry test")

    # --- output ---------------------------------------------------------------
    out_path = Path(
        args.out
        or ROOT / "outputs" / "circuits" / f"{(structure.pdb_id if structure else 'seq')}_{mutation.raw}.json"
    )
    payload = circuit.as_dict()
    payload["geometry"] = geometry
    payload["args"] = vars(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[circuits] wrote {out_path}")
    print("[circuits] view it with:  python -m streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
