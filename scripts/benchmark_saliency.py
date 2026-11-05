#!/usr/bin/env python
"""Benchmark causal path patching against post-hoc saliency.

    python scripts/benchmark_saliency.py
    python scripts/benchmark_saliency.py model=esm2_35m data.source=synthetic
    python scripts/benchmark_saliency.py +n_variants=50 +ablation_fractions=[0.05,0.1,0.2]

Two metrics, both of which require *intervening* on the model rather than
merely ranking residues.

**Causal precision.** Take each method's top-n residues and mutate them one at a
time to alanine (glycine where the residue is already alanine). The precision is
the fraction whose mutation moves the model's variant-effect metric by more than
a threshold. A method that highlights residues which do nothing when changed
scores badly here regardless of how convincing its heatmap looks.

**Faithfulness.** Ablate each method's top-n residues together and measure how
far the metric falls, against ablating the same number of random residues. The
gap between the two curves is the quantity of interest; the absolute drop on its
own says more about the protein than about the method.

The random baseline is reported throughout. On faithfulness in particular it is
a stronger competitor than people expect, and a method that fails to separate
from it has not demonstrated anything.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from interpretability.baselines import (  # noqa: E402
    attention_saliency,
    integrated_gradients_saliency,
    random_saliency,
    sae_circuit_saliency,
)
from interpretability.circuit_extraction import discover_circuit  # noqa: E402
from interpretability.path_patching import CausalPatcher, PatchingSetup  # noqa: E402
from models.esm_hooks import ESMWrapper  # noqa: E402
from models.sparse_autoencoder import TopKSparseAutoencoder  # noqa: E402
from utils.dataloaders import load_variants  # noqa: E402
from utils.device import device_report, resolve_device  # noqa: E402
from utils.protein import AA_ALPHABET, Mutation, parse_mutation  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402

METHODS = ("causal_sae", "attention", "integrated_gradients", "random")


def alanine_substitution(sequence: str, seq_pos: int) -> Mutation | None:
    """Build an alanine-scanning mutation at ``seq_pos``.

    Alanine scanning is the standard functional probe: alanine removes the side
    chain past the beta carbon while leaving the backbone intact, so a large
    effect implicates the side chain rather than the fold. Positions already
    alanine are mutated to glycine instead, and positions holding a
    non-canonical residue are skipped rather than forced.
    """
    wt = sequence[seq_pos]
    if wt not in AA_ALPHABET:
        return None
    target = "G" if wt == "A" else "A"
    return parse_mutation(f"{wt}{seq_pos + 1}{target}")


@torch.no_grad()
def causal_precision(
    wrapper: ESMWrapper,
    sequence: str,
    residues_1indexed: list[int],
    *,
    threshold: float,
) -> tuple[float, list[float]]:
    """Fraction of selected residues whose mutation actually moves the model.

    Uses the masked-marginal score, which asks the model what belongs at a site
    given its context — the same readout the circuit was discovered against, so
    a high score here is not an artefact of measuring a different quantity.
    """
    effects: list[float] = []
    for residue in residues_1indexed:
        seq_pos = residue - 1
        if not 0 <= seq_pos < len(sequence):
            continue
        mutation = alanine_substitution(sequence, seq_pos)
        if mutation is None:
            continue
        score = wrapper.masked_marginal_score(
            sequence, seq_pos, mutation.wt_aa, mutation.mut_aa
        )
        effects.append(abs(score))
    if not effects:
        return float("nan"), []
    return float(np.mean([e > threshold for e in effects])), effects


@torch.no_grad()
def faithfulness_curve(
    wrapper: ESMWrapper,
    setup: PatchingSetup,
    scores: torch.Tensor,
    fractions: list[float],
    *,
    protected: set[int],
) -> list[float]:
    """Metric after masking the top-scoring fraction of residues.

    Masking (replacing with ``<mask>``) rather than deleting keeps the sequence
    length fixed, so the mutated position stays where the metric expects it and
    no positional shift contaminates the measurement. The mutated site itself is
    protected from ablation — masking it would destroy the readout rather than
    test the circuit.
    """
    seq_len = scores.shape[0]
    order = torch.argsort(scores, descending=True).tolist()
    mask_id = getattr(wrapper.tokenizer, "mask_token_id", None)
    if mask_id is None:
        raise RuntimeError("tokenizer has no mask token")

    out: list[float] = []
    for fraction in fractions:
        n = max(1, int(round(fraction * seq_len)))
        chosen = [p for p in order if (p + 1) not in protected][:n]
        ids = setup.corrupted.input_ids.clone()
        for seq_pos in chosen:
            ids[0, seq_pos + 1] = int(mask_id)  # +1 for <cls>
        from models.esm_hooks import TokenBatch

        batch = TokenBatch(ids, setup.corrupted.attention_mask, setup.corrupted.sequences)
        out.append(float(setup.metric(wrapper.model(**batch.as_model_kwargs()).logits)))
    return out


def find_sae(layer: int, sae_dir: Path | None) -> Path:
    if sae_dir is not None:
        p = Path(sae_dir) / f"sae_layer{layer}.pt"
        if p.exists():
            return p
        raise SystemExit(f"no SAE for layer {layer} under {sae_dir}")
    matches = sorted(
        (ROOT / "outputs").rglob(f"sae_layer{layer}.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        raise SystemExit(
            f"no trained SAE for layer {layer}; run `python scripts/train_sae.py` first"
        )
    return matches[0]


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.device)
    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    n_variants = int(cfg.get("n_variants", 25))
    top_n = int(cfg.get("top_n_residues", 10))
    effect_threshold = float(cfg.get("effect_threshold", 0.5))
    fractions = [float(f) for f in cfg.get("ablation_fractions", [0.02, 0.05, 0.1, 0.2, 0.3])]
    layers = [int(x) for x in cfg.get("bench_layers", [int(cfg.model.primary_layer)])]
    sae_dir = cfg.get("sae_dir", None)

    print(f"[bench] {device_report(device)}")
    print(f"[bench] model={cfg.model.hf_id}  layers={layers}  variants={n_variants}")

    # Attentions need the eager implementation to be readable.
    wrapper = ESMWrapper.from_pretrained(
        cfg.model.hf_id,
        device=device,
        cache_dir=str(cfg.paths.models),
        max_seq_len=int(cfg.model.max_seq_len),
    )
    try:
        wrapper.model.config._attn_implementation = "eager"
    except Exception:  # noqa: BLE001 - older transformers have no such attribute
        pass

    patchers = {}
    for layer in layers:
        path = find_sae(layer, Path(sae_dir) if sae_dir else None)
        sae = TopKSparseAutoencoder.load(
            path, device=device, expect_model=str(cfg.model.hf_id), expect_layer=layer
        )
        patchers[layer] = CausalPatcher(wrapper, sae, layer)
        print(f"[bench] layer {layer}: {path}")

    variants = load_variants(cfg.data, seed=int(cfg.seed))
    variants = [v for v in variants if len(v.wt_sequence) <= int(cfg.model.max_seq_len)]
    variants = variants[:n_variants]
    if not variants:
        raise SystemExit("no usable variants; check the data config")
    print(f"[bench] evaluating {len(variants)} variants")

    records: list[dict] = []
    for variant in tqdm(variants, desc="variants"):
        sequence = variant.wt_sequence
        seq_len = len(sequence)
        try:
            setup = PatchingSetup.from_mutation(wrapper, sequence, variant.mutation)
        except (ValueError, IndexError) as exc:
            print(f"[bench] skipping {variant.name}: {exc}")
            continue

        base = patchers[layers[0]].baselines(setup)
        gap = base["clean_metric"] - base["corrupted_metric"]
        if abs(gap) < 1e-3:
            # No causal effect to explain; including these would let every
            # method score identically on noise.
            continue

        # --- per-method residue scores ---------------------------------------
        scores: dict[str, torch.Tensor] = {}

        circuit = discover_circuit(
            patchers,
            setup,
            top_k_attribution=int(cfg.get("top_k_attribution", 48)),
            max_features_per_layer=int(cfg.get("max_features_per_layer", 3)),
            find_edges=False,
            progress=False,
        )
        residue_scores: dict[int, float] = {}
        for node in circuit.nodes:
            for residue, effect in zip(node.residues, node.position_effects, strict=False):
                residue_scores[residue] = max(residue_scores.get(residue, 0.0), abs(effect))
        scores["causal_sae"] = sae_circuit_saliency(
            list(residue_scores), list(residue_scores.values()), seq_len
        )

        try:
            scores["attention"] = attention_saliency(
                wrapper, setup.corrupted, query_token_pos=setup.token_pos
            )
        except RuntimeError as exc:
            print(f"[bench] attention unavailable: {exc}")
            scores["attention"] = random_saliency(seq_len, seed=int(cfg.seed))

        scores["integrated_gradients"] = integrated_gradients_saliency(
            wrapper, setup.corrupted, setup.metric, steps=int(cfg.get("ig_steps", 32))
        )
        scores["random"] = random_saliency(seq_len, seed=int(cfg.seed) + len(records))

        # Align lengths — saliency vectors are trimmed to the residue count, but
        # truncation can leave them shorter than the sequence.
        for name, vec in list(scores.items()):
            if vec.shape[0] < seq_len:
                padded = torch.zeros(seq_len)
                padded[: vec.shape[0]] = vec
                scores[name] = padded
            elif vec.shape[0] > seq_len:
                scores[name] = vec[:seq_len]

        protected = {variant.mutation.one_indexed_pos}

        record: dict = {
            "variant": variant.name,
            "mutation": variant.mutation.raw,
            "seq_len": seq_len,
            "gap": gap,
            "recovered_fraction": circuit.recovered_fraction,
            "n_circuit_residues": len(circuit.residues),
            "methods": {},
        }

        for name, vec in scores.items():
            top = (torch.argsort(vec, descending=True)[:top_n] + 1).tolist()
            precision, effects = causal_precision(
                wrapper, sequence, top, threshold=effect_threshold
            )
            curve = faithfulness_curve(
                wrapper, setup, vec, fractions, protected=protected
            )
            # Normalise the curve so variants of different effect sizes are
            # comparable: 0 = metric unchanged, 1 = fully collapsed to the
            # corrupted baseline's distance from clean.
            drops = [(base["corrupted_metric"] - c) / abs(gap) for c in curve]
            record["methods"][name] = {
                "top_residues": top,
                "causal_precision": precision,
                "mean_abs_effect": float(np.mean(effects)) if effects else float("nan"),
                "faithfulness_curve": curve,
                "normalized_drop": drops,
                "auc_drop": float(np.trapz(drops, fractions) / (fractions[-1] - fractions[0])),
            }

        records.append(record)

    if not records:
        raise SystemExit(
            "no variants produced a measurable effect; try data.source=synthetic or a "
            "different assay"
        )

    # --- aggregate ------------------------------------------------------------
    summary: dict[str, dict] = {}
    for method in METHODS:
        precisions = [
            r["methods"][method]["causal_precision"]
            for r in records
            if method in r["methods"] and not np.isnan(r["methods"][method]["causal_precision"])
        ]
        aucs = [r["methods"][method]["auc_drop"] for r in records if method in r["methods"]]
        curves = np.array(
            [r["methods"][method]["normalized_drop"] for r in records if method in r["methods"]]
        )
        summary[method] = {
            "causal_precision_mean": float(np.mean(precisions)) if precisions else float("nan"),
            "causal_precision_sem": (
                float(np.std(precisions) / np.sqrt(len(precisions))) if precisions else float("nan")
            ),
            "auc_drop_mean": float(np.mean(aucs)) if aucs else float("nan"),
            "mean_curve": curves.mean(axis=0).tolist() if curves.size else [],
            "n": len(precisions),
        }

    payload = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "fractions": fractions,
        "summary": summary,
        "records": records,
    }
    (out_dir / "benchmark.json").write_text(json.dumps(payload, indent=2, default=float))

    print(f"\n[bench] Causal precision (top-{top_n} residues) and faithfulness AUC")
    print(f"{'method':>22}  {'precision':>18}  {'AUC(drop)':>10}  {'n':>4}")
    for method in METHODS:
        s = summary[method]
        print(
            f"{method:>22}  {s['causal_precision_mean']:>10.3f} +/- {s['causal_precision_sem']:<5.3f}  "
            f"{s['auc_drop_mean']:>10.3f}  {s['n']:>4}"
        )

    _plot(payload, out_dir)
    print(f"\n[bench] artefacts in {out_dir}")


def _plot(payload: dict, out_dir: Path) -> None:
    """Figure 2: precision bars and faithfulness curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fractions = payload["fractions"]
    summary = payload["summary"]
    # Colour-blind-safe, and distinguishable in greyscale print.
    colors = {
        "causal_sae": "#0173B2",
        "attention": "#DE8F05",
        "integrated_gradients": "#029E73",
        "random": "#949494",
    }
    labels = {
        "causal_sae": "Causal SAE patching",
        "attention": "Raw attention",
        "integrated_gradients": "Integrated gradients",
        "random": "Random",
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    methods = [m for m in METHODS if summary.get(m, {}).get("n", 0) > 0]
    values = [summary[m]["causal_precision_mean"] for m in methods]
    errors = [summary[m]["causal_precision_sem"] for m in methods]
    ax.bar(
        range(len(methods)),
        values,
        yerr=errors,
        capsize=4,
        color=[colors[m] for m in methods],
    )
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([labels[m] for m in methods], rotation=20, ha="right")
    ax.set_ylabel("Causal precision")
    ax.set_title("Do the highlighted residues matter?")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    for method in methods:
        curve = summary[method]["mean_curve"]
        if curve:
            ax.plot(
                fractions, curve, marker="o", ms=4, color=colors[method], label=labels[method]
            )
    ax.set_xlabel("Fraction of residues ablated")
    ax.set_ylabel("Normalised metric drop")
    ax.set_title("Faithfulness")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"figure2_benchmark.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[bench] wrote {out_dir / 'figure2_benchmark.png'}")


if __name__ == "__main__":
    main()
