#!/usr/bin/env python
"""Train Top-K sparse autoencoders on ESM-2 residual activations.

    python scripts/train_sae.py                       # 8M model, layers from config
    python scripts/train_sae.py model=esm2_35m
    python scripts/train_sae.py sae.arch.k=64 sae.train.total_steps=50000
    python scripts/train_sae.py model.target_layers=[3]

One SAE is trained per layer in ``model.target_layers``. Checkpoints and metrics
land in the Hydra run directory under ``outputs/``.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# MPS fallback must be set before torch is imported anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydra  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from models.esm_hooks import ESMActivationExtractor, ESMWrapper  # noqa: E402
from models.sparse_autoencoder import (  # noqa: E402
    ActivationNormalizer,
    SAEConfig,
    TopKSparseAutoencoder,
)
from utils.dataloaders import ActivationBuffer, load_sequence_corpus  # noqa: E402
from utils.device import device_report, empty_cache, resolve_device  # noqa: E402
from utils.seeding import seed_everything  # noqa: E402
from validation.monosemanticity_metrics import evaluate_reconstruction  # noqa: E402


def build_lr_schedule(optimizer, warmup: int, total: int):
    """Linear warmup then cosine decay to 10% of peak.

    Warmup matters more than usual here: ``b_dec`` starts at the data mean and
    the encoder starts as the decoder transpose, so the first few hundred steps
    involve large, badly-scaled updates that can knock the decoder off the unit
    sphere faster than renormalisation recovers.
    """
    import math

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_layer(
    cfg: DictConfig,
    wrapper: ESMWrapper,
    sequences: list[str],
    layer_idx: int,
    out_dir: Path,
) -> dict:
    """Train a single SAE on one layer's residual stream."""
    device = wrapper.device
    arch, tcfg = cfg.sae.arch, cfg.sae.train

    extractor = ESMActivationExtractor(wrapper.model, [layer_idx])

    def extract(batch_sequences):
        """Forward pass -> [n_real_tokens, d_model], padding removed."""
        batch = wrapper.tokenize(list(batch_sequences))
        with torch.no_grad(), extractor:
            wrapper.model(**batch.as_model_kwargs())
            return extractor.flat(layer_idx, batch.attention_mask)

    buffer = ActivationBuffer(
        extract,
        sequences,
        d_model=wrapper.d_model,
        buffer_tokens=int(tcfg.buffer_tokens),
        batch_size=int(tcfg.batch_size),
        extract_batch_size=int(cfg.model.extract_batch_size),
        refill_at=float(tcfg.refill_at),
        loop=True,
        seed=int(cfg.seed) + layer_idx,
        device=device,
    )

    # Fit normalisation and b_dec on a real sample before the first step.
    warm = buffer.peek(min(int(tcfg.batch_size) * 4, int(tcfg.buffer_tokens)))
    normalizer = (
        ActivationNormalizer.fit(warm)
        if bool(tcfg.normalize_activations)
        else ActivationNormalizer()
    )
    print(f"[train] layer {layer_idx}: activation scale = {normalizer.scale:.6g}")

    sae_cfg = SAEConfig(
        d_in=wrapper.d_model,
        dict_mult=int(arch.dict_mult),
        k=int(arch.k),
        aux_k=int(arch.aux_k),
        aux_alpha=float(arch.aux_alpha),
        dead_after_tokens=int(arch.dead_after_tokens),
        center_input=bool(arch.center_input),
        model_name=str(cfg.model.hf_id),
        layer_idx=layer_idx,
        activation_scale=float(normalizer.scale),
    )
    sae = TopKSparseAutoencoder(sae_cfg).to(device)
    if bool(arch.init_b_dec_from_data):
        sae.init_b_dec_from_data(normalizer(warm))
    del warm

    optimizer = torch.optim.Adam(
        sae.parameters(), lr=float(tcfg.lr), betas=tuple(tcfg.betas)
    )
    scheduler = build_lr_schedule(
        optimizer, int(tcfg.warmup_steps), int(tcfg.total_steps)
    )

    sae.train()
    history: list[dict] = []
    started = time.time()
    total_steps = int(tcfg.total_steps)
    bar = tqdm(range(total_steps), desc=f"SAE L{layer_idx}")

    for step in bar:
        try:
            raw = next(buffer)
        except StopIteration:
            print(f"[train] corpus exhausted at step {step}")
            break
        x = normalizer(raw)

        loss, metrics = sae.loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Order matters: strip the radial gradient component, clip, step, then
        # renormalise. Clipping before the projection would mix the discarded
        # component into the norm computation.
        sae.remove_parallel_gradient()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), float(tcfg.grad_clip))
        optimizer.step()
        scheduler.step()
        sae.normalize_decoder()

        if step % int(tcfg.log_every) == 0:
            metrics["step"] = step
            metrics["lr"] = scheduler.get_last_lr()[0]
            history.append(metrics)
            bar.set_postfix(
                fvu=f"{metrics['fvu']:.4f}",
                dead=f"{metrics['dead_frac']:.1%}",
            )

        if step > 0 and step % int(tcfg.checkpoint_every) == 0:
            sae.save(out_dir / f"sae_layer{layer_idx}.pt")

    elapsed = time.time() - started

    # Held-out evaluation on freshly extracted activations.
    sae.eval()
    eval_tokens = int(cfg.sae.eval.num_eval_tokens)
    chunks, have = [], 0
    while have < eval_tokens:
        try:
            chunk = next(buffer)
        except StopIteration:
            break
        chunks.append(chunk.cpu())
        have += chunk.shape[0]
    if chunks:
        eval_acts = normalizer(torch.cat(chunks, dim=0)[:eval_tokens])
        report = evaluate_reconstruction(sae, eval_acts, layer_idx=layer_idx)
    else:
        report = None

    path = sae.save(out_dir / f"sae_layer{layer_idx}.pt")
    summary = {
        "layer": layer_idx,
        "checkpoint": str(path),
        "activation_scale": normalizer.scale,
        "train_seconds": elapsed,
        "steps": len(history) * int(tcfg.log_every),
        "final_dead_fraction": sae.dead_fraction(),
        "history": history,
        "eval": report.as_dict() if report else None,
    }
    if report:
        print(f"[train] layer {layer_idx}: {report}")
    print(f"[train] layer {layer_idx}: saved {path} ({elapsed / 60:.1f} min)")

    del sae, buffer
    empty_cache(device)
    return summary


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    seed_everything(int(cfg.seed))
    device = resolve_device(cfg.device)
    print(f"[train] {device_report(device)}")
    print(f"[train] model: {cfg.model.hf_id}")

    out_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    wrapper = ESMWrapper.from_pretrained(
        cfg.model.hf_id,
        device=device,
        cache_dir=str(cfg.paths.models),
        max_seq_len=int(cfg.model.max_seq_len),
    )
    print(f"[train] loaded: {wrapper.n_layers} layers, d_model={wrapper.d_model}")

    sequences = load_sequence_corpus(
        cfg.data.sae_corpus.fasta,
        min_len=int(cfg.data.sae_corpus.min_len),
        max_len=int(cfg.data.sae_corpus.max_len),
        limit=int(cfg.sae.train.num_sequences),
        allow_synthetic_fallback=bool(cfg.data.sae_corpus.allow_synthetic_fallback),
        seed=int(cfg.seed),
    )
    print(f"[train] corpus: {len(sequences)} sequences")

    summaries = []
    for layer_idx in list(cfg.model.target_layers):
        summaries.append(train_one_layer(cfg, wrapper, sequences, int(layer_idx), out_dir))

    (out_dir / "training_summary.json").write_text(json.dumps(summaries, indent=2))

    print("\n[train] Table 1 — reconstruction quality by layer")
    print(f"{'layer':>6}  {'FVU':>8}  {'EV':>8}  {'L0':>6}  {'dead':>7}  {'cos':>7}")
    for s in summaries:
        ev = s.get("eval")
        if not ev:
            continue
        print(
            f"{ev['layer']:>6}  {ev['fvu']:>8.4f}  {ev['explained_variance']:>8.4f}  "
            f"{ev['l0']:>6.1f}  {ev['dead_fraction']:>6.1%}  {ev['cosine_similarity']:>7.4f}"
        )
    print(f"\n[train] artefacts in {out_dir}")


if __name__ == "__main__":
    main()
