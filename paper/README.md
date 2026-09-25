# Paper

Deferred until there are real numbers to write up.

The figures and tables the workshop paper needs are already produced by the
pipeline, so writing it is a matter of assembling them rather than generating
anything new:

- **Table 1** — reconstruction quality per layer (FVU, explained variance, L0,
  dead fraction, cosine similarity). Printed at the end of
  `scripts/train_sae.py` and saved to `training_summary.json` in that run's
  Hydra output directory.
- **Figure 1** — the causal circuit on its 3D structure, with the Cα geometry
  test. Produced by `scripts/discover_circuits.py` (JSON) and rendered by
  `dashboard/app.py`.
- **Figure 2** — causal precision and faithfulness curves against raw attention,
  integrated gradients and random. Written as `figure2_benchmark.png` / `.pdf`
  by `scripts/benchmark_saliency.py`.

Before writing: the results section needs at least one experiment this
repository does not yet run, namely whether a circuit found for one variant
generalises to other variants of the same protein. A per-variant circuit is a
case study; a shared circuit is a finding.
