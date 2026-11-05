# CausalMech-Bio

**Causal mechanistic circuit discovery in protein language models.**

Attention maps tell you what a protein language model *looked at*. They do not
tell you what it *used*. This repository finds the computational subgraphs —
circuits — that ESM-2 actually runs when it evaluates a point mutation, by
training sparse autoencoders on its residual stream and then intervening on
individual dictionary features with path patching. Circuits are then checked
against 3D structure: if the residues a circuit routes through form a real
pocket, the model has learned biophysics; if they are scattered, it has learned
a dataset shortcut.

Everything runs locally on Apple Silicon with native MPS acceleration. The
default configuration is sized for an 8 GB M2.

---

## New to this? Read the guide

**[`docs/GUIDE.md`](docs/GUIDE.md)** — or **[`docs/GUIDE.pdf`](docs/GUIDE.pdf)**
for the typeset version — is a complete walkthrough written for a reader with no
prior background. It defines every term it uses, from amino acids and the PDB
through transformers, sparse autoencoders and the causal-inference machinery,
and explains why each design decision was made rather than only what it does.

Part 0 is two pages and assumes nothing. Part 9 is a self-contained run guide.
Appendix A is a glossary. The rest fills in between.

The README below is the reference; the guide is the explanation.

---

## Quickstart

```bash
conda create -n causalmech python=3.10 -y
conda activate causalmech

pip install -r requirements.txt
pip install -e .

python scripts/smoke_test.py          # verifies the install, no downloads
```

`smoke_test.py` builds a tiny ESM-2 from config (random weights, no network),
trains an SAE on it, discovers a circuit and validates it — the whole pipeline
in under a minute. Run it before spending an hour downloading assets.

Then the real thing:

```bash
python scripts/fetch_assets.py --all                          # ESM-2, PDB, UniProt, ProteinGym
python scripts/train_sae.py                                   # one SAE per target layer
python scripts/discover_circuits.py --pdb 1A2Y --mutation A45T --layer 3
python -m streamlit run dashboard/app.py                      # inspect it in 3D
```

`make help` lists the same steps as targets.

---

## What the pipeline does

```
  Wild-type S_wt ──► ESM-2 ──► residual stream @ layer L ──► SAE ──► z_clean
                                                                       │
                                                                       │ restore feature f
                                                                       ▼
  Mutant   S_mut ──► ESM-2 ──► residual stream @ layer L ──► SAE ──► z_corrupt
                                    │                                  │
                                    └──── a + (decode(z_patched) − decode(z_corrupt)) ────┐
                                                                                          ▼
                                                                          Δ variant-effect metric
                                                                                = DCE(f)
```

1. **Hook the residual stream.** `src/models/esm_hooks.py` attaches forward
   hooks to ESM-2's transformer blocks. Because ESM-2 is pre-LayerNorm, a block's
   output *is* the residual stream, so the tensor we read is the tensor we later
   write to.
2. **Learn a dictionary.** `src/models/sparse_autoencoder.py` trains a Top-K SAE
   with a unit-norm decoder and an AuxK dead-latent revival term. Top-K fixes
   L0 = k exactly, which removes the sparsity-coefficient sweep and makes layers
   directly comparable.
3. **Intervene.** `src/interpretability/path_patching.py` restores individual
   clean feature values inside the mutant forward pass and measures the change
   in a variant-effect metric — the direct causal effect.
4. **Assemble a circuit.** `src/interpretability/circuit_extraction.py` prunes
   to active features, ranks them with attribution patching, confirms the
   shortlist with exact patching, then greedily grows a feature set by *joint*
   recovery and localises each feature to specific residues.
5. **Ground it in structure.** `src/validation/pdb_aligner.py` aligns the
   sequence to a PDB chain and tests whether the circuit's residues are more
   spatially clustered than size-matched random residue sets.
6. **Benchmark.** `scripts/benchmark_saliency.py` compares all of this against
   raw attention, integrated gradients and a random baseline on causal precision
   and faithfulness.

---

## Five decisions that differ from the obvious implementation

These are the places where the straightforward version of this pipeline is
wrong in ways that produce believable numbers rather than errors. Each is
documented at the relevant call site; they are collected here because they are
the substance of the method.

### 1. Error-preserving splicing

The obvious patch replaces the layer's activation with the SAE's decode of the
edited latents. That injects the SAE's reconstruction error into the measurement,
and for a single feature the error term is usually *larger* than the effect being
measured. We instead add only the difference:

```
a_patched = a_corrupt + (decode(z_patched) − decode(z_corrupt))
```

The reconstruction error is identical in both decodes and cancels exactly. The
consequence is testable and tested: patching *zero* features is then a perfect
no-op, which it is not under the naive scheme (`test_error_preserving_splice_cancels_reconstruction_error`,
`test_naive_splice_does_not_cancel_reconstruction_error`). Pass
`--splice-mode replace` to reproduce the naive behaviour and see the gap.

### 2. The corrupted baseline is measured before any hook exists

A PyTorch forward hook fires on *every* forward pass until removed. Computing
the unpatched baseline inside the patched context — easy to do, since both are
just `model(**batch)` calls — makes the baseline equal the patched value and
every direct effect exactly zero. `CausalPatcher.baselines()` runs first and
caches, and every hook in the codebase is scoped by a context manager so an
exception cannot leak one into the rest of the session.

### 3. Spatial clustering is tested against a size-matched null, not a fixed threshold

"Mean pairwise Cα distance ≤ 6 Å" depends heavily on how many residues are in
the set and how large the protein is. Three residues drawn at random from a
small domain are often within 6 Å; the same three in a large multi-domain
protein essentially never are. So the threshold partly tests the protein rather
than the circuit.

`spatial_clustering_test` therefore builds a null by sampling same-size residue
sets from the same structure and reports an empirical p-value and z-score. The
6 Å threshold is still reported — it is easy to read — but the p-value is what
carries the claim. Sampling is restricted to *resolved* residues, because
disordered regions are systematically surface-exposed and including them would
bias the null toward looser sets and make almost anything look significant.

### 4. Sequence position ≠ structure residue number

Crystal structures have disordered termini and loops with no coordinates, often
number according to a mature protein while the sequence follows the precursor,
and routinely contain expression tags. Indexing straight into the structure
shifts every residue by a constant and produces a clustering result that is
pure noise while looking entirely normal. `align_sequence_to_structure` runs a
BLOSUM62 global alignment with affine gaps and free end gaps, and returns an
explicit mapping plus identity and coverage. Residues with no coordinates are
counted and reported rather than dropped silently.

### 5. Circuits are grown greedily, not taken as a top-n list

SAE features are often redundant — several encode the same thing, and each alone
recovers most of a mutation's effect. Taking the top 5 by individual DCE
therefore yields five copies of one mechanism. `greedy_select` instead adds, at
each step, the feature that most improves the recovery of the set *already
chosen*, and stops when the marginal gain falls below a threshold. The reported
headline number is the **joint** recovery of all selected features patched
together in a single forward pass, not a sum of individual effects — the model
is nonlinear and those are not the same quantity.

---

## Repository layout

```
causalmech-bio/
├── configs/                      Hydra configuration
│   ├── config.yaml               root: seed, device, paths
│   ├── model/esm2_8m.yaml        default — 6 layers, d=320
│   ├── model/esm2_35m.yaml       opt-in — 12 layers, d=480
│   ├── sae/topk_sae.yaml         dictionary size, k, AuxK, training schedule
│   └── data/proteingym.yaml      ProteinGym / ClinVar / synthetic sources
├── src/
│   ├── models/
│   │   ├── esm_hooks.py          residual-stream capture and patching primitives
│   │   └── sparse_autoencoder.py Top-K SAE, AuxK, unit-norm decoder
│   ├── interpretability/
│   │   ├── path_patching.py      DCE, splicing, ablation, attribution patching
│   │   ├── circuit_extraction.py greedy selection, positional localisation, edges
│   │   └── baselines.py          attention / integrated gradients / random
│   ├── validation/
│   │   ├── pdb_aligner.py        alignment, Cα geometry, permutation null
│   │   └── monosemanticity_metrics.py  FVU, L0, CE-recovered, feature purity
│   └── utils/
│       ├── protein.py            mutation parsing and the index conventions
│       ├── dataloaders.py        ProteinGym/ClinVar + streaming activation buffer
│       ├── device.py             MPS selection and its caveats
│       └── seeding.py            reproducibility
├── scripts/
│   ├── smoke_test.py             full pipeline, no downloads — run this first
│   ├── fetch_assets.py           all downloads live here, and only here
│   ├── train_sae.py              Hydra-driven SAE training
│   ├── discover_circuits.py      the single-command demo
│   └── benchmark_saliency.py     causal precision + faithfulness vs baselines
├── dashboard/app.py              Streamlit + py3Dmol circuit inspector
├── docs/
│   ├── GUIDE.md                  the complete guide — start here if new
│   └── GUIDE.pdf                 the same, typeset (76pp, A4)
└── tests/                        79 tests, no network, no pretrained weights
```

---

## Index conventions

Off-by-one errors are the most common silent corruption in this kind of
pipeline, so the conventions are fixed in `src/utils/protein.py` and nothing is
allowed to convert between them inline:

| Quantity | Convention | Example |
|---|---|---|
| Mutation strings | 1-indexed | `A45T` → residue 45 |
| `Mutation.seq_pos` | 0-indexed into the sequence | `44` |
| Token positions | 0-indexed, shifted by `<cls>` | `45` |
| Reported circuit residues | 1-indexed | `45` |

`seq_to_token_pos` / `token_to_seq_pos` are the only sanctioned conversions, and
`token_to_seq_pos(0)` raises because `<cls>` is not a residue.

---

## Tests

```bash
make test        # or: python -m pytest
```

79 tests, no network access and no pretrained weights. The suite builds a real
`EsmForMaskedLM` from `EsmConfig` at toy dimensions, so the actual transformers
module graph is exercised — layer resolution, hook placement, output-tuple
handling and token offsets all run against the real ESM-2 code path.

The tests worth knowing about:

| Test | What it catches |
|---|---|
| `test_patching_clean_into_clean_is_identity` | wrong splice algebra, asymmetric activation scaling, writing to the wrong tuple element |
| `test_causal_conservation_full_patch_reaches_clean_behaviour` | substituting the whole clean activation must reproduce the clean run exactly |
| `test_baseline_is_measured_without_hooks` | the baseline-under-hook bug that zeroes every effect |
| `test_hooks_removed_even_when_forward_raises` | leaked hooks corrupting later passes |
| `test_error_preserving_splice_cancels_reconstruction_error` | an empty patch must be a perfect no-op |
| `test_reconstruction_error_below_five_percent` | SAE fits to FVU < 0.05 on a well-conditioned recovery problem |
| `test_alignment_recovers_an_n_terminal_offset` | the sequence/structure numbering shift |
| `test_planted_pocket_is_significantly_clustered` | geometry test finds a constructed pocket and rejects a spread-out set |
| `test_inactive_features_have_zero_effect` | soundness of the pruning stage |

A note on conservation: **individual feature effects do not sum to the total
effect.** The model is nonlinear and features interact, so a test asserting
additivity would be asserting something false. What does hold exactly, and is
what `test_causal_conservation_full_patch_reaches_clean_behaviour` checks, is
that replacing the entire residual stream at a layer makes everything downstream
identical to the clean forward pass.

---

## Running on an 8 GB M2

The defaults are chosen for this machine, and the constraints shape the design:

- **ESM-2 8M is the default**, 35M is one flag away (`model=esm2_35m`). Iterate
  on 8M; produce final numbers on 35M.
- **Activations are streamed, never materialised.** 10,000 sequences × ~300
  tokens × 320 dims in fp32 is ~4 GB, alongside the model, on a machine with
  8 GB total. `ActivationBuffer` keeps a fixed 256k-token shuffled pool and
  refills from fresh forward passes, so memory is bounded by `buffer_tokens`
  regardless of corpus size. The shuffle is not cosmetic: consecutive tokens
  from one protein are strongly correlated, and unshuffled batches teach the SAE
  per-protein idiosyncrasies that look like features.
- **Attribution patching prescreens.** Exact patching costs one forward pass per
  feature; scoring a 5120-element dictionary across layers and variants that way
  does not finish. One backward pass ranks the whole dictionary, and only the
  shortlist gets exact treatment. Attribution is used for *ranking only* — it is
  unreliable in magnitude, particularly where the model saturates, which is
  often where the interesting behaviour is.
- **MPS caveats are handled explicitly.** `PYTORCH_ENABLE_MPS_FALLBACK=1` is set
  before torch is imported in every entry point, so a missing kernel degrades to
  CPU instead of crashing a long sweep. MPS has no float64, so the permutation
  null runs on CPU in numpy.

Rough timings on an M2 Air (8M model): SAE training ~15–20 min per layer at
default settings, circuit discovery for one variant ~30–60 s, the benchmark
~20 min for 25 variants.

---

## Configuration

Hydra, so everything is overridable from the command line:

```bash
python scripts/train_sae.py model=esm2_35m
python scripts/train_sae.py sae.arch.k=64 sae.arch.dict_mult=32
python scripts/train_sae.py model.target_layers=[3]
python scripts/benchmark_saliency.py data.source=synthetic +n_variants=50
```

`data.source=synthetic` generates proteins with a planted `HExxH` motif and
labels variants inside it as deleterious. That gives circuit discovery a known
ground truth: if the pipeline cannot recover a motif it was handed, it will not
recover an active site. Useful for validating changes to the discovery
algorithm before spending model time on real assays.

---

## Data sources

`scripts/fetch_assets.py` is the only place anything is downloaded, so that an
analysis run cannot quietly depend on network state and works offline once the
assets exist.

- **ESM-2** (`facebook/esm2_t6_8M_UR50D`, `facebook/esm2_t12_35M_UR50D`) from
  HuggingFace.
- **PDB** structures from RCSB.
- **UniProt/SwissProt** reviewed sequences for the SAE training corpus.
- **ProteinGym** DMS substitution assays. Release URLs move between versions; if
  the download 404s, check <https://proteingym.org> for the current link.
- **ClinVar** is not auto-downloadable in a usable form — the variant summary
  gives HGVS notation, and turning that into `(wt_sequence, mutation)` needs a
  RefSeq/UniProt transcript join whose choice would silently determine the
  results. The script writes the expected schema and leaves that join to you.

---

## Status and limitations

What is implemented and tested: Phases 1–5 of the plan — environment, hooks,
SAE, path patching, circuit extraction, biophysical validation, saliency
benchmarking, and the dashboard.

Not yet written: the workshop paper (`paper/`). It is deliberately deferred
until there are real numbers to write up.

Known limitations worth stating before anyone reads results from this:

- **Single-substitution variants only.** `parse_mutation` rejects multi-mutants
  explicitly, because single-site causal attribution is not well defined for a
  joint mutant.
- **Circuits are per-variant.** Nothing here yet establishes that the same
  circuit generalises across variants of the same protein, let alone across
  proteins. That is the obvious next experiment and the one that would turn this
  into a result.
- **The variant-effect metric is a logit difference at the mutated site.** It is
  the standard readout and it correlates with function, but it is the model's
  opinion, not an assay. `SequenceLogLikelihoodMetric` is provided as a coarser
  robustness check.
- **Attribution prescreening can miss features.** The shortlist is built from a
  first-order approximation, so a feature whose effect is invisible to the
  gradient can be dropped before exact patching ever sees it.
- **SAE dictionaries are layer- and checkpoint-specific.** Loading refuses
  obvious mismatches, but nothing can detect a dictionary trained on a
  meaningfully different corpus.

---

## License

MIT.
