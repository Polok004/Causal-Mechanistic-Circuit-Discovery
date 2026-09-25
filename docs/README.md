# Documentation

| File | What it is |
|---|---|
| `GUIDE.md` | The complete guide — ~23,000 words, from "what is an amino acid" through the causal-inference derivations. Written for a reader with no prior background. |
| `GUIDE.pdf` | The same document typeset for reading offline or printing (76 pages, A4). |
| `pdf-header.tex` | LaTeX preamble used when rendering the PDF. |

## Where to start

- **No background at all** → `GUIDE.md` Part 0. Two pages, assumes nothing.
- **Know ML, not biology** → Part 1, skip Part 2, resume at Part 3.
- **Know biology, not ML** → skip Part 1, start at Part 2.
- **Just want to run it** → Part 9 is self-contained.
- **Stuck on a word** → Appendix A is a glossary of every term the guide defines.

The sections that carry the actual method are 5.7 (error-preserving splicing),
5.12 (greedy selection) and 6.5 (the permutation test). If you read three
technical sections, read those.

## Rebuilding the PDF

Needs `pandoc`, `xelatex`, the `lmodern` LaTeX package, and the DejaVu fonts.

```bash
# macOS
brew install pandoc
brew install --cask mactex-no-gui      # or basictex + tlmgr install lmodern fvextra titlesec fancyhdr

# Debian/Ubuntu
sudo apt install pandoc texlive-xetex texlive-fonts-recommended lmodern fonts-dejavu
```

Then, from the repository root:

```bash
pandoc docs/GUIDE.md \
  -o docs/GUIDE.pdf \
  --pdf-engine=xelatex \
  --from=markdown+pipe_tables+backtick_code_blocks \
  --toc --toc-depth=2 \
  --highlight-style=tango \
  --include-in-header=docs/pdf-header.tex \
  -V documentclass=report \
  -V papersize=a4 \
  -V geometry:"margin=2.4cm" \
  -V mainfont="DejaVu Serif" \
  -V sansfont="DejaVu Sans" \
  -V monofont="DejaVu Sans Mono" \
  -V fontsize=10pt \
  -V linestretch=1.12 \
  -V colorlinks=true \
  -V linkcolor="[HTML]{1A4D7A}" \
  -V urlcolor="[HTML]{1A4D7A}" \
  -V toccolor="[HTML]{1A4D7A}" \
  -M title="CausalMech-Bio" \
  -M subtitle="A Complete Guide to Causal Mechanistic Circuit Discovery in Protein Language Models" \
  -M date="September 2026"
```

Two things that will bite you if you change the preamble:

**Do not put `\hypersetup` in `pdf-header.tex`.** Pandoc injects
`header-includes` *before* it loads `hyperref`, so `\hypersetup` is still
undefined at that point and the build fails with "Undefined control sequence".
Set link colours through pandoc's `-V linkcolor` / `-V urlcolor` variables
instead, as the command above does.

**Keep `fvextra` and its `Highlighting` redefinition.** Without it, long lines in
code blocks run off the right margin instead of wrapping.

## Keeping the guide accurate

The guide quotes specific default values (`k = 32`, `dict_mult = 16`,
`top_k_attribution = 64`, `min_identity = 0.8`, and others) and specific line
counts per module. If you change a default in `configs/` or a script's argparse
defaults, the guide is now wrong in a way no test catches.

The values it cites live in:

- `configs/sae/topk_sae.yaml` — `dict_mult`, `k`, `aux_k`, `aux_alpha`,
  `dead_after_tokens`, `buffer_tokens`, `refill_at`
- `configs/model/esm2_*.yaml` — layer counts, widths, `target_layers`
- `scripts/discover_circuits.py` — `--top-k-attribution`,
  `--max-features-per-layer`, `--min-gain`, `--position-threshold`,
  `--permutations`, `--distance-threshold`
- `scripts/benchmark_saliency.py` — `n_variants`, `top_n_residues`,
  `effect_threshold`, `ablation_fractions`, `ig_steps`
- `src/validation/pdb_aligner.py` — `min_identity`, gap penalties
