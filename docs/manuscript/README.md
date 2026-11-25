# ACFL journal manuscript (IEEE TAES)

LaTeX manuscript for **IEEE Transactions on Aerospace and Electronic Systems (TAES)** using `IEEEtran`.

## Files

| File | Purpose |
|------|---------|
| `main.tex` | IEEEtran journal scaffold (title, abstract, keywords, intro, related work, conclusion) |
| `method.tex` | §Method — threat model, FL baseline, Layers 1–3, gating |
| `experiments.tex` | §Experimental Evaluation — setup, results, ablation, limitations |
| `tables.tex` | Static setup/model tables + `\input{tables_data}` |
| `tables_data.tex` | Auto-generated benchmark tables (do not edit by hand) |
| `figures.tex` | Figure environments referencing generated PDFs |
| `references.bib` | IEEE BibTeX database (FL, Byzantine, poisoning, UAS cyber) |
| `generate_figures.py` | Build `figures/fig_main_f1.pdf` and `fig_backdoor_dual.pdf` |

## Build

```bash
# From repo root — figures + tables + PDF with bibliography
./acfl_env/bin/python scripts/sync_manuscript_numbers.py
./acfl_env/bin/python scripts/compute_significance_tests.py
./acfl_env/bin/python docs/manuscript/generate_figures.py

cd docs/manuscript
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Requires `matplotlib` (in `acfl_env`) and TeX Live with `IEEEtran.cls` / `IEEEtran.bst`.

## IEEE TAES notes

- Document class: `\documentclass[journal]{IEEEtran}`
- Bibliography: `references.bib` + `\bibliographystyle{IEEEtran}`
- Keywords: `\begin{IEEEkeywords}...\end{IEEEkeywords}`
- Double-blind: anonymous author block is active in `main.tex`
- Camera-ready: uncomment the named `\author{...}` example, fill affiliations/emails, add IEEE bios if the TAES author kit requires them
- Use the official IEEE TAES / Author Portal templates for final upload metadata (copyright, ORCID, etc.)

## Data sources

Benchmark numbers come from JSON summaries under `evaluation_results/`. Regenerate tables with `scripts/sync_manuscript_numbers.py`.

Sources include journal / ablation / threat-gap / scrub ablation / 5-client / combined Layer~3 / full-stack / Layer~3 operational summaries (see script header for the full list).

## Status

| Section | Status |
|---------|--------|
| Abstract / keywords | Synced to current results |
| Introduction | Draft with current backdoor / label-flip claims |
| Method | Includes shortlist-veto, post-agg scrub, shortlist-clean |
| Experiments | Synced to current 5-client / scrub ablation means |
| Related work | Expanded with IEEEtran bibliography |
| Conclusion | Draft with limitations + future work |
| Figures | Auto-generated from JSON |
| Tables | Auto-generated via `scripts/sync_manuscript_numbers.py` → `tables_data.tex` |
| Bibliography | `references.bib` (IEEEtran, 28 refs) |
| Layout | `IEEEtran` journal class |

## Include fragments only

If you already have a venue template, input individual files:

```latex
\input{docs/manuscript/method}
\input{docs/manuscript/experiments}
\input{docs/manuscript/figures}
\input{docs/manuscript/tables}
```
