# BAPR: code and manuscript

This publication branch includes the implementation, experiment protocols, analysis and test code,
Markdown reports, and the standalone Lean development in `reproducibility/lean`.

- `paper/paper.pdf`: compiled manuscript.
- `paper/paper.tex`: editable manuscript, with its appendix, bibliography and figures.
- `paper/paper.zip`: self-contained source and PDF archive for the current manuscript.

Build the manuscript with `latexmk -pdf -halt-on-error paper.tex` from
`paper/`. Experiment outputs, datasets, weights and checkpoints are
excluded from this branch. See [Reproducing the manuscript results](reproducibility/README.md)
for the separate compact results supplement and runtime instructions.
