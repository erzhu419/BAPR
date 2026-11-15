# BAPR: code and manuscript

This lightweight publication snapshot is based on BAPR commit `40b320d`.
It includes the implementation, experiment protocols, analysis and test code,
Markdown reports, and the standalone Lean development in `reproducibility/lean`.

- `paper/paper.pdf`: compiled manuscript.
- `paper/paper.tex`: editable manuscript, with its appendix, bibliography and figures.
- `paper/paper.zip`: self-contained source and PDF archive for the current manuscript.

Build the manuscript by running `pdflatex -halt-on-error paper.tex` twice from
`paper/`. This snapshot excludes recorded experiment outputs, datasets, model
weights and checkpoints. Re-running historical audits, plots or artifact-based
tests requires their corresponding reproduction inputs.
