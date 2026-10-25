# End-to-End Reproducibility Guide

This guide provides instructions to regenerate all experimental benchmarks, bootstrap confidence intervals, and vector publication figures directly from raw logs and source scripts.

---

## 💻 Environment Setup

The repository is built to be lightweight and requires only standard scientific Python packages:

```bash
# Optional: create a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install core dependencies
pip install numpy scipy pandas scikit-learn matplotlib seaborn
```

For LaTeX compilation:
```bash
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra
```

---

## 📊 1. Re-Calculate QWK & 95% Bootstrap Confidence Intervals

To reproduce the Quadratic Weighted Kappa (QWK) scores and paired bootstrap p-values reported in Table I and Section IV of the manuscript:

```bash
python3 scripts/calculate_bootstrap_qwk.py
```

- **Output Artifacts**: Verifies bootstrap differences against [`results/bootstrap_qwk_results.json`](../results/bootstrap_qwk_results.json).
- **Target Value**: Overall QWK $0.8437$ with 95% CI $[0.812, 0.871]$.

---

## 🎨 2. Regenerate Publication Figures

All manuscript figures can be rendered into dual formats (vector PDF for IEEE publication, 400 DPI PNG for presentations):

```bash
# Generate all 8 figures simultaneously
python3 scripts/generate_all_professional_figures.py

# Generate individual figures:
python3 scripts/generate_architecture_fig.py   # Figure 2: Neural Architecture
python3 scripts/generate_preprocessing_fig.py    # Figure 1: Retinal Normalization
python3 scripts/build_full_gradcam_grid.py       # Figure 5: 5-Class Grad-CAM Heatmaps
```

Rendered figures are automatically saved to `figures/` and `paper/figures/`.

---

## 📑 3. Compile the Complete Manuscript

Compile the 7-page IEEEtran manuscript with balanced columns (`flushend`):

```bash
cd paper
bash compile.sh
```

The script runs a 4-pass compilation pipeline (`pdflatex` $\to$ `bibtex` $\to$ `pdflatex` $\to$ `pdflatex`) ensuring zero unresolved citations and producing `main.pdf`.
