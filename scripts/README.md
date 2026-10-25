# Scripts Directory

Standalone, reproducible execution utilities for paper generation, statistical evaluation, and double-blind audits.

---

## 📂 Scripts Index

| Script | Purpose | Execution Command |
| :--- | :--- | :--- |
| **`calculate_bootstrap_qwk.py`** | Computes Quadratic Weighted Kappa with 1,000 bootstrap iterations and 95% CI. | `python3 scripts/calculate_bootstrap_qwk.py` |
| **`verify_anonymization.py`** | Automated double-blind compliance linter for LaTeX and docs. | `python3 scripts/verify_anonymization.py` |
| **`export_clean_paper_tables.py`** | Formats CSV benchmark results into clean LaTeX booktabs code. | `python3 scripts/export_clean_paper_tables.py` |
| **`generate_all_professional_figures.py`** | Regenerates all 8 publication vector PDFs and 400 DPI PNGs. | `python3 scripts/generate_all_professional_figures.py` |
| **`generate_architecture_fig.py`** | Generates neural architecture schematic (Figure 2). | `python3 scripts/generate_architecture_fig.py` |
| **`generate_preprocessing_fig.py`** | Generates retinal preprocessing pipeline diagram (Figure 1). | `python3 scripts/generate_preprocessing_fig.py` |
| **`build_full_gradcam_grid.py`** | Compiles 5-class Grad-CAM clinical case grid (Figure 5). | `python3 scripts/build_full_gradcam_grid.py` |
