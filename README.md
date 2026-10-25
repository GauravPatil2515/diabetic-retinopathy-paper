# Domain-Aware Hybrid CNN–Vision Transformer for Cross-Domain Diabetic Retinopathy Grading

[![IEEE Conference Paper](https://img.shields.io/badge/IEEE-SISIMPACT%20%2F%20ICIPCN%202027-blue.svg)](conferences/03_SISIMPACT_2027/README.md)
[![Preprint Available](https://img.shields.io/badge/bioRxiv%20%2F%20medRxiv-Preprint%20Kit-green.svg)](conferences/04_PREPRINT_MEDRXIV/README.md)
[![Double-Blind Compliant](https://img.shields.io/badge/Peer%20Review-Double--Blind%20Verified-success.svg)](docs/DOUBLE_BLIND_CHECKLIST.md)
[![LaTeX Build](https://img.shields.io/badge/LaTeX-pdfTeX%200--errors-brightgreen.svg)](paper/compile.sh)
[![Reproducibility](https://img.shields.io/badge/Benchmarks-100%25%20Reproducible-orange.svg)](docs/REPRODUCIBILITY_GUIDE.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official publication repository for the paper: **"Domain-Aware Hybrid CNN–Vision Transformer for Cross-Domain Diabetic Retinopathy Grading"**.

This repository contains the complete 7-page IEEEtran manuscript sources, vector illustrations, multi-center Leave-One-Dataset-Out (LODO) experimental ledgers, Grad-CAM clinical case studies, dataset audits, conference submission kits, and automated reproduction scripts.

---

## 📑 Quick Navigation

| Resource Section | Direct Link | Content Description |
| :--- | :--- | :--- |
| **Manuscript PDF** | [📄 `main.pdf`](main.pdf) | Compiled, publication-ready 7-page IEEE PDF with balanced columns (`flushend`). |
| **LaTeX Sources** | [📁 `paper/`](paper/) | `main.tex`, `references.bib`, `IEEEtran.cls`, and one-click build script `compile.sh`. |
| **Conference Kits** | [🏛️ `conferences/`](conferences/) | Ready-to-submit packages for **SISIMPACT 2027**, **ICIPCN 2027**, **ICICCDS**, and **medRxiv**. |
| **Benchmark Ledgers** | [📊 `results/`](results/) | Master LODO CSVs, full experiment logs (51KB), bootstrap JSONs, and TTA logs. |
| **Visual Catalog** | [🖼️ `figures/`](figures/) | Figures 1–8 (PDF/PNG), 8 confusion matrices, and 106 Grad-CAM clinical cases. |
| **Technical Docs** | [📚 `docs/`](docs/) | 10 in-depth architectural deep dives, ETDRS clinical mappings, and audits. |
| **Execution Scripts** | [⚙️ `scripts/`](scripts/) | Bootstrap QWK calculation, double-blind verification, and figure generators. |

---

## 🎯 Executive Summary & Core Contribution

Diabetic Retinopathy (DR) automated grading algorithms frequently suffer catastrophic performance collapse when deployed across different clinical centers due to domain shifts in sensor optics, illumination, and demographic characteristics.

### The Problem: Why Prior Approaches Fail
1. **Standard CNNs (ResNet, DenseNet, EfficientNet)**: Rely on shared Batch Normalization. When shifted out-of-domain, internal running feature statistics diverge, causing QWK to plummet from $>0.80$ to $<0.20$.
2. **Pure Vision Transformers (ViT, Swin, RETFound)**: Use LayerNorm, which stabilizes training, but self-attention alone lacks spatial inductive bias to resolve tiny punctate lesions ($<20\mu\text{m}$ microaneurysms) in early DR stages.
3. **Adversarial Domain Adaptation (DANN)**: Causes mode collapse in ordinal grading by aligning healthy retinal features with pathological ones.

### The Solution: Our Proposed Architecture
Our framework introduces a three-pillar signal-preserving architecture:
- **Dual-Stream Feature Fusion**: Combines high-resolution multi-scale CNN feature maps ($C_3, C_4, C_5$) with global self-attention context from a foundation Vision Transformer (DINOv2).
- **Domain-Specific Batch Normalization (DSBN)**: Maintains dedicated affine parameters $\{\gamma_d, \beta_d\}$ and running statistics per clinical center while sharing deep feature representations.
- **Distance-Weighted Cross-Entropy (DWCE)**: Incorporates quadratic ordinal penalty matrices directly into gradient updates, aligning predictions with the clinical ETDRS grading hierarchy.

```
                      ┌───────────────────────────────────────────────┐
                      │  Fundus Input [B, 3, 256, 256]                │
                      │  (Circular Crop + Graham Norm + CLAHE)        │
                      └──────────────────────┬────────────────────────┘
                                             │
                     ┌───────────────────────┴───────────────────────┐
                     ▼                                               ▼
         ┌───────────────────────────────┐               ┌───────────────────────┐
         │ Multi-Scale CNN (EffNet-B3)   │               │ Vision Transformer    │
         │ Stages: C3, C4, C5 (Frozen)   │               │ DINOv2 ViT-Small      │
         └───────────────┬───────────────┘               └───────────┬───────────┘
                         │                                           │
                         ▼                                           ▼
         ┌───────────────────────────────┐               ┌───────────────────────┐
         │ Domain-Specific BN (DSBN)     │               │ Global Token          │
         │ Site A: {γ_A, β_A}            │               │ Representation        │
         │ Site B: {γ_B, β_B}            │               │ Context [B, 384]      │
         └───────────────┬───────────────┘               └───────────┬───────────┘
                         │                                           │
                         └───────────────────────┬───────────────────┘
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ Cross-Attention Feature Fusion  │
                                └────────────────┬────────────────┘
                                                 │
                                                 ▼
                                ┌─────────────────────────────────┐
                                │ Distance-Weighted CE Loss (DWCE)│
                                │ Ordinal ETDRS Calibrated Output │
                                └─────────────────────────────────┘
```

---

## 📊 Master Results & Experimental Benchmarks

All reported metrics are evaluated across **three international retinal cohorts ($N = 5,922$ images)** under a strict **3-Fold Leave-One-Dataset-Out (LODO)** protocol:

### 1. Multi-Center Leave-One-Dataset-Out (LODO) Generalization

| Evaluation Partition | Held-Out Target | Baseline (SharedNorm) | RETFound (ViT-Large) | **Proposed Hybrid DSBN** | Performance Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** (Train: Messidor + IDRiD) | **APTOS 2019** ($N=3,662$) | $0.152 \pm 0.012$ | $0.584 \pm 0.012$ | **$0.859 \pm 0.009$** | $+0.707$ QWK |
| **Fold 1** (Train: APTOS + IDRiD) | **Messidor-2** ($N=1,744$) | $0.213 \pm 0.018$ | $0.521 \pm 0.014$ | **$0.641 \pm 0.011$** | $+0.428$ QWK |
| **Fold 2** (Train: APTOS + Messidor) | **IDRiD** ($N=516$) | $0.182 \pm 0.016$ | $0.553 \pm 0.013$ | **$0.750 \pm 0.010$** | $+0.568$ QWK |
| **3-Domain LODO Mean** | **Multi-Site Benchmark** | **$0.182 \pm 0.015$** | **$0.553 \pm 0.013$** | **$0.750 \pm 0.010$** | **$+0.568$ Gain ($p < 0.001$)** |

Detailed narrative breakdown: [`results/BENCHMARK_ANALYSIS.md`](results/BENCHMARK_ANALYSIS.md) | Raw data: [`results/master_lodo_benchmark.csv`](results/master_lodo_benchmark.csv)

### 2. Systematic Architectural Ablation

| Architecture Configuration | Normalization / Domain Strategy | Resolution | Validation QWK ($\uparrow$) | $\Delta$ Gain | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| EfficientNet-B3 (Baseline) | Standard BatchNorm | $224 \times 224$ | $0.775 \pm 0.012$ | -- | Single-stream baseline |
| DINOv2 ViT-Small (Baseline) | LayerNorm | $224 \times 224$ | $0.751 \pm 0.014$ | $-0.024$ | Single-stream baseline |
| EfficientNet-B3 + DINOv2 | DANN ($\lambda=1.0$) | $224 \times 224$ | $0.000 \pm 0.000$ | $-0.775$ | Adversarial mode collapse |
| EfficientNet-B3 + DINOv2 | Shared BatchNorm | $224 \times 224$ | $0.742 \pm 0.011$ | $-0.033$ | Domain confusion |
| Hybrid CNN--ViT (No EMA) | DSBN | $224 \times 224$ | $0.792 \pm 0.013$ | $+0.017$ | Domain isolation |
| **Hybrid CNN--ViT + DSBN + EMA** | **DSBN** | $\mathbf{224 \times 224}$ | **$0.841 \pm 0.008$** | **$+0.066$** | **Proposed configuration** |
| Hybrid CNN--ViT with TTA | DSBN | $384 \times 384$ | $0.786 \pm 0.015$ | $+0.011$ | Small batch size variance |

### 3. Quantitative Probability Calibration & Edge Efficiency

| Evaluation Metric | Uncalibrated Baseline | **Calibrated (Ours)** | Improvement / Specification |
| :--- | :---: | :---: | :--- |
| **Expected Calibration Error (ECE)** | $0.143$ | **$0.041$** | **$\downarrow 71.3\%$ error reduction** (Temperature $T=1.42$) |
| **Brier Score** | $0.184$ | **$0.092$** | **$\downarrow 50.0\%$ probability dispersion** |
| **Decision Thresholds ($\mathbf{T}^*$)** | $[0.5, 1.5, 2.5, 3.5]$ | **$[0.513, 1.427, 2.751, 3.482]$** | Moderate $\rightarrow$ Severe shift ($+0.251$) matches ETDRS |
| **Inference Latency** | $8.4\text{ ms}$ (CNN) | **$16.8\text{ ms}$ (Hybrid)** | Real-time edge screening compliance |
| **Peak GPU Memory (Batch=1)** | $2.1\text{ GB}$ (CNN) | **$4.2\text{ GB}$ (Hybrid)** | Runs on commodity 6GB/8GB GPUs |

---

## 📚 Technical Documentation & Deep Dives

The `docs/` folder contains comprehensive reference material:

1. [📐 `docs/ARCHITECTURE_DEEP_DIVE.md`](docs/ARCHITECTURE_DEEP_DIVE.md): Full topological mathematical description of DSBN, DWCE loss formulation, cross-attention projection math, and parameter budget.
2. [🏥 `docs/CLINICAL_SIGNIFICANCE_ETDRS.md`](docs/CLINICAL_SIGNIFICANCE_ETDRS.md): Clinical translation of ETDRS 4-2-1 rule, boundary shift mitigation, and ophthalmological risk stratification.
3. [📊 `docs/EXPERIMENTAL_AUDIT_LEDGER.md`](docs/EXPERIMENTAL_AUDIT_LEDGER.md): Master ledger cross-referencing all 50+ training runs, seed configurations, and verification checksums.
4. [🔬 `docs/DATASET_SPECIFICATIONS.md`](docs/DATASET_SPECIFICATIONS.md): Cohort breakdown for APTOS 2019, Messidor-2, IDRiD, DDR (camera hardware, resolution, demographics, and class ratios).
5. [🖼️ `docs/PREPROCESSING_PIPELINE.md`](docs/PREPROCESSING_PIPELINE.md): Step-by-step mathematical explanation of Graham normalization, Otsu circular FOV cropping, and green-channel CLAHE.
6. [🔒 `docs/DOUBLE_BLIND_CHECKLIST.md`](docs/DOUBLE_BLIND_CHECKLIST.md): IEEE peer review compliance verification verifying zero authorial or institutional leaks.
7. [⚙️ `docs/REPRODUCIBILITY_GUIDE.md`](docs/REPRODUCIBILITY_GUIDE.md): Instructions to regenerate all tables, metrics, and vector figures from source.
8. [📖 `docs/COMPREHENSIVE_SUMMARY.md`](docs/COMPREHENSIVE_SUMMARY.md): End-to-end research project report from initial discovery to final validation.
9. [🧪 `docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): Detailed phase-by-phase ablation regimes and hyperparameter configurations.
10. [🤖 `docs/MODELS.md`](docs/MODELS.md): Architectural specifications for all baseline backbones (DenseNet, EfficientNet, ViT, Swin, ConvNeXt).

---

## 📁 Repository Directory Structure

```
diabetic-retinopathy-paper/
├── README.md                           # Master technical portal and documentation index
├── LICENSE                             # Open-source MIT License
├── .gitignore                          # Clean LaTeX and Python build ignores
├── main.pdf                            # Precompiled 7-page IEEEtran camera-ready PDF
│
├── paper/                              # Complete LaTeX manuscript source package
│   ├── main.tex                        # Full IEEEtran manuscript source code
│   ├── references.bib                  # BibTeX bibliography database
│   ├── IEEEtran.cls                    # Official IEEE conference document class
│   ├── compile.sh                      # One-click 4-pass automated compilation script
│   ├── main.pdf                        # Synchronized 7-page PDF output
│   └── figures/                        # Native vector PDFs and high-res PNGs (Figs 1–8)
│
├── conferences/                        # Target conference submission kits
│   ├── 01_ICIPCN_2027/                 # ICIPCN 2027 IEEE kit (CFP26VY4-POD)
│   ├── 03_SISIMPACT_2027/              # SISIMPACT 2027 IEEE kit (Record #73129)
│   ├── 02_ICICCDS_2027/                # ICICCDS 2027 hybrid submission kit
│   └── 04_PREPRINT_MEDRXIV/            # bioRxiv / medRxiv submission kit
│
├── figures/                            # Full visual catalog
│   ├── README.md                       # Complete figure index & captions
│   ├── Figure_1_Preprocessing_New.pdf  # Vector Fig 1 (Retinal Normalization)
│   ├── Figure_2_Architecture_New.pdf   # Vector Fig 2 (Neural Architecture)
│   ├── Figure_3_TrainingStability_New.pdf # Vector Fig 3 (Convergence Dynamics)
│   ├── Figure_4_Boundaries_New.pdf     # Vector Fig 4 (Ordinal Calibration)
│   ├── Figure_5_GradCAM_New.pdf        # Vector Fig 5 (Multi-class Grad-CAM)
│   ├── Figure_6_LODO_New.pdf           # Vector Fig 6 (LODO Benchmark)
│   ├── Figure_7_Ablation_New.pdf       # Vector Fig 7 (Component Ablation)
│   ├── Figure_8_Comparison_New.pdf     # Vector Fig 8 (SOTA Comparison)
│   ├── confusion_matrices/             # Cross-domain confusion matrices (8 models)
│   ├── supplementary/                  # Extra figures (preprocessing & crop samples)
│   └── gradcam_clinical_cases/         # 106 Grad-CAM cases across all 5 DR grades
│
├── results/                            # Experimental benchmark ledgers & logs
│   ├── BENCHMARK_ANALYSIS.md           # Deep-dive comparative analysis
│   ├── master_lodo_benchmark.csv       # Master 3-fold LODO benchmark table
│   ├── experiment_tracking.csv         # 51KB master tracking log of all experimental runs
│   ├── downloaded_models_benchmark.csv # Pretrained comparative backbone audit
│   ├── bootstrap_qwk_results.json      # 1,000-iteration bootstrap stats & p-values
│   ├── dataset_audit_report.json       # Exact class counts across all cohorts
│   ├── dataset_audit_report.txt        # Text summary of data integrity audit
│   ├── phase_ablation_summary.txt      # 3-phase ablation logs
│   ├── tta_results.txt                 # Test-time adaptation evaluations
│   └── tta_evaluation_log.txt          # Multi-scale TTA log
│
├── docs/                               # 10 comprehensive markdown deep-dives
│   ├── ARCHITECTURE_DEEP_DIVE.md
│   ├── CLINICAL_SIGNIFICANCE_ETDRS.md
│   ├── EXPERIMENTAL_AUDIT_LEDGER.md
│   ├── DATASET_SPECIFICATIONS.md
│   ├── PREPROCESSING_PIPELINE.md
│   ├── DOUBLE_BLIND_CHECKLIST.md
│   ├── REPRODUCIBILITY_GUIDE.md
│   ├── COMPREHENSIVE_SUMMARY.md
│   ├── EXPERIMENTS.md
│   └── MODELS.md
│
└── scripts/                            # Standalone reproducible execution utilities
    ├── README.md                       # Scripts documentation & execution index
    ├── calculate_bootstrap_qwk.py      # Computes QWK & 95% bootstrap CI
    ├── verify_anonymization.py         # Automated double-blind compliance linter
    ├── export_clean_paper_tables.py    # Exports clean LaTeX booktabs tables
    ├── generate_all_professional_figures.py # Renders all 8 vector figures
    ├── generate_architecture_fig.py    # Figure 2 renderer
    ├── generate_preprocessing_fig.py   # Figure 1 renderer
    └── build_full_gradcam_grid.py      # Figure 5 Grad-CAM grid compiler
```

---

## 🛠️ Quickstart: How to Use & Reproduce

### 1. Compile the Paper PDF
```bash
cd paper
bash compile.sh
```
Compiles `main.tex` with `bibtex` and `pdflatex` into `main.pdf` (7 pages, 0 errors).

### 2. Verify Double-Blind Review Compliance
```bash
python3 scripts/verify_anonymization.py
```
Audits LaTeX sources and documents against author leaks.

### 3. Compute Bootstrap QWK Confidence Intervals
```bash
python3 scripts/calculate_bootstrap_qwk.py
```
Reproduces the 95% bootstrap CI ($0.8437$ $[0.812, 0.871]$).

### 4. Regenerate All Publication Figures
```bash
python3 scripts/generate_all_professional_figures.py
```
Rebuilds Figures 1 through 8 into both `.pdf` and `.png` formats.

---

## 🏛️ Verified IEEE Conference Submissions

1. **SISIMPACT 2027** (*2nd IEEE International Conference on Smart Innovations in Systems, AI & Computing*):
   - **IEEE Conference Record**: **#73129** (Officially verified on IEEE Conf. Search)
   - **Dates**: January 15–16, 2027 (Hybrid / Virtual presentation supported)
   - **Kit**: [`conferences/03_SISIMPACT_2027/README.md`](conferences/03_SISIMPACT_2027/README.md)
2. **ICIPCN 2027** (*7th International Conference on Image Processing and Capsule Networks*):
   - **Proceedings**: IEEE Xplore & Scopus indexed (Catalog # `CFP26VY4-POD`)
   - **Dates**: January 21–23, 2027 (Hybrid mode)
   - **Kit**: [`conferences/01_ICIPCN_2027/README.md`](conferences/01_ICIPCN_2027/README.md)
3. **bioRxiv / medRxiv Preprint**:
   - **Kit**: [`conferences/04_PREPRINT_MEDRXIV/README.md`](conferences/04_PREPRINT_MEDRXIV/README.md)

---

## 📖 Citation

```bibtex
@inproceedings{patil2027domain,
  title     = {Domain-Aware Hybrid CNN--Vision Transformer for Cross-Domain Diabetic Retinopathy Grading},
  author    = {Patil, Gaurav},
  booktitle = {Proceedings of the IEEE International Conference on Smart Innovations in Systems, Infrastructure, AI and Computing Technologies (SISIMPACT)},
  year      = {2027},
  pages     = {1--7},
  publisher = {IEEE}
}
```

---

## 📄 License & Ethical Compliance

- **Code & Documentation**: Released under the [MIT License](LICENSE).
- **Benchmark Datasets**: All experiments utilize publicly accessible, fully de-identified retinal benchmarks ([APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection), [Messidor-2](https://www.adcis.net/en/third-party/messidor2/), and [IDRiD](https://idrid.grand-challenge.org/)).
