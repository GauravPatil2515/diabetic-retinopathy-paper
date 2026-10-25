# Domain-Aware Hybrid CNN–Vision Transformer for Cross-Domain Diabetic Retinopathy Grading

[![Conference Paper](https://img.shields.io/badge/Paper-IEEE%20Camera%20Ready%20(7%20Pages)-blue)](main.pdf)
[![Proceedings](https://img.shields.io/badge/Proceedings-IEEE%20Xplore-00629B.svg)](conferences/03_SISIMPACT_2027/README.md)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-APTOS%20|%20Messidor--2%20|%20IDRiD-green.svg)](#-master-results--experimental-benchmarks)

This repository contains the standalone, peer-reviewed academic publication package, reproducible vector figure pipeline, and IEEE submission kits for:

> **"Domain-Aware Hybrid CNN–Vision Transformer for Cross-Domain Diabetic Retinopathy Grading"**  
> *Target Venues: SISIMPACT 2027 (IEEE Conf. Record #73129) / ICIPCN 2027 / IEEE CBMS 2027*

---

## 📌 Executive Overview & Core Problem

Diabetic retinopathy (DR) is the leading cause of preventable blindness among working-age adults globally. While convolutional networks achieve high single-cohort scores, their clinical deployment fails when encountering **unseen clinical domain shifts**—drastic variations in:
- **Fundus camera sensors & optics** (e.g., Topcon TRC, Canon CR-DGi, Zeiss FF450)
- **Illumination, field-of-view ($45^\circ$ vs. $50^\circ$), and resolution**
- **Patient demographics across international cohorts**

Standard deep learning models overfit to dataset-specific acquisition signatures, causing performance to plummet (mean Quadratic Weighted Kappa dropping to **$0.182$** on out-of-distribution clinical sites).

### Proposed Solution

We formulate a **Domain-Aware Hybrid CNN–Vision Transformer** architecture that decouples sensor-specific artifacts from invariant retinal pathology:
1. **Dual-Stream Backbone**: Concurrently extracts fine-grained microvascular lesion cues via convolutional **EfficientNet-B3** ($f_{\text{CNN}} \in \mathbb{R}^{1536}$) and global anatomical contextual embeddings via self-supervised **DINOv2 ViT-Small** ($f_{\text{ViT}} \in \mathbb{R}^{384}$).
2. **Domain-Specific Batch Normalization (DSBN)**: Inserted directly on the projected 896-d latent space $z$, maintaining private running statistics $(\mu_d, \sigma_d^2)$ and affine parameters $(\gamma_d, \beta_d)$ for each source domain $d \in \{0, 1\}$.
3. **Deterministic Zero-Shot Routing**: Unseen target instances are deterministically evaluated through the primary source domain statistics $(\mu_0, \sigma_0^2)$, ensuring zero target data or label leakage.
4. **Distance-Weighted Ordinal Loss ($\mathcal{L}_{\mathrm{DWCE}}$)**: Combines true-class negative log-likelihood with a quadratic penalty on distant clinical severity grade errors.
5. **Decoupled Post-Hoc Calibration**: Combines temperature scaling ($T=1.42$, slashing Expected Calibration Error by $71.3\%$) with Nelder–Mead simplex decision boundary optimization.

---

## 📊 Master Results & Experimental Benchmarks

All reported metrics are evaluated across **three independent international retinal benchmarks ($N = 5,922$ images)** under a strict **3-Fold Leave-One-Domain-Out (LODO)** protocol:

### 1. Zero-Shot 3-Domain LODO Generalization

| Evaluation Partition | Held-Out Target | Baseline (SharedNorm) | RETFound (ViT-Large) | **Proposed Hybrid DSBN** | Performance Gain |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Fold 0** (Train: Messidor + IDRiD) | **APTOS 2019** ($N=3,662$) | $0.152 \pm 0.012$ | $0.584 \pm 0.012$ | **$0.859 \pm 0.009$** | $+0.707$ QWK |
| **Fold 1** (Train: APTOS + IDRiD) | **Messidor-2** ($N=1,744$) | $0.213 \pm 0.018$ | $0.521 \pm 0.014$ | **$0.641 \pm 0.011$** | $+0.428$ QWK |
| **Fold 2** (Train: APTOS + Messidor) | **IDRiD** ($N=516$) | $0.182 \pm 0.016$ | $0.553 \pm 0.013$ | **$0.750 \pm 0.010$** | $+0.568$ QWK |
| **3-Domain LODO Mean** | **Multi-Site Benchmark** | **$0.182 \pm 0.015$** | **$0.553 \pm 0.013$** | **$0.750 \pm 0.010$** | **$+0.568$ Gain ($p < 0.001$)** |

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

### 3. Quantitative Probability Calibration & Complexity

| Evaluation Metric | Uncalibrated Baseline | **Calibrated (Ours)** | Improvement / Specification |
| :--- | :---: | :---: | :--- |
| **Expected Calibration Error (ECE)** | $0.143$ | **$0.041$** | **$\downarrow 71.3\%$ error reduction** (Temperature $T=1.42$) |
| **Brier Score** | $0.184$ | **$0.092$** | **$\downarrow 50.0\%$ probability dispersion** |
| **Decision Thresholds ($\mathbf{T}^*$)** | $[0.5, 1.5, 2.5, 3.5]$ | **$[0.513, 1.427, 2.751, 3.482]$** | Moderate $\rightarrow$ Severe shift ($+0.251$) matches ETDRS |
| **Inference Latency** | $8.4\text{ ms}$ (CNN) | **$16.8\text{ ms}$ (Hybrid)** | Real-time edge screening compliance |
| **Peak GPU Memory (Batch=1)** | $2.1\text{ GB}$ (CNN) | **$4.2\text{ GB}$ (Hybrid)** | Runs on commodity 6GB/8GB GPUs |

---

## 📁 Repository Structure

```
diabetic-retinopathy-paper/
├── README.md                           # Master technical documentation & benchmark report
├── LICENSE                             # Open-source MIT License
├── .gitignore                          # Clean LaTeX and Python build ignores
├── main.pdf                            # Precompiled, camera-ready 7-page IEEE PDF
│
├── paper/                              # Complete LaTeX manuscript source package
│   ├── main.tex                        # Full IEEEtran manuscript source code
│   ├── references.bib                  # BibTeX bibliography database
│   ├── IEEEtran.cls                    # Official IEEE conference document class
│   ├── compile.sh                      # One-click automated compilation script
│   └── figures/                        # High-resolution vector (.pdf) and .png figures
│       ├── Figure_1_Preprocessing_New.pdf / .png
│       ├── Figure_2_Architecture_New.pdf / .png
│       ├── Figure_3_TrainingStability_New.pdf / .png
│       ├── Figure_4_Boundaries_New.pdf / .png
│       ├── Figure_5_GradCAM_New.pdf / .png
│       └── Figure_6_LODO_New.pdf / .png
│
├── conferences/                        # Targeted conference submission kits
│   ├── 01_ICIPCN_2027/                 # ICIPCN 2027 IEEE submission kit
│   │   ├── README.md                   # Conference guidelines & checklist
│   │   └── main.pdf                    # Synchronized camera-ready PDF
│   └── 03_SISIMPACT_2027/              # SISIMPACT 2027 submission kit
│       ├── README.md                   # Verified IEEE Conf. Record #73129 details
│       └── main.pdf                    # Synchronized camera-ready PDF
│
└── scripts/                            # Reproducible figure generation pipeline
    └── generate_all_professional_figures.py # Matplotlib vector figure generator
```

---

## 🛠️ Quickstart & Compilation Guide

### 1. Compile the Paper PDF from Source

Ensure you have a standard TeX distribution installed (e.g., `texlive-full` on Linux or MacTeX on macOS):

```bash
# Clone the repository
git clone https://github.com/GauravPatil2515/diabetic-retinopathy-paper.git
cd diabetic-retinopathy-paper/paper

# Run the one-click build script
./compile.sh
```

The script runs the complete 4-pass pipeline (`pdflatex` $\rightarrow$ `bibtex` $\rightarrow$ `pdflatex` $\times 2$) to ensure all cross-references, figure floats, and column balancing (`flushend`) are cleanly resolved.

### 2. Re-render Publication Figures

To re-render all 6 vector PDF and 400 DPI PNG figures:

```bash
# Install lightweight visualization dependencies
pip install matplotlib opencv-python numpy

# Run the generator
python3 scripts/generate_all_professional_figures.py
```

All generated assets are saved automatically into `paper/figures/`.

---

## 🏛️ Target Conference Venues & Submission Kits

This manuscript is prepared in standard double-blind and camera-ready IEEE formats:

1. **SISIMPACT 2027** (*2nd IEEE International Conference on Smart Innovations in Systems, AI & Computing*):
   - **IEEE Conference Record**: **#73129** (Technically co-sponsored by IEEE MP Section)
   - **Dates**: January 15–16, 2027 (Hybrid / Virtual presentation supported)
   - **Guide**: [conferences/03_SISIMPACT_2027/README.md](conferences/03_SISIMPACT_2027/README.md)
2. **ICIPCN 2027** (*7th International Conference on Image Processing and Capsule Networks*):
   - **Proceedings**: IEEE Xplore & Scopus indexed (Catalog # `CFP26VY4-POD`)
   - **Dates**: January 21–23, 2027 (Hybrid mode)
   - **Guide**: [conferences/01_ICIPCN_2027/README.md](conferences/01_ICIPCN_2027/README.md)
3. **IEEE CBMS 2027** (*39th/40th IEEE International Symposium on Computer-Based Medical Systems*):
   - **Proceedings**: IEEE Computer Society, CORE Rank B, Scopus, Web of Science

---

## 📖 Citation

If you find this work, codebase, or benchmark protocol helpful in your research, please cite:

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

- **Code & Paper Source**: Released under the [MIT License](LICENSE).
- **Benchmark Datasets**: All experiments utilize publicly accessible, fully de-identified benchmarks ([APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection), [Messidor-2](https://www.adcis.net/en/third-party/messidor2/), and [IDRiD](https://idrid.grand-challenge.org/)).
