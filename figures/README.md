# Figures & Visual Assets Catalog

This catalog indexes all publication-ready vector diagrams, Grad-CAM clinical case studies, and confusion matrices available in this repository.

---

## 📄 Primary Manuscript Figures (PDF Vector & 400 DPI PNG)

| Figure | Filename (PDF / PNG) | Description | Relevant Manuscript Section | Generating Script |
| :---: | :--- | :--- | :--- | :--- |
| **Fig. 1** | [`Figure_1_Preprocessing_New.pdf`](Figure_1_Preprocessing_New.pdf) | End-to-end retinal image normalization pipeline (Crop, Graham, CLAHE, Resize). | Section III-A (*Data Harmonization*) | `scripts/generate_preprocessing_fig.py` |
| **Fig. 2** | [`Figure_2_Architecture_New.pdf`](Figure_2_Architecture_New.pdf) | Dual-Stream CNN-ViT architecture with DSBN routing and DWCE loss. | Section III-B (*Model Architecture*) | `scripts/generate_architecture_fig.py` |
| **Fig. 3** | [`Figure_3_TrainingStability_New.pdf`](Figure_3_TrainingStability_New.pdf) | Convergence curves: Training loss, validation loss, and QWK over 30 epochs. | Section IV-C (*Training Dynamics*) | `scripts/generate_all_professional_figures.py` |
| **Fig. 4** | [`Figure_4_Boundaries_New.pdf`](Figure_4_Boundaries_New.pdf) | Ordinal boundary calibration and score distributions across grades 0–4. | Section III-D (*Ordinal Calibration*) | `scripts/generate_all_professional_figures.py` |
| **Fig. 5** | [`Figure_5_GradCAM_New.pdf`](Figure_5_GradCAM_New.pdf) | Multi-class Grad-CAM interpretability grid across all 5 ETDRS DR severity stages. | Section IV-E (*Clinical Interpretability*)| `scripts/build_full_gradcam_grid.py` |
| **Fig. 6** | [`Figure_6_LODO_New.pdf`](Figure_6_LODO_New.pdf) | Cross-center Leave-One-Dataset-Out (LODO) generalization comparison bar charts. | Section IV-B (*Cross-Domain Generalization*)| `scripts/generate_all_professional_figures.py` |
| **Fig. 7** | [`Figure_7_Ablation_New.pdf`](Figure_7_Ablation_New.pdf) | Multi-component ablation study (Freezing, DSBN, DWCE, Preprocessing). | Section IV-D (*Ablation Analysis*) | `scripts/generate_all_professional_figures.py` |
| **Fig. 8** | [`Figure_8_Comparison_New.pdf`](Figure_8_Comparison_New.pdf) | SOTA comparison: Proposed framework vs. DenseNet, EfficientNet, ViT, RETFound. | Section IV-A (*SOTA Benchmarking*) | `scripts/generate_all_professional_figures.py` |

---

## 🩺 Clinical Interpretability Case Studies (`figures/gradcam_clinical_cases/`)
Contains localized attention heatmaps, raw fundus photographs, and metadata for 5 patient cases per DR severity class:
- `No_DR/`: Validates model focus on vascular arcades and optic disc rather than background noise.
- `Mild/`: Highlights isolated punctate microaneurysms.
- `Moderate/`: Identifies hard exudate rings and blot hemorrhages.
- `Severe/`: Pinpoints extensive intraretinal hemorrhages and venous beading.
- `Proliferate_DR/`: Detects neovascular tufts and fibrous proliferations.

---

## 🔲 Domain Confusion Matrices (`figures/confusion_matrices/`)
Contains cross-domain confusion matrices for comparative backbones across datasets:
- `cm_densenet121_aptos.png` vs. `cm_densenet121_messidor.png`
- `cm_efficientnet_b3_aptos.png` vs. `cm_efficientnet_b3_messidor.png`
- `cm_vit_base_patch16_224_aptos.png` vs. `cm_vit_base_patch16_224_messidor.png`
- `cm_vit_small_patch16_224_aptos.png` vs. `cm_vit_small_patch16_224_messidor.png`
