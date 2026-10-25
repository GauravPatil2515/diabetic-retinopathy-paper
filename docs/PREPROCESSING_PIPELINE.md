# Retinal Preprocessing & Signal Normalization Pipeline

This document details the standardized multi-stage preprocessing pipeline applied across all retinal fundus cohorts to mitigate sensor-level domain shifts before neural feature extraction.

---

## 🖼️ Architectural Visual Overview

Refer to the high-resolution pipeline diagram in the manuscript:
- Vector Diagram: [`figures/Figure_1_Preprocessing_New.pdf`](../figures/Figure_1_Preprocessing_New.pdf)
- Supplementary Stage Breakdown: [`figures/supplementary/advanced_preprocessing_steps.png`](../figures/supplementary/advanced_preprocessing_steps.png)

```
Raw Color Fundus Image (Diverse Sensor / Resolution)
         │
         ▼ [Stage 1: Circular Field-of-View Segmentation]
Automated Thresholding + Largest Contour Bounding Box Extraction
         │
         ▼ [Stage 2: Ben Graham Color Constancy Normalization]
Local Gaussian Subtraction: I_norm = α·I + β·G(I, σ) + γ
         │
         ▼ [Stage 3: Multiscale Green-Channel Contrast Enhancement]
L*a*b* Luminance Equalization & Micro-Lesion Contrast Amplification
         │
         ▼ [Stage 4: Aspect-Ratio Preserved Isotropic Resizing & Zero-Padding]
Square Harmonization: Uniform 256 × 256 × 3 Tensor Input
```

---

## 📐 Mathematical Formulation of Pipeline Stages

### 1. Circular Field-of-View (FOV) Segmentation & Tight Cropping
Clinical fundus photographs capture circular retinal projections surrounded by variable black borders and device metadata. To remove non-informative pixels:
1. The RGB image $I(x, y)$ is converted to grayscale $I_{\text{gray}}(x, y)$.
2. An Otsu threshold $T_{\text{otsu}}$ masks out the background perimeter:
   $$\mathcal{M}(x, y) = \begin{cases} 1 & \text{if } I_{\text{gray}}(x, y) > T_{\text{otsu}} \\ 0 & \text{otherwise} \end{cases}$$
3. The largest connected contour $C_{\text{fundus}}$ is extracted, and the tight enclosing bounding box $[x_{\min}, y_{\min}, x_{\max}, y_{\max}]$ crops the image.

### 2. Ben Graham Local Color Constancy Normalization
Because different camera flash tubes, pupil dilation levels, and pigmentations cause extreme chromatic variations across centers, we apply Ben Graham's method:
$$I_{\text{norm}}(x, y) = 4 \cdot I(x, y) - 4 \cdot \mathcal{G}_{\sigma} * I(x, y) + 128$$
where $\mathcal{G}_{\sigma}$ is a 2D Gaussian blur kernel with standard deviation $\sigma = \frac{\min(H, W)}{30}$.
- **Effect**: Subtracts slow-varying illuminance trends across the retina while preserving high-frequency textural cues (microaneurysms, hemorrhages, and exudates).

### 3. Green-Channel Vascular Sharpening
The green spectral band exhibits optimal hemoglobin absorption contrast against the retinal pigment epithelium:
- The normalized image is projected to CIE $L^*a^*b^*$ color space.
- Contrast Limited Adaptive Histogram Equalization (CLAHE) is executed with `clipLimit=2.0` and `tileGridSize=(8, 8)` strictly on the luminance $L^*$ channel.
- Re-merging with $a^*$ and $b^*$ maintains chromatic balance while accentuating subtle micro-lesions in Grade 1 and Grade 2 eyes.

### 4. Isotropic Scale Harmonization
- Images are scaled uniformly along the major axis to $256$ pixels.
- The minor axis is padded symmetrically with zeros (black border) to form an exact $256 \times 256$ square tensor.
- This prevents geometric distortion of circular retinal lesions and anatomical landmarks (optic disc, fovea).

---

## ⚡ Ablation: Impact of Preprocessing on Domain Adaptation

| Preprocessing Regime | Messidor-2 Zero-Shot QWK | Cross-Center Lesion Recall |
| :--- | :---: | :---: |
| Raw Images (Direct Resize) | 0.384 | 42.1% |
| Circular Crop Only | 0.491 | 51.6% |
| Crop + Graham Normalization | 0.612 | 68.4% |
| **Full Pipeline (Crop + Graham + CLAHE)** | **0.663** | **78.2%** |
