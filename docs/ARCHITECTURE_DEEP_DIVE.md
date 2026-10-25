# Technical Architecture Deep Dive: Domain-Aware Hybrid CNN-ViT

This document details the architectural principles, layer topologies, mathematical formulations, and tensor dimensions of the **Domain-Aware Hybrid CNN–Vision Transformer** for cross-domain diabetic retinopathy grading.

---

## 1. Topological Overview

```
                      Input Retinal Tensor X ∈ R^(B × 3 × 224 × 224)
                                    │
            ┌───────────────────────┴───────────────────────┐
            ▼                                               ▼
┌───────────────────────────────┐               ┌───────────────────────────────┐
│ Stream A: EfficientNet-B3     │               │ Stream B: DINOv2 ViT-Small    │
│ • Pretrained Conv Backbone    │               │ • Self-Supervised Prior       │
│ • Local texture / lesions     │               │ • Global structural context   │
│ • Feature maps [C3, C4, C5]   │               │ • 14 × 14 patch tokens        │
│ • Global Average Pooling      │               │ • Class-Token [CLS] Embedding │
└───────────────┬───────────────┘               └───────────────┬───────────────┘
                │ f_CNN ∈ R^(B × 1536)                          │ f_ViT ∈ R^(B × 384)
                └───────────────────────┬───────────────────────┘
                                        ▼
                  Feature Concatenation: f_fused = [f_CNN ∥ f_ViT] ∈ R^(B × 1920)
                                        │
                         Linear Projection: W_proj ∈ R^(896 × 1920)
                                        │
                                        ▼
                   Projected Latent Embedding: z ∈ R^(B × 896)
                                        │
                                        ▼
           ┌────────────────────────────────────────────────────────┐
           │ Domain-Specific Batch Normalization (DSBN)             │
           │ • Source branch d ∈ {0, 1}                             │
           │ • Private Running Mean μ_d & Variance σ_d^2            │
           │ • Private Learnable Affine Scaling (γ_d, β_d)          │
           │ • Zero-Shot Target Rule: Evaluate via primary branch 0 │
           └────────────────────────────┬───────────────────────────┘
                                        │ z_norm ∈ R^(B × 896)
                                        ▼
           ┌────────────────────────────────────────────────────────┐
           │ Ordinal Classification Head (MLP)                      │
           │ • Linear(896 → 256)                                    │
           │ • ReLU Activation + Dropout(p = 0.3)                   │
           │ • Linear(256 → 5)                                      │
           │ • Output Logits L ∈ R^(B × 5)                          │
           └────────────────────────────┬───────────────────────────┘
                                        │
                        Softmax Posterior Probabilities p
                                        │
            ┌───────────────────────────┴───────────────────────────┐
            ▼                                                       ▼
┌───────────────────────────────────────┐   ┌───────────────────────────────────────┐
│ Loss Function (Training Phase)        │   │ Post-Hoc Calibration (Inference Phase)│
│ • True-class cross-entropy L_CE       │   │ • Temperature Scaling (T = 1.42)      │
│ • Quadratic ordinal penalty L_ord     │   │ • Expected score E[c] = Σ c · p_c     │
│ • Composite loss: L_CE + λ L_ord      │   │ • Nelder-Mead Simplex Thresholds T*   │
└───────────────────────────────────────┘   └───────────────────────────────────────┘
```

---

## 2. Mathematical Formulations

### 2.1 Feature Concatenation & Latent Projection

Given an input fundus image $\mathbf{X} \in \mathbb{R}^{3 	imes 224 	imes 224}$:
- Convolutional stream: $\mathbf{f}_{	ext{CNN}} = 	ext{GAP}(	ext{EfficientNet-B3}(\mathbf{X})) \in \mathbb{R}^{1536}$
- Vision Transformer stream: $\mathbf{f}_{	ext{ViT}} = 	ext{DINOv2-ViT-S}(\mathbf{X})_{[	ext{CLS}]} \in \mathbb{R}^{384}$

The combined representation is formed via vector concatenation followed by an affine projection into an 896-dimensional latent space:
$$\mathbf{f}_{	ext{fused}} = [\mathbf{f}_{	ext{CNN}} \,\|\, \mathbf{f}_{	ext{ViT}}] \in \mathbb{R}^{1920}$$
$$\mathbf{z} = \mathbf{W}_{	ext{proj}} \mathbf{f}_{	ext{fused}} + \mathbf{b}_{	ext{proj}} \in \mathbb{R}^{896}$$

### 2.2 Domain-Specific Batch Normalization (DSBN)

To decouple sensor-specific acquisition noise from invariant retinal pathology without destroying delicate microvascular lesion signals, DSBN maintains independent statistical estimators and affine parameters for each source domain $d \in \{0, 1\}$:

$$\hat{\mathbf{z}}_d = rac{\mathbf{z} - oldsymbol{\mu}_d}{\sqrt{oldsymbol{\sigma}_d^2 + \epsilon}}$$
$$	ilde{\mathbf{z}} = oldsymbol{\gamma}_d \odot \hat{\mathbf{z}}_d + oldsymbol{eta}_d$$

where:
- $oldsymbol{\mu}_d, oldsymbol{\sigma}_d^2 \in \mathbb{R}^{896}$ are domain-specific running mean and variance trackers updated via exponential decay during training.
- $oldsymbol{\gamma}_d, oldsymbol{eta}_d \in \mathbb{R}^{896}$ are domain-specific learnable scale and shift parameters.
- $\epsilon = 10^{-5}$ is a numerical stability constant.

### 2.3 Distance-Weighted Cross-Entropy (DWCE) Loss

Standard multi-class cross-entropy treats all classification mistakes equally. In diabetic retinopathy grading, predicting Grade 0 instead of Grade 1 is a minor discrepancy, whereas predicting Grade 0 instead of Grade 4 (Proliferative DR) is clinically catastrophic. The DWCE loss explicitly penalizes ordinal distance:

$$\mathcal{L}_{\mathrm{DWCE}} = \mathcal{L}_{\mathrm{CE}} + \lambda_{\mathrm{dist}} \mathcal{L}_{\mathrm{ord}}$$

$$\mathcal{L}_{\mathrm{CE}} = -rac{1}{N} \sum_{i=1}^N \log p_{i, y_i}$$

$$\mathcal{L}_{\mathrm{ord}} = rac{1}{N} \sum_{i=1}^N \sum_{c=0}^4 rac{|y_i - c|^2}{(C - 1)^2} p_{i,c}$$

with $C = 5$, $\lambda_{\mathrm{dist}} = 1.0$, and $(C - 1)^2 = 16$.

---

## 3. Computational Parameter Budget

| Module Component | Backbone / Layer | Trainable Parameters | Output Dimension | FLOPs (G) |
| :--- | :--- | :---: | :---: | :---: |
| **Stream A (CNN)** | EfficientNet-B3 | $12.2	ext{ M}$ | $1,536$ | $1.8	ext{ G}$ |
| **Stream B (ViT)** | DINOv2 ViT-Small/14 | $22.1	ext{ M}$ | $384$ | $4.6	ext{ G}$ |
| **Projection Layer** | Linear($1920 ightarrow 896$) | $1.72	ext{ M}$ | $896$ | $< 0.01	ext{ G}$ |
| **DSBN Module** | 2 Branches $	imes (oldsymbol{\gamma}, oldsymbol{eta})$ | $3.58	ext{ K}$ | $896$ | $< 0.001	ext{ G}$ |
| **Classifier Head** | MLP($896 ightarrow 256 ightarrow 5$) | $0.23	ext{ M}$ | $5$ | $< 0.001	ext{ G}$ |
| **Total System** | **Hybrid CNN-ViT + DSBN** | **$34.3	ext{ M}$** | **$5$ (Classes)** | **$6.4	ext{ G}$** |
