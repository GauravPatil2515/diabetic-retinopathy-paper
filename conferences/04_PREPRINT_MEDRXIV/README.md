# bioRxiv / medRxiv Submission Guide & Pre-Formatted Metadata

**Manuscript Title**: *A Domain-Aware Hybrid CNN–Vision Transformer Framework for Cross-Domain Diabetic Retinopathy Grading*  
**Ready-to-Upload File**: `preprint_package/manuscript.pdf`

---

## 📌 Step-by-Step Upload Instructions for bioRxiv / medRxiv

### Step 1: Login to Portal
- **medRxiv Portal**: [https://submit.medrxiv.org](https://submit.medrxiv.org) *(Recommended for medical imaging/ophthalmology)*
- **bioRxiv Portal**: [https://submit.biorxiv.org](https://submit.biorxiv.org)
- Log in with your **ORCID** or email account.

### Step 2: Select Article Type & Category
- **Article Type**: **Original Research Article**
- **Subject Area**: 
  - On **medRxiv**: Choose **Ophthalmology** or **Health Informatics** or **Medical Imaging**.
  - On **bioRxiv**: Choose **Bioengineering** or **Bioinformatics**.

### Step 3: Copy-Paste Metadata Fields
Copy and paste the exact strings below directly into the web form fields:

#### 1. Title
```
A Domain-Aware Hybrid CNN-Vision Transformer Framework for Cross-Domain Diabetic Retinopathy Grading
```

#### 2. Running Title / Short Title
```
Domain-Aware Hybrid Framework for DR Grading
```

#### 3. Abstract
```
Deep learning models for Diabetic Retinopathy (DR) grading often exhibit reduced performance when deployed across heterogeneous clinical domains. Heterogeneous fundus camera profiles, lighting variances, and acquisition artifacts cause feature distribution drift. To address this challenge, we propose a domain-aware hybrid framework fusing self-supervised DINOv2 Vision Transformer (ViT) embeddings with EfficientNet-B3 CNN representations. The framework integrates four complementary components. First, Domain-Specific Batch Normalization (DSBN) isolates camera-specific style statistics. Second, Exponential Moving Average (EMA) stabilizes optimization. Third, Distance-Weighted Cross-Entropy (DWCE) addresses ordinal grading penalty structure. Finally, post-hoc continuous decision boundary calibration aligns continuous predictions with clinical severity thresholds. Evaluated under a 3-domain Leave-One-Domain-Out (LODO) protocol across APTOS 2019, Messidor-2, and IDRiD benchmarks, our architecture achieves an average cross-domain Quadratic Weighted Kappa (QWK) of 0.750 +/- 0.010, achieving higher cross-domain performance than standard shared-normalization baselines (0.182 +/- 0.015). Furthermore, Nelder-Mead boundary optimization reduces Expected Calibration Error (ECE) from 0.143 to 0.041 (down 71.3%), expanding the Moderate-to-Severe boundary (2.500 -> 2.751). The optimized threshold numerically resembles the conservative diagnostic boundary used in the ETDRS 4-2-1 rule, although no prospective clinical validation was performed. Grad-CAM visualizations confirm domain-invariant targeting of pathological exudates and hemorrhages across diverse camera domains.
```

#### 4. Keywords
```
Diabetic Retinopathy, Domain Adaptation, Vision Transformers, DINOv2, Domain-Specific Batch Normalization, Decision Boundary Calibration, Cross-Domain Generalization, Medical Image Analysis
```

#### 5. Data Availability Statement
```
All fundus photograph datasets analyzed in this study are fully anonymized, open-access public research benchmarks: APTOS 2019 Blindness Detection, Messidor-2, and IDRiD. No new human subject data or patient clinical trials were conducted by the authors. Source code, evaluation scripts, and pretrained weights are available in the project repository.
```

#### 6. Ethics & Human Subjects Statement
```
This study relies exclusively on anonymized, publicly available retrospective benchmarks (APTOS 2019, Messidor-2, IDRiD). No direct human subject intervention or clinical trial was conducted. Ethical compliance was managed by the original benchmark providers.
```

#### 7. Conflict of Interest / Financial Disclosure
```
The authors declare no competing financial or non-financial interests directly related to this work.
```

#### 8. License Selection
```
Creative Commons Attribution 4.0 International (CC-BY 4.0)
```

---

### Step 4: Upload PDF Manuscript
- In the file upload tab, select **Single PDF File** and upload:
  📁 `preprint_package/manuscript.pdf`

### Step 5: Finalize & Submit
- Review the generated web proof PDF and click **Submit Manuscript**.
- You will receive an official bioRxiv/medRxiv DOI link via email within **24–48 hours**!
