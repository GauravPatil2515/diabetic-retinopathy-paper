# Multi-Center Dataset Specifications & Cohort Demographics

This document provides complete, verified clinical and technical specifications for all datasets evaluated in our Leave-One-Dataset-Out (LODO) cross-domain benchmark.

---

## 📊 Summary of Evaluated Datasets

| Dataset | Cohort Source / Country | Total Images | Resolution (Raw) | Acquisition Modality / Sensor | ETDRS Classes | Primary Domain Shift Challenge |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **APTOS 2019** | Aravind Eye Hospital, India | 3,662 | $1536 \times 2048$ to $3456 \times 5184$ | Multiple camera models (rural mobile camps) | 5 Grades (0–4) | Heterogeneous lighting, field-of-view vignetting, dust artifacts |
| **Messidor-2** | France (Brest, Paris, Saint-Étienne) | 1,744 | $1440 \times 960$ to $2304 \times 1536$ | Topcon TRC NW6 non-mydriatic camera (45° FOV) | 5 Grades (0–4) | Standardized hospital lighting, narrow contrast, subtle microaneurysms |
| **IDRiD** | Nanded, India | 516 | $4288 \times 2848$ | Kowa VX-10α digital fundus camera (50° FOV) | 5 Grades (0–4) | High spatial resolution, dense micro-lesions, pixel-level annotation |
| **DDR** | Multi-hospital clinical registry, China | 13,673 | Diverse ($1000^2$ to $3000^2$) | Multi-vendor fundus sensors | 5 Grades (0–4) + Quality | Severe real-world class imbalance, low-quality clinical discards |
| **EyePACS (Aux)**| California Healthcare Foundation, USA | 4,200 (lite subset) | High-res varied | Telemedicine fundus cameras | Binary / 5 Grades | Severe compression artifacts, non-uniform aspect ratios |

---

## 🔬 Class Distribution & Imbalance Audit

The following ground-truth distribution was determined via the automated dataset audit (`results/dataset_audit_report.json`):

### 1. APTOS 2019 Blindness Detection
- **Total Validated Images:** 3,662
- **Class Breakdown:**
  - Grade 0 (No DR): **1,805** (49.29%)
  - Grade 1 (Mild DR): **370** (10.10%)
  - Grade 2 (Moderate DR): **999** (27.28%)
  - Grade 3 (Severe DR): **193** (5.27%)
  - Grade 4 (Proliferative DR): **295** (8.06%)
- **Imbalance Ratio:** 9.35 : 1 (Grade 0 vs. Grade 3)

### 2. Messidor-2 Reference Cohort
- **Total Validated Images:** 1,744 (874 paired patient eyes)
- **Class Breakdown:**
  - Grade 0 (No DR): **1,017** (58.31%)
  - Grade 1 (Mild DR): **270** (15.48%)
  - Grade 2 (Moderate DR): **347** (19.90%)
  - Grade 3 (Severe DR): **75** (4.30%)
  - Grade 4 (Proliferative DR): **35** (2.01%)
- **Imbalance Ratio:** 29.06 : 1 (Grade 0 vs. Grade 4)

### 3. IDRiD Benchmark Cohort
- **Total Images:** 516 (413 train, 103 test)
- **High-grade prevalence:** Includes detailed lesion masks for microaneurysms, hemorrhages, hard exudates, and soft exudates (cotton wool spots).

---

## 🏥 Clinical Ground-Truth Definition: ICDR Scale

All cohorts adhere to the **International Clinical Diabetic Retinopathy (ICDR)** severity scale, derived from the landmark Early Treatment Diabetic Retinopathy Study (ETDRS):

| Grade | Clinical Designation | ETDRS Diagnostic Criteria | Hallmark Lesions Visualized |
| :---: | :--- | :--- | :--- |
| **0** | **No DR** | Absence of any observable retinal vascular lesions. | Clear fovea, intact vascular arcades. |
| **1** | **Mild NPDR** | Presence of microaneurysms (MAs) only. | Tiny red punctate dots isolated in vascular periphery. |
| **2** | **Moderate NPDR** | More than microaneurysms but less than Severe NPDR. | Microaneurysms, blot hemorrhages, hard exudates, venous beading in <2 quadrants. |
| **3** | **Severe NPDR** | Adheres to the **ETDRS 4-2-1 Rule**: <br> • >20 intraretinal hemorrhages in each of 4 quadrants, OR <br> • Definite venous beading in $\ge$ 2 quadrants, OR <br> • Prominent IRMA in $\ge$ 1 quadrant. | Extensive deep retinal hemorrhages, severe vascular leakage, cotton wool spots. |
| **4** | **Proliferative DR** | Neovascularization (NVD/NVE) and/or vitreous or preretinal hemorrhage. | Fragile new blood vessels sprouting from disc or retina, fibrous proliferation. |

---

## ⚡ Domain Shift Characterization

When training on APTOS 2019 and testing directly on Messidor-2 without domain adaptation:
1. **Sensor Color Bias:** APTOS images exhibit pronounced yellow-green tint from rural mobile sensors; Messidor-2 exhibits cool-magenta calibration from Topcon hospital cameras.
2. **Field-of-View Variations:** APTOS images vary between 40° and 50° FOV with wide black circular borders; Messidor-2 uses standardized 45° FOV with tight peripheral masking.
3. **Severe Grade Misalignment:** Standard models collapse on Grade 1 and Grade 3 when shifted across sensors because localized microaneurysms are suppressed by sensor noise. Our **Dual-Stream DSBN** decouples these center-specific distributions to preserve lesion fidelity.
