# IEEE Double-Blind Submission Audit & Anonymization Checklist

This document records the rigorous double-blind compliance audit conducted on all manuscript source files, figures, tables, and supplementary code for IEEE conference review.

---

## 🔒 Anonymization Compliance Status: 100% VERIFIED

| Audit Criterion | Verification Method | Result | Notes |
| :--- | :--- | :---: | :--- |
| **Author Names & Affiliations** | Regex scan of `paper/main.tex` and `conferences/*/main.tex` | ✅ PASSED | Replaced with `Anonymous Author(s)` and `Anonymous Institution` under `\iftrue ... \fi` block. |
| **Grant & Funding Acknowledgments** | Text inspection of acknowledgments section | ✅ PASSED | Funding numbers and institutional grant names omitted during review. |
| **Figure Metadata & EXIF** | ExifTool / PNG metadata parser across `figures/` | ✅ PASSED | Zero author usernames, email addresses, or workstation paths embedded. |
| **PDF Properties & Creator Tags** | `pdfinfo` inspection of compiled `main.pdf` | ✅ PASSED | `Author` field blank, `Creator` set to standard TeX Live / pdfTeX. |
| **Self-Citations (Third-Person)** | Reference analysis in `references.bib` | ✅ PASSED | All self-references phrased in objective third person ("Patil et al. previously observed..." rather than "In our previous work..."). |
| **GitHub Repository Links** | Hyperlink audit in abstract and conclusion | ✅ PASSED | Public release URLs anonymized with placeholder `<https://github.com/anonymized-dr-review/dr-paper>` for peer review. |

---

## 🛠️ Automated Anonymization Verification Script

To re-verify zero-leakage compliance at any time, execute:
```bash
python3 scripts/verify_anonymization.py
```
This utility scans all `.tex`, `.bib`, and `.md` files for author handles, local machine directory paths, and institutional markers.
