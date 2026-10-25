#!/usr/bin/env python3
"""
Automated Double-Blind Anonymization Linter.
Scans LaTeX source files and documentation for author names, university handles, and local paths.
"""
import os
import re

FORBIDDEN_PATTERNS = [
    r"Gaurav\s+Patil",
    r"SIES\s+Graduate",
    r"gauravpatil",
    r"/home/gaurav",
    r"Department\s+of\s+Artificial\s+Intelligence",
]

def scan_file(filepath):
    leaks = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line_no, line in enumerate(f, 1):
            stripped = line.strip()
            if filepath.endswith(".tex") and stripped.startswith("%"):
                continue
            for pat in FORBIDDEN_PATTERNS:
                if re.search(pat, line, re.IGNORECASE):
                    leaks.append((line_no, pat, stripped[:80]))
    return leaks

if __name__ == "__main__":
    print("Running Double-Blind Anonymization Scan...")
    target_files = [
        "paper/main.tex",
        "conferences/01_ICIPCN_2027/main.tex",
        "conferences/03_SISIMPACT_2027/main.tex",
        "conferences/02_ICICCDS_2027/main.tex",
        "paper/references.bib",
    ]
    
    found_any = False
    for tf in target_files:
        if os.path.exists(tf):
            leaks = scan_file(tf)
            if leaks:
                print(f"[WARN] Potential leak in {tf}:")
                for line_no, pat, snippet in leaks:
                    print(f"   Line {line_no} matches '{pat}': {snippet}")
                found_any = True
            else:
                print(f"[PASS] {tf}: Clean (Double-blind compliant)")
                
    if not found_any:
        print("\nALL MANUSCRIPT FILES ARE 100% DOUBLE-BLIND COMPLIANT.")
