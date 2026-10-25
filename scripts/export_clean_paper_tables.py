#!/usr/bin/env python3
"""
Export Clean LaTeX Booktabs Tables from Benchmark Results CSVs.
"""
import os
import pandas as pd

def export_lodo_table():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    csv_path = os.path.join(repo_root, "results", "master_lodo_benchmark.csv")
    if not os.path.exists(csv_path):
        print(f"Missing {csv_path}")
        return
    df = pd.read_csv(csv_path)
    print("=== Master LODO Benchmark LaTeX Table ===")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Leave-One-Dataset-Out (LODO) Cross-Domain Generalization Benchmark (QWK)}")
    print(r"\label{tab:master_lodo}")
    print(r"\begin{tabular}{llccc}")
    print(r"\toprule")
    print(r"Held-Out Target & Source Cohorts & SharedNorm Baseline & RETFound \cite{zhou2023foundation} & \textbf{Proposed Hybrid DSBN} \\")
    print(r"\midrule")
    for _, row in df.iterrows():
        tgt = row["Target_Domain"]
        src = row["Source_Domains"]
        sn = f"{row['Baseline_SharedNorm_QWK']:.3f}"
        rf = f"{row['RETFound_QWK']:.3f}"
        prop = f"\\textbf{{{row['Proposed_Hybrid_DSBN_QWK']:.3f}}}"
        print(f"{tgt} & {src} & {sn} & {rf} & {prop} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")

if __name__ == "__main__":
    export_lodo_table()
