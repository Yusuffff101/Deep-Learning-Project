"""
plot_results.py — Generate all evaluation plots and summary tables from results.csv.

Usage:
  python src/plot_results.py
  python src/plot_results.py --csv results/results.csv --output_dir results
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

COLORS = {
    "ce": "#555555",
    "logit_kd": "#E45C3A",
    "attention_transfer": "#3A82E4"
}
LABELS = {
    "ce": "CE (baseline)",
    "logit_kd": "Logit KD (Hinton 2015)",
    "attention_transfer": "Attention Transfer (AT)"
}


def generate_plots_and_summary(csv_path: str = "results/results.csv", output_dir: str = "results") -> None:
    csv_file = Path(csv_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_file.exists():
        print(f"[ERROR] Results file not found at: {csv_file}")
        print("Run the training grid first before generating plots.")
        return

    df = pd.read_csv(csv_file)
    df["f_S"] = df["f_S"].astype(float)
    total_runs = len(df)
    print(f"\n{'='*60}")
    print(f"  KD-LowData: Analyzing Results ({total_runs} completed runs)")
    print(f"{'='*60}\n")

    # Summary table
    summary = (
        df.groupby(["regime", "method", "f_S"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .round(2)
    )
    print("Test Accuracy Summary:")
    print(summary.to_string(index=False))
    print()

    # ----------------------------------------------------------
    # PLOT 1: Accuracy vs Student Fraction (Regime A and B)
    # ----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for regime, ax in zip(["A", "B"], axes):
        regime_df = df[df["regime"] == regime]
        for method in ["ce", "logit_kd", "attention_transfer"]:
            m_df = regime_df[regime_df["method"] == method]
            if m_df.empty:
                continue
            g = m_df.groupby("f_S")["test_acc"].agg(["mean", "std"]).reset_index()
            yerr = g["std"].fillna(0)
            ax.errorbar(
                g["f_S"] * 100, g["mean"], yerr=yerr,
                marker="o", label=LABELS.get(method, method), color=COLORS.get(method, "#333333"),
                linewidth=2, capsize=4
            )
        if regime == "A":
            title = "Regime A: Teacher trained on 100% data"
        else:
            title = "Regime B: Teacher trained on same subset as student"
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Student Data Fraction (%)", fontsize=11)
        ax.set_ylabel("Test Accuracy (%)", fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.set_xticks([1, 5, 10, 25, 100])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.suptitle(
        "KD-LowData: CIFAR-100 Test Accuracy vs Student Data Fraction",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    p1 = out_dir / "kd_gain_vs_fraction.png"
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p1}")

    # ----------------------------------------------------------
    # PLOT 2: Regime A vs B comparison (does teacher data matter?)
    # ----------------------------------------------------------
    shared_fracs = [0.05, 0.10, 0.25]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    RCOL = {"A": "#E45C3A", "B": "#3A82E4"}

    for ax, method in zip(axes, ["ce", "logit_kd", "attention_transfer"]):
        for regime in ["A", "B"]:
            mask = (
                (df["regime"] == regime) &
                (df["method"] == method) &
                (df["f_S"].isin(shared_fracs))
            )
            r_df = df[mask]
            if r_df.empty:
                continue
            g = r_df.groupby("f_S")["test_acc"].agg(["mean", "std"]).reset_index()
            yerr = g["std"].fillna(0)
            rlabel = "Regime A (teacher=100%)" if regime == "A" else "Regime B (teacher=student)"
            ax.errorbar(
                g["f_S"] * 100, g["mean"], yerr=yerr,
                marker="s", label=rlabel,
                color=RCOL[regime], linewidth=2, capsize=4
            )
        ax.set_title(LABELS.get(method, method), fontsize=12)
        ax.set_xlabel("Student Data Fraction (%)", fontsize=11)
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(mtick.ScalarFormatter())
        ax.set_xticks([5, 10, 25])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Test Accuracy (%)", fontsize=11)
    plt.suptitle(
        "Does Teacher Training Data Matter? Regime A vs Regime B",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    p2 = out_dir / "regime_comparison.png"
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {p2}")

    # ----------------------------------------------------------
    # PLOT 3: KD Gain Heatmap vs CE baseline (Regime A)
    # ----------------------------------------------------------
    pivot = (
        df[df["regime"] == "A"]
        .groupby(["method", "f_S"])["test_acc"]
        .mean()
        .unstack("f_S")
    )

    if "ce" in pivot.index:
        gain = pivot.subtract(pivot.loc["ce"]).drop("ce")

        fig, ax = plt.subplots(figsize=(9, 3))
        im = ax.imshow(gain.values, cmap="RdYlGn", aspect="auto", vmin=-2, vmax=8)
        plt.colorbar(im, ax=ax, label="KD Gain over CE baseline (%)")

        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([str(int(f * 100)) + "%" for f in pivot.columns])
        ax.set_yticks(range(len(gain.index)))
        ax.set_yticklabels([LABELS.get(m, m) for m in gain.index])
        ax.set_xlabel("Student Data Fraction")
        ax.set_title("KD Gain vs CE Baseline (Regime A, Teacher=100% data)", fontweight="bold")

        for i in range(len(gain.index)):
            for j in range(len(pivot.columns)):
                v = gain.values[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:+.1f}%", ha="center", va="center",
                            fontsize=11, fontweight="bold")

        plt.tight_layout()
        p3 = out_dir / "kd_gain_heatmap.png"
        plt.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {p3}")

    # Package output into zip
    zip_out = out_dir.parent / "kd_results"
    shutil.make_archive(str(zip_out), "zip", str(out_dir))
    print(f"\nPackaged all results and plots into: {zip_out}.zip")


def main():
    p = argparse.ArgumentParser(description="Generate plots and summary from results.csv")
    p.add_argument("--csv", default="results/results.csv", help="Path to results.csv")
    p.add_argument("--output_dir", default="results", help="Directory to save plots")
    args = p.parse_args()
    generate_plots_and_summary(args.csv, args.output_dir)


if __name__ == "__main__":
    main()
