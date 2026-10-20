# ============================================================
# KD-LowData: Knowledge Distillation under Data Scarcity
# Universal GPU Notebook (Works 100% Free on Kaggle & Google Colab)
#
# SETUP INSTRUCTIONS:
#   1. Push your kd-lowdata repository to GitHub.
#   2. Platform Choice:
#      - On Kaggle: New Notebook -> Settings -> Accelerator -> GPU P100 (or T4 x1).
#      - On Colab:  New Notebook -> Runtime -> Change runtime type -> T4 GPU.
#   3. Ensure Internet is enabled (Kaggle: Settings -> Internet ON).
#   4. Run cells in order: CELL 1 -> 2 -> 3 -> 4 -> (PHASE 1) -> PHASE 2 -> PHASE 3.
#
# TOTAL ESTIMATED TIME ON FREE GPU:
#   Phase 2 Grid (Seed 0, fast budget): ~3.5 hours on P100 (~4.5 hours on T4)
#   Phase 3 Plots & Summary tables   : < 5 minutes
#   Total: ~3.5 to 4.5 hours total (fits inside a single free session!)
#
# The grid is FULLY RESUMABLE if interrupted.
# ============================================================


# -----------------------------------------------------------
# CELL 1: Clone repo, check GPU, install deps
# -----------------------------------------------------------
import os
import subprocess
import sys

# Auto-detect platform: Google Colab vs Kaggle vs Local
IS_COLAB = "google.colab" in sys.modules or os.path.exists("/content")
IS_KAGGLE = os.path.exists("/kaggle")
BASE_DIR = "/content" if IS_COLAB else ("/kaggle/working" if IS_KAGGLE else os.path.abspath("."))
print(f"Platform: {'Google Colab' if IS_COLAB else ('Kaggle' if IS_KAGGLE else 'Local/Other')}")
print(f"Base Directory: {BASE_DIR}")

REPO_URL = "https://github.com/YOUR_USERNAME/kd-lowdata.git"  # !! CHANGE THIS !!
REPO_DIR = os.path.join(BASE_DIR, "kd-lowdata")

if not os.path.exists(REPO_DIR):
    print("Cloning repository...")
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
else:
    print("Repo exists, pulling latest...")
    subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)

print("Repo contents:", os.listdir(REPO_DIR))

import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"VRAM: {vram:.1f} GB")
else:
    print("WARNING: No GPU found! On Colab: Runtime -> Change runtime type -> T4 GPU. On Kaggle: Settings -> Accelerator -> GPU P100/T4.")

subprocess.run(["pip", "install", "-q", "pyyaml", "tqdm", "matplotlib", "pandas"], check=True)
print("Dependencies OK.")


# -----------------------------------------------------------
# CELL 2: Configure paths and load config
# -----------------------------------------------------------
from pathlib import Path
import yaml

DATA_DIR   = os.path.join(BASE_DIR, "data")          # CIFAR-100 auto-downloads here (~170 MB)
OUTPUT_DIR = os.path.join(BASE_DIR, "results")       # CSV files and plots
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")   # Model checkpoints

os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR,   exist_ok=True)

SRC_DIR     = os.path.join(REPO_DIR, "src")
CONFIG_PATH = os.path.join(REPO_DIR, "configs", "kaggle_fast.yaml")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

with open(CONFIG_PATH) as f:
    cfg = yaml.safe_load(f)

print("Config loaded:", CONFIG_PATH)
print("  Methods:           ", cfg["methods"])
print("  Student fractions: ", cfg["student_fractions"])
print("  Teacher fractions: ", cfg["teacher_fractions"])
print("  Seeds:             ", cfg["seeds"])
print("  Batch size:        ", cfg["batch_size"])
print("  Total student runs: 24 (Seed 0)")
print("  Total teacher runs: 4  (Seed 0)")
print("  Estimated time:     ~3.5h on P100 / ~4.5h on T4")

# Shorthand for building subprocess commands
RUN_GRID    = [sys.executable, os.path.join(REPO_DIR, "src", "run_grid.py")]
COMMON_ARGS = [
    "--config", CONFIG_PATH,
    "--data_dir", DATA_DIR,
    "--output_dir", OUTPUT_DIR,
    "--checkpoint_dir", CKPT_DIR,
    "--num_workers", "2",
]


# -----------------------------------------------------------
# CELL 3: Smoke tests (validates environment, ~2 minutes)
# -----------------------------------------------------------
smoke = subprocess.run(
    [sys.executable, "-m", "pytest",
     os.path.join(REPO_DIR, "tests", "test_smoke.py"),
     "-v", "--tb=short", "-x"],
    capture_output=True, text=True,
    env=dict(os.environ, PYTHONPATH=SRC_DIR)
)
out = smoke.stdout
print(out[-3000:] if len(out) > 3000 else out)
if smoke.returncode != 0:
    print("STDERR:", smoke.stderr[-500:])
    raise RuntimeError("Smoke tests FAILED. Fix errors before continuing!")
print("All smoke tests passed!")


# -----------------------------------------------------------
# CELL 4: Dry run - preview the plan (no training)
# -----------------------------------------------------------
print("=" * 60)
print("MAIN GRID PLAN (24 student runs + 4 teacher runs)")
print("=" * 60)
subprocess.run(RUN_GRID + COMMON_ARGS + ["--dry-run"])


# -----------------------------------------------------------
# PHASE 1 (OPTIONAL): Hyperparameter Tuning
#
# NOTE: Standard CIFAR-100 distillation hyperparameters
# (lr=0.1, T=4.0, alpha=0.9, at_beta=1000.0) are already built-in.
#
# - To finish in ~3.5h: LEAVE RUN_TUNING = False and proceed to Phase 2!
# - If you wish to tune: set RUN_TUNING = True (~1.2h with tune-max-steps=2000).
# -----------------------------------------------------------
RUN_TUNING = False

if RUN_TUNING:
    TUNE_MAX_STEPS = 2000
    print("Starting Phase 1: Hyperparameter Tuning...")
    tune_res = subprocess.run(
        RUN_GRID + COMMON_ARGS + [
            "--tune",
            "--tune-fractions", "0.01", "0.10", "1.0",
            "--tune-max-steps", str(TUNE_MAX_STEPS),
        ],
        cwd=REPO_DIR
    )
    print("Phase 1 finished. Exit code:", tune_res.returncode)
else:
    print("Skipping Phase 1 tuning. Using standard CIFAR-100 defaults.")


# -----------------------------------------------------------
# PHASE 2: Experiment Grid (~3.5h on P100, ~4.5h on T4)
#
# 24 student runs + 4 teacher checkpoints on Seed 0.
# - Automatically skips already-completed runs (SAFE TO RE-RUN).
# - Appends one row to results.csv per completed run.
# - Each teacher checkpoint is trained once and reused.
# -----------------------------------------------------------
print("Starting Phase 2: Experiment Grid...")
print("(Resumable: re-running skips completed runs)")
grid_res = subprocess.run(RUN_GRID + COMMON_ARGS, cwd=REPO_DIR)
print("Phase 2 finished. Exit code:", grid_res.returncode)


# -----------------------------------------------------------
# PHASE 3: Results Analysis & Plots
# Run at any time during or after training to see progress.
# -----------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd

results_csv = os.path.join(OUTPUT_DIR, "results.csv")

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

if not os.path.exists(results_csv):
    print("results.csv not found yet. Run Phase 2 first.")
else:
    df = pd.read_csv(results_csv)
    df["f_S"] = df["f_S"].astype(float)

    print("Completed runs:", len(df), "/ 24")
    print()

    # Summary table
    summary = (
        df.groupby(["regime", "method", "f_S"])["test_acc"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .round(2)
    )
    print(summary.to_string(index=False))

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
    plt.savefig(os.path.join(OUTPUT_DIR, "kd_gain_vs_fraction.png"), dpi=150, bbox_inches="tight")
    plt.show()

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
    plt.savefig(os.path.join(OUTPUT_DIR, "regime_comparison.png"), dpi=150, bbox_inches="tight")
    plt.show()

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
        plt.savefig(os.path.join(OUTPUT_DIR, "kd_gain_heatmap.png"), dpi=150, bbox_inches="tight")
        plt.show()

    # Throughput summary
    mean_sps = df["steps_per_sec"].mean()
    remaining = max(0, 24 - len(df))
    print()
    print("Throughput summary:")
    th = (
        df.groupby("f_S")["steps_per_sec"]
        .agg(["mean", "min", "max"])
        .round(1)
        .reset_index()
    )
    print(th.to_string(index=False))
    if mean_sps > 0 and remaining > 0:
        hrs = remaining * 20000 / mean_sps / 3600
        print(f"\nMean throughput: {mean_sps:.1f} steps/sec")
        print(f"Estimated time remaining ({remaining} runs): ~{hrs:.1f} hours")


# -----------------------------------------------------------
# FINAL CELL: Zip results for download
# -----------------------------------------------------------
import shutil

zip_path = os.path.join(BASE_DIR, "kd_results")
shutil.make_archive(zip_path, "zip", OUTPUT_DIR)
print(f"Zipped results to: {zip_path}.zip")
print("Contents:")
for p in sorted(Path(OUTPUT_DIR).rglob("*")):
    if p.is_file():
        mb = p.stat().st_size / 1e6
        rel = str(p.relative_to(OUTPUT_DIR))
        print(f"  {rel}  ({mb:.2f} MB)")

if IS_COLAB:
    try:
        from google.colab import files
        print("\nTriggering browser download for kd_results.zip...")
        files.download(f"{zip_path}.zip")
    except Exception as e:
        print(f"\nNote: You can download {zip_path}.zip from the Files panel on the left sidebar.")
