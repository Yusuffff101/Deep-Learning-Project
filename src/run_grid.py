"""
run_grid.py — Enumerate and execute the full KD experiment grid.

Regime A: teacher trained on f_T = 1.0 (full data), student on f_S.
Regime B: teacher trained on f_T = f_S (same subset as student).

The grid loops over:
  (regime, method, f_T, f_S, seed)

Design decisions
-----------------
- SKIP runs whose result row already exists in results.csv.
  This makes the grid safely resumable after Kaggle session interruptions.
- --dry-run prints all planned runs without training.
- One CSV row is appended per completed run; the file is never rewritten,
  so partial results are preserved if the process is killed.
- Teacher checkpoints for Regime A (f_T=1.0) are reused across students
  with the same seed, so the teacher is only trained once per seed.

Usage
-----
# See what will run (no training):
python src/run_grid.py --dry-run

# Run the full grid:
python src/run_grid.py --config configs/default.yaml

# Override paths for Kaggle:
python src/run_grid.py --data_dir /kaggle/input/cifar100 \\
    --output_dir /kaggle/working/results \\
    --checkpoint_dir /kaggle/working/checkpoints
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Optional

# Ensure src/ is on the path when called from the project root
sys.path.insert(0, str(Path(__file__).parent))

from train import build_config, train
from evaluate import evaluate_checkpoint
from utils import load_yaml_config, make_run_id, save_yaml, setup_logging

logger = logging.getLogger(__name__)

# ── CSV schema ────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "regime", "method", "f_T", "f_S", "seed",
    "val_acc", "test_acc", "agreement", "ece",
    "train_steps", "wall_time_s", "steps_per_sec",
]

TUNING_CSV_COLUMNS = [
    "stage", "method", "f_S", "param_name", "param_val",
    "lr", "T", "alpha", "at_beta", "tune_max_steps",
    "val_acc", "train_steps", "wall_time_s", "steps_per_sec",
]


# ── Result existence check ────────────────────────────────────────────────────

def _load_existing_results(csv_path: Path) -> set[tuple]:
    """Return a set of (regime, method, f_T, f_S, seed) tuples already in CSV."""
    done: set[tuple] = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                regime = row["regime"]
                method = row["method"]
                f_t = float(row["f_T"])
                f_s = float(row["f_S"])
                seed = int(row["seed"])
                done.add((regime, method, f_t, f_s, seed))
                # Treat Regime A f_S=1.0 and Regime B f_T=f_S=1.0 as the same experiment
                if abs(f_t - 1.0) < 1e-5 and abs(f_s - 1.0) < 1e-5:
                    done.add(("A", method, 1.0, 1.0, seed))
                    done.add(("B", method, 1.0, 1.0, seed))
            except (KeyError, ValueError):
                pass  # Skip malformed rows
    return done


def _append_result_row(csv_path: Path, row: dict) -> None:
    """Append a single row to the results CSV, creating the header if needed."""
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── Grid construction ─────────────────────────────────────────────────────────

def build_grid(cfg: dict) -> list[dict]:
    """Return the full list of run specs as a list of dicts.

    Each dict has keys: regime, method, f_T, f_S, seed, + all cfg fields.
    """
    runs = []
    teacher_fractions = cfg["teacher_fractions"]
    student_fractions = cfg["student_fractions"]
    methods = cfg["methods"]
    seeds = cfg["seeds"]

    default_lr = cfg.get("lr", 0.1)
    default_T = cfg.get("T", cfg.get("temperature", 4.0))
    default_alpha = cfg.get("alpha", 0.9)
    default_beta = cfg.get("at_beta", 1000.0)

    for seed, method in product(seeds, methods):
        # ── Regime A: teacher always on 100% data ───────────────────────────
        f_T_A = 1.0
        for f_S in student_fractions:
            runs.append({
                **cfg,
                "regime": "A",
                "method": method,
                "f_t": f_T_A,
                "f_s": f_S,
                "seed": seed,
                "lr": default_lr,
                "T": default_T,
                "alpha": default_alpha,
                "at_beta": default_beta,
            })

        # ── Regime B: teacher on same subset as student ─────────────────────
        for f_S in student_fractions:
            # In regime B, f_T = f_S; only include teacher_fractions values.
            # Skip f_S=1.0 because it is identical to Regime A (f_T=1.0, f_S=1.0)
            if f_S not in teacher_fractions or abs(f_S - 1.0) < 1e-5:
                continue  # Skip f_S values that don't exist as f_T options or duplicate Regime A
            runs.append({
                **cfg,
                "regime": "B",
                "method": method,
                "f_t": f_S,   # f_T == f_S
                "f_s": f_S,
                "seed": seed,
                "lr": default_lr,
                "T": default_T,
                "alpha": default_alpha,
                "at_beta": default_beta,
            })

    return runs


# ── Hyperparameter tuning helpers ─────────────────────────────────────────────

def _parse_fractions(raw) -> list[float]:
    """Parse fractions from a list, tuple, or comma-separated string."""
    if isinstance(raw, (list, tuple)):
        items = []
        for x in raw:
            if isinstance(x, str) and "," in x:
                items.extend(float(v.strip()) for v in x.split(",") if v.strip())
            else:
                items.append(float(x))
        return items
    elif isinstance(raw, str):
        return [float(v.strip()) for v in raw.split(",") if v.strip()]
    return [0.01, 0.10, 1.0]


def build_sequential_tuning_plan(
    cfg: dict,
    tune_fractions: list[float] | None = None,
) -> list[dict]:
    """Build the planned sequence of tuning steps across (method, f_S) cells.

    Candidate grids:
      lr candidates (3 values): [0.05, 0.1, 0.2]
      For CE: only lr is tuned (3 runs per fraction).
      For logit_kd:
        Stage 1: lr (3 values) with default T=4.0, alpha=0.9
        Stage 2: T (3 values: [2.0, 4.0, 8.0]) with best lr
        Stage 3: alpha (3 values: [0.5, 0.7, 0.9]) with best lr, best T
        (Total 9 runs per fraction)
      For attention_transfer:
        Total loss = CE + at_beta * at_loss (no logit-KD term).
        Stage 1: lr (3 values) with default at_beta=1000.0
        Stage 2: at_beta (4 values: [100.0, 500.0, 1000.0, 5000.0]) with best lr
        (Total 7 runs per fraction)

    Total per fraction = 3 + 9 + 7 = 19 runs.
    For 3 default fractions (0.01, 0.10, 1.0) = 57 runs.
    """
    if tune_fractions is None:
        tune_fractions = [0.01, 0.10, 1.0]

    methods = [m for m in cfg.get("methods", ["ce", "logit_kd", "attention_transfer"])]
    lr_candidates = [0.05, 0.1, 0.2]
    T_candidates = [2.0, 4.0, 8.0]
    alpha_candidates = [0.5, 0.7, 0.9]
    beta_candidates = [100.0, 500.0, 1000.0, 5000.0]

    default_lr = cfg.get("lr", 0.1)
    default_T = cfg.get("T", cfg.get("temperature", 4.0))
    default_alpha = cfg.get("alpha", 0.9)
    default_beta = cfg.get("at_beta", 1000.0)

    steps = []
    for f_S in tune_fractions:
        for method in methods:
            if method == "ce":
                # Stage 1: lr only
                for lr in lr_candidates:
                    steps.append({
                        **cfg,
                        "stage": "lr",
                        "param_name": "lr",
                        "param_val": lr,
                        "role": "student",
                        "regime": "A",
                        "method": "ce",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": lr,
                        "T": default_T,
                        "alpha": default_alpha,
                        "at_beta": default_beta,
                    })
            elif method == "logit_kd":
                # Stage 1: lr with default T and alpha
                for lr in lr_candidates:
                    steps.append({
                        **cfg,
                        "stage": "lr",
                        "param_name": "lr",
                        "param_val": lr,
                        "role": "student",
                        "regime": "A",
                        "method": "logit_kd",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": lr,
                        "T": default_T,
                        "alpha": default_alpha,
                        "at_beta": default_beta,
                    })
                # Stage 2: T with best lr (using default_lr as placeholder in plan)
                for T in T_candidates:
                    steps.append({
                        **cfg,
                        "stage": "T",
                        "param_name": "T",
                        "param_val": T,
                        "role": "student",
                        "regime": "A",
                        "method": "logit_kd",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": default_lr,
                        "T": T,
                        "alpha": default_alpha,
                        "at_beta": default_beta,
                    })
                # Stage 3: alpha with best lr and T (using default_T as placeholder)
                for alpha in alpha_candidates:
                    steps.append({
                        **cfg,
                        "stage": "alpha",
                        "param_name": "alpha",
                        "param_val": alpha,
                        "role": "student",
                        "regime": "A",
                        "method": "logit_kd",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": default_lr,
                        "T": default_T,
                        "alpha": alpha,
                        "at_beta": default_beta,
                    })
            elif method == "attention_transfer":
                # Total loss has no logit-KD term: CE + at_beta * AT
                # Stage 1: lr with default at_beta
                for lr in lr_candidates:
                    steps.append({
                        **cfg,
                        "stage": "lr",
                        "param_name": "lr",
                        "param_val": lr,
                        "role": "student",
                        "regime": "A",
                        "method": "attention_transfer",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": lr,
                        "T": default_T,
                        "alpha": default_alpha,
                        "at_beta": default_beta,
                    })
                # Stage 2: at_beta with best lr
                for beta in beta_candidates:
                    steps.append({
                        **cfg,
                        "stage": "at_beta",
                        "param_name": "at_beta",
                        "param_val": beta,
                        "role": "student",
                        "regime": "A",
                        "method": "attention_transfer",
                        "f_t": 1.0,
                        "f_s": f_S,
                        "seed": 0,
                        "lr": default_lr,
                        "T": default_T,
                        "alpha": default_alpha,
                        "at_beta": beta,
                    })
    return steps


# Alias for backward-compatibility with tests
build_tuning_grid = build_sequential_tuning_plan


def _load_existing_tuning_results(csv_path: Path) -> dict[tuple, dict]:
    """Return a mapping of (method, f_S, lr, T, alpha, at_beta) -> row dict."""
    done: dict[tuple, dict] = {}
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                method = row["method"]
                f_s = float(row["f_S"])
                lr = float(row["lr"])
                T = float(row["T"]) if row.get("T") else 4.0
                alpha = float(row["alpha"]) if row.get("alpha") else 0.9
                at_beta = float(row["at_beta"]) if row.get("at_beta") else 1000.0
                done[(method, f_s, lr, T, alpha, at_beta)] = row
            except (KeyError, ValueError):
                pass
    return done


def _append_tuning_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TUNING_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def select_and_save_best_hyperparams(tuning_csv_path: Path, best_config_path: Path) -> dict:
    """Parse tuning.csv, select best hyperparams by val_acc per (method, f_S), and save YAML."""
    best: dict[str, dict] = {}
    if not tuning_csv_path.exists():
        return best

    rows_by_group: dict[tuple, list[dict]] = {}
    with open(tuning_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                group_key = (row["method"], float(row["f_S"]))
                rows_by_group.setdefault(group_key, []).append(row)
            except (KeyError, ValueError):
                pass

    for (method, f_S), group_rows in rows_by_group.items():
        if method not in best:
            best[method] = {}

        has_stages = any("stage" in r and r["stage"] for r in group_rows)
        if has_stages:
            lr_rows = [r for r in group_rows if r.get("stage") == "lr"]
            best_lr_row = max(lr_rows, key=lambda r: float(r["val_acc"])) if lr_rows else max(group_rows, key=lambda r: float(r["val_acc"]))
            best_lr = float(best_lr_row["lr"])
            entry = {"lr": best_lr}

            if method == "logit_kd":
                t_rows = [r for r in group_rows if r.get("stage") == "T"]
                best_t_row = max(t_rows, key=lambda r: float(r["val_acc"])) if t_rows else best_lr_row
                best_T = float(best_t_row.get("T", 4.0))

                alpha_rows = [r for r in group_rows if r.get("stage") == "alpha"]
                best_alpha_row = max(alpha_rows, key=lambda r: float(r["val_acc"])) if alpha_rows else best_t_row
                best_alpha = float(best_alpha_row.get("alpha", 0.9))

                entry["T"] = best_T
                entry["alpha"] = best_alpha
                entry["best_val_acc"] = float(best_alpha_row["val_acc"])
            elif method == "attention_transfer":
                beta_rows = [r for r in group_rows if r.get("stage") == "at_beta"]
                best_beta_row = max(beta_rows, key=lambda r: float(r["val_acc"])) if beta_rows else best_lr_row
                best_beta = float(best_beta_row.get("at_beta", 1000.0))

                entry["at_beta"] = best_beta
                entry["best_val_acc"] = float(best_beta_row["val_acc"])
            else:
                entry["best_val_acc"] = float(best_lr_row["val_acc"])
        else:
            best_row = max(group_rows, key=lambda r: float(r["val_acc"]))
            entry = {
                "lr": float(best_row["lr"]),
                "best_val_acc": float(best_row["val_acc"]),
            }
            if "T" in best_row and best_row["T"] != "":
                entry["T"] = float(best_row["T"])
            if "alpha" in best_row and best_row["alpha"] != "":
                entry["alpha"] = float(best_row["alpha"])
            if "at_beta" in best_row and best_row["at_beta"] != "":
                entry["at_beta"] = float(best_row["at_beta"])

        best[method][f"{f_S:.2f}"] = entry

    save_yaml(best, best_config_path)
    logger.info("Saved best hyperparameters to %s", best_config_path)
    return best


def apply_tuned_hyperparams(grid: list[dict], tuned_config_path: Path) -> list[dict]:
    """Override grid hyperparams with tuned settings if available.

    For fractions that were not explicitly tuned, applies settings from the
    nearest tuned fraction (by minimum absolute difference |f_tuned - f_S|).
    """
    if not tuned_config_path.exists():
        return grid
    tuned = load_yaml_config(tuned_config_path)
    if not tuned:
        return grid

    for run in grid:
        method = run["method"]
        if method not in tuned:
            continue
        method_settings = tuned[method]
        if not method_settings:
            continue

        tuned_fractions = [float(k) for k in method_settings.keys()]
        run_fs = float(run["f_s"])
        nearest_f = min(tuned_fractions, key=lambda tf: abs(tf - run_fs))

        # Match key in method_settings
        matched_key = None
        for k in method_settings.keys():
            if abs(float(k) - nearest_f) < 1e-5:
                matched_key = k
                break
        if matched_key is None:
            continue

        settings = method_settings[matched_key]
        if "lr" in settings:
            run["lr"] = settings["lr"]
        if "T" in settings:
            run["T"] = settings["T"]
        if "alpha" in settings:
            run["alpha"] = settings["alpha"]
        if "at_beta" in settings:
            run["at_beta"] = settings["at_beta"]

    logger.info("Applied tuned hyperparameters from %s to grid.", tuned_config_path)
    return grid


# ── Teacher checkpoint resolution ─────────────────────────────────────────────

def get_or_train_teacher(
    f_t: float,
    seed: int,
    cfg: dict,
    dry_run: bool = False,
) -> Optional[Path]:
    """Return path to teacher checkpoint, training if not yet available.

    Parameters
    ----------
    f_t:
        Teacher data fraction.
    seed:
        Experiment seed.
    cfg:
        Global config dict.
    dry_run:
        If True, return the expected path without training.

    Returns
    -------
    Path to best.pth, or None in dry_run mode if it doesn't exist yet.
    """
    teacher_run_id = make_run_id(
        role="teacher",
        model=cfg["teacher_model"],
        method="ce",
        f_t=f_t,
        f_s=f_t,  # teacher's "f_s" is same as f_t for naming
        seed=seed,
    )
    ckpt_path = Path(cfg["checkpoint_dir"]) / teacher_run_id / "best.pth"

    if ckpt_path.exists():
        return ckpt_path

    if dry_run:
        return None  # Will be trained during the real run

    # Train teacher
    logger.info("Training teacher: f_T=%.2f seed=%d", f_t, seed)
    teacher_cfg = {
        **cfg,
        "role": "teacher",
        "method": "ce",
        "f_t": f_t,
        "f_s": f_t,
        "seed": seed,
        "teacher_model": cfg["teacher_model"],
    }
    train(teacher_cfg)
    return ckpt_path


# ── Single run execution ──────────────────────────────────────────────────────

def run_one(run_spec: dict, csv_path: Path, dry_run: bool = False) -> None:
    """Execute or describe one student training + evaluation run."""
    regime = run_spec["regime"]
    method = run_spec["method"]
    f_T = run_spec["f_t"]
    f_S = run_spec["f_s"]
    seed = run_spec["seed"]

    label = f"[{regime}] {method} f_T={f_T:.2f} f_S={f_S:.2f} seed={seed}"

    if dry_run:
        print(f"  PLAN  {label}")
        return

    # 1. Ensure teacher checkpoint exists
    teacher_ckpt = get_or_train_teacher(f_T, seed, run_spec, dry_run=False)
    if teacher_ckpt is None or not teacher_ckpt.exists():
        logger.error("Teacher checkpoint not found after training: %s", teacher_ckpt)
        return

    # 2. Train student
    logger.info("Training student: %s", label)
    student_cfg = {
        **run_spec,
        "role": "student",
        "teacher_ckpt": str(teacher_ckpt),
    }
    t0 = time.time()
    summary = train(student_cfg)
    wall_time = time.time() - t0

    # 3. Evaluate on test set
    student_run_id = make_run_id(
        role="student",
        model=run_spec["student_model"],
        method=method,
        f_t=f_T,
        f_s=f_S,
        seed=seed,
    )
    student_ckpt_path = Path(run_spec["checkpoint_dir"]) / student_run_id / "best.pth"
    eval_output_dir = Path(run_spec["output_dir"]) / "eval" / student_run_id

    try:
        eval_results = evaluate_checkpoint(
            student_ckpt_path=student_ckpt_path,
            data_dir=run_spec["data_dir"],
            output_dir=eval_output_dir,
            teacher_ckpt_path=teacher_ckpt,
            num_classes=run_spec.get("num_classes", 100),
            batch_size=run_spec.get("batch_size", 128),
            num_workers=run_spec.get("num_workers", 4),
            ece_bins=run_spec.get("ece_bins", 15),
        )
        test_acc = eval_results["top1_acc"]
        agreement = eval_results.get("agreement")
        ece = eval_results["ece"]
    except Exception as exc:
        logger.error("Evaluation failed for %s: %s", label, exc)
        test_acc = agreement = ece = None

    # 4. Append to CSV
    steps_per_sec = summary.get(
        "steps_per_sec",
        round(summary["total_steps"] / max(wall_time, 1e-6), 2)
    )

    _append_result_row(csv_path, {
        "regime": regime,
        "method": method,
        "f_T": f_T,
        "f_S": f_S,
        "seed": seed,
        "val_acc": round(summary["best_val_acc"], 4),
        "test_acc": round(test_acc, 4) if test_acc is not None else "",
        "agreement": round(agreement, 4) if agreement is not None else "",
        "ece": round(ece, 6) if ece is not None else "",
        "train_steps": summary["total_steps"],
        "wall_time_s": round(wall_time, 1),
        "steps_per_sec": steps_per_sec,
    })
    logger.info("Appended result to %s (throughput: %.2f steps/s)", csv_path, steps_per_sec)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the kd-lowdata experiment grid.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--max-steps-override", "--max_steps_override", type=int, default=None,
                   dest="max_steps_override", help="Cap total optimizer steps for pilot runs.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the full list of planned runs without training.")
    p.add_argument("--tune", action="store_true",
                   help="Run hyperparameter tuning over lr, T/beta, and alpha for seed 0, selecting by validation accuracy.")
    p.add_argument("--tune-fractions", "--tune_fractions", nargs="+", default=["0.01", "0.10", "1.0"],
                   dest="tune_fractions", help="Student data fractions to tune on (default: 0.01 0.10 1.0).")
    p.add_argument("--tune-max-steps", "--tune_max_steps", type=int, default=None,
                   dest="tune_max_steps", help="Cap optimizer steps for tuning runs.")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override seeds list (e.g. --seeds 0 for single-seed fast mode).")
    p.add_argument("--tuned-config", default=None,
                   help="Path to YAML file with tuned hyperparameters (defaults to <output_dir>/best_hyperparams.yaml if present).")
    return p.parse_args(argv)


def run_tuning(
    cfg: dict,
    tune_fractions: list[float] | None = None,
    tune_max_steps: int | None = None,
    dry_run: bool = False,
) -> None:
    """Execute or plan sequential hyperparameter tuning runs on seed 0."""
    if tune_fractions is None:
        tune_fractions = [0.01, 0.10, 1.0]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = output_dir / "tuning.csv"
    best_config_path = output_dir / "best_hyperparams.yaml"

    done = _load_existing_tuning_results(tuning_csv)
    plan = build_sequential_tuning_plan(cfg, tune_fractions=tune_fractions)

    print(f"\n{'='*60}")
    print(f"  KD-LowData Hyperparameter Tuning -- {'DRY RUN' if dry_run else 'EXECUTION'}")
    print(f"{'='*60}")

    if dry_run:
        pending = 0
        skipped = 0
        for step in plan:
            key = (step["method"], step["f_s"], step["lr"], step["T"], step["alpha"], step["at_beta"])
            if key in done:
                skipped += 1
                continue
            pending += 1
            if step["method"] == "ce":
                desc = f"lr={step['param_val']}"
            elif step["method"] == "logit_kd":
                desc = f"{step['param_name']}={step['param_val']} (lr={step['lr']}, T={step['T']}, alpha={step['alpha']})"
            else:  # attention_transfer
                desc = f"{step['param_name']}={step['param_val']} (lr={step['lr']}, at_beta={step['at_beta']})"
            print(f"  PLAN  [stage={step['stage']}] {step['method']} f_S={step['f_s']:.2f} {desc}")

        print(f"\n{'='*60}")
        print(f"  Total tuning candidate runs: {len(plan)}")
        print(f"  Already done (will skip): {skipped}")
        print(f"  Pending: {pending}")
        print(f"  Tuning fractions: {tune_fractions}")
        print(f"  Tuning max steps: {tune_max_steps}")
        print(f"  Tuning CSV: {tuning_csv}")
        print(f"  Best params will save to: {best_config_path}")
        print(f"{'='*60}\n")
        return

    # Real execution: sequential per (method, f_S) cell
    methods = [m for m in cfg.get("methods", ["ce", "logit_kd", "attention_transfer"])]
    lr_candidates = [0.05, 0.1, 0.2]
    T_candidates = [2.0, 4.0, 8.0]
    alpha_candidates = [0.5, 0.7, 0.9]
    beta_candidates = [100.0, 500.0, 1000.0, 5000.0]

    default_lr = cfg.get("lr", 0.1)
    default_T = cfg.get("T", cfg.get("temperature", 4.0))
    default_alpha = cfg.get("alpha", 0.9)
    default_beta = cfg.get("at_beta", 1000.0)

    def _eval_step(method, f_s, stage, param_name, param_val, lr, T, alpha, at_beta) -> float:
        key = (method, f_s, lr, T, alpha, at_beta)
        if key in done:
            cached_acc = float(done[key]["val_acc"])
            logger.info("SKIP (cached in %s): %s f_S=%.2f lr=%s T=%s alpha=%s at_beta=%s -> val_acc=%.2f%%",
                        tuning_csv, method, f_s, lr, T, alpha, at_beta, cached_acc)
            return cached_acc

        label = f"[TUNE stage={stage}] {method} f_S={f_s:.2f} {param_name}={param_val} (lr={lr}, T={T}, alpha={alpha}, at_beta={at_beta})"
        logger.info("Running: %s", label)

        teacher_ckpt = None
        if method != "ce":
            teacher_ckpt = get_or_train_teacher(1.0, 0, cfg, dry_run=False)

        student_cfg = {
            **cfg,
            "role": "student",
            "regime": "A",
            "method": method,
            "f_t": 1.0,
            "f_s": f_s,
            "seed": 0,
            "lr": lr,
            "T": T,
            "alpha": alpha,
            "at_beta": at_beta,
            "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else None,
        }
        if tune_max_steps is not None:
            student_cfg["max_steps_override"] = tune_max_steps

        t0 = time.time()
        summary = train(student_cfg)
        wall_time = time.time() - t0
        steps_per_sec = summary.get("steps_per_sec", round(summary["total_steps"] / max(wall_time, 1e-6), 2))
        val_acc = round(summary["best_val_acc"], 4)

        row = {
            "stage": stage,
            "method": method,
            "f_S": f_s,
            "param_name": param_name,
            "param_val": param_val,
            "lr": lr,
            "T": T,
            "alpha": alpha,
            "at_beta": at_beta,
            "tune_max_steps": tune_max_steps if tune_max_steps is not None else "",
            "val_acc": val_acc,
            "train_steps": summary["total_steps"],
            "wall_time_s": round(wall_time, 1),
            "steps_per_sec": steps_per_sec,
        }
        _append_tuning_row(tuning_csv, row)
        done[key] = row
        return val_acc

    best_hyperparams: dict[str, dict] = {}
    for f_S in tune_fractions:
        for method in methods:
            if method not in best_hyperparams:
                best_hyperparams[method] = {}

            if method == "ce":
                # Stage 1: tune lr
                lr_accs = [
                    (lr, _eval_step("ce", f_S, "lr", "lr", lr, lr, default_T, default_alpha, default_beta))
                    for lr in lr_candidates
                ]
                best_lr, best_acc = max(lr_accs, key=lambda pair: pair[1])
                best_hyperparams["ce"][f"{f_S:.2f}"] = {
                    "lr": best_lr,
                    "best_val_acc": best_acc,
                }
            elif method == "logit_kd":
                # Stage 1: tune lr with default T and alpha
                lr_accs = [
                    (lr, _eval_step("logit_kd", f_S, "lr", "lr", lr, lr, default_T, default_alpha, default_beta))
                    for lr in lr_candidates
                ]
                best_lr, _ = max(lr_accs, key=lambda pair: pair[1])

                # Stage 2: tune T with best lr and default alpha
                t_accs = [
                    (T, _eval_step("logit_kd", f_S, "T", "T", T, best_lr, T, default_alpha, default_beta))
                    for T in T_candidates
                ]
                best_T, _ = max(t_accs, key=lambda pair: pair[1])

                # Stage 3: tune alpha with best lr and best T
                alpha_accs = [
                    (alpha, _eval_step("logit_kd", f_S, "alpha", "alpha", alpha, best_lr, best_T, alpha, default_beta))
                    for alpha in alpha_candidates
                ]
                best_alpha, best_acc = max(alpha_accs, key=lambda pair: pair[1])

                best_hyperparams["logit_kd"][f"{f_S:.2f}"] = {
                    "lr": best_lr,
                    "T": best_T,
                    "alpha": best_alpha,
                    "best_val_acc": best_acc,
                }
            elif method == "attention_transfer":
                # Total loss = CE + at_beta * AT (no logit-KD term)
                # Stage 1: tune lr with default at_beta
                lr_accs = [
                    (lr, _eval_step("attention_transfer", f_S, "lr", "lr", lr, lr, default_T, default_alpha, default_beta))
                    for lr in lr_candidates
                ]
                best_lr, _ = max(lr_accs, key=lambda pair: pair[1])

                # Stage 2: tune at_beta with best lr
                beta_accs = [
                    (beta, _eval_step("attention_transfer", f_S, "at_beta", "at_beta", beta, best_lr, default_T, default_alpha, beta))
                    for beta in beta_candidates
                ]
                best_beta, best_acc = max(beta_accs, key=lambda pair: pair[1])

                best_hyperparams["attention_transfer"][f"{f_S:.2f}"] = {
                    "lr": best_lr,
                    "at_beta": best_beta,
                    "best_val_acc": best_acc,
                }

    save_yaml(best_hyperparams, best_config_path)
    logger.info("Saved best hyperparameters to %s", best_config_path)


def main(argv=None) -> None:
    setup_logging()
    args = parse_args(argv)

    cfg = load_yaml_config(args.config)

    # Apply CLI path overrides
    if args.data_dir:
        cfg["data_dir"] = args.data_dir
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.checkpoint_dir:
        cfg["checkpoint_dir"] = args.checkpoint_dir
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers
    if args.max_steps_override is not None:
        cfg["max_steps_override"] = args.max_steps_override
    if args.seeds is not None:
        cfg["seeds"] = args.seeds

    # If --tune requested, run tuning pipeline instead
    if args.tune:
        tune_fractions = _parse_fractions(args.tune_fractions)
        run_tuning(
            cfg,
            tune_fractions=tune_fractions,
            tune_max_steps=args.tune_max_steps,
            dry_run=args.dry_run,
        )
        return

    csv_path = Path(cfg["output_dir"]) / "results.csv"
    grid = build_grid(cfg)

    # Check for tuned config file to apply
    tuned_config_path = (
        Path(args.tuned_config)
        if args.tuned_config
        else Path(cfg["output_dir"]) / "best_hyperparams.yaml"
    )
    if tuned_config_path.exists():
        grid = apply_tuned_hyperparams(grid, tuned_config_path)

    if args.dry_run:
        print(f"\n{'='*60}")
        print(f"  KD-LowData Experiment Grid -- DRY RUN")
        print(f"{'='*60}")
        # Show teacher runs
        seen_teachers: set[tuple] = set()
        for run in grid:
            key = (run["f_t"], run["seed"])
            if key not in seen_teachers:
                teacher_run_id = make_run_id(
                    role="teacher", model=cfg["teacher_model"], method="ce",
                    f_t=run["f_t"], f_s=run["f_t"], seed=run["seed"]
                )
                ckpt = Path(cfg["checkpoint_dir"]) / teacher_run_id / "best.pth"
                status = "EXISTS" if ckpt.exists() else "PENDING"
                print(f"  TEACHER [{status}] f_T={run['f_t']:.2f} seed={run['seed']}")
                seen_teachers.add(key)

        print()
        done = _load_existing_results(csv_path)
        pending = 0
        skipped = 0
        for run in grid:
            key = (run["regime"], run["method"], run["f_t"], run["f_s"], run["seed"])
            if key in done:
                skipped += 1
                continue
            run_one(run, csv_path, dry_run=True)
            pending += 1

        print(f"\n{'='*60}")
        print(f"  Total student runs: {len(grid)}")
        print(f"  Already done (will skip): {skipped}")
        print(f"  Pending: {pending}")
        print(f"  Unique teacher checkpoints: {len(seen_teachers)}")
        print(f"  Results CSV: {csv_path}")
        print(f"{'='*60}\n")
        return

    # Real run
    done = _load_existing_results(csv_path)
    total = len(grid)
    skipped_count = 0

    for i, run in enumerate(grid, 1):
        key = (run["regime"], run["method"], run["f_t"], run["f_s"], run["seed"])
        if key in done:
            logger.info("SKIP (%d/%d): already in results.csv", i, total)
            skipped_count += 1
            continue
        logger.info("RUN (%d/%d): regime=%s method=%s f_T=%.2f f_S=%.2f seed=%d",
                    i, total, run["regime"], run["method"],
                    run["f_t"], run["f_s"], run["seed"])
        run_one(run, csv_path, dry_run=False)

    logger.info(
        "Grid complete. %d/%d runs executed, %d skipped.",
        total - skipped_count, total, skipped_count,
    )


if __name__ == "__main__":
    main()
