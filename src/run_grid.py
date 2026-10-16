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
    "method", "f_S", "lr", "T", "alpha",
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
            })

    return runs


# ── Hyperparameter tuning helpers ─────────────────────────────────────────────

def build_tuning_grid(cfg: dict) -> list[dict]:
    """Build the hyperparameter tuning grid for seed 0."""
    runs = []
    student_fractions = cfg["student_fractions"]
    methods = cfg["methods"]

    # 3 lr values
    lr_candidates = [0.05, 0.1, 0.2]

    for method in methods:
        is_kd = method in ("logit_kd", "attention_transfer")
        T_candidates = [2.0, 4.0, 8.0] if is_kd else [cfg.get("temperature", 4.0)]
        alpha_candidates = [0.5, 0.9] if is_kd else [cfg.get("alpha", 0.9)]

        for f_S in student_fractions:
            for lr, T, alpha in product(lr_candidates, T_candidates, alpha_candidates):
                runs.append({
                    **cfg,
                    "role": "student",
                    "regime": "A",
                    "method": method,
                    "f_t": 1.0,
                    "f_s": f_S,
                    "seed": 0,
                    "lr": lr,
                    "T": T,
                    "alpha": alpha,
                })
    return runs


def _load_existing_tuning_results(csv_path: Path) -> set[tuple]:
    """Return set of (method, f_S, lr, T, alpha) already recorded."""
    done = set()
    if not csv_path.exists():
        return done
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                key = (
                    row["method"],
                    float(row["f_S"]),
                    float(row["lr"]),
                    float(row["T"]),
                    float(row["alpha"]),
                )
                done.add(key)
            except (KeyError, ValueError):
                pass
    return done


def _append_tuning_row(csv_path: Path, row: dict) -> None:
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TUNING_CSV_COLUMNS)
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
        best_row = max(group_rows, key=lambda r: float(r["val_acc"]))
        if method not in best:
            best[method] = {}
        best[method][f"{f_S:.2f}"] = {
            "lr": float(best_row["lr"]),
            "T": float(best_row["T"]),
            "alpha": float(best_row["alpha"]),
            "best_val_acc": float(best_row["val_acc"]),
        }

    save_yaml(best, best_config_path)
    logger.info("Saved best hyperparameters to %s", best_config_path)
    return best


def apply_tuned_hyperparams(grid: list[dict], tuned_config_path: Path) -> list[dict]:
    """Override grid hyperparams with tuned settings if available."""
    if not tuned_config_path.exists():
        return grid
    tuned = load_yaml_config(tuned_config_path)
    if not tuned:
        return grid

    for run in grid:
        method = run["method"]
        f_s_key = f"{run['f_s']:.2f}"
        if method in tuned and f_s_key in tuned[method]:
            settings = tuned[method][f_s_key]
            if "lr" in settings:
                run["lr"] = settings["lr"]
            if "T" in settings:
                run["T"] = settings["T"]
            if "alpha" in settings:
                run["alpha"] = settings["alpha"]
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
                   help="Run hyperparameter tuning over lr, T, and alpha for seed 0, selecting by validation accuracy.")
    p.add_argument("--tuned-config", default=None,
                   help="Path to YAML file with tuned hyperparameters (defaults to <output_dir>/best_hyperparams.yaml if present).")
    return p.parse_args(argv)


def run_tuning(cfg: dict, dry_run: bool = False) -> None:
    """Execute or plan hyperparameter tuning runs on seed 0."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tuning_csv = output_dir / "tuning.csv"
    best_config_path = output_dir / "best_hyperparams.yaml"

    tuning_grid = build_tuning_grid(cfg)
    done = _load_existing_tuning_results(tuning_csv)

    print(f"\n{'='*60}")
    print(f"  KD-LowData Hyperparameter Tuning — {'DRY RUN' if dry_run else 'EXECUTION'}")
    print(f"{'='*60}")

    pending = 0
    skipped = 0

    for run in tuning_grid:
        key = (run["method"], run["f_s"], run["lr"], run["T"], run["alpha"])
        if key in done:
            skipped += 1
            continue

        pending += 1
        label = (
            f"TUNE {run['method']} f_S={run['f_s']:.2f} lr={run['lr']} "
            f"T={run['T']} alpha={run['alpha']}"
        )
        if dry_run:
            print(f"  PLAN  {label}")
            continue

        logger.info("Running: %s", label)
        # Ensure teacher checkpoint exists (f_T=1.0, seed=0)
        teacher_ckpt = None
        if run["method"] != "ce":
            teacher_ckpt = get_or_train_teacher(1.0, 0, run, dry_run=False)

        student_cfg = {
            **run,
            "role": "student",
            "teacher_ckpt": str(teacher_ckpt) if teacher_ckpt else None,
        }
        t0 = time.time()
        summary = train(student_cfg)
        wall_time = time.time() - t0
        steps_per_sec = summary.get("steps_per_sec", round(summary["total_steps"] / max(wall_time, 1e-6), 2))

        _append_tuning_row(tuning_csv, {
            "method": run["method"],
            "f_S": run["f_s"],
            "lr": run["lr"],
            "T": run["T"],
            "alpha": run["alpha"],
            "val_acc": round(summary["best_val_acc"], 4),
            "train_steps": summary["total_steps"],
            "wall_time_s": round(wall_time, 1),
            "steps_per_sec": steps_per_sec,
        })

    print(f"\n{'='*60}")
    print(f"  Total tuning candidate runs: {len(tuning_grid)}")
    print(f"  Already done (will skip): {skipped}")
    print(f"  Pending: {pending}")
    print(f"  Tuning CSV: {tuning_csv}")
    print(f"  Best params will save to: {best_config_path}")
    print(f"{'='*60}\n")

    if not dry_run and tuning_csv.exists():
        select_and_save_best_hyperparams(tuning_csv, best_config_path)


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

    # If --tune requested, run tuning pipeline instead
    if args.tune:
        run_tuning(cfg, dry_run=args.dry_run)
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
        print(f"  KD-LowData Experiment Grid — DRY RUN")
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
