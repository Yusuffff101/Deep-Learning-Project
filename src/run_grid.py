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
from utils import load_yaml_config, make_run_id, setup_logging

logger = logging.getLogger(__name__)

# ── CSV schema ────────────────────────────────────────────────────────────────
CSV_COLUMNS = [
    "regime", "method", "f_T", "f_S", "seed",
    "val_acc", "test_acc", "agreement", "ece",
    "train_steps", "wall_time_s",
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
                key = (
                    row["regime"],
                    row["method"],
                    float(row["f_T"]),
                    float(row["f_S"]),
                    int(row["seed"]),
                )
                done.add(key)
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
            # In regime B, f_T = f_S; only include teacher_fractions values
            if f_S not in teacher_fractions:
                continue  # Skip f_S values that don't exist as f_T options
            runs.append({
                **cfg,
                "regime": "B",
                "method": method,
                "f_t": f_S,   # f_T == f_S
                "f_s": f_S,
                "seed": seed,
            })

    return runs


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
    })
    logger.info("Appended result to %s", csv_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the kd-lowdata experiment grid.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--checkpoint_dir", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--dry-run", action="store_true",
                   help="Print the full list of planned runs without training.")
    return p.parse_args(argv)


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

    csv_path = Path(cfg["output_dir"]) / "results.csv"
    grid = build_grid(cfg)

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
