# KD-LowData

> **Research question:** How does knowledge distillation (KD) benefit change
> as labeled data shrinks, and does it matter whether the teacher was trained
> on full data or on the same small subset as the student?

---

## Project Goal

Study KD on CIFAR-100 under data scarcity.

**Two experimental regimes:**

### Data Splits & Training Pool Definition
- **CIFAR-100 Train Split**: 50,000 images total.
- **Fixed Validation Split**: A fixed, class-stratified split of **5,000 images** (50 per class) is carved out using seed `42` (`VAL_SEED`) and held out for validation across all experiments.
- **Training Pool ("100%")**: The remaining **45,000 images** (450 per class) form the training pool.
- **Fractions are strictly of the 45k pool**: All percentages/fractions ($f_S, f_T$) denote exact subsets of this 45,000-image pool:
  - **1%** = 450 images (4–5 images per class)
  - **5%** = 2,250 images (22–23 images per class)
  - **10%** = 4,500 images (45 images per class)
  - **25%** = 11,250 images (~112 images per class)
  - **100%** = 45,000 images (450 images per class)
- The 5,000 validation images and 10,000 test images are **never** included in any training subset.

---

## Installation

```bash
# Clone and enter the project
git clone <your-repo-url>
cd kd-lowdata

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Smoke Tests

Verifies data pipeline, losses, and 2-step training on CPU (no GPU required).
CIFAR-100 will be downloaded automatically (~170 MB) on first run.

```bash
cd kd-lowdata
pytest tests/test_smoke.py -v
```

Expected output: all tests pass in ~3–5 minutes on CPU.

---

## Running a Single Pilot Experiment

```bash
# Train the teacher (ResNet-56, 100% data, seed 0):
python src/train.py --role teacher --f_t 1.0 --seed 0

# Train a student (ResNet-20, logit KD, 5% data, seed 0):
python src/train.py \
    --role student \
    --method logit_kd \
    --f_s 0.05 \
    --f_t 1.0 \
    --seed 0 \
    --teacher_ckpt checkpoints/teacher_resnet56_ce_fT1.00_fS1.00_s0/best.pth

# Evaluate:
python src/evaluate.py \
    --student_ckpt checkpoints/student_resnet20_logit_kd_fT1.00_fS0.05_s0/best.pth \
    --teacher_ckpt checkpoints/teacher_resnet56_ce_fT1.00_fS1.00_s0/best.pth \
    --output_dir results/eval/pilot
```

---

## Running the Full Grid

```bash
# Preview what will run (no training, 72 student runs + 12 teachers):
python src/run_grid.py --dry-run

# Run pilot with step capping:
python src/run_grid.py --dry-run --max-steps-override 5

# Preview sequential tuning runs (57 candidate runs):
python src/run_grid.py --tune --dry-run

# Run sequential hyperparameter tuning on seed 0:
python src/run_grid.py --tune --tune-fractions 0.01 0.10 1.0 --tune-max-steps 5000

# Run all experiments (safely resumable, applying tuned hyperparams):
python src/run_grid.py --config configs/default.yaml
```

The grid skips any configuration whose result row already exists in
`results/results.csv`, so interrupted Kaggle sessions can be resumed safely.
Regime A with $f_S=1.0$ and Regime B with $f_T=f_S=1.0$ are deduplicated and executed only once.
Each run prints a timing summary with step throughput (`steps/s`) to easily extrapolate GPU runtimes.

---

## Sequential Hyperparameter Tuning

Tuning runs on seed 0 across `(method, f_S)` cells to select optimal parameters before launching the full grid:

1. **Stage Sequence per Cell**:
   - **`ce`**: Tunes `lr` over `[0.05, 0.1, 0.2]` (3 runs).
   - **`logit_kd`**:
     - *Stage 1*: Tune `lr` (`[0.05, 0.1, 0.2]`) with default $T=4.0, \alpha=0.9$ (3 runs).
     - *Stage 2*: Tune $T$ (`[2.0, 4.0, 8.0]`) with best `lr` and default $\alpha=0.9$ (3 runs).
     - *Stage 3*: Tune $\alpha$ (`[0.5, 0.7, 0.9]`) with best `lr` and best $T$ (3 runs).
     - Total: 9 runs per fraction.
   - **`attention_transfer`**:
     - Loss formulation has **no logit-KD term**: $\mathcal{L}_{total} = \mathcal{L}_{CE} + \beta \mathcal{L}_{AT}$.
     - *Stage 1*: Tune `lr` (`[0.05, 0.1, 0.2]`) with default $\beta=1000.0$ (3 runs).
     - *Stage 2*: Tune AT loss weight $\beta$ (`[100.0, 500.0, 1000.0, 5000.0]`) with best `lr` (4 runs).
     - Total: 7 runs per fraction.
   - Total per fraction: $3 + 9 + 7 = 19$ runs. Across 3 fractions = **57 candidate runs**.

2. **`--tune-fractions` Flag** (default: `0.01 0.10 1.0`):
   - Tuning runs only on these designated fractions to conserve compute.
   - When the main grid loads `best_hyperparams.yaml`, hyperparameter settings from the **nearest tuned fraction** (by minimum absolute difference $|f_{tuned} - f_S|$) are automatically interpolated for untuned fractions:
     - $f_S = 0.01 \to 0.01$ (exact)
     - $f_S = 0.05 \to 0.01$ ($|0.05 - 0.01| = 0.04 < |0.05 - 0.10| = 0.05$)
     - $f_S = 0.10 \to 0.10$ (exact)
     - $f_S = 0.25 \to 0.10$ ($|0.25 - 0.10| = 0.15 < |0.25 - 1.00| = 0.75$)
     - $f_S = 1.00 \to 1.00$ (exact)

3. **`--tune-max-steps` Flag**:
   - Reduces the step budget for faster turnaround during tuning.
   - The step budget used is logged under the `tune_max_steps` column in `results/tuning.csv`.
   - Results are saved to `results/best_hyperparams.yaml` and read automatically by the main grid.

---

## Using on Kaggle

```python
# In a Kaggle notebook cell:
!git clone <your-repo-url> kd-lowdata
%cd kd-lowdata
!pip install -r requirements.txt -q

# Run with Kaggle paths:
!python src/run_grid.py \
    --data_dir /kaggle/input \
    --output_dir /kaggle/working/results \
    --checkpoint_dir /kaggle/working/checkpoints \
    --num_workers 2
```

CIFAR-100 dataset can be added via Kaggle's dataset interface and pointed to
via `--data_dir`. The code auto-detects CUDA and uses mixed precision on T4/P100.

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `epochs` | 200 | Baseline; may be tuned per cell on val set |
| `min_steps` | 30000 | Prevents under-training on tiny subsets |
| `lr` | 0.1 | SGD with cosine annealing |
| `temperature` | 4.0 | KD softening temperature |
| `alpha` | 0.9 | KL weight in logit KD |
| `at_beta` | 1000.0 | AT loss weight |

Override any parameter via CLI flags (see `python src/train.py --help`).

---

## Project Structure

```
kd-lowdata/
├── configs/
│   └── default.yaml        # All hyperparameters
├── src/
│   ├── data.py             # CIFAR-100 loading, nested subset construction
│   ├── models.py           # ResNet-20, ResNet-56 (CIFAR-style)
│   ├── losses.py           # CE, logit KD, attention transfer + registry
│   ├── train.py            # Teacher/student training loop
│   ├── evaluate.py         # Test-set evaluation (acc, ECE, agreement)
│   ├── run_grid.py         # Grid runner (resumable, Regime A+B)
│   └── utils.py            # Seeds, checkpointing, meters, logging
├── tests/
│   └── test_smoke.py       # 11 smoke tests
├── notebooks/
│   └── analysis.ipynb      # Placeholder for results analysis
├── results/                # CSV + eval outputs (gitignored except .gitkeep)
├── checkpoints/            # Model checkpoints (gitignored except .gitkeep)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Extending with New KD Methods

To add DKD, CRD, or any future method:

1. Write a loss function in `src/losses.py` with the signature:
   ```python
   def dkd_loss(student_logits, teacher_logits, labels,
                student_features, teacher_features, **kwargs) -> Tensor:
       ...
   ```
2. Register it: `LOSS_REGISTRY["dkd"] = dkd_loss`
3. Add it to `methods:` in `configs/default.yaml`

That's it — no changes needed elsewhere.

---

## Results

*To be filled in after experiments are run. Placeholder for tables and plots.*

| Regime | Method | f_S | Val Acc | Test Acc | Agreement | ECE |
|--------|--------|-----|---------|----------|-----------|-----|
| — | — | — | — | — | — | — |

---

## Citation / References

- He et al. (2016). Deep Residual Learning for Image Recognition. CVPR.
- Hinton et al. (2015). Distilling the Knowledge in a Neural Network. NeurIPS Workshop.
- Zagoruyko & Komodakis (2017). Paying More Attention to Attention. ICLR.

---

## Push to GitHub

After running the experiments locally, push to GitHub with:

```bash
# Create the remote (do this on github.com first, then):
git remote add origin https://github.com/<your-username>/kd-lowdata.git
git branch -M main
git push -u origin main
```
