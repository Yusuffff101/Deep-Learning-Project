# KD-LowData

> **Research question:** How does knowledge distillation (KD) benefit change
> as labeled data shrinks, and does it matter whether the teacher was trained
> on full data or on the same small subset as the student?

---

## Project Goal

Study KD on CIFAR-100 under data scarcity.

**Two experimental regimes:**

| Regime | Teacher data | Student data |
|--------|-------------|-------------|
| **A** | 100% (f_T = 1.0) | f_S ∈ {1%, 5%, 10%, 25%, 100%} |
| **B** | Same fraction as student (f_T = f_S) | f_S ∈ {5%, 10%, 25%} |

**Methods compared:**
- `ce` — student trained from scratch (cross-entropy only, baseline)
- `logit_kd` — Hinton et al. 2015 (softened logit matching)
- `attention_transfer` — Zagoruyko & Komodakis 2017 (spatial attention maps)

**Models:** ResNet-56 (teacher) → ResNet-20 (student), CIFAR-style.

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
# Preview what will run (no training):
python src/run_grid.py --dry-run

# Run all experiments (safely resumable):
python src/run_grid.py --config configs/default.yaml
```

The grid skips any configuration whose result row already exists in
`results/results.csv`, so interrupted Kaggle sessions can be resumed safely.

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
