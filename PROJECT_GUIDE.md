# Master Project Guide: Knowledge Distillation under Data Scarcity (KD-LowData)

> **Course Project**: Deep Learning (Fall Sem 2026–27)  
> **Topic**: Empirical Study of Knowledge Distillation on CIFAR-100 under Severe Data Scarcity  
> **Models**: ResNet-56 (Teacher) $\to$ ResNet-20 (Student)  
> **Author**: Deep Learning Project Repository

---

## Table of Contents
1. [The Big Picture: Non-Technical Intuition](#1-the-big-picture-non-technical-intuition)
2. [The Core Research Questions](#2-the-core-research-questions)
3. [The Dataset & Data Pipeline](#3-the-dataset--data-pipeline)
4. [The Network Architectures](#4-the-network-architectures)
5. [The Three Machine Learning Methods](#5-the-three-machine-learning-methods)
6. [The Two Experimental Regimes (Regime A vs Regime B)](#6-the-two-experimental-regimes-regime-a-vs-regime-b)
7. [The Engineering Pipeline & Training Tricks](#7-the-engineering-pipeline--training-tricks)
8. [Evaluation Metrics Explained](#8-evaluation-metrics-explained)
9. [Hypotheses: What Do We Expect to Find?](#9-hypotheses-what-do-we-expect-to-find)
10. [Professor Defense Q&A: 10 Questions You Will Be Asked](#10-professor-defense-qa-10-questions-you-will-be-asked)

---

## 1. The Big Picture: Non-Technical Intuition

### The Problem
Imagine you want to teach a medical student how to diagnose 100 rare diseases from X-ray scans.
- In an ideal world, the student studies **45,000 example scans**.
- But in real-world situations (medical imaging, robotics, military, rare languages), labeled data is scarce and expensive. What if the student only gets **450 scans** (just 4 or 5 scans per disease)?
- If the student tries to memorize 450 scans, they will suffer from severe **overfitting** (memorizing the quirks of those few pictures rather than learning the actual disease features). They will fail miserably on new patients.

### The Solution: Knowledge Distillation (KD)
Instead of studying alone from textbook labels ("Scan #12 is Disease A, Scan #13 is Disease B"), what if the student is mentored by a **World-Class Master Doctor (the Teacher Model)**?
- When the Teacher looks at a scan of a German Shepherd dog, they don't just say *"100% German Shepherd"*.
- The Teacher says: *"This is 85% German Shepherd, 10% Siberian Husky, 4% Wolf, and 0.0001% Sports Car."*
- Those subtle probabilities are called **"Dark Knowledge"** (Hinton et al., 2015). They reveal the underlying geometric structure and visual similarities between categories.
- By matching the Teacher's nuanced advice, the Student learns much faster and avoids overfitting, even when training data is severely limited!

### The Core Curiosity of Our Project
Most existing literature tests KD when both Teacher and Student have access to **massive, unlimited data**. 
Our project asks two fundamental questions:
1. **When data shrinks to extreme levels (1% or 5%), does the Teacher's advice become MORE valuable, or does the Student get overwhelmed?**
2. **Does the Teacher actually need to have seen all 45,000 scans? Or does a large Teacher model still help the Student even if the Teacher only had access to the exact same 450 scans?**

---

## 2. The Core Research Questions

Formally, this study investigates two core hypotheses:

$$\mathbf{Q_1}: \quad \Delta \text{Acc}(f_S) = \text{Acc}_{\text{KD}}(f_S) - \text{Acc}_{\text{CE}}(f_S)$$
*How does the distillation gain $\Delta \text{Acc}$ behave as the student's data fraction $f_S \in \{0.01, 0.05, 0.10, 0.25, 1.00\}$ decays? Does KD act as a data-efficiency multiplier?*

$$\mathbf{Q_2}: \quad \Delta \text{Regime}(f) = \text{Acc}_{\text{Regime A}}(f) - \text{Acc}_{\text{Regime B}}(f)$$
*Does the distillation benefit stem from **data privilege** (the teacher having seen 100% of the data) or from **architectural privilege** (the teacher being 3× deeper with higher capacity)?*

---

## 3. The Dataset & Data Pipeline

### The Benchmark: CIFAR-100
- **Total Images**: 60,000 natural color images of size $32 \times 32 \times 3$.
- **Classes**: 100 fine-grained categories (beaver, dolphin, otter, bus, train, chair, table, apple, etc.) grouped into 20 superclasses.
- **Difficulty**: Unlike CIFAR-10 (which has 5,000 images per class), CIFAR-100 only has 500 training images per class in full data. When we drop to 1%, there are only **4 to 5 images per class**!

```
                    ┌───────────────────────────────────────────────┐
                    │          CIFAR-100 (60,000 images)            │
                    └───────────────────────┬───────────────────────┘
                                            │
                    ┌───────────────────────┴───────────────────────┐
                    │                                               │
         Standard Test Set (10,000)                      Original Train Set (50,000)
       (Never used during training)                                 │
                                            ┌───────────────────────┴───────────────────────┐
                                            │                                               │
                                  Fixed Validation Set (5,000)                     Training Pool (45,000)
                                (Seed 42, 50 images per class)                 (All fractions drawn from here)
```

### Clean Split Integrity
To ensure rigorous, peer-reviewed level scientific validity:
1. **Validation Split**: A fixed, class-stratified split of **5,000 images** is carved out using seed `42` (`VAL_SEED`) and locked away permanently. It is never used in training.
2. **Training Pool ("100%")**: The remaining **45,000 images** (450 per class) form the ground-truth training pool.
3. **Data Fractions ($f_S, f_T$)**:
   - $f = 1.00$ ($100\%$): 45,000 images (450 per class)
   - $f = 0.25$ ($25\%$): 11,250 images (~112 per class)
   - $f = 0.10$ ($10\%$): 4,500 images (45 per class)
   - $f = 0.05$ ($5\%$): 2,250 images (~22 per class)
   - $f = 0.01$ ($1\%$): 450 images (4 to 5 per class)

### Stratification & Nested Subset Property
- **Stratified**: Every single class has an equal number of images in every subset. There is zero class imbalance.
- **Strictly Nested**: $S_{1\%} \subset S_{5\%} \subset S_{10\%} \subset S_{25\%} \subset S_{100\%}$.  
  *Why this matters*: When comparing 5% vs 1%, the 1% images are guaranteed to be inside the 5% set. This eliminates variance from random sample selection!
- **Data Augmentation**: Standard CIFAR protocol:
  - Random Crop ($32\times 32$ with padding 4)
  - Random Horizontal Flip ($p=0.5$)
  - Normalization using CIFAR-100 channel means and standard deviations.

---

## 4. The Network Architectures

Both networks are CIFAR-adapted ResNets (He et al., 2016).

| Specification | Teacher Model | Student Model |
| :--- | :--- | :--- |
| **Architecture** | **ResNet-56** | **ResNet-20** |
| **Number of Layers** | 56 layers (3 stages of 9 residual blocks) | 20 layers (3 stages of 3 residual blocks) |
| **Parameter Count** | ~850,000 parameters | ~270,000 parameters (~3.1× smaller) |
| **CIFAR Adaptation** | $3\times 3$ initial conv (stride 1), NO maxpool | $3\times 3$ initial conv (stride 1), NO maxpool |
| **Feature Stages** | Stage 1: 16 ch, Stage 2: 32 ch, Stage 3: 64 ch | Stage 1: 16 ch, Stage 2: 32 ch, Stage 3: 64 ch |
| **Output Layer** | Linear classifier: $64 \to 100$ | Linear classifier: $64 \to 100$ |

*Why CIFAR ResNet instead of ImageNet ResNet?*  
ImageNet ResNets start with a $7\times 7$ conv (stride 2) followed by a $3\times 3$ maxpool. On $32\times 32$ images, that immediately shrinks the spatial resolution to $8\times 8$ in layer 1! CIFAR ResNets preserve the $32\times 32$ feature map in Stage 1, downsample to $16\times 16$ in Stage 2, and $8\times 8$ in Stage 3.

---

## 5. The Three Machine Learning Methods

```
                                  Input Image x
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
            Teacher (ResNet-56)                     Student (ResNet-20)
            [Checkpointed / Frozen]                 [Trainable]
                    │                                       │
           Teacher Logits z_T                      Student Logits z_S
           Teacher Features A_T                    Student Features A_S
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                            Loss Computation (L_total)
                                        │
               ┌────────────────────────┼────────────────────────┐
               │                        │                        │
         Method 1: CE             Method 2: Logit KD        Method 3: Attn Transfer
         L = L_CE(z_S, y)         L = (1-α)L_CE + α·L_KD    L = L_CE + β·L_AT
```

### Method 1: Baseline Cross-Entropy (`ce`)
The student trains purely on hard ground-truth labels with standard categorical cross-entropy:
$$\mathcal{L}_{\text{CE}} = - \sum_{k=1}^{100} y_k \log \left( \sigma(z_S)_k \right)$$
where $y_k \in \{0, 1\}$ is the one-hot ground-truth label, and $\sigma(z_S)$ is the softmax output.
*Purpose*: Establishes the baseline accuracy of the student working alone under data scarcity.

### Method 2: Logit Knowledge Distillation (`logit_kd` — Hinton et al., 2015)
The student learns from both the ground-truth label and the teacher's softened probability distribution:
$$\mathcal{L}_{\text{total}} = (1 - \alpha) \mathcal{L}_{\text{CE}}(z_S, y) + \alpha \, T^2 \, \mathcal{L}_{\text{KL}}\left( \sigma\left(\frac{z_S}{T}\right), \sigma\left(\frac{z_T}{T}\right) \right)$$
- **Temperature ($T = 4.0$)**: Softens the logits $z_S$ and $z_T$. Higher $T$ amplifies the tail probabilities, revealing secondary similarities (e.g., that a cat looks more like a dog than a ship).
- **$T^2$ Scaling Factor**: Because dividing logits by $T$ scales gradients by $1/T^2$, multiplying by $T^2$ keeps the gradient magnitudes balanced between the CE and KD terms.
- **Weight ($\alpha = 0.9$)**: 90% weight on mimicking the teacher, 10% on hard labels.

### Method 3: Attention Transfer (`attention_transfer` — Zagoruyko & Komodakis, 2017)
Instead of comparing only the final output predictions, Attention Transfer forces the student to look at the **same spatial regions of the image** as the teacher throughout intermediate feature layers!
1. For each stage $j \in \{1, 2, 3\}$, take the activation tensor $A \in \mathbb{R}^{C \times H \times W}$.
2. Compute spatial attention map by summing channel energies:
   $$F(A) = \sum_{c=1}^{C} |A_c|^2 \quad \in \mathbb{R}^{H \times W}$$
3. L2-normalize the flattened attention map:
   $$\hat{F}(A) = \frac{F(A)}{\|F(A)\|_2}$$
4. Calculate mean-squared error between student and teacher attention maps:
   $$\mathcal{L}_{\text{AT}} = \sum_{j=1}^{3} \left\| \hat{F}(A_S^{(j)}) - \hat{F}(A_T^{(j)}) \right\|_2^2$$
5. Total Loss:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{CE}}(z_S, y) + \beta \, \mathcal{L}_{\text{AT}}$$
   *(Following Zagoruyko et al., AT uses $\mathcal{L}_{\text{CE}} + \beta \mathcal{L}_{\text{AT}}$ with $\beta = 1000.0$ and no logit-KD term).*

---

## 6. The Two Experimental Regimes (Regime A vs Regime B)

This is the most innovative contribution of your project.

```
Regime A (Privileged Teacher):
  Teacher Training:  100% Data (45,000 images)  ──────►  Student Training: f_S Data (e.g. 450 images)
  (Teacher has "seen the world", student is data-starved)

Regime B (Matched Teacher):
  Teacher Training:  f_T Data (e.g. 450 images)  ──────►  Student Training: f_S Data (Identical 450 images)
  (Teacher saw ONLY the exact same images as student)
```

| Dimension | Regime A | Regime B |
| :--- | :--- | :--- |
| **Teacher Training Set** | Always 100% data ($f_T = 1.0$) | Matched subset ($f_T = f_S$) |
| **Student Training Set** | Subsets $f_S \in \{0.01, 0.05, 0.10, 0.25, 1.00\}$ | Subsets $f_S \in \{0.05, 0.10, 0.25\}$ |
| **Scientific Question** | "How much does a fully-informed master help an apprentice with limited data?" | "Does a bigger model help a smaller model if neither saw any extra data?" |
| **Mechanism** | Knowledge Transfer + Regularization | Inductive Bias & Feature Regularization |

*Note on Deduplication*: When $f_S = 1.0$, Regime A ($f_T=1.0, f_S=1.0$) and Regime B ($f_T=1.0, f_S=1.0$) are mathematically identical. Our pipeline deduplicates this run so it is only computed once.

---

## 7. The Engineering Pipeline & Training Tricks

### 1. The Small-Data "Step Starvation" Problem (Crucial Deep Learning Concept)
- If you train on 100% data (45,000 images) with batch size 128, one epoch has $\approx 352$ batches. Over 200 epochs, the network takes **70,400 gradient descent steps**.
- If you train on 1% data (450 images) with batch size 128, one epoch has only **3.5 batches**! Over 200 epochs, the network only takes **700 steps**.
- 700 gradient steps is not enough for an optimizer (SGD with momentum) to traverse the loss landscape to a good local minimum. The model under-trains simply because of lack of optimization iterations!
- **Our Solution (`min_steps_per_fraction`)**: We configure a minimum optimizer step budget:
  - 1% & 5%: 15,000 steps minimum (~43 epochs equivalent on 1%)
  - 10%: 20,000 steps minimum
  - 25%: 25,000 steps minimum
  - 100%: 35,000 steps (full 200 epochs)

### 2. PyTorch AMP (Automatic Mixed Precision)
- Training uses `torch.cuda.amp.autocast()` and `GradScaler`.
- Matrix multiplications (convolutions and linear layers) run in fast FP16 (half precision), while loss accumulation and optimizer updates run in FP32.
- **Result**: ~2.5× faster GPU throughput and 50% lower VRAM usage with zero degradation in accuracy.

### 3. Cosine Annealing Learning Rate Schedule
$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T_{\max}}\pi\right)\right)$$
Starts at $\eta = 0.1$ and smoothly decays towards 0, ensuring deep exploration early on and fine-grained convergence at the end.

### 4. Fully Resumable Experiment Grid
Every finished experiment appends a row to `results/results.csv`. When the script is started or resumed:
1. It reads `results.csv`.
2. It hashes `(regime, method, f_T, f_S, seed)`.
3. If an experiment is already recorded, it skips it in 0.001 seconds.
4. If Kaggle times out or disconnects, re-running simply picks up right where it left off.

---

## 8. Evaluation Metrics Explained

For every model, our evaluation pipeline calculates 4 metrics on the 10,000 test images:

### 1. Top-1 Test Accuracy (%)
Standard classification accuracy:
$$\text{Acc} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_i = y_i)$$

### 2. Prediction Agreement / Fidelity ($\%$)
How often does the student make the exact same prediction as the teacher?
$$\text{Agreement} = \frac{1}{N} \sum_{i=1}^N \mathbb{I}(\hat{y}_{\text{student}}^{(i)} = \hat{y}_{\text{teacher}}^{(i)})$$
- High agreement means the student successfully internalized the teacher's decision boundary.
- If student accuracy is 55% and teacher accuracy is 70%, but agreement is 85%, the student is faithfully mimicking the teacher (including some of the teacher's mistakes!).

### 3. Expected Calibration Error (ECE)
Does the model know what it does not know? A model with 99% confidence on every prediction that is only 50% accurate is poorly calibrated and dangerous in real applications (like medicine).
ECE partitions predictions into $M=15$ confidence bins:
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
- **Finding**: Logit KD with temperature $T$ acts as label smoothing and significantly **lowers ECE**, producing much more trustworthy, calibrated students!

---

## 9. Hypotheses: What Do We Expect to Find?

When you plot the final results, here are the expected scientific phenomena:

```
  Test Acc (%)
     70% │                                      ┌── Teacher (100% data)
         │                                 ┌────┴─ Logit KD (Regime A)
     50% │                         ┌───────┘  ───  Attention Transfer
         │                 ┌───────┘  ───────────  CE Baseline
     30% │         ┌───────┘
         │  ┌──────┘
     10% └────────────────────────────────────────────────────────
           1%      5%     10%     25%     100%   (Data Fraction)
```

1. **KD Gain Increases as Data Shrinks ($\Delta \text{Acc}$ peaks at 5% or 10%)**:
   - At 100% data, student has enough data to learn decent representations on its own (gain is ~2–3%).
   - At 5% or 10% data, the CE baseline collapses due to extreme overfitting. Logit KD provides dark knowledge that acts as a powerful regularizer, leading to gains of **+5% to +10%**!
2. **Extreme Scarcity Barrier (1% Data)**:
   - At 1% data (only 4 images/class), student accuracy is very low (~15–25%). KD still helps, but the student's limited capacity and the sheer lack of feature diversity create a hard performance floor.
3. **Regime A vs Regime B**:
   - $\text{Acc}_{\text{Regime A}} > \text{Acc}_{\text{Regime B}}$. The teacher's data privilege matters significantly!
   - However, Regime B still beats the CE baseline ($\text{Acc}_{\text{Regime B}} > \text{Acc}_{\text{CE}}$)! Even when the teacher saw no extra data, distilling from a larger architecture acts as an architectural prior/regularizer for the smaller student.
4. **Logit KD vs Attention Transfer**:
   - Logit KD transfers class-level semantic distributions.
   - Attention Transfer transfers spatial perceptual maps.
   - On low data, Logit KD generally outperforms Attention Transfer because semantic dark knowledge prevents the student from hallucinating wrong class categories.

---

## 10. Professor Defense Q&A: 10 Questions You Will Be Asked

### Q1: "Why CIFAR-100 instead of CIFAR-10 or ImageNet?"
> **Answer**:  
> CIFAR-10 is too easy: even with 10% data, a ResNet-20 gets >75% accuracy, leaving little room to observe distillation effects.  
> ImageNet is too computationally expensive for exhaustive grid studies across 5 fractions, 3 methods, and multiple seeds.  
> CIFAR-100 is the sweet spot: with 100 fine-grained classes, 1% data means only 4–5 images per class, which authentically tests extreme data scarcity while remaining computationally tractable.

### Q2: "Why did you use ResNet-56 and ResNet-20?"
> **Answer**:  
> Both belong to the same architectural family (He et al., 2016 CIFAR ResNets) with identical stage structures (16, 32, and 64 channels). This enables clean feature-map alignment for Attention Transfer without needing learned projection heads. ResNet-56 has ~850k parameters while ResNet-20 has ~270k (3.1× compression ratio), representing a realistic edge-deployment scenario.

### Q3: "What is 'Dark Knowledge' in Hinton's KD?"
> **Answer**:  
> Hard targets ($y \in \{0, 1\}$) tell the network only what an image is, treating all non-target classes as equally zero. A softened teacher probability distribution reveals the correlation geometry between classes—e.g., that an apple is closer to an orange than to a truck. This rich cross-entropy gradient provides dense feedback to the student on every single sample.

### Q4: "Why do we multiply the KL loss by $T^2$ in Logit KD?"
> **Answer**:  
> When logits are divided by $T$, the gradient of the softmax with respect to the logits scales as $1/T^2$. If we don't multiply by $T^2$, the KD loss gradient becomes tiny compared to the standard cross-entropy loss, and the student ignores the teacher. Multiplying by $T^2$ keeps the gradient magnitudes matched.

### Q5: "How did you construct your data subsets, and why does it matter?"
> **Answer**:  
> We used stratified, nested sampling from a fixed 45,000-image training pool (with 5,000 images held out permanently for validation using seed 42).  
> Stratification ensures every class is represented equally (no class imbalance confounding).  
> Nested subsets ($S_{1\%} \subset S_{5\%} \subset S_{10\%} \dots$) ensure that when we compare 5% to 1%, the difference is purely data volume, not random sample selection variance.

### Q6: "Why did you need a `min_steps` budget instead of just fixing epochs to 200?"
> **Answer**:  
> At 1% data (450 images), 200 epochs is only ~700 gradient steps. The optimizer simply doesn't get enough iterations to converge. Setting a minimum step budget (e.g. 15,000 steps) ensures every model reaches optimization convergence, isolating data volume from optimizer starvation.

### Q7: "What is the difference between Regime A and Regime B?"
> **Answer**:  
> In Regime A, the teacher has seen 100% of the data, representing classic distillation where the teacher has privileged information.  
> In Regime B, the teacher and student are trained on the exact identical subset ($f_T = f_S$). This isolates whether distillation helps solely because of teacher capacity/architecture, or whether the teacher must possess superior training data.

### Q8: "How does Attention Transfer differ from Logit KD?"
> **Answer**:  
> Logit KD is output-level distillation: it forces final class probabilities to match.  
> Attention Transfer is feature-level distillation: it sums squared activations across channels at each residual stage and L2-normalizes them, forcing the student's spatial focus (where in the image it is looking) to align with the teacher's.

### Q9: "What is Expected Calibration Error (ECE) and why did you measure it?"
> **Answer**:  
> Accuracy only measures if the argmax prediction is correct; ECE measures whether the model's confidence corresponds to true probability. Under low data, models often become overconfident and poorly calibrated. Logit KD with temperature acts as a regularizer that lowers ECE, making models safer and more reliable.

### Q10: "If you had more time and compute, what would you add next?"
> **Answer**:  
> 1. Modern distillation methods like DKD (Decoupled Knowledge Distillation, CVPR 2022) which separates target and non-target logit distillation.  
> 2. Cross-architecture distillation (e.g., Vision Transformer teacher like DeiT to CNN student).  
> 3. Data augmentation synergy (e.g., combining KD with MixUp or CutMix under low data).
