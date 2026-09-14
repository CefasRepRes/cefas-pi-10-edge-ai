# training_defaults.yaml — Explained

This document explains **every configurable training knob** exposed in `training_defaults.yaml`, with particular focus on **minority-class handling**, **cost asymmetry**, and **decision behaviour**. It is intended to be **self-contained**: all rationale, guidance, and presets live in this file.

The configuration is intentionally split into **three conceptual layers**:

1. **Base model & optimisation** – how the network learns at all
2. **Imbalance handling during training** – what the optimiser is encouraged to care about
3. **Decision-making after training** – how probabilities are turned into labels

Understanding which layer a knob affects is critical: these knobs are **not redundant** and are designed to work together.

---

## 1. Base model & optimisation (context)

These settings are conventional and mostly orthogonal to class imbalance. They define *capacity*, *training stability*, and *regularisation*.

```yaml
model:
  architecture: resnet18
  pretrained: true

data:
  image_size: [256, 256]
  normalisation: imagenet

training:
  batch_size: 64
  epochs: 50
  learning_rate: 3e-4
  optimizer: adam
  weight_decay: 1e-4
  num_workers: 8
  val_fraction: 0.2
  seed: 1337
```

Nothing here encodes preferences about *which* mistakes matter more.

---

## 2. Loss selection — what gradients focus on

### 2.1 Cross-entropy (baseline)

```yaml
loss:
  name: cross_entropy
```

**Behaviour**
- All samples contribute equally
- Easy majority-class examples dominate gradients
- Optimises global accuracy

Use when:
- Classes are roughly balanced
- Establishing a neutral baseline

---

### 2.2 Focal loss — focus on hard cases

```yaml
loss:
  name: focal
  focal_gamma: 2.0
```

Focal loss multiplies cross-entropy by `(1 - p_t)^γ`, where `p_t` is the model’s confidence in the *true* class.

**Behaviour**
- Easy, already-correct examples fade out
- Misclassified and borderline samples dominate learning
- Minority classes benefit indirectly, in a way which is relatively impartial

**What it does NOT do**
- Does not encode class importance
- Does not encode cost asymmetry

Gamma guidance:
- γ = 1 → mild
- γ = 2 → safe default
- γ ≥ 3 → aggressive; risk of instability

---

## 3. Class weighting — which mistakes hurt more

```yaml
class_weighting:
  enabled: true
  minority:
    mode: bottom_k | below_fraction | below_count
    k: 1
    fraction: 0.2
    count: 50
    multiplier: 5.0
    min_count: 1
```

Class weighting multiplies the loss **only for samples whose true class is minority**.

> This is the primary mechanism for saying: “false negatives on class X matter more.”

### 3.1 Minority selection modes

| Mode | Meaning |
|------|--------|
| bottom_k | Select k rarest classes |
| below_fraction | Bottom X% of classes |
| below_count | ≤ N samples |

### 3.2 Multiplier (P)

```yaml
multiplier: 5.0
```

- Minority-class mistakes cost ~P× more gradient
- Strongly increases recall pressure

Expected effects:
- ✅ Fewer false negatives
- ❌ More false positives unless corrected downstream

### 3.3 Interaction with focal loss

| Combination | Effect |
|------------|--------|
| CE + weights | All minority errors matter more |
| Focal + weights | Hard minority errors dominate |

---

## 4. Threshold calibration — when to predict a class

This stage is **post-training**. No gradients. No retraining.

```yaml
threshold_calibration:
  enabled: true
  target: minority | all | list
  objective: min_cost | maximize_fbeta
  fn_cost: 5.0
  fp_cost: 1.0
  beta: 2.0
  grid_size: 201
```

Training produces probabilities. Deployment requires decisions.

Argmax decisions are *not* cost-aware.

Threshold calibration explicitly separates:

> “How sure is the model?” from “What should we do?”

### 4.1 Target selection

| Target | Meaning |
|-------|--------|
| minority | Only recalibrate minority classes |
| all | Recalibrate every class |
| list | Explicit list |

### 4.2 Objectives

#### min_cost

```yaml
objective: min_cost
fn_cost: 10.0
fp_cost: 1.0
```

Optimises:

```
FN * fn_cost + FP * fp_cost
```

This is the **most interpretable and auditable** way to encode domain preferences.

#### maximize_fbeta

```yaml
objective: maximize_fbeta
beta: 2.0
```

Optimises recall-weighted harmonic mean when costs are unknown.

---

## 5. Recommended presets (copy-paste ready)

### Conservative baseline

```yaml
training:
  loss:
    name: cross_entropy
  class_weighting:
    enabled: false
  threshold_calibration:
    enabled: false
```

### Balanced default

```yaml
training:
  loss:
    name: focal
    focal_gamma: 2.0
  class_weighting:
    enabled: true
    minority:
      mode: below_fraction
      fraction: 0.2
      multiplier: 3.0
  threshold_calibration:
    enabled: false
```

### Cost-aware deployment

```yaml
training:
  loss:
    name: focal
  class_weighting:
    enabled: true
    minority:
      mode: bottom_k
      k: 1
      multiplier: 5.0
  threshold_calibration:
    enabled: true
    target: minority
    objective: min_cost
    fn_cost: 5.0
    fp_cost: 1.0
```

### Aggressive recall

```yaml
training:
  loss:
    name: cross_entropy
  class_weighting:
    enabled: true
    minority:
      mode: bottom_k
      k: 1
      multiplier: 10.0
  threshold_calibration:
    enabled: true
    target: minority
    objective: min_cost
    fn_cost: 10.0
    fp_cost: 1.0
```

---
