# Fine-Tuning (FT) & Super-Fine-Tuning (SFT) Backdoor Defense Guide

> Based on: *"Fine-Tuning Is All You Need to Mitigate Backdoor Attacks"* (Sha et al., 2022)

---

## Overview

Both FT and SFT remove backdoors from ML models by fine-tuning on a clean dataset. The key insight is that **large learning rates force the model to forget backdoor triggers**, while **small learning rates preserve clean accuracy**. SFT combines both via a dynamic learning rate schedule.

**When to use which:**

| Scenario | Recommended Method |
|---|---|
| Encoder-based (e.g., SimCLR + BadEncoder) | Conventional FT (whole model) |
| Transfer-based (pre-trained → fine-tuned) | SFT preferred; FT often sufficient |
| Standalone (user directly deploys model) | SFT required; FT usually fails |

---

## Method 1: Conventional Fine-Tuning (FT)

Fine-tune the **whole model** (not just the classifier) on a clean dataset using the same learning rate as pre-training.

**Key requirements:**
- Whole model fine-tuning, not just the last layer
- Use the same LR as pre-training (e.g., `lr = 0.0003`)
- Clean dataset required (can use as little as 20% of training data)

**Hyperparameters:**

```yaml
lr: 0.0003
lr_scheduler: none
client_optimizer: sgd
sgd_momentum: 0.9
wd: 5.0e-4
epochs: 100        # FT needs more epochs than SFT
ratio: 1.0         # Use full clean dataset for FT
```

**Limitation:** In the standalone scenario, FT often fails — ASR can remain high (e.g., 0.978 for Blended on CIFAR10 after 100 epochs). Use SFT instead.

---

## Method 2: Super-Fine-Tuning (SFT)

SFT uses a two-phase oscillating learning rate schedule inspired by super-convergence. It is more effective than FT in all scenarios and especially necessary for standalone defense.

### Learning Rate Schedule

```
Phase 1 (epochs 0 → phase1_epochs):
  LR oscillates: LR_BASE → LR_MAX1 → LR_BASE (repeat each cycle_len epochs)

Phase 2 (epochs phase1_epochs → total_epochs):
  LR oscillates: LR_BASE → LR_MAX2 → LR_BASE (lower max, reduces overfitting)
```

- **LR_MAX1** (large): forces the model to forget backdoor triggers quickly
- **LR_MAX2** (medium): continues refinement without too much utility loss
- **LR_BASE** (small): anchors the model back to maintain clean accuracy

### SFT Config (your provided config)

```yaml
device: 'cuda'
amp: True
pin_memory: True
non_blocking: True
prefetch: False

checkpoint_load:
checkpoint_save:
log:
dataset_path: './data'
dataset: 'cifar10'

# Training
epochs: 6              # Total epochs (Phase 1: 3, Phase 2: 3)
batch_size: 256
num_workers: 4
model: 'resnet20'

# Optimizer
client_optimizer: 'sgd'
lr: 0.0003             # LR_BASE — initial / anchor learning rate
sgd_momentum: 0.9
wd: 5.0e-4
lr_scheduler: none     # SFT manages its own LR schedule
frequency_save: 0

# Data
ratio: 0.05            # Use 5% of clean dataset
index:
random_seed: 0

# SFT-specific parameters
lr_base: 0.0003        # Anchor LR (same as lr)
lr_max1: 0.1           # Phase 1 peak — aggressively removes backdoor
lr_max2: 0.001         # Phase 2 peak — gentle refinement
phase1_epochs: 3       # Epochs in first phase
cycle_len: 2           # LR ramp-up/down period in iterations (×500 iterations)
```

### SFT Config Parameters Explained

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `lr` / `lr_base` | 0.0003 | Base LR — maintains clean accuracy when active |
| `lr_max1` | 0.1 | Peak LR in Phase 1 — critical for forgetting backdoor triggers |
| `lr_max2` | 0.001 | Peak LR in Phase 2 — reduces overfitting after Phase 1 |
| `phase1_epochs` | 3 | Number of epochs for aggressive backdoor removal |
| `cycle_len` | 2 | Controls oscillation frequency within each phase |
| `epochs` | 6 | Total epochs (phase1=3, phase2=3) |
| `ratio` | 0.05 | Only 5% of clean data needed — very efficient |

### Tuning Advice

**If backdoor is not fully removed (ASR still high):**
- Increase `lr_max1` (e.g., `0.1` → `0.3`) — especially when `ratio` is small
- Increase `phase1_epochs` (e.g., `3` → `5`)

**If clean accuracy (CA) drops too much:**
- Decrease `lr_max1` slightly
- Increase `lr_base` to maintain utility
- Add more `phase2_epochs` (extend `epochs - phase1_epochs`)

**If you have very limited data (`ratio` ≤ 0.1):**
- Must increase `lr_max1` to compensate (e.g., 0.3 or higher)
- The paper shows that with only 10% data, LR_MAX1=0.1 may fail for Blended/LF attacks

---

## Implementing SFT in PyTorch

```python
import torch
import torch.nn as nn
import numpy as np

def get_sft_lr(
    current_iter,      # global iteration count
    iterations_per_epoch,
    lr_base,
    lr_max1,
    lr_max2,
    phase1_epochs,
    cycle_len,         # half-cycle length in epochs
):
    """Compute the SFT learning rate at a given iteration."""
    cycle_iters = cycle_len * iterations_per_epoch
    phase1_iters = phase1_epochs * iterations_per_epoch

    if current_iter < phase1_iters:
        # Phase 1: oscillate between lr_base and lr_max1
        pos_in_cycle = current_iter % cycle_iters
        if pos_in_cycle < cycle_iters / 2:
            # Ramp up
            lr = lr_base + (lr_max1 - lr_base) * (pos_in_cycle / (cycle_iters / 2))
        else:
            # Ramp down
            lr = lr_max1 - (lr_max1 - lr_base) * ((pos_in_cycle - cycle_iters / 2) / (cycle_iters / 2))
    else:
        # Phase 2: oscillate between lr_base and lr_max2
        pos_in_cycle = (current_iter - phase1_iters) % cycle_iters
        if pos_in_cycle < cycle_iters / 2:
            lr = lr_base + (lr_max2 - lr_base) * (pos_in_cycle / (cycle_iters / 2))
        else:
            lr = lr_max2 - (lr_max2 - lr_base) * ((pos_in_cycle - cycle_iters / 2) / (cycle_iters / 2))

    return lr


def super_fine_tune(model, clean_loader, config, device='cuda'):
    """
    Super-fine-tune a backdoored model using the SFT schedule.

    Args:
        model:        backdoored model (whole model, not just classifier)
        clean_loader: DataLoader of clean data (even 5% of training set works)
        config:       dict with lr_base, lr_max1, lr_max2, phase1_epochs,
                      cycle_len, epochs, wd, sgd_momentum
        device:       'cuda' or 'cpu'
    """
    model = model.to(device)
    model.train()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config['lr_base'],
        momentum=config['sgd_momentum'],
        weight_decay=config['wd'],
    )
    criterion = nn.CrossEntropyLoss()

    iterations_per_epoch = len(clean_loader)
    global_iter = 0

    for epoch in range(config['epochs']):
        for inputs, labels in clean_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Update LR
            lr = get_sft_lr(
                current_iter=global_iter,
                iterations_per_epoch=iterations_per_epoch,
                lr_base=config['lr_base'],
                lr_max1=config['lr_max1'],
                lr_max2=config['lr_max2'],
                phase1_epochs=config['phase1_epochs'],
                cycle_len=config['cycle_len'],
            )
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            global_iter += 1

        print(f"Epoch [{epoch+1}/{config['epochs']}] | LR: {lr:.6f}")

    return model
```

### Usage Example

```python
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random

# Load backdoored model
model = torch.load('backdoored_model.pt')

# Prepare clean dataset (only 5% needed per config)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])
full_dataset = datasets.CIFAR10('./data', train=True, transform=transform)

ratio = 0.05
subset_size = int(len(full_dataset) * ratio)
indices = random.sample(range(len(full_dataset)), subset_size)
clean_subset = Subset(full_dataset, indices)
clean_loader = DataLoader(clean_subset, batch_size=256, shuffle=True, num_workers=4)

# SFT config (matches provided YAML)
config = {
    'lr_base':       0.0003,
    'lr_max1':       0.1,
    'lr_max2':       0.001,
    'phase1_epochs': 3,
    'cycle_len':     2,
    'epochs':        6,
    'sgd_momentum':  0.9,
    'wd':            5e-4,
}

# Run SFT
defended_model = super_fine_tune(model, clean_loader, config, device='cuda')

# Save
torch.save(defended_model.state_dict(), 'defended_model.pt')
```

---

## Integrating as a Landseer Post-Training Tool

This defense fits the **`post_training`** pipeline stage in Landseer.

### `main.py` skeleton

```python
import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from config_model import config as get_model

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/data")
parser.add_argument("--output",    default="/output")
parser.add_argument("--method",    default="sft",    choices=["ft", "sft"])
parser.add_argument("--epochs",    type=int, default=6)
parser.add_argument("--ratio",     type=float, default=0.05)
parser.add_argument("--lr-base",   type=float, default=3e-4)
parser.add_argument("--lr-max1",   type=float, default=0.1)
parser.add_argument("--lr-max2",   type=float, default=0.001)
parser.add_argument("--phase1-epochs", type=int, default=3)
parser.add_argument("--cycle-len", type=int, default=2)
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
model = get_model().to(device)
state = torch.load(os.path.join(args.input_dir, "model.pt"), map_location=device)
model.load_state_dict(state)

# Load clean data
X = np.load(os.path.join(args.input_dir, "data.npy"))
Y = np.load(os.path.join(args.input_dir, "labels.npy"))

# Subsample by ratio
n = int(len(X) * args.ratio)
idx = np.random.choice(len(X), n, replace=False)
X, Y = X[idx], Y[idx]

dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(Y))
loader  = DataLoader(dataset, batch_size=256, shuffle=True)

# Run defense
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_base,
                             momentum=0.9, weight_decay=5e-4)

iters_per_epoch = len(loader)
global_iter = 0

for epoch in range(args.epochs):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        if args.method == "sft":
            cycle_iters = args.cycle_len * iters_per_epoch
            phase1_iters = args.phase1_epochs * iters_per_epoch
            pos = global_iter % cycle_iters
            half = cycle_iters / 2
            if global_iter < phase1_iters:
                peak = args.lr_max1
            else:
                peak = args.lr_max2
            if pos < half:
                lr = args.lr_base + (peak - args.lr_base) * (pos / half)
            else:
                lr = peak - (peak - args.lr_base) * ((pos - half) / half)
            for g in optimizer.param_groups:
                g['lr'] = lr

        optimizer.zero_grad()
        loss = criterion(model(xb), yb)
        loss.backward()
        optimizer.step()
        global_iter += 1

    print(f"[{args.method.upper()}] Epoch {epoch+1}/{args.epochs}")

# Save
os.makedirs(args.output, exist_ok=True)
m = model.module if hasattr(model, 'module') else model
torch.save(m.state_dict(), os.path.join(args.output, "model.pt"))
print("Done. Saved to", args.output)
```

### Apptainer Definition (`post_sft.def`)

```singularity
Bootstrap: docker
From: python:3.10-slim

%labels
    org.opencontainers.image.dataset="CIFAR-10"
    org.opencontainers.image.defense_stage="post_training"
    org.opencontainers.image.defense_type="adversarial_training"
    org.opencontainers.image.framework="pytorch"

%post
    set -e
    apt-get update && apt-get install -y --no-install-recommends tini
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir torch torchvision numpy tqdm

%files
    main.py /app/main.py

%environment
    export PYTHONUNBUFFERED=1

%runscript
    exec /usr/bin/tini -- python /app/main.py \
        --input-dir /data --output /output \
        --method sft --epochs 6 --ratio 0.05 \
        --lr-base 0.0003 --lr-max1 0.1 --lr-max2 0.001 \
        --phase1-epochs 3 --cycle-len 2
```

### Landseer Pipeline Config Snippet

```yaml
post_training:
  tools:
    - name: post-sft
      container:
        image: oras://ghcr.io/landseer-project/post_sft:v1
        command: python /app/main.py --method sft --epochs 6 --ratio 0.05
  noop:
    name: post_noop
    container:
      image: docker://ghcr.io/landseer-project/post_noop_new:v1
      command: python main.py
```

---

## Expected Results Summary

| Attack | Scenario | FT (100ep) ASR | SFT (6ep) ASR | SFT CA |
|--------|----------|---------------|---------------|--------|
| BadNets | Standalone | ~0.1 | **0.009** | 0.932 |
| Blended | Standalone | 0.978 | **0.081** | 0.937 |
| Inputaware | Standalone | ~0.8 | **~0.05** | ~0.93 |
| LF | Standalone | ~0.5 | **~0.05** | ~0.92 |
| WaNet | Standalone | ~0.6 | **~0.05** | ~0.93 |

SFT uses only **0.089 GPU hours** vs NC's **0.997 GPU hours** for the same task.

---

## Key Takeaways

1. **SFT only needs ~5% of clean data** and ~6 epochs — very cheap
2. **LR_MAX1 is the most important hyperparameter** — raise it if ASR stays high
3. **SFT does not hurt privacy** — membership inference risk actually decreases after SFT
4. **Re-injection caveat** — once defended, the model is somewhat easier to re-backdoor; this affects all defense methods, not just SFT
5. For Landseer integration, this is a **`post_training`** stage tool
