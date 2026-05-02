# LANDSEER Setup & Tool Integration Guide

> **Purpose**: This guide is written for Claude Code to help set up Landseer on the Gilbreth HPC cluster and integrate new ML defense tools into the pipeline.

---

## Overview

Landseer is a containerized ML security evaluation pipeline that benchmarks defense tools across four stages: pre-training, during-training, post-training, and deployment. Tools run inside containers (Apptainer on HPC) and communicate via standardized `/data` and `/output` directories.

**Key constraint**: Gilbreth does NOT support Docker — always use **Apptainer**.

---

## Part 1: Environment Setup (Gilbreth)

### 1.1 SSH Key for GitHub

```bash
ssh-keygen -t rsa -C <your_github_email>
# Press Enter to accept default path; optionally set a passphrase
cat ~/.ssh/id_rsa.pub   # Copy this output
# Add it at https://github.com/settings/keys
ssh -T git@github.com   # Verify; type "yes" when prompted
```

### 1.2 Fork & Clone the Repository

1. Fork `https://github.com/landseer-project/Landseer` on GitHub
2. Go to your fork → **<> Code** → **SSH** → copy the link
3. On Gilbreth:

```bash
git clone <your-ssh-fork-url>
cd Landseer
```

### 1.3 Create Conda Environment

```bash
conda create -n landseer python=3.11 -y
conda activate landseer
```

> **Required on Gilbreth** — never skip this step.

### 1.4 Install Poetry & Dependencies

```bash
python3 -m pip install --user --upgrade pip
python3 -m pip install --user poetry

echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

poetry --version   # Verify

poetry install     # Takes 5-10 min (PyTorch, scikit-learn, etc.)
```

### 1.5 Configure Apptainer Cache (Critical!)

Apptainer image caches will fill your `$HOME` quota if not redirected. Add to `~/.bashrc`:

```bash
export APPTAINER_CACHEDIR=/scratch/gilbreth/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/gilbreth/$USER/apptainer_tmp
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"
```

Then reload:

```bash
source ~/.bashrc
apptainer --version   # Verify
```

---

## Part 2: Running Landseer

### Basic Usage

```bash
conda activate landseer
cd /scratch/gilbreth/$USER/Landseer

poetry run landseer \
  -c configs/pipeline/trades.yaml \
  -a configs/attack/test_config_1.yaml
```

### SLURM Job Script (`run_landseer.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=landseer
#SBATCH --output=landseer_%j.out
#SBATCH --error=landseer_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu

module load anaconda
module load cuda

conda activate landseer

export APPTAINER_CACHEDIR=/scratch/gilbreth/$USER/apptainer_cache
export APPTAINER_TMPDIR=/scratch/gilbreth/$USER/apptainer_tmp

cd /scratch/gilbreth/$USER/Landseer

poetry run landseer \
  -c configs/pipeline/trades.yaml \
  -a configs/attack/test_config_1.yaml
```

```bash
sbatch run_landseer.sh
squeue -u $USER          # Check status
tail -f landseer_<job_id>.out   # Watch output live
```

### Output Files

Results land in `results/`:

| File | Description |
|------|-------------|
| `results_combinations.csv` | Results for each tool combination |
| `results_tools.csv` | Per-tool execution details |
| `logs/` | Detailed execution logs |

---

## Part 3: Integrating a New Tool into Landseer

### 3.1 Understand Pipeline Stages

Your tool must target exactly one stage:

| Stage | Purpose | Input | Output |
|-------|---------|-------|--------|
| `pre_training` | Data preprocessing / outlier detection | `.npy` dataset | Cleaned `.npy` dataset |
| `during_training` | Training with defenses | `.npy` dataset | `model.pt` |
| `post_training` | Model refinement | Model + dataset | `model.pt` |
| `deployment` | Runtime defenses | Model + dataset | Processed artifacts |

### 3.2 Modify Your Tool's Code

#### `main.py` — Required Argument Interface

```python
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", default="/data",
                    help="Input directory (mounted by Landseer)")
parser.add_argument("--output", default="/output",
                    help="Output directory (mounted by Landseer)")
args = parser.parse_args()
```

#### Model Config Import (all stages except pre-training)

Landseer injects `config_model.py` into the container. Use it like this:

```python
from config_model import config

model = config()   # Returns the shared model architecture
```

#### Dataset Loading — NumPy Format

```python
import numpy as np, os

X_train = np.load(os.path.join(args.input_dir, "data.npy"))        # (N, C, H, W), float32, [0,1]
Y_train = np.load(os.path.join(args.input_dir, "labels.npy"))      # (N,), int64
X_test  = np.load(os.path.join(args.input_dir, "test_data.npy"))   # (M, C, H, W), float32, [0,1]
Y_test  = np.load(os.path.join(args.input_dir, "test_labels.npy")) # (M,), int64
```

> **TensorFlow users**: TF uses `(N, H, W, C)` — transpose before saving:
> ```python
> X_train = np.transpose(X_train, (0, 3, 1, 2))
> ```

#### Saving Output

**Pre-training tools** — save cleaned dataset:

```python
os.makedirs(args.output, exist_ok=True)
np.save(os.path.join(args.output, "data.npy"),        X_train.astype(np.float32))
np.save(os.path.join(args.output, "labels.npy"),      Y_train.astype(np.int64))
np.save(os.path.join(args.output, "test_data.npy"),   X_test.astype(np.float32))
np.save(os.path.join(args.output, "test_labels.npy"), Y_test.astype(np.int64))
```

**Training / post-training tools** — save model:

```python
import torch

# Handle DataParallel wrapper
model_to_save = model.module if hasattr(model, 'module') else model
torch.save(model_to_save.state_dict(), os.path.join(args.output, "model.pt"))
```

### 3.3 Create the Apptainer Definition File

**Naming convention**: `<stage_prefix>_<toolname>.def`

| Stage | Prefix | Example |
|-------|--------|---------|
| Pre-training | `pre_` | `pre_myfilter` |
| During-training | `in_` | `in_mytrain` |
| Post-training | `post_` | `post_myprune` |
| Deployment | `deploy_` | `deploy_mydp` |

**Template** (`in_mytool.def`):

```singularity
Bootstrap: docker
From: python:3.10-slim

%labels
    org.opencontainers.image.dataset="CIFAR-10"
    org.opencontainers.image.defense_stage="during_training"
    org.opencontainers.image.defense_type="adversarial_training"
    org.opencontainers.image.framework="pytorch"

%post
    set -e
    apt-get update && apt-get install -y --no-install-recommends tini
    rm -rf /var/lib/apt/lists/*
    pip install --no-cache-dir torch torchvision numpy tqdm scikit-learn
    mkdir -p /app

%files
    main.py /app/main.py

%environment
    export PYTHONUNBUFFERED=1
    export PIP_NO_CACHE_DIR=1

%runscript
    exec /usr/bin/tini -- python /app/main.py --input-dir /data --output /output
```

**Valid label values**:

- `defense_stage`: `pre_training`, `during_training`, `post_training`, `deployment`
- `defense_type`: `adversarial_training`, `differential_privacy`, `watermarking`, `outlier_removal`, `explanation`, `fairness`, `fingerprinting`
- `dataset`: `CIFAR-10`, `CelebA`, `MNIST`
- `framework`: `pytorch`, `tensorflow`

### 3.4 Build the Apptainer Image

```bash
# Preferred: local build with fakeroot
apptainer build --fakeroot in_mytool.sif in_mytool.def

# Alternative: remote build (no fakeroot required)
apptainer remote add --no-default sylabs cloud.sylabs.io
apptainer remote use sylabs
apptainer key new   # First time only
apptainer build --remote in_mytool.sif in_mytool.def
```

### 3.5 Test Locally

```bash
apptainer exec --nv \
  -B /path/to/data:/data \
  -B /path/to/output:/output \
  in_mytool.sif python /app/main.py --input-dir /data --output /output
```

### 3.6 Push to GitHub Container Registry (GHCR)

**One-time setup**:

1. Create a GitHub PAT with `write:packages` and `delete:packages` scope
2. Login:

```bash
apptainer registry login -u YOUR_GITHUB_USERNAME oras://ghcr.io
# Enter PAT as password
```

**Push image**:

```bash
apptainer push in_mytool.sif oras://ghcr.io/landseer-project/in_mytool:v1
```

**Make package public**:

1. Go to `https://github.com/orgs/landseer-project/packages`
2. Click your package → **Package settings** → scroll to bottom → **Change visibility → Public**

---

## Part 4: Pipeline Configuration

### 4.1 Testing Only Your Tool

When testing a single tool, use `noop` containers at all other stages. Remove the `tools:` section from stages that don't involve your tool.

**Example for a `during_training` tool** (`configs/pipeline/my_tool_test.yaml`):

```yaml
dataset:
  name: cifar10        # Options: cifar10, celeba, mnist
  variant: clean       # Options: clean, poisoned

model:
  script: configs/model/config_model.py
  framework: pytorch

pipeline:
  pre_training:
    # No tools here — baseline passthrough
    noop:
      name: noop
      container:
        image: docker://ghcr.io/landseer-project/pre_noop:v1
        command: python main.py

  during_training:
    tools:
      - name: my-tool
        container:
          image: oras://ghcr.io/landseer-project/in_mytool:v1   # oras:// for Apptainer!
          command: python /app/main.py
    noop:
      name: in_noop
      container:
        image: docker://ghcr.io/landseer-project/in_noop:v5
        command: python main.py

  post_training:
    noop:
      name: post_noop
      container:
        image: docker://ghcr.io/landseer-project/post_noop_new:v1
        command: python main.py

  deployment:
    noop:
      name: deploy_noop
      container:
        image: docker://ghcr.io/landseer-project/deploy_noop_docker:v1
        command: python main.py
```

### 4.2 Quick Stage Reference

| Your Tool's Stage | Keep `tools:` in | Use only `noop:` in |
|---|---|---|
| Pre-training | `pre_training` | during, post, deployment |
| During-training | `during_training` | pre, post, deployment |
| Post-training | `post_training` | pre, during, deployment |
| Deployment | `deployment` | pre, during, post |

### 4.3 Available Noop Images

| Stage | Image (Docker prefix) |
|-------|-----------------------|
| Pre-training | `ghcr.io/landseer-project/pre_noop:v1` |
| During-training | `ghcr.io/landseer-project/in_noop:v5` |
| Post-training | `ghcr.io/landseer-project/post_noop_new:v1` |
| Deployment | `ghcr.io/landseer-project/deploy_noop_docker:v1` |

> **Image prefix rule**: Docker images use `docker://`, Apptainer `.sif` images use `oras://`. Landseer defaults to Docker format if no prefix is given.

### 4.4 Attack Configuration (`configs/attack/`)

```yaml
attacks:
  backdoor: True        # Backdoor attack
  adversarial: True     # PGD, FGSM, C&W
  outlier: True         # Out-of-distribution detection
  carlini: True         # Carlini & Wagner L2
  watermarking: True    # Watermark detection
  fingerprinting: True  # Model fingerprinting
  inference: True       # Membership inference, etc.
  other: True
```

### 4.5 Model Configuration (`configs/model/`)

Create a Python file with a `config()` function:

```python
def config():
    return resnet20()   # Return your model architecture
```

---

## Part 5: Caching Behavior

- Cache key = hash of (tool image + input data + config)
- Cached results are reused automatically — tool is skipped if cache hit
- If you update your image but **keep the same tag** (e.g., `v1`), the old cache may still be used

**Fix**: either bump the version tag (`v1` → `v2`) or delete the cache folder:

```bash
rm -rf cache/<relevant_subfolder>
```

---

## Part 6: Storage Management on Gilbreth

Storage issues are responsible for ~50% of HPC failures.

```bash
myquota   # Check current usage

# Find what's eating space
du -h --max-depth=2 ~ | sort -hr | head -20
du -sh ~/.apptainer/

# Clear Apptainer cache from home (move to scratch)
rm -rf ~/.apptainer/cache
export APPTAINER_CACHEDIR=/scratch/gilbreth/$USER/apptainer_cache
```

---

## Part 7: Complete Workflow Checklist

For deploying a new tool end-to-end:

```
[ ] 1. Write main.py with --input-dir and --output args
[ ] 2. Import config_model and load/save data in .npy format
[ ] 3. Save output as model.pt (training tools) or .npy (pre-training tools)
[ ] 4. Create <stage>_<toolname>.def with required LABEL fields
[ ] 5. Build: apptainer build --fakeroot <stage>_<toolname>.sif <stage>_<toolname>.def
[ ] 6. Test: apptainer exec --nv -B data:/data -B output:/output <img>.sif python /app/main.py
[ ] 7. Push: apptainer push <img>.sif oras://ghcr.io/landseer-project/<img>:v1
[ ] 8. Make GHCR package public
[ ] 9. Create configs/pipeline/my_config.yaml with oras:// prefix for your image
[ ] 10. Run: poetry run landseer -c configs/pipeline/my_config.yaml -a configs/attack/test_config_1.yaml
[ ] 11. Check results/ for output CSVs and logs/
```
