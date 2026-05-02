# BackdoorBench Post-Training Wrapper

This integration wraps BackdoorBench `ft` and `sft` so they can run as `Landseer` `post_training` tools.

## What It Assumes

- Input is the standard Landseer tool contract:
  - `/data/data.npy`
  - `/data/labels.npy`
  - `/data/test_data.npy`
  - `/data/test_labels.npy`
  - `/data/model.pt`
- The model architecture is provided by `/app/config_model.py`.
- The dataset is currently `cifar10`.
- Landseer's poisoned training set is constructed by appending poisoned samples after the clean training set.

## Important Limitation

Landseer does not currently pass the original clean training split into `post_training`. To make `ft` and `sft` usable anyway, this wrapper reconstructs the clean portion by taking the first `N` samples of `data.npy`, where:

`N ~= total_train_size / (1 + poison_fraction)`

If you know the exact clean training size, pass `--clean-train-size` to avoid inference.

## Build

Run from this directory:

```bash
docker build -t backdoorbench-landseer-post:latest .
```

## Runtime Mounts

The container expects the BackdoorBench repository to be mounted at `/opt/backdoorbench`.
In Landseer this is done with `auxiliary_files`.

## Example Commands

```bash
python /app/main.py \
  --method ft \
  --output /output \
  --backdoorbench-root /opt/backdoorbench \
  --poison-fraction 0.05 \
  --target-class 0 \
  --trigger-size 3 \
  --trigger-value 1.0
```

```bash
python /app/main.py \
  --method sft \
  --output /output \
  --backdoorbench-root /opt/backdoorbench \
  --poison-fraction 0.05 \
  --target-class 0 \
  --trigger-size 3 \
  --trigger-value 1.0 \
  --lr-max1 0.1 \
  --lr-max2 0.001
```
