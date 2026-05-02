#!/usr/bin/env python3
import argparse
import importlib.util
import json
import logging
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

LOG = logging.getLogger("backdoorbench_post")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Landseer wrapper for BackdoorBench ft/sft post-training defenses")
    parser.add_argument("--output", required=True, help="Output directory mounted by Landseer")
    parser.add_argument("--method", choices=["ft", "sft"], required=True, help="BackdoorBench defense to run")
    parser.add_argument("--backdoorbench-root", default="/opt/backdoorbench", help="Mounted BackdoorBench repository path")
    parser.add_argument("--model-config", default="/app/config_model.py", help="Mounted Landseer model config path")
    parser.add_argument("--work-dir", default="/tmp/backdoorbench-landseer", help="Scratch area inside container")
    parser.add_argument("--dataset", default="cifar10", help="Dataset name expected by BackdoorBench")
    parser.add_argument("--device", default=None, help="Torch device override, defaults to cuda when available")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--ratio", type=float, default=None, help="Fraction of clean train data used by ft/sft")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--poison-fraction", type=float, default=0.05, help="Landseer poison fraction used to infer clean train size")
    parser.add_argument("--target-class", type=int, default=0, help="Target label used to build poisoned test set for ft/sft evaluation")
    parser.add_argument("--trigger-size", type=int, default=3)
    parser.add_argument("--trigger-value", type=float, default=1.0)
    parser.add_argument("--clean-train-size", type=int, default=None, help="Optional explicit clean train size before Landseer appended poisoned samples")
    parser.add_argument("--model-name", default="landseer_model", help="Logical model name recorded in BackdoorBench outputs")
    parser.add_argument("--lr-base", type=float, default=None)
    parser.add_argument("--lr-max1", type=float, default=None)
    parser.add_argument("--lr-max2", type=float, default=None)
    parser.add_argument("--phase1-epochs", type=int, default=None)
    parser.add_argument("--cycle-len", type=int, default=None)
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def load_config_callable(config_path: Path):
    spec = importlib.util.spec_from_file_location("landseer_config_model", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load model config from {config_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "config"):
        raise RuntimeError(f"Model config {config_path} does not expose config()")
    return module.config


def normalize_to_uint8(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images)
    if arr.ndim != 4:
        raise ValueError(f"Expected 4D image tensor, got shape {arr.shape}")
    if arr.shape[1] in (1, 3):
        arr = np.transpose(arr, (0, 2, 3, 1))
    arr = arr.astype(np.float32)
    if arr.min() < 0:
        arr = (arr + 1.0) / 2.0
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


class NumpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray):
        self.images = normalize_to_uint8(images)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        image = self.images[index]
        if image.shape[-1] == 1:
            pil = Image.fromarray(image.squeeze(-1), mode="L")
        else:
            pil = Image.fromarray(image)
        return pil, int(self.labels[index])


def build_backdoor_test_dataset(
    test_images: np.ndarray,
    test_labels: np.ndarray,
    trigger_size: int,
    trigger_value: float,
    target_class: int,
):
    from utils.bd_dataset_v2 import prepro_cls_DatasetBD_v2

    clean_test_dataset = NumpyImageDataset(test_images, test_labels)
    poisoned_test_images = apply_trigger(test_images, trigger_size, trigger_value)

    def image_transform(_img, target=None, image_serial_id=None):
        _ = target
        if image_serial_id is None:
            raise ValueError("image_serial_id is required for BackdoorBench bd_test construction")
        poisoned_image = normalize_to_uint8(poisoned_test_images[image_serial_id:image_serial_id + 1])[0]
        if poisoned_image.shape[-1] == 1:
            return Image.fromarray(poisoned_image.squeeze(-1), mode="L")
        return Image.fromarray(poisoned_image)

    def label_transform(_label):
        return int(target_class)

    return prepro_cls_DatasetBD_v2(
        clean_test_dataset,
        poison_indicator=np.ones(len(test_labels), dtype=np.int64),
        bd_image_pre_transform=image_transform,
        bd_label_pre_transform=label_transform,
    )


def apply_trigger(images: np.ndarray, trigger_size: int, trigger_value: float) -> np.ndarray:
    poisoned = np.array(images, copy=True)
    value = trigger_value
    if poisoned.dtype.kind in {"u", "i"}:
        value = np.clip(trigger_value * 255.0 if trigger_value <= 1.0 else trigger_value, 0, 255)
    if poisoned.shape[1] in (1, 3):
        poisoned[:, :, -trigger_size:, -trigger_size:] = value
    else:
        poisoned[:, -trigger_size:, -trigger_size:, :] = value
    return poisoned


def infer_clean_train_size(total_size: int, poison_fraction: float) -> int:
    if poison_fraction <= 0:
        return total_size
    inferred = round(total_size / (1.0 + poison_fraction))
    return min(max(inferred, 1), total_size)


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required input file: {path}")
    return np.load(path)


def prepare_backdoorbench_result(args: argparse.Namespace, backdoorbench_root: Path):
    data_dir = Path("/data")
    train_images = load_array(data_dir / "data.npy")
    train_labels = load_array(data_dir / "labels.npy")
    test_images = load_array(data_dir / "test_data.npy")
    test_labels = load_array(data_dir / "test_labels.npy")

    clean_train_size = args.clean_train_size or infer_clean_train_size(len(train_labels), args.poison_fraction)
    if clean_train_size > len(train_labels):
        raise ValueError(f"clean_train_size {clean_train_size} exceeds total train size {len(train_labels)}")

    clean_train_images = train_images[:clean_train_size]
    clean_train_labels = train_labels[:clean_train_size]
    from utils.bd_dataset_v2 import dataset_wrapper_with_transform

    result = {
        "model": torch.load(data_dir / "model.pt", map_location="cpu"),
        "clean_train": dataset_wrapper_with_transform(NumpyImageDataset(clean_train_images, clean_train_labels)),
        "clean_test": dataset_wrapper_with_transform(NumpyImageDataset(test_images, test_labels)),
        "bd_test": dataset_wrapper_with_transform(
            build_backdoor_test_dataset(
                test_images=test_images,
                test_labels=test_labels,
                trigger_size=args.trigger_size,
                trigger_value=args.trigger_value,
                target_class=args.target_class,
            )
        ),
    }

    metadata = {
        "train_total": int(len(train_labels)),
        "clean_train_size": int(clean_train_size),
        "test_size": int(len(test_labels)),
        "poison_fraction": args.poison_fraction,
        "target_class": args.target_class,
        "trigger_size": args.trigger_size,
        "trigger_value": args.trigger_value,
        "backdoorbench_root": str(backdoorbench_root),
    }
    return result, metadata


def build_landseer_model_builder(config_path: Path):
    config_fn = load_config_callable(config_path)

    def _builder(model_name: str, num_classes: int, image_size: int = 32, **kwargs):
        _ = model_name, num_classes, image_size, kwargs
        model = config_fn()
        if not isinstance(model, torch.nn.Module):
            raise TypeError("config() must return a torch.nn.Module")
        return model

    return _builder


def make_defense_args(cli_args: argparse.Namespace, yaml_path: Path, device: str) -> argparse.Namespace:
    ns = argparse.Namespace(
        yaml_path=str(yaml_path),
        result_file=None,
        device=device,
        dataset=cli_args.dataset,
        dataset_path="/data",
        model=cli_args.model_name,
        random_seed=cli_args.random_seed,
        epochs=cli_args.epochs,
        batch_size=cli_args.batch_size,
        num_workers=cli_args.num_workers,
        ratio=cli_args.ratio,
        lr_base=cli_args.lr_base,
        lr_max1=cli_args.lr_max1,
        lr_max2=cli_args.lr_max2,
        phase1_epochs=cli_args.phase1_epochs,
        cycle_len=cli_args.cycle_len,
    )
    return ns


def run_defense(cli_args: argparse.Namespace, work_root: Path, result_payload: dict) -> Path:
    backdoorbench_root = Path(cli_args.backdoorbench_root).resolve()
    if not backdoorbench_root.exists():
        raise FileNotFoundError(f"BackdoorBench root not found: {backdoorbench_root}")

    os.chdir(backdoorbench_root)
    sys.path.insert(0, str(backdoorbench_root))

    device = cli_args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if cli_args.method == "ft":
        from defense import ft as defense_module
        defense_class = defense_module.ft
    else:
        from defense import sft as defense_module
        defense_class = defense_module.sft

    defense_module.generate_cls_model = build_landseer_model_builder(Path(cli_args.model_config))

    yaml_path = backdoorbench_root / "config" / "defense" / "ft" / "config.yaml"
    defense_args = make_defense_args(cli_args, yaml_path, device)
    defender = defense_class(defense_args)

    save_root = work_root / "record" / "landseer_input" / "defense" / cli_args.method
    checkpoint_root = save_root / "checkpoint"
    log_root = save_root / "log"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    defender.args.save_path = str(save_root) + "/"
    defender.args.checkpoint_save = str(checkpoint_root) + "/"
    defender.args.log = str(log_root) + "/"
    defender.args.device = device
    if getattr(defender.args, "num_workers", None) is not None:
        defender.args.num_workers = int(defender.args.num_workers)

    defender.result = result_payload
    defense_module.args = defender.args
    defender.set_logger()
    LOG.info("Running BackdoorBench %s with save_path=%s", cli_args.method, defender.args.save_path)
    defender.mitigation()
    return save_root / "defense_result.pt"


def extract_state_dict(defense_result_path: Path) -> dict:
    payload = torch.load(defense_result_path, map_location="cpu")
    if isinstance(payload, dict) and "model" in payload:
        return payload["model"]
    raise RuntimeError(f"Unexpected defense result payload at {defense_result_path}")


def main() -> None:
    configure_logging()
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_root = Path(args.work_dir).resolve()
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True, exist_ok=True)

    backdoorbench_root = Path(args.backdoorbench_root).resolve()
    if str(backdoorbench_root) not in sys.path:
        sys.path.insert(0, str(backdoorbench_root))
    result_payload, metadata = prepare_backdoorbench_result(args, backdoorbench_root)
    defense_result_path = run_defense(args, work_root, result_payload)
    model_state = extract_state_dict(defense_result_path)

    torch.save(model_state, output_dir / "model.pt")
    shutil.copy2(defense_result_path, output_dir / "defense_result.pt")
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "method": args.method,
                "device": args.device or ("cuda" if torch.cuda.is_available() else "cpu"),
                **metadata,
            },
            handle,
            indent=2,
        )
    LOG.info("Saved defended model to %s", output_dir / "model.pt")


if __name__ == "__main__":
    main()
