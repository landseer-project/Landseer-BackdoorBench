#!/usr/bin/env python3
import argparse
import csv
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload a Landseer run directory to Weights & Biases")
    parser.add_argument("--run-dir", required=True, help="Landseer run directory, e.g. results/<pipeline_id>/<timestamp>")
    parser.add_argument("--project", required=True, help="wandb project name")
    parser.add_argument("--entity", default=None, help="wandb entity/team")
    parser.add_argument("--name", default=None, help="wandb run name override")
    parser.add_argument("--job-type", default="landseer-eval")
    parser.add_argument("--group", default=None)
    parser.add_argument("--tags", nargs="*", default=[])
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value: str):
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None


def summarize(rows: List[Dict[str, str]]) -> Dict[str, float]:
    if not rows:
        return {}
    metrics = {
        "acc_test_clean": [],
        "pgd_acc": [],
        "ood_auc": [],
        "asr": [],
        "mia_auc": [],
        "eps_estimate": [],
        "total_duration": [],
    }
    for row in rows:
        for key in list(metrics.keys()):
            val = maybe_float(row.get(key))
            if val is not None and val >= 0:
                metrics[key].append(val)
    summary = {}
    for key, values in metrics.items():
        if values:
            summary[f"summary/{key}_mean"] = sum(values) / len(values)
            summary[f"summary/{key}_best"] = max(values) if key != "asr" else min(values)
    summary["summary/num_combinations"] = len(rows)
    return summary


def _clear_stale_wandb_service_env() -> None:
    os.environ.pop("WANDB_SERVICE", None)


def _diagnose_wandb_core_startup(wandb) -> str:
    try:
        from wandb.util import get_core_path
    except Exception as exc:
        return f"Unable to import wandb core path helper: {exc}"

    try:
        core_path = get_core_path()
    except Exception as exc:
        return f"Unable to locate wandb-core binary: {exc}"

    env = os.environ.copy()
    cache_dir = Path(env.get("WANDB_CACHE_DIR", str(Path.home() / ".cache" / "wandb")))
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    with tempfile.TemporaryDirectory(prefix="wandb-core-check-") as tmpdir:
        port_file = Path(tmpdir) / f"port-{os.getpid()}.txt"
        cmd = [core_path, "--port-filename", str(port_file), "--pid", str(os.getpid())]
        try:
            proc = subprocess.run(
                cmd,
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = (exc.stdout or "").strip()
            stderr = (exc.stderr or "").strip()
            details = stderr or stdout or "wandb-core timed out without output"
            return f"wandb-core probe timed out after 10s. Details: {details}"
        except Exception as exc:
            return f"wandb-core probe failed to execute: {exc}"

    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    details = stderr or stdout or "wandb-core exited without any stdout/stderr"
    return f"wandb-core exited with code {proc.returncode}. Details: {details}"


def _init_wandb_run(wandb, *, project, entity, name, job_type, group, tags, config):
    settings = wandb.Settings(
        x_service_wait=60,
        init_timeout=120,
        x_disable_stats=True,
        x_disable_meta=True,
        console="off",
    )
    last_exc = None

    for transport in (None, "tcp"):
        if transport is None:
            os.environ.pop("WANDB_SERVICE_TRANSPORT", None)
        else:
            os.environ["WANDB_SERVICE_TRANSPORT"] = transport

        _clear_stale_wandb_service_env()

        try:
            return wandb.init(
                project=project,
                entity=entity,
                name=name,
                job_type=job_type,
                group=group,
                tags=tags,
                config=config,
                settings=settings,
            )
        except Exception as exc:
            last_exc = exc
            if exc.__class__.__name__ != "ServicePollForTokenError":
                raise
            time.sleep(2)

    diagnostic = _diagnose_wandb_core_startup(wandb)
    raise RuntimeError(
        "wandb.init failed after retrying default and tcp service transport. "
        f"Last error: {last_exc}. Diagnostic: {diagnostic}"
    ) from last_exc


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    combos_csv = run_dir / "results_combinations.csv"
    tools_csv = run_dir / "results_tools.csv"
    mappings_json = run_dir / "artifact_mappings.json"

    rows = read_csv_rows(combos_csv)
    tool_rows = read_csv_rows(tools_csv)

    try:
        import wandb
    except Exception as exc:
        raise SystemExit(f"wandb import failed: {exc}")

    pipeline_id = run_dir.parent.name
    timestamp = run_dir.name
    run_name = args.name or f"landseer-{pipeline_id}-{timestamp}"

    run = _init_wandb_run(
        wandb,
        project=args.project,
        entity=args.entity,
        name=run_name,
        job_type=args.job_type,
        group=args.group,
        tags=args.tags,
        config={
            "pipeline_id": pipeline_id,
            "timestamp": timestamp,
            "run_dir": str(run_dir),
        },
    )

    summary = summarize(rows)
    if summary:
        wandb.log(summary)

    if rows:
        combo_table = wandb.Table(columns=list(rows[0].keys()))
        for row in rows:
            combo_table.add_data(*[row.get(col, "") for col in combo_table.columns])
        wandb.log({"results/combinations": combo_table})

    if tool_rows:
        tool_table = wandb.Table(columns=list(tool_rows[0].keys()))
        for row in tool_rows:
            tool_table.add_data(*[row.get(col, "") for col in tool_table.columns])
        wandb.log({"results/tools": tool_table})

    artifact = wandb.Artifact(f"landseer-run-{pipeline_id}-{timestamp}", type="landseer-results")
    for path in [combos_csv, tools_csv, mappings_json]:
        if path.exists():
            artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)

    if mappings_json.exists():
        with mappings_json.open("r", encoding="utf-8") as handle:
            mappings = json.load(handle)
        run.summary["summary/artifact_mapping_keys"] = len(mappings)

    run.finish()


if __name__ == "__main__":
    main()
