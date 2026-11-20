"""Verify a completed run and compare its summary with matching NVIDIA observations."""

import argparse
import json
import math
from pathlib import Path
from pickle import UnpicklingError
from zipfile import BadZipFile

import torch

from evaluate import evaluate
from ppu_core import build_model
from runtime import file_sha256
from model_config import CELLS, PARAMETER_COUNT, settings

ROOT = Path(__file__).resolve().parent
BATCH_KEYS = ("micro_groups_per_rank", "micro_cells_per_rank",
              "global_groups_per_update", "global_cells_per_update")


def compare(actual, expected, path):
    """Compare the complete record, allowing rounding only for finite floating-point metrics."""
    if isinstance(expected, dict) and isinstance(actual, dict) and actual.keys() == expected.keys():
        for key, value in expected.items():
            compare(actual[key], value, f"{path}.{key}")
    elif isinstance(expected, list) and isinstance(actual, list) and len(actual) == len(expected):
        for i, value in enumerate(expected):
            compare(actual[i], value, f"{path}[{i}]")
    elif type(expected) is float and type(actual) in (int, float):
        if not (math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12)):
            raise ValueError(f"{path} differs: {actual} vs {expected}")
    elif type(actual) is not type(expected) or actual != expected:
        raise ValueError(f"{path} differs")


def training_records(path):
    """Read torchrun JSON records, including multiple ranks writing on the same line."""
    decoder = json.JSONDecoder()
    with path.open() as stream:
        for line in stream:
            start = line.find("{")
            while start >= 0:
                try:
                    record, end = decoder.raw_decode(line, start)
                except ValueError:
                    end = start + 1
                else:
                    if isinstance(record, dict) and "stage" in record:
                        yield record
                start = line.find("{", end)


def check_training(checkpoint, log):
    config = settings()
    compare(checkpoint["model_settings"], config, "model_settings")
    compare(checkpoint["parameter_count"], PARAMETER_COUNT, "parameter_count")
    # A meta model supplies exact keys, shapes and dtypes without allocating weight storage.
    with torch.device("meta"):
        expected = build_model(config).state_dict()
    for key in ("model", "ema"):
        state = checkpoint[key]
        if state.keys() != expected.keys() or any(
            state[k].shape != t.shape or state[k].dtype != t.dtype for k, t in expected.items()
        ):
            raise ValueError(f"{key} weights differ from the model layout")

    for key in ("step", "accumulation", "rank_count", "training_seed", "warmup_steps"):
        minimum = 0 if key in ("training_seed", "warmup_steps") else 1
        if type(checkpoint[key]) is not int or checkpoint[key] < minimum:
            raise ValueError(f"Invalid training {key}")
    if (checkpoint["precision"] not in ("bf16", "fp32") or not checkpoint["device"]
            or any(not math.isfinite(checkpoint[k]) or checkpoint[k] < 0
                   for k in ("validation_loss", "head_update_max", "grad_norm"))
            or checkpoint["head_update_max"] == 0):
        raise ValueError("Invalid training device, precision or update statistics")

    world = checkpoint["rank_count"]
    micro = checkpoint["micro_groups_per_rank"]
    if type(micro) is not int or micro not in (1, 2) or CELLS != 128:
        raise ValueError("Expected one or two groups of 128 cells per rank per microbatch")
    groups = micro * world * checkpoint["accumulation"]
    batch = dict(micro_groups_per_rank=micro, micro_cells_per_rank=micro * CELLS,
                 global_groups_per_update=groups, global_cells_per_update=groups * CELLS)
    for key, value in batch.items():
        compare(checkpoint[key], value, f"checkpoint.{key}")
    ranks = set()
    train_steps = 0
    final_step = receipt = None
    for entry in training_records(log):
        if entry["stage"] in ("device", "train"):
            for key, value in batch.items():
                compare(entry[key], value, f"{entry['stage']}.{key}")
        if entry["stage"] == "train":
            train_steps += 1
            compare(entry["step"], train_steps, "train.step")
            compare(entry["global_cells"], groups * CELLS, "train.global_cells")
            if not math.isfinite(entry["grad_norm"]) or entry["grad_norm"] <= 0:
                raise ValueError(f"Invalid gradient norm at training step {train_steps}")
            final_step = entry
        elif entry["stage"] == "device":
            compare(entry["world_size"], world, "device.world_size")
            compare(entry["parameter_count"], PARAMETER_COUNT, "device.parameter_count")
            compare(entry["model_settings"], config, "device.model_settings")
            if entry["rank"] == 0:
                compare(entry["device"], checkpoint["device"], "rank 0 device")
            ranks.add(entry["rank"])
        elif entry["stage"] == "checkpoint":
            receipt = entry
    if sorted(ranks) != list(range(world)) or train_steps != checkpoint["step"] or receipt is None:
        raise ValueError("Missing training rank, step or checkpoint record")
    for key, value in checkpoint.items():
        if key not in ("model", "ema"):
            compare(receipt[key], value, f"checkpoint log.{key}")
    compare(final_step["grad_norm"], checkpoint["grad_norm"], "final step.grad_norm")
    if not math.isfinite(final_step["loss"]) or final_step["loss"] < 0:
        raise ValueError("Invalid final training loss")
    return final_step["loss"]


def check_run(run, data):
    checkpoint_path = run / "checkpoint.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)
    train_loss = check_training(ckpt, run / "train.log")
    report = evaluate(data, run / "predictions.npz")
    compare(json.loads((run / "metrics.json").read_text()), report, "metrics")
    inference = report["inference"]
    compare(inference["data_sha256"], ckpt["data_sha256"], "fixture fingerprint")
    compare(inference["checkpoint_sha256"], file_sha256(checkpoint_path), "checkpoint fingerprint")
    compare(inference["checkpoint_step"], ckpt["step"], "prediction checkpoint step")

    keys = ("rank_count", "step", "accumulation", "warmup_steps", "training_seed", "precision", *BATCH_KEYS)
    case = {k: ckpt[k] for k in keys}
    case.update(infer_steps=inference["steps"], infer_seed=inference["seed"])
    scores = dict(train_loss=train_loss, validation_loss=ckpt["validation_loss"], macro_nre=report["macro_nre"])
    refs = json.loads((ROOT / "references/runs.json").read_text())
    delta = None
    if refs["data_sha256"] == ckpt["data_sha256"] and inference["precision"] == ckpt["precision"]:
        for row in refs["runs"]:
            if all(row.get(k) == v for k, v in case.items()):
                delta = {k: v - row[k] for k, v in scores.items()}
                break
    return dict(shape=report["shape"], **case, **scores, delta_vs_nvidia=delta)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dummy.npz")
    args = parser.parse_args()
    try:
        result = check_run(args.run_dir, args.data)
    except (OSError, RuntimeError, ValueError, TypeError, KeyError, EOFError, UnpicklingError, BadZipFile) as exc:
        print(json.dumps({"stage": "run_verification", "pass": False, "error": str(exc)}))
        return 1
    print(json.dumps({"stage": "run_verification", "pass": True, **result}, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
