"""Score a complete dummy prediction on CPU using count and ranking metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from fixture import load_fixture
from ppu_core import COUNT_SIZE, COUNT_START, decode_counts, encode_counts
from runtime import atomic_output, file_sha256


def response_metrics(predicted_mean, true_mean, control_mean, target):
    """Calculate off-own-target normalized response error and companion errors."""
    pred, truth, ctrl = [np.asarray(x, dtype=np.float64) for x in (predicted_mean, true_mean, control_mean)]
    if (pred.ndim != 1 or truth.shape != pred.shape or ctrl.shape != pred.shape
            or not 0 <= target < len(pred)
            or any(not np.isfinite(x).all() or (x < 0).any() for x in (pred, truth, ctrl))):
        raise ValueError("Invalid target or population means")
    off_target = np.arange(len(pred)) != target
    error = np.log1p(pred) - np.log1p(truth)
    response = np.log1p(truth) - np.log1p(ctrl)
    energy = np.square(response[off_target]).sum()
    if not np.isfinite(energy) or energy <= 0:
        raise ValueError("Expected positive off-target response energy")
    return {
        "nre": float(np.square(error[off_target]).sum() / energy),
        "nre_full": float(np.square(error).sum() / np.square(response).sum()),
        "response_energy": float(energy),
        "log_mean_mse": float(np.square(error[off_target]).mean()),
        "log_mean_mse_full": float(np.square(error).mean()),
        "raw_mean_mae": float(np.abs(pred[off_target] - truth[off_target]).mean()),
        "raw_mean_mae_full": float(np.abs(pred - truth).mean()),
        "target_mean": float(pred[target]),
        "target_true_mean": float(truth[target]),
    }


def pds_response_rank(predicted, truths, control, targets):
    """Rank each matching response against other targets on the common gene axis."""
    axis = np.ones(truths.shape[1], dtype=bool)
    axis[targets] = False
    if len(targets) < 2 or not axis.any():
        raise ValueError("PDS requires two distinct targets and a remaining comparison gene")
    normalized = []
    zero = []
    for means in (predicted, truths):
        response = np.log1p(means[:, axis]) - np.log1p(control[axis])
        norm = np.linalg.norm(response, axis=1, keepdims=True)
        zero.append(int((norm == 0).sum()))
        normalized.append(np.divide(response, norm, out=np.zeros_like(response), where=norm != 0))
    scores = np.einsum("ig,jg->ij", normalized[0], normalized[1])
    correct = scores.diagonal()[:, None]
    ranking = (scores < correct).astype(float) + .5 * (scores == correct)
    np.fill_diagonal(ranking, 0)
    per_target = ranking.sum(axis=1) / (len(targets) - 1)
    return {"mean": float(per_target.mean()), "per_target": per_target.tolist(),
            "zero_prediction_vectors": zero[0], "zero_true_vectors": zero[1], "gene_count": int(axis.sum())}


def check_sampled_counts(counts, targets):
    """Detect constant or copied outputs in independently sampled full-length cells."""
    seen_cells = {}
    for index, target in enumerate(targets):
        block = counts[index]
        if block.min() == block.max():
            raise ValueError(f"Target {target} has constant predictions for all cells and genes")
        for previous in range(index):
            if np.array_equal(block, counts[previous]):
                raise ValueError(f"Target {target} repeats the predictions for target {targets[previous]}")
        for cell, row in enumerate(block):
            profile = row.tobytes()
            if profile in seen_cells:
                old_target, old_cell = seen_cells[profile]
                raise ValueError(f"Target {target} cell {cell} repeats the complete prediction "
                                 f"for target {old_target} cell {old_cell}")
            seen_cells[profile] = (target, cell)


def evaluate(data_path, prediction_path):
    fixture = load_fixture(data_path)
    true_means = np.asarray(fixture["true_mean"], dtype=np.float64)
    control = np.asarray(fixture["control_mean"], dtype=np.float64)
    observations = fixture["eval_response"]
    targets = fixture["targets"]
    with np.load(prediction_path, allow_pickle=False) as prediction:
        counts = prediction["counts"]
        identifiers = prediction["targets"]
        if identifiers.dtype.kind not in "iu" or not np.array_equal(identifiers, targets):
            raise ValueError("Predictions must cover the fixture targets in order")
        keys = ("data_sha256", "checkpoint_sha256", "checkpoint_step", "weights", "device", "precision", "steps", "seed")
        binding = {key: prediction[key].item() for key in keys}
    if binding["data_sha256"] != file_sha256(data_path):
        raise ValueError("Predictions were produced with a different fixture")
    if (re.fullmatch(r"[a-f0-9]{64}", str(binding["checkpoint_sha256"])) is None
            or binding["checkpoint_step"] < 1 or binding["steps"] < 1
            or binding["weights"] != "ema-0.05" or binding["precision"] not in ("fp32", "bf16")
            or not binding["device"]):
        raise ValueError("Invalid inference metadata")
    if counts.dtype != np.uint32 or counts.shape != observations.shape or len(targets) != len(counts):
        raise ValueError("Predictions must be complete target x cell x gene uint32 counts")
    valid_counts = decode_counts(np.arange(COUNT_START, COUNT_START + COUNT_SIZE, dtype=np.int64))
    if not np.isin(counts, valid_counts).all():
        raise ValueError("Predicted counts exceed the model's decoded count support")
    check_sampled_counts(counts, targets)

    means = counts.mean(axis=1)
    per_target = []
    for index, target in enumerate(targets.tolist()):
        truth = true_means[index]
        reference = observations[index]
        item = response_metrics(means[index], truth, control, target)
        quantized = decode_counts(encode_counts(reference))
        item.update(target=target, raw_reference_nre=response_metrics(reference.mean(0), truth, control, target)["nre"],
                    quantized_reference_nre=response_metrics(quantized.mean(0), truth, control, target)["nre"],
                    pred_zero_fraction=float((counts[index] == 0).mean()),
                    reference_zero_fraction=float((reference == 0).mean()),
                    pred_cap_fraction=float((counts[index] == valid_counts[-1]).mean()),
                    marginal_variance_mae=float(np.abs(counts[index].var(0) - reference.var(0)).mean()))
        per_target.append(item)
    return {
        "evaluation_device": "cpu", "inference": binding, "shape": list(counts.shape),
        "macro_nre": float(np.mean([row["nre"] for row in per_target])),
        "raw_reference_macro_nre": float(np.mean([row["raw_reference_nre"] for row in per_target])),
        "quantized_reference_macro_nre": float(np.mean([row["quantized_reference_nre"] for row in per_target])),
        "pds_response_rank": pds_response_rank(means, true_means, control, targets),
        "per_target": per_target,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/dummy.npz"))
    parser.add_argument("--predictions", type=Path, default=Path("runs/predictions.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/metrics.json"))
    args = parser.parse_args()
    if args.output.resolve() in {args.data.resolve(), args.predictions.resolve()}:
        parser.error("The output path must differ from both inputs")
    report = evaluate(args.data, args.predictions)
    payload = (json.dumps(report, allow_nan=False, indent=2) + "\n").encode()
    with atomic_output(args.output) as stream:
        stream.write(payload)
    print(json.dumps({"stage": "evaluation", "macro_nre": report["macro_nre"],
                      "pds_response_rank": report["pds_response_rank"]["mean"],
                      "shape": report["shape"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
