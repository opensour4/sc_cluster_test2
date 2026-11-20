"""Check fixed-input logits, training loss and output-head gradients against NVIDIA."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path

import numpy as np
import torch

from fixture import load_fixture
from infer import sample_count_tokens
from ppu_core import COUNT_SIZE, ModelInput, ResponseCrossEntropy, build_model, encode_counts
from ppu_core.interfaces import LossInputContext
from runtime import add_device_args, file_sha256, resolve_device
from model_config import CELLS, GENES, PARAMETER_COUNT, settings


ROOT = Path(__file__).resolve().parent
SEED = 3407
MASKS_PER_CELL = 32
PROBE_CELLS = (0, 9, 31, 48, 65, 89, 113, 127)
PROBE_SLOTS = (0, 6, 9, 13, 18, 24, 29, 31)
GRAD_COLUMNS = (1, 79, 203, 481)
SHAPES = {"logits": (8, 48), "loss": (), "head_gradient": (48, 4), "head_gradient_norm": ()}

# Bounds are (absolute, relative). The absolute floor handles entries close to zero.
# BF16 reductions and GEMMs have fewer significant bits than the full FP32 path.
TOLERANCES = {
    "fp32": {"logits": (3e-4, 5e-4), "loss": (1e-4, 1e-4),
             "head_gradient": (2e-5, 1e-3), "head_gradient_norm": (2e-4, 5e-4)},
    "bf16": {"logits": (0.035, 0.025), "loss": (0.01, 0.002),
             "head_gradient": (0.0007, 0.035), "head_gradient_norm": (0.003, 0.02)},
}


def fixed_case(data, layout, device):
    """Use all cells and genes; every cell has the same fixed mask density."""
    target = int(data["targets"][0])
    if target != 3:
        raise ValueError("Expected dummy target 3")
    b = encode_counts(data["eval_control"])
    y = encode_counts(data["eval_response"][0])
    tokens = np.full((CELLS, layout.block_size), layout.eos, dtype=np.int64)
    tokens[:, :layout.prefix] = layout.prefix_tokens(target)
    tokens[:, layout.prefix:layout.seq_length] = y
    mask = np.zeros_like(tokens, dtype=bool)
    rows = np.arange(CELLS)[:, None]
    genes = (113 * rows + 487 * np.arange(MASKS_PER_CELL)) % GENES
    mask[rows, layout.prefix + genes] = True
    ids = torch.as_tensor(tokens, device=device)
    selected = torch.as_tensor(mask, device=device)
    control = torch.as_tensor(b.astype(np.int64), device=device)
    group = torch.zeros(CELLS, dtype=torch.long, device=device)
    inputs = ModelInput(ids.masked_fill(selected, layout.mask_token), {
        "control_token_ids": control, "control_available": torch.ones_like(control, dtype=torch.bool),
        "set_id": group, "source_group_id": group, "mask_indices": selected,
    })
    probability = selected.float() * (MASKS_PER_CELL / GENES)
    ctx = LossInputContext(ids, layout.prefix, layout.seq_length, selected, probability)
    return inputs, ctx


def measure(data_path, device, precision):
    """Run the complete model forward and native masked CE, then differentiate the head."""
    config = settings()
    if (config["genes"], config["cell_set_size"], config["n_layer"], config["n_embd"],
            config["n_head"], config["intermediate_size"], config["attention_axes"],
            config["attn_module"]) != (18533, 128, 13, 640, 10, 2560, ["L", "N"] * 6 + ["L"], "SDPA"):
        raise ValueError("Calibration requires the complete model architecture")
    # Seed, initialize and construct all stochastic weights on the host before moving them.
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(SEED)
        with torch.device("cpu"):
            model = build_model(config)
    finally:
        torch.set_num_threads(previous_threads)
    count = sum(p.numel() for p in model.parameters())
    if count != PARAMETER_COUNT or count != 123_543_040 or COUNT_SIZE != 48:
        raise ValueError("Unexpected model size or output vocabulary")
    model.requires_grad_(False)
    model.lm_head.weight.requires_grad_(True)
    model.to(device).eval()
    if model.lm_head.weight.device != device:
        raise RuntimeError("Model was not moved to the requested device")
    inputs, ctx = fixed_case(load_fixture(data_path), model.layout, device)
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with amp:
        output = model(inputs)
        dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if (output.logits.device != device or output.logits.dtype != dtype
                or output.logits.shape != (CELLS * MASKS_PER_CELL, COUNT_SIZE)):
            raise RuntimeError("Fixed-case logits have unexpected shape, device or precision")
        loss = ResponseCrossEntropy(model.layout).compute(ctx, output).loss
    loss.backward()
    grad = model.lm_head.weight.grad
    if grad is None or grad.device != device or grad.shape != (48, 640):
        raise RuntimeError("Missing or misplaced output-head gradient")
    for tensor in (output.logits, loss, grad):
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("Nonfinite fixed-case logits, loss or gradients")
    if not grad.any():
        raise RuntimeError("Fixed-case head gradient is zero")
    probe_rows = [row * MASKS_PER_CELL + slot for row, slot in zip(PROBE_CELLS, PROBE_SLOTS)]
    tensors = {"logits": output.logits[probe_rows], "loss": loss,
               "head_gradient": grad[:, list(GRAD_COLUMNS)], "head_gradient_norm": grad.norm()}
    values = {name: tensor.detach().float().cpu().tolist() for name, tensor in tensors.items()}
    identity = {"case_version": 1, "fixture_sha256": file_sha256(data_path), "seed": SEED,
                "model": config, "parameters": count, "target": 3, "mask_count": int(ctx.mask_indices.sum()),
                "probe_cells": list(PROBE_CELLS), "sorted_mask_slots": list(PROBE_SLOTS),
                "head_gradient_columns": list(GRAD_COLUMNS)}
    return identity, values


def compare(observed, expected, precision):
    """Check every value with its absolute-plus-relative error bound."""
    if set(observed) != set(SHAPES) or set(expected) != set(SHAPES):
        raise ValueError("Reference and measurement must contain all four numerical checks")
    result = {}
    for name, shape in SHAPES.items():
        reference = np.asarray(expected[name], dtype=np.float64)
        measured = np.asarray(observed[name], dtype=np.float64)
        if reference.shape != shape or measured.shape != shape:
            raise ValueError(f"{name}: expected shape {shape}")
        if not np.isfinite(reference).all() or not np.isfinite(measured).all():
            raise ValueError(f"{name}: nonfinite reference or measurement")
        absolute, relative = TOLERANCES[precision][name]
        error = np.abs(measured - reference)
        limit = absolute + relative * np.abs(reference)
        ratio = error / limit
        worst = int(ratio.argmax())
        result[name] = {"pass": bool(np.all(ratio <= 1)), "max_abs_error": float(error.max()),
                        "worst_error": float(error.flat[worst]), "allowed_at_worst": float(limit.flat[worst]),
                        "max_error_to_limit": float(ratio.flat[worst])}
    return result


def check_count_sampling(device):
    """Exercise the inference Gumbel-max operation on fixed FP64 draws."""
    logits = torch.zeros((4, COUNT_SIZE), dtype=torch.float64, device=device)
    uniform = torch.full_like(logits, 0.5)
    # First and last classes, a lower-logit class that wins after noise,
    # and two high quantiles that would both round to 1.0 in float32.
    logits[0, 0], logits[1, -1] = 3.0, 3.0
    logits[2, 15], uniform[2, 12] = 1.0, 0.9
    uniform[3, 30], uniform[3, 31] = 1.0 - 2**-30, 1.0 - 2**-45
    expected = [5, 52, 17, 36]
    tokens = sample_count_tokens(logits, uniform)
    if tokens.device != device or tokens.dtype != torch.long:
        raise RuntimeError("Sampled count token IDs have an unexpected device or dtype")
    actual = tokens.cpu().tolist()
    return {"pass": actual == expected, "expected": expected, "observed": actual}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_args(parser)
    parser.add_argument("--data", type=Path, default=ROOT / "data/dummy.npz")
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device = resolve_device(args.device, args.plugin)
    reference = json.loads((ROOT / "references/numeric.json").read_text())
    if file_sha256(args.data) != reference["identity"]["fixture_sha256"]:
        raise ValueError("Fixture SHA-256 differs from the numerical reference")
    identity, measured = measure(args.data, device, args.precision)
    if reference["identity"] != identity:
        raise ValueError("Reference identity differs from the measured case")
    checks = compare(measured, reference["values"][args.precision], args.precision)
    checks["count_sampling"] = check_count_sampling(device)
    passed = all(check["pass"] for check in checks.values())
    print(json.dumps({"stage": "numeric_verification", "pass": passed, "device": str(device),
                      "precision": args.precision, "checks": checks}, allow_nan=False), flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
