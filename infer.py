"""Generate counts from a trained EMA checkpoint and held-out control cells."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path

import numpy as np
import torch

from fixture import load_fixture
from ppu_core import COUNT_SIZE, COUNT_START, ModelInput, build_model, decode_counts, encode_counts
from ppu_core.identities import public_seed
from runtime import add_device_args, atomic_output, file_sha256, resolve_device
from model_config import PARAMETER_COUNT, settings


def unmask_schedule(steps, genes, device):
    """Allocate whole genes per step using a cosine schedule and the largest-remainder rule."""
    if steps <= 0:
        raise ValueError("--steps must be positive")
    if steps >= genes:
        schedule = torch.zeros(steps, dtype=torch.long, device=device)
        schedule[:genes] = 1
        return schedule
    times = torch.cos(torch.linspace(0, math.pi / 2, steps + 1, device=device)).clamp(0, 1)
    fractions = times[:-1] - times[1:]
    ideal = (fractions / fractions.sum()) * genes
    schedule = ideal.floor().long()
    remainder = int(genes - schedule.sum())
    if remainder:
        schedule[(ideal - schedule.float()).topk(remainder).indices] += 1
    # Borrow from the earliest current maximum for each empty step.
    empty = schedule == 0
    deficit = int(empty.sum())
    schedule[empty] = 1
    for _ in range(deficit):
        donor = torch.where(schedule == schedule.max())[0][0]
        schedule[donor] -= 1
    if int(schedule.sum()) != genes or not torch.all(schedule >= 1):
        raise RuntimeError("Cosine schedule did not allocate the complete gene axis")
    return schedule


def group_seed(seed, target, group, group_size):
    """Give each synthetic target and control set a repeatable independent stream."""
    return public_seed("cluster-sampling", seed, 0, "dummy", target, group, group_size)


def sample_count_tokens(logits, uniform):
    """Draw count token IDs by applying Gumbel-max to a row of count logits."""
    gumbel = -torch.log(-torch.log(uniform.clamp_min(torch.finfo(torch.float64).tiny)))
    return (logits + gumbel).argmax(dim=-1) + COUNT_START


@torch.no_grad()
def sample_group(model, controls, target, *, device, steps, precision, seed):
    """Iteratively reveal Y count tokens from the prefix and one observed control set."""
    layout = model.layout
    cells = layout.group_size
    if model.training or controls.shape != (cells, layout.genes):
        raise ValueError("Sampling requires an eval-mode model and a complete control set")
    control_ids = torch.as_tensor(encode_counts(controls).astype(np.int64), device=device)
    y = torch.full((cells, layout.block_size), layout.eos, dtype=torch.long, device=device)
    y[:, :layout.prefix] = torch.as_tensor(layout.prefix_tokens(target), device=device)
    gene_span = slice(layout.prefix, layout.seq_length)
    y[:, gene_span] = layout.mask_token
    group_ids = torch.zeros(cells, dtype=torch.long, device=device)
    metadata = {
        "control_token_ids": control_ids,
        "control_available": torch.ones_like(control_ids, dtype=torch.bool),
        "set_id": group_ids,
        "source_group_id": group_ids.clone(),
    }
    rng = torch.Generator(device=y.device).manual_seed(seed)
    lookup = torch.empty_like(y)
    schedule = unmask_schedule(steps, layout.genes, device)
    for budget in schedule:
        selected = y == layout.mask_token
        if not selected.any():
            break
        amp = torch.autocast(y.device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
        with amp:
            output = model(ModelInput(y, {**metadata, "mask_indices": selected}))
        n_masked = int(selected.sum())
        if (output.metadata.get("response_contract_sha256") != layout.sha256
                or not output.metadata.get("is_logits_masked_only")
                or output.logits.shape != (n_masked, COUNT_SIZE)
                or output.logits.device != y.device):
            raise RuntimeError("Sampler received logits with an incompatible layout or device")
        expected_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if output.logits.dtype != expected_dtype:
            raise RuntimeError(f"Requested {precision}, model produced logits in {output.logits.dtype}")
        if not torch.isfinite(output.logits).all():
            raise FloatingPointError("Nonfinite sampling logits")

        # All cells started fully masked and share the same per-step integer budget.
        k = int(budget)
        scores = torch.rand(y.shape, device=device, dtype=torch.float32, generator=rng)
        scores.masked_fill_(~selected, -1.)
        columns = scores.topk(k, dim=-1).indices
        rows = torch.arange(cells, device=device)[:, None].expand(cells, k)
        lookup[selected] = torch.arange(n_masked, device=device)
        logits = output.logits[lookup[rows, columns].reshape(-1)].double()
        uniform = torch.rand(logits.shape, dtype=torch.float64, device=device, generator=rng)
        tokens = sample_count_tokens(logits, uniform)
        y[rows, columns] = tokens.reshape(cells, k)
    if torch.any(y[:, gene_span] == layout.mask_token):
        raise RuntimeError("Reverse diffusion left a masked gene")
    return decode_counts(y[:, gene_span].cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/checkpoint.pt"))
    parser.add_argument("--data", type=Path, default=Path("data/dummy.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/predictions.npz"))
    parser.add_argument("--steps", type=int, default=4, help="Reverse diffusion steps")
    parser.add_argument("--seed", type=int, default=3407)
    add_device_args(parser)
    args = parser.parse_args()
    if args.output.resolve() in {args.checkpoint.resolve(), args.data.resolve()}:
        parser.error("The output path must differ from the checkpoint and fixture")
    device = resolve_device(args.device, args.plugin)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    data_sha = file_sha256(args.data)
    if checkpoint["data_sha256"] != data_sha:
        raise ValueError("Fixture SHA-256 differs from the training checkpoint")
    expected_settings = settings()
    recorded = dict(checkpoint["model_settings"])
    if isinstance(recorded.get("attention_axes"), tuple):
        recorded["attention_axes"] = list(recorded["attention_axes"])
    if recorded != expected_settings:
        raise ValueError("Checkpoint geometry differs from the configured model settings")
    if checkpoint.get("parameter_count", PARAMETER_COUNT) != PARAMETER_COUNT:
        raise ValueError("Checkpoint parameter count differs from the configured model")
    model = build_model(expected_settings)
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.to(device).eval()
    actual_device = str(next(model.parameters()).device)
    fixture = load_fixture(args.data)
    controls = fixture["eval_control"]
    targets = fixture["targets"]
    group_size = model.layout.group_size
    if (controls.ndim != 2 or not len(controls) or len(controls) % group_size
            or controls.shape[1] != model.layout.genes):
        raise ValueError("Held-out controls must form complete sets on the checkpoint gene axis")
    counts = []
    for target in targets.tolist():
        groups = []
        for index, start in enumerate(range(0, len(controls), group_size)):
            groups.append(sample_group(model, controls[start:start + group_size], target,
                                       device=device, precision=args.precision, steps=args.steps,
                                       seed=group_seed(args.seed, target, index, group_size)))
        counts.append(np.concatenate(groups))
    counts = np.stack(counts)
    binding = dict(checkpoint_step=int(checkpoint["step"]), data_sha256=data_sha,
                   checkpoint_sha256=file_sha256(args.checkpoint), weights="ema-0.05",
                   device=actual_device, precision=args.precision, steps=args.steps, seed=args.seed)
    with atomic_output(args.output) as stream:
        np.savez_compressed(stream, counts=counts, targets=targets, **binding)
    print(json.dumps({"stage": "inference", "shape": list(counts.shape), "output": str(args.output), **binding}))


if __name__ == "__main__":
    main()
