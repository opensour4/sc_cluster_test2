"""Train the cluster test model on fixed synthetic counts with one device or torchrun."""
from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from fixture import load_fixture
from ppu_core import (PowerFunctionEMA, ResponseCrossEntropy, SequenceBatch, build_model,
                      encode_counts, make_diffusion, prepare_step)
from runtime import add_device_args, atomic_output, file_sha256, resolve_device
from model_config import (CELLS, DEFAULT_MICRO_GROUPS, SUPPORTED_MICRO_GROUPS, PARAMETER_COUNT,
                          default_accumulation, settings)


def batch_for_draw(data, layout, draw, seed, device, *, validation=False):
    split = "eval" if validation else "train"
    n = layout.group_size
    # The first two test draws cover NT and a perturbation; subsequent training draws use p(NT)=0.1.
    rng = np.random.default_rng([seed, draw, int(validation)])
    nt = len(data["targets"])
    target_idx = int(rng.integers(nt))
    if not validation and (draw == 0 or (draw > 1 and rng.random() < 0.1)):
        target_idx = nt
    target = None if target_idx == nt else data["targets"][target_idx]
    control = data[f"{split}_control"]
    response = data[f"{split}_response"][target_idx]
    b = encode_counts(control[rng.choice(len(control), size=n, replace=False)])
    y = encode_counts(response[rng.choice(len(response), size=n, replace=False)])
    ids = torch.full((n, layout.block_size), layout.eos, dtype=torch.long, device=device)
    ids[:, :layout.prefix] = torch.as_tensor(layout.prefix_tokens(target), device=device)
    body = slice(layout.prefix, layout.seq_length)
    ids[:, body] = torch.as_tensor(y, device=device)
    eligible = torch.zeros_like(ids, dtype=torch.bool)
    eligible[:, body] = True
    groups = torch.full((n,), draw, dtype=torch.long, device=device)
    slots = torch.arange(n, device=device)
    controls = torch.as_tensor(b.astype(np.int64), device=device)
    return SequenceBatch(ids, layout.prefix, layout.seq_length, {
        "control_token_ids": controls, "control_available": torch.ones_like(controls, dtype=torch.bool),
        "eligible_mask": eligible, "set_id": groups, "source_group_id": groups.clone(),
        "cell_slot": slots, "global_cell_draw": groups * n + slots})


def training_batch(data, layout, first_draw, micro_groups, seed, device):
    """Pack whole, independently sampled sets into one model forward."""
    groups = [batch_for_draw(data, layout, first_draw + offset, seed, device)
              for offset in range(micro_groups)]
    if len(groups) == 1:
        return groups[0]
    return SequenceBatch(
        torch.cat([group.token_ids for group in groups], dim=0),
        layout.prefix, layout.seq_length,
        {key: torch.cat([group.metadata[key] for group in groups], dim=0)
         for key in groups[0].metadata},
    )


def measure_batch(batch, world, accumulation, expected_groups):
    """Read group membership and cell counts from the local input to this forward pass."""
    ids = batch.token_ids
    set_ids = batch.metadata["set_id"]
    if ids.ndim != 2 or set_ids.shape != (ids.shape[0],) or ids.device != set_ids.device:
        raise RuntimeError("Expected one set ID per input cell on the training device")
    groups, counts = torch.unique(set_ids, return_counts=True)
    micro_groups, micro_cells = groups.numel(), ids.shape[0]
    if micro_groups != expected_groups or not torch.all(counts == CELLS):
        raise RuntimeError(f"Expected {expected_groups} complete sets of {CELLS} cells per rank; "
                           f"observed {micro_groups} sets and {micro_cells} cells")
    return dict(micro_groups_per_rank=micro_groups, micro_cells_per_rank=micro_cells,
                global_groups_per_update=micro_groups * world * accumulation,
                global_cells_per_update=micro_cells * world * accumulation)


def forward_loss(model, batch, diffusion, criterion, seed, device, precision):
    inputs, context = prepare_step(batch, diffusion, seed)
    amp = torch.autocast(device.type, dtype=torch.bfloat16) if precision == "bf16" else nullcontext()
    with amp:
        output = model(inputs)
        expected_dtype = torch.bfloat16 if precision == "bf16" else torch.float32
        if output.logits.device != device or output.logits.dtype != expected_dtype:
            raise RuntimeError(f"Expected {device}/{expected_dtype}, got {output.logits.device}/{output.logits.dtype}")
        loss = criterion.compute(context, output).loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite masked loss")
    return loss


def train(args, device, rank, world, backend):
    torch.manual_seed(args.seed)
    config = settings()
    data = load_fixture(args.data)
    model = build_model(config)
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != PARAMETER_COUNT:
        raise RuntimeError(f"Expected {PARAMETER_COUNT:,} trainable parameters, got {parameter_count:,}")
    model = model.to(device)
    wrapped = (DistributedDataParallel(model, device_ids=None if device.type == "cpu" else [device.index],
                                       broadcast_buffers=False) if world > 1 else model)
    device_module = getattr(torch, device.type, None)
    name = device_module.get_device_name(device) if hasattr(device_module, "get_device_name") else device.type
    layout = model.layout
    diffusion = make_diffusion(layout.group_size)
    criterion = ResponseCrossEntropy(layout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.1, betas=(0.9, 0.95))
    ema = PowerFunctionEMA(model=model, enabled=True, stds=[0.05, 0.10])
    ema.register(model)
    initial_head = model.lm_head.weight.detach().clone()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        lr = 2e-4 * min(step / args.warmup, 1.) if args.warmup else 2e-4
        for group in optimizer.param_groups:
            group["lr"] = lr
        total_loss = torch.zeros((), device=device)
        for micro in range(args.accum):
            first_draw = (((step - 1) * args.accum + micro) * world + rank) * args.micro_groups
            batch = training_batch(data, layout, first_draw, args.micro_groups, args.seed, device)
            batch_sizes = measure_batch(batch, world, args.accum, args.micro_groups)
            if step == 1 and micro == 0:
                print(json.dumps(dict(stage="device", rank=rank, world_size=world, device=str(device),
                                      hardware=name, backend=backend, torch=str(torch.__version__),
                                      parameter_count=parameter_count, model_settings=config, **batch_sizes)), flush=True)
            sync = wrapped.no_sync() if world > 1 and micro + 1 < args.accum else nullcontext()
            with sync:
                loss = forward_loss(wrapped, batch, diffusion, criterion, args.seed, device, args.precision)
                (loss / args.accum).backward()
            total_loss += loss.detach() / args.accum
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        valid_ranks = (torch.isfinite(norm) & (norm > 0)).int()
        if world > 1:
            dist.all_reduce(valid_ranks)
        if valid_ranks.item() != world:
            raise FloatingPointError("Each rank must have a finite positive gradient norm before the update")
        optimizer.step()
        global_cells = batch_sizes["global_cells_per_update"]
        ema.update(current_num_samples=step * global_cells, batch_size=global_cells)
        if world > 1:
            dist.all_reduce(total_loss)
        if rank == 0:
            print(json.dumps(dict(stage="train", step=step, loss=total_loss.item() / world,
                                  grad_norm=norm.item(), lr=lr, global_cells=global_cells, **batch_sizes)), flush=True)

    head = model.lm_head.weight.detach()
    delta = (head - initial_head).abs().max().item()
    if not np.isfinite(delta) or delta <= 0:
        raise RuntimeError("No finite optimizer update detected in the output head")
    if world > 1:
        reference = head.clone()
        dist.broadcast(reference, src=0)
        different = (head != reference).any().long()
        dist.all_reduce(different)
        if different.item():
            raise RuntimeError("Ranks disagree on the trained output head")
    if rank == 0:
        model.eval()
        with torch.no_grad():
            # Fixed held-out loss always evaluates one complete set.
            validation = batch_for_draw(data, layout, 0, args.seed + 1, device, validation=True)
            val = forward_loss(model, validation, diffusion, criterion, args.seed + 1, device, args.precision)
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        shadow = {**state, **{k: v.detach().cpu() for k, v in ema.shadow_params[0].items()}}
        result = dict(step=args.steps, data_sha256=file_sha256(args.data), model_settings=config,
                      rank_count=world, device=str(device), precision=args.precision,
                      training_seed=args.seed, accumulation=args.accum, warmup_steps=args.warmup,
                      parameter_count=parameter_count, **batch_sizes,
                      grad_norm=norm.item(), head_update_max=delta, validation_loss=val.item())
        with atomic_output(args.output) as stream:
            torch.save({**result, "model": state, "ema": shadow}, stream)
        print(json.dumps(dict(stage="checkpoint", **result, output=str(args.output))), flush=True)
    if world > 1:
        dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_device_args(parser)
    parser.add_argument("--backend", help="torch.distributed collective backend, e.g. nccl or gloo")
    parser.add_argument("--data", type=Path, default=Path("data/dummy.npz"))
    parser.add_argument("--output", type=Path, default=Path("runs/checkpoint.pt"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--micro-groups", type=int, choices=SUPPORTED_MICRO_GROUPS, default=DEFAULT_MICRO_GROUPS,
                        help="Complete 128-cell sets per rank in each forward")
    parser.add_argument("--accum", type=int, help="Microbatches per rank per update (default: 16 / micro-groups)")
    parser.add_argument("--warmup", type=int, default=1000, help="One-based linear warmup steps")
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()
    if args.accum is None:
        args.accum = default_accumulation(args.micro_groups)
    if min(args.steps, args.accum) < 1 or min(args.warmup, args.seed) < 0:
        parser.error("steps and accum must be positive; warmup and seed must be nonnegative")
    if args.output.resolve() == args.data.resolve():
        parser.error("Checkpoint output must differ from the fixture")
    if args.device.split(":")[0] == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    rank, world = int(os.getenv("RANK", "0")), int(os.getenv("WORLD_SIZE", "1"))
    local_rank = int(os.environ["LOCAL_RANK"]) if world > 1 else None
    device = resolve_device(args.device, args.plugin, local_rank)
    backend = None
    if world > 1:
        backend = args.backend or {"cpu": "gloo", "cuda": "nccl"}.get(device.type)
        if not backend:
            parser.error("Specify the vendor collective --backend for this device")
        dist.init_process_group(backend=backend, timeout=timedelta(minutes=5))
    try:
        train(args, device, rank, world, backend)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
