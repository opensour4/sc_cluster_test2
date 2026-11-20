"""Shared CLI, device selection and artifact I/O."""
from __future__ import annotations

import hashlib
import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def add_device_args(parser):
    parser.add_argument("--device", required=True, help="Registered PyTorch device, e.g. cpu or cuda")
    parser.add_argument("--plugin", help="Vendor module to import before selecting the device")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="bf16")


def resolve_device(name, plugin=None, local_rank=None):
    if plugin:
        importlib.import_module(plugin)
    requested = torch.device(name)
    if local_rank is not None and requested.type != "cpu":
        if requested.index is not None:
            raise ValueError("Use a device type without an index with torchrun")
        requested = torch.device(requested.type, local_rank)
    if requested.type == "meta":
        raise ValueError("The test requires physical tensor storage")
    if requested.type != "cpu":
        backend = getattr(torch, requested.type, None)
        if requested.index is not None and hasattr(backend, "set_device"):
            backend.set_device(requested.index)
    probe = torch.ones(1, device=requested)
    if probe.device.type != requested.type or (requested.index is not None and probe.device != requested):
        raise RuntimeError(f"Requested {requested}, tensor was allocated on {probe.device}")
    return probe.device


def file_sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextmanager
def atomic_output(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}-", delete=False) as temp:
        try:
            yield temp
            temp.flush()
            temp.close()
            os.replace(temp.name, path)
        finally:
            Path(temp.name).unlink(missing_ok=True)
