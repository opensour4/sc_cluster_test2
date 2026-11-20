"""Generate and read synthetic count matrices on an anonymous integer gene axis."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


from model_config import CELLS, GENES

TARGETS = (3, 11)


def generate(seed=3407):
    rng = np.random.default_rng(seed)
    genes, cells = GENES, CELLS
    targets = np.array(TARGETS, dtype=np.int64)
    control = rng.uniform(0.3, 4.5, size=genes)
    truth = np.tile(control, (len(targets), 1))
    for i, target in enumerate(targets):
        truth[i, target] *= 0.5
        truth[i, i::4] *= 1.6
    # Last training condition is NT. B, Y and held-out observations are independent draws.
    train_rates = np.concatenate([truth, control[None, :]])
    return dict(targets=targets, control_mean=control, true_mean=truth,
                train_control=rng.poisson(control, size=(cells, genes)).astype(np.uint16),
                train_response=rng.poisson(train_rates[:, None, :],
                                           size=(len(train_rates), cells, genes)).astype(np.uint16),
                eval_control=rng.poisson(control, size=(cells, genes)).astype(np.uint16),
                eval_response=rng.poisson(truth[:, None, :],
                                          size=(len(targets), cells, genes)).astype(np.uint16))


def load_fixture(path):
    with np.load(path, allow_pickle=False) as archive:
        expected = {"targets", "control_mean", "true_mean", "train_control",
                    "train_response", "eval_control", "eval_response"}
        if set(archive.files) != expected or len(archive.files) != len(expected):
            raise ValueError("Unexpected synthetic fixture arrays")
        data = {key: archive[key] for key in archive.files}
    targets, ctrl, means = data["targets"], data["control_mean"], data["true_mean"]
    if (ctrl.shape != (GENES,) or targets.dtype.kind not in "iu" or not np.array_equal(targets, TARGETS)
            or ctrl.dtype != np.float64 or means.dtype != np.float64
            or means.shape != (len(targets), len(ctrl))
            or any((a < 0).any() or not np.isfinite(a).all() for a in (ctrl, means))):
        raise ValueError("Invalid fixture targets or population means")
    for split, extra in (("train", 1), ("eval", 0)):
        b, y = data[f"{split}_control"], data[f"{split}_response"]
        if (b.shape != (CELLS, GENES)
                or y.shape != (len(targets) + extra, *b.shape)
                or b.dtype != np.uint16 or y.dtype != np.uint16):
            raise ValueError(f"Invalid {split} count matrices")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        np.savez_compressed(stream, **generate())
