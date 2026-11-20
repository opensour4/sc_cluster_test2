"""Run from the repository root with python -m unittest discover -s tests."""
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from evaluate import evaluate
from fixture import TARGETS
from model_config import CELLS, GENES
from runtime import file_sha256


class PredictionChecks(unittest.TestCase):
    data_path = Path(__file__).resolve().parents[1] / "data/dummy.npz"

    def setUp(self):
        # The targets have equal gene means, but each sampled cell has a distinct profile.
        self.counts = np.zeros((len(TARGETS), CELLS, GENES), dtype=np.uint32)
        cells = np.arange(CELLS)
        for index in range(len(TARGETS)):
            self.counts[index, cells, cells] = 1
            self.counts[index, cells, CELLS + (cells + index) % CELLS] = 1

    def evaluate_counts(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.npz"
            np.savez(path, counts=self.counts, targets=TARGETS, data_sha256=file_sha256(self.data_path),
                     checkpoint_sha256="0" * 64, checkpoint_step=1, weights="ema-0.05",
                     device="cpu", precision="fp32", steps=4, seed=123)
            return evaluate(self.data_path, path)

    def test_sparse_distinct_cells_with_equal_target_means_pass(self):
        report = self.evaluate_counts()
        self.assertEqual(report["shape"], list(self.counts.shape))

    def test_all_zero_predictions_fail(self):
        self.counts.fill(0)
        with self.assertRaisesRegex(ValueError, "constant predictions"):
            self.evaluate_counts()

    def test_one_constant_nonzero_target_fails(self):
        self.counts[1].fill(4)
        with self.assertRaisesRegex(ValueError, "constant predictions"):
            self.evaluate_counts()

    def test_repeated_target_block_fails(self):
        self.counts[1] = self.counts[0]
        with self.assertRaisesRegex(ValueError, "predictions for target"):
            self.evaluate_counts()

    def test_repeated_cell_in_target_fails(self):
        self.counts[0, 1] = self.counts[0, 0]
        with self.assertRaisesRegex(ValueError, "repeats the complete prediction"):
            self.evaluate_counts()

    def test_repeated_cell_across_targets_fails(self):
        self.counts[1, 2] = self.counts[0, 3]
        with self.assertRaisesRegex(ValueError, "repeats the complete prediction"):
            self.evaluate_counts()
