"""Token layout and integer count codec for grouped cell data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import numpy as np
from .identities import fingerprint


GENE_COUNT = 4096
REPRESENTATIVES = (
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
    20, 24, 30, 37, 45, 56, 68, 84, 104, 128, 157, 193, 238,
    293, 361, 444, 546, 672, 827, 1018, 1253, 1543, 1899, 2337,
    2876, 3540, 4357, 5363, 6601, 8124, 9999,
)
COUNT_START, COUNT_SIZE = 5, 48
TARGET_START, NT_TOKEN, BOOKEND, EOS = 306, 305, 4, 1
MODEL_METADATA = frozenset({"control_token_ids", "control_available", "set_id",
                            "source_group_id", "mask_indices"})



@dataclass(frozen=True)
class Layout:
    """Complete layout; seq_length excludes the trailing EOS token.

    Args:
        group_size: Explicit cells per set, at least two and a divisor of 512.
        genes: Number of real genes; tail EOS padding is excluded from this axis.
        prefix: Fixed five-token condition prefix.
        patch_size: Gene patch width, fixed at 32 and independent of group_size.
        target_start: First public target token.
        nt_token: Non-targeting condition token.
        bookend: Constant value in unused prefix slots.
        eos: Trailing end-of-sequence token.
    """

    group_size: int
    genes: int = GENE_COUNT
    prefix: int = 5
    patch_size: int = 32
    target_start: int = TARGET_START
    nt_token: int = NT_TOKEN
    bookend: int = BOOKEND
    eos: int = EOS

    def __post_init__(self):
        if (isinstance(self.genes, bool) or not isinstance(self.genes, int)
                or not 1 <= self.genes <= 65536 or self.prefix != 5 or self.patch_size != 32):
            raise ValueError("This data contract requires a positive gene axis / prefix5 / patch32")
        if (isinstance(self.group_size, bool) or not isinstance(self.group_size, int)
                or self.group_size < 2 or self.group_size > 512 or 512 % self.group_size):
            raise ValueError("Group size must be >=2 and divide the fixed 512-cell response pool")

    @property
    def seq_length(self):
        return self.prefix + self.genes

    @property
    def block_size(self):
        return self.seq_length + max(1, (-self.genes) % self.patch_size)

    @property
    def vocabulary(self):
        return self.target_start + self.genes

    @property
    def padded_vocabulary(self):
        return ((self.vocabulary + 63) // 64) * 64

    @property
    def mask_token(self):
        return self.padded_vocabulary

    @property
    def contract(self):
        return {"schema": "cluster-counts", **asdict(self),
                "seq_length": self.seq_length, "block_size": self.block_size,
                "vocabulary": self.vocabulary, "padded_vocabulary": self.padded_vocabulary,
                "mask_token": self.mask_token, "count_start": COUNT_START,
                "representatives": list(REPRESENTATIVES), "count_encoding": "nearest_ties_lower",
                "overflow": "saturate_9999", "background_embedding": False}

    @property
    def sha256(self):
        return fingerprint(self.contract)

    def prefix_tokens(self, target_gene_id):
        """Build only the public target/NT prefix, with constant other slots."""
        if target_gene_id is not None and (isinstance(target_gene_id, (bool, np.bool_))
                or not isinstance(target_gene_id, (int, np.integer)) or not 0 <= target_gene_id < self.genes):
            raise ValueError("Target must be a public gene-axis coordinate or None for NT")
        target = self.nt_token if target_gene_id is None else self.target_start + int(target_gene_id)
        return np.asarray([self.bookend, self.bookend, target, self.bookend, self.bookend], dtype=np.int64)


def encode_counts(counts):
    """Encode nonnegative integer observations into uint8 count token IDs.

    Args:
        counts: Integer count array, with arbitrary shape.

    Returns:
        Same-shaped uint8 array containing IDs 5..52.

    Raises:
        ValueError: For negative or non-integer observations.
    """
    values = np.asarray(counts)
    if values.dtype.kind not in "ui" or np.any(values < 0):
        raise ValueError("Counts must be nonnegative integers")
    bounded = np.minimum(values, 9999).astype(np.int64)
    reps = np.asarray(REPRESENTATIVES)
    return (COUNT_START + np.searchsorted(reps[1:] + reps[:-1], 2 * bounded, side="left")).astype(np.uint8)


def decode_counts(tokens):
    """Decode count-only IDs, rejecting prefix, padding and MASK tokens."""
    ids = np.asarray(tokens)
    if ids.dtype.kind not in "ui" or np.any((ids < COUNT_START) | (ids >= COUNT_START + COUNT_SIZE)):
        raise ValueError("Only count IDs 5..52 are decodable")
    return np.asarray(REPRESENTATIVES, dtype=np.uint32)[ids.astype(np.int64) - COUNT_START]
