"""Model geometry shared by the fixture, training and inference."""

GENES = 18533
CELLS = 128
SUPPORTED_MICRO_GROUPS = (1, 2)
DEFAULT_MICRO_GROUPS = 1
PARAMETER_COUNT = 123_543_040


def default_accumulation(micro_groups):
    """Keep 16 complete groups per rank and optimizer update by default."""
    if micro_groups not in SUPPORTED_MICRO_GROUPS:
        raise ValueError(f"Unsupported microbatch group count: {micro_groups}")
    return 16 // micro_groups


def settings():
    return dict(genes=GENES, cell_set_size=CELLS, n_layer=13, n_embd=640, n_head=10,
                intermediate_size=2560, attention_axes=["L", "N"] * 6 + ["L"], attn_module="SDPA")
