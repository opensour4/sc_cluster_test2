"""Patch transformer conditioned on observed control counts."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .codec import COUNT_SIZE, COUNT_START, Layout, MODEL_METADATA
from .interfaces import ModelOutput
from .layers import Block, RMSNorm, SelfAttention, rotary_cache


@dataclass(frozen=True)
class Settings:
    genes: int
    cell_set_size: int
    n_layer: int
    n_embd: int
    n_head: int
    intermediate_size: int
    attention_axes: tuple
    attn_module: str = "SDPA"

    def __post_init__(self):
        if (self.attn_module != "SDPA" or self.n_head <= 0 or self.n_embd <= 0
                or self.n_embd % (2 * self.n_head) or self.intermediate_size <= 0):
            raise ValueError("SDPA requires positive dimensions and an even rotary head width")
        if (self.n_layer < 1 or len(self.attention_axes) != self.n_layer
                or set(self.attention_axes) - {"L", "N"}):
            raise ValueError("Specify one L or N axis per transformer block")


class CountModel(nn.Module):
    def __init__(self, settings):
        super().__init__()
        self.config = settings
        self.layout = Layout(group_size=settings.cell_set_size, genes=settings.genes)
        layout, width = self.layout, settings.n_embd
        self.gene_pad = (-layout.genes) % layout.patch_size
        self.num_patches = (layout.genes + self.gene_pad) // layout.patch_size
        # Initialize the full head and then retain the original 48 count rows.
        self.lm_head = nn.Linear(width, layout.padded_vocabulary, bias=False)
        self.patch_proj = nn.Linear(layout.patch_size * width, width, bias=False)
        self.depatch_expand = nn.Linear(width, layout.patch_size * width, bias=False)
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(layout.padded_vocabulary + 1, width),
            h=nn.ModuleList(Block(settings) for _ in range(settings.n_layer)),
            ln_f=RMSNorm(width),
        ))
        generator = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(layout.genes, generator=generator, device="cpu")
        inverse = torch.empty_like(perm)
        inverse[perm] = torch.arange(layout.genes, device="cpu")
        self.register_buffer("gene_perm", perm)
        self.register_buffer("gene_inv_perm", inverse)
        self.rope_cache = None
        self.apply(self._init_weights)
        self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach()[COUNT_START:COUNT_START + COUNT_SIZE].clone())
        self.lm_head.out_features = COUNT_SIZE

    def _init_weights(self, module):
        width = self.config.n_embd
        if isinstance(module, (nn.Embedding, nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=math.sqrt(2.0 / 5 / width))
        if isinstance(module, SelfAttention):
            nn.init.normal_(module.proj.weight, mean=0.0, std=1 / math.sqrt(width) / self.config.n_layer)

    def _validate(self, ids, metadata):
        layout = self.layout
        if set(metadata) - MODEL_METADATA:
            raise ValueError("Invalid model metadata")
        if ids.ndim != 2 or not len(ids) or len(ids) % layout.group_size or ids.shape[1] != layout.block_size:
            raise ValueError("Expected complete response cell sets in the padded token layout")
        controls = metadata["control_token_ids"]
        available = metadata["control_available"]
        if (controls.shape != (len(ids), layout.genes) or available.shape != controls.shape
                or not available.all() or ((controls < COUNT_START) | (controls >= COUNT_START + COUNT_SIZE)).any()):
            raise ValueError("All control counts must be observed and in the count vocabulary")
        targets = ids[:, 2].reshape(-1, layout.group_size)
        if (torch.any(ids[:, [0, 1, 3, 4]] != layout.bookend) or torch.any(ids[:, layout.seq_length:] != layout.eos)
                or ((targets != layout.nt_token) & ((targets < layout.target_start) | (targets >= layout.vocabulary))).any()
                or (targets != targets[:, :1]).any()):
            raise ValueError("Invalid prefix, target grouping or EOS padding")
        body = ids[:, layout.prefix:layout.seq_length]
        if ((body != layout.mask_token) & ((body < COUNT_START) | (body >= COUNT_START + COUNT_SIZE))).any():
            raise ValueError("Expected count or MASK tokens in the response")
        for key in ("set_id", "source_group_id"):
            groups = metadata[key].reshape(-1, layout.group_size)
            if len(groups) != len(targets) or (groups != groups[:, :1]).any():
                raise ValueError("Mixed cell sets")
        mask = metadata.get("mask_indices")
        if mask is not None and (mask.shape != ids.shape or mask.dtype != torch.bool or
                                 mask[:, :layout.prefix].any() or mask[:, layout.seq_length:].any()):
            raise ValueError("Expected a gene-body-only supervision mask")

    def _embed_patches(self, ids, metadata):
        """Limit the lifetime of full gene embeddings to patch projection."""
        layout = self.layout
        cells, width = len(ids), self.config.n_embd
        padded_genes = self.num_patches * layout.patch_size
        body = ids[:, layout.prefix:layout.seq_length].index_select(1, self.gene_perm)
        ordered = torch.cat((ids[:, :layout.prefix], body, ids[:, layout.seq_length:]), dim=1)
        prefix, response, tail = self.transformer.wte(ordered).split(
            (layout.prefix, padded_genes, ids.shape[1] - layout.prefix - padded_genes), dim=1
        )
        response = response.reshape(cells, self.num_patches, layout.patch_size * width)
        controls = metadata["control_token_ids"].long().index_select(1, self.gene_perm)
        if self.gene_pad:
            controls = torch.cat((controls, ids[:, layout.seq_length:layout.seq_length+self.gene_pad]), dim=1)
        controls = self.transformer.wte(controls).reshape_as(response)
        half = width // 2
        patches = torch.cat((F.linear(controls, self.patch_proj.weight[:half]),
                             F.linear(response, self.patch_proj.weight[half:])), dim=-1)
        return torch.cat((prefix, patches, tail), dim=1)

    def forward(self, inputs):
        ids, metadata = inputs.noisy_token_ids, inputs.metadata
        self._validate(ids, metadata)
        layout, config = self.layout, self.config
        cells, width = len(ids), config.n_embd
        padded_genes = self.num_patches * layout.patch_size
        x = self._embed_patches(ids, metadata)
        x = torch.cat((x[:, :1], torch.zeros_like(x[:, 1:2]), x[:, 2:]), dim=1)
        if self.rope_cache is None or self.rope_cache[0].device != x.device:
            self.rope_cache = rotary_cache(x.shape[1], width // config.n_head, x.device)
        group_size, length = layout.group_size, x.shape[1]
        for axis, block in zip(config.attention_axes, self.transformer.h):
            if axis == "L":
                x = block(x, self.rope_cache)
            else:
                by_cell = x.reshape(-1, group_size, length, width).transpose(1, 2).reshape(-1, group_size, width)
                by_cell = block(by_cell)
                x = by_cell.reshape(-1, length, group_size, width).transpose(1, 2).reshape(cells, length, width)
        x = self.transformer.ln_f(x)
        genes = self.depatch_expand(x[:, layout.prefix:layout.prefix+self.num_patches]).reshape(cells, padded_genes, width)
        mask = metadata.get("mask_indices")
        if mask is None:
            hidden = torch.cat((x[:, :layout.prefix], genes[:, :layout.genes].index_select(1, self.gene_inv_perm),
                                genes[:, layout.genes:], x[:, layout.prefix+self.num_patches:]), dim=1)
        else:
            cell, position = mask.nonzero(as_tuple=True)
            hidden = genes[cell, self.gene_inv_perm[position - layout.prefix]]
        # The head only needs the gathered features, which have their own storage.
        del genes
        return ModelOutput(self.lm_head(hidden), {
            "is_logits_masked_only": mask is not None, "output_token_support": "count_only",
            "logit_token_start": COUNT_START, "logit_token_count": COUNT_SIZE,
            "response_contract_sha256": layout.sha256,
        })


def build_model(settings):
    return CountModel(Settings(**settings))
