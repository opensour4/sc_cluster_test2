"""Probability-weighted cross entropy for masked count tokens."""
from __future__ import annotations

import torch
from torch.nn import functional as F
from .interfaces import LossResult
from .codec import COUNT_SIZE, COUNT_START


class ResponseCrossEntropy:
    """Weight each masked count by the inverse of its corruption probability.

    Args:
        layout: Explicit response layout, including the configured cell-set size.
    """

    loss_metric_names = ("gene_loss", "total_loss", "nt_loss", "response_loss")

    def __init__(self, layout):
        self.layout = layout

    def compute(self, ctx, output):
        """Average weighted masked CE over all cells, including zero-mask cells."""
        logits, mask = output.logits.float(), ctx.mask_indices
        expected = self.layout.sha256
        if (output.metadata.get("response_contract_sha256") != expected
                or logits.shape[-1] != COUNT_SIZE or ctx.token_ids.shape[1] != self.layout.block_size
                or torch.any(mask[:, :self.layout.prefix]) or torch.any(mask[:, self.layout.seq_length:])):
            raise ValueError("Loss input does not match the new count/layout contract")
        labels = ctx.token_ids[mask] - COUNT_START
        if torch.any((labels < 0) | (labels >= COUNT_SIZE)):
            raise ValueError("Only Y count positions may be supervised")
        selected = logits if output.metadata.get("is_logits_masked_only") else logits[mask]
        if selected.shape != (int(mask.sum()), COUNT_SIZE):
            raise ValueError("Masked logits and labels are misaligned")
        per_cell = torch.zeros(len(mask), dtype=torch.float32, device=logits.device)
        if len(labels):
            probabilities = ctx.p_mask[mask].float()
            if torch.any(probabilities <= 0):
                raise ValueError("Masked positions require positive mask probability")
            weighted = F.cross_entropy(selected, labels, reduction="none") / probabilities
            per_cell.scatter_add_(0, mask.nonzero(as_tuple=True)[0], weighted)
        else:
            per_cell = per_cell + logits.sum() * 0
        normalized = per_cell / (self.layout.genes + 1)
        value = normalized.mean()
        nt = ctx.token_ids[:, 2] == self.layout.nt_token
        metrics = {"loss": float(value.detach()), "gene_loss": float(value.detach()),
            "total_loss": float(value.detach()), "mask_rate": float(mask.float().mean()),
            "eligible_mask_rate": float(mask[:, self.layout.prefix:self.layout.seq_length].float().mean()),
            "eligible_genes": float(self.layout.genes), "nt_fraction": float(nt.float().mean())}
        weights = {}
        for name, flag in (("nt_loss", nt), ("response_loss", ~nt)):
            count = int(flag.sum())
            if count:
                metrics[name] = float(normalized.detach()[flag].mean())
                weights[name] = count
        return LossResult(value, metrics, metadata={"loss_metric_weights": weights})
