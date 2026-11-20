"""Exponential-peak corruption with a shared set time and independent cell masks."""
from __future__ import annotations

import math
import numpy as np
import torch

from .codec import Layout, MODEL_METADATA
from .interfaces import LossInputContext, ModelInput


class CellSetDiffusion:
    def __init__(self, group_size):
        Layout(group_size=group_size)
        self.cell_set_size = group_size
        self.t_sampling_params = {"t_exp_w": 0.1, "t_exp_q": 0.25, "t_exp_b_max": 200.0}
        self._exp_peak_params = None

    @staticmethod
    def _uniform_draws(length, seeds, device):
        times, masks = [], []
        for seed in seeds.tolist():
            generator = torch.Generator(device=device).manual_seed(int(seed))
            times.append(torch.rand((), device=device, generator=generator, dtype=torch.float32))
            masks.append(torch.rand(length, device=device, generator=generator, dtype=torch.float32))
        return torch.stack(times), torch.stack(masks)

    def _sample_noise(self, batch, seeds):
        n = self.cell_set_size
        device = batch.token_ids.device
        count = len(batch.token_ids)
        if count == 0 or count % n:
            raise ValueError("Shared-t noise requires complete sets")
        fields = {}
        for key in ("set_id", "cell_slot", "global_cell_draw"):
            value = batch.metadata.get(key)
            if value is None or value.shape != (count,):
                raise ValueError(f"Missing batch-aligned {key}")
            fields[key] = value.to(device=device, dtype=torch.long)
        sets = fields["set_id"].reshape(-1, n)
        slots = fields["cell_slot"].reshape(-1, n)
        if (bool((sets < 0).any()) or not torch.equal(sets, sets[:, :1].expand_as(sets))
                or sets[:, 0].unique().numel() != len(sets)
                or not torch.equal(slots.sort(dim=1).values,
                                   torch.arange(n, device=device).expand_as(slots))
                or not torch.equal(fields["global_cell_draw"],
                                   fields["set_id"] * n + fields["cell_slot"])):
            raise ValueError("Invalid complete-set draw identities for shared-t noise")
        # Each cell's mask stream starts after its initial time draw.
        _, uniform = self._uniform_draws(batch.token_ids.shape[1], seeds, device)
        time_seeds = sets[:, 0] + (seeds - fields["global_cell_draw"])[0] + 1000000000022
        times, _ = self._uniform_draws(1, time_seeds, device)
        return self._uniform_to_exponential_t(times.repeat_interleave(n)), uniform

    def _uniform_to_exponential_t(self, u: torch.Tensor) -> torch.Tensor:
        w = float(self.t_sampling_params.get("t_exp_w", 0.1))
        q = float(self.t_sampling_params.get("t_exp_q", 0.25))
        b_max = float(self.t_sampling_params.get("t_exp_b_max", 200.0))
        if self._exp_peak_params is None:
            self._exp_peak_params = self._solve_exponential_t_abc(w, q, b_max=b_max)
        a, b, c = self._exp_peak_params

        from scipy.special import lambertw  # type: ignore[import-not-found]

        u = torch.as_tensor(u)
        u_dtype = u.dtype
        u = u.to(dtype=torch.float64)
        a_t = torch.tensor(a, dtype=torch.float64, device=u.device)
        b_t = torch.tensor(b, dtype=torch.float64, device=u.device)
        c_t = torch.tensor(c, dtype=torch.float64, device=u.device)

        K = a_t * torch.exp(-b_t) / b_t
        S = u + K
        z = (b_t * K / c_t) * torch.exp(b_t * S / c_t)
        z_np = z.detach().cpu().numpy()
        w_np = lambertw(z_np, k=0)
        w_real = torch.from_numpy(np.asarray(w_np.real, dtype=z_np.dtype)).to(device=u.device, dtype=torch.float64)
        t = S / c_t - w_real / b_t
        return t.to(dtype=u_dtype)

    def _solve_exponential_t_abc(self, w: float, q: float, *, b_max: float = 200.0) -> tuple[float, float, float]:
        w = float(w)
        q = float(q)
        b_max = float(b_max)
        if not (0.0 < w < 0.5):
            raise ValueError(
                "Require 0 < w < 0.5 to keep a peak near t=1 under the 50% window constraint."
            )
        if not (0.0 < q < 0.5):
            raise ValueError("Require 0 < q < 0.5.")
        if not (b_max > 0.0):
            raise ValueError("b_max must be > 0.")

        def compute_a_c(b: float) -> tuple[float, float] | tuple[None, None]:
            exp_neg_b = math.exp(-b)
            denom = (1.0 - math.exp(-b * w)) - w * (1.0 - exp_neg_b)
            if denom <= 0.0 or not math.isfinite(denom):
                return None, None
            a = b * (0.5 - w) / denom
            c = 1.0 - a * (1.0 - exp_neg_b) / b
            if not (math.isfinite(a) and math.isfinite(c)):
                return None, None
            return a, c

        def g(b: float) -> float | None:
            a, c = compute_a_c(b)
            if a is None or c is None:
                return None
            val = (a / b) * (math.exp(-b / 2.0) - math.exp(-b)) + 0.5 * c - q
            if not math.isfinite(val):
                return None
            return float(val)

        from scipy.optimize import root_scalar  # type: ignore[import-not-found]

        lo, hi = 1.0e-3, b_max
        glo, ghi = g(lo), g(hi)
        if glo is None or ghi is None or glo * ghi > 0.0:
            raise RuntimeError(
                "No sign change found for g(b) on the default bracket; (w, q) may be infeasible."
            )

        sol = root_scalar(
            lambda x: g(float(x)),
            bracket=(lo, hi),
            method="brentq",
            xtol=1.0e-12,
            rtol=1.0e-10,
            maxiter=200,
        )
        if not sol.converged:
            raise RuntimeError("SciPy root solver did not converge for b.")

        b_sol = float(sol.root)
        a_sol, c_sol = compute_a_c(b_sol)
        if a_sol is None or c_sol is None:
            raise RuntimeError("Solved b but got invalid a,c (unexpected).")
        return float(a_sol), float(b_sol), float(c_sol)


def make_diffusion(group_size):
    return CellSetDiffusion(group_size)


def prepare_step(batch, diffusion, seed=3407):
    """Corrupt eligible response counts and preserve the observed control inputs."""
    ids = batch.token_ids
    prefix, length = batch.condition_prefix_length, batch.seq_length
    layout = Layout(group_size=diffusion.cell_set_size, genes=length-prefix)
    if prefix != layout.prefix or ids.shape[1] != layout.block_size:
        raise ValueError("Batch does not match the response-token layout")
    eligible = batch.metadata["eligible_mask"].to(device=ids.device, dtype=torch.bool)
    positions = torch.arange(ids.shape[1], device=ids.device)[None, :]
    body = (positions >= prefix) & (positions < length)
    if eligible.shape != ids.shape or (eligible & ~body).any() or not eligible.any(dim=1).all():
        raise ValueError("Invalid gene-body supervision mask")
    draws = batch.metadata["global_cell_draw"].to(device=ids.device, dtype=torch.long)
    times, uniforms = diffusion._sample_noise(batch, draws + int(seed) + 1000000000039)
    probabilities = (0.995 * times[:, None] + 0.005).expand_as(ids).masked_fill(~eligible, 0).clamp(0., 1.)
    selected = uniforms < probabilities
    noisy = ids.masked_fill(selected | (body & ~eligible), layout.mask_token)
    metadata = {key: value.to(ids.device) for key, value in batch.metadata.items() if key in MODEL_METADATA}
    metadata["mask_indices"] = selected
    return ModelInput(noisy, metadata), LossInputContext(ids, prefix, length, selected, probabilities)
