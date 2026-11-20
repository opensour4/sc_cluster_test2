"""Sample-count power-function EMA for response training checkpoints."""
import math
import torch


def _power_exponent(std):
    """Solve the normalized power kernel's relative standard deviation.

    For exponent g, the squared relative width is
    (g+1) / ((g+2)^2 (g+3)). Use the nonnegative solution.
    """
    if not (0 < std < 1 / math.sqrt(12)):
        raise ValueError("EMA standard deviation must be in (0, 1/sqrt(12))")
    def width_squared(exponent):
        return (1.0 + exponent) / ((2.0 + exponent) ** 2 * (3.0 + exponent))

    target = std * std
    low, high = 0.0, 1.0 / std
    for _ in range(64):
        mid = (low + high) / 2
        if width_squared(mid) > target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


class PowerFunctionEMA:
    """Maintain one trainable-parameter shadow per configured EMA width."""

    def __init__(self, *, model, enabled, stds):
        self.model = model
        self.enabled = enabled
        self.stds = tuple(stds)
        self._powers = [_power_exponent(std) + 1 for std in self.stds]
        self.shadow_params = []

    def register(self, model):
        self.model = model
        if self.enabled:
            self.shadow_params = [
                {key: value.detach().clone() for key, value in model.named_parameters() if value.requires_grad}
                for _ in self.stds
            ]

    @torch.no_grad()
    def update(self, *, current_num_samples, batch_size):
        if not self.enabled:
            return
        if not 0 < batch_size <= current_num_samples or not self.shadow_params:
            raise ValueError("EMA requires a registered model and a completed positive batch")
        retained = 1.0 - batch_size / current_num_samples
        for power, shadow in zip(self._powers, self.shadow_params):
            weight = 1.0 - retained ** power
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    shadow[name].lerp_(param.detach(), weight)
