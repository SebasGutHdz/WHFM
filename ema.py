"""Internal exponential moving average for HFM velocity models."""

from __future__ import annotations

import copy

import torch


class ExponentialMovingAverage:
    """Maintain a detached EMA copy of a torch module.

    ``mode='fixed'`` uses a constant decay. ``mode='posthoc'`` uses
    ``beta_gamma(t) = (1 - 1 / t) ** (gamma + 1)`` with update count ``t``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        decay: float = 0.995,
        *,
        mode: str = "fixed",
        gamma: float = 6.99,
    ):
        if mode not in {"fixed", "posthoc"}:
            raise ValueError("mode must be 'fixed' or 'posthoc'.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("decay must be in [0, 1).")
        if gamma <= 0.0:
            raise ValueError("gamma must be positive.")
        self.mode = mode
        self.decay = float(decay)
        self.gamma = float(gamma)
        self.num_updates = 0
        self.ema_model = copy.deepcopy(model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)

    def current_decay(self) -> float:
        if self.mode == "fixed":
            return self.decay
        t = max(self.num_updates + 1, 1)
        return float((1.0 - 1.0 / float(t)) ** (self.gamma + 1.0))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        beta = self.current_decay()
        self.num_updates += 1
        online_params = dict(model.named_parameters())
        for name, ema_param in self.ema_model.named_parameters():
            ema_param.mul_(beta).add_(online_params[name], alpha=1.0 - beta)

        online_buffers = dict(model.named_buffers())
        for name, ema_buffer in self.ema_model.named_buffers():
            ema_buffer.copy_(online_buffers[name])

    def to(self, device=None, dtype=None):
        self.ema_model.to(device=device, dtype=dtype)
        return self

    def state_dict(self):
        return {
            "mode": self.mode,
            "decay": self.decay,
            "gamma": self.gamma,
            "num_updates": self.num_updates,
            "ema_model": self.ema_model.state_dict(),
        }

    def load_state_dict(self, state) -> None:
        self.mode = str(state.get("mode", "fixed"))
        self.decay = float(state["decay"])
        self.gamma = float(state.get("gamma", 6.99))
        self.num_updates = int(state["num_updates"])
        self.ema_model.load_state_dict(state["ema_model"])
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad_(False)
