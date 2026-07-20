"""Configured composite potential wrapper."""

from __future__ import annotations

import torch
from torch import Tensor

from .interaction_potentials import build_interaction_potential
from .internal_potentials import build_internal_potential
from .linear_potentials import build_linear_potential
from .potentials import Potential


def _as_like(value, x: Tensor) -> Tensor:
    return torch.as_tensor(value, dtype=x.dtype, device=x.device)


class ConfiguredPotential(Potential):
    """Batch-aware configured potential wrapper.

    The configuration must contain exactly ``linear``, ``internal``, and
    ``interaction`` entries. Each entry is either ``None`` or ``(name,
    coefficient[, parameters])``. Calling the wrapper returns one weighted
    energy value per sample in the input batch.
    """

    _REQUIRED_KEYS = {"linear", "internal", "interaction"}

    def __init__(self, cfg):
        self._validate_keys(cfg)
        self.cfg = dict(cfg)

        self.linear_name, self.linear_coefficient, self.linear = self._build_component(
            cfg["linear"], "linear", build_linear_potential
        )
        self.internal_name, self.internal_coefficient, self.internal = self._build_component(
            cfg["internal"], "internal", build_internal_potential
        )
        (
            self.interaction_name,
            self.interaction_coefficient,
            self.interaction,
        ) = self._build_component(cfg["interaction"], "interaction", build_interaction_potential)

        self.has_linear = self.linear is not None
        self.has_internal = self.internal is not None
        self.has_interaction = self.interaction is not None

    @classmethod
    def _validate_keys(cls, cfg) -> None:
        keys = set(cfg)
        if keys != cls._REQUIRED_KEYS:
            missing = ", ".join(sorted(cls._REQUIRED_KEYS - keys))
            extra = ", ".join(sorted(keys - cls._REQUIRED_KEYS))
            details = []
            if missing:
                details.append(f"missing keys: {missing}")
            if extra:
                details.append(f"extra keys: {extra}")
            suffix = f" ({'; '.join(details)})" if details else ""
            raise ValueError(
                "ConfiguredPotential cfg must contain exactly linear, internal, interaction"
                + suffix
            )

    @staticmethod
    def _build_component(component_cfg, key: str, builder):
        if component_cfg is None:
            return None, 0.0, None
        if not isinstance(component_cfg, tuple) or len(component_cfg) not in {2, 3}:
            raise ValueError(
                f"cfg['{key}'] must be None or a tuple of the form "
                "(name, coefficient[, parameters])."
            )
        if len(component_cfg) == 2:
            name, coefficient = component_cfg
            parameters = {}
        else:
            name, coefficient, parameters = component_cfg
        if not isinstance(name, str):
            raise TypeError(f"cfg['{key}'][0] must be a potential name string.")
        if isinstance(coefficient, bool):
            raise TypeError(f"cfg['{key}'][1] must be a numeric coefficient.")
        try:
            coefficient_value = float(coefficient)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"cfg['{key}'][1] must be a numeric coefficient.") from exc
        if parameters is None:
            parameters = {}
        if not isinstance(parameters, dict):
            raise TypeError(f"cfg['{key}'][2] must be a parameters dict.")
        if key == "linear":
            component = builder(name, parameters)
        elif parameters:
            raise ValueError(f"cfg['{key}'] does not support parameters yet.")
        else:
            component = builder(name)
        return name, coefficient_value, component

    @staticmethod
    def _zeros(x: Tensor) -> Tensor:
        return torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)

    def _shuffled_batch(self, x: Tensor) -> Tensor:
        if x.shape[0] <= 1:
            return x
        return x[torch.randperm(x.shape[0], device=x.device)]

    def linear_energy(self, x: Tensor) -> Tensor:
        if self.linear is None:
            return self._zeros(x)
        return _as_like(self.linear_coefficient, x) * self.linear.energy(x)

    def linear_gradient(self, x: Tensor) -> Tensor:
        if self.linear is None:
            return torch.zeros_like(x)
        return _as_like(self.linear_coefficient, x) * self.linear.gradient(x)

    def internal_energy(self, x: Tensor) -> Tensor:
        if self.internal is None:
            return self._zeros(x)
        if not hasattr(self.internal, "batch_energy"):
            raise TypeError(
                f"Internal potential '{self.internal_name}' does not provide batch_energy(x)."
            )
        return _as_like(self.internal_coefficient, x) * self.internal.batch_energy(x)

    def internal_gradient_from_gaussian_mixture(
        self, x: Tensor, means: Tensor, stds: Tensor
    ) -> Tensor:
        if self.internal is None:
            return torch.zeros_like(x)
        if not hasattr(self.internal, "score_from_gaussian_mixture"):
            raise TypeError(
                f"Internal potential '{self.internal_name}' does not provide "
                "score_from_gaussian_mixture(x, means, stds)."
            )
        coefficient = _as_like(self.internal_coefficient, x)
        return coefficient * self.internal.score_from_gaussian_mixture(x, means, stds)

    def interaction_energy(self, x: Tensor, y: Tensor = None) -> Tensor:
        if self.interaction is None:
            return self._zeros(x)
        y = self._shuffled_batch(x) if y is None else y
        coefficient = 0.5 * _as_like(self.interaction_coefficient, x)
        return coefficient * self.interaction.interaction_energy(x, y)

    def interaction_gradient(self, x: Tensor, y: Tensor = None) -> Tensor:
        if self.interaction is None:
            return torch.zeros_like(x)
        y = self._shuffled_batch(x) if y is None else y
        return _as_like(self.interaction_coefficient, x) * self.interaction.interaction_gradient(x, y)

    def energy(self, x: Tensor) -> Tensor:
        return self.linear_energy(x) + self.internal_energy(x) + self.interaction_energy(x)
