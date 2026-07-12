"""Base potential interface for Hamiltonian flow matching."""

from __future__ import annotations

from abc import ABC

import torch
from torch import Tensor


class Potential(ABC):
    """Root potential interface.

    Subclasses provide the domain-specific energy API. The default gradient is
    autograd-based and is suitable for composed potentials whose energy returns
    one scalar value per input sample.
    """

    def __call__(self, x: Tensor) -> Tensor:
        return self.energy(x)

    def energy(self, x: Tensor) -> Tensor:
        """Return potential energy values for ``x``."""

        raise NotImplementedError(f"{type(self).__name__} does not implement energy(x).")

    def gradient(self, x: Tensor) -> Tensor:
        """Return ``grad_x V(x)`` with the same shape as ``x``."""

        with torch.enable_grad():
            x_req = x if x.requires_grad else x.detach().requires_grad_(True)
            energy = self.energy(x_req)
            if not energy.requires_grad:
                return torch.zeros_like(x_req)
            (grad,) = torch.autograd.grad(
                energy.sum(),
                x_req,
                create_graph=x.requires_grad,
                retain_graph=x.requires_grad,
                allow_unused=True,
            )
        return torch.zeros_like(x_req) if grad is None else grad

    def force(self, x: Tensor) -> Tensor:
        """Return Hamiltonian force ``-grad_x V(x)``."""

        return -self.gradient(x)
