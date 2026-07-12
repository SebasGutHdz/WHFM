"""Interaction potentials and kernels."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .potentials import Potential
from .schedules import SigmaSchedule
from ..solvers import rk4_integrate, scipy_solve_ivp


class InteractionKernel(ABC):
    """Pairwise interaction kernel ``W(x, y)``."""

    @abstractmethod
    def kernel(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        """Return pairwise kernel values."""

    def __call__(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        return self.kernel(x, y, **kwargs)


class InteractionPotential(Potential):
    """Potential of the form ``int int W(x, y) rho(x) rho(y) dx dy``."""

    @abstractmethod
    def interaction_energy(self, x: Tensor, y: Tensor) -> Tensor:
        """Return pairwise interaction energy values."""

    @abstractmethod
    def sigma_rhs(self) -> Callable:
        """Return the ODE right-hand side governing sigma evolution."""

    @abstractmethod
    def compute_sigma_schedule(
        self,
        sigma_0,
        sigma_dot_0=None,
        *,
        t_grid=None,
        n_steps: int = 200,
        method: str = "scipy",
    ) -> SigmaSchedule:
        """Precompute a sigma schedule on ``[0, 1]``."""


class RepulsiveKernel(InteractionKernel):
    """Repulsive kernel from ``example_interaction.ipynb``."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-3):
        self.alpha = alpha
        self.eps = eps

    def kernel(self, x: Tensor, y: Tensor, *, alpha=None) -> Tensor:
        alpha_value = self.alpha if alpha is None else alpha
        alpha_t = torch.as_tensor(alpha_value, dtype=x.dtype, device=x.device)
        eps_t = torch.as_tensor(self.eps, dtype=x.dtype, device=x.device)
        diff = x - y
        if diff.dim() == 0:
            sq_norm = diff.pow(2)
        elif diff.dim() == 1:
            sq_norm = diff.pow(2)
        else:
            sq_norm = diff.pow(2).sum(dim=-1)
        return -sq_norm / (alpha_t.pow(2) * sq_norm + eps_t).pow(2)


class RationalQuadraticInteractionKernel(InteractionKernel):
    """Kernel W(x, y) = 2 / (||x - y||^2 + 1) with analytic gradient."""

    def kernel(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        diff = x - y
        if diff.dim() == 0:
            sq_norm = diff.pow(2)
        elif diff.dim() == 1:
            sq_norm = diff.pow(2)
        else:
            sq_norm = diff.pow(2).sum(dim=-1)
        return 2.0 / (sq_norm + 1.0)

    def gradient(self, x: Tensor, y: Tensor, **kwargs) -> Tensor:
        diff = x - y
        if diff.dim() == 0:
            sq_norm = diff.pow(2)
        elif diff.dim() == 1:
            sq_norm = diff.pow(2)
        else:
            sq_norm = diff.pow(2).sum(dim=-1, keepdim=True)
        return -4.0 * diff / (sq_norm + 1.0).pow(2)


class KernelInteractionPotential(InteractionPotential):
    """Interaction potential driven by Monte Carlo samples and a pairwise kernel."""

    def __init__(
        self,
        kernel: InteractionKernel,
        coupling_samples: Tensor = None,
        *,
        n_samples: int = 1000,
        dim: int = 1,
        sigma_as_kernel_alpha: bool = True,
        acceleration_sign: float = -1.0,
    ):
        self.kernel = kernel
        if coupling_samples is None:
            coupling_samples = torch.randn(n_samples, 2, dim)
        self.coupling_samples = torch.as_tensor(coupling_samples, dtype=torch.get_default_dtype())
        self.sigma_as_kernel_alpha = sigma_as_kernel_alpha
        self.acceleration_sign = acceleration_sign

    def interaction_energy(self, x: Tensor, y: Tensor) -> Tensor:
        return self.kernel(x, y)

    def interaction_gradient(self, x: Tensor, y: Tensor) -> Tensor:
        if hasattr(self.kernel, "gradient"):
            return self.kernel.gradient(x, y)

        with torch.enable_grad():
            x_req = x if x.requires_grad else x.detach().requires_grad_(True)
            energy = self.interaction_energy(x_req, y)
            (grad,) = torch.autograd.grad(
                energy.sum(),
                x_req,
                create_graph=x.requires_grad,
                retain_graph=x.requires_grad,
            )
        return grad

    def _acceleration(self, sigma: Tensor) -> Tensor:
        samples = self.coupling_samples.to(device=sigma.device, dtype=sigma.dtype)
        z1 = samples[:, 0]
        z2 = samples[:, 1]
        alpha = sigma.reshape(-1)[0] if self.sigma_as_kernel_alpha else None
        values = self.kernel(z1, z2, alpha=alpha)
        return self.acceleration_sign * values.mean()

    def sigma_rhs(self):
        samples = self.coupling_samples.detach().cpu()

        def rhs(_t, state):
            sigma = torch.as_tensor(float(state[0]), dtype=samples.dtype)
            z1 = samples[:, 0]
            z2 = samples[:, 1]
            alpha = sigma if self.sigma_as_kernel_alpha else None
            values = self.kernel(z1, z2, alpha=alpha)
            acceleration = self.acceleration_sign * values.mean()
            return np.asarray([state[1], float(acceleration)], dtype=float)

        return rhs

    def compute_sigma_schedule(
        self,
        sigma_0,
        sigma_dot_0=None,
        *,
        t_grid=None,
        n_steps: int = 200,
        method: str = "scipy",
    ) -> SigmaSchedule:
        sigma_0_t = torch.as_tensor(sigma_0, dtype=torch.get_default_dtype()).reshape(1)
        sigma_dot = 0.0 if sigma_dot_0 is None else float(torch.as_tensor(sigma_dot_0).reshape(-1)[0])
        y0 = torch.stack([sigma_0_t.reshape(()), torch.as_tensor(sigma_dot, dtype=sigma_0_t.dtype)])
        t_t = (
            torch.linspace(0.0, 1.0, n_steps + 1, dtype=sigma_0_t.dtype)
            if t_grid is None
            else torch.as_tensor(t_grid, dtype=sigma_0_t.dtype).reshape(-1)
        )

        if method == "scipy":
            result = scipy_solve_ivp(
                self.sigma_rhs(),
                (float(t_t[0]), float(t_t[-1])),
                y0,
                t_eval=t_t,
                dense_output=True,
                to_tensor=True,
            )
            sigma = result.y[0].reshape(-1, 1)
            sigma_prime = result.y[1].reshape(-1, 1)
        elif method == "rk4":
            def rhs(t, state):
                sigma = state[0:1]
                sigma_prime = state[1:2]
                acceleration = self._acceleration(sigma).reshape(1)
                return torch.cat([sigma_prime, acceleration], dim=0)

            trajectory = rk4_integrate(rhs, y0, t_t)
            sigma = trajectory[:, 0:1]
            sigma_prime = trajectory[:, 1:2]
        else:
            raise ValueError(f"Unknown sigma schedule method: {method}")

        return SigmaSchedule(t=t_t.to(sigma.device), sigma=sigma, sigma_prime=sigma_prime)


def build_interaction_potential(name: str) -> InteractionPotential:
    """Build a no-argument interaction potential by name."""

    builders = {
        "repulsive": lambda: KernelInteractionPotential(
            RepulsiveKernel(), sigma_as_kernel_alpha=False
        ),
        "rational_quadratic": lambda: KernelInteractionPotential(
            RationalQuadraticInteractionKernel(), sigma_as_kernel_alpha=False
        ),
    }
    try:
        return builders[name]()
    except KeyError as exc:
        options = ", ".join(sorted(builders))
        raise ValueError(f"Unknown interaction potential '{name}'. Options: {options}.") from exc
