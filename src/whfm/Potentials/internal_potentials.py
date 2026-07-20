"""Internal potentials for Hamiltonian flow matching."""

from __future__ import annotations

from abc import abstractmethod
from typing import Callable

import numpy as np
import torch
from torch import Tensor

from .potentials import Potential
from .schedules import SigmaSchedule
from ..solvers import rk4_integrate, scipy_solve_ivp


class InternalPotential(Potential):
    """Potential of the form ``int U(rho(x)) dx``."""

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


class EntropyPotential(InternalPotential):
    """Entropy-style internal potential with notebook-compatible sigma dynamics.

    The notebooks precompute sigma with the first-order ODE
    ``sigma' = -sqrt(2 * beta * log(sigma) + constant)`` and use the positive
    square root as ``sigma_prime`` when forming the conditional flow.
    """

    def __init__(self, beta: float = 2.0, coeff: float = 0.75, constant=None):
        self.beta = beta
        self.coeff = coeff
        self.constant = constant

    def score_from_gaussian_mixture(
        self,
        x: Tensor,
        means: Tensor,
        stds: Tensor,
        *,
        sigma_floor: float = 1e-12,
    ) -> Tensor:
        """Return grad_x log rho(x) for an isotropic Gaussian mixture.

        x has shape (..., dim). means and stds have matching
        leading dimensions plus a component axis: (..., components, dim)
        and (..., components, 1). Component weights are uniform.
        """

        x = torch.as_tensor(x)
        means = torch.as_tensor(means, dtype=x.dtype, device=x.device)
        stds = torch.as_tensor(stds, dtype=x.dtype, device=x.device).clamp_min(
            torch.as_tensor(sigma_floor, dtype=x.dtype, device=x.device)
        )
        dim = x.shape[-1]
        x_expanded = x.unsqueeze(-2)
        means_expanded = means.unsqueeze(-3)
        stds_expanded = stds.unsqueeze(-3)
        variance = stds_expanded.pow(2)
        diff = x_expanded - means_expanded
        sq_dist = diff.pow(2).sum(dim=-1)
        log_prob = -float(dim) * torch.log(stds_expanded.squeeze(-1)) - 0.5 * sq_dist / variance.squeeze(-1)
        responsibilities = torch.softmax(log_prob, dim=-1)
        component_scores = -diff / variance
        return (responsibilities.unsqueeze(-1) * component_scores).sum(dim=-2)

    def batch_energy(self, x: Tensor, bandwidth=None) -> Tensor:
        """Return per-sample KDE entropy energy for a batch of samples.

        The KDE includes self terms and uses a scalar Silverman-style bandwidth
        when no explicit bandwidth is supplied.
        """

        if x.shape[0] == 0:
            return x.new_empty((0,))

        x_flat = x.reshape(x.shape[0], -1)
        batch_size, dim = x_flat.shape
        eps = torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)

        if bandwidth is None:
            scale = x_flat.std(dim=0, unbiased=False).mean()
            h = scale * float(batch_size) ** (-1.0 / (float(dim) + 4.0))
            h = h.clamp_min(torch.as_tensor(1e-6, dtype=x.dtype, device=x.device))
        else:
            h = torch.as_tensor(bandwidth, dtype=x.dtype, device=x.device).clamp_min(eps)

        diff = x_flat[:, None, :] - x_flat[None, :, :]
        sq_dist = diff.pow(2).sum(dim=-1)
        kernel = torch.exp(-0.5 * sq_dist / h.pow(2))
        two_pi = torch.as_tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)
        normalizer = two_pi.pow(-0.5 * dim) * h.pow(-dim)
        density = normalizer * kernel.mean(dim=1)
        beta = torch.as_tensor(self.beta, dtype=x.dtype, device=x.device)
        return beta * torch.log(density.clamp_min(eps))

    def _constant_for(self, sigma_0):
        sigma_0 = np.asarray(sigma_0, dtype=float)
        if self.constant is not None:
            return np.asarray(self.constant, dtype=float)
        return -2.0 * self.beta * np.log(sigma_0) * (1.0 + self.coeff)

    def sigma_rhs(self):
        def rhs(_t, sigma):
            sigma = np.asarray(sigma, dtype=float)
            constant = self._constant_for(sigma)
            radicand = 2.0 * self.beta * np.log(np.clip(sigma, 1e-12, None)) + constant
            return -np.sqrt(np.maximum(radicand, 0.0))

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
        sigma_0_t = torch.as_tensor(sigma_0, dtype=torch.get_default_dtype()).reshape(-1)
        t_t = (
            torch.linspace(0.0, 1.0, n_steps + 1, dtype=sigma_0_t.dtype)
            if t_grid is None
            else torch.as_tensor(t_grid, dtype=sigma_0_t.dtype).reshape(-1)
        )
        constant_np = self._constant_for(sigma_0_t.detach().cpu().numpy())

        if method == "scipy":
            def rhs(_t, sigma):
                radicand = 2.0 * self.beta * np.log(np.clip(sigma, 1e-12, None)) + constant_np
                return -np.sqrt(np.maximum(radicand, 0.0))

            result = scipy_solve_ivp(
                rhs,
                (float(t_t[0]), float(t_t[-1])),
                sigma_0_t,
                t_eval=t_t,
                dense_output=True,
                to_tensor=True,
            )
            sigma = result.y.T
        elif method == "rk4":
            constant_t = torch.as_tensor(constant_np, dtype=sigma_0_t.dtype, device=sigma_0_t.device)

            def rhs(_t, sigma):
                radicand = 2.0 * self.beta * torch.log(sigma.clamp_min(1e-12)) + constant_t
                return -torch.sqrt(radicand.clamp_min(0.0))

            sigma = rk4_integrate(rhs, sigma_0_t, t_t)
        else:
            raise ValueError(f"Unknown sigma schedule method: {method}")

        constant_t = torch.as_tensor(constant_np, dtype=sigma.dtype, device=sigma.device)
        sigma_prime = torch.sqrt(
            (2.0 * self.beta * torch.log(sigma.clamp_min(1e-12)) + constant_t).clamp_min(0.0)
        )
        return SigmaSchedule(t=t_t.to(sigma.device), sigma=sigma, sigma_prime=sigma_prime)


def build_internal_potential(name: str) -> InternalPotential:
    """Build a no-argument internal potential by name."""

    builders = {
        "entropy": EntropyPotential,
    }
    try:
        return builders[name]()
    except KeyError as exc:
        options = ", ".join(sorted(builders))
        raise ValueError(f"Unknown internal potential '{name}'. Options: {options}.") from exc
