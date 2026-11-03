"""Internal potentials for Hamiltonian flow matching."""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from torch import Tensor

from .potentials import Potential


def gaussian_mixture_log_density(
    x: Tensor,
    means: Tensor,
    stds: Tensor,
    *,
    sigma_floor: float = 1e-12,
) -> Tensor:
    """Return the log density of a uniform isotropic Gaussian mixture.

    ``x`` has shape ``(..., samples, dim)``. ``means`` and ``stds`` have
    broadcast-compatible leading dimensions and shapes
    ``(..., components, dim)`` and ``(..., components, 1)``. The result has
    shape ``(..., samples)``.
    """

    x = torch.as_tensor(x)
    means = torch.as_tensor(means, dtype=x.dtype, device=x.device)
    stds = torch.as_tensor(stds, dtype=x.dtype, device=x.device)
    if x.ndim < 2 or means.ndim < 2 or stds.ndim < 2:
        raise ValueError("x, means, and stds must each have at least two dimensions.")
    if x.shape[-1] != means.shape[-1]:
        raise ValueError("x and means must have the same final dimension.")
    if stds.shape[-1] != 1:
        raise ValueError("stds must have final dimension 1.")
    if means.shape[-2] != stds.shape[-2]:
        raise ValueError("means and stds must have the same component count.")
    if means.shape[-2] == 0:
        raise ValueError("The Gaussian mixture must contain at least one component.")
    if sigma_floor <= 0.0:
        raise ValueError("sigma_floor must be positive.")
    if torch.any(stds <= 0):
        raise ValueError("stds must contain only positive values.")

    stds = stds.clamp_min(
        torch.as_tensor(sigma_floor, dtype=x.dtype, device=x.device)
    )
    dim = x.shape[-1]
    diff = x.unsqueeze(-2) - means.unsqueeze(-3)
    expanded_stds = stds.unsqueeze(-3).squeeze(-1)
    component_log_density = (
        -0.5 * diff.square().sum(dim=-1) / expanded_stds.square()
        - float(dim) * torch.log(expanded_stds)
        - 0.5 * float(dim) * math.log(2.0 * math.pi)
    )
    return torch.logsumexp(component_log_density, dim=-1) - math.log(
        means.shape[-2]
    )


def entropy_cost(
    xt: Tensor,
    t: Tensor,
    gpath,
    *,
    sigma_floor: float = 1e-12,
) -> Tensor:
    """Return ``log rho_t(xt)`` for the supplied Gaussian path mixture.

    ``xt`` has shape ``(batch, samples, time, dim)`` and the result has shape
    ``(batch, samples, time)``. ``gpath.mean(t)`` and ``gpath.gamma(t)`` must
    return ``(batch, time, dim)`` and ``(batch, time, 1)``. Path parameters are
    detached while gradients with respect to ``xt`` are retained.
    """

    xt = torch.as_tensor(xt)
    t = torch.as_tensor(t, dtype=xt.dtype, device=xt.device)
    if xt.ndim != 4:
        raise ValueError(f"xt must have shape (B, N, T, D); got {tuple(xt.shape)}.")
    batch, _, n_time, dim = xt.shape
    if tuple(t.shape) != (n_time,):
        raise ValueError(f"t must have shape ({n_time},); got {tuple(t.shape)}.")

    means = torch.as_tensor(gpath.mean(t)).detach().to(dtype=xt.dtype, device=xt.device)
    stds = torch.as_tensor(gpath.gamma(t)).detach().to(dtype=xt.dtype, device=xt.device)
    if tuple(means.shape) != (batch, n_time, dim):
        raise ValueError(
            "gpath.mean(t) must have shape "
            f"{(batch, n_time, dim)}; got {tuple(means.shape)}."
        )
    if tuple(stds.shape) != (batch, n_time, 1):
        raise ValueError(
            "gpath.gamma(t) must have shape "
            f"{(batch, n_time, 1)}; got {tuple(stds.shape)}."
        )

    queries = xt.permute(0, 2, 1, 3)
    mixture_means = means.permute(1, 0, 2).unsqueeze(0).expand(batch, -1, -1, -1)
    mixture_stds = stds.permute(1, 0, 2).unsqueeze(0).expand(batch, -1, -1, -1)
    return gaussian_mixture_log_density(
        queries, mixture_means, mixture_stds, sigma_floor=sigma_floor
    ).permute(0, 2, 1)


class InternalPotential(Potential):
    """Marker base for density-dependent potential energies."""


class EntropyPotential(InternalPotential):
    """Entropy potential with coefficient-free energy and score primitives.

    For ``F[rho] = c * int rho log(rho) dx``, :meth:`batch_energy` returns
    the per-sample ``log(rho)`` energy integrand and
    :meth:`score_from_gaussian_mixture` returns ``grad_x log(rho)``. The
    configured potential wrapper applies the YAML coefficient ``c`` exactly
    once. The spatially constant ``1`` in the first variation
    ``1 + log(rho)`` does not enter the reported action or its force.
    """

    def __init__(self, bandwidth: float | None = None):
        if bandwidth is not None and float(bandwidth) <= 0.0:
            raise ValueError("bandwidth must be positive when provided.")
        self.bandwidth = None if bandwidth is None else float(bandwidth)

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
        log_prob = (
            -float(dim) * torch.log(stds_expanded.squeeze(-1))
            - 0.5 * sq_dist / variance.squeeze(-1)
        )
        responsibilities = torch.softmax(log_prob, dim=-1)
        component_scores = -diff / variance
        return (responsibilities.unsqueeze(-1) * component_scores).sum(dim=-2)

    def cost_from_gaussian_mixture(
        self,
        x: Tensor,
        means: Tensor,
        stds: Tensor,
        *,
        sigma_floor: float = 1e-12,
    ) -> Tensor:
        """Return the scalar log-density whose gradient is the mixture score."""

        return gaussian_mixture_log_density(
            x, means, stds, sigma_floor=sigma_floor
        )

    def empirical_log_density(
        self,
        x: Tensor,
        samples: Tensor,
        *,
        bandwidth=None,
    ) -> Tensor:
        """Return KDE log-density at query points from empirical samples."""

        if x.shape[0] == 0:
            return x.new_empty((0,))

        x_flat = x.reshape(x.shape[0], -1)
        samples_flat = samples.reshape(samples.shape[0], -1).to(
            device=x.device, dtype=x.dtype
        )
        if samples_flat.shape[0] == 0:
            raise ValueError("empirical density samples must be nonempty.")
        if x_flat.shape[-1] != samples_flat.shape[-1]:
            raise ValueError("query points and samples must have matching dimensions.")

        sample_count, dim = samples_flat.shape
        eps = torch.as_tensor(1e-12, dtype=x.dtype, device=x.device)
        selected_bandwidth = self.bandwidth if bandwidth is None else float(bandwidth)

        if selected_bandwidth is None:
            scale = samples_flat.std(dim=0, unbiased=False).mean()
            h = scale * float(sample_count) ** (-1.0 / (float(dim) + 4.0))
            h = h.clamp_min(torch.as_tensor(1e-6, dtype=x.dtype, device=x.device))
        else:
            if selected_bandwidth <= 0.0:
                raise ValueError("bandwidth must be positive when provided.")
            h = torch.as_tensor(selected_bandwidth, dtype=x.dtype, device=x.device)

        diff = x_flat[:, None, :] - samples_flat[None, :, :]
        sq_dist = diff.pow(2).sum(dim=-1)
        kernel = torch.exp(-0.5 * sq_dist / h.pow(2))
        two_pi = torch.as_tensor(2.0 * torch.pi, dtype=x.dtype, device=x.device)
        normalizer = two_pi.pow(-0.5 * dim) * h.pow(-dim)
        density = normalizer * kernel.mean(dim=1)
        return torch.log(density.clamp_min(eps))

    def score_from_empirical_samples(
        self,
        x: Tensor,
        samples: Tensor,
        *,
        bandwidth=None,
    ) -> Tensor:
        """Return ``grad_x log rho(x)`` for a KDE over empirical samples."""

        with torch.enable_grad():
            x_req = x if x.requires_grad else x.detach().requires_grad_(True)
            cost = self.empirical_log_density(
                x_req,
                samples,
                bandwidth=bandwidth,
            )
            (score,) = torch.autograd.grad(
                cost.sum(),
                x_req,
                create_graph=x.requires_grad,
                retain_graph=x.requires_grad,
            )
        return score

    def batch_energy(self, x: Tensor, bandwidth=None) -> Tensor:
        """Return the per-sample KDE ``log(rho)`` energy integrand.

        The KDE includes self terms. It uses the explicit method argument when
        provided, otherwise the configured bandwidth, and finally a scalar
        Silverman-style bandwidth when neither is set.
        """
        return self.empirical_log_density(x, x, bandwidth=bandwidth)


class EntropyCostPotential(EntropyPotential):
    """Autograd implementation of the proposed scalar entropy cost.

    This is an independent comparison path for the faster, exactly equivalent
    analytic mixture score implemented by :class:`EntropyPotential`.
    """

    def score_from_gaussian_mixture(
        self,
        x: Tensor,
        means: Tensor,
        stds: Tensor,
        *,
        sigma_floor: float = 1e-12,
    ) -> Tensor:
        with torch.enable_grad():
            x_req = x if x.requires_grad else x.detach().requires_grad_(True)
            cost = self.cost_from_gaussian_mixture(
                x_req, means, stds, sigma_floor=sigma_floor
            )
            (score,) = torch.autograd.grad(
                cost.sum(),
                x_req,
                create_graph=x.requires_grad,
                retain_graph=x.requires_grad,
            )
        return score


def build_internal_potential(
    name: str, parameters: Mapping[str, object] | None = None
) -> InternalPotential:
    """Build a configured internal potential by name."""

    builders = {
        "entropy": EntropyPotential,
        "entropy_cost": EntropyCostPotential,
    }
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, Mapping):
        raise TypeError("internal potential parameters must be a mapping.")
    try:
        builder = builders[name]
    except KeyError as exc:
        options = ", ".join(sorted(builders))
        raise ValueError(
            f"Unknown internal potential '{name}'. Options: {options}."
        ) from exc
    return builder(**dict(parameters))
