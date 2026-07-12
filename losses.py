"""Loss helpers for Hamiltonian flow matching."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def flow_matching_loss(vt: Tensor, ut: Tensor) -> Tensor:
    """Mean-squared flow matching loss."""

    return F.mse_loss(vt, ut)


def _call_model(model, xt: Tensor, t: Tensor) -> Tensor:
    t_col = t.reshape(t.shape[0], -1)[:, :1] if t.dim() > 1 else t.reshape(-1, 1)
    if xt.dim() == 2:
        try:
            return model(torch.cat([xt, t_col.to(device=xt.device, dtype=xt.dtype)], dim=-1))
        except TypeError:
            pass
    return model(t, xt)


def finite_difference_time_derivative(
    model,
    xt: Tensor,
    t: Tensor,
    epsilon: float = 1e-3,
) -> Tensor:
    """Central finite-difference estimate of ``d model(x, t) / dt`` at fixed ``x``."""

    eps = torch.as_tensor(epsilon, dtype=t.dtype, device=t.device)
    vt_plus = _call_model(model, xt, t + eps)
    vt_minus = _call_model(model, xt, t - eps)
    return (vt_plus - vt_minus) / (2.0 * eps)


def hamiltonian_physics_loss(vt_t: Tensor, grad_V: Tensor) -> Tensor:
    """Mean-squared residual for ``dv/dt = -grad V(x)``."""

    return F.mse_loss(vt_t, -grad_V)


def combined_loss(
    flow_loss: Tensor,
    physics_loss: Tensor,
    lambda_: float,
    use_physics_loss: bool = False,
) -> Tensor:
    """Combine flow matching and optional physics losses."""

    if not use_physics_loss:
        return flow_loss
    return flow_loss + lambda_ * physics_loss
