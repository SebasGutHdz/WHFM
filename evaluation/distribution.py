"""Distribution matching metrics."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor


def sliced_wasserstein2(x: Tensor, y: Tensor, num_projections: int, *, generator=None) -> float:
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    if x_flat.shape[0] != y_flat.shape[0]:
        n = min(x_flat.shape[0], y_flat.shape[0])
        x_flat = x_flat[:n]
        y_flat = y_flat[:n]
    dim = x_flat.shape[1]
    projections = torch.randn(
        (int(num_projections), dim),
        generator=generator,
        device=x.device,
        dtype=x.dtype,
    )
    projections = projections / projections.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x_proj = x_flat @ projections.T
    y_proj = y_flat @ projections.T
    x_sorted = torch.sort(x_proj, dim=0).values
    y_sorted = torch.sort(y_proj, dim=0).values
    return float(torch.sqrt((x_sorted - y_sorted).pow(2).mean()).detach().cpu())


def _median_bandwidth(x: Tensor, y: Tensor) -> Tensor:
    z = torch.cat([x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)], dim=0)
    distances = torch.pdist(z).pow(2)
    positive = distances[distances > 0]
    if positive.numel() == 0:
        return torch.ones((), dtype=x.dtype, device=x.device)
    return torch.median(positive).clamp_min(1e-12)


def rbf_mmd2(x: Tensor, y: Tensor, bandwidth: Optional[float] = None) -> float:
    x_flat = x.reshape(x.shape[0], -1)
    y_flat = y.reshape(y.shape[0], -1)
    sigma2 = (
        torch.as_tensor(float(bandwidth), dtype=x.dtype, device=x.device).clamp_min(1e-12)
        if bandwidth is not None
        else _median_bandwidth(x_flat, y_flat)
    )
    k_xx = torch.exp(-torch.cdist(x_flat, x_flat).pow(2) / (2.0 * sigma2))
    k_yy = torch.exp(-torch.cdist(y_flat, y_flat).pow(2) / (2.0 * sigma2))
    k_xy = torch.exp(-torch.cdist(x_flat, y_flat).pow(2) / (2.0 * sigma2))
    return float((k_xx.mean() + k_yy.mean() - 2.0 * k_xy.mean()).detach().cpu())
