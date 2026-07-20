"""Diagnostics for Hamiltonian flow matching."""

from __future__ import annotations

import os

import numpy as np
import torch

from .losses import _call_model


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def hamiltonian_energy(model, potential, traj, t_span) -> np.ndarray:
    """Compute mean ``0.5 * ||v(x_t, t)||^2 + V(x_t)`` along a trajectory."""

    traj_t = traj if torch.is_tensor(traj) else torch.as_tensor(traj, dtype=torch.get_default_dtype())
    t_t = t_span if torch.is_tensor(t_span) else torch.as_tensor(t_span, dtype=traj_t.dtype)
    t_t = t_t.to(device=traj_t.device, dtype=traj_t.dtype).reshape(-1)

    if traj_t.shape[0] == t_t.numel():
        time_first = traj_t
    elif traj_t.dim() >= 3 and traj_t.shape[1] == t_t.numel():
        time_first = traj_t.transpose(0, 1)
    else:
        raise ValueError("traj must have time as the first or second dimension.")

    values = []
    with torch.no_grad():
        for i, t_value in enumerate(t_t):
            x = time_first[i]
            t_batch = torch.full((x.shape[0], 1), t_value, dtype=x.dtype, device=x.device)
            v = _call_model(model, x, t_batch)
            kinetic = 0.5 * v.reshape(v.shape[0], -1).pow(2).sum(dim=-1)
            potential_energy = potential.energy(x)
            values.append((kinetic + potential_energy).mean())
    return torch.stack(values).detach().cpu().numpy()


def relative_hamiltonian_drift(H: np.ndarray) -> np.ndarray:
    """Return ``(H(t) - H(0)) / H(0)`` with a small zero guard."""

    H = np.asarray(H)
    denom = H[0] if abs(H[0]) > 1e-12 else 1e-12
    return (H - H[0]) / denom


def plot_hamiltonian(t, H, title: str = ""):
    """Plot Hamiltonian energy over time."""

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(_to_numpy(t), _to_numpy(H))
    ax.set_xlabel("t")
    ax.set_ylabel("H(t)")
    if title:
        ax.set_title(title)
    return fig


def plot_trajectories_with_potential(traj, X, Y, Z, n: int = 2000):
    """Plot 2D trajectories over a potential contour."""

    import matplotlib.pyplot as plt

    traj_np = _to_numpy(traj)
    if traj_np.shape[-1] != 2:
        raise ValueError("plot_trajectories_with_potential expects 2D trajectories.")
    if traj_np.shape[0] < traj_np.shape[1]:
        time_first = traj_np
    else:
        time_first = np.swapaxes(traj_np, 0, 1)

    fig, ax = plt.subplots()
    ax.contourf(_to_numpy(X), _to_numpy(Y), _to_numpy(Z), levels=40, alpha=0.6)
    samples = time_first[:, : min(n, time_first.shape[1]), :]
    for i in range(samples.shape[1]):
        ax.plot(samples[:, i, 0], samples[:, i, 1], alpha=0.15, linewidth=0.7)
    ax.scatter(samples[0, :, 0], samples[0, :, 1], s=4, alpha=0.6)
    ax.scatter(samples[-1, :, 0], samples[-1, :, 1], s=4, alpha=0.6)
    return fig


def quiver_animation(model, x0_samples, save_dir, N: int = 50, **kwargs) -> None:
    """Wrap ``examples_HF.utils_hf.gif_quiver``."""

    try:
        from examples_HF.utils_hf import gif_quiver
    except ImportError as exc:
        raise ImportError("Could not import examples_HF.utils_hf.gif_quiver.") from exc

    os.makedirs(os.path.join(save_dir, "figs_gif"), exist_ok=True)
    gif_quiver(model, save_dir, x0_samples, x0_samples.shape[0], N=N, **kwargs)


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Compute the Frechet distance between two Gaussian summaries."""

    from scipy.linalg import sqrtm

    mu1_np = _to_numpy(mu1)
    mu2_np = _to_numpy(mu2)
    sigma1_np = _to_numpy(sigma1)
    sigma2_np = _to_numpy(sigma2)

    diff = mu1_np - mu2_np
    covmean = sqrtm(sigma1_np @ sigma2_np)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1_np) + np.trace(sigma2_np) - 2.0 * np.trace(covmean)
    return float(value)
