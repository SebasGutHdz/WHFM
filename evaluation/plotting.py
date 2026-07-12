"""Evaluation plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import Tensor


def _projection_indices(config, dim: int):
    i = int(config.plot_dir1)
    j = int(config.plot_dir2)
    if i < 0 or j < 0 or i >= dim or j >= dim:
        raise ValueError("plot_dir1 and plot_dir2 must be valid dimensions.")
    if i == j and dim != 1:
        raise ValueError("plot_dir1 and plot_dir2 must be distinct unless dimension is 1.")
    return i, j


def _project(x: Tensor, config) -> Tensor:
    i, j = _projection_indices(config, x.shape[-1])
    if x.shape[-1] == 1 and i == j:
        return torch.stack([x[..., i], torch.zeros_like(x[..., i])], dim=-1)
    return x[..., [i, j]]


def save_evaluation_plots(
    *,
    figures_dir: Path,
    tag: str,
    traj: Tensor,
    time_grid: Tensor,
    generated: Tensor,
    reference: Tensor,
    drift_samples: Tensor,
    potential,
    source_reference: Tensor,
    evaluation_config,
) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(evaluation_config.plot_trajectory_count), traj.shape[1])
    if count <= 0:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"trajectory_plot": f"plot unavailable: {exc}"}

    traj_plot = _project(traj[:, :count], evaluation_config).detach().cpu().numpy()
    generated_plot = _project(generated[:count], evaluation_config).detach().cpu().numpy()
    reference_plot = _project(reference[:count], evaluation_config).detach().cpu().numpy()
    t_np = time_grid.detach().cpu().numpy()
    cmap = plt.get_cmap(evaluation_config.plot_colormap)
    norm = plt.Normalize(vmin=float(t_np[0]), vmax=float(t_np[-1]))

    def add_time_scatter(ax):
        time_values = np.repeat(t_np, traj_plot.shape[1])
        ax.scatter(
            traj_plot[..., 0].reshape(-1),
            traj_plot[..., 1].reshape(-1),
            c=time_values,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.55,
            linewidths=0,
        )
        ax.autoscale()
        ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
        ax.set_ylabel(f"x{evaluation_config.plot_dir2}")

    def add_terminal_points(ax):
        ax.scatter(
            generated_plot[:, 0],
            generated_plot[:, 1],
            s=12,
            alpha=0.75,
            label="generated terminal",
        )
        ax.scatter(reference_plot[:, 0], reference_plot[:, 1], s=12, alpha=0.75, label="target")
        ax.legend(markerscale=1.5)
        ax.autoscale()

    fig, ax = plt.subplots()
    add_time_scatter(ax)
    add_terminal_points(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="t")
    trajectory_path = figures_dir / f"{tag}_trajectories.png"
    fig.savefig(trajectory_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(traj[:, :count], source_reference, generated, reference),
    )
    add_time_scatter(ax)
    add_terminal_points(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="t")
    contour_path = figures_dir / f"{tag}_linear_potential.png"
    fig.savefig(contour_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(source_reference, generated, reference),
    )
    add_terminal_points(ax)
    ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
    ax.set_ylabel(f"x{evaluation_config.plot_dir2}")
    terminal_path = figures_dir / f"{tag}_terminal_scatter.png"
    fig.savefig(terminal_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    drift_np = drift_samples.detach().cpu().numpy()
    ax.hist(drift_np, bins="auto", alpha=0.8)
    ax.set_xlabel("Hamiltonian drift integral")
    ax.set_ylabel("count")
    ax.set_title(tag)
    histogram_path = figures_dir / f"{tag}_hamiltonian_drift_histogram.png"
    fig.savefig(histogram_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "trajectory_plot": str(trajectory_path),
        "linear_potential_plot": str(contour_path),
        "terminal_scatter_plot": str(terminal_path),
        "hamiltonian_histogram_plot": str(histogram_path),
    }


def save_warmup_plots(
    *,
    figures_dir: Path,
    tag: str,
    traj: Tensor,
    time_grid: Tensor,
    generated: Tensor,
    reference: Tensor,
    potential,
    source_reference: Tensor,
    evaluation_config,
) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    count = min(int(evaluation_config.plot_trajectory_count), traj.shape[1])
    if count <= 0:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"trajectory_plot": f"plot unavailable: {exc}"}

    traj_plot = _project(traj[:, :count], evaluation_config).detach().cpu().numpy()
    generated_plot = _project(generated[:count], evaluation_config).detach().cpu().numpy()
    reference_plot = _project(reference[:count], evaluation_config).detach().cpu().numpy()
    t_np = time_grid.detach().cpu().numpy()
    cmap = plt.get_cmap(evaluation_config.plot_colormap)
    norm = plt.Normalize(vmin=float(t_np[0]), vmax=float(t_np[-1]))

    def add_time_scatter(ax):
        time_values = np.repeat(t_np, traj_plot.shape[1])
        ax.scatter(
            traj_plot[..., 0].reshape(-1),
            traj_plot[..., 1].reshape(-1),
            c=time_values,
            cmap=cmap,
            norm=norm,
            s=8,
            alpha=0.55,
            linewidths=0,
        )
        ax.autoscale()
        ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
        ax.set_ylabel(f"x{evaluation_config.plot_dir2}")

    def add_terminal_points(ax):
        ax.scatter(
            generated_plot[:, 0],
            generated_plot[:, 1],
            s=12,
            alpha=0.75,
            label="generated terminal",
        )
        ax.scatter(reference_plot[:, 0], reference_plot[:, 1], s=12, alpha=0.75, label="target")
        ax.legend(markerscale=1.5)
        ax.autoscale()

    fig, ax = plt.subplots()
    add_time_scatter(ax)
    add_terminal_points(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="t")
    trajectory_path = figures_dir / f"{tag}_trajectories.png"
    fig.savefig(trajectory_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(traj[:, :count], source_reference, generated, reference),
    )
    add_time_scatter(ax)
    add_terminal_points(ax)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, label="t")
    contour_path = figures_dir / f"{tag}_linear_potential.png"
    fig.savefig(contour_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(source_reference, generated, reference),
    )
    add_terminal_points(ax)
    ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
    ax.set_ylabel(f"x{evaluation_config.plot_dir2}")
    terminal_path = figures_dir / f"{tag}_terminal_scatter.png"
    fig.savefig(terminal_path, bbox_inches="tight")
    plt.close(fig)
    return {
        "trajectory_plot": str(trajectory_path),
        "linear_potential_plot": str(contour_path),
        "terminal_scatter_plot": str(terminal_path),
    }


def save_bridge_solution_plots(
    *,
    figures_dir: Path,
    tag: str,
    mean: Tensor,
    std: Tensor,
    time_grid: Tensor,
    potential,
    source_reference: Tensor,
    evaluation_config,
) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if mean.shape[0] == 0:
        return {}
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"plot_error": f"plot unavailable: {exc}"}

    t_np = time_grid.detach().cpu().numpy()
    mean_plot = _project(mean, evaluation_config).detach().cpu().numpy()
    std_plot = std[..., 0].detach().cpu().numpy()

    fig, ax = plt.subplots()
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(mean, source_reference),
    )
    for path_mean in mean_plot:
        ax.plot(path_mean[:, 0], path_mean[:, 1], alpha=0.45, linewidth=0.9)
    ax.scatter(mean_plot[:, 0, 0], mean_plot[:, 0, 1], s=10, alpha=0.7, label="x0")
    ax.scatter(mean_plot[:, -1, 0], mean_plot[:, -1, 1], s=10, alpha=0.7, label="x1")
    ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
    ax.set_ylabel(f"x{evaluation_config.plot_dir2}")
    ax.legend(markerscale=1.5)
    mean_path = figures_dir / f"{tag}_mean_trajectories.png"
    fig.savefig(mean_path, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(t_np, std_plot.T, alpha=0.5, linewidth=0.9)
    ax.set_xlabel("t")
    ax.set_ylabel("std")
    std_path = figures_dir / f"{tag}_std_paths.png"
    fig.savefig(std_path, bbox_inches="tight")
    plt.close(fig)

    return {"mean_plot": str(mean_path), "std_plot": str(std_path)}


def _projected_plot_domain(
    tensors: Iterable[Tensor],
    evaluation_config,
    dim: int,
    *,
    padding_fraction: float = 0.1,
    min_padding: float = 1.0,
) -> tuple[Tensor, Tensor]:
    i, j = _projection_indices(evaluation_config, dim)
    projected = []
    device = None
    dtype = None
    for tensor in tensors:
        if tensor is None or tensor.numel() == 0:
            continue
        if tensor.shape[-1] != dim:
            raise ValueError("all contour domain tensors must have the same final dimension.")
        if device is None:
            device = tensor.device
            dtype = tensor.dtype
        values = tensor.detach().to(device=device, dtype=dtype).reshape(-1, dim)[:, [i, j]]
        projected.append(values)
    if not projected:
        raise ValueError("contour domain requires at least one non-empty tensor.")
    points = torch.cat(projected, dim=0)
    low = points.min(dim=0).values
    high = points.max(dim=0).values
    span = (high - low).clamp_min(0.0)
    padding = torch.maximum(
        span * float(padding_fraction),
        torch.full_like(span, float(min_padding)),
    )
    return low - padding, high + padding


def _plot_linear_contour(
    ax,
    potential,
    source_reference: Tensor,
    evaluation_config,
    *,
    domain_tensors: Iterable[Tensor] | None = None,
) -> None:
    if not getattr(potential, "has_linear", False):
        return
    i, j = _projection_indices(evaluation_config, source_reference.shape[-1])
    ref = source_reference.mean(dim=0)
    low, high = _projected_plot_domain(
        domain_tensors if domain_tensors is not None else (source_reference,),
        evaluation_config,
        source_reference.shape[-1],
    )
    xs = torch.linspace(low[0], high[0], 80, device=source_reference.device, dtype=source_reference.dtype)
    ys = torch.linspace(low[1], high[1], 80, device=source_reference.device, dtype=source_reference.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    points = ref.reshape(1, -1).repeat(grid_x.numel(), 1)
    points[:, i] = grid_x.reshape(-1)
    points[:, j] = grid_y.reshape(-1)
    with torch.no_grad():
        values = potential.linear_energy(points).reshape(grid_x.shape).detach().cpu().numpy()
    grid_x_np = grid_x.detach().cpu().numpy()
    grid_y_np = grid_y.detach().cpu().numpy()
    contour_fill = ax.contourf(
        grid_x_np,
        grid_y_np,
        values,
        levels=35,
        alpha=0.10,
        cmap="Greys",
    )
    ax.contour(
        grid_x_np,
        grid_y_np,
        values,
        levels=contour_fill.levels,
        colors="0.25",
        linewidths=0.6,
        alpha=0.7,
    )
