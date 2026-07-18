"""Evaluation plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch import Tensor


TIME_COLORMAP = "autumn"
POTENTIAL_COLORMAP = "viridis"
PUBLICATION_FIGSIZE = (7.2, 5.8)
PUBLICATION_SQUARE_FIGSIZE = (7.2, 7.2)
PUBLICATION_HIST_FIGSIZE = (7.2, 4.8)
PUBLICATION_GIF_FIGSIZE = (8.6, 8.6)
PUBLICATION_DPI = 300
PUBLICATION_FONT_SIZE = 15
PUBLICATION_AXIS_LABEL_SIZE = 18
PUBLICATION_TITLE_SIZE = 18
PUBLICATION_LEGEND_SIZE = 14
PUBLICATION_COLORBAR_LABEL_SIZE = 17
PUBLICATION_TICK_SIZE = 15
PUBLICATION_SPINE_WIDTH = 1.25
PUBLICATION_TICK_WIDTH = 1.2
PUBLICATION_LINE_WIDTH = 1.6
PUBLICATION_TRAJECTORY_MARKER_SIZE = 5
PUBLICATION_TERMINAL_MARKER_SIZE = 10
PUBLICATION_GIF_STATIC_MARKER_SIZE = 10
PUBLICATION_GIF_CURRENT_MARKER_SIZE = 5


@dataclass(frozen=True)
class _ParticlePlotDescriptor:
    n_particles: int
    particle_dim: int


def _apply_publication_style(plt) -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "mathtext.fontset": "dejavuserif",
            "font.size": PUBLICATION_FONT_SIZE,
            "axes.labelsize": PUBLICATION_AXIS_LABEL_SIZE,
            "axes.titlesize": PUBLICATION_TITLE_SIZE,
            "xtick.labelsize": PUBLICATION_TICK_SIZE,
            "ytick.labelsize": PUBLICATION_TICK_SIZE,
            "legend.fontsize": PUBLICATION_LEGEND_SIZE,
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "axes.linewidth": PUBLICATION_SPINE_WIDTH,
            "xtick.major.width": PUBLICATION_TICK_WIDTH,
            "ytick.major.width": PUBLICATION_TICK_WIDTH,
            "xtick.major.size": 5.5,
            "ytick.major.size": 5.5,
            "lines.linewidth": PUBLICATION_LINE_WIDTH,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _publication_subplots(plt, *, figsize=PUBLICATION_FIGSIZE):
    _apply_publication_style(plt)
    return plt.subplots(figsize=figsize, constrained_layout=True)


def _style_axis(ax, *, title: str | None = None) -> None:
    if title is not None:
        ax.set_title(title, fontsize=PUBLICATION_TITLE_SIZE, pad=10)
    ax.xaxis.label.set_size(PUBLICATION_AXIS_LABEL_SIZE)
    ax.yaxis.label.set_size(PUBLICATION_AXIS_LABEL_SIZE)
    ax.tick_params(
        axis="both",
        which="major",
        labelsize=PUBLICATION_TICK_SIZE,
        width=PUBLICATION_TICK_WIDTH,
        length=5.5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(PUBLICATION_SPINE_WIDTH)


def _style_legend(ax, *, loc: str = "best", markerscale: float = 1.4):
    legend = ax.legend(
        loc=loc,
        markerscale=markerscale,
        frameon=True,
        framealpha=0.92,
        borderpad=0.45,
        handlelength=1.4,
        handletextpad=0.5,
        labelspacing=0.35,
    )
    if legend is not None:
        legend.get_frame().set_linewidth(0.8)
    return legend


def _publication_colorbar(fig, ax, mappable, *, label: str):
    colorbar = fig.colorbar(mappable, ax=ax)
    colorbar.set_label(label, fontsize=PUBLICATION_COLORBAR_LABEL_SIZE)
    colorbar.ax.tick_params(labelsize=PUBLICATION_TICK_SIZE, width=PUBLICATION_TICK_WIDTH, length=5.0)
    return colorbar


def _particle_plot_descriptor(potential, dim: int | None = None) -> _ParticlePlotDescriptor | None:
    linear = getattr(potential, "linear", None)
    component = linear if linear is not None else potential
    class_name = component.__class__.__name__
    if class_name not in {"FixedCenterThreeBodyPotential", "GridSpringPotential", "SmoothCoulombPotential"}:
        return None

    n_particles = getattr(component, "n_particles", None)
    if n_particles is None:
        n_particles = getattr(component, "n_moving", None)
    particle_dim = getattr(component, "particle_dim", None)
    if n_particles is None or particle_dim is None:
        return None
    try:
        n_particles = int(n_particles)
        particle_dim = int(particle_dim)
    except (TypeError, ValueError):
        return None
    if n_particles <= 0 or particle_dim != 2:
        return None
    if dim is not None and n_particles * particle_dim != int(dim):
        return None
    return _ParticlePlotDescriptor(n_particles=n_particles, particle_dim=particle_dim)


def _reshape_particle_positions(x: Tensor, descriptor: _ParticlePlotDescriptor) -> Tensor:
    expected_dim = descriptor.n_particles * descriptor.particle_dim
    if x.shape[-1] != expected_dim:
        raise ValueError(
            "particle plotting expected final dimension "
            f"{expected_dim}, got {x.shape[-1]}."
        )
    return x.reshape(*x.shape[:-1], descriptor.n_particles, descriptor.particle_dim)


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


def _plot_positions(x: Tensor, config, descriptor: _ParticlePlotDescriptor | None) -> Tensor:
    if descriptor is not None:
        return _reshape_particle_positions(x, descriptor)
    return _project(x, config)


def _time_values_for_plot(t_np: np.ndarray, traj_plot: np.ndarray) -> np.ndarray:
    points_per_time = int(np.prod(traj_plot.shape[1:-1]))
    return np.repeat(t_np, points_per_time)


def _set_position_labels(ax, evaluation_config, descriptor: _ParticlePlotDescriptor | None) -> None:
    if descriptor is not None:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        return
    ax.set_xlabel(f"x{evaluation_config.plot_dir1}")
    ax.set_ylabel(f"x{evaluation_config.plot_dir2}")


def _axis_limit_from_config(evaluation_config, attr: str) -> tuple[float, float] | None:
    values = getattr(evaluation_config, attr, [0.0, 0.0])
    low = float(values[0])
    high = float(values[1])
    if low == 0.0 and high == 0.0:
        return None
    return low, high


def _resolve_plot_domain(evaluation_config, low: Tensor, high: Tensor) -> tuple[Tensor, Tensor]:
    resolved_low = low.clone()
    resolved_high = high.clone()
    xlim = _axis_limit_from_config(evaluation_config, "plot_xlim")
    ylim = _axis_limit_from_config(evaluation_config, "plot_ylim")
    if xlim is not None:
        resolved_low[0] = xlim[0]
        resolved_high[0] = xlim[1]
    if ylim is not None:
        resolved_low[1] = ylim[0]
        resolved_high[1] = ylim[1]
    return resolved_low, resolved_high


def _apply_axis_limits(ax, evaluation_config) -> None:
    xlim = _axis_limit_from_config(evaluation_config, "plot_xlim")
    ylim = _axis_limit_from_config(evaluation_config, "plot_ylim")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)


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

    particle_descriptor = _particle_plot_descriptor(potential, traj.shape[-1])
    traj_plot = _plot_positions(traj[:, :count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    generated_plot = _plot_positions(generated[:count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    reference_plot = _plot_positions(reference[:count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    t_np = time_grid.detach().cpu().numpy()
    cmap = plt.get_cmap(TIME_COLORMAP)
    norm = plt.Normalize(vmin=float(t_np[0]), vmax=float(t_np[-1]))

    def add_time_scatter(ax):
        time_values = _time_values_for_plot(t_np, traj_plot)
        ax.scatter(
            traj_plot[..., 0].reshape(-1),
            traj_plot[..., 1].reshape(-1),
            c=time_values,
            cmap=cmap,
            norm=norm,
            s=PUBLICATION_TRAJECTORY_MARKER_SIZE,
            alpha=0.55,
            linewidths=0,
        )
        ax.autoscale()
        _set_position_labels(ax, evaluation_config, particle_descriptor)
        _apply_axis_limits(ax, evaluation_config)

    def add_terminal_points(ax):
        ax.scatter(
            generated_plot[..., 0].reshape(-1),
            generated_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.75,
            label=r"$\tilde{\nu}$",
        )
        ax.scatter(
            reference_plot[..., 0].reshape(-1),
            reference_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.75,
            label=r"$\nu$",
        )
        _style_legend(ax, markerscale=1.5)
        ax.autoscale()
        _apply_axis_limits(ax, evaluation_config)

    fig, ax = _publication_subplots(plt)
    add_time_scatter(ax)
    add_terminal_points(ax)
    _publication_colorbar(fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="t")
    trajectory_path = figures_dir / f"{tag}_trajectories.png"
    _style_axis(ax)
    fig.savefig(trajectory_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(traj[:, :count], source_reference, generated, reference),
    )
    add_time_scatter(ax)
    add_terminal_points(ax)
    _publication_colorbar(fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="t")
    contour_path = figures_dir / f"{tag}_linear_potential.png"
    _style_axis(ax)
    fig.savefig(contour_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(source_reference, generated, reference),
    )
    add_terminal_points(ax)
    _set_position_labels(ax, evaluation_config, particle_descriptor)
    _apply_axis_limits(ax, evaluation_config)
    terminal_path = figures_dir / f"{tag}_terminal_scatter.png"
    _style_axis(ax)
    fig.savefig(terminal_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    drift_np = drift_samples.detach().cpu().numpy()
    ax.hist(drift_np, bins="auto", alpha=0.8)
    ax.set_xlim(0,evaluation_config.xaxis_hist)
    ax.set_xlabel("Hamiltonian drift integral")
    ax.set_ylabel("count")
    rectf_num = tag[1]
    ax.set_title(f'Hamltonian drift historgram, rectification {rectf_num}')
    histogram_path = figures_dir / f"{tag}_hamiltonian_drift_histogram.png"
    _style_axis(ax)
    fig.savefig(histogram_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
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

    particle_descriptor = _particle_plot_descriptor(potential, traj.shape[-1])
    traj_plot = _plot_positions(traj[:, :count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    generated_plot = _plot_positions(generated[:count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    reference_plot = _plot_positions(reference[:count], evaluation_config, particle_descriptor).detach().cpu().numpy()
    t_np = time_grid.detach().cpu().numpy()
    cmap = plt.get_cmap(TIME_COLORMAP)
    norm = plt.Normalize(vmin=float(t_np[0]), vmax=float(t_np[-1]))

    def add_time_scatter(ax):
        time_values = _time_values_for_plot(t_np, traj_plot)
        ax.scatter(
            traj_plot[..., 0].reshape(-1),
            traj_plot[..., 1].reshape(-1),
            c=time_values,
            cmap=cmap,
            norm=norm,
            s=PUBLICATION_TRAJECTORY_MARKER_SIZE,
            alpha=0.55,
            linewidths=0,
        )
        ax.autoscale()
        _set_position_labels(ax, evaluation_config, particle_descriptor)
        _apply_axis_limits(ax, evaluation_config)

    def add_terminal_points(ax):
        ax.scatter(
            generated_plot[..., 0].reshape(-1),
            generated_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.75,
            label=r"$\tilde{\nu}$",
        )
        ax.scatter(
            reference_plot[..., 0].reshape(-1),
            reference_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.75,
            label=r"$\nu$",
        )
        _style_legend(ax, markerscale=1.5)
        ax.autoscale()
        _apply_axis_limits(ax, evaluation_config)

    fig, ax = _publication_subplots(plt)
    add_time_scatter(ax)
    add_terminal_points(ax)
    _publication_colorbar(fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="t")
    trajectory_path = figures_dir / f"{tag}_trajectories.png"
    _style_axis(ax)
    fig.savefig(trajectory_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(traj[:, :count], source_reference, generated, reference),
    )
    add_time_scatter(ax)
    add_terminal_points(ax)
    _publication_colorbar(fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="t")
    contour_path = figures_dir / f"{tag}_linear_potential.png"
    _style_axis(ax)
    fig.savefig(contour_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(source_reference, generated, reference),
    )
    add_terminal_points(ax)
    _set_position_labels(ax, evaluation_config, particle_descriptor)
    _apply_axis_limits(ax, evaluation_config)
    terminal_path = figures_dir / f"{tag}_terminal_scatter.png"
    _style_axis(ax)
    fig.savefig(terminal_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
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
    particle_descriptor = _particle_plot_descriptor(potential, mean.shape[-1])
    mean_plot = _plot_positions(mean, evaluation_config, particle_descriptor).detach().cpu().numpy()
    std_plot = std[..., 0].detach().cpu().numpy()

    fig, ax = _publication_subplots(plt)
    _plot_linear_contour(
        ax,
        potential,
        source_reference,
        evaluation_config,
        domain_tensors=(mean, source_reference),
    )
    if particle_descriptor is None:
        for path_mean in mean_plot:
            ax.plot(path_mean[:, 0], path_mean[:, 1], alpha=0.45, linewidth=PUBLICATION_LINE_WIDTH)
        ax.scatter(mean_plot[:, 0, 0], mean_plot[:, 0, 1], s=PUBLICATION_TERMINAL_MARKER_SIZE, alpha=0.7, label="x0")
        ax.scatter(mean_plot[:, -1, 0], mean_plot[:, -1, 1], s=PUBLICATION_TERMINAL_MARKER_SIZE, alpha=0.7, label="x1")
    else:
        for path_mean in mean_plot:
            ax.plot(path_mean[..., 0], path_mean[..., 1], alpha=0.45, linewidth=PUBLICATION_LINE_WIDTH)
        ax.scatter(
            mean_plot[:, 0, :, 0].reshape(-1),
            mean_plot[:, 0, :, 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.7,
            label="x0",
        )
        ax.scatter(
            mean_plot[:, -1, :, 0].reshape(-1),
            mean_plot[:, -1, :, 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.7,
            label="x1",
        )
    _set_position_labels(ax, evaluation_config, particle_descriptor)
    _apply_axis_limits(ax, evaluation_config)
    _style_legend(ax, markerscale=1.5)
    mean_path = figures_dir / f"{tag}_mean_trajectories.png"
    _style_axis(ax)
    fig.savefig(mean_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    fig, ax = _publication_subplots(plt)
    ax.plot(t_np, std_plot.T, alpha=0.5, linewidth=PUBLICATION_LINE_WIDTH)
    ax.set_xlabel("t")
    ax.set_ylabel("std")
    std_path = figures_dir / f"{tag}_std_paths.png"
    _style_axis(ax)
    fig.savefig(std_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)

    return {"mean_plot": str(mean_path), "std_plot": str(std_path)}


def _projected_plot_domain(
    tensors: Iterable[Tensor],
    evaluation_config,
    dim: int,
    *,
    padding_fraction: float = 0.1,
    min_padding: float = 1.0,
    particle_descriptor: _ParticlePlotDescriptor | None = None,
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
        values = tensor.detach().to(device=device, dtype=dtype)
        if particle_descriptor is None:
            values = values.reshape(-1, dim)[:, [i, j]]
        else:
            values = _reshape_particle_positions(values, particle_descriptor).reshape(-1, 2)
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


def _linear_contour_values(
    potential,
    source_reference: Tensor,
    evaluation_config,
    grid_x: Tensor,
    grid_y: Tensor,
    particle_descriptor,
) -> Tensor:
    i, j = _projection_indices(evaluation_config, source_reference.shape[-1])
    ref = source_reference.mean(dim=0)
    grid_points = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)

    if particle_descriptor is None:
        points = ref.reshape(1, -1).repeat(grid_points.shape[0], 1)
        points[:, i] = grid_points[:, 0]
        points[:, j] = grid_points[:, 1]
        return potential.linear_energy(points).reshape(grid_x.shape)

    ref_particles = _reshape_particle_positions(ref, particle_descriptor)
    n_particles = particle_descriptor.n_particles
    points = ref_particles.reshape(1, n_particles, particle_descriptor.particle_dim)
    points = points.repeat(grid_points.shape[0] * n_particles, 1, 1)
    particle_indices = torch.arange(n_particles, device=grid_points.device).repeat(grid_points.shape[0])
    batch_indices = torch.arange(points.shape[0], device=grid_points.device)
    points[batch_indices, particle_indices] = grid_points.repeat_interleave(n_particles, dim=0)
    values = potential.linear_energy(points.reshape(points.shape[0], source_reference.shape[-1]))
    return values.reshape(grid_points.shape[0], n_particles).mean(dim=1).reshape(grid_x.shape)


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
    particle_descriptor = _particle_plot_descriptor(potential, source_reference.shape[-1])
    low, high = _projected_plot_domain(
        domain_tensors if domain_tensors is not None else (source_reference,),
        evaluation_config,
        source_reference.shape[-1],
        particle_descriptor=particle_descriptor,
    )
    low, high = _resolve_plot_domain(evaluation_config, low, high)
    xs = torch.linspace(low[0], high[0], 80, device=source_reference.device, dtype=source_reference.dtype)
    ys = torch.linspace(low[1], high[1], 80, device=source_reference.device, dtype=source_reference.dtype)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
    with torch.no_grad():
        values = _linear_contour_values(
            potential,
            source_reference,
            evaluation_config,
            grid_x,
            grid_y,
            particle_descriptor,
        ).detach().cpu().numpy()
    grid_x_np = grid_x.detach().cpu().numpy()
    grid_y_np = grid_y.detach().cpu().numpy()
    contour_fill = ax.contourf(
        grid_x_np,
        grid_y_np,
        values,
        levels=35,
        alpha=0.10,
        cmap=POTENTIAL_COLORMAP,
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
