"""Evaluation metrics and plots for Hamiltonian Flow Matching training."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch import Tensor

from ..optimal_transport import wasserstein
from .node import NodeSolver, call_velocity_model


METRIC_FIELDNAMES = [
    "timestamp",
    "rectification_index",
    "direction",
    "model_kind",
    "num_eval_samples",
    "w2",
    "sliced_w2",
    "mmd2_rbf",
    "hamiltonian_drift_integral_mean",
    "hamiltonian_drift_integral_max",
    "action_mean",
    "kinetic_integral_mean",
    "potential_integral_mean",
    "terminal_mean_error",
    "terminal_cov_error",
    "terminal_displacement_mean",
    "latest_loss",
    "bridge_success_rate",
    "bridge_failed_pairs",
    "trajectory_plot",
    "linear_potential_plot",
    "terminal_scatter_plot",
    "hamiltonian_histogram_plot",
]

WARMUP_METRIC_FIELDNAMES = [
    "timestamp",
    "direction",
    "model_kind",
    "num_eval_samples",
    "w2",
    "sliced_w2",
    "mmd2_rbf",
    "terminal_mean_error",
    "terminal_cov_error",
    "terminal_displacement_mean",
    "latest_warmup_loss",
    "trajectory_plot",
    "linear_potential_plot",
    "terminal_scatter_plot",
    "sample_path",
]


def cap_pair(x: Tensor, y: Tensor, max_samples: int) -> tuple[Tensor, Tensor]:
    n = min(int(max_samples), x.shape[0], y.shape[0])
    if n <= 0:
        raise ValueError("evaluation requires at least one sample.")
    return x[:n], y[:n]


@torch.no_grad()
def integrate_model(model, node_solver: NodeSolver, x0: Tensor):
    return node_solver.integrate(model, x0)


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


@torch.no_grad()
def trajectory_quantities(model, potential, traj, time_grid: Tensor) -> Dict[str, float | Tensor]:
    h_values = []
    kinetic_values = []
    potential_values = []
    for i, t_value in enumerate(time_grid.reshape(-1)):
        x_t = traj.states[i]
        t_batch = t_value.reshape(1, 1).expand(x_t.shape[0], 1)
        velocity = call_velocity_model(model, x_t, t_batch)
        kinetic = 0.5 * velocity.reshape(velocity.shape[0], -1).pow(2).sum(dim=-1)
        potential_energy = potential.energy(x_t)
        kinetic_values.append(kinetic)
        potential_values.append(potential_energy)
        h_values.append(kinetic + potential_energy)

    kinetic_t = torch.stack(kinetic_values, dim=0)
    potential_t = torch.stack(potential_values, dim=0)
    h_t = torch.stack(h_values, dim=0)
    t = time_grid.reshape(-1)
    h0 = h_t[0]
    drift = torch.trapz((h_t - h0).abs(), t, dim=0) / (h0.abs() + 1e-6)
    kinetic_integral = torch.trapz(kinetic_t, t, dim=0)
    potential_integral = torch.trapz(potential_t, t, dim=0)
    action = kinetic_integral - potential_integral
    return {
        "hamiltonian_drift_integral_mean": float(drift.mean().detach().cpu()),
        "hamiltonian_drift_integral_max": float(drift.max().detach().cpu()),
        "action_mean": float(action.mean().detach().cpu()),
        "kinetic_integral_mean": float(kinetic_integral.mean().detach().cpu()),
        "potential_integral_mean": float(potential_integral.mean().detach().cpu()),
        "hamiltonian_drift_samples": drift.detach().cpu(),
    }


def covariance_matrix(x: Tensor) -> Tensor:
    x_flat = x.reshape(x.shape[0], -1)
    if x_flat.shape[0] <= 1:
        return torch.zeros((x_flat.shape[1], x_flat.shape[1]), dtype=x.dtype, device=x.device)
    centered = x_flat - x_flat.mean(dim=0, keepdim=True)
    return centered.T @ centered / float(x_flat.shape[0] - 1)


def terminal_summary(generated: Tensor, target: Tensor) -> Dict[str, float]:
    generated_flat = generated.reshape(generated.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    mean_error = (generated_flat.mean(dim=0) - target_flat.mean(dim=0)).norm()
    cov_error = (covariance_matrix(generated_flat) - covariance_matrix(target_flat)).norm()
    displacement = (generated_flat - target_flat).norm(dim=-1).mean()
    return {
        "terminal_mean_error": float(mean_error.detach().cpu()),
        "terminal_cov_error": float(cov_error.detach().cpu()),
        "terminal_displacement_mean": float(displacement.detach().cpu()),
    }


def bridge_summary(bridge_records: Iterable[dict], rectification_index: int, direction: str) -> Dict[str, float]:
    records = [
        row
        for row in bridge_records
        if row.get("rectification_index") == rectification_index and row.get("direction") == direction
    ]
    requested = sum(int(row.get("requested_pairs", 0)) for row in records)
    successful = sum(int(row.get("successful_pairs", 0)) for row in records)
    failed = sum(int(row.get("failed_pairs", 0)) for row in records)
    success_rate = float(successful / requested) if requested else float("nan")
    return {"bridge_success_rate": success_rate, "bridge_failed_pairs": failed}


def append_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in METRIC_FIELDNAMES})


def append_warmup_metrics_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=WARMUP_METRIC_FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in WARMUP_METRIC_FIELDNAMES})


def evaluate_warmup_model(
    *,
    model,
    model_kind: str,
    direction: str,
    node_solver: NodeSolver,
    potential,
    source_test: Tensor,
    target_test: Tensor,
    evaluation_config,
    latest_warmup_loss: float,
    figures_dir: Path,
    samples_dir: Path,
) -> Dict[str, object]:
    source_eval, target_eval = cap_pair(source_test, target_test, evaluation_config.max_metric_samples)
    traj = integrate_model(model, node_solver, source_eval)
    generated = traj.states[-1]
    generator = torch.Generator(device=source_eval.device)
    generator.manual_seed(811)
    tag = f"warmup_{direction}_{model_kind}"
    plot_paths = save_warmup_plots(
        figures_dir=figures_dir,
        tag=tag,
        traj=traj.states,
        time_grid=traj.time_grid,
        generated=generated,
        reference=target_eval,
        potential=potential,
        source_reference=source_eval,
        evaluation_config=evaluation_config,
    )
    samples_dir.mkdir(parents=True, exist_ok=True)
    sample_path = samples_dir / f"{tag}.pt"
    torch.save(
        {
            "direction": direction,
            "model_kind": model_kind,
            "time_grid": traj.time_grid.detach().cpu(),
            "states": traj.states.detach().cpu(),
            "start": source_eval.detach().cpu(),
            "generated_terminal": generated.detach().cpu(),
            "reference_terminal": target_eval.detach().cpu(),
        },
        sample_path,
    )
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "direction": direction,
        "model_kind": model_kind,
        "num_eval_samples": int(source_eval.shape[0]),
        "w2": wasserstein(generated, target_eval, method="exact", power=2),
        "sliced_w2": sliced_wasserstein2(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        ),
        "mmd2_rbf": rbf_mmd2(generated, target_eval, evaluation_config.mmd_bandwidth),
        "latest_warmup_loss": latest_warmup_loss,
        "sample_path": str(sample_path),
    }
    row.update(terminal_summary(generated, target_eval))
    row.update(plot_paths)
    return row


def evaluate_model(
    *,
    model,
    model_kind: str,
    direction: str,
    rectification_index: int,
    node_solver: NodeSolver,
    potential,
    source_test: Tensor,
    target_test: Tensor,
    evaluation_config,
    latest_loss: float,
    bridge_metrics,
    figures_dir: Path,
) -> Dict[str, object]:
    source_eval, target_eval = cap_pair(source_test, target_test, evaluation_config.max_metric_samples)
    traj = integrate_model(model, node_solver, source_eval)
    generated = traj.states[-1]
    generator = torch.Generator(device=source_eval.device)
    generator.manual_seed(17 + int(rectification_index))
    q = trajectory_quantities(model, potential, traj, traj.time_grid)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rectification_index": rectification_index,
        "direction": direction,
        "model_kind": model_kind,
        "num_eval_samples": int(source_eval.shape[0]),
        "w2": wasserstein(generated, target_eval, method="exact", power=2),
        "sliced_w2": sliced_wasserstein2(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        ),
        "mmd2_rbf": rbf_mmd2(generated, target_eval, evaluation_config.mmd_bandwidth),
        "latest_loss": latest_loss,
    }
    row.update({key: value for key, value in q.items() if key != "hamiltonian_drift_samples"})
    row.update(terminal_summary(generated, target_eval))
    row.update(bridge_summary(bridge_metrics, rectification_index, direction))
    row.update(
        save_evaluation_plots(
            figures_dir=figures_dir,
            tag=f"r{rectification_index}_{direction}_{model_kind}",
            traj=traj.states,
            time_grid=traj.time_grid,
            generated=generated,
            reference=target_eval,
            drift_samples=q["hamiltonian_drift_samples"],
            potential=potential,
            source_reference=source_eval,
            evaluation_config=evaluation_config,
        )
    )
    return row


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
