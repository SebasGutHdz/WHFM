"""CLI helpers for plotting single-sample Gaussian bridge solutions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import Tensor

from ..Potentials import ConfiguredPotential
from ..bridge import GaussianBridgeSolver
from ..config import load_problem_config, load_train_config
from ..node import NodeSolver
from .animate import (
    _build_model,
    _json_ready,
    _model_state_dict,
    _prepare_evaluation_split,
    _resolve_device,
    _resolve_dtype,
    available_model_specs,
    resolve_checkpoint_path,
)
from .plotting import (
    PUBLICATION_DPI,
    PUBLICATION_GIF_FIGSIZE,
    PUBLICATION_GIF_STATIC_MARKER_SIZE,
    PUBLICATION_LINE_WIDTH,
    PUBLICATION_SQUARE_FIGSIZE,
    PUBLICATION_TERMINAL_MARKER_SIZE,
    TIME_COLORMAP,
    _apply_publication_style,
    _particle_plot_descriptor,
    _plot_linear_contour,
    _plot_positions,
    _projected_plot_domain,
    _publication_colorbar,
    _publication_subplots,
    _resolve_plot_domain,
    _set_position_labels,
    _style_axis,
    _style_legend,
)


def create_gaussian_bridge_visualizations(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    direction: str = "all",
    model_kind: str = "all",
    sample_index: int = 0,
    device: str | None = None,
    output_dir: str | Path | None = None,
    node_steps: int | None = None,
    fps: int = 12,
    dpi: int = 160,
    num_frames: int = 8,
    num_cloud_samples: int = 300,
    density_levels: str | list[float] | tuple[float, ...] = "1,2,3",
    visual_seed: int = 12345,
    std_visual_scale: float = 1.0,
    potential_background: bool = True,
) -> list[dict]:
    """Create static and animated plots for one rollout-induced Gaussian BVP."""

    run_dir = Path(run_dir)
    train_config = load_train_config(run_dir / "resolved_train.yaml")
    problem_config = load_problem_config(run_dir / "resolved_problem.yaml")
    dtype = _resolve_dtype(train_config.dtype)
    resolved_device = _resolve_device(device or train_config.device)

    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    checkpoint_state = torch.load(checkpoint_path, map_location=resolved_device, weights_only=False)
    specs = available_model_specs(checkpoint_state, direction=direction, model_kind=model_kind)

    potential = ConfiguredPotential(problem_config.functional.to_potential_cfg())
    node_solver = NodeSolver(
        train_config.node_solver.method,
        node_steps=train_config.node_solver.node_steps if node_steps is None else int(node_steps),
    )
    bridge_solver = _build_bridge_solver(train_config, potential)
    source_test, target_test = _prepare_evaluation_split(
        train_config,
        problem_config,
        device=resolved_device,
        dtype=dtype,
    )

    sample_index = int(sample_index)
    if sample_index < 0:
        raise ValueError("sample_index must be non-negative.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    density_level_values = _parse_density_levels(density_levels)
    num_cloud_samples = int(num_cloud_samples)
    if num_cloud_samples <= 0:
        raise ValueError("num_cloud_samples must be positive.")
    visual_seed = int(visual_seed)
    std_visual_scale = float(std_visual_scale)
    if not np.isfinite(std_visual_scale) or std_visual_scale <= 0.0:
        raise ValueError("std_visual_scale must be positive and finite.")

    figures_dir = _resolve_output_dir(run_dir, output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for direction_name, kind in specs:
        model = _build_model(train_config, problem_config, device=resolved_device, dtype=dtype)
        model.load_state_dict(_model_state_dict(checkpoint_state, direction_name, kind))
        model.eval()

        if direction_name == "forward":
            pool = source_test
        elif direction_name == "backward":
            pool = target_test
        else:
            raise ValueError(f"unsupported direction in checkpoint: {direction_name!r}")
        if sample_index >= pool.shape[0]:
            raise ValueError(
                f"sample_index {sample_index} is out of range for {direction_name} "
                f"evaluation split with {pool.shape[0]} samples."
            )

        x0 = pool[sample_index : sample_index + 1]
        with torch.no_grad():
            node_trajectory = node_solver.integrate(model, x0)
        x1 = node_trajectory.states[-1]
        mean_guess, velocity_guess = node_solver.prepare_bridge_guess(
            node_trajectory.states,
            node_trajectory.velocities,
            x0,
            x1,
            reverse=False,
            bridge_steps=train_config.bridge_solver.bridge_steps,
        )
        solution = bridge_solver.solve_batch(
            x0,
            x1,
            mean_guess=mean_guess,
            mean_velocity_guess=velocity_guess,
        )
        if solution.num_successful <= 0:
            raise RuntimeError(
                "Gaussian BVP solve failed for "
                f"{direction_name}/{kind} in checkpoint {checkpoint_path}."
            )

        tag = f"{direction_name}_{kind}_{checkpoint_path.stem}_sample{sample_index}_gaussian_bridge"
        static_path = figures_dir / f"{tag}.png"
        gif_path = figures_dir / f"{tag}.gif"
        _save_gaussian_bridge_static_plot(
            output_path=static_path,
            tag=tag,
            solution=solution,
            potential=potential,
            evaluation_config=problem_config.evaluation,
            num_frames=num_frames,
            density_levels=density_level_values,
            std_visual_scale=std_visual_scale,
            potential_background=potential_background,
        )
        _save_gaussian_bridge_gif(
            gif_path=gif_path,
            tag=tag,
            solution=solution,
            potential=potential,
            evaluation_config=problem_config.evaluation,
            fps=fps,
            dpi=dpi,
            num_cloud_samples=num_cloud_samples,
            visual_seed=visual_seed,
            std_visual_scale=std_visual_scale,
            potential_background=potential_background,
        )

        rows.append(
            _json_ready(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "run_dir": str(run_dir),
                    "checkpoint": str(checkpoint_path),
                    "direction": direction_name,
                    "model_kind": kind,
                    "sample_index": sample_index,
                    "static_path": str(static_path),
                    "gif_path": str(gif_path),
                    "num_bridge_steps": int(solution.time_grid.numel() - 1),
                    "num_static_frames": len(_snapshot_indices(solution.time_grid.numel(), num_frames)),
                    "num_cloud_samples": num_cloud_samples,
                    "density_levels": density_level_values,
                    "visual_seed": visual_seed,
                    "std_visual_scale": std_visual_scale,
                    "endpoint_errors": solution.endpoint_errors[0],
                    "solve_time_seconds": float(solution.solve_time_seconds),
                }
            )
        )

    manifest_path = metrics_dir / "gaussian_bridge_paths.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one learned rollout-induced Gaussian bridge per saved HFM model."
    )
    parser.add_argument("run_dir", help="Trainer run directory containing resolved configs and checkpoints.")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path, filename, or stem. Defaults to checkpoints/final.pt.",
    )
    parser.add_argument(
        "--direction",
        choices=("forward", "backward", "all"),
        default="all",
        help="Direction to plot. Defaults to all directions stored in the checkpoint.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("online", "ema", "all"),
        default="all",
        help="Model kind to plot. Defaults to online plus EMA when present.",
    )
    parser.add_argument("--sample-index", type=int, default=0, help="Held-out initial sample index.")
    parser.add_argument("--device", default=None, help="Evaluation device override, e.g. cpu or cuda:0.")
    parser.add_argument("--output-dir", default=None, help="Directory for plots. Defaults under the run directory.")
    parser.add_argument(
        "--node-steps",
        type=int,
        default=None,
        help="Override NODE integration steps for the learned rollout guess.",
    )
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--dpi", type=int, default=160, help="GIF DPI.")
    parser.add_argument(
        "--num-frames",
        type=int,
        default=8,
        help="Number of evenly spaced BVP times shown in the static plot.",
    )
    parser.add_argument(
        "--num-cloud-samples",
        type=int,
        default=300,
        help="Gaussian samples per mean center in each GIF frame.",
    )
    parser.add_argument(
        "--density-levels",
        default="1,2,3",
        help="Comma-separated std radii for static filled confidence regions.",
    )
    parser.add_argument(
        "--visual-seed",
        type=int,
        default=12345,
        help="Local RNG seed for deterministic Gaussian visualization samples.",
    )
    parser.add_argument(
        "--std-visual-scale",
        type=float,
        default=1.0,
        help="Visualization-only multiplier for Gaussian std in density regions and GIF clouds.",
    )
    parser.add_argument(
        "--no-potential-background",
        action="store_true",
        help="Disable the linear potential contour background.",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    rows = create_gaussian_bridge_visualizations(
        args.run_dir,
        checkpoint=args.checkpoint,
        direction=args.direction,
        model_kind=args.model_kind,
        sample_index=args.sample_index,
        device=args.device,
        output_dir=args.output_dir,
        node_steps=args.node_steps,
        fps=args.fps,
        dpi=args.dpi,
        num_frames=args.num_frames,
        num_cloud_samples=args.num_cloud_samples,
        density_levels=args.density_levels,
        visual_seed=args.visual_seed,
        std_visual_scale=args.std_visual_scale,
        potential_background=not args.no_potential_background,
    )
    for row in rows:
        print(f"{row['direction']} {row['model_kind']}: {row['static_path']} {row['gif_path']}")
    return rows


def _build_bridge_solver(train_config, potential) -> GaussianBridgeSolver:
    cfg = train_config.bridge_solver
    return GaussianBridgeSolver(
        potential,
        sigma_source=cfg.sigma_source,
        sigma_target=cfg.sigma_target,
        bridge_steps=cfg.bridge_steps,
        tol=cfg.tol,
        max_nodes=cfg.max_nodes,
        quadrature_order=cfg.quadrature_order,
        use_monte_carlo=cfg.use_monte_carlo,
        monte_carlo_samples=cfg.monte_carlo_samples,
        num_workers=getattr(cfg, "num_workers", 1),
        failure_policy=cfg.failure_policy,
    )


def _resolve_output_dir(run_dir: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return run_dir / "figures" / "gaussian_bridge_paths"
    return Path(output_dir)


def _snapshot_indices(num_time_points: int, num_frames: int) -> list[int]:
    if num_time_points <= 0:
        raise ValueError("num_time_points must be positive.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    count = min(int(num_frames), int(num_time_points))
    values = np.linspace(0, int(num_time_points) - 1, count)
    return sorted(set(int(round(value)) for value in values))


def _parse_density_levels(value) -> list[float]:
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw_values = list(value)
    if not raw_values:
        raise ValueError("density_levels must contain at least one value.")
    levels = []
    for raw in raw_values:
        level = float(raw)
        if not np.isfinite(level) or level <= 0.0:
            raise ValueError("density_levels must be positive finite values.")
        levels.append(level)
    return sorted(set(levels))


def _sample_gaussian_clouds(
    mean_plot: np.ndarray,
    std: np.ndarray,
    *,
    num_cloud_samples: int,
    seed: int,
    std_visual_scale: float = 1.0,
) -> np.ndarray:
    num_cloud_samples = int(num_cloud_samples)
    if num_cloud_samples <= 0:
        raise ValueError("num_cloud_samples must be positive.")
    centers = np.asarray(mean_plot, dtype=np.float64).reshape(mean_plot.shape[0], -1, 2)
    std_values = np.asarray(std, dtype=np.float64).reshape(-1)
    if centers.shape[0] != std_values.shape[0]:
        raise ValueError("mean_plot and std must have matching time dimensions.")
    std_visual_scale = float(std_visual_scale)
    if not np.isfinite(std_visual_scale) or std_visual_scale <= 0.0:
        raise ValueError("std_visual_scale must be positive and finite.")
    std_values = std_values * std_visual_scale
    rng = np.random.default_rng(int(seed))
    noise = rng.standard_normal((centers.shape[0], centers.shape[1], num_cloud_samples, 2))
    samples = centers[:, :, None, :] + std_values[:, None, None, None] * noise
    return samples.reshape(centers.shape[0], centers.shape[1] * num_cloud_samples, 2)


def _save_gaussian_bridge_static_plot(
    *,
    output_path: Path,
    tag: str,
    solution,
    potential,
    evaluation_config,
    num_frames: int,
    density_levels: list[float] | tuple[float, ...],
    std_visual_scale: float,
    potential_background: bool,
) -> Path:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError(f"Gaussian bridge plotting dependencies are unavailable: {exc}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    bridge = _first_bridge_arrays(solution, potential, evaluation_config, std_visual_scale=std_visual_scale)
    indices = _snapshot_indices(bridge.time_grid.shape[0], num_frames)
    cmap = plt.get_cmap(TIME_COLORMAP)
    norm = plt.Normalize(vmin=float(bridge.time_grid[0]), vmax=float(bridge.time_grid[-1]))

    fig, ax = _publication_subplots(plt, figsize=PUBLICATION_SQUARE_FIGSIZE)
    _draw_bridge_background(
        ax,
        bridge=bridge,
        potential=potential,
        evaluation_config=evaluation_config,
        potential_background=potential_background,
    )
    sorted_levels = sorted(density_levels, reverse=True)
    for plot_count, index in enumerate(indices):
        color = cmap(norm(float(bridge.time_grid[index])))
        centers = _flatten_positions(bridge.mean_plot[index])
        density_label = "Gaussian confidence regions" if plot_count == 0 else None
        for center in centers:
            for rank, level in enumerate(sorted_levels):
                alpha = 0.05 + 0.08 * (len(sorted_levels) - rank - 1)
                ax.add_patch(
                    Circle(
                        (float(center[0]), float(center[1])),
                        float(level * bridge.std[index] * std_visual_scale),
                        edgecolor=color,
                        facecolor=color,
                        alpha=alpha,
                        linewidth=0.8,
                        label=density_label,
                        zorder=2 + 0.1 * rank,
                    )
                )
                density_label = None
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            color=color,
            edgecolors="black",
            linewidths=0.1,
            alpha=0.1,
            label="mean snapshots" if plot_count == 0 else None,
            zorder=1,
        )

    _draw_endpoints(ax, bridge)
    _publication_colorbar(fig, ax, plt.cm.ScalarMappable(norm=norm, cmap=cmap), label="t")
    _set_position_labels(ax, evaluation_config, bridge.descriptor)
    _style_axis(ax, title="Gaussian bridge")
    _style_legend(ax, markerscale=1.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=PUBLICATION_DPI)
    plt.close(fig)
    return output_path


def _save_gaussian_bridge_gif(
    *,
    gif_path: Path,
    tag: str,
    solution,
    potential,
    evaluation_config,
    fps: int,
    dpi: int,
    num_cloud_samples: int,
    visual_seed: int,
    std_visual_scale: float,
    potential_background: bool,
) -> Path:
    if fps <= 0:
        raise ValueError("fps must be positive.")
    if dpi <= 0:
        raise ValueError("dpi must be positive.")
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
        from matplotlib.patches import Circle
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError(f"Gaussian bridge GIF dependencies are unavailable: {exc}") from exc

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    bridge = _first_bridge_arrays(solution, potential, evaluation_config, std_visual_scale=std_visual_scale)
    cmap = plt.get_cmap(TIME_COLORMAP)
    norm = plt.Normalize(vmin=float(bridge.time_grid[0]), vmax=float(bridge.time_grid[-1]))

    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=PUBLICATION_GIF_FIGSIZE, constrained_layout=True)
    _draw_bridge_background(
        ax,
        bridge=bridge,
        potential=potential,
        evaluation_config=evaluation_config,
        potential_background=potential_background,
    )
    _draw_endpoints(ax, bridge)
    clouds = _sample_gaussian_clouds(
        bridge.mean_plot,
        bridge.std,
        num_cloud_samples=num_cloud_samples,
        seed=visual_seed,
        std_visual_scale=std_visual_scale,
    )
    initial_centers = _flatten_positions(bridge.mean_plot[0])
    initial_color = cmap(norm(float(bridge.time_grid[0])))
    cloud = ax.scatter(
        clouds[0, :, 0],
        clouds[0, :, 1],
        s=12,
        color=initial_color,
        alpha=0.48,
        linewidths=0,
        label="Gaussian samples",
        zorder=3,
    )
    current = ax.scatter(
        initial_centers[:, 0],
        initial_centers[:, 1],
        s=PUBLICATION_GIF_STATIC_MARKER_SIZE * 1.1,
        color=initial_color,
        edgecolors="black",
        linewidths=0.5,
        label="current mean",
        zorder=5,
    )
    text = ax.text(0.02, 0.98, "", transform=ax.transAxes, va="top")
    _set_position_labels(ax, evaluation_config, bridge.descriptor)
    _style_axis(ax, title="Gaussian bridge")
    _style_legend(ax, markerscale=1.2)

    def update(frame: int):
        centers = _flatten_positions(bridge.mean_plot[frame])
        color = cmap(norm(float(bridge.time_grid[frame])))
        cloud.set_offsets(clouds[frame])
        cloud.set_color([color])
        current.set_offsets(centers)
        current.set_color([color])
        text.set_text(f"t={float(bridge.time_grid[frame]):.3f}")
        return [cloud, current, text]

    animation = FuncAnimation(fig, update, frames=bridge.mean_plot.shape[0], interval=1000.0 / fps, blit=False)
    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    except Exception as exc:  # pragma: no cover - depends on optional writer install
        raise RuntimeError(f"failed to write Gaussian bridge GIF with Pillow writer: {exc}") from exc
    finally:
        plt.close(fig)
    return gif_path


def _first_bridge_arrays(solution, potential, evaluation_config, *, std_visual_scale: float = 1.0):
    if solution.num_successful <= 0:
        raise ValueError("Gaussian bridge solution contains no successful pairs.")
    mean = solution.mean[:1]
    std = solution.std[0, :, 0].detach().cpu().numpy()
    x0 = solution.x0[:1]
    x1 = solution.x1[:1]
    descriptor = _particle_plot_descriptor(potential, mean.shape[-1])
    mean_plot = _plot_positions(mean[0], evaluation_config, descriptor).detach().cpu().numpy()
    x0_plot = _plot_positions(x0, evaluation_config, descriptor).detach().cpu().numpy()
    x1_plot = _plot_positions(x1, evaluation_config, descriptor).detach().cpu().numpy()
    time_grid = solution.time_grid.detach().cpu().numpy()
    low, high = _projected_plot_domain(
        (mean, x0, x1),
        evaluation_config,
        mean.shape[-1],
        particle_descriptor=descriptor,
    )
    std_visual_scale = float(std_visual_scale)
    if not np.isfinite(std_visual_scale) or std_visual_scale <= 0.0:
        raise ValueError("std_visual_scale must be positive and finite.")
    radius = 3.0 * std_visual_scale * solution.std[0, :, 0].detach().max()
    low = low - radius
    high = high + radius
    low, high = _resolve_plot_domain(evaluation_config, low, high)
    return SimpleNamespace(
        mean=mean,
        std=std,
        x0=x0,
        x1=x1,
        mean_plot=mean_plot,
        x0_plot=x0_plot,
        x1_plot=x1_plot,
        time_grid=time_grid,
        low=low.detach().cpu().numpy(),
        high=high.detach().cpu().numpy(),
        descriptor=descriptor,
    )


def _draw_bridge_background(
    ax,
    *,
    bridge,
    potential,
    evaluation_config,
    potential_background: bool,
) -> None:
    if potential_background:
        _plot_linear_contour(
            ax,
            potential,
            bridge.x0,
            evaluation_config,
            domain_tensors=(bridge.mean, bridge.x0, bridge.x1),
        )
    ax.set_xlim(float(bridge.low[0]), float(bridge.high[0]))
    ax.set_ylim(float(bridge.low[1]), float(bridge.high[1]))


def _draw_endpoints(ax, bridge) -> None:
    start = _flatten_positions(bridge.x0_plot)
    terminal = _flatten_positions(bridge.x1_plot)
    ax.scatter(
        start[:, 0],
        start[:, 1],
        s=PUBLICATION_TERMINAL_MARKER_SIZE * 1.25,
        color="black",
        marker="o",
        label="start",
        zorder=6,
    )
    ax.scatter(
        terminal[:, 0],
        terminal[:, 1],
        s=PUBLICATION_TERMINAL_MARKER_SIZE * 1.25,
        color="white",
        edgecolors="black",
        marker="s",
        linewidths=1.1,
        label="NODE terminal",
        zorder=6,
    )


def _flatten_positions(values: np.ndarray) -> np.ndarray:
    return np.asarray(values).reshape(-1, 2)


if __name__ == "__main__":
    main()
