"""CLI helpers for evaluating saved HFM models and writing rollout GIFs."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from ...models.models_v2 import FourierTimeResidualMLP
from ..Potentials import ConfiguredPotential
from ..config import (
    BoundaryConfig,
    ProblemConfig,
    TrainConfig,
    load_problem_config,
    load_train_config,
)
from ..datasets import _dataset_samples
from ..node import NodeSolver
from .checkpoint import load_checkpoint
from .model import cap_pair, distribution_summary, trajectory_quantities
from .plotting import (
    PUBLICATION_GIF_CURRENT_MARKER_SIZE,
    PUBLICATION_GIF_FIGSIZE,
    PUBLICATION_GIF_STATIC_MARKER_SIZE,
    POTENTIAL_COLORMAP,
    PUBLICATION_TITLE_SIZE,
    _apply_publication_style,
    _linear_contour_values,
    _particle_plot_descriptor,
    _plot_positions,
    _projected_plot_domain,
    _resolve_plot_domain,
    _set_position_labels,
    _style_axis,
    _style_legend,
)


def resolve_checkpoint_path(run_dir: str | Path, checkpoint: str | Path | None = None) -> Path:
    """Resolve a checkpoint path from a run directory and optional CLI value."""

    run_dir = Path(run_dir)
    if checkpoint is None:
        candidates = [run_dir / "checkpoints" / "final.pt"]
    else:
        raw = Path(checkpoint)
        if raw.is_absolute():
            candidates = [raw]
        else:
            names = [raw]
            if raw.suffix == "":
                names.append(raw.with_suffix(".pt"))
            candidates = []
            for name in names:
                candidates.extend([name, run_dir / name, run_dir / "checkpoints" / name])

    for path in candidates:
        if path.exists():
            return path
    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"could not resolve checkpoint; searched: {searched}")


def available_model_specs(
    checkpoint: dict,
    *,
    direction: str = "all",
    model_kind: str = "all",
) -> list[tuple[str, str]]:
    """Return ``(direction, model_kind)`` pairs available in a trainer checkpoint."""

    if direction not in {"forward", "backward", "all"}:
        raise ValueError("direction must be 'forward', 'backward', or 'all'.")
    if model_kind not in {"online", "ema", "all"}:
        raise ValueError("model_kind must be 'online', 'ema', or 'all'.")

    direction_states = checkpoint.get("directions")
    if not isinstance(direction_states, dict) or not direction_states:
        raise ValueError("checkpoint does not contain any direction states.")

    ordered_directions = [
        value
        for value in ("forward", "backward")
        if value in direction_states
    ]
    ordered_directions.extend(
        value for value in direction_states if value not in {"forward", "backward"}
    )

    specs: list[tuple[str, str]] = []
    for direction_name in ordered_directions:
        if direction != "all" and direction_name != direction:
            continue
        state = direction_states[direction_name]
        if model_kind in {"online", "all"} and state.get("model") is not None:
            specs.append((direction_name, "online"))
        if model_kind in {"ema", "all"} and state.get("ema") is not None:
            ema_state = state["ema"]
            if isinstance(ema_state, dict) and ema_state.get("ema_model") is not None:
                specs.append((direction_name, "ema"))
    if not specs:
        raise ValueError("no checkpoint models matched the requested filters.")
    return specs


def create_particle_evolution_gifs(
    run_dir: str | Path,
    *,
    checkpoint: str | Path | None = None,
    direction: str = "all",
    model_kind: str = "all",
    num_samples: int | None = None,
    device: str | None = None,
    output_dir: str | Path | None = None,
    fps: int = 12,
    dpi: int = 160,
    node_steps: int | None = None,
    potential_background: bool = True,
) -> list[dict]:
    """Load a trainer run, evaluate saved models, and create rollout GIFs."""

    run_dir = Path(run_dir)
    train_config = load_train_config(run_dir / "resolved_train.yaml")
    problem_config = load_problem_config(run_dir / "resolved_problem.yaml")
    dtype = _resolve_dtype(train_config.dtype)
    resolved_device = _resolve_device(device or train_config.device)

    checkpoint_path = resolve_checkpoint_path(run_dir, checkpoint)
    checkpoint_state = load_checkpoint(checkpoint_path, map_location=resolved_device)
    specs = available_model_specs(checkpoint_state, direction=direction, model_kind=model_kind)

    potential = ConfiguredPotential(problem_config.functional.to_potential_cfg())
    node_solver = NodeSolver(
        train_config.node_solver.method,
        node_steps=train_config.node_solver.node_steps if node_steps is None else int(node_steps),
    )
    source_test, target_test = _prepare_evaluation_split(
        train_config,
        problem_config,
        device=resolved_device,
        dtype=dtype,
    )
    if source_test.shape[0] == 0 or target_test.shape[0] == 0:
        raise ValueError("run configuration has no held-out evaluation samples.")

    sample_count = int(num_samples or problem_config.evaluation.max_metric_samples)
    if sample_count <= 0:
        raise ValueError("num_samples must be positive.")

    figures_dir = _resolve_output_dir(run_dir, output_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = run_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for direction_name, kind in specs:
        model = _build_model(train_config, problem_config, device=resolved_device, dtype=dtype)
        model_state = _model_state_dict(checkpoint_state, direction_name, kind)
        model.load_state_dict(model_state)
        model.eval()

        if direction_name == "forward":
            start, reference = cap_pair(source_test, target_test, sample_count)
        elif direction_name == "backward":
            start, reference = cap_pair(target_test, source_test, sample_count)
        else:
            raise ValueError(f"unsupported direction in checkpoint: {direction_name!r}")

        with torch.no_grad():
            traj = node_solver.integrate(model, start)
        generated = traj.states[-1]

        tag = f"{direction_name}_{kind}_{checkpoint_path.stem}"
        gif_path = figures_dir / f"{tag}_evolution.gif"
        _save_particle_evolution_gif(
            gif_path=gif_path,
            tag=tag,
            traj=traj.states,
            time_grid=traj.time_grid,
            generated=generated,
            reference=reference,
            potential=potential,
            source_reference=start,
            evaluation_config=problem_config.evaluation,
            fps=fps,
            dpi=dpi,
            potential_background=potential_background,
        )

        generator = torch.Generator(device=start.device)
        generator.manual_seed(9001 + len(rows))
        quantities = trajectory_quantities(model, potential, traj, traj.time_grid)
        row = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_dir": str(run_dir),
            "checkpoint": str(checkpoint_path),
            "direction": direction_name,
            "model_kind": kind,
            "num_eval_samples": int(start.shape[0]),
            "gif_path": str(gif_path),
        }
        row.update(
            distribution_summary(
                generated,
                reference,
                problem_config.evaluation.num_sliced_projections,
                generator=generator,
            )
        )
        row.update(
            {
                key: value
                for key, value in quantities.items()
                if key != "hamiltonian_drift_samples"
            }
        )
        rows.append(_json_ready(row))

    manifest_path = metrics_dir / "evolution_gifs.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate saved HFM checkpoints and create particle evolution GIFs."
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
        help="Direction to animate. Defaults to all directions stored in the checkpoint.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("online", "ema", "all"),
        default="all",
        help="Model kind to animate. Defaults to online plus EMA when present.",
    )
    parser.add_argument("--num-samples", type=int, default=None, help="Number of held-out samples to animate.")
    parser.add_argument("--device", default=None, help="Evaluation device override, e.g. cpu or cuda:0.")
    parser.add_argument("--output-dir", default=None, help="Directory for GIFs. Defaults under the run directory.")
    parser.add_argument("--fps", type=int, default=12, help="GIF frames per second.")
    parser.add_argument("--dpi", type=int, default=160, help="Animation DPI.")
    parser.add_argument(
        "--node-steps",
        type=int,
        default=None,
        help="Override NODE integration steps for animation. Defaults to the saved train config.",
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
    rows = create_particle_evolution_gifs(
        args.run_dir,
        checkpoint=args.checkpoint,
        direction=args.direction,
        model_kind=args.model_kind,
        num_samples=args.num_samples,
        device=args.device,
        output_dir=args.output_dir,
        fps=args.fps,
        dpi=args.dpi,
        node_steps=args.node_steps,
        potential_background=not args.no_potential_background,
    )
    for row in rows:
        print(f"{row['direction']} {row['model_kind']}: {row['gif_path']}")
    return rows


def _resolve_output_dir(run_dir: Path, output_dir: str | Path | None) -> Path:
    if output_dir is None:
        return run_dir / "figures" / "evolution_gifs"
    return Path(output_dir)


def _model_state_dict(checkpoint: dict, direction: str, model_kind: str) -> dict:
    state = checkpoint["directions"][direction]
    if model_kind == "online":
        return state["model"]
    if model_kind == "ema":
        return state["ema"]["ema_model"]
    raise ValueError("model_kind must be 'online' or 'ema'.")


def _build_model(
    train_config: TrainConfig,
    problem_config: ProblemConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    return FourierTimeResidualMLP(
        dim=problem_config.dimension,
        out_dim=problem_config.dimension,
        w=train_config.model.width,
        hidden=train_config.model.hidden,
        m=train_config.model.fourier_modes,
        time_varying=True,
    ).to(device=device, dtype=dtype)


def _prepare_evaluation_split(
    train_config: TrainConfig,
    problem_config: ProblemConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    cfg = train_config.data
    source = _sample_boundary(
        problem_config.boundaries.source,
        problem_config,
        cfg.total_samples,
        seed=train_config.seed,
        device=device,
        dtype=dtype,
    )
    target = _sample_boundary(
        problem_config.boundaries.target,
        problem_config,
        cfg.total_samples,
        seed=train_config.seed + 1,
        device=device,
        dtype=dtype,
    )
    n_train = int(round(cfg.total_samples * (1.0 - cfg.test_fraction)))
    generator = torch.Generator(device=device)
    generator.manual_seed(train_config.seed + 2)
    source_perm = torch.randperm(cfg.total_samples, generator=generator, device=device)
    generator.manual_seed(train_config.seed + 3)
    target_perm = torch.randperm(cfg.total_samples, generator=generator, device=device)
    return source[source_perm[n_train:]], target[target_perm[n_train:]]


def _sample_boundary(
    boundary: BoundaryConfig,
    problem_config: ProblemConfig,
    n_samples: int,
    *,
    seed: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    x, _ = _dataset_samples(boundary.name, seed, n_samples, boundary.parameters)
    x = x.to(device=device, dtype=dtype)
    if x.shape[-1] != problem_config.dimension:
        raise ValueError(
            f"Boundary dataset '{boundary.name}' returned dimension {x.shape[-1]}, "
            f"expected {problem_config.dimension}."
        )
    if boundary.sample_noise > 0.0:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 104729)
        x = x + boundary.sample_noise * torch.randn(
            x.shape,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    return x


def _dynamic_linear_contour_frame_values(
    potential,
    frame_reference: Tensor,
    evaluation_config,
    grid_x: Tensor,
    grid_y: Tensor,
    particle_descriptor,
) -> tuple[np.ndarray, np.ndarray] | None:
    if not getattr(potential, "has_linear", False):
        return None

    with torch.no_grad():
        values = _linear_contour_values(
            potential,
            frame_reference,
            evaluation_config,
            grid_x,
            grid_y,
            particle_descriptor,
        )
    values_np = values.detach().cpu().numpy()
    finite_values = values_np[np.isfinite(values_np)]
    if finite_values.size == 0:
        return None
    vmin, vmax = np.percentile(finite_values, [2.0, 98.0])
    vmin = float(vmin)
    vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(finite_values.min())
        vmax = float(finite_values.max())
    if vmin == vmax:
        delta = max(abs(vmin) * 0.01, 1.0)
        vmin -= delta
        vmax += delta
    levels = np.linspace(vmin, vmax, 35)
    return values_np, levels


def _save_particle_evolution_gif(
    *,
    gif_path: Path,
    tag: str,
    traj: Tensor,
    time_grid: Tensor,
    generated: Tensor,
    reference: Tensor,
    potential,
    source_reference: Tensor,
    evaluation_config,
    fps: int,
    dpi: int,
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
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        raise RuntimeError(f"GIF plotting dependencies are unavailable: {exc}") from exc

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = _particle_plot_descriptor(potential, traj.shape[-1])
    low, high = _projected_plot_domain(
        (traj, source_reference, generated, reference),
        evaluation_config,
        traj.shape[-1],
        particle_descriptor=descriptor,
    )
    padding = (high - low).abs().max().clamp_min(torch.as_tensor(1.0, device=high.device, dtype=high.dtype))
    low = low - 0.02 * padding
    high = high + 0.02 * padding
    low, high = _resolve_plot_domain(evaluation_config, low, high)

    traj_plot = _plot_positions(traj, evaluation_config, descriptor).detach().cpu().numpy()
    source_plot = _plot_positions(source_reference, evaluation_config, descriptor).detach().cpu().numpy()
    generated_plot = _plot_positions(generated, evaluation_config, descriptor).detach().cpu().numpy()
    reference_plot = _plot_positions(reference, evaluation_config, descriptor).detach().cpu().numpy()
    t_np = time_grid.detach().cpu().numpy()

    dynamic_background = None
    grid_x = None
    grid_y = None
    grid_x_np = None
    grid_y_np = None
    if potential_background and getattr(potential, "has_linear", False):
        xs = torch.linspace(low[0], high[0], 80, device=traj.device, dtype=traj.dtype)
        ys = torch.linspace(low[1], high[1], 80, device=traj.device, dtype=traj.dtype)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="xy")
        dynamic_background = _dynamic_linear_contour_frame_values(
            potential,
            traj[0],
            evaluation_config,
            grid_x,
            grid_y,
            descriptor,
        )
        if dynamic_background is not None:
            grid_x_np = grid_x.detach().cpu().numpy()
            grid_y_np = grid_y.detach().cpu().numpy()

    _apply_publication_style(plt)
    fig, ax = plt.subplots(figsize=PUBLICATION_GIF_FIGSIZE, constrained_layout=True)
    contour_sets = []

    def draw_contour(frame: int):
        if dynamic_background is None:
            return []
        frame_background = _dynamic_linear_contour_frame_values(
            potential,
            traj[frame],
            evaluation_config,
            grid_x,
            grid_y,
            descriptor,
        )
        if frame_background is None:
            return []
        values_np, levels = frame_background
        contour_fill = ax.contourf(
            grid_x_np,
            grid_y_np,
            values_np,
            levels=levels,
            alpha=0.25,
            cmap=POTENTIAL_COLORMAP,
            zorder=0,
            extend="both",
        )
        contour_lines = ax.contour(
            grid_x_np,
            grid_y_np,
            values_np,
            levels=levels,
            colors="0.20",
            linewidths=0.8,
            alpha=0.85,
            zorder=1,
        )
        return [contour_fill, contour_lines]

    def clear_contours():
        for contour_set in contour_sets:
            for artist in getattr(contour_set, "collections", ()):
                artist.remove()
        contour_sets.clear()

    contour_sets.extend(draw_contour(0))
    ax.scatter(
        _flatten_positions(source_plot)[0],
        _flatten_positions(source_plot)[1],
        s=PUBLICATION_GIF_STATIC_MARKER_SIZE,
        alpha=0.25,
        zorder=4,
        label=r"$\mu$",
    )
    ax.scatter(
        _flatten_positions(reference_plot)[0],
        _flatten_positions(reference_plot)[1],
        s=PUBLICATION_GIF_STATIC_MARKER_SIZE,
        alpha=0.35,
        zorder=4,
        label=r"$\nu$",
    )
    ax.scatter(
        _flatten_positions(generated_plot)[0],
        _flatten_positions(generated_plot)[1],
        s=PUBLICATION_GIF_STATIC_MARKER_SIZE,
        alpha=0.35,
        zorder=4,
        label=r"$\tilde{\nu}$",
    )
    current = ax.scatter(
        [],
        [],
        s=PUBLICATION_GIF_CURRENT_MARKER_SIZE,
        alpha=0.9,
        zorder=5,
        label=r"$Z_t$",
    )
    text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=PUBLICATION_TITLE_SIZE,
        zorder=6,
    )
    ax.set_xlim(float(low[0].detach().cpu()), float(high[0].detach().cpu()))
    ax.set_ylim(float(low[1].detach().cpu()), float(high[1].detach().cpu()))
    # ax.set_aspect("equal", adjustable="box")
    _set_position_labels(ax, evaluation_config, descriptor)
    _style_axis(ax, title="Particle evolution")
    _style_legend(ax, loc="best", markerscale=1.2)
    layout_engine = fig.get_layout_engine() if hasattr(fig, "get_layout_engine") else None
    if layout_engine is not None:
        layout_engine.set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.02)

    def update(frame: int):
        if dynamic_background is not None:
            clear_contours()
            contour_sets.extend(draw_contour(frame))
        x, y = _flatten_positions(traj_plot[frame])
        current.set_offsets(np.column_stack([x, y]))
        text.set_text(f"t={float(t_np[frame]):.3f}")
        return current, text

    animation = FuncAnimation(fig, update, frames=traj_plot.shape[0], interval=1000.0 / fps, blit=False)
    try:
        animation.save(gif_path, writer=PillowWriter(fps=fps), dpi=dpi)
    except Exception as exc:  # pragma: no cover - depends on optional writer install
        raise RuntimeError(f"failed to write GIF with Pillow writer: {exc}") from exc
    finally:
        plt.close(fig)
    return gif_path


def _flatten_positions(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(values).reshape(-1, 2)
    return points[:, 0], points[:, 1]


def _resolve_dtype(name: str) -> torch.dtype:
    if name == "float64":
        return torch.float64
    if name == "float32":
        return torch.float32
    raise ValueError("dtype must be 'float32' or 'float64'.")


def _resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def _json_ready(row: dict) -> dict:
    clean = {}
    for key, value in row.items():
        if isinstance(value, Tensor):
            if value.numel() != 1:
                clean[key] = value.detach().cpu().tolist()
            else:
                clean[key] = float(value.detach().cpu())
        elif isinstance(value, np.generic):
            clean[key] = value.item()
        else:
            clean[key] = value
    return clean


if __name__ == "__main__":
    main()
