"""Postprocessing plots for completed WHFM trainer runs."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor

from .plotting import (
    PUBLICATION_DPI,
    PUBLICATION_FIGSIZE,
    PUBLICATION_TERMINAL_MARKER_SIZE,
    _apply_axis_limits,
    _particle_plot_descriptor,
    _plot_positions,
    _publication_subplots,
    _set_position_labels,
    _style_axis,
    _style_legend,
)

RECTIFICATION_SERIES_METRICS = (
    "sliced_w2",
    "hamiltonian_drift_integral_mean",
    "action_mean",
    "kinetic_integral_mean",
    "potential_integral_mean",
    "linear_potential_integral_mean",
    "rectification_residual",
    "latest_loss",
)

METRIC_TITLES = {
    "sliced_w2": "Terminal Distribution Error",
    "hamiltonian_drift_integral_mean": "Hamiltonian Drift Along Learned Trajectories",
    "action_mean": "Mean Action of Learned Trajectories",
    "kinetic_integral_mean": "Kinetic Energy Along Learned Trajectories",
    "potential_integral_mean": "Potential Energy Along Learned Trajectories",
    "linear_potential_integral_mean": "Obscatle Avoidance Along Learned Trajectories",
    "rectification_residual": "Change Between Consecutive Rectifications",
    "latest_loss": "Training Loss Across Rectifications",
}

METRIC_YLABELS = {
    "sliced_w2": r"$SW_2(\nu,\tilde{\nu})$",
    "hamiltonian_drift_integral_mean": r"$\mathbb{E}\,\mathcal{D}_H[Z]$",
    "action_mean": r"$\mathbb{E}\,\mathcal{A}[Z]$",
    "kinetic_integral_mean": r"$\int_0^1 \mathbb{E}_{x\sim\mu}\left[\frac{1}{2}\|Z_t'(x)\|^2\right]\,dt$",
    "potential_integral_mean": r"$\int_0^1 \mathcal{F}[\rho_t]\,dt$",
    "linear_potential_integral_mean": r"$\mathbb{E}\int_0^1 V(Z_t(x))\,dt$",
    "rectification_residual": r"$\|\pi-\mathcal{R}^H(\pi)\|$",
    "latest_loss": r"$\mathcal{L}_{CFM}$",
}


def _as_finite_float(value) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _as_int(value) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_terminal_output(path: str | Path) -> Tensor:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "generated_terminal" not in payload:
        raise ValueError(f"terminal artifact does not contain generated_terminal: {path}")
    terminal = payload["generated_terminal"]
    if not isinstance(terminal, Tensor):
        raise TypeError(f"generated_terminal in {path} is not a tensor.")
    return terminal.detach().cpu()


def _row_label(row: dict) -> str:
    direction = str(row.get("direction", "direction"))
    model_kind = str(row.get("model_kind", "model"))
    direction_label = direction.replace("_", " ").title()
    if model_kind == "ema":
        model_label = "EMA"
    else:
        model_label = model_kind.replace("_", " ")
    return f"{direction_label} {model_label}"


def _safe_row_tag(row: dict) -> str:
    rectification = row.get("rectification_index", "unknown")
    direction = str(row.get("direction", "direction")).replace("/", "_")
    model_kind = str(row.get("model_kind", "model")).replace("/", "_")
    return f"r{rectification}_{direction}_{model_kind}"


def plot_rectification_metric_series(
    run_dir: str | Path,
    rows: Iterable[dict],
    *,
    metrics: Iterable[str] = RECTIFICATION_SERIES_METRICS,
) -> dict[str, str]:
    """Plot rectification metrics against rectification index for saved rows."""
    rows = list(rows)
    output_dir = Path(run_dir) / "figures" / "rectification_metrics"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"plot_error": f"plot unavailable: {exc}"}

    paths: dict[str, str] = {}
    for metric in metrics:
        grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for row in rows:
            rectification = _as_int(row.get("rectification_index"))
            value = _as_finite_float(row.get(metric))
            if rectification is None or value is None:
                continue
            grouped[_row_label(row)].append((rectification, value))
        if not grouped:
            continue

        fig, ax = _publication_subplots(plt, figsize=PUBLICATION_FIGSIZE)
        for label, values in sorted(grouped.items()):
            ordered = sorted(values, key=lambda item: item[0])
            x_values = [item[0] for item in ordered]
            y_values = [item[1] for item in ordered]
            ax.plot(x_values, y_values, marker="o", label=label)
        ax.set_xlabel("Rectification step")
        ax.set_ylabel(METRIC_YLABELS.get(metric, metric))
        _style_legend(ax)
        _style_axis(ax, title=METRIC_TITLES.get(metric, metric))
        path = output_dir / f"{metric}_by_rectification.png"
        fig.savefig(path, bbox_inches="tight", dpi=PUBLICATION_DPI)
        plt.close(fig)
        paths[metric] = str(path)
    return paths


def plot_rectification_residual_samples(
    run_dir: str | Path,
    rows: Iterable[dict],
    potential,
    evaluation_config,
) -> dict[str, str]:
    """Plot current and previous generated terminal samples used by residuals."""
    rows = list(rows)
    output_dir = Path(run_dir) / "figures" / "rectification_residual_samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        return {"plot_error": f"plot unavailable: {exc}"}

    paths: dict[str, str] = {}
    for row in rows:
        terminal_path = row.get("terminal_path")
        previous_terminal_path = row.get("previous_terminal_path")
        if not terminal_path or not previous_terminal_path:
            continue
        current = _load_terminal_output(terminal_path)
        previous = _load_terminal_output(previous_terminal_path)
        if current.shape != previous.shape:
            raise ValueError(
                "residual sample plotting requires matching terminal shapes; "
                f"got current {tuple(current.shape)} and previous {tuple(previous.shape)}."
            )

        count = min(int(getattr(evaluation_config, "plot_trajectory_count", current.shape[0])), current.shape[0])
        if count <= 0:
            continue
        descriptor = _particle_plot_descriptor(potential, current.shape[-1])
        current_plot = _plot_positions(current[:count], evaluation_config, descriptor).detach().cpu().numpy()
        previous_plot = _plot_positions(previous[:count], evaluation_config, descriptor).detach().cpu().numpy()

        fig, ax = _publication_subplots(plt, figsize=PUBLICATION_FIGSIZE)
        ax.scatter(
            previous_plot[..., 0].reshape(-1),
            previous_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.72,
            label="Previous rectification",
        )
        ax.scatter(
            current_plot[..., 0].reshape(-1),
            current_plot[..., 1].reshape(-1),
            s=PUBLICATION_TERMINAL_MARKER_SIZE,
            alpha=0.72,
            label="Current rectification",
        )
        _set_position_labels(ax, evaluation_config, descriptor)
        _apply_axis_limits(ax, evaluation_config)
        _style_legend(ax)
        tag = _safe_row_tag(row)
        _style_axis(ax, title="Terminal Samples Used for Rectification Residual")
        path = output_dir / f"{tag}_residual_samples.png"
        fig.savefig(path, bbox_inches="tight", dpi=PUBLICATION_DPI)
        plt.close(fig)
        paths[tag] = str(path)
    return paths


def postprocess_rectification_results(
    run_dir: str | Path,
    rows: Iterable[dict],
    potential,
    evaluation_config,
) -> dict[str, object]:
    """Run all postprocessing plots for a completed rectification run."""
    rows = list(rows)
    return {
        "metric_series_plots": plot_rectification_metric_series(run_dir, rows),
        "rectification_residual_sample_plots": plot_rectification_residual_samples(
            run_dir,
            rows,
            potential,
            evaluation_config,
        ),
    }
