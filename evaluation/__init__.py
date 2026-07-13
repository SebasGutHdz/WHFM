"""Evaluation metrics, model summaries, plots, and CSV helpers."""

from .distribution import _median_bandwidth, rbf_mmd2, sliced_wasserstein2
from .io import (
    METRIC_FIELDNAMES,
    WARMUP_METRIC_FIELDNAMES,
    append_metrics_row,
    append_warmup_metrics_row,
)
from .model import (
    bridge_summary,
    cap_pair,
    covariance_matrix,
    evaluate_model,
    evaluate_warmup_model,
    integrate_model,
    terminal_summary,
    trajectory_quantities,
)
from .plotting import (
    POTENTIAL_COLORMAP,
    TIME_COLORMAP,
    _particle_plot_descriptor,
    _plot_linear_contour,
    _plot_positions,
    _project,
    _projected_plot_domain,
    _projection_indices,
    _reshape_particle_positions,
    save_bridge_solution_plots,
    save_evaluation_plots,
    save_warmup_plots,
)

__all__ = [
    "POTENTIAL_COLORMAP",
    "TIME_COLORMAP",
    "METRIC_FIELDNAMES",
    "WARMUP_METRIC_FIELDNAMES",
    "_median_bandwidth",
    "_particle_plot_descriptor",
    "_plot_linear_contour",
    "_plot_positions",
    "_project",
    "_projected_plot_domain",
    "_projection_indices",
    "_reshape_particle_positions",
    "append_metrics_row",
    "append_warmup_metrics_row",
    "bridge_summary",
    "cap_pair",
    "covariance_matrix",
    "evaluate_model",
    "evaluate_warmup_model",
    "integrate_model",
    "rbf_mmd2",
    "save_bridge_solution_plots",
    "save_evaluation_plots",
    "save_warmup_plots",
    "sliced_wasserstein2",
    "terminal_summary",
    "trajectory_quantities",
]
