"""Model rollout evaluation and trajectory summaries."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Iterable

import torch
from torch import Tensor

from ..node import NodeSolver, call_velocity_model
from .distribution import MMD_loss, sinkhorn, sliced_wasserstein2
from .plotting import save_evaluation_plots, save_warmup_plots


def cap_pair(x: Tensor, y: Tensor, max_samples: int) -> tuple[Tensor, Tensor]:
    n = min(int(max_samples), x.shape[0], y.shape[0])
    if n <= 0:
        raise ValueError("evaluation requires at least one sample.")
    return x[:n], y[:n]


def _metric_float(value) -> float:
    if isinstance(value, Tensor):
        return float(value.detach().cpu())
    return float(value)


def distribution_summary(
    generated: Tensor,
    target: Tensor,
    num_sliced_projections: int,
    *,
    generator=None,
) -> Dict[str, float]:
    generated_flat = generated.reshape(generated.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    mmd = MMD_loss()(generated_flat, target_flat)
    return {
        "sliced_w2": _metric_float(
            sliced_wasserstein2(
                generated,
                target,
                num_sliced_projections,
                generator=generator,
            )
        ),
        "sinkhorn": _metric_float(sinkhorn(generated, target)),
        "mmd": _metric_float(mmd),
    }


@torch.no_grad()
def integrate_model(model, node_solver: NodeSolver, x0: Tensor):
    return node_solver.integrate(model, x0)


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


def _artifact_metadata(
    *,
    stage: str,
    direction: str,
    model_kind: str,
    epoch: int | None = None,
    rectification_index: int | None = None,
) -> Dict[str, object]:
    metadata = {
        "stage": stage,
        "direction": direction,
        "model_kind": model_kind,
    }
    if epoch is not None:
        metadata["epoch"] = int(epoch)
    if rectification_index is not None:
        metadata["rectification_index"] = int(rectification_index)
    return metadata


def save_evaluation_artifacts(
    *,
    samples_dir: Path,
    tag: str,
    stage: str,
    direction: str,
    model_kind: str,
    time_grid: Tensor,
    states: Tensor,
    start: Tensor,
    generated_terminal: Tensor,
    reference_terminal: Tensor,
    epoch: int | None = None,
    rectification_index: int | None = None,
) -> Dict[str, str]:
    """Save trajectory and terminal endpoint artifacts for one evaluation."""
    trajectory_dir = samples_dir / "trajectories"
    terminal_dir = samples_dir / "terminals"
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    terminal_dir.mkdir(parents=True, exist_ok=True)
    metadata = _artifact_metadata(
        stage=stage,
        direction=direction,
        model_kind=model_kind,
        epoch=epoch,
        rectification_index=rectification_index,
    )
    trajectory_path = trajectory_dir / f"{tag}_trajectory.pt"
    terminal_path = terminal_dir / f"{tag}_terminal.pt"
    torch.save(
        {
            **metadata,
            "time_grid": time_grid.detach().cpu(),
            "states": states.detach().cpu(),
            "start": start.detach().cpu(),
            "reference_terminal": reference_terminal.detach().cpu(),
        },
        trajectory_path,
    )
    torch.save(
        {
            **metadata,
            "generated_terminal": generated_terminal.detach().cpu(),
            "num_eval_samples": int(generated_terminal.shape[0]),
        },
        terminal_path,
    )
    return {"trajectory_path": str(trajectory_path), "terminal_path": str(terminal_path)}


def load_terminal_output(path: str | Path) -> Tensor:
    """Load a generated terminal tensor from a terminal artifact file."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "generated_terminal" not in payload:
        raise ValueError(f"terminal artifact does not contain generated_terminal: {path}")
    terminal = payload["generated_terminal"]
    if not isinstance(terminal, Tensor):
        raise TypeError(f"generated_terminal in {path} is not a tensor.")
    return terminal.detach().cpu()


def rectification_residual(current_terminal: Tensor, previous_terminal: Tensor) -> float:
    """Mean squared distance between matching generated terminal endpoints."""
    current_flat = current_terminal.detach().cpu().reshape(current_terminal.shape[0], -1)
    previous_flat = previous_terminal.detach().cpu().reshape(previous_terminal.shape[0], -1)
    if current_flat.shape != previous_flat.shape:
        raise ValueError(
            "rectification residual requires matching terminal shapes; "
            f"got current {tuple(current_flat.shape)} and previous {tuple(previous_flat.shape)}."
        )
    squared_distance = (current_flat - previous_flat).pow(2).sum(dim=-1)
    return float(squared_distance.mean().detach().cpu())


def rectification_residual_from_file(
    current_terminal: Tensor,
    previous_terminal_path: str | Path | None,
) -> float:
    if previous_terminal_path is None:
        return float("nan")
    return rectification_residual(current_terminal, load_terminal_output(previous_terminal_path))


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
    epoch: int | None = None,
    previous_terminal_path: str | Path | None = None,
    save_plots: bool = True,
    plot_mode: str = "full",
) -> Dict[str, object]:
    row, _ = evaluate_warmup_model_with_terminal(
        model=model,
        model_kind=model_kind,
        direction=direction,
        node_solver=node_solver,
        potential=potential,
        source_test=source_test,
        target_test=target_test,
        evaluation_config=evaluation_config,
        latest_warmup_loss=latest_warmup_loss,
        figures_dir=figures_dir,
        samples_dir=samples_dir,
        epoch=epoch,
        previous_terminal_path=previous_terminal_path,
        save_plots=save_plots,
        plot_mode=plot_mode,
    )
    return row


def evaluate_warmup_model_with_terminal(
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
    epoch: int | None = None,
    previous_terminal_path: str | Path | None = None,
    save_plots: bool = True,
    plot_mode: str = "full",
) -> tuple[Dict[str, object], Path]:
    source_eval, target_eval = cap_pair(source_test, target_test, evaluation_config.max_metric_samples)
    traj = integrate_model(model, node_solver, source_eval)
    generated = traj.states[-1]
    generator = torch.Generator(device=source_eval.device)
    generator.manual_seed(811)
    tag = (
        f"warmup_epoch{int(epoch)}_{direction}_{model_kind}"
        if epoch is not None
        else f"warmup_final_{direction}_{model_kind}"
    )
    plot_paths = {
        "trajectory_plot": "",
        "linear_potential_plot": "",
        "terminal_scatter_plot": "",
    }
    if save_plots:
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
            plot_mode=plot_mode,
        )
    artifact_paths = save_evaluation_artifacts(
        samples_dir=samples_dir,
        tag=tag,
        stage="warmup",
        direction=direction,
        model_kind=model_kind,
        epoch=epoch,
        time_grid=traj.time_grid,
        states=traj.states,
        start=source_eval,
        generated_terminal=generated,
        reference_terminal=target_eval,
    )
    previous_path_value = "" if previous_terminal_path is None else str(previous_terminal_path)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "epoch": "" if epoch is None else int(epoch),
        "direction": direction,
        "model_kind": model_kind,
        "num_eval_samples": int(source_eval.shape[0]),
        "latest_warmup_loss": latest_warmup_loss,
        "sample_path": artifact_paths["trajectory_path"],
        "previous_terminal_path": previous_path_value,
        "rectification_residual": rectification_residual_from_file(
            generated.detach().cpu(),
            previous_terminal_path,
        ),
    }
    row.update(artifact_paths)
    row.update(
        distribution_summary(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        )
    )
    row.update(plot_paths)
    return row, Path(artifact_paths["terminal_path"])


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
    samples_dir: Path | None = None,
    previous_terminal_path: str | Path | None = None,
) -> Dict[str, object]:
    row, _ = evaluate_model_with_terminal(
        model=model,
        model_kind=model_kind,
        direction=direction,
        rectification_index=rectification_index,
        node_solver=node_solver,
        potential=potential,
        source_test=source_test,
        target_test=target_test,
        evaluation_config=evaluation_config,
        latest_loss=latest_loss,
        bridge_metrics=bridge_metrics,
        figures_dir=figures_dir,
        samples_dir=samples_dir,
        previous_terminal_path=previous_terminal_path,
    )
    return row


def evaluate_model_with_terminal(
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
    samples_dir: Path | None = None,
    previous_terminal_path: str | Path | None = None,
) -> tuple[Dict[str, object], Path | None]:
    source_eval, target_eval = cap_pair(source_test, target_test, evaluation_config.max_metric_samples)
    traj = integrate_model(model, node_solver, source_eval)
    generated = traj.states[-1]
    generator = torch.Generator(device=source_eval.device)
    generator.manual_seed(17 + int(rectification_index))
    q = trajectory_quantities(model, potential, traj, traj.time_grid)
    tag = f"rectification_r{rectification_index}_{direction}_{model_kind}"
    artifact_paths: Dict[str, str] = {}
    terminal_path = None
    if samples_dir is not None:
        artifact_paths = save_evaluation_artifacts(
            samples_dir=samples_dir,
            tag=tag,
            stage="rectification",
            direction=direction,
            model_kind=model_kind,
            rectification_index=rectification_index,
            time_grid=traj.time_grid,
            states=traj.states,
            start=source_eval,
            generated_terminal=generated,
            reference_terminal=target_eval,
        )
        terminal_path = Path(artifact_paths["terminal_path"])
    previous_path_value = "" if previous_terminal_path is None else str(previous_terminal_path)
    row = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "rectification_index": rectification_index,
        "direction": direction,
        "model_kind": model_kind,
        "num_eval_samples": int(source_eval.shape[0]),
        "latest_loss": latest_loss,
        "previous_terminal_path": previous_path_value,
        "rectification_residual": rectification_residual_from_file(
            generated.detach().cpu(),
            previous_terminal_path,
        ),
    }
    row.update(artifact_paths)
    row.update({key: value for key, value in q.items() if key != "hamiltonian_drift_samples"})
    row.update(
        distribution_summary(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        )
    )
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
    return row, terminal_path
