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
        "latest_warmup_loss": latest_warmup_loss,
        "sample_path": str(sample_path),
    }
    row.update(
        distribution_summary(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        )
    )
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
        "latest_loss": latest_loss,
    }
    row.update({key: value for key, value in q.items() if key != "hamiltonian_drift_samples"})
    row.update(
        distribution_summary(
            generated,
            target_eval,
            evaluation_config.num_sliced_projections,
            generator=generator,
        )
    )
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
