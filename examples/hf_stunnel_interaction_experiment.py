from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import MultivariateNormal

from whfm import (
    CrowdNavObstaclePotential,
    KernelInteractionPotential,
    MeanStdBVPGaussianPath,
    RationalQuadraticInteractionKernel,
)
from whfm.optimal_transport import OTPlanSampler


class ZeroLinearPotential:
    def energy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[:-1], dtype=x.dtype, device=x.device)

    def gradient(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)


def resolve_device(requested: str) -> tuple[torch.device, str]:
    if not requested.startswith("cuda"):
        return torch.device(requested), f"Using requested device {requested}."
    if not torch.cuda.is_available():
        return torch.device("cpu"), f"Requested {requested}, but CUDA is unavailable; using CPU."

    parts = requested.split(":", 1)
    requested_index = int(parts[1]) if len(parts) == 2 and parts[1] else 0
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    count = torch.cuda.device_count()
    if requested_index < count:
        return torch.device(requested), f"Using requested device {requested}."
    if visible and count > 0:
        return (
            torch.device("cuda:0"),
            f"Requested {requested}; CUDA_VISIBLE_DEVICES={visible} exposes {count} device(s), "
            "so using cuda:0 within the masked process.",
        )
    return torch.device("cpu"), f"Requested {requested}, but only {count} CUDA devices are visible; using CPU."


def sample_gaussian(mean: torch.Tensor, std: float, n: int) -> torch.Tensor:
    dim = mean.numel()
    dist = MultivariateNormal(mean, std**2 * torch.eye(dim, dtype=mean.dtype))
    return dist.sample((n,))


def compute_stunnel_grid(potential, dtype: torch.dtype):
    x_grid = np.linspace(-13, 13, 180)
    y_grid = np.linspace(-9, 9, 140)
    X, Y = np.meshgrid(x_grid, y_grid)
    xy = torch.tensor(np.stack([X.ravel(), Y.ravel()], axis=1), dtype=dtype)
    with torch.no_grad():
        Z = potential.energy(xy).detach().cpu().numpy().reshape(Y.shape)
    return X, Y, Z


def plot_single_path(X, Y, Z, state: torch.Tensor, output_dir: Path):
    dim = 2
    traj = state[:, :dim].detach().cpu().numpy()
    sigma = state[:, 2 * dim].detach().cpu().numpy()
    t = np.linspace(0.0, 1.0, state.shape[0])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].contourf(X, Y, Z, levels=35, cmap="RdBu_r", alpha=0.55)
    axes[0].plot(traj[:, 0], traj[:, 1], color="black", linewidth=2.0)
    axes[0].scatter(traj[[0, -1], 0], traj[[0, -1], 1], c=["steelblue", "tomato"], s=45)
    axes[0].set_title("Single Gaussian mean path")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(-13, 13)
    axes[0].set_ylim(-9, 9)

    axes[1].plot(t, sigma, color="darkgreen", linewidth=2.0)
    axes[1].set_title("Single Gaussian sigma(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("sigma")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    path = output_dir / "single_gaussian_interaction.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_batch_paths(X, Y, Z, states: torch.Tensor, output_dir: Path):
    dim = 2
    traj = states[:, :, :dim].detach().cpu().numpy()
    sigma = states[:, :, 2 * dim].detach().cpu().numpy()
    t = np.linspace(0.0, 1.0, states.shape[1])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].contourf(X, Y, Z, levels=35, cmap="RdBu_r", alpha=0.55)
    for i in range(traj.shape[0]):
        axes[0].plot(traj[i, :, 0], traj[i, :, 1], color="black", alpha=0.35, linewidth=0.8)
    axes[0].scatter(traj[:, 0, 0], traj[:, 0, 1], c="steelblue", s=10, label="source")
    axes[0].scatter(traj[:, -1, 0], traj[:, -1, 1], c="tomato", s=10, label="target")
    axes[0].legend(markerscale=2)
    axes[0].set_title("Filtered stunnel+interaction BVP paths")
    axes[0].set_xlabel("x1")
    axes[0].set_ylabel("x2")
    axes[0].set_xlim(-13, 13)
    axes[0].set_ylim(-9, 9)

    for i in range(sigma.shape[0]):
        axes[1].plot(t, sigma[i], color="darkgreen", alpha=0.25, linewidth=0.8)
    axes[1].plot(t, sigma.mean(axis=0), color="black", linewidth=2.0, label="mean")
    axes[1].set_title("Batch sigma(t)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("sigma")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.tight_layout()
    path = output_dir / "stunnel_interaction_batch.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def state_metrics(states: torch.Tensor) -> dict:
    dim = 2
    sigma = states[:, :, 2 * dim]
    mu = states[:, :, :dim]
    diffs = mu[:, 1:] - mu[:, :-1]
    lengths = torch.linalg.vector_norm(diffs, dim=-1).sum(dim=-1)
    return {
        "n_paths": int(states.shape[0]),
        "sigma_min": float(sigma.min().item()),
        "sigma_max": float(sigma.max().item()),
        "sigma_mid_mean": float(sigma[:, sigma.shape[1] // 2].mean().item()),
        "path_length_mean": float(lengths.mean().item()),
        "path_length_min": float(lengths.min().item()),
        "path_length_max": float(lengths.max().item()),
    }


def write_report(report_path: Path, metrics: dict, args, device_message: str, figures: dict):
    failure_preview = metrics["batch"].get("failure_preview", [])
    lines = [
        "# Stunnel + Interaction Gaussian BVP Report",
        "",
        "## Setup",
        "",
        f"- Device request: `{args.device}`",
        f"- Device resolution: {device_message}",
        f"- Source mean: `[-11, -1]`; target mean: `[11, 1]`",
        f"- Gaussian sampling std: `{args.gaussian_std}`",
        f"- Path endpoint sigma: `{args.sigma_path}`",
        f"- Interaction kernel: `W(x-y) = 2 / (||x-y||^2 + 1)`",
        f"- Interaction coefficient: `{args.interaction_coefficient}`",
        f"- Sigma initial guess bump: `{args.sigma_guess_bump}`",
        f"- BVP settings: n_steps={args.n_steps}, tol={args.tol}, max_nodes={args.max_nodes}, quadrature_order={args.quadrature_order}",
        "",
        "## Single Gaussian Interaction-Only BVP",
        "",
        f"- Sigma min/max: `{metrics['single']['sigma_min']:.6g}` / `{metrics['single']['sigma_max']:.6g}`",
        f"- Midpoint sigma: `{metrics['single']['sigma_mid_mean']:.6g}`",
        f"- Mean path length: `{metrics['single']['path_length_mean']:.6g}`",
        f"- Figure: `{figures['single']}`",
        "",
        "## Stunnel + Interaction Batch BVP",
        "",
        f"- Requested pairs: `{metrics['batch']['requested_pairs']}`",
        f"- Successful pairs: `{metrics['batch']['n_paths']}`",
        f"- Failed pairs: `{metrics['batch']['failed_pairs']}`",
        f"- Sigma min/max: `{metrics['batch']['sigma_min']:.6g}` / `{metrics['batch']['sigma_max']:.6g}`",
        f"- Midpoint sigma mean: `{metrics['batch']['sigma_mid_mean']:.6g}`",
        f"- Mean path length: `{metrics['batch']['path_length_mean']:.6g}`",
        f"- Figure: `{figures['batch']}`",
        "",
        "## Notes",
        "",
        "- Non-converged BVP samples are filtered before plotting or downstream training.",
        "- The interaction term is evaluated deterministically with Gauss-Hermite quadrature using `z1-z2 ~ N(0, 2I)`.",
    ]
    if failure_preview:
        lines.extend(["", "## Failure Preview", ""])
        for idx, message in failure_preview:
            lines.append(f"- Pair {idx}: {message}")
    report_path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run stunnel plus rational-quadratic interaction BVP experiment.")
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-dataset", type=int, default=24)
    parser.add_argument("--n-steps", type=int, default=40)
    parser.add_argument("--tol", type=float, default=0.5)
    parser.add_argument("--max-nodes", type=int, default=2000)
    parser.add_argument("--quadrature-order", type=int, default=5)
    parser.add_argument("--sigma-path", type=float, default=0.01)
    parser.add_argument("--gaussian-std", type=float, default=0.5)
    parser.add_argument("--interaction-coefficient", type=float, default=1.0)
    parser.add_argument("--sigma-guess-bump", type=float, default=0.5)
    parser.add_argument(
        "--output-dir",
        default="examples/stunnel_interaction_results",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_default_dtype(torch.float64)
    device, device_message = resolve_device(args.device)
    print(device_message)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dim = 2
    source_mean = torch.tensor([-11.0, -1.0], dtype=torch.float64)
    target_mean = torch.tensor([11.0, 1.0], dtype=torch.float64)
    stunnel = CrowdNavObstaclePotential("stunnel")
    X, Y, Z = compute_stunnel_grid(stunnel, dtype=torch.float64)

    interaction = KernelInteractionPotential(
        RationalQuadraticInteractionKernel(),
        coupling_samples=torch.zeros(2, 2, dim, dtype=torch.float64),
        sigma_as_kernel_alpha=False,
    )

    grid = torch.linspace(0.0, 1.0, args.n_steps + 1, dtype=torch.float64)
    sigma_profile = args.sigma_path + args.sigma_guess_bump * 4.0 * grid * (1.0 - grid)
    sigma_dot_profile = args.sigma_guess_bump * 4.0 * (1.0 - 2.0 * grid)

    single_path = MeanStdBVPGaussianPath(
        ZeroLinearPotential(),
        sigma=args.sigma_path,
        n_steps=args.n_steps,
        tol=args.tol,
        max_nodes=args.max_nodes,
        quadrature_order=args.quadrature_order,
        sigma_guess=sigma_profile.reshape(1, -1),
        sigma_dot_guess=sigma_dot_profile.reshape(1, -1),
        interaction_potential=interaction,
        interaction_coefficient=args.interaction_coefficient,
    )
    single_states = single_path.batch_solve(source_mean.reshape(1, -1), target_mean.reshape(1, -1))

    x0_all = sample_gaussian(source_mean, args.gaussian_std, args.n_dataset)
    x1_all = sample_gaussian(target_mean, args.gaussian_std, args.n_dataset)
    x0_coupled, x1_coupled = OTPlanSampler(method="exact").sample_plan(x0_all, x1_all)

    batch_sigma_guess = sigma_profile.reshape(1, -1).repeat(args.n_dataset, 1)
    batch_sigma_dot_guess = sigma_dot_profile.reshape(1, -1).repeat(args.n_dataset, 1)
    batch_path = MeanStdBVPGaussianPath(
        stunnel,
        sigma=args.sigma_path,
        n_steps=args.n_steps,
        tol=args.tol,
        max_nodes=args.max_nodes,
        quadrature_order=args.quadrature_order,
        sigma_guess=batch_sigma_guess,
        sigma_dot_guess=batch_sigma_dot_guess,
        interaction_potential=interaction,
        interaction_coefficient=args.interaction_coefficient,
    )
    batch_states = batch_path.batch_solve(x0_coupled, x1_coupled)
    keep = batch_path.success_mask
    x0_coupled = x0_coupled[keep]
    x1_coupled = x1_coupled[keep]
    print(f"Kept {x0_coupled.shape[0]} / {args.n_dataset} stunnel+interaction BVP pairs.")

    figures = {
        "single": str(plot_single_path(X, Y, Z, single_states[0], output_dir)),
        "batch": str(plot_batch_paths(X, Y, Z, batch_states, output_dir)),
    }

    metrics = {
        "single": state_metrics(single_states),
        "batch": state_metrics(batch_states),
    }
    metrics["batch"].update(
        {
            "requested_pairs": int(args.n_dataset),
            "failed_pairs": int((~batch_path.success_mask).sum().item()),
            "success_indices": batch_path.success_indices.tolist(),
            "failed_indices": batch_path.failed_indices.tolist(),
            "failure_preview": list(batch_path.failure_messages.items())[:5],
        }
    )
    metrics["device_message"] = device_message
    metrics["resolved_device"] = str(device)
    metrics["args"] = vars(args)

    metrics_path = output_dir / "stunnel_interaction_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    report_path = Path("examples/stunnel_interaction_report.md")
    write_report(report_path, metrics, args, device_message, figures)

    print(f"Saved metrics to {metrics_path}")
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
