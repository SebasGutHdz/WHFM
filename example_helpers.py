from __future__ import annotations

from typing import Callable, Optional

import torch
from torch import Tensor

from .losses import flow_matching_loss
from .gaussian_paths import MeanStdBVPGaussianPath


def to_numpy(x):
    return x.detach().cpu().numpy()


def as_particles(q: Tensor, n_particles: int, particle_dim: int) -> Tensor:
    return q.reshape(*q.shape[:-1], n_particles, particle_dim)


def make_hamiltonian_node(model, *, sensitivity: str = "adjoint", solver: str = "euler"):
    from torchdyn.core import NeuralODE
    from torchcfm.utils import torch_wrapper
    return NeuralODE(torch_wrapper(model), sensitivity=sensitivity, solver=solver)


def make_mean_std_bvp_path(potential, *, sigma, n_steps: int, tol: float, quadrature_order: int, **kwargs):
    return MeanStdBVPGaussianPath(
        potential,
        sigma=sigma,
        n_steps=n_steps,
        tol=tol,
        quadrature_order=quadrature_order,
        **kwargs,
    )


def solve_bvp_paths(make_path: Callable, x0: Tensor, x1: Tensor, *, label: str = "path", description: str = "BVPs"):
    path = make_path()
    print(f"Solving {x0.shape[0]} {label} {description}...")
    states = path.batch_solve(x0, x1)
    keep = path.success_mask.to(device=x0.device)
    x0_keep = x0[keep]
    x1_keep = x1[keep]
    n_failed = int((~keep).sum().item())
    print(f"{label}: kept {x0_keep.shape[0]} / {keep.numel()} BVPs; failed {n_failed}; states: {states.shape}")
    if n_failed:
        preview = list(path.failure_messages.items())[:5]
        print(f"{label}: first failures: {preview}")
    return path, x0_keep, x1_keep, states

def sample_time_endpoint_biased(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    p_uniform: float = 0.50,
    beta_concentration: float = 0.1,
    eps: float = 1e-4,
):
    """
    Samples t in (0,1), biased toward both endpoints.

    p_uniform controls how much ordinary uniform sampling remains.
    beta_concentration < 1 gives more mass near 0 and 1.
    """
    shape = (batch_size, 1)

    # Uniform samples.
    t_uniform = torch.rand(shape, device=device, dtype=dtype)

    # U-shaped beta samples: mass near 0 and 1.
    alpha = torch.tensor(beta_concentration, device=device, dtype=dtype)
    beta = torch.tensor(beta_concentration, device=device, dtype=dtype)
    beta_dist = torch.distributions.Beta(alpha, beta)
    t_beta = beta_dist.sample(shape).to(device=device, dtype=dtype)

    # Mixture mask.
    use_uniform = torch.rand(shape, device=device, dtype=dtype) < p_uniform

    t = torch.where(use_uniform, t_uniform, t_beta)

    # Avoid exact endpoints if your formulas divide by t, 1-t, sigma_t, etc.
    return t.clamp(eps, 1.0 - eps)


def train_on_cached_path_pairs(
    model,
    optimizer,
    path,
    x0: Tensor,
    x1: Tensor,
    n_steps_train: int,
    label: str,
    ema: None,
    *,
    batch_size: int,
    device=None,
    log_every: int = 200,
    scheduler: Optional[object] = None,
    no_pairs_message: str = "no successful BVP pairs to train on",
):
    model.train()
    step_losses = []
    n_pairs = x0.shape[0]
    if n_pairs == 0:
        raise RuntimeError(f"{label}: {no_pairs_message}.")
    device = x0.device if device is None else device
    x0 = x0.to(device)
    x1 = x1.to(device)
    for step in range(n_steps_train):
        optimizer.zero_grad()
        idx = torch.randint(0, n_pairs, (batch_size,), device=device)
        x0_b = x0[idx]
        x1_b = x1[idx]
        # t = torch.rand((batch_size, 1), device=device, dtype=x0_b.dtype)
        t = sample_time_endpoint_biased(batch_size,device = device,dtype=x0_b.dtype)
        epsilon = torch.randn_like(x0_b)
        xt = path.sample_xt(x0_b, x1_b, t, epsilon)
        ut = path.compute_ut(x0_b, x1_b, t, xt)
        vt = model(torch.cat([xt, t], dim=-1))
        loss = flow_matching_loss(vt, ut)
        loss.backward()
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # if not torch.isfinite(grad_norm):
        #     optimizer.zero_grad(set_to_none=True)
        #     continue
        optimizer.step()
        if ema is not None:
            ema.update()
        if scheduler is not None:
            scheduler.step()
        step_losses.append(loss.item())
        if step % log_every == 0 or step == n_steps_train - 1:
            print(f"{label} step {step:5d}: loss = {loss.item():.5f}")
    return step_losses


def train_on_ot_pairs(
    model,
    optimizer,
    x0: Tensor,
    x1: Tensor,
    n_steps_train: int,
    label: str,
    *,
    batch_size: int,
    device=None,
    log_every: int = 500,
):
    model.train()
    step_losses = []
    n_pairs = x0.shape[0]
    if n_pairs == 0:
        raise RuntimeError(f"{label}: no OT pairs to train on.")
    device = x0.device if device is None else device
    x0 = x0.to(device)
    x1 = x1.to(device)
    for step in range(n_steps_train):
        optimizer.zero_grad()
        idx = torch.randint(0, n_pairs, (batch_size,), device=device)
        x0_b = x0[idx]
        x1_b = x1[idx]
        t = torch.rand((batch_size, 1), device=device, dtype=x0_b.dtype)
        xt = (1.0 - t) * x0_b + t * x1_b
        ut = x1_b - x0_b
        vt = model(torch.cat([xt, t], dim=-1))
        loss = flow_matching_loss(vt, ut)
        loss.backward()
        optimizer.step()
        step_losses.append(loss.item())
        if step % log_every == 0 or step == n_steps_train - 1:
            print(f"{label} step {step:5d}: loss = {loss.item():.5f}")
    return step_losses


def simulate_model_trajectory(model, x0: Tensor, t_span: Tensor, make_node: Callable = make_hamiltonian_node):
    node = make_node(model)
    model.eval()
    with torch.no_grad():
        return node.trajectory(x0, t_span=t_span)


def cached_mean_trajectory(path, x0: Tensor, x1: Tensor, t_span: Tensor):
    means = []
    for t_value in t_span:
        t_batch = torch.full((x0.shape[0], 1), float(t_value), device=x0.device, dtype=x0.dtype)
        mu_t, _ = path.compute(x0, x1, t_batch, return_derivatives=False)
        means.append(mu_t)
    return torch.stack(means, dim=0)


def trajectory_hamiltonian(traj: Tensor, t_span: Tensor, potential):
    dt = t_span[1] - t_span[0]
    velocity = torch.empty_like(traj)
    velocity[0] = (traj[1] - traj[0]) / dt
    velocity[-1] = (traj[-1] - traj[-2]) / dt
    velocity[1:-1] = (traj[2:] - traj[:-2]) / (2 * dt)
    kinetic = 0.5 * velocity.pow(2).sum(dim=-1)
    potential_energy = torch.stack([potential.energy(x_t) for x_t in traj], dim=0)
    return (kinetic + potential_energy).mean(dim=1)
