"""Package-local trainable Hamiltonian Flow Matching v1 trainer."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

import tqdm as tqdm

from ..models.models_v2 import FourierTimeResidualMLP
from .Potentials import ConfiguredPotential
from .bridge import BridgeSolution, GaussianBridgeSolver
from .config import BoundaryConfig, ProblemConfig, TrainConfig, dump_resolved_yaml
from .couplings import Coupler
from .datasets import _dataset_samples
from .directions import Direction, DirectionState
from .evaluation import (
    append_metrics_row,
    append_warmup_metrics_row,
    evaluate_model,
    evaluate_warmup_model,
    save_bridge_solution_plots,
)
from .losses import flow_matching_loss
from .node import NodeSolver, call_velocity_model


@dataclass
class BoundaryProblem:
    """Dataset-backed boundary problem plus configured Hamiltonian potential."""

    config: ProblemConfig
    device: torch.device
    dtype: torch.dtype

    def __post_init__(self):
        self.potential = ConfiguredPotential(self.config.functional.to_potential_cfg())

    @property
    def dimension(self) -> int:
        return self.config.dimension

    def sample_boundary(self, boundary: BoundaryConfig, n_samples: int, *, seed: int) -> Tensor:
        x, _ = _dataset_samples(boundary.name, seed, n_samples, boundary.parameters)
        x = x.to(device=self.device, dtype=self.dtype)
        if x.shape[-1] != self.dimension:
            raise ValueError(
                f"Boundary dataset '{boundary.name}' returned dimension {x.shape[-1]}, "
                f"expected {self.dimension}."
            )
        if boundary.sample_noise > 0.0:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(int(seed) + 104729)
            x = x + boundary.sample_noise * torch.randn(
                x.shape, generator=generator, device=self.device, dtype=self.dtype
            )
        return x

    def sample_source(self, n_samples: int, *, seed: int) -> Tensor:
        return self.sample_boundary(self.config.boundaries.source, n_samples, seed=seed)

    def sample_target(self, n_samples: int, *, seed: int) -> Tensor:
        return self.sample_boundary(self.config.boundaries.target, n_samples, seed=seed)


class HamiltonianTrainer:
    """End-to-end trainable HFM orchestration."""

    def __init__(self, train_config: TrainConfig, problem_config: ProblemConfig):
        self.train_config = train_config
        self.problem_config = problem_config
        self.dtype = self._resolve_dtype(train_config.dtype)
        self.device = self._resolve_device(train_config.device)
        self._set_seed(int(train_config.seed))

        self.problem = BoundaryProblem(problem_config, self.device, self.dtype)
        self.dataset = self._prepare_datasets()
        self.node_solver = NodeSolver(
            train_config.node_solver.method,
            node_steps=train_config.node_solver.node_steps,
        )
        self.initial_coupler = Coupler(train_config.initial_fit.coupling)
        self.rectification_coupler = Coupler("ot")
        self.bridge_solver = GaussianBridgeSolver(
            self.problem.potential,
            sigma=train_config.bridge_solver.sigma,
            bridge_steps=train_config.bridge_solver.bridge_steps,
            tol=train_config.bridge_solver.tol,
            max_nodes=train_config.bridge_solver.max_nodes,
            quadrature_order=train_config.bridge_solver.quadrature_order,
            use_monte_carlo=train_config.bridge_solver.use_monte_carlo,
            monte_carlo_samples=train_config.bridge_solver.monte_carlo_samples,
            failure_policy=train_config.bridge_solver.failure_policy,
        )

        self.mode = train_config.rectification.coupling_generation
        directions = [Direction.FORWARD]
        if self.mode == "opposite_ema":
            directions.append(Direction.BACKWARD)
        self.total_steps_by_direction = self._estimate_total_steps(directions)
        self.states = {direction: self._make_direction_state(direction) for direction in directions}
        self.metrics = {
            "losses": {direction.value: [] for direction in directions},
            "bridge": [],
            "checkpoints": [],
            "diagnostics": [],
            "evaluation": [],
            "warmup": [],
        }
        self.stage = "created"
        self.completed_passes = []
        self._diagnostic_saved = set()
        self.run_dir = self._create_run_dir()
        dump_resolved_yaml(self.run_dir / "resolved_train.yaml", train_config)
        dump_resolved_yaml(self.run_dir / "resolved_problem.yaml", problem_config)

    def train(self) -> Dict:
        self.stage = "initial_fit"
        print("Running initial fit")
        self.run_initial_fit()
        print("Finished initial fit")
        for state in self.states.values():
            state.initialize_ema(self.train_config.ema)
        self.evaluate_warmup()

        self.stage = "rectification"
        print("Starting rectifications")
        self.run_rectifications()
        self.stage = "complete"
        self.save_metrics()
        self.save_checkpoint("final", rectification_index=None, direction=None)
        return self.metrics

    def run_initial_fit(self) -> None:
        cfg = self.train_config.initial_fit
        for epoch in range(cfg.epochs):
            source_loader, target_loader = self._make_loaders(
                self.dataset["source_train"], self.dataset["target_train"]
            )
            for x_source, y_target in zip(source_loader, target_loader):
                x_source = x_source[0].to(self.device, dtype=self.dtype)
                y_target = y_target[0].to(self.device, dtype=self.dtype)
                x0, x1 = self.initial_coupler.pair(x_source, y_target)
                self._train_straight_batch(
                    self.states[Direction.FORWARD],
                    x0,
                    x1,
                    steps=cfg.steps_per_batch,
                )
                if self.mode == "opposite_ema":
                    self._train_straight_batch(
                        self.states[Direction.BACKWARD],
                        x1,
                        x0,
                        steps=cfg.steps_per_batch,
                    )
            self.completed_passes.append({"stage": "initial_fit", "epoch": epoch})
        self.save_checkpoint("after_initial_fit", rectification_index=None, direction=None)

    def run_rectifications(self) -> None:
        rect_cfg = self.train_config.rectification
       
        for rectification_index in range(rect_cfg.num_rectifications):
            print(f"Running rectification {rectification_index}")
            for direction_name in rect_cfg.direction_order:
                direction = Direction(direction_name)
                if direction not in self.states:
                    continue
                self.run_direction_pass(direction, rectification_index)
                if self.train_config.output.checkpoint_every_direction_pass:
                    self.save_checkpoint(
                        "direction_pass",
                        rectification_index=rectification_index,
                        direction=direction,
                    )
            self.evaluate_rectification(rectification_index)

    def run_direction_pass(self, direction: Direction, rectification_index: int) -> None:
        x_train, y_train = self._rectification_subdataset(rectification_index)
        source_loader, target_loader = self._make_loaders(x_train, y_train)
        train_state = self.states[direction]
        generation_state = self._generation_state_for(direction)
        # epoch_data = enumerate(zip(source_loader,target_loader))
        pbar = tqdm.tqdm(zip(source_loader,target_loader), desc=f"Running {direction.value} pass")
        for batch_index, (x_source, y_target) in enumerate(pbar):
            x_source = x_source[0].to(self.device, dtype=self.dtype)
            y_target = y_target[0].to(self.device, dtype=self.dtype)
            if self.mode == "own_ema":
                solution = self._generate_own_forward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            elif direction is Direction.FORWARD:
                solution = self._generate_forward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            else:
                solution = self._generate_backward_bridge(
                    x_source, y_target, generation_state.generation_model
                )
            self._record_bridge_metrics(direction, rectification_index, batch_index, solution)
            if solution.num_successful == 0:
                if self.train_config.bridge_solver.failure_policy == "raise":
                    raise RuntimeError(f"{direction.value}: no successful bridge pairs.")
                continue
            self._train_bridge_solution(
                train_state,
                solution,
                steps=self.train_config.rectification.steps_per_batch,
            )
        self.completed_passes.append(
            {
                "stage": "rectification",
                "rectification_index": rectification_index,
                "direction": direction.value,
            }
        )

    @torch.no_grad()
    def _generate_own_forward_bridge(
        self,
        x_source: Tensor,
        y_target: Tensor,
        generation_model: torch.nn.Module,
    ) -> BridgeSolution:
        trajectory = self.node_solver.integrate(generation_model, x_source)
        generated_target = trajectory.states[-1].detach()
        coupled = self.rectification_coupler.pair_with_labels(
            generated_target,
            y_target,
            y0=trajectory.states.permute(1, 0, 2).contiguous(),
        )
        aligned_traj = coupled.y0.permute(1, 0, 2).contiguous()
        aligned_source = aligned_traj[0].detach()
        aligned_target = coupled.x1.detach()
        velocities = self.node_solver.evaluate_velocities(
            generation_model, aligned_traj, trajectory.time_grid
        )
        mean_guess, velocity_guess = self.node_solver.prepare_bridge_guess(
            aligned_traj,
            velocities,
            aligned_source,
            aligned_target,
            reverse=False,
            bridge_steps=self.train_config.bridge_solver.bridge_steps,
        )
        return self.bridge_solver.solve_batch(
            aligned_source,
            aligned_target,
            mean_guess=mean_guess,
            mean_velocity_guess=velocity_guess,
        )

    @torch.no_grad()
    def _generate_forward_bridge(
        self,
        x_source: Tensor,
        y_target: Tensor,
        generation_model: torch.nn.Module,
    ) -> BridgeSolution:
        trajectory = self.node_solver.integrate(generation_model, y_target)
        generated_source = trajectory.states[-1].detach()
        coupled = self.rectification_coupler.pair_with_labels(
            generated_source,
            x_source,
            y0=trajectory.states.permute(1, 0, 2).contiguous(),
        )
        aligned_traj = coupled.y0.permute(1, 0, 2).contiguous()
        aligned_target = aligned_traj[0].detach()
        aligned_source = coupled.x1.detach()
        velocities = self.node_solver.evaluate_velocities(
            generation_model, aligned_traj, trajectory.time_grid
        )
        mean_guess, velocity_guess = self.node_solver.prepare_bridge_guess(
            aligned_traj,
            velocities,
            aligned_source,
            aligned_target,
            reverse=True,
            bridge_steps=self.train_config.bridge_solver.bridge_steps,
        )
        return self.bridge_solver.solve_batch(
            aligned_source,
            aligned_target,
            mean_guess=mean_guess,
            mean_velocity_guess=velocity_guess,
        )

    @torch.no_grad()
    def _generate_backward_bridge(
        self,
        x_source: Tensor,
        y_target: Tensor,
        generation_model: torch.nn.Module,
    ) -> BridgeSolution:
        trajectory = self.node_solver.integrate(generation_model, x_source)
        generated_target = trajectory.states[-1].detach()
        coupled = self.rectification_coupler.pair_with_labels(
            generated_target,
            y_target,
            y0=trajectory.states.permute(1, 0, 2).contiguous(),
        )
        aligned_traj = coupled.y0.permute(1, 0, 2).contiguous()
        aligned_source = aligned_traj[0].detach()
        aligned_target = coupled.x1.detach()
        velocities = self.node_solver.evaluate_velocities(
            generation_model, aligned_traj, trajectory.time_grid
        )
        mean_guess, velocity_guess = self.node_solver.prepare_bridge_guess(
            aligned_traj,
            velocities,
            aligned_target,
            aligned_source,
            reverse=True,
            bridge_steps=self.train_config.bridge_solver.bridge_steps,
        )
        return self.bridge_solver.solve_batch(
            aligned_target,
            aligned_source,
            mean_guess=mean_guess,
            mean_velocity_guess=velocity_guess,
        )

    def _train_straight_batch(
        self,
        state: DirectionState,
        x0: Tensor,
        x1: Tensor,
        *,
        steps: int,
    ) -> None:
        state.model.train()
        for _ in range(steps):
            state.optimizer.zero_grad(set_to_none=True)
            t = torch.rand((x0.shape[0], 1), device=x0.device, dtype=x0.dtype)
            xt = (1.0 - t) * x0 + t * x1
            if self.train_config.initial_fit.noise_std > 0.0:
                xt = xt + self.train_config.initial_fit.noise_std * torch.randn_like(xt)
            target = x1 - x0
            prediction = call_velocity_model(state.model, xt, t)
            loss = flow_matching_loss(prediction, target)
            self._check_finite(loss, "straight loss")
            loss.backward()
            state.optimizer.step()
            state.scheduler_step()
            self.metrics["losses"][state.direction.value].append(float(loss.detach().cpu()))

    def _train_bridge_solution(
        self,
        state: DirectionState,
        solution: BridgeSolution,
        *,
        steps: int,
    ) -> None:
        state.model.train()
        for _ in range(steps):
            state.optimizer.zero_grad(set_to_none=True)
            batch_size = min(self.train_config.data.batch_size, solution.x0.shape[0])
            idx = torch.randint(solution.x0.shape[0], (batch_size,), device=self.device)
            x0 = solution.x0[idx]
            x1 = solution.x1[idx]
            t = torch.rand((x0.shape[0], 1), device=self.device, dtype=self.dtype)
            epsilon = torch.randn_like(x0)
            xt = solution.path.sample_xt(x0, x1, t, epsilon)
            target = solution.path.compute_ut(x0, x1, t, xt)
            prediction = call_velocity_model(state.model, xt, t)
            loss = flow_matching_loss(prediction, target)
            self._check_finite(loss, "bridge loss")
            loss.backward()
            state.optimizer.step()
            state.scheduler_step()
            if state.ema is not None:
                state.ema.update(state.model)
            self.metrics["losses"][state.direction.value].append(float(loss.detach().cpu()))

    def evaluate_warmup(self) -> None:
        source_test = self.dataset["source_test"]
        target_test = self.dataset["target_test"]
        if source_test.shape[0] == 0 or target_test.shape[0] == 0:
            return
        if self.mode == "own_ema":
            self._evaluate_warmup_direction(Direction.FORWARD, source_test, target_test)
            return
        self._evaluate_warmup_direction(Direction.FORWARD, source_test, target_test)
        self._evaluate_warmup_direction(Direction.BACKWARD, target_test, source_test)

    def _evaluate_warmup_direction(
        self,
        direction: Direction,
        source_test: Tensor,
        target_test: Tensor,
    ) -> None:
        state = self.states[direction]
        losses = self.metrics["losses"].get(direction.value, [])
        latest_loss = losses[-1] if losses else float("nan")
        models = [("online", state.model)]
        if state.ema is not None:
            models.append(("ema", state.ema.ema_model))
        figures_dir = self.run_dir / "figures" / "warmup"
        samples_dir = self.run_dir / "samples" / "warmup"
        csv_path = self.run_dir / "metrics" / "warmup_metrics.csv"
        for model_kind, model in models:
            was_training = model.training
            model.eval()
            row = evaluate_warmup_model(
                model=model,
                model_kind=model_kind,
                direction=direction.value,
                node_solver=self.node_solver,
                potential=self.problem.potential,
                source_test=source_test,
                target_test=target_test,
                evaluation_config=self.problem_config.evaluation,
                latest_warmup_loss=latest_loss,
                figures_dir=figures_dir,
                samples_dir=samples_dir,
            )
            append_warmup_metrics_row(csv_path, row)
            self.metrics["warmup"].append(row)
            if was_training:
                model.train()

    def evaluate_rectification(self, rectification_index: int) -> None:
        source_test = self.dataset["source_test"]
        target_test = self.dataset["target_test"]
        if source_test.shape[0] == 0 or target_test.shape[0] == 0:
            return
        if self.mode == "own_ema":
            self._evaluate_direction_models(
                Direction.FORWARD,
                rectification_index,
                source_test,
                target_test,
                self.run_dir / "metrics" / "metrics.csv",
            )
            return
        self._evaluate_direction_models(
            Direction.FORWARD,
            rectification_index,
            source_test,
            target_test,
            self.run_dir / "metrics" / "metrics_forward.csv",
        )
        self._evaluate_direction_models(
            Direction.BACKWARD,
            rectification_index,
            target_test,
            source_test,
            self.run_dir / "metrics" / "metrics_backward.csv",
        )

    def _evaluate_direction_models(
        self,
        direction: Direction,
        rectification_index: int,
        source_test: Tensor,
        target_test: Tensor,
        csv_path: Path,
    ) -> None:
        state = self.states[direction]
        losses = self.metrics["losses"].get(direction.value, [])
        latest_loss = losses[-1] if losses else float("nan")
        models = [("online", state.model)]
        if state.ema is not None:
            models.append(("ema", state.ema.ema_model))
        figures_dir = self.run_dir / "figures" / "evaluation"
        for model_kind, model in models:
            was_training = model.training
            model.eval()
            row = evaluate_model(
                model=model,
                model_kind=model_kind,
                direction=direction.value,
                rectification_index=rectification_index,
                node_solver=self.node_solver,
                potential=self.problem.potential,
                source_test=source_test,
                target_test=target_test,
                evaluation_config=self.problem_config.evaluation,
                latest_loss=latest_loss,
                bridge_metrics=self.metrics["bridge"],
                figures_dir=figures_dir,
            )
            append_metrics_row(csv_path, row)
            self.metrics["evaluation"].append(row)
            if was_training:
                model.train()

    def save_checkpoint(
        self,
        tag: str,
        *,
        rectification_index: int | None,
        direction: Direction | None,
    ) -> Path:
        checkpoint_dir = self.run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        suffix = tag
        if rectification_index is not None:
            suffix += f"_r{rectification_index}"
        if direction is not None:
            suffix += f"_{direction.value}"
        path = checkpoint_dir / f"{suffix}.pt"
        state = {
            "tag": tag,
            "stage": self.stage,
            "rectification_index": rectification_index,
            "direction": None if direction is None else direction.value,
            "completed_passes": list(self.completed_passes),
            "train_config": self.train_config,
            "problem_config": self.problem_config,
            "directions": {key.value: value.state_dict() for key, value in self.states.items()},
            "rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "metrics": self.metrics,
        }
        torch.save(state, path)
        self.metrics["checkpoints"].append(str(path))
        return path

    def save_metrics(self) -> Path:
        metrics_dir = self.run_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        path = metrics_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)
        return path

    def _make_direction_state(self, direction: Direction) -> DirectionState:
        model = FourierTimeResidualMLP(
            dim=self.problem.dimension,
            out_dim=self.problem.dimension,
            w=self.train_config.model.width,
            hidden=self.train_config.model.hidden,
            m=self.train_config.model.fourier_modes,
            time_varying=True,
        ).to(device=self.device, dtype=self.dtype)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.train_config.optimization.learning_rate,
            weight_decay=self.train_config.optimization.weight_decay,
        )
        scheduler = self._make_scheduler(optimizer, self.total_steps_by_direction[direction])
        return DirectionState(
            direction=direction,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    def _generation_state_for(self, direction: Direction) -> DirectionState:
        if self.mode == "opposite_ema":
            return self.states[direction.opposite]
        return self.states[direction]

    def _prepare_datasets(self) -> Dict[str, Tensor]:
        cfg = self.train_config.data
        source = self.problem.sample_source(cfg.total_samples, seed=self.train_config.seed)
        target = self.problem.sample_target(cfg.total_samples, seed=self.train_config.seed + 1)
        n_train = int(round(cfg.total_samples * (1.0 - cfg.test_fraction)))
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.train_config.seed + 2)
        source_perm = torch.randperm(cfg.total_samples, generator=generator, device=self.device)
        generator.manual_seed(self.train_config.seed + 3)
        target_perm = torch.randperm(cfg.total_samples, generator=generator, device=self.device)
        return {
            "source_train": source[source_perm[:n_train]],
            "source_test": source[source_perm[n_train:]],
            "target_train": target[target_perm[:n_train]],
            "target_test": target[target_perm[n_train:]],
        }

    def _rectification_subdataset(self, rectification_index: int) -> Tuple[Tensor, Tensor]:
        size = self.train_config.data.n_dataset
        source = self.dataset["source_train"]
        target = self.dataset["target_train"]
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.train_config.seed + 1009 + rectification_index)
        source_idx = torch.randperm(source.shape[0], generator=generator, device=self.device)[:size]
        generator.manual_seed(self.train_config.seed + 2003 + rectification_index)
        target_idx = torch.randperm(target.shape[0], generator=generator, device=self.device)[:size]
        return source[source_idx], target[target_idx]

    def _make_loaders(self, source: Tensor, target: Tensor) -> Tuple[DataLoader, DataLoader]:
        cfg = self.train_config.data
        source_loader = DataLoader(
            TensorDataset(source),
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
        )
        target_loader = DataLoader(
            TensorDataset(target),
            batch_size=cfg.batch_size,
            shuffle=cfg.shuffle,
            drop_last=cfg.drop_last,
            num_workers=cfg.num_workers,
        )
        return source_loader, target_loader

    def _record_bridge_metrics(
        self,
        direction: Direction,
        rectification_index: int,
        batch_index: int,
        solution: BridgeSolution,
    ) -> None:
        endpoint_error = (
            float(solution.endpoint_errors.max().detach().cpu())
            if solution.endpoint_errors.numel()
            else float("nan")
        )
        solve_time_seconds = float(solution.solve_time_seconds)
        solve_time_per_successful_pair = (
            solve_time_seconds / solution.num_successful if solution.num_successful else float("nan")
        )
        solve_time_per_requested_pair = (
            solve_time_seconds / solution.num_pairs if solution.num_pairs else float("nan")
        )
        solver_iterations = [
            int(value) for value in solution.solver_iterations if value is not None
        ]
        solver_mesh_nodes = [
            int(value) for value in solution.solver_mesh_nodes if value is not None
        ]
        mean_solver_iterations = (
            float(sum(solver_iterations) / len(solver_iterations))
            if solver_iterations
            else float("nan")
        )
        max_solver_iterations = max(solver_iterations) if solver_iterations else float("nan")
        mean_solver_mesh_nodes = (
            float(sum(solver_mesh_nodes) / len(solver_mesh_nodes))
            if solver_mesh_nodes
            else float("nan")
        )
        diagnostic_paths = self._save_bridge_diagnostics(
            direction, rectification_index, batch_index, solution
        )
        self.metrics["bridge"].append(
            {
                "direction": direction.value,
                "rectification_index": rectification_index,
                "batch_index": batch_index,
                "requested_pairs": solution.num_pairs,
                "successful_pairs": solution.num_successful,
                "failed_pairs": int(solution.failed_indices.numel()),
                "max_endpoint_error": endpoint_error,
                "solve_time_seconds": solve_time_seconds,
                "solve_time_per_successful_pair": solve_time_per_successful_pair,
                "solve_time_per_requested_pair": solve_time_per_requested_pair,
                "mean_solver_iterations": mean_solver_iterations,
                "max_solver_iterations": max_solver_iterations,
                "mean_solver_mesh_nodes": mean_solver_mesh_nodes,
                "guess_source": solution.guess_source,
                "failure_preview": list(solution.failure_messages.items())[:5],
                "diagnostics": diagnostic_paths,
            }
        )

    def _save_bridge_diagnostics(
        self,
        direction: Direction,
        rectification_index: int,
        batch_index: int,
        solution: BridgeSolution,
    ) -> Dict[str, str]:
        key = (rectification_index, direction.value)
        if key in self._diagnostic_saved or solution.num_successful == 0:
            return {}
        count = min(self.train_config.output.diagnostic_plot_count, solution.num_successful)
        if count <= 0:
            return {}
        self._diagnostic_saved.add(key)
        tag = f"r{rectification_index}_{direction.value}_b{batch_index}"
        samples_dir = self.run_dir / "samples"
        figures_dir = self.run_dir / "figures"
        samples_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        data_path = samples_dir / f"bridge_{tag}.pt"
        torch.save(
            {
                "time_grid": solution.time_grid.detach().cpu(),
                "mean": solution.mean[:count].detach().cpu(),
                "std": solution.std[:count].detach().cpu(),
                "x0": solution.x0[:count].detach().cpu(),
                "x1": solution.x1[:count].detach().cpu(),
            },
            data_path,
        )
        paths = {"data": str(data_path)}
        bridge_figures_dir = self.run_dir / "figures" / "bridge"
        paths.update(
            save_bridge_solution_plots(
                figures_dir=bridge_figures_dir,
                tag=f"r{rectification_index}_{direction.value}",
                mean=solution.mean[:count],
                std=solution.std[:count],
                time_grid=solution.time_grid,
                potential=self.problem.potential,
                source_reference=solution.x0[:count],
                evaluation_config=self.problem_config.evaluation,
            )
        )
        self.metrics["diagnostics"].append(paths)
        return paths

    def _create_run_dir(self) -> Path:
        root = Path(self.train_config.output.root)
        run_name = self.train_config.output.run_name or time.strftime("%Y%m%d_%H%M%S")
        run_dir = root / self.problem_config.name / run_name
        index = 1
        base = run_dir
        while run_dir.exists():
            run_dir = Path(f"{base}_{index}")
            index += 1
        for child in ("checkpoints", "metrics", "samples", "figures", "logs"):
            (run_dir / child).mkdir(parents=True, exist_ok=True)
        return run_dir

    def _estimate_total_steps(self, directions) -> Dict[Direction, int]:
        cfg = self.train_config
        train_count = self._train_count()
        warmup_batches = self._num_batches(train_count)
        rect_batches = self._num_batches(cfg.data.n_dataset)
        totals = {}
        for direction in directions:
            warmup_steps = warmup_batches * cfg.initial_fit.epochs * cfg.initial_fit.steps_per_batch
            rect_occurrences = sum(
                1 for value in cfg.rectification.direction_order if value == direction.value
            )
            rect_steps = (
                rect_batches
                * cfg.rectification.num_rectifications
                * rect_occurrences
                * cfg.rectification.steps_per_batch
            )
            totals[direction] = max(1, warmup_steps + rect_steps)
        return totals

    def _make_scheduler(self, optimizer, total_steps: int):
        scheduler_cfg = self.train_config.optimization.scheduler
        if scheduler_cfg.kind == "none":
            return None
        base_lr = self.train_config.optimization.learning_rate
        min_ratio = scheduler_cfg.min_lr / base_lr

        def lr_lambda(step: int):
            progress = min(max(step, 0), total_steps) / float(max(total_steps, 1))
            if scheduler_cfg.kind == "linear":
                factor = 1.0 - progress
            else:
                factor = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_ratio + (1.0 - min_ratio) * factor

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    def _num_batches(self, n_samples: int) -> int:
        batch_size = self.train_config.data.batch_size
        if self.train_config.data.drop_last:
            return max(1, n_samples // batch_size)
        return max(1, math.ceil(n_samples / batch_size))

    def _train_count(self) -> int:
        return int(round(self.train_config.data.total_samples * (1.0 - self.train_config.data.test_fraction)))

    @staticmethod
    def _resolve_dtype(name: str) -> torch.dtype:
        if name == "float64":
            return torch.float64
        if name == "float32":
            return torch.float32
        raise ValueError("dtype must be 'float32' or 'float64'.")

    @staticmethod
    def _resolve_device(name: str) -> torch.device:
        if name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if name.startswith("cuda") and not torch.cuda.is_available():
            return torch.device("cpu")
        return torch.device(name)

    @staticmethod
    def _set_seed(seed: int) -> None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _check_finite(value: Tensor, label: str) -> None:
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{label} is not finite.")
