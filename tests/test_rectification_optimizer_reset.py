from __future__ import annotations

import importlib
from types import SimpleNamespace

import torch


def _load_package_module(monkeypatch, module_name: str):
    return importlib.import_module(f"whfm.{module_name}")


def test_direction_state_reset_optimization_preserves_model_direction_and_ema(monkeypatch):
    directions = _load_package_module(monkeypatch, "directions")
    model = torch.nn.Linear(2, 2)
    old_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = directions.DirectionState(
        direction=directions.Direction.FORWARD,
        model=model,
        optimizer=old_optimizer,
        scheduler=None,
    )
    ema = object()
    state.ema = ema
    new_optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    new_scheduler = torch.optim.lr_scheduler.LambdaLR(new_optimizer, lr_lambda=lambda step: 1.0)

    state.reset_optimization(new_optimizer, new_scheduler)

    assert state.direction is directions.Direction.FORWARD
    assert state.model is model
    assert state.ema is ema
    assert state.optimizer is new_optimizer
    assert state.scheduler is new_scheduler
    assert state.optimizer is not old_optimizer


def test_trainer_rectification_reset_uses_existing_model_parameters(monkeypatch):
    trainer_module = _load_package_module(monkeypatch, "trainer")
    directions = importlib.import_module("whfm.directions")
    trainer = object.__new__(trainer_module.HamiltonianTrainer)
    model = torch.nn.Linear(2, 2)
    initial_optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    state = directions.DirectionState(
        direction=directions.Direction.FORWARD,
        model=model,
        optimizer=initial_optimizer,
        scheduler=None,
    )
    state.ema = object()
    trainer.states = {directions.Direction.FORWARD: state}
    trainer.train_config = SimpleNamespace(
        data=SimpleNamespace(n_dataset=8, batch_size=4, drop_last=False),
        rectification=SimpleNamespace(
            direction_order=["forward", "forward"],
            steps_per_batch=3,
        ),
        optimization=SimpleNamespace(
            learning_rate=1e-2,
            weight_decay=0.1,
            scheduler=SimpleNamespace(kind="linear", min_lr=1e-3),
        ),
    )

    trainer._reset_rectification_optimization()

    assert state.model is model
    assert state.ema is not None
    assert state.optimizer is not initial_optimizer
    assert len(state.optimizer.state) == 0
    assert state.optimizer.param_groups[0]["lr"] == 1e-2
    optimizer_params = {
        id(param)
        for group in state.optimizer.param_groups
        for param in group["params"]
    }
    model_params = {id(param) for param in model.parameters()}
    assert optimizer_params == model_params
    assert state.scheduler is not None
    assert state.scheduler.optimizer is state.optimizer
    assert state.scheduler.last_epoch == 0
    assert trainer._rectification_steps_for_direction(directions.Direction.FORWARD) == 12


def test_trainer_rectification_terminal_path_bookkeeping_is_per_model(monkeypatch, tmp_path):
    trainer_module = _load_package_module(monkeypatch, "trainer")
    directions = importlib.import_module("whfm.directions")
    trainer = object.__new__(trainer_module.HamiltonianTrainer)
    trainer.states = {
        directions.Direction.FORWARD: SimpleNamespace(
            model=torch.nn.Linear(2, 2),
            ema=SimpleNamespace(ema_model=torch.nn.Linear(2, 2)),
        ),
        directions.Direction.BACKWARD: SimpleNamespace(
            model=torch.nn.Linear(2, 2),
            ema=None,
        ),
    }
    trainer.metrics = {
        "losses": {"forward": [1.0], "backward": [2.0]},
        "bridge": [],
        "evaluation": [],
    }
    trainer.run_dir = tmp_path
    trainer.node_solver = object()
    trainer.problem = SimpleNamespace(potential=object())
    trainer.problem_config = SimpleNamespace(evaluation=object())
    trainer.run_dir = tmp_path
    trainer._previous_terminal_paths = {
        ("forward", "online"): tmp_path / "previous_forward_online.pt",
    }
    rows = []
    calls = []

    def fake_append_metrics_row(path, row):
        rows.append((path, row))

    def fake_evaluate_model_with_terminal(**kwargs):
        terminal_path = tmp_path / f"{kwargs['direction']}_{kwargs['model_kind']}_terminal.pt"
        calls.append(
            (
                kwargs["direction"],
                kwargs["model_kind"],
                kwargs.get("previous_terminal_path"),
                kwargs.get("samples_dir"),
            )
        )
        return {"direction": kwargs["direction"], "model_kind": kwargs["model_kind"]}, terminal_path

    monkeypatch.setattr(trainer_module, "append_metrics_row", fake_append_metrics_row)
    monkeypatch.setattr(trainer_module, "evaluate_model_with_terminal", fake_evaluate_model_with_terminal)

    trainer._evaluate_direction_models(
        directions.Direction.FORWARD,
        0,
        torch.zeros(2, 2),
        torch.ones(2, 2),
        tmp_path / "metrics.csv",
    )
    trainer._evaluate_direction_models(
        directions.Direction.BACKWARD,
        0,
        torch.zeros(2, 2),
        torch.ones(2, 2),
        tmp_path / "metrics_backward.csv",
    )

    assert calls == [
        ("forward", "online", tmp_path / "previous_forward_online.pt", tmp_path / "samples"),
        ("forward", "ema", None, tmp_path / "samples"),
        ("backward", "online", None, tmp_path / "samples"),
    ]
    assert trainer._previous_terminal_paths[("forward", "online")] == tmp_path / "forward_online_terminal.pt"
    assert trainer._previous_terminal_paths[("forward", "ema")] == tmp_path / "forward_ema_terminal.pt"
    assert trainer._previous_terminal_paths[("backward", "online")] == tmp_path / "backward_online_terminal.pt"
    assert len(rows) == 3
    assert len(trainer.metrics["evaluation"]) == 3


def test_run_initial_fit_does_not_evaluate_warmup_each_epoch(monkeypatch, tmp_path):
    trainer_module = _load_package_module(monkeypatch, "trainer")
    directions = importlib.import_module("whfm.directions")
    trainer = object.__new__(trainer_module.HamiltonianTrainer)
    trainer.mode = "own_ema"
    trainer.device = torch.device("cpu")
    trainer.dtype = torch.float32
    trainer.dataset = {
        "source_train": torch.zeros(2, 2),
        "target_train": torch.ones(2, 2),
    }
    trainer.train_config = SimpleNamespace(
        initial_fit=SimpleNamespace(epochs=2, steps_per_batch=1),
    )
    trainer.states = {directions.Direction.FORWARD: SimpleNamespace()}
    trainer.initial_coupler = SimpleNamespace(pair=lambda x, y: (x, y))
    trainer.completed_passes = []
    trainer._make_loaders = lambda source, target: (
        [(torch.zeros(1, 2),)],
        [(torch.ones(1, 2),)],
    )
    train_calls = []
    trainer._train_straight_batch = lambda *args, **kwargs: train_calls.append((args, kwargs))
    trainer.save_checkpoint = lambda *args, **kwargs: tmp_path / "checkpoint.pt"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("warmup evaluation should happen after EMA initialization, not inside run_initial_fit")

    trainer.evaluate_warmup = fail_if_called

    trainer.run_initial_fit()

    assert len(train_calls) == 2
    assert trainer.completed_passes == [
        {"stage": "initial_fit", "epoch": 0},
        {"stage": "initial_fit", "epoch": 1},
    ]


def test_train_evaluates_final_warmup_with_trajectory_plot_mode(monkeypatch, tmp_path):
    trainer_module = _load_package_module(monkeypatch, "trainer")
    trainer = object.__new__(trainer_module.HamiltonianTrainer)
    states = [SimpleNamespace(initialize_ema=lambda ema_config: None)]
    trainer.states = {"forward": states[0]}
    trainer.train_config = SimpleNamespace(ema=object())
    trainer.metrics = {"evaluation": []}
    trainer.problem = SimpleNamespace(potential=object())
    trainer.problem_config = SimpleNamespace(evaluation=object())
    trainer.run_dir = tmp_path
    trainer.stage = "created"
    trainer.run_initial_fit = lambda: None
    trainer.run_rectifications = lambda: None
    trainer.save_metrics = lambda: tmp_path / "metrics.json"
    trainer.save_checkpoint = lambda *args, **kwargs: tmp_path / "final.pt"
    calls = []

    def fake_evaluate_warmup(**kwargs):
        calls.append(kwargs)

    trainer.evaluate_warmup = fake_evaluate_warmup

    result = trainer.train()

    assert result is trainer.metrics
    assert calls == [{"model_kinds": ("online", "ema"), "plot_mode": "trajectory"}]
    assert trainer.stage == "complete"
