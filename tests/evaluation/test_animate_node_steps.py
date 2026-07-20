from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

import importlib


def _load_animate(monkeypatch):
    return importlib.import_module("whfm.evaluation.animate")


def test_parser_accepts_node_steps(monkeypatch):
    animate = _load_animate(monkeypatch)

    args = animate.build_parser().parse_args(["results/stunnel/run", "--node-steps", "50"])

    assert args.node_steps == 50


def test_create_particle_evolution_gifs_overrides_node_steps(monkeypatch, tmp_path):
    animate = _load_animate(monkeypatch)
    captured = {}

    class DummyNodeSolver:
        def __init__(self, method, node_steps):
            captured["method"] = method
            captured["node_steps"] = node_steps

    train_config = SimpleNamespace(
        dtype="float32",
        device="cpu",
        node_solver=SimpleNamespace(method="euler", node_steps=500),
    )
    problem_config = SimpleNamespace(
        functional=SimpleNamespace(
            to_potential_cfg=lambda: {"linear": None, "internal": None, "interaction": None}
        )
    )
    monkeypatch.setattr(animate, "load_train_config", lambda path: train_config)
    monkeypatch.setattr(animate, "load_problem_config", lambda path: problem_config)
    monkeypatch.setattr(animate, "resolve_checkpoint_path", lambda run_dir, checkpoint=None: tmp_path / "final.pt")
    monkeypatch.setattr(animate.torch, "load", lambda *args, **kwargs: {"directions": {"forward": {"model": {}}}})
    monkeypatch.setattr(animate, "ConfiguredPotential", lambda cfg: SimpleNamespace())
    monkeypatch.setattr(animate, "NodeSolver", DummyNodeSolver)
    monkeypatch.setattr(
        animate,
        "_prepare_evaluation_split",
        lambda *args, **kwargs: (torch.empty(0, 2), torch.empty(0, 2)),
    )

    with pytest.raises(ValueError, match="no held-out evaluation samples"):
        animate.create_particle_evolution_gifs(tmp_path, node_steps=37)

    assert captured == {"method": "euler", "node_steps": 37}
