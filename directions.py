"""Direction-specific state for forward/backward HFM training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch

from .config import EMAConfig
from .ema import ExponentialMovingAverage


class Direction(str, Enum):
    """Training direction for boundary transport."""

    FORWARD = "forward"
    BACKWARD = "backward"

    @property
    def opposite(self) -> "Direction":
        return Direction.BACKWARD if self is Direction.FORWARD else Direction.FORWARD


@dataclass
class DirectionState:
    """Online model, optimizer, scheduler, and EMA state owned by one direction."""

    direction: Direction
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
    ema: Optional[ExponentialMovingAverage] = None

    def initialize_ema(self, config: EMAConfig) -> None:
        self.ema = ExponentialMovingAverage(
            self.model,
            decay=config.decay,
            mode=config.mode,
            gamma=config.gamma,
        )

    @property
    def generation_model(self) -> torch.nn.Module:
        if self.ema is None:
            return self.model
        return self.ema.ema_model

    def scheduler_step(self) -> None:
        if self.scheduler is not None:
            self.scheduler.step()

    def reset_optimization(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    ) -> None:
        self.optimizer = optimizer
        self.scheduler = scheduler

    def state_dict(self):
        return {
            "direction": self.direction.value,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": None if self.scheduler is None else self.scheduler.state_dict(),
            "ema": None if self.ema is None else self.ema.state_dict(),
        }

    def load_state_dict(self, state, ema_config: EMAConfig = None) -> None:
        self.model.load_state_dict(state["model"])
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scheduler is not None and state.get("scheduler") is not None:
            self.scheduler.load_state_dict(state["scheduler"])
        if state.get("ema") is not None:
            if self.ema is None:
                if ema_config is None:
                    ema_config = EMAConfig(
                        mode=state["ema"].get("mode", "fixed"),
                        decay=float(state["ema"]["decay"]),
                        gamma=float(state["ema"].get("gamma", 6.99)),
                    )
                self.initialize_ema(ema_config)
            self.ema.load_state_dict(state["ema"])
