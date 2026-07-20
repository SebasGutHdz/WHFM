"""Endpoint coupling utilities for HFM training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor

from .optimal_transport import OTPlanSampler


@dataclass
class CoupledBatch:
    x0: Tensor
    x1: Tensor
    y0: Optional[Tensor] = None
    y1: Optional[Tensor] = None


class Coupler:
    """Pair source and target minibatches while preserving optional labels."""

    def __init__(
        self,
        kind: str,
        *,
        ot_method: str = "exact",
        ot_reg: float = 0.05,
        ot_reg_m: float = 1.0,
    ):
        if kind not in {"independent", "ot"}:
            raise ValueError("kind must be 'independent' or 'ot'.")
        self.kind = kind
        self.ot_sampler = None
        if kind == "ot":
            self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg, reg_m=ot_reg_m)

    def pair(self, x0: Tensor, x1: Tensor) -> Tuple[Tensor, Tensor]:
        if self.kind == "ot":
            return self.ot_sampler.sample_plan(x0, x1)
        if x1.shape[0] <= 1:
            return x0, x1
        perm = torch.randperm(x1.shape[0], device=x1.device)
        return x0, x1[perm]

    def pair_with_labels(
        self,
        x0: Tensor,
        x1: Tensor,
        *,
        y0: Optional[Tensor] = None,
        y1: Optional[Tensor] = None,
    ) -> CoupledBatch:
        if self.kind == "ot":
            x0_p, x1_p, y0_p, y1_p = self.ot_sampler.sample_plan_with_labels(
                x0, x1, y0=y0, y1=y1
            )
            return CoupledBatch(x0_p, x1_p, y0_p, y1_p)

        if x1.shape[0] <= 1:
            return CoupledBatch(x0, x1, y0, y1)
        perm = torch.randperm(x1.shape[0], device=x1.device)
        return CoupledBatch(
            x0,
            x1[perm],
            y0,
            None if y1 is None else y1[perm],
        )
