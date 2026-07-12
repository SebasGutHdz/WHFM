"""Hamiltonian flow matcher orchestration."""

from __future__ import annotations

import torch

from ..optimal_transport import OTPlanSampler


class HamiltonianFlowMatcher:
    """OT coupling plus Hamiltonian Gaussian path sampling."""

    def __init__(
        self,
        path,
        coupling: str = "ot",
        ot_method: str = "exact",
        ot_reg: float = 0.05,
        ot_reg_m: float = 1.0,
    ):
        if coupling == "ipmf":
            raise NotImplementedError("IPMF coupling is planned for Phase 2.")
        if coupling != "ot":
            raise ValueError(f"Unknown coupling: {coupling}")
        self.path = path
        self.coupling = coupling
        self.ot_sampler = OTPlanSampler(method=ot_method, reg=ot_reg, reg_m=ot_reg_m)

    def sample_location_and_conditional_flow(self, x0, x1, return_noise: bool = False):
        """Return ``(t, xt, ut)`` after OT coupling."""

        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        t = torch.rand((x0.shape[0], 1), dtype=x0.dtype, device=x0.device)
        epsilon = torch.randn_like(x0)
        xt = self.path.sample_xt(x0, x1, t, epsilon)
        ut = self.path.compute_ut(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, epsilon
        return t, xt, ut
