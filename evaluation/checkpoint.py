"""Checkpoint loading helpers for WHFM evaluation tools."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

import torch


def install_legacy_checkpoint_aliases(module_names: Iterable[str] = ()) -> None:
    """Register v2 trainer classes on modules used by old pickled checkpoints."""

    try:
        from ..trainer_v2 import BridgeTargetSet, RectificationV2Config, TrainV2Config
    except Exception:
        return

    aliases = {
        "BridgeTargetSet": BridgeTargetSet,
        "RectificationV2Config": RectificationV2Config,
        "TrainV2Config": TrainV2Config,
    }
    names = {
        "__main__",
        "torchcfm.WHFM-standalone.evaluation.animate",
        "torchcfm.WHFM-standalone.evaluation.gaussian_bridge_plot",
        "torchcfm.WHFM_standalone.evaluation.animate",
        "torchcfm.WHFM_standalone.evaluation.gaussian_bridge_plot",
        *module_names,
    }
    for name in names:
        module = sys.modules.get(name)
        if isinstance(module, ModuleType):
            for attr, value in aliases.items():
                if not hasattr(module, attr):
                    setattr(module, attr, value)


def load_checkpoint(path: str | Path, *, map_location):
    """Load a checkpoint after installing aliases for legacy v2 config pickles."""

    install_legacy_checkpoint_aliases()
    return torch.load(Path(path), map_location=map_location, weights_only=False)
