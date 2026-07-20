"""CLI entry point for package-local HFM training."""

from __future__ import annotations

import argparse
from dataclasses import replace

from .config import load_problem_config, load_train_config
from .trainer import HamiltonianTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Hamiltonian Flow Matching v1.")
    parser.add_argument("--train-config", required=True, help="Path to the training YAML config.")
    parser.add_argument("--problem-config", required=True, help="Path to the problem YAML config.")
    parser.add_argument("--cuda-device", type=int, default=None, help="CUDA device index to use, overriding the training YAML.")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    train_config = load_train_config(args.train_config)
    if args.cuda_device is not None:
        if args.cuda_device < 0:
            raise ValueError("--cuda-device must be nonnegative.")
        train_config = replace(
            train_config,
            device=f"cuda:{args.cuda_device}",
            cuda_device=args.cuda_device,
        )
    problem_config = load_problem_config(args.problem_config)
    trainer = HamiltonianTrainer(train_config, problem_config)
    trainer.train()
    print(f"Run directory: {trainer.run_dir}")


if __name__ == "__main__":
    main()
