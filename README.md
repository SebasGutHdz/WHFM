# WHFM Standalone

WHFM Standalone implements Wasserstein Hamiltonian Flow Matching with scalar
Gaussian Hamiltonian bridge targets. The installable distribution is
`whfm-standalone`; its Python import package is `whfm`.

## Installation

Install PyTorch for the CUDA version required by the host first, then install
this repository in editable mode:

```bash
python -m pip install -e .
```

Visualization tools and test/build dependencies are optional extras:

```bash
python -m pip install -e '.[viz]'
python -m pip install -e '.[test,viz]'
```

## Training

Training uses separate train and problem YAML files. Existing configuration
paths remain valid:

```bash
whfm-train --train-config configs/train.yaml \
  --problem-config configs/problems/stunnel.yaml

whfm-train-v2 --train-config configs/train_v2.yaml \
  --problem-config configs/problems/stunnel.yaml
```

The root wrappers provide equivalent commands:

```bash
python train.py --train-config configs/train.yaml \
  --problem-config configs/problems/stunnel.yaml
python trainer_v2.py --train-config configs/train_v2.yaml \
  --problem-config configs/problems/stunnel.yaml
```

## Seed Sweeps

Sweep manifests define the trainer, base configuration, problem list, seeds,
and available physical GPU IDs. Each GPU child is masked with
`CUDA_VISIBLE_DEVICES` and sees its assigned device as `cuda:0`.

```bash
whfm-sweep --sweep-config configs/sweeps/vneck_seed_sweep_v2.yaml --dry-run
whfm-sweep --sweep-config configs/sweeps/vneck_seed_sweep_v2.yaml
whfm-sweep --sweep-config configs/sweeps/vneck_seed_sweep_v2.yaml --resume
```

`python sweep.py` remains an equivalent wrapper. Child jobs run as
`python -m whfm.sweep` and inherit the caller's working directory, so relative
output roots continue to resolve from the launch directory.

## Evaluation

Install the `viz` extra before using visualization commands:

```bash
whfm-animate results/problem/run
whfm-bridge-plot results/problem/run
```

## Current Limitations

- The scalar Gaussian bridge uses serial SciPy BVP solves inside each worker.
- Tensor-product Gauss-Hermite quadrature scales exponentially with dimension.
- Configured internal potentials are not yet included in the trainable
  mean/std bridge force.
- Bridge residual norms are not yet evaluated.
- Trainer checkpoints do not yet provide exact training resume semantics.
