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

### Zero-terminal Gaussian bridges

The v2 forward `own_ema` configuration can set
`bridge_solver.sigma_source > 0` and `bridge_solver.sigma_target: 0.0`.
Set top-level `tau` in `(0, 1)`. NODE rollout, evaluation, and training-time
sampling then end at `t = 1 - tau`; the Gaussian BVP and its boundary
conditions remain on `[0, 1]`. The conditional target is evaluated only at
positive standard deviation. Existing equal-scale configurations remain valid
with `tau: 0`.

`node_solver.correct_terminal_guess: true` replaces the NODE guess's terminal
position with the exact paired target. Setting it to `false` retains the NODE
terminal position as an initializer ablation. The BVP still enforces its
configured endpoint in either case.

### Entropy functional

The `entropy` internal potential evaluates a Gaussian-mixture log density and
uses its analytic score in the BVP force. `entropy_cost` computes the same
score by autograd for comparison. The configured coefficient is applied once.
The reference mixture is frozen during each solve; the optional outer
reference iteration updates it between solves. The density floor stabilizes
the reference computation near a zero target scale without changing the BVP's
physical terminal boundary.

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
