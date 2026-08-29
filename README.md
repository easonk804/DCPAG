# DCPAG

Reproducible PyTorch experiments for **dynamic channel pruning with activation
gates**. The original DCPAG work was developed by Shunqiang Liu in 2021 and is
described in the associated
[paper](https://link.springer.com/article/10.1007/s10489-022-03383-w).

This repository now provides a maintained CIFAR experiment path while retaining
the original ImageNet and Nori scripts as historical references.

## What is included

- Baselines: CifarNet and ResNet.
- DCPAG variants based on Dynamic ReLU.
- Feature Boosting and Suppression (FBS) comparison models.
- CIFAR-10, CIFAR-100, and offline synthetic-data runs.
- Reproducible seeding, deterministic execution, AMP, gradient clipping,
  TensorBoard logging, JSONL metrics, and resumable checkpoints.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements-dev.txt
```

Install the CUDA build of PyTorch recommended for your GPU from
[pytorch.org](https://pytorch.org/get-started/locally/) when GPU training is
required.

## Quick validation

Run the unit tests:

```bash
pytest
```

Run a one-batch CPU smoke test without downloading a dataset:

```bash
python train_cifar.py \
  --dataset fake \
  --model dyrelu-resnet \
  --ratio 0.5 \
  --device cpu \
  --epochs 1 \
  --max-train-batches 1 \
  --max-val-batches 1
```

## CIFAR experiments

Train the main DCPAG configuration:

```bash
python train_cifar.py \
  --dataset cifar10 \
  --model dyrelu-resnet \
  --depth 18 \
  --ratio 0.5 \
  --gate-lambda 1e-8 \
  --epochs 300 \
  --amp
```

Available models:

| Model | Returns gate penalty | Purpose |
| --- | --- | --- |
| `cifarnet` | No | Width-scaled baseline |
| `dyrelu-cifarnet` | Yes | Dynamic ReLU channel gating |
| `fbs-cifarnet` | Yes | FBS comparison |
| `resnet` | No | ResNet baseline |
| `dyrelu-resnet` | Yes | DCPAG ResNet |
| `fbs-resnet` | Yes | FBS ResNet comparison |

Useful options:

- `--seed`: experiment seed; defaults to `1337`.
- `--non-deterministic`: trade repeatability for additional CUDA speed.
- `--resume runs/<run>/last.pt`: restore model, optimizer, scheduler, scaler,
  epoch, and best accuracy.
- `--evaluate --resume <checkpoint>`: evaluate a checkpoint only.
- `--tensorboard`: write TensorBoard events into the run directory.
- `--no-download`: fail instead of downloading a missing CIFAR dataset.
- `--compile`: opt into `torch.compile` on supported systems.

Each run writes `train.log`, `metrics.jsonl`, `last.pt`, and `best.pt` beneath
`runs/<run-name>/`.

## Correctness fixes in the maintained path

- Dynamic ReLU coefficients preserve the batch dimension instead of mixing
  samples during reshape.
- Gate regularization is stored per model instance rather than in process-wide
  mutable state.
- Validation data is no longer shuffled.
- Dataset and output paths are configurable and platform-independent.
- Checkpoints include optimizer, scheduler, scaler, and experiment settings.
- Fixed Dynamic ReLU coefficients are registered as buffers and therefore move
  correctly between CPU and GPU.

## Legacy ImageNet experiments

The extensionless scripts in the repository root and `ImageNet/` are retained
for provenance. Some depend on the internal `megrec`/Nori data stack and are not
part of the portable test suite. Porting that data pipeline should be treated
as a separate experiment-migration task so reported ImageNet results remain
comparable with the original environment.

## License

MIT
