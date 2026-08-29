"""Reproducible CIFAR experiment runner for DCPAG models."""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from cifar import GATED_MODELS, MODEL_NAMES, build_model

LOGGER = logging.getLogger("dcpag")


@dataclass
class EpochMetrics:
    epoch: int
    learning_rate: float
    train_loss: float
    train_accuracy: float
    validation_loss: float
    validation_accuracy: float
    elapsed_seconds: float


class AverageMeter:
    """Track a sample-weighted running average."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * count
        self.count += count
        self.average = self.total / self.count


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline, Dynamic ReLU, and FBS models on CIFAR."
    )
    parser.add_argument("--model", choices=MODEL_NAMES, default="dyrelu-resnet")
    parser.add_argument(
        "--dataset", choices=("cifar10", "cifar100", "fake"), default="cifar10"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--depth", type=int, choices=(18, 34, 50, 101), default=18)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--gate-lambda", type=float, default=1e-8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument(
        "--milestones",
        type=int,
        nargs="*",
        default=None,
        help="Epochs at which to multiply the learning rate by --gamma.",
    )
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--amp", action="store_true", help="Use mixed precision on CUDA.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile when available.")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--no-download", dest="download", action="store_false")
    parser.add_argument("--non-deterministic", dest="deterministic", action="store_false")
    parser.add_argument("--tensorboard", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument(
        "--fake-data-size",
        type=int,
        default=1024,
        help="Sample count used by --dataset fake for smoke tests.",
    )
    parser.set_defaults(augment=True, deterministic=True, download=True)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.ratio <= 1.0:
        raise ValueError("--ratio must be in (0, 1]")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers cannot be negative")
    if args.gate_lambda < 0 or args.gradient_clip < 0:
        raise ValueError("regularization and clipping values cannot be negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")


def configure_logging(run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    handlers = (
        logging.StreamHandler(),
        logging.FileHandler(run_dir / "train.log", encoding="utf-8"),
    )
    for handler in handlers:
        handler.setFormatter(formatter)
        LOGGER.addHandler(handler)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_worker(_worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def _normalization(dataset_name: str) -> transforms.Normalize:
    if dataset_name == "cifar100":
        return transforms.Normalize(
            (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        )
    return transforms.Normalize(
        (0.4914, 0.4824, 0.4467), (0.2470, 0.2435, 0.2616)
    )


def build_loaders(args: argparse.Namespace, device: torch.device):
    normalize = _normalization(args.dataset)
    train_ops = []
    if args.augment:
        train_ops.extend(
            (transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip())
        )
    train_ops.extend((transforms.ToTensor(), normalize))
    train_transform = transforms.Compose(train_ops)
    validation_transform = transforms.Compose((transforms.ToTensor(), normalize))

    if args.dataset == "fake":
        num_classes = 10
        train_set = datasets.FakeData(
            size=args.fake_data_size,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_transform,
            random_offset=args.seed,
        )
        validation_set = datasets.FakeData(
            size=max(args.batch_size, args.fake_data_size // 4),
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=validation_transform,
            random_offset=args.seed + args.fake_data_size,
        )
    else:
        dataset_class = (
            datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        )
        num_classes = 10 if args.dataset == "cifar10" else 100
        train_set = dataset_class(
            root=args.data_dir, train=True, download=args.download, transform=train_transform
        )
        validation_set = dataset_class(
            root=args.data_dir, train=False, download=args.download, transform=validation_transform
        )

    generator = torch.Generator().manual_seed(args.seed)
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.workers > 0,
        "generator": generator,
        "worker_init_fn": seed_worker,
    }
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    validation_loader = DataLoader(validation_set, shuffle=False, **loader_kwargs)
    return train_loader, validation_loader, num_classes


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return predictions.eq(targets).float().mean().mul(100.0).item()


def unpack_output(output, device: torch.device):
    if isinstance(output, tuple):
        logits, gate_penalty = output
        if not torch.is_tensor(gate_penalty):
            gate_penalty = torch.tensor(gate_penalty, device=device)
        return logits, gate_penalty.mean()
    return output, torch.zeros((), device=device)


def run_epoch(
    loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.cuda.amp.GradScaler,
    gate_lambda: float,
    gradient_clip: float,
    amp_enabled: bool,
    log_interval: int,
    max_batches: int | None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    losses = AverageMeter()
    accuracies = AverageMeter()

    for batch_index, (inputs, targets) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                logits, gate_penalty = unpack_output(model(inputs), device)
                loss = criterion(logits, targets)
                if training and gate_lambda:
                    loss = loss + gate_lambda * gate_penalty

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                if gradient_clip:
                    nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
                scaler.step(optimizer)
                scaler.update()

        batch_size = inputs.size(0)
        losses.update(loss.item(), batch_size)
        accuracies.update(accuracy(logits, targets), batch_size)
        if training and batch_index % log_interval == 0:
            LOGGER.info(
                "batch=%d/%d loss=%.4f accuracy=%.2f",
                batch_index,
                len(loader),
                losses.average,
                accuracies.average,
            )

    return losses.average, accuracies.average


def default_milestones(model_name: str) -> Sequence[int]:
    return (100, 200) if "cifarnet" in model_name else (150, 225)


def save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    best_accuracy: float,
    args: argparse.Namespace,
) -> None:
    state_model = getattr(model, "_orig_mod", model)
    torch.save(
        {
            "epoch": epoch,
            "model": state_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_accuracy": best_accuracy,
            "args": vars(args),
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
) -> tuple[int, float]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model", checkpoint.get("state_dict"))
    if state_dict is None:
        raise ValueError(f"checkpoint {path} does not contain model weights")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    model.load_state_dict(state_dict)
    if "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint.get("epoch", 0)), float(
        checkpoint.get("best_accuracy", checkpoint.get("best_prec1", 0.0))
    )


def append_metrics(path: Path, metrics: EpochMetrics) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(asdict(metrics), ensure_ascii=False) + "\n")


def main(argv: Iterable[str] | None = None) -> float:
    args = create_parser().parse_args(argv)
    validate_args(args)
    device = resolve_device(args.device)
    seed_everything(args.seed, args.deterministic)

    run_name = args.run_name or f"{args.dataset}-{args.model}-r{args.ratio:g}-s{args.seed}"
    run_dir = args.output_dir / run_name
    configure_logging(run_dir)
    LOGGER.info("configuration=%s", json.dumps(vars(args), default=str, sort_keys=True))
    LOGGER.info("device=%s", device)

    train_loader, validation_loader, num_classes = build_loaders(args, device)
    model = build_model(
        args.model, num_classes=num_classes, depth=args.depth, ratio=args.ratio
    ).to(device)
    LOGGER.info("parameters=%d", sum(parameter.numel() for parameter in model.parameters()))

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        nesterov=True,
        weight_decay=args.weight_decay,
    )
    milestones = args.milestones or default_milestones(args.model)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=milestones, gamma=args.gamma
    )
    amp_enabled = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_epoch = 0
    best_accuracy = float("-inf")

    if args.resume:
        start_epoch, best_accuracy = load_checkpoint(
            args.resume, model, optimizer, scheduler, scaler, device
        )
        LOGGER.info(
            "resumed=%s epoch=%d best_accuracy=%.2f",
            args.resume,
            start_epoch,
            best_accuracy,
        )

    if args.compile:
        if not hasattr(torch, "compile"):
            raise RuntimeError("--compile requires PyTorch 2.0 or newer")
        model = torch.compile(model)

    if args.evaluate:
        validation_loss, validation_accuracy = run_epoch(
            validation_loader,
            model,
            criterion,
            device,
            optimizer=None,
            scaler=scaler,
            gate_lambda=0.0,
            gradient_clip=0.0,
            amp_enabled=amp_enabled,
            log_interval=args.log_interval,
            max_batches=args.max_val_batches,
        )
        LOGGER.info(
            "validation_loss=%.4f validation_accuracy=%.2f",
            validation_loss,
            validation_accuracy,
        )
        return validation_accuracy

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as exc:
            raise RuntimeError("Install tensorboard to use --tensorboard") from exc
        writer = SummaryWriter(run_dir)

    try:
        for epoch in range(start_epoch, args.epochs):
            started = time.perf_counter()
            train_loss, train_accuracy = run_epoch(
                train_loader,
                model,
                criterion,
                device,
                optimizer=optimizer,
                scaler=scaler,
                gate_lambda=args.gate_lambda if args.model in GATED_MODELS else 0.0,
                gradient_clip=args.gradient_clip,
                amp_enabled=amp_enabled,
                log_interval=args.log_interval,
                max_batches=args.max_train_batches,
            )
            validation_loss, validation_accuracy = run_epoch(
                validation_loader,
                model,
                criterion,
                device,
                optimizer=None,
                scaler=scaler,
                gate_lambda=0.0,
                gradient_clip=0.0,
                amp_enabled=amp_enabled,
                log_interval=args.log_interval,
                max_batches=args.max_val_batches,
            )
            current_learning_rate = optimizer.param_groups[0]["lr"]
            is_best = validation_accuracy > best_accuracy
            best_accuracy = max(best_accuracy, validation_accuracy)
            metrics = EpochMetrics(
                epoch=epoch + 1,
                learning_rate=current_learning_rate,
                train_loss=train_loss,
                train_accuracy=train_accuracy,
                validation_loss=validation_loss,
                validation_accuracy=validation_accuracy,
                elapsed_seconds=time.perf_counter() - started,
            )
            scheduler.step()
            append_metrics(run_dir / "metrics.jsonl", metrics)
            save_checkpoint(
                run_dir / "last.pt",
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                best_accuracy=best_accuracy,
                args=args,
            )
            if is_best:
                save_checkpoint(
                    run_dir / "best.pt",
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_accuracy=best_accuracy,
                    args=args,
                )
            LOGGER.info("epoch=%d metrics=%s", epoch + 1, json.dumps(asdict(metrics)))
            if writer:
                for key, value in asdict(metrics).items():
                    if key not in {"epoch", "elapsed_seconds"}:
                        writer.add_scalar(key, value, epoch + 1)
    finally:
        if writer:
            writer.close()

    LOGGER.info("best_validation_accuracy=%.2f", best_accuracy)
    return best_accuracy


if __name__ == "__main__":
    main()
