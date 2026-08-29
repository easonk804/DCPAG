"""CIFAR model registry for the DCPAG experiments."""

from .cifar_net import CifarNet
from .dyrelu_cifar_net import DyReLUCifarNet
from .dyrelu_resnet import DyReLUResNetCifar
from .fbs_cifar_net import FBSCifarNet
from .fbs_resnet import FBSResNetCifar
from .resnet import ResNetCifar

MODEL_NAMES = (
    "cifarnet",
    "dyrelu-cifarnet",
    "fbs-cifarnet",
    "resnet",
    "dyrelu-resnet",
    "fbs-resnet",
)

GATED_MODELS = frozenset(
    {"dyrelu-cifarnet", "fbs-cifarnet", "dyrelu-resnet", "fbs-resnet"}
)


def build_model(name, *, num_classes=10, depth=18, ratio=1.0):
    """Build a CIFAR model from an explicit, stable experiment name."""
    if name == "dyrelu-resnet" and depth != 18:
        raise ValueError("dyrelu-resnet currently supports depth 18 only")
    if name == "fbs-resnet" and depth not in (18, 34):
        raise ValueError("fbs-resnet currently supports depths 18 and 34 only")
    factories = {
        "cifarnet": lambda: CifarNet(ratio=ratio, num_classes=num_classes),
        "dyrelu-cifarnet": lambda: DyReLUCifarNet(
            ratio=ratio, num_classes=num_classes
        ),
        "fbs-cifarnet": lambda: FBSCifarNet(
            ratio=ratio, num_classes=num_classes
        ),
        "resnet": lambda: ResNetCifar(depth, num_classes),
        "dyrelu-resnet": lambda: DyReLUResNetCifar(depth, num_classes, ratio),
        "fbs-resnet": lambda: FBSResNetCifar(depth, num_classes, ratio),
    }
    try:
        return factories[name]()
    except KeyError as exc:
        raise ValueError(f"unknown model {name!r}; choose from {MODEL_NAMES}") from exc


__all__ = ["GATED_MODELS", "MODEL_NAMES", "build_model"]
