import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from cifar import GATED_MODELS, MODEL_NAMES, build_model
from cifar.modules import DynamicReLU, DynamicReLUV1, validate_ratio


@pytest.mark.parametrize("module_class", [DynamicReLU, DynamicReLUV1])
def test_dynamic_coefficients_are_batch_permutation_equivariant(module_class):
    torch.manual_seed(7)
    module = module_class(4, 6).eval()
    inputs = torch.randn(3, 4, 5, 5)
    permutation = torch.tensor([2, 0, 1])

    coefficients_a, offsets_a = module(inputs)
    coefficients_b, offsets_b = module(inputs[permutation])

    assert coefficients_a.shape == (2, 3, 6, 1, 1)
    assert offsets_a.shape == (2, 3, 6, 1, 1)
    torch.testing.assert_close(coefficients_a[:, permutation], coefficients_b)
    torch.testing.assert_close(offsets_a[:, permutation], offsets_b)


@pytest.mark.parametrize("ratio", [0, -0.1, 1.01])
def test_invalid_ratios_are_rejected(ratio):
    with pytest.raises(ValueError, match="ratio"):
        validate_ratio(ratio)


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_model_registry_forward_contract(model_name):
    model = build_model(model_name, num_classes=10, depth=18, ratio=0.5).eval()
    with torch.no_grad():
        output = model(torch.randn(2, 3, 32, 32))

    if model_name in GATED_MODELS:
        logits, gate_penalty = output
        assert gate_penalty.ndim == 0
    else:
        logits = output
    assert logits.shape == (2, 10)
