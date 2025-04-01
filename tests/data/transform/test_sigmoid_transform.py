from __future__ import annotations

import torch

from transformertf.data.transform import SigmoidTransform


def test_create_sigmoid_transform_default_parameters() -> None:
    transform = SigmoidTransform()

    assert transform.k_ == 1.0
    assert transform.x0_ == 0.0


def test_create_sigmoid_transform_custom_parameters() -> None:
    transform = SigmoidTransform(k_=2.0, x0_=3.0)

    assert transform.k_ == 2.0
    assert transform.x0_ == 3.0


def test_sigmoid_transform_fit() -> None:
    transform = SigmoidTransform()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    transform_fitted = transform.fit(x)

    assert transform_fitted == transform


def test_sigmoid_transform_forward() -> None:
    transform = SigmoidTransform()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    y_transformed = transform.transform(x)

    target = torch.tensor([0.7310586, 0.8807971, 0.9525741, 0.9820138, 0.9933072])

    assert torch.allclose(y_transformed, target)


def test_sigmoid_transform_inverse_transform() -> None:
    transform = SigmoidTransform()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    x_inverse_transformed = transform.inverse_transform(transform.transform(x))

    assert torch.allclose(x_inverse_transformed, x)


def test_sigmoid_transform_transform() -> None:
    transform = SigmoidTransform()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    y_transformed = transform.transform(x)

    target = torch.tensor([0.7310586, 0.8807971, 0.9525741, 0.9820138, 0.9933072])

    assert torch.allclose(y_transformed, target)


def test_sigmoid_fit_transform() -> None:
    transform = SigmoidTransform()

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    y_transformed = transform.fit_transform(x)

    target = torch.tensor([0.7310586, 0.8807971, 0.9525741, 0.9820138, 0.9933072])

    assert torch.allclose(y_transformed, target)
