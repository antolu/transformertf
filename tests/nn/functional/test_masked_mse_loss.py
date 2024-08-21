from __future__ import annotations

import torch

from transformertf.nn.functional import masked_mse_loss


def test_masked_mse_loss_with_perfect_prediction() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([1.0, 1.0, 1.0])
    assert masked_mse_loss(input_, target, mask) == 0.0


def test_masked_mse_loss_with_partial_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    assert masked_mse_loss(input_, target, mask) == 0.0


def test_masked_mse_loss_with_full_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([0.0, 0.0, 0.0])
    assert torch.isnan(masked_mse_loss(input_, target, mask))


def test_masked_mse_loss_with_all_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 1.0])
    assert masked_mse_loss(input_, target, mask) == 1.0 / 3.0


def test_masked_mse_loss_with_different_shapes() -> None:
    input_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    assert masked_mse_loss(input_, target, mask) == 0.0
