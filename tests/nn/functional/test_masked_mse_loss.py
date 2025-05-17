from __future__ import annotations

import torch

from transformertf.nn.functional import mse_loss


def test_masked_mse_loss_with_perfect_prediction() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.tensor([1.0, 1.0, 1.0])
    assert mse_loss(input_, target, mask=mask) == 0.0


def test_masked_mse_loss_with_partial_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 0.0])
    assert mse_loss(input_, target, mask=mask) == 0.0


def test_masked_mse_loss_with_full_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([0.0, 0.0, 0.0])
    assert torch.isnan(mse_loss(input_, target, mask=mask))


def test_masked_mse_loss_with_all_mask() -> None:
    input_ = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    mask = torch.tensor([1.0, 1.0, 1.0])
    assert mse_loss(input_, target, mask=mask) == 1.0 / 3.0


def test_masked_mse_loss_with_different_shapes() -> None:
    input_ = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.0, 2.0], [3.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0], [1.0, 0.0]])
    assert mse_loss(input_, target, mask=mask) == 0.0


def test_batched_mse_loss() -> None:
    input_ = torch.rand((2, 10, 3))
    target = torch.rand((2, 10, 3))

    weight = torch.tensor([[0.5, 0.75]]).T
    mask = torch.ones_like(input_)
    mask[0, 5:, :] = 0.0
    mask[1, 7:, :] = 0.0

    value = mse_loss(input_, target, weight, mask)
    # make sure we get a single value back
    assert value.shape == ()
