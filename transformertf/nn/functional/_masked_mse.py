from __future__ import annotations

import torch

from ...utils import maybe_compile


@maybe_compile
def masked_mse_loss(
    input: torch.Tensor,  # noqa: A002
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    torch.nn.functional.mse_loss(input, target, reduction="none")
    return torch.nansum((input - target) ** 2 * mask) / mask.sum()
