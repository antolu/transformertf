from __future__ import annotations

import torch
import torch.nn as nn


class GradientTorch(nn.Module):
    """Calculate the gradient along the sequence dimension."""

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """Run the module's forward pass.

        This accepts a 3D input with ``(batch, sequence, channels)``
        dimensions. On each forward iteration, it calculates the
        symmetric difference quotient between neighboring elements along
        the sequence dimension.
        """
        # pylint: disable = no-self-use
        # Replicate the last and first element respectively.
        after = torch.cat((data[:, 1:, :], data[:, -1:, :]), dim=1)
        before = torch.cat((data[:, 0:1, :], data[:, :-1, :]), dim=1)
        # The first and last gradient receive unit weight, the others
        # double that.
        weights = torch.ones_like(data)
        weights[:, 1:-1, :] = weights[:, 1:-1, :] * 2.0
        return (after - before) / weights
