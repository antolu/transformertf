"""
Simple module that adds two tensors and normalizes the result.
"""
from __future__ import annotations

import torch


class AddAndNorm(torch.nn.Module):
    def __init__(
        self, input_dim: int, residual_block: torch.nn.Module | None = None
    ):
        """
        Add and Normalize module, based on the equation:

        .. math::
            \\text{AddAndNorm}(x, y) = \\text{LayerNorm}(x + ResidualBlock(y))

        Where `ResidualBlock` is an optional residual block to be applied to `y` before adding it to `x`.

        Parameters
        ----------
        input_dim: int
            Input dimension
        residual_block: torch.nn.Module, default=None
            Residual block to be applied to `y` before adding it to `x`
        """
        super().__init__()
        self.norm = torch.nn.LayerNorm(input_dim)
        self.residual_block = residual_block

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim, n_features)
        y: torch.Tensor
            Second input tensor of shape (batch_size, input_dim, ...)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, input_dim, n_features)
        """
        y = y if self.residual_block is None else self.residual_block(y)
        return self.norm(x + y)
