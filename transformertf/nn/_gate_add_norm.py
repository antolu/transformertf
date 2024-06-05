"""
Implementation of GateAddNorm block that combines a feedforward network with a gating mechanism and
layer normalization, and residual connection.
"""

from __future__ import annotations

import torch

from ._add_norm import AddNorm
from ._glu import GatedLinearUnit


class GateAddNorm(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        dropout: float = 0.0,
        *,
        trainable_add: bool = False,
    ):
        """
        GateAddNorm block that combines a feedforward network with a gating mechanism and
        layer normalization, and residual connection.

        Parameters
        ----------
        input_dim: int
            Input dimension
        output_dim: int
            Output dimension. The residual connection must have the same dimension as the output.
        dropout: float, default=0.0
            Dropout rate
        trainable_add: bool, default=False
            Whether the skip connection should be trainable. If True, a trainable mask will be added
            to the skip connection.
        """
        super().__init__()
        self.input_dim = input_dim
        output_dim = output_dim or input_dim
        self.output_dim = output_dim
        self.glu = GatedLinearUnit(input_dim, output_dim, dropout)

        self.add_norm = AddNorm(output_dim, trainable_add=trainable_add)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim, n_features)
        residual: torch.Tensor
            Residual tensor of shape (batch_size, output_dim, n_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, output_dim, n_features)
        """
        x = self.glu(x)
        return self.add_norm(x, residual)
