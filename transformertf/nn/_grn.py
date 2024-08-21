"""
Implementation of Gated Residual Network (GRN) module as described in the paper
"Temporal Fustion Transformers for Interpretable Multi-horizon Time Series Forecasting"
by Lim et al.
https://arxiv.org/abs/1912.09363
"""

from __future__ import annotations

import typing

import torch

from ._get_activation import VALID_ACTIVATIONS, get_activation
from ._glu import GatedLinearUnit
from ._resample_norm import ResampleNorm


class GatedResidualNetwork(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        context_dim: int | None = None,
        dropout: float = 0.1,
        activation: VALID_ACTIVATIONS = "elu",
        projection: typing.Literal["linear", "interpolate"] = "linear",
    ):
        """
        Gated Residual Network (GRN) module, based on the equation:

        .. math::
            \\text{GRN}(x) = x + \\text{GLU}(fc2(\\text{ELU}(fc1(x))) + fc3(c))

        Parameters
        ----------
        input_dim: int
            Input dimension
        hidden_dim: int
            Hidden dimension
        output_dim: int
            Output dimension
        context_dim: int, default=None
            Context dimension. If not None, the context tensor will be used to condition the GRN
        activation: ACTIVATIONS, default="elu"
            Activation function
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.context_dim = context_dim

        hidden_dim = hidden_dim or input_dim
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc3 = (
            torch.nn.Linear(context_dim, hidden_dim, bias=False)
            if context_dim
            else None
        )

        if input_dim != output_dim:
            if projection == "linear":
                self.project = torch.nn.Linear(input_dim, output_dim)
            elif projection == "interpolate":
                self.project = ResampleNorm(input_dim, output_dim)
            else:
                msg = (
                    f"Unknown resampling method: {projection} "
                    f"(choose from 'linear' or 'interpolate')"
                )
                raise ValueError(msg)
        else:
            self.project = torch.nn.Identity()

        self.glu1 = GatedLinearUnit(output_dim, dropout=dropout)
        self.norm = torch.nn.LayerNorm(output_dim)

        self.activation = get_activation(activation)

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim, n_features)
        context: torch.Tensor, default=None
            Context tensor of shape (batch_size, context_dim, n_features)
            Will be used to condition the GRN if not None. Required if context_dim is not None.

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, input_dim, n_features)
        """
        residual = self.project(x)

        x = self.fc1(x)
        if self.fc3 is not None:
            x += self.fc3(context)

        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        x = self.glu1(x)

        return self.norm(x + residual)
