from __future__ import annotations

import torch

from ..utils._activation import ACTIVATIONS, get_activation


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int | tuple[int, ...] | None = None,
        output_dim: int = 1,
        dropout: float = 0.1,
        activation: ACTIVATIONS = "relu",
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        if hidden_dim is not None:
            if isinstance(hidden_dim, int):
                hidden_dim = (hidden_dim,)
            for dim in hidden_dim:
                layers.append(torch.nn.Linear(prev_dim, dim))
                layers.append(get_activation(activation))
                if dropout > 0:
                    layers.append(torch.nn.Dropout(dropout))
                prev_dim = dim
        layers.append(torch.nn.Linear(prev_dim, output_dim))

        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
