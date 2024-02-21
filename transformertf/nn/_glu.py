"""
Gate Linear Unit (GLU) module as described in the paper
"Language Modeling with Gated Convolutional Networks" by Dauphin et al.
https://arxiv.org/abs/1612.08083
"""

from __future__ import annotations

import torch


class GatedLinearUnit(torch.nn.Module):
    def __init__(self, input_dim: int):
        """
        Gate Linear Unit (GLU) module, based on the equation:

        .. math::
            \\text{GLU}(x) = fc1(x) \\odot \\sigma(fc2(x))

        This implementation is a faster version of the original GLU, as it uses a single linear layer
        instead of two separate layers, using only one matrix multiplication instead of two.

        Parameters
        ----------
        input_dim: int
            Input dimension
        """
        super().__init__()
        self.input_dim = input_dim
        self.fc1 = torch.nn.Linear(input_dim, input_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x: torch.Tensor
            Input tensor of shape (batch_size, input_dim, n_features)

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, input_dim, n_features)
        """
        x = self.fc1(x)
        chunks = x.chunk(2, dim=-1)
        return chunks[0] * torch.sigmoid(chunks[1])
