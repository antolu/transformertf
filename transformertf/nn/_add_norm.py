"""
Implemention of AddNorm layer, which is just an addition of two tensors followed by layer normalization.

This specific implementation also adds the `trainable_add` parameter, which allows
the skip connection to be trainable.
"""

from __future__ import annotations


import torch


class AddNorm(torch.nn.Module):
    def __init__(self, input_size: int, trainable_add: bool = True):
        """
        AddNorm layer, which is just an addition of two tensors followed by layer normalization.

        This specific implementation also adds the `trainable_add` parameter, which allows
        the skip connection to be trainable with a sigmoid mask.

        The mathematical equation is:

        .. math::
            \\text{AddNorm}(x, y) = \\text{LayerNorm}(x + y * \\sigma(\\text{mask}) * 2)

        Parameters
        ----------
        input_size : int
            Input size. The output size will be the same as the input size.
        trainable_add : bool, default=True
            Whether the skip connection should be trainable. If True, a trainable mask will be added
            to the skip connection.
        """
        super().__init__()
        self.input_size = input_size
        self.trainable_add = trainable_add

        if self.trainable_add:
            self.mask = torch.nn.Parameter(
                torch.zeros(self.input_size, dtype=torch.float)
            )
        else:
            self.mask = None

        self.norm = torch.nn.LayerNorm(self.input_size)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Forward pass implementation.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        y : torch.Tensor
            Skip connection tensor. Must have the same shape as `x`.

        Returns
        -------
        torch.Tensor
            Output tensor, with the same shape as `x`.
        """
        if self.trainable_add:
            assert self.mask is not None
            return self.norm(
                x + y * torch.nn.functional.sigmoid(self.mask) * 2.0
            )
        else:
            return self.norm(x + y)
