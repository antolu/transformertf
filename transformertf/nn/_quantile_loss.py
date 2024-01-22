"""
Implementation of quantile loss
"""
from __future__ import annotations

import typing

import einops
import torch


class QuantileLoss(torch.nn.Module):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    quantiles: torch.Tensor

    def __init__(
        self,
        quantiles: typing.Sequence[float] | None = None,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__()
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        quantiles = torch.tensor(list(sorted(quantiles)))
        self.register_buffer("quantiles", quantiles)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # calculate quantile loss
        error = (
            einops.repeat(target, "... 1 -> ... n", n=len(self.quantiles))
            - y_pred
        )

        loss = torch.max((self.quantiles - 1) * error, self.quantiles * error)

        return torch.sum(loss, dim=1)

    def forward(
        self, y_pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # calculate quantile loss
        return self.loss(y_pred, target).mean()

    def point_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = len(self.quantiles) // 2
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred
