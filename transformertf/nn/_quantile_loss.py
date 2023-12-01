"""
Implementation of quantile loss
"""
from __future__ import annotations

import typing
import torch


class QuantileLoss(torch.nn.Module):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

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
        quantiles = list(sorted(quantiles))
        self.quantiles = quantiles

    def forward(
        self, y_pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[..., i]
            losses.append(
                torch.max((q - 1) * errors, q * errors).unsqueeze(-1)
            )
        losses = 2 * torch.cat(losses, dim=2)

        return losses

    def point_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
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
