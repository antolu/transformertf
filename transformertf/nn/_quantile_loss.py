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
        quantiles_t = torch.tensor(sorted(quantiles))
        self.register_buffer("quantiles", quantiles_t)

    def loss(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # calculate quantile loss
        if y_pred.ndim != target.ndim and y_pred.ndim != target.ndim + 1:
            msg = f"y_pred must have shape [batch_size, ..., n_quantiles], got {y_pred.shape} and {target.shape}"
            raise ValueError(msg)
        if (
            weights is not None
            and weights.ndim > 2
            and weights.shape[0] == target.shape[0]
        ):
            msg = f"weights must have shape [batch_size], got {weights.shape} and {target.shape}"
            raise ValueError(msg)

        if target.ndim <= 2:
            target = target.unsqueeze(-1)
        if y_pred.ndim <= 2:
            y_pred = y_pred.unsqueeze(0)
        error = einops.repeat(target, "... 1 -> ... n", n=len(self.quantiles)) - y_pred

        loss = torch.max((self.quantiles - 1) * error, self.quantiles * error)

        if weights is not None:
            if weights.ndim == 1:
                weights = weights.unsqueeze(-1)
            return torch.sum(loss, dim=1) * weights
        return torch.sum(loss, dim=1)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # calculate quantile loss
        return self.loss(y_pred, target, weights=weights).mean()

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
