"""
Quantile Loss implementation for uncertainty quantification in neural networks.

This module implements the quantile loss function, which is essential for
probabilistic forecasting and uncertainty quantification. The quantile loss
enables training neural networks to predict multiple quantiles of the target
distribution rather than just point estimates.

Classes
-------
QuantileLoss : torch.nn.Module
    Quantile loss function for multi-quantile prediction and uncertainty estimation.

Notes
-----
The quantile loss is asymmetric and penalizes under-prediction and over-prediction
differently based on the quantile level. This property makes it particularly
suitable for tasks where prediction intervals and uncertainty estimates are
important, such as time series forecasting and risk assessment.

The quantile loss is widely used in:
- Probabilistic time series forecasting
- Risk management applications
- Confidence interval estimation
- Robust regression problems

References
----------
.. [1] Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles."
   Econometrica (1978): 33-50.
.. [2] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
   multi-horizon time series forecasting." ICML 2021.
"""

from __future__ import annotations

import typing

import einops
import torch


class QuantileLoss(torch.nn.Module):
    """
    Quantile loss function for uncertainty quantification and probabilistic forecasting.

    This loss function enables training neural networks to predict multiple quantiles
    of the target distribution simultaneously. It provides asymmetric penalties that
    encourage the model to learn different aspects of the prediction uncertainty.

    The quantile loss for a given quantile level τ ∈ (0,1) is defined as:

    .. math::
        L_τ(y, ŷ) = \\max(τ(y - ŷ), (τ - 1)(y - ŷ))

    which can also be written as:

    .. math::
        L_τ(y, ŷ) = \\begin{cases}
        τ(y - ŷ) & \\text{if } y ≥ ŷ \\\\
        (1 - τ)(ŷ - y) & \\text{if } y < ŷ
        \\end{cases}

    For multiple quantiles, the total loss is the sum across all quantile levels:

    .. math::
        L_{total} = \\sum_{i=1}^{Q} L_{τ_i}(y, ŷ_i)

    where Q is the number of quantiles and τᵢ are the quantile levels.

    Parameters
    ----------
    quantiles : Sequence[float], optional
        List of quantile levels to predict. Each value must be in (0, 1).
        Default is [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98] which provides
        a comprehensive range for uncertainty quantification.

    Attributes
    ----------
    quantiles : torch.Tensor
        Tensor containing the quantile levels as a registered buffer.
        Shape: (num_quantiles,)

    Notes
    -----
    The quantile loss has several important properties:
    1. **Asymmetric penalties**: Under-prediction and over-prediction are
       penalized differently based on the quantile level
    2. **Probabilistic interpretation**: Each quantile represents a different
       confidence level for the prediction
    3. **Robustness**: Less sensitive to outliers compared to squared loss
    4. **Median special case**: τ=0.5 gives half the mean absolute error

    The default quantiles provide:
    - 0.02, 0.98: 96% prediction interval
    - 0.1, 0.9: 80% prediction interval
    - 0.25, 0.75: 50% prediction interval (interquartile range)
    - 0.5: Median prediction (point estimate)

    This loss is particularly effective for:
    - Time series forecasting with uncertainty
    - Risk-aware decision making
    - Financial modeling and risk management
    - Weather forecasting and climate modeling

    Examples
    --------
    >>> import torch
    >>> from transformertf.nn import QuantileLoss
    >>>
    >>> # Default quantiles
    >>> qloss = QuantileLoss()
    >>> print(f"Number of quantiles: {qloss.num_quantiles}")  # 7
    >>> print(f"Quantile levels: {qloss.quantiles}")
    >>>
    >>> # Custom quantiles for 90% prediction interval
    >>> qloss_custom = QuantileLoss([0.05, 0.5, 0.95])
    >>>
    >>> # Multi-quantile predictions
    >>> batch_size, seq_len, num_quantiles = 32, 128, 7
    >>> y_pred = torch.randn(batch_size, seq_len, num_quantiles)
    >>> target = torch.randn(batch_size, seq_len)
    >>>
    >>> # Compute loss
    >>> loss = qloss(y_pred, target)
    >>> print(f"Quantile loss: {loss.item():.4f}")
    >>>
    >>> # Get point prediction (median)
    >>> point_pred = qloss.point_prediction(y_pred)
    >>> print(f"Point prediction shape: {point_pred.shape}")
    >>>
    >>> # With sample weights
    >>> weights = torch.ones(batch_size, seq_len)
    >>> weighted_loss = qloss(y_pred, target, weights=weights)
    >>>
    >>> # With masking for variable-length sequences
    >>> mask = torch.ones(batch_size, seq_len)
    >>> mask[0, 10:] = 0  # Mask out padding positions
    >>> masked_loss = qloss(y_pred, target, mask=mask)
    >>>
    >>> # Combine mask and weights
    >>> combined_loss = qloss(y_pred, target, weights=weights, mask=mask)
    >>>
    >>> # Analyze prediction intervals
    >>> quantiles_pred = qloss.to_quantiles(y_pred)
    >>> q05, q50, q95 = quantiles_pred[..., 1], quantiles_pred[..., 3], quantiles_pred[..., 5]
    >>> interval_width = q95 - q05  # 90% prediction interval width
    >>> print(f"Average interval width: {interval_width.mean():.4f}")
    >>>
    >>> # Time series forecasting example
    >>> horizon = 24  # 24-hour forecast
    >>> y_pred_ts = torch.randn(64, horizon, 7)  # 64 series, 24 steps, 7 quantiles
    >>> target_ts = torch.randn(64, horizon)
    >>> ts_loss = qloss(y_pred_ts, target_ts)
    >>>
    >>> # Extract confidence intervals
    >>> median_forecast = qloss.point_prediction(y_pred_ts)
    >>> lower_bound = y_pred_ts[..., 1]  # 0.1 quantile
    >>> upper_bound = y_pred_ts[..., 5]  # 0.9 quantile
    >>> # Now you have 80% prediction intervals

    See Also
    --------
    transformertf.nn.WeightedMSELoss : Alternative loss for point predictions
    torch.nn.L1Loss : Standard L1 loss (equivalent to τ=0.5 quantile loss)
    torch.nn.MSELoss : Standard squared loss

    References
    ----------
    .. [1] Koenker, Roger, and Gilbert Bassett Jr. "Regression quantiles."
       Econometrica: journal of the Econometric Society (1978): 33-50.
    .. [2] Lim, Bryan, et al. "Temporal fusion transformers for interpretable
       multi-horizon time series forecasting." ICML 2021.
    .. [3] Salinas, David, et al. "DeepAR: Probabilistic forecasting with
       autoregressive recurrent networks." International Journal of Forecasting 36.3 (2020): 1181-1191.
    """

    quantiles: torch.Tensor

    def __init__(
        self,
        quantiles: typing.Sequence[float] | None = None,
    ):
        super().__init__()
        if quantiles is None:
            quantiles = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98]
        quantiles_t = torch.tensor(sorted(quantiles))
        self.register_buffer("quantiles", quantiles_t)

    def loss(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        *,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # calculate quantile loss
        if y_pred.ndim != target.ndim and y_pred.ndim != target.ndim + 1:
            msg = f"y_pred must have shape [batch_size, ..., n_quantiles], got {y_pred.shape} and {target.shape}"
            raise ValueError(msg)

        # Combine mask and weights
        if mask is not None and weights is not None:
            weights = weights * mask.float()
        elif mask is not None:
            weights = mask.float()
        elif weights is None:
            weights = 1.0

        if target.ndim <= 2:
            target = target.unsqueeze(-1)
        if y_pred.ndim == target.ndim + 1:
            y_pred = y_pred.unsqueeze(-1)

        # Expand weights to match quantile dimension if needed
        if isinstance(weights, torch.Tensor) and weights.dim() > 0:
            if weights.ndim == y_pred.ndim - 1:
                # Expand weights from (...) to (..., n_quantiles) to match y_pred
                weights = einops.repeat(weights, "... -> ... n", n=len(self.quantiles))
            elif weights.ndim == y_pred.ndim:
                if weights.shape[-1] != len(self.quantiles) and weights.shape[-1] != 1:
                    msg = (
                        f"weights must have shape [batch_size, ..., n_quantiles], "
                        f"got {weights.shape} and {y_pred.shape}"
                    )
                    raise ValueError(msg)

                weights = einops.repeat(
                    weights, "... 1 -> ... n", n=len(self.quantiles)
                )

        error = (
            einops.repeat(target, "... 1 -> ... n", n=len(self.quantiles)) - y_pred
        ) * weights

        loss = torch.max((self.quantiles - 1) * error, self.quantiles * error)

        return torch.sum(loss, dim=1)

    def forward(
        self,
        y_pred: torch.Tensor,
        target: torch.Tensor,
        weights: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # calculate quantile loss
        return self.loss(y_pred, target, weights=weights, mask=mask).mean()

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

    @property
    def num_quantiles(self) -> int:
        """
        Number of quantiles
        """
        return len(self.quantiles)
