"""
Implementation of a sigmoid transformation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

from ._base import BaseTransform, _as_torch

log = logging.getLogger(__name__)


class SigmoidTransform(BaseTransform):
    """
    Sigmoid transformation. Does not require fitting, but is controlled only by the
    parameters k and x0.

    The sigmoid function is defined as:

    .. math::
        f(x) = \\frac{1}{1 + e^{-k(x - x0)}}

    Parameters
    ----------
    k : float, default=1.0
        The slope of the sigmoid function.
    x0 : float, default=0.0
        The x value at which the sigmoid function crosses 0.5.

    """

    _transform_type = BaseTransform.TransformType.X

    def __init__(self, k_: float = 1.0, x0_: float = 0.0):
        super().__init__()
        self.k_ = k_
        self.x0_ = x0_

    def fit(
        self,
        x: np.ndarray | torch.Tensor | None = None,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> SigmoidTransform:
        if x is not None:
            msg = "Sigmoid transform does not require fitting, ignoring input data"
            log.warning(msg)
        if y is not None:
            msg = "Sigmoid transform does not require fitting, ignoring target data"
            log.warning(msg)

        return self

    def transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)

        return torch.nn.functional.sigmoid(self.k_ * (x - self.x0_))

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for Sigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)

        return self.x0_ + torch.log(x / (1 - x)) / self.k_


class AdaptiveSigmoidTransform(BaseTransform):
    """
    Adaptive sigmoid transformation. The sigmoid function has a slope that decreases
    as x moves away from x0. The slope is defined as:
    .. math::
        k = k0 * (1 - alpha * np.abs(x - x0))

    The sigmoid function is defined as:

    .. math::
        f(x) = \\frac{1}{1 + e^{-k(x - x0)}}

    Parameters
    ----------
    k0_: float, default=1.0
        The maximum slope of the sigmoid function.
    alpha_: float, default=0.1
        The rate at which the slope decreases as x moves away from x0.
    x0_ : float, default=0.0
        The x value at which the sigmoid function crosses 0.5.
    atol_: float, default=1e-8
        The absolute tolerance for the convergence of the inverse transform.
    max_iter_: int, default=10000
        The maximum number of iterations for the convergence of the inverse transform.

    Notes
    -----
    The inverse transform is computed using Newton's method. The maximum number of iterations
    is controlled by the `max_iter_` parameter. The inverse transform may not converge for
    certain values of x, in which case a warning is raised and the last computed value is returned.
    The default value of `atol_` is 1e-8, which should be sufficient for most applications.
    """

    _transform_type = BaseTransform.TransformType.X

    def __init__(
        self,
        k0_: float = 1.0,
        alpha_: float = 0.1,
        x0_: float = 0.0,
        atol_: float = 1e-8,
        max_iter_: int = 10000,
    ):
        super().__init__()

        self.k0_ = k0_
        self.alpha_ = alpha_

        self.x0_ = x0_
        self.atol_ = atol_
        self.max_iter_ = max_iter_

    def fit(
        self,
        x: np.ndarray | torch.Tensor | None = None,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> AdaptiveSigmoidTransform:
        if x is not None:
            msg = "AdaptiveSigmoid transform does not require fitting, ignoring input data"
            log.warning(msg)
        if y is not None:
            msg = "AdaptiveSigmoid transform does not require fitting, ignoring target data"
            log.warning(msg)

        return self

    def transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for AdaptiveSigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)

        k = self.k0_ * (1 - self.alpha_ * torch.abs(x - self.x0_))

        return torch.nn.functional.sigmoid(k * (x - self.x0_))

    def inverse_transform(
        self,
        x: np.ndarray | torch.Tensor,
        y: np.ndarray | torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y is not None:
            msg = "y must be None for AdaptiveSigmoid transform"
            raise ValueError(msg)

        x = _as_torch(x)
        dtype = x.dtype
        x = x.to(torch.float64)

        logx = torch.log(x / (1 - x))
        x_guess = logx / self.k0_ + self.x0_

        prev_x_guess = torch.zeros_like(x_guess) - 1e6
        i = 0
        while (
            torch.max(torch.abs(x_guess - prev_x_guess)) > self.atol_
            and i < self.max_iter_
        ):
            if i == self.max_iter_ - 1:
                msg = "AdaptiveSigmoidTransform did not converge"
                log.warning(msg)
                break

            prev_x_guess = x_guess.clone()
            k = self.k0_ * (1 - self.alpha_ * torch.abs(x_guess - self.x0_))

            # newton iteration to find the root of the equation
            sig = torch.nn.functional.sigmoid(k * (x_guess - self.x0_))
            f = x - sig
            f_prime = -k * sig * (1 - sig)
            x_guess = x_guess - f / f_prime
            i += 1

        log.debug(f"AdaptiveSigmoidTransform converged in {i} iterations")

        return x_guess.to(dtype)
