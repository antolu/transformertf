"""
Implementation of Perona-Malik PDE solution for time series smoothing.

As written in https://arxiv.org/pdf/1412.6291.pdf
and implemented in https://wire.insiderfinance.io/preserving-edges-when-smoothening-time-series-data-90f9d965132e
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal as signal
from numba import njit, prange

__all__ = [
    "perona_malik_smooth",
    "convolve_PDE",
    "butter_lowpass_filter",
    "butter_lowpass",
    "mean_filter",
]


@njit
def convolve_PDE(
    U: npt.NDArray[np.float64 | np.float32], sigma: float = 1, k: float = 0.05
) -> npt.NDArray[np.float64]:
    """
    Perform Gaussian convolution by solving the heat equation with Neumann
    boundary conditions

    :param U: The array to perform convolution on.
    :param sigma: The standard deviation of the gaussian convolution.
    :param k: The step-size for the finite difference scheme (keep < 0.1 for accuracy)

    :return: The convolved function
    """

    t_end = sigma**2 / 2

    factor = 1 - 2 * k

    C = U.copy().astype(np.float64)
    for t in prange(int(t_end / k)):
        # Implementing the neumann boundary conditions
        C[0] = 2 * k * C[1] + factor * C[0]
        C[-1] = 2 * k * C[-2] + factor * C[-1]

        # Scheme on the interior nodes
        C[1:-1] = k * (C[2:] + C[:-2]) + factor * C[1:-1]

    return C


@njit
def gradient1(x: np.ndarray) -> np.ndarray:
    """
    Calculate the first order gradient for an array, excluding the edge values.

    :param x: array of length N

    :return: array of length N-2
    """
    return (x[2:] - x[:-2]) / 2


@njit
def gradient2(x: np.ndarray) -> np.ndarray:
    """
    Calculate the second order gradient for an array, excluding the edge values.

    :param x: array of length N

    :return: array of length N-2
    """
    return x[2:] - 2 * x[1:-1] + x[:-2]


@njit(parallel=True, nogil=True, cache=True)
def perona_malik_smooth(
    p: npt.NDArray[np.float32 | np.float64],
    alpha: float = 50.0,
    k: float = 0.05,
    t_end: float = 5.0,
) -> npt.NDArray[np.float64]:
    """
    Solve the Gaussian convolved Perona-Malik PDE using a basic finite
    difference scheme.

    Total number of iteration steps: t_end / k

    Parameters
    ----------
    p : np.array
        The signal to smoothen.
    alpha : float, optional
        A parameter to control how much the PDE resembles the heat equation,
        the perona malik PDE -> heat equation as alpha -> infinity
    k : float, optional
        The step size in time (keep < 0.1 for accuracy)
    t_end : float, optional
        When to termininate the algorithm the larger the t_end, the smoother
        the series
    Returns
    -------
    U : np.array=
        The Perona-Malik smoothened time series
    """

    U = p.astype(np.float64)

    for t in prange(int(t_end / k)):
        # Find the convolution of U with the gaussian, this ensures that the
        # PDE problem is well posed
        C = convolve_PDE(U, k=k)

        # Determine the derivatives by using matrix multiplication
        Cx = gradient1(C)
        Cxx = gradient2(C)

        Ux = gradient1(U)
        Uxx = gradient2(U)

        # Find the spatial component of the PDE
        PDE_space = (
            alpha * Uxx / (alpha + Cx**2)
            - 2 * alpha * Ux * Cx * Cxx / (alpha + Cx**2) ** 2
        )

        # Solve the PDE for the next time-step
        U[1:-1] += k * PDE_space

    return U


def butter_lowpass(cutoff: float, fs: float, order: int = 5) -> np.ndarray:
    """
    :param cutoff: The cutoff frequency of the filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.

    """
    return signal.butter(
        order, cutoff, fs=fs, btype="low", analog=False, output="sos"
    )


def butter_lowpass_filter(
    data: np.ndarray | pd.Series, cutoff: int, fs: float, order: int
) -> np.ndarray:
    """
    :param data: The data to filter.
    :param cutoff: The cutoff frequency of the filter.
    :param fs: The sampling frequency of the signal.
    :param order: The order of the filter.
    """
    sos = butter_lowpass(cutoff, fs, order)
    y = signal.sosfiltfilt(sos, data)
    return y


@njit
def mean_filter(
    value: np.ndarray,
    window_size: int = 10,
    stride: int = 1,
    threshold: float = 1e-6,
) -> np.ndarray:
    """
    Remove small amplitude fluctuations from a time series by
    replacing the values in a window with the mean of the window if
    the values in the window are close to each other.

    :param value: The array to smooth.
    :param window_size: The size of the window to smooth.
    :param stride: The stride between the windows.
    :param threshold: The threshold for the difference between the
        values in the window.
    """
    new_arr = value.copy()

    smoothed_last = False
    for i in range((value.size - window_size) // stride):
        s = slice(i * stride, i * stride + window_size)

        arr_s = value[s]
        condition = np.abs(arr_s - arr_s[0]) < threshold
        if np.all(condition):
            new_arr[s] = np.mean(arr_s)
            smoothed_last = True
        elif smoothed_last and np.argmin(condition) > 1:
            first_false = np.argmin(condition)
            new_arr[s.start : s.start + first_false] = np.mean(
                arr_s[:first_false]
            )
        else:
            new_arr[s.start : s.start + stride] = arr_s[:stride]
            smoothed_last = False

    return new_arr
