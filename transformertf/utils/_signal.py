"""
Implementation of Perona-Malik PDE solution for time series smoothing.

As written in https://arxiv.org/pdf/1412.6291.pdf
and implemented in https://wire.insiderfinance.io/preserving-edges-when-smoothening-time-series-data-90f9d965132e
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from numba import njit, prange
from scipy import signal

__all__ = [
    "butter_lowpass",
    "butter_lowpass_filter",
    "convolve_PDE",
    "mean_filter",
    "perona_malik_smooth",
]


@njit(cache=True)
def convolve_PDE(  # noqa: N802
    U: npt.NDArray[np.float64 | np.float32],  # noqa: N803
    sigma: float = 1,
    k: float = 0.05,
) -> npt.NDArray[np.float64]:
    """
    Perform Gaussian convolution by solving the heat equation with Neumann boundary conditions.

    This function implements a finite difference scheme to solve the heat equation,
    which produces a Gaussian convolution of the input signal. It uses Neumann
    boundary conditions to handle the edges of the signal appropriately.

    Parameters
    ----------
    U : npt.NDArray[np.float64 | np.float32]
        The 1D array to perform convolution on. Input signal to be smoothed.
    sigma : float, optional
        The standard deviation of the Gaussian convolution kernel. Controls
        the amount of smoothing. Default is 1.0.
    k : float, optional
        The time step size for the finite difference scheme. Must be less than
        0.1 for numerical stability and accuracy. Default is 0.05.

    Returns
    -------
    npt.NDArray[np.float64]
        The convolved (smoothed) signal as a float64 array with the same
        length as the input.

    Notes
    -----
    This function implements the heat equation:

    ∂u/∂t = ∂²u/∂x²

    The relationship between the convolution width σ and time t is:
    t = σ²/2

    Neumann boundary conditions (zero flux) are applied at the boundaries,
    meaning the derivative at the edges is zero. This preserves the signal
    energy and prevents artificial edge effects.

    The function is compiled with Numba for performance and cached for
    efficiency in repeated calls.

    Examples
    --------
    Basic Gaussian smoothing:

    >>> import numpy as np
    >>> from transformertf.utils._signal import convolve_PDE
    >>>
    >>> # Create noisy signal
    >>> x = np.linspace(0, 10, 1000)
    >>> signal = np.sin(x) + 0.1 * np.random.randn(len(x))
    >>>
    >>> # Apply Gaussian smoothing
    >>> smoothed = convolve_PDE(signal, sigma=0.5)
    >>> print(f"Original shape: {signal.shape}, Smoothed shape: {smoothed.shape}")
    Original shape: (1000,) Smoothed shape: (1000,)

    Different levels of smoothing:

    >>> # Light smoothing
    >>> light_smooth = convolve_PDE(signal, sigma=0.2)
    >>>
    >>> # Heavy smoothing
    >>> heavy_smooth = convolve_PDE(signal, sigma=2.0)

    Custom time step for high accuracy:

    >>> # Use smaller time step for better accuracy
    >>> accurate_smooth = convolve_PDE(signal, sigma=1.0, k=0.01)

    See Also
    --------
    perona_malik_smooth : Edge-preserving smoothing using Perona-Malik PDE
    butter_lowpass_filter : Butterworth lowpass filtering
    scipy.ndimage.gaussian_filter1d : SciPy's Gaussian filter
    """

    t_end = sigma**2 / 2

    factor = 1 - 2 * k

    C = U.copy().astype(np.float64)
    for _t in prange(int(t_end / k)):
        # Implementing the neumann boundary conditions
        C[0] = 2 * k * C[1] + factor * C[0]
        C[-1] = 2 * k * C[-2] + factor * C[-1]

        # Scheme on the interior nodes
        C[1:-1] = k * (C[2:] + C[:-2]) + factor * C[1:-1]

    return C


@njit(cache=True)
def gradient1(x: np.ndarray) -> np.ndarray:
    """
    Calculate the first-order gradient using central differences.

    This function computes the first derivative of the input array using
    central difference approximation, excluding the edge values to avoid
    boundary artifacts.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array of length N for which to compute the gradient.

    Returns
    -------
    np.ndarray
        First-order gradient array of length N-2. The gradient at position i
        corresponds to the slope at position i+1 in the original array.

    Notes
    -----
    The central difference formula used is:

    gradient[i] = (x[i+2] - x[i]) / 2

    This provides a second-order accurate approximation to the first derivative.
    The function is compiled with Numba for high performance.

    Examples
    --------
    >>> import numpy as np
    >>> from transformertf.utils._signal import gradient1
    >>>
    >>> # Linear function - constant gradient
    >>> x = np.linspace(0, 10, 11)  # [0, 1, 2, ..., 10]
    >>> grad = gradient1(x)
    >>> print(grad)  # Should be approximately [1, 1, 1, ..., 1]

    See Also
    --------
    gradient2 : Second-order gradient calculation
    numpy.gradient : NumPy's gradient function with edge handling
    """
    return (x[2:] - x[:-2]) / 2


@njit(cache=True)
def gradient2(x: np.ndarray) -> np.ndarray:
    """
    Calculate the second-order gradient (discrete Laplacian) using finite differences.

    This function computes the second derivative of the input array using
    finite difference approximation, excluding the edge values to avoid
    boundary artifacts.

    Parameters
    ----------
    x : np.ndarray
        Input 1D array of length N for which to compute the second derivative.

    Returns
    -------
    np.ndarray
        Second-order gradient array of length N-2. The second derivative at
        position i corresponds to the curvature at position i+1 in the original array.

    Notes
    -----
    The finite difference formula used is:

    gradient2[i] = x[i+2] - 2*x[i+1] + x[i]

    This is the discrete second derivative or Laplacian operator, which measures
    the local curvature of the signal. Positive values indicate concave up
    regions, negative values indicate concave down regions.

    The function is compiled with Numba for high performance and is used
    internally by the Perona-Malik smoothing algorithm.

    Examples
    --------
    >>> import numpy as np
    >>> from transformertf.utils._signal import gradient2
    >>>
    >>> # Quadratic function - constant second derivative
    >>> x = np.array([0, 1, 4, 9, 16, 25])  # x^2 for x=[0,1,2,3,4,5]
    >>> grad2 = gradient2(x)
    >>> print(grad2)  # Should be approximately [2, 2, 2, 2]

    See Also
    --------
    gradient1 : First-order gradient calculation
    perona_malik_smooth : Uses this function for edge detection
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
    Apply edge-preserving smoothing using the Perona-Malik PDE method.

    This function implements the Perona-Malik partial differential equation
    for anisotropic diffusion, which smooths signals while preserving edges.
    Unlike Gaussian smoothing, this method reduces smoothing near edges
    (high gradients) and increases smoothing in flat regions.

    Parameters
    ----------
    p : npt.NDArray[np.float32 | np.float64]
        The 1D signal to smooth. Input time series or signal data.
    alpha : float, optional
        Edge preservation parameter. Controls the transition between
        edge-preserving and isotropic diffusion:

        - Small values (< 10): Strong edge preservation
        - Large values (> 100): Approaches standard heat equation
        - Default is 50.0 for balanced smoothing
    k : float, optional
        Time step size for the finite difference scheme. Must be less than
        0.1 for numerical stability. Smaller values give higher accuracy
        but require more computation. Default is 0.05.
    t_end : float, optional
        Total evolution time. Larger values produce more smoothing.
        The algorithm runs for int(t_end / k) iterations. Default is 5.0.

    Returns
    -------
    npt.NDArray[np.float64]
        The edge-preserving smoothed signal as a float64 array with the
        same length as the input.

    Notes
    -----
    The Perona-Malik equation is:

    ∂u/∂t = ∇ · (g(|∇u|) ∇u)

    where g(s) = α/(α + s²) is the diffusion function that reduces
    diffusion near edges (high gradients).

    This implementation:

    - Uses Gaussian pre-smoothing to ensure well-posed PDE
    - Applies finite difference scheme for spatial derivatives
    - Employs parallel processing for improved performance
    - Maintains signal boundaries with appropriate conditions

    The method is particularly effective for:

    - Financial time series with regime changes
    - Sensor data with step changes
    - Signals with important discontinuities
    - Noise reduction while preserving features

    Examples
    --------
    Basic edge-preserving smoothing:

    >>> import numpy as np
    >>> from transformertf.utils._signal import perona_malik_smooth
    >>>
    >>> # Create signal with edges and noise
    >>> t = np.linspace(0, 10, 1000)
    >>> signal = np.piecewise(t, [t < 3, (t >= 3) & (t < 7), t >= 7], [1, 3, 1])
    >>> noisy_signal = signal + 0.1 * np.random.randn(len(signal))
    >>>
    >>> # Apply edge-preserving smoothing
    >>> smoothed = perona_malik_smooth(noisy_signal)
    >>> print(f"Signal preserved edges better than Gaussian smoothing")

    Different edge preservation levels:

    >>> # Strong edge preservation
    >>> strong_edges = perona_malik_smooth(noisy_signal, alpha=10.0)
    >>>
    >>> # Moderate edge preservation
    >>> moderate_edges = perona_malik_smooth(noisy_signal, alpha=50.0)
    >>>
    >>> # Weak edge preservation (closer to Gaussian)
    >>> weak_edges = perona_malik_smooth(noisy_signal, alpha=200.0)

    Custom evolution parameters:

    >>> # Light smoothing with short evolution
    >>> light_smooth = perona_malik_smooth(noisy_signal, t_end=2.0)
    >>>
    >>> # Heavy smoothing with long evolution
    >>> heavy_smooth = perona_malik_smooth(noisy_signal, t_end=10.0)

    High-accuracy processing:

    >>> # Use smaller time step for critical applications
    >>> accurate = perona_malik_smooth(
    ...     noisy_signal, alpha=30.0, k=0.01, t_end=3.0
    ... )

    Financial time series application:

    >>> # Smooth price data while preserving trend changes
    >>> price_data = np.random.cumsum(np.random.randn(1000)) + 100
    >>> smoothed_prices = perona_malik_smooth(
    ...     price_data, alpha=25.0, t_end=3.0
    ... )

    See Also
    --------
    convolve_PDE : Gaussian convolution via heat equation
    butter_lowpass_filter : Butterworth lowpass filtering
    mean_filter : Simple mean filtering with thresholding
    """

    U = p.astype(np.float64)

    for _t in prange(int(t_end / k)):
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
    Design a Butterworth lowpass filter in second-order sections form.

    This function creates the filter coefficients for a Butterworth lowpass
    filter, which provides a maximally flat passband response and smooth
    transition to the stopband.

    Parameters
    ----------
    cutoff : float
        The cutoff frequency of the filter in Hz. Frequencies above this
        value will be attenuated.
    fs : float
        The sampling frequency of the signal in Hz. Must be at least twice
        the cutoff frequency (Nyquist criterion).
    order : int, optional
        The order of the filter. Higher orders provide sharper transitions
        but may introduce more phase distortion. Default is 5.

    Returns
    -------
    np.ndarray
        Second-order sections representation of the filter coefficients.
        Shape is (n_sections, 6) where each row contains [b0, b1, b2, a0, a1, a2]
        for one second-order section.

    Notes
    -----
    Butterworth filters are characterized by:

    - Maximally flat magnitude response in the passband
    - Monotonic response (no ripples in pass or stopband)
    - -3dB attenuation at the cutoff frequency
    - Roll-off rate of -20n dB/decade (where n is the order)

    The second-order sections (SOS) format provides better numerical
    stability compared to transfer function coefficients, especially
    for high-order filters.

    Examples
    --------
    Design a lowpass filter for audio processing:

    >>> import numpy as np
    >>> from transformertf.utils._signal import butter_lowpass
    >>>
    >>> # Design filter to remove frequencies above 1kHz from 44.1kHz audio
    >>> sos = butter_lowpass(cutoff=1000, fs=44100, order=4)
    >>> print(f"Filter sections: {sos.shape[0]}")
    Filter sections: 2

    Different filter orders:

    >>> # Low order - gentle roll-off
    >>> gentle_filter = butter_lowpass(100, 1000, order=2)
    >>>
    >>> # High order - sharp roll-off
    >>> sharp_filter = butter_lowpass(100, 1000, order=8)

    Filter for time series data:

    >>> # Remove high-frequency noise from sensor data sampled at 100Hz
    >>> sensor_filter = butter_lowpass(cutoff=10, fs=100, order=6)

    See Also
    --------
    butter_lowpass_filter : Apply the designed filter to data
    scipy.signal.butter : SciPy's Butterworth filter design
    scipy.signal.filtfilt : Zero-phase filtering
    """
    return signal.butter(order, cutoff, fs=fs, btype="low", analog=False, output="sos")


def butter_lowpass_filter(
    data: np.ndarray, cutoff: int, fs: float, order: int
) -> np.ndarray:
    """
    Apply a Butterworth lowpass filter to the input data with zero-phase distortion.

    This function designs a Butterworth lowpass filter and applies it to the input
    data using zero-phase filtering, which eliminates phase distortion by filtering
    the data forward and backward.

    Parameters
    ----------
    data : np.ndarray
        The input signal to filter. Can be 1D or multi-dimensional, with filtering
        applied along the last axis.
    cutoff : int
        The cutoff frequency of the filter in Hz. Frequencies above this value
        will be attenuated.
    fs : float
        The sampling frequency of the signal in Hz. Must be greater than twice
        the cutoff frequency.
    order : int
        The order of the Butterworth filter. Higher orders provide sharper
        transitions but may be more sensitive to numerical errors.

    Returns
    -------
    np.ndarray
        The filtered signal with the same shape as the input. High-frequency
        components above the cutoff are attenuated.

    Notes
    -----
    This function combines filter design and application:

    1. Designs Butterworth filter using `butter_lowpass`
    2. Applies zero-phase filtering using `scipy.signal.sosfiltfilt`

    Zero-phase filtering advantages:

    - No phase distortion (important for preserving signal timing)
    - Effective filter order is doubled (steeper roll-off)
    - Suitable for offline processing of complete signals

    The function handles edge effects automatically and works with
    signals of various lengths, though very short signals may have
    artifacts near the boundaries.

    Examples
    --------
    Filter noisy time series data:

    >>> import numpy as np
    >>> from transformertf.utils._signal import butter_lowpass_filter
    >>>
    >>> # Create noisy signal: sine wave + high-frequency noise
    >>> t = np.linspace(0, 2, 1000)  # 2 seconds at 500 Hz
    >>> clean_signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
    >>> noise = 0.3 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
    >>> noisy_signal = clean_signal + noise
    >>>
    >>> # Apply lowpass filter to remove noise
    >>> filtered = butter_lowpass_filter(noisy_signal, cutoff=10, fs=500, order=4)
    >>> print(f"Noise reduced, signal preserved")

    Process sensor data:

    >>> # Remove high-frequency vibrations from accelerometer data
    >>> fs = 1000  # 1 kHz sampling rate
    >>> cutoff = 50  # Keep frequencies below 50 Hz
    >>> filtered_accel = butter_lowpass_filter(accel_data, cutoff, fs, order=6)

    Financial data smoothing:

    >>> # Smooth daily price data (remove high-frequency fluctuations)
    >>> # Assuming daily data for 2 years (roughly 500 samples)
    >>> daily_prices = np.random.cumsum(np.random.randn(500)) + 100
    >>> smoothed_prices = butter_lowpass_filter(
    ...     daily_prices, cutoff=0.1, fs=1.0, order=3
    ... )

    Multi-channel data:

    >>> # Filter multiple channels simultaneously
    >>> multi_channel = np.random.randn(1000, 3)  # 3 channels, 1000 samples
    >>> filtered_channels = butter_lowpass_filter(
    ...     multi_channel, cutoff=20, fs=200, order=5
    ... )
    >>> print(f"Filtered shape: {filtered_channels.shape}")
    Filtered shape: (1000, 3)

    See Also
    --------
    butter_lowpass : Design Butterworth lowpass filter
    perona_malik_smooth : Edge-preserving smoothing
    scipy.signal.sosfiltfilt : Zero-phase filtering function
    """
    sos = butter_lowpass(cutoff, fs, order)
    return signal.sosfiltfilt(sos, data)


@njit(cache=True)
def mean_filter(
    value: np.ndarray,
    window_size: int = 10,
    stride: int = 1,
    threshold: float = 1e-6,
) -> np.ndarray:
    """
    Apply adaptive mean filtering to remove small amplitude fluctuations.

    This function applies mean filtering selectively to regions where values
    are close to each other (within a threshold), effectively removing small
    fluctuations while preserving larger variations and trends in the signal.

    Parameters
    ----------
    value : np.ndarray
        The 1D input array to smooth. Time series or signal data.
    window_size : int, optional
        The size of the sliding window for computing means. Larger windows
        provide more smoothing but may over-smooth important features.
        Default is 10.
    stride : int, optional
        The step size between consecutive windows. A stride of 1 processes
        every possible window position. Larger strides process fewer windows
        but run faster. Default is 1.
    threshold : float, optional
        The maximum allowed difference between values in a window for mean
        filtering to be applied. Only windows where all values are within
        this threshold of the first value get smoothed. Default is 1e-6.

    Returns
    -------
    np.ndarray
        The filtered array with the same shape as the input. Regions with
        small fluctuations are smoothed, while larger variations are preserved.

    Notes
    -----
    The algorithm works as follows:

    1. Slide a window across the signal with the specified stride
    2. Check if all values in the window are within `threshold` of the first value
    3. If yes, replace the window values with their mean
    4. Handle boundary cases to ensure smooth transitions

    This selective approach preserves important signal features while removing
    small-amplitude noise. It's particularly effective for:

    - Removing measurement noise from relatively stable signals
    - Smoothing plateaus in step-like signals
    - Preprocessing signals before feature extraction
    - Reducing storage requirements for quasi-constant regions

    The function is compiled with Numba for high performance on large arrays.

    Examples
    --------
    Remove small fluctuations from a step signal:

    >>> import numpy as np
    >>> from transformertf.utils._signal import mean_filter
    >>>
    >>> # Create step signal with small noise
    >>> signal = np.concatenate([
    ...     np.ones(100) + 0.001 * np.random.randn(100),  # Noisy plateau at 1
    ...     3 * np.ones(100) + 0.001 * np.random.randn(100),  # Noisy plateau at 3
    ...     np.ones(100) + 0.001 * np.random.randn(100)   # Back to 1
    ... ])
    >>>
    >>> # Apply adaptive mean filter
    >>> filtered = mean_filter(signal, window_size=20, threshold=0.01)
    >>> print(f"Small noise removed, steps preserved")

    Process sensor data with different parameters:

    >>> # Fine-scale smoothing
    >>> fine_smooth = mean_filter(sensor_data, window_size=5, threshold=0.001)
    >>>
    >>> # Coarse-scale smoothing
    >>> coarse_smooth = mean_filter(sensor_data, window_size=50, threshold=0.1)

    Efficient processing with larger stride:

    >>> # Process every 5th position for faster computation
    >>> fast_filter = mean_filter(
    ...     large_signal, window_size=20, stride=5, threshold=0.01
    ... )

    Different threshold levels:

    >>> # Very sensitive - only smooth nearly constant regions
    >>> sensitive = mean_filter(signal, threshold=1e-8)
    >>>
    >>> # Less sensitive - smooth larger fluctuations
    >>> permissive = mean_filter(signal, threshold=0.1)

    Compare with other smoothing methods:

    >>> # Adaptive mean filter preserves edges better than uniform smoothing
    >>> from scipy.ndimage import uniform_filter1d
    >>> uniform_smooth = uniform_filter1d(signal, size=20)
    >>> adaptive_smooth = mean_filter(signal, window_size=20, threshold=0.01)
    >>> # adaptive_smooth will preserve steps better than uniform_smooth

    See Also
    --------
    perona_malik_smooth : Edge-preserving PDE-based smoothing
    butter_lowpass_filter : Frequency-domain filtering
    scipy.ndimage.uniform_filter1d : Uniform mean filtering
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
            new_arr[s.start : s.start + first_false] = np.mean(arr_s[:first_false])
        else:
            new_arr[s.start : s.start + stride] = arr_s[:stride]
            smoothed_last = False

    return new_arr
