"""
This module contains functions for preprocessing data.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

import logging
import typing

import numpy as np

__all__ = ["WindowGenerator"]

log = logging.getLogger(__name__)


class WindowGenerator(typing.Sequence[slice]):
    def __init__(
        self,
        num_points: int,
        window_size: int,
        stride: int = 1,
        zero_pad: bool = False,
    ):
        """
        Class to generate sliding windows for input and label data.

        The class is agnostic to the data source and can be used with
        any data source that supports indexing and slicing. The only
        thing that is required is that the input and label data have
        the same length.

        Parameters
        ----------
        input_data : DATA_SOURCE
            Input data. Can be a pandas DataFrame, Series, or numpy array.
        in_window_size : int
            Input window size.
        label_data : DATA_SOURCE, optional
            Label data. Can be a pandas DataFrame, Series, or numpy array.
            The default is None.
        label_seq_len : int, optional
            Label window size. If set, the label window size is used instead of the
            The default is None.
        stride : int, optional
            Stride for the sliding window. The default is 1.
        zero_pad : bool, optional
            If True, the input and label data are zero padded to fit the stride.
            The default is False.

        Raises
        ------
        ValueError
            If the input and label data have different lengths.
        ValueError
            If the input data length is less than the input window size.
        """
        if window_size > num_points and not zero_pad:
            raise ValueError(
                "Input window size ({}) must be less than the input data "
                "length ({})".format(window_size, num_points)
            )

        self._window_size: int = window_size
        self._stride: int = stride
        self._label_data: np.ndarray | None

        round_op: typing.Callable[[float], float] = (  # type: ignore[assignment]
            np.ceil if zero_pad else np.floor  # type: ignore[assignment]
        )
        self._num_samples = int(
            round_op((num_points - window_size + stride) / stride)
        )

        if zero_pad:
            # extend input and label data to fit stride
            self._source_len = stride * self._num_samples + (
                window_size - stride
            )
        else:
            self._source_len = num_points

    @property
    def num_samples(self) -> int:
        """
        Number of samples that can be generated from the input data.

        Returns
        -------
        int
            Number of samples.
        """
        return self._num_samples

    @property
    def real_data_len(self) -> int:
        """
        Number of points including zero-padding.
        """
        return self._source_len

    def __len__(self) -> int:
        """
        Number of samples that can be generated from the input data.

        Returns
        -------
        int
            Number of samples.
        """
        return self.num_samples

    def calc_slice(self, idx: int) -> slice:
        """
        Calculate the slice (window size) for the input or label data at a
        specific index. First performs an index check to ensure that the index
        is within bounds.

        Parameters
        ----------
        idx : int
            Index of the sample. If index is out of bounds an IndexError is raised.

        Returns
        -------
        slice
            Slice object representing the window size.

        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        # Check index
        if idx < 0:
            idx = idx + self.num_samples

        if idx >= self._num_samples:
            raise IndexError(f"Index {idx} is out of bounds")

        # Calculate slice
        start = idx * self._stride
        stop = start + self._window_size

        return slice(start, stop)

    @typing.overload
    def __getitem__(self, idx: int) -> slice: ...

    @typing.overload
    def __getitem__(self, idx: slice) -> typing.Sequence[slice]: ...

    def __getitem__(self, idx: int | slice) -> slice | typing.Sequence[slice]:
        """
        Determine the type of sample to return based on the label window size.
        """
        if isinstance(idx, int):
            return self.calc_slice(idx)
        else:
            return [
                self.calc_slice(i)
                for i in range(
                    idx.start or 0, idx.stop or len(self), idx.step or 1
                )
            ]

    def __iter__(self) -> typing.Iterator[slice]:
        for i in range(len(self)):
            yield self[i]

    def __str__(self) -> str:
        return (
            f"WindowGenerator(window_size={self._window_size}, "
            f"stride={self._stride}, "
            f"num_samples={self._num_samples})"
        )

    def __repr__(self) -> str:
        return str(self)
