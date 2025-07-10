"""
Window Strategy pattern for decoupling windowing logic from data modules.

This module provides the strategy pattern for handling different windowing
approaches for time series data, allowing for flexible window generation
without tight coupling to specific data modules.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ._sample_generator import (
    TimeSeriesSampleGenerator,
    TransformerSampleGenerator,
)

__all__ = [
    "PredictWindowStrategy",
    "RandomizedWindowStrategy",
    "TimeSeriesWindowStrategy",
    "TransformerWindowStrategy",
    "WindowStrategy",
    "WindowStrategyFactory",
]


class WindowStrategy(ABC):
    """
    Abstract base class for window generation strategies.

    This class defines the interface for different windowing approaches
    used in time series data processing. Concrete implementations handle
    specific windowing logic while maintaining a consistent interface.
    """

    @abstractmethod
    def create_sample_generators(
        self,
        input_data: list[pd.DataFrame],
        target_data: list[pd.DataFrame] | list[None],
        **kwargs: Any,
    ) -> list[Any]:
        """
        Create sample generators for the given data.

        Parameters
        ----------
        input_data : List[pd.DataFrame]
            List of input DataFrames.
        target_data : List[pd.DataFrame] | List[None]
            List of target DataFrames or None values.
        **kwargs : Any
            Additional strategy-specific parameters.

        Returns
        -------
        List[Any]
            List of sample generators.
        """

    @abstractmethod
    def get_window_parameters(self) -> dict[str, Any]:
        """
        Get the window parameters for this strategy.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing window parameters.
        """

    @abstractmethod
    def validate_parameters(self, **kwargs: Any) -> None:
        """
        Validate strategy-specific parameters.

        Parameters
        ----------
        **kwargs : Any
            Parameters to validate.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """


class TimeSeriesWindowStrategy(WindowStrategy):
    """
    Window strategy for time series datasets.

    This strategy creates windows suitable for sequence-to-sequence
    time series modeling where all features are known throughout
    the sequence.

    Parameters
    ----------
    seq_len : int
        Length of each sequence window.
    stride : int, optional
        Step size for sliding window. Default is 1.
    min_seq_len : int | None, optional
        Minimum sequence length for randomization. Default is None.
    randomize_seq_len : bool, optional
        Whether to randomize sequence lengths. Default is False.
    predict : bool, optional
        Whether windowing is for prediction. Default is False.
    """

    def __init__(
        self,
        seq_len: int,
        stride: int = 1,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        predict: bool = False,
    ):
        self.seq_len = seq_len
        self.stride = stride
        self.min_seq_len = min_seq_len
        self.randomize_seq_len = randomize_seq_len
        self.predict = predict

        self.validate_parameters()

    def validate_parameters(self, **kwargs: Any) -> None:
        """Validate time series window parameters."""
        if self.seq_len <= 0:
            msg = "seq_len must be positive"
            raise ValueError(msg)

        if self.stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)

        if self.randomize_seq_len:
            if self.min_seq_len is None:
                msg = "min_seq_len must be specified when randomize_seq_len is True"
                raise ValueError(msg)
            if self.min_seq_len <= 0:
                msg = "min_seq_len must be positive"
                raise ValueError(msg)
            if self.min_seq_len > self.seq_len:
                msg = "min_seq_len must be less than or equal to seq_len"
                raise ValueError(msg)

    def create_sample_generators(
        self,
        input_data: list[pd.DataFrame],
        target_data: list[pd.DataFrame] | list[None],
        **kwargs: Any,
    ) -> list[TimeSeriesSampleGenerator]:
        """Create time series sample generators."""
        # Apply stride by creating multiple generators with different offsets
        generators = []

        for start_offset in range(self.stride):
            for input_df, target_df in zip(input_data, target_data, strict=False):
                # Apply stride offset to data
                strided_input = input_df.iloc[start_offset :: self.stride]
                strided_target = (
                    target_df.iloc[start_offset :: self.stride]
                    if target_df is not None
                    else None
                )

                generator = TimeSeriesSampleGenerator(
                    input_data=strided_input,
                    window_size=self.seq_len,
                    label_data=strided_target,
                    stride=self.seq_len if self.predict else 1,
                    zero_pad=self.predict,
                )
                generators.append(generator)

        return generators

    def get_window_parameters(self) -> dict[str, Any]:
        """Get window parameters for time series strategy."""
        return {
            "seq_len": self.seq_len,
            "stride": self.stride,
            "min_seq_len": self.min_seq_len,
            "randomize_seq_len": self.randomize_seq_len,
            "predict": self.predict,
        }


class TransformerWindowStrategy(WindowStrategy):
    """
    Window strategy for transformer-based models.

    This strategy creates windows suitable for encoder-decoder
    transformer architectures with separate context and target
    sequence lengths.

    Parameters
    ----------
    ctx_seq_len : int
        Context (encoder) sequence length.
    tgt_seq_len : int
        Target (decoder) sequence length.
    stride : int, optional
        Step size for sliding window. Default is 1.
    min_ctx_seq_len : int | None, optional
        Minimum context sequence length. Default is None.
    min_tgt_seq_len : int | None, optional
        Minimum target sequence length. Default is None.
    randomize_seq_len : bool, optional
        Whether to randomize sequence lengths. Default is False.
    predict : bool, optional
        Whether windowing is for prediction. Default is False.
    """

    def __init__(
        self,
        ctx_seq_len: int,
        tgt_seq_len: int,
        stride: int = 1,
        min_ctx_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        predict: bool = False,
    ):
        self.ctx_seq_len = ctx_seq_len
        self.tgt_seq_len = tgt_seq_len
        self.stride = stride
        self.min_ctx_seq_len = min_ctx_seq_len
        self.min_tgt_seq_len = min_tgt_seq_len
        self.randomize_seq_len = randomize_seq_len
        self.predict = predict

        self.validate_parameters()

    def validate_parameters(self, **kwargs: Any) -> None:
        """Validate transformer window parameters."""
        if self.ctx_seq_len <= 0:
            msg = "ctx_seq_len must be positive"
            raise ValueError(msg)

        if self.tgt_seq_len <= 0:
            msg = "tgt_seq_len must be positive"
            raise ValueError(msg)

        if self.stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)

        if self.randomize_seq_len:
            if self.min_ctx_seq_len is None:
                msg = "min_ctx_seq_len must be specified when randomize_seq_len is True"
                raise ValueError(msg)
            if self.min_tgt_seq_len is None:
                msg = "min_tgt_seq_len must be specified when randomize_seq_len is True"
                raise ValueError(msg)
            if self.min_ctx_seq_len <= 0 or self.min_tgt_seq_len <= 0:
                msg = "min_ctx_seq_len and min_tgt_seq_len must be positive"
                raise ValueError(msg)
            if self.min_ctx_seq_len > self.ctx_seq_len:
                msg = "min_ctx_seq_len must be less than or equal to ctx_seq_len"
                raise ValueError(msg)
            if self.min_tgt_seq_len > self.tgt_seq_len:
                msg = "min_tgt_seq_len must be less than or equal to tgt_seq_len"
                raise ValueError(msg)

    def create_sample_generators(
        self,
        input_data: list[pd.DataFrame],
        target_data: list[pd.DataFrame] | list[None],
        known_past_data: list[pd.DataFrame] | None = None,
        time_data: list[pd.DataFrame] | None = None,
        **kwargs: Any,
    ) -> list[TransformerSampleGenerator]:
        """Create transformer sample generators."""
        generators = []

        # Handle optional data lists
        if known_past_data is None:
            known_past_data = [None] * len(input_data)
        if time_data is None:
            time_data = [None] * len(input_data)

        for input_df, target_df, past_df, _time_df in zip(
            input_data, target_data, known_past_data, time_data, strict=False
        ):
            generator = TransformerSampleGenerator(
                input_data=input_df,
                target_data=target_df,
                src_seq_len=self.ctx_seq_len,
                tgt_seq_len=self.tgt_seq_len,
                known_past_data=past_df,
                stride=self.stride,
                **kwargs,
            )
            generators.append(generator)

        return generators

    def get_window_parameters(self) -> dict[str, Any]:
        """Get window parameters for transformer strategy."""
        return {
            "ctx_seq_len": self.ctx_seq_len,
            "tgt_seq_len": self.tgt_seq_len,
            "stride": self.stride,
            "min_ctx_seq_len": self.min_ctx_seq_len,
            "min_tgt_seq_len": self.min_tgt_seq_len,
            "randomize_seq_len": self.randomize_seq_len,
            "predict": self.predict,
        }


class PredictWindowStrategy(WindowStrategy):
    """
    Window strategy for prediction tasks.

    This strategy creates windows specifically for prediction,
    using overlapping windows with zero-padding to handle
    sequence boundaries.

    Parameters
    ----------
    window_size : int
        Size of prediction windows.
    stride : int, optional
        Step size for sliding window. Default is 1.
    zero_pad : bool, optional
        Whether to zero-pad incomplete windows. Default is True.
    """

    def __init__(
        self,
        window_size: int,
        stride: int = 1,
        zero_pad: bool = True,
    ):
        self.window_size = window_size
        self.stride = stride
        self.zero_pad = zero_pad

        self.validate_parameters()

    def validate_parameters(self, **kwargs: Any) -> None:
        """Validate prediction window parameters."""
        if self.window_size <= 0:
            msg = "window_size must be positive"
            raise ValueError(msg)

        if self.stride <= 0:
            msg = "stride must be positive"
            raise ValueError(msg)

    def create_sample_generators(
        self,
        input_data: list[pd.DataFrame],
        target_data: list[pd.DataFrame] | list[None],
        **kwargs: Any,
    ) -> list[TimeSeriesSampleGenerator]:
        """Create prediction sample generators."""
        generators = []

        for input_df in input_data:
            generator = TimeSeriesSampleGenerator(
                input_data=input_df,
                window_size=self.window_size,
                label_data=None,  # No labels in prediction
                stride=self.stride,
                zero_pad=self.zero_pad,
            )
            generators.append(generator)

        return generators

    def get_window_parameters(self) -> dict[str, Any]:
        """Get window parameters for prediction strategy."""
        return {
            "window_size": self.window_size,
            "stride": self.stride,
            "zero_pad": self.zero_pad,
        }


class RandomizedWindowStrategy(WindowStrategy):
    """
    Window strategy with randomized window sizes.

    This strategy wraps another strategy and adds randomization
    to window sizes during training for improved robustness.

    Parameters
    ----------
    base_strategy : WindowStrategy
        Base strategy to wrap with randomization.
    randomize_training : bool, optional
        Whether to randomize during training. Default is True.
    """

    def __init__(
        self,
        base_strategy: WindowStrategy,
        randomize_training: bool = True,
    ):
        self.base_strategy = base_strategy
        self.randomize_training = randomize_training

        self.validate_parameters()

    def validate_parameters(self, **kwargs: Any) -> None:
        """Validate randomized window parameters."""
        # Delegate validation to base strategy
        self.base_strategy.validate_parameters(**kwargs)

    def create_sample_generators(
        self,
        input_data: list[pd.DataFrame],
        target_data: list[pd.DataFrame] | list[None],
        **kwargs: Any,
    ) -> list[Any]:
        """Create randomized sample generators."""
        # Add randomization parameters to kwargs if training
        if self.randomize_training and not kwargs.get("predict"):
            kwargs["randomize_seq_len"] = True

        return self.base_strategy.create_sample_generators(
            input_data, target_data, **kwargs
        )

    def get_window_parameters(self) -> dict[str, Any]:
        """Get window parameters including randomization."""
        params = self.base_strategy.get_window_parameters()
        params["randomize_training"] = self.randomize_training
        return params


class WindowStrategyFactory:
    """
    Factory for creating window strategies.

    This factory provides a centralized way to create different
    window strategies based on configuration parameters.
    """

    @staticmethod
    def create_timeseries_strategy(
        seq_len: int,
        stride: int = 1,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        predict: bool = False,
    ) -> TimeSeriesWindowStrategy:
        """
        Create a time series window strategy.

        Parameters
        ----------
        seq_len : int
            Length of each sequence window.
        stride : int, optional
            Step size for sliding window. Default is 1.
        min_seq_len : int | None, optional
            Minimum sequence length for randomization. Default is None.
        randomize_seq_len : bool, optional
            Whether to randomize sequence lengths. Default is False.
        predict : bool, optional
            Whether windowing is for prediction. Default is False.

        Returns
        -------
        TimeSeriesWindowStrategy
            Configured time series window strategy.
        """
        return TimeSeriesWindowStrategy(
            seq_len=seq_len,
            stride=stride,
            min_seq_len=min_seq_len,
            randomize_seq_len=randomize_seq_len,
            predict=predict,
        )

    @staticmethod
    def create_transformer_strategy(
        ctx_seq_len: int,
        tgt_seq_len: int,
        stride: int = 1,
        min_ctx_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        predict: bool = False,
    ) -> TransformerWindowStrategy:
        """
        Create a transformer window strategy.

        Parameters
        ----------
        ctx_seq_len : int
            Context (encoder) sequence length.
        tgt_seq_len : int
            Target (decoder) sequence length.
        stride : int, optional
            Step size for sliding window. Default is 1.
        min_ctx_seq_len : int | None, optional
            Minimum context sequence length. Default is None.
        min_tgt_seq_len : int | None, optional
            Minimum target sequence length. Default is None.
        randomize_seq_len : bool, optional
            Whether to randomize sequence lengths. Default is False.
        predict : bool, optional
            Whether windowing is for prediction. Default is False.

        Returns
        -------
        TransformerWindowStrategy
            Configured transformer window strategy.
        """
        return TransformerWindowStrategy(
            ctx_seq_len=ctx_seq_len,
            tgt_seq_len=tgt_seq_len,
            stride=stride,
            min_ctx_seq_len=min_ctx_seq_len,
            min_tgt_seq_len=min_tgt_seq_len,
            randomize_seq_len=randomize_seq_len,
            predict=predict,
        )

    @staticmethod
    def create_predict_strategy(
        window_size: int,
        stride: int = 1,
        zero_pad: bool = True,
    ) -> PredictWindowStrategy:
        """
        Create a prediction window strategy.

        Parameters
        ----------
        window_size : int
            Size of prediction windows.
        stride : int, optional
            Step size for sliding window. Default is 1.
        zero_pad : bool, optional
            Whether to zero-pad incomplete windows. Default is True.

        Returns
        -------
        PredictWindowStrategy
            Configured prediction window strategy.
        """
        return PredictWindowStrategy(
            window_size=window_size,
            stride=stride,
            zero_pad=zero_pad,
        )

    @staticmethod
    def create_randomized_strategy(
        base_strategy: WindowStrategy,
        randomize_training: bool = True,
    ) -> RandomizedWindowStrategy:
        """
        Create a randomized window strategy.

        Parameters
        ----------
        base_strategy : WindowStrategy
            Base strategy to wrap with randomization.
        randomize_training : bool, optional
            Whether to randomize during training. Default is True.

        Returns
        -------
        RandomizedWindowStrategy
            Configured randomized window strategy.
        """
        return RandomizedWindowStrategy(
            base_strategy=base_strategy,
            randomize_training=randomize_training,
        )
