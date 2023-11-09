from __future__ import annotations

import logging
import typing

import pandas as pd

from ...utils import signal
from ...data import TimeSeriesDataModule

from ._config import PhyLSTMConfig

__all__ = [
    "PhyLSTMDataModule",
]

pd.options.mode.chained_assignment = None  # type: ignore

if typing.TYPE_CHECKING:
    STAGES = typing.Literal["train", "val", "test", "predict"]
    EVAL_TYPES = typing.Literal["val", "test", "predict"]
    SameType = typing.TypeVar("SameType", bound="PhyLSTMDataModule")

CURRENT = "I_meas_A"
FIELD = "B_meas_T"

log = logging.getLogger(__name__)


class PhyLSTMDataModule(TimeSeriesDataModule):
    """
    A convenience class created to load and do low level preprocessing of the data.

    The class can load multiple files and preprocess them independently.
    The files are not merged until after individual samples are created in the
    :class:`TimeSeriesDataset` class

    The preprocessing steps are:
    - Downsample the data. The downsampling can use different factors per input file.
    - Remove linear component of the field

    The processing then partitions data into training, validation and test sets (optional).

    The training, validation and test sets can then be retrieved using the respective properties
    :attr:`train_data`, :attr:`val_data` and :attr:`test_data`.
    """

    TRANSFORMS = ["polynomial", "normalize"]

    def __init__(
        self,
        train_df: pd.DataFrame | list[pd.DataFrame] | None,
        val_df: pd.DataFrame | list[pd.DataFrame] | None,
        seq_len: int = 500,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        downsample: int = 1,
        lowpass_filter: bool = False,
        mean_filter: bool = False,
        remove_polynomial: bool = True,
        polynomial_degree: int = 1,
        polynomial_iterations: int = 1000,
        input_columns: str = CURRENT,
        target_column: str = FIELD,
        batch_size: int = 128,
        num_workers: int = 0,
        model_dir: str | None = None,
        dtype: str = "float32",
    ):
        """
        The raw dataset used to train the hysteresis model.
        The class can load multiple files and preprocess them independently.

        Loads the data from the given path and preprocesses it.
        Retrieve the training, validation and test data using the
        respective properties.

        :param batch_size:
        :param num_workers:

        """
        super().__init__(
            train_df=train_df,
            val_df=val_df,
            input_columns=input_columns,
            target_column=target_column,
            normalize=True,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            randomize_seq_len=randomize_seq_len,
            stride=stride,
            remove_polynomial=remove_polynomial,
            polynomial_degree=polynomial_degree,
            polynomial_iterations=polynomial_iterations,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        self.save_hyperparameters(ignore=["train_df", "val_df"])

    @classmethod
    def parse_config_kwargs(  # type: ignore[override]
        cls,
        config: PhyLSTMConfig,
        **kwargs: typing.Any,
    ) -> dict[str, typing.Any]:
        kwargs = super().parse_config_kwargs(config, **kwargs)
        default_kwargs = {
            "lowpass_filter": config.lowpass_filter,
            "mean_filter": config.mean_filter,
        }
        default_kwargs.update(kwargs)

        for key in ("normalize",):
            if key in default_kwargs:
                del default_kwargs[key]

        return default_kwargs

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe into the format expected by the model.
        This function should be chaininvoked with the ``read_input`` function.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to preprocess.

        Returns
        -------
        pd.DataFrame
            The preprocessed dataframe.
        """
        df = super().preprocess_dataframe(df)

        current: str = self.hparams["input_columns"][0]
        field: str | None = self.hparams["target_column"]

        # signal processing: lowpass filter
        if self.hparams["lowpass_filter"]:
            df[current] = signal.butter_lowpass_filter(
                df[current].to_numpy(), cutoff=32, fs=1e3, order=10
            )

            if field is not None and field in df:
                df[field] = signal.butter_lowpass_filter(
                    df[field].to_numpy(), cutoff=32, fs=1e3, order=10
                )

        # signal processing: mean filter to remove low amplitude fluctuations
        if self.hparams["mean_filter"]:
            df[current] = signal.mean_filter(
                df[current].to_numpy(),
                window_size=100,
                stride=1,
                threshold=35e-3,
            )

            if field is not None and field in df:
                df[field] = signal.mean_filter(
                    df[field].to_numpy(),
                    window_size=100,
                    stride=1,
                    threshold=6e-6,
                )

        return df
