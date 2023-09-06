from __future__ import annotations

import logging
import typing

import numpy as np
import pandas as pd

from transformertf.utils import signal

from ...data import DataModuleBase
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


class PhyLSTMDataModule(DataModuleBase):
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

    def __init__(
        self,
        train_dataset: str | typing.Sequence[str] | None = None,
        val_dataset: str | typing.Sequence[str] | None = None,
        seq_len: int = 500,
        min_seq_len: int | None = None,
        out_seq_len: int = 0,
        randomize_seq_len: bool = False,
        stride: int = 1,
        lowpass_filter: bool = False,
        mean_filter: bool = False,
        downsample: int = 1,
        remove_polynomial: bool = True,
        polynomial_degree: int = 1,
        polynomial_iterations: int = 1000,
        batch_size: int = 128,
        num_workers: int = 0,
        current_column: str = CURRENT,
        field_column: str = FIELD,
        model_dir: str | None = None,
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
            input_columns=[current_column],
            target_columns=[field_column],
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            normalize=True,
            seq_len=seq_len,
            min_seq_len=min_seq_len,
            out_seq_len=out_seq_len,
            randomize_seq_len=randomize_seq_len,
            stride=stride,
            remove_polynomial=remove_polynomial,
            polynomial_degree=polynomial_degree,
            polynomial_iterations=polynomial_iterations,
        )
        self.save_hyperparameters(ignore=["current_column", "field_column"])

        self._check_args()

    @classmethod
    def from_config(  # type: ignore[override]
        cls,
        config: PhyLSTMConfig,
        **kwargs: typing.Any,
    ) -> PhyLSTMDataModule:
        default_kwargs = {
            "train_dataset": config.train_dataset,
            "val_dataset": config.val_dataset,
            "seq_len": config.seq_len,
            "min_seq_len": config.min_seq_len,
            "randomize_seq_len": config.randomize_seq_len,
            "stride": config.stride,
            "downsample": config.downsample,
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "model_dir": config.model_dir,
            "lowpass_filter": config.lowpass_filter,
            "mean_filter": config.mean_filter,
            "remove_polynomial": config.remove_polynomial,
            "polynomial_degree": config.polynomial_degree,
            "polynomial_iterations": config.polynomial_iterations,
        }
        default_kwargs.update(kwargs)
        return PhyLSTMDataModule(
            **default_kwargs,  # type: ignore[arg-type]
        )

    def read_input(  # type: ignore[override]
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: typing.Sequence[str] | None = None,
        target_columns: typing.Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Transforms the input data into a dataframe with the specified columns.

        Parameters
        ----------
        input_ : np.ndarray | pd.Series | pd.DataFrame
            The input data.
        target : np.ndarray | pd.Series | pd.DataFrame | None
            The target data.
        timestamp : np.ndarray | pd.Series | pd.DataFrame | None
            The timestamps of the data.
        input_columns : typing.Sequence[str] | None
            The names of the input columns.
        target_columns : typing.Sequence[str] | None
            The names of the target columns.
        """

        df = super().read_input(
            input_=input_,
            target=target,
            timestamp=timestamp,
            input_columns=input_columns,
            target_columns=target_columns,
        )

        return df

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
        field: str | None = self.hparams["target_columns"][0]

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

        # downsample
        df = df.iloc[:: self.hparams["downsample"]].reset_index()

        return df

    def _check_args(self) -> None:
        if self.hparams["stride"] < 1:
            raise ValueError(
                f"Stride must be at least 1, but got {self.hparams['stride']}."
            )
