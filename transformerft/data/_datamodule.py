from __future__ import annotations

import logging
import os.path as path
import tempfile
import typing
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from ..config import BaseConfig

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="DataModuleBase")


TIME = "time_ms"


class DataModuleBase(L.LightningDataModule):
    """
    Abstract base class for all data modules.

    Don't forget to call :meth:`save_hyperparameters` in your
    subclass constructor.
    """

    def __init__(
        self,
        train_dataset: str | typing.Sequence[str] | None = None,
        val_dataset: str | typing.Sequence[str] | None = None,
        test_dataset: str | None = None,
        predict_dataset: str | None = None,
    ):
        self._train_data_pth = train_dataset
        self._val_data_pth = val_dataset
        self._test_data_pth = test_dataset
        self._predict_data_pth = predict_dataset

        self._train_df: list[pd.DataFrame] | None = None
        self._val_df: list[pd.DataFrame] | None = None
        self._test_df: list[pd.DataFrame] | None = None
        self._predict_df: list[pd.DataFrame] | None = None

        self._tmp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def from_config(
        cls: typing.Type[SameType], config: BaseConfig, **kwargs: typing.Any
    ) -> SameType:
        raise NotImplementedError

    def prepare_data(self, save: bool = True) -> None:
        """
        Loads and preprocesses data dataframes.

        The dataframes are purposely *not* concatenated to keep
        distinct data from different sources separate.
        The data will be concatenated in :class:`TimeSeriesDataset`
        after data has been split and samples created using the sliding window technique.

        Parameters
        ----------
        save : bool
            Whether to save the dataframes to parquet files.

        """
        self._train_df = self._load_and_preprocess_data(
            self._train_data_pth or []
        )
        self._val_df = self._load_and_preprocess_data(self._val_data_pth or [])
        self._test_df = self._load_and_preprocess_data(
            [self._test_data_pth] if self._test_data_pth is not None else []
        )
        self._predict_df = self._load_and_preprocess_data(
            [self._predict_data_pth]
            if self._predict_data_pth is not None
            else []
        )

        if save:
            self.save_data(self._tmp_dir.name)

    def setup(
        self,
        stage: typing.Literal["fit", "train", "val", "test", "predict"]
        | None = None,
    ) -> None:
        """
        Sets up the data for training, validation or testing.

        Parameters
        ----------
        stage : typing.Literal["fit", "train", "val", "test", "predict"] | None
            The stage to setup for. If None, all stages are setup.
        """

        def load_parquet(
            name: typing.Literal["train", "val", "test", "predict"],
            dir_: str | None = None,
        ) -> list[pd.DataFrame]:
            """Convenience function to load data from parquet files."""
            paths = self._tmp_data_paths(name, dir_)
            dfs = []
            for pth in paths:
                df = pd.read_parquet(pth)
                dfs.append(df)
            return dfs

        if stage in ("train", "fit"):
            self._train_df = load_parquet("train", self._tmp_dir.name)
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "val":
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "test":
            self._test_df = load_parquet("test", self._tmp_dir.name)
        elif stage == "predict":
            self._predict_df = load_parquet("predict", self._tmp_dir.name)
        else:
            raise ValueError(f"Unknown stage {stage}.")

    def transform_input(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: typing.Sequence[str] | None = None,
        target_columns: typing.Sequence[str] | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Transforms the input data into a dataframe with the specified columns.

        If the inputs and targets are numpy arrays, the columns will be named
        ``input_0``, ``input_1``, ..., ``input_n`` and ``target_0``,
        ``target_1``, ..., ``target_m``, if ``input_columns`` and
        ``

        Parameters
        ----------
        input_ : np.ndarray | pd.Series | pd.DataFrame
            The input data.
        target : np.ndarray | pd.Series | pd.DataFrame | None
            The target data.
        input_columns : typing.Sequence[str] | None
            The columns to use from the input data, or which columns to create
        target_columns : typing.Sequence[str] | None
            The columns to use from the target data, or which columns to create
        timestamp : np.ndarray | pd.Series | pd.DataFrame | None
            The timestamps of the data.
        """
        # convert inputs to numpy
        if isinstance(input_, pd.Series):
            input_ = input_.to_numpy()
        if target is not None and isinstance(target, pd.Series):
            target = target.to_numpy()

        n_input_cols = input_.shape[1]
        n_target_cols = target.shape[1] if target is not None else 0

        # generate column names if necessary
        generated_input_cols = False
        if input_columns is None:
            input_columns = [f"input_{i}" for i in range(n_input_cols)]
            generated_input_cols = True
        elif len(input_columns) != n_input_cols:
            raise ValueError(
                f"Expected {n_input_cols} input columns, got {len(input_columns)}."
            )
        else:
            input_columns = list(input_columns)

        generated_target_cols = False
        if target_columns is None:
            target_columns = [f"target_{i}" for i in range(n_target_cols)]
            generated_target_cols = True
        elif len(target_columns) != n_target_cols:
            raise ValueError(
                f"Expected {n_target_cols} target columns, got {len(target_columns)}."
            )
        else:
            target_columns = list(target_columns)

        # convert inputs to dataframe
        if isinstance(input_, np.ndarray):
            df_dict: dict[str, np.ndarray | pd.Series] = {
                input_columns[i]: input_[:, i] for i in range(n_input_cols)
            }
            if target is not None:
                if isinstance(target, np.ndarray):
                    for i in range(n_target_cols):
                        df_dict[target_columns[i]] = target[:, i]
                elif isinstance(target, pd.DataFrame):
                    if generated_target_cols:
                        raise ValueError(
                            "Target columns were generated, "
                            "but target is a dataframe. "
                            "Please specify target columns."
                        )
                    for i, col in enumerate(target_columns):
                        df_dict[col] = target.iloc[:, i]
                else:
                    raise TypeError(
                        "Expected np.ndarray, pd.Series or pd.DataFrame, "
                        f"got {type(target)}."
                    )

            if timestamp is not None:
                if isinstance(timestamp, (np.ndarray, pd.Series)):
                    df_dict[TIME] = timestamp
                elif isinstance(timestamp, (int, float)):
                    time = np.arange(len(input_)) + timestamp
                    df_dict[TIME] = time
                elif isinstance(timestamp, pd.DataFrame):
                    df_dict[TIME] = timestamp.to_numpy()
                else:
                    time = np.arange(len(input_)) + 0.0  # for type checking
                    df_dict[TIME] = time

            df = pd.DataFrame.from_dict(df_dict)
        elif isinstance(input_, pd.DataFrame):
            if (
                generated_input_cols and generated_target_cols
            ):  # no columns supplied
                df = input_.dropna(how="all", axis="columns")
            elif generated_input_cols:  # no input columns supplied
                df = input_.dropna(how="all", axis="columns")
                df = df[target_columns]
            elif generated_target_cols:  # no target columns supplied
                df = input_.dropna(how="all", axis="columns")
                df = df[input_columns]
            else:  # both input and target columns supplied
                df = input_.dropna(how="all", axis="columns")
                df = df[input_columns + target_columns]
        else:
            raise TypeError(
                f"Expected np.ndarray or pd.DataFrame, got {type(input_)}"
            )

        return df

    def make_dataset(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: typing.Sequence[str] | None = None,
        target_columns: typing.Sequence[str] | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        predict: bool = False,
    ) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def make_dataloader(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: typing.Sequence[str] | None = None,
        target_columns: typing.Sequence[str] | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        predict: bool = False,
        **kwargs: typing.Any,
    ) -> torch.utils.data.DataLoader:
        raise NotImplementedError

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @property
    def val_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @property
    def test_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    @property
    def predict_dataset(self) -> torch.utils.data.Dataset:
        raise NotImplementedError

    def save_data(self, save_dir: typing.Optional[str] = None) -> None:
        """
        Saves the data to the model directory.
        """
        if save_dir is None:
            save_dir = self.hparams["model_dir"]

        def save_parquet(
            dfs: typing.Optional[list[pd.DataFrame]],
            name: typing.Literal["train", "val", "test", "predict"],
        ) -> None:
            if dfs is None:
                return

            paths = self._tmp_data_paths(name, save_dir)
            for i, df in enumerate(dfs):
                if len(df) == 0:
                    continue
                dfs[i].reset_index(drop=True).to_parquet(paths[i])

        save_parquet(self._train_df, "train")
        save_parquet(self._val_df, "val")
        save_parquet(self._test_df, "test")
        save_parquet(self._predict_df, "predict")

    def _tmp_data_paths(
        self,
        name: typing.Literal["train", "val", "test", "predict"],
        dir_: str | None = None,
    ) -> list[Path]:
        """
        Returns a list of paths to the data files.
        If the data is not saved yet, the paths are generated,
        otherwise the paths are searched for in the model directory.

        :param name: Name of the data set.
        :param dir_: Directory to look for the data files in.
        :return: List of paths to the data files.
        """
        if dir_ is None:
            dir_ = typing.cast(str, self.hparams["model_dir"])
        dfs = getattr(self, f"_{name}_df")

        if dfs is not None:  # for saving
            if len(dfs) == 1 or isinstance(dfs, pd.DataFrame):
                return [Path(dir_) / f"{name}.parquet"]
            else:
                return [
                    Path(dir_) / f"{name}_{i}.parquet" for i in range(len(dfs))
                ]
        else:  # for loading
            # try to find the data files
            single_df_path = Path(dir_) / f"{name}.parquet"
            if single_df_path.exists():
                return [single_df_path]
            else:
                paths = []
                multi_file_stem = path.join(
                    self.hparams["model_dir"], f"{name}_{{}}.parquet"
                )
                # find all files with the same stem
                i = 0
                while True:
                    p = Path(multi_file_stem.format(i))
                    if p.exists():
                        paths.append(p)
                    else:
                        break

                return paths

    def _load_and_preprocess_data(
        self, df_pths: str | typing.Sequence[str]
    ) -> list[pd.DataFrame]:
        """
        Loads and preprocesses data dataframes.

        The dataframes are purposely *not* concatenated to keep
        distinct data from different sources separate.
        The data will be concatenated in :class:`HysteresisDataset`
        after data has been split and samples created using the sliding window technique.

        :return: List of processed dataframes.
        """

        def load_and_preprocess(pth: str) -> pd.DataFrame:
            log.info(f"Reading data from {pth}.")
            df = pd.read_parquet(pth)

            return self.transform_input(df)

        if isinstance(df_pths, str):
            return [load_and_preprocess(df_pths)]
        else:
            return [load_and_preprocess(pth) for pth in df_pths]
