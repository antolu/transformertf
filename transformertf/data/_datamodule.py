from __future__ import annotations

import logging
import os.path as path
import tempfile
import typing
from pathlib import Path
import functools

import lightning as L
import numpy as np
import pandas as pd
import sklearn.exceptions
import sklearn.utils.validation
import torch
import torch.utils.data

from ..config import BaseConfig
from ..modules import RunningNormalizer
from ._dataset import TimeSeriesDataset
from ._transform import PolynomialTransform

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="DataModuleBase")


TIME = "time_ms"


class DataModuleBase(L.LightningDataModule):
    _train_data_pth: list[str]
    _val_data_pth: list[str]

    _input_normalizer: RunningNormalizer | None
    _target_normalizer: RunningNormalizer | None

    """
    Abstract base class for all data modules.

    Don't forget to call :meth:`save_hyperparameters` in your
    subclass constructor.
    """

    def __init__(
        self,
        input_columns: typing.Sequence[str],
        target_columns: typing.Sequence[str],
        train_dataset: str | typing.Sequence[str] | None = None,
        val_dataset: str | typing.Sequence[str] | None = None,
        normalize: bool = True,
        seq_len: int = 500,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        out_seq_len: int = 0,
        stride: int = 1,
        remove_polynomial: bool = False,
        polynomial_degree: int = 2,
        polynomial_iterations: int = 1000,
        batch_size: int = 128,
        num_workers: int = 0,
    ):
        super().__init__()
        min_seq_len = min_seq_len or seq_len
        self.save_hyperparameters()

        for attr_name, arg in (
            ("_train_data_pth", train_dataset),
            ("_val_data_pth", val_dataset),
        ):
            if isinstance(arg, str):
                setattr(self, attr_name, [arg])
            elif isinstance(arg, (list, tuple)):
                setattr(self, attr_name, arg)
            elif arg is None:
                setattr(self, attr_name, [])
            else:
                raise TypeError(
                    f"Expected str, list or tuple, got {type(arg)}."
                )

        self._train_df: list[pd.DataFrame] = []
        self._val_df: list[pd.DataFrame] = []

        if normalize:
            if input_columns is None or target_columns is None:
                raise ValueError(
                    "input_columns and target_columns must "
                    "be set if normalize=True"
                )
            self._input_normalizer = RunningNormalizer(
                num_features=len(input_columns)
            )
            self._target_normalizer = RunningNormalizer(
                num_features=len(target_columns)
            )
        else:
            self._input_normalizer = None
            self._target_normalizer = None

        self._polynomial_transform: dict[str, PolynomialTransform] | None
        if remove_polynomial:
            self._polynomial_transform = {
                col: PolynomialTransform(
                    degree=polynomial_degree,
                    num_iterations=polynomial_iterations,
                )
                for col in target_columns
            }
        else:
            self._polynomial_transform = None

        self._tmp_dir = tempfile.TemporaryDirectory()

    @classmethod
    def from_config(
        cls: typing.Type[SameType], config: BaseConfig, **kwargs: typing.Any
    ) -> SameType:
        raise NotImplementedError

    def state_dict(self) -> dict[str, typing.Any]:
        state = super().state_dict()
        if self._input_normalizer is not None:
            state["input_normalizer"] = self._input_normalizer.state_dict()
        if self._target_normalizer is not None:
            state["target_normalizer"] = self._target_normalizer.state_dict()

        if self._polynomial_transform is not None:
            state["polynomial_transform"] = {
                col: transform.state_dict()
                for col, transform in self._polynomial_transform.items()
            }

        return state

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        if "input_normalizer" in state:
            if self._input_normalizer is None:
                raise RuntimeError("input_normalizer is None")

            self._input_normalizer.load_state_dict(
                state.pop("input_normalizer")
            )
        if "target_normalizer" in state:
            if self._target_normalizer is None:
                raise RuntimeError("target_normalizer is None")

            self._target_normalizer.load_state_dict(
                state.pop("target_normalizer")
            )

        if "polynomial_transform" in state:
            if self._polynomial_transform is None:
                raise RuntimeError("polynomial_transform is None")

            for col, transform in self._polynomial_transform.items():
                transform.load_state_dict(state["polynomial_transform"][col])
            state.pop("polynomial_transform")

        super().load_state_dict(state)

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
        # load all data into memory and then apply transforms
        self._train_df = list(
            map(
                self.preprocess_dataframe,
                map(
                    self._load_and_preprocess_data, self._train_data_pth or []
                ),
            )
        )
        self._val_df = list(
            map(
                self.preprocess_dataframe,
                map(self._load_and_preprocess_data, self._val_data_pth or []),
            )
        )

        self._try_fit_polynomial_transform(pd.concat(self._train_df))

        try:
            self._try_scalers_fitted()
            raise RuntimeError("Scalers have already been fitted.")
        except sklearn.exceptions.NotFittedError:
            if self._polynomial_transform is not None:
                self._fit_scalers(
                    self._try_transform_polynomial(pd.concat(self._train_df))
                )
            else:
                self._fit_scalers(pd.concat(self._train_df))

        self._train_df = list(map(self.apply_transforms, self._train_df))
        self._val_df = list(map(self.apply_transforms, self._val_df))

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

        if stage is None:
            self._train_df = load_parquet("train", self._tmp_dir.name)
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage in ("train", "fit"):
            self._train_df = load_parquet("train", self._tmp_dir.name)
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "val":
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "test":
            raise NotImplementedError(
                "Datamodule does not support using the test set.\n"
                "Use the 'make_dataset' or 'make_dataloader' methods instead."
            )
        elif stage == "predict":
            raise NotImplementedError(
                "Datamodule does not support using the predict set.\n"
                "Use the 'make_dataset' or 'make_dataloader' methods instead."
            )
        else:
            raise ValueError(f"Unknown stage {stage}.")

    def read_input(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: str | typing.Sequence[str] | None = None,
        target_columns: str | typing.Sequence[str] | None = None,
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
        timestamp : np.ndarray | pd.Series | pd.DataFrame | None
            The timestamps of the data.
        """
        # convert inputs to numpy
        if isinstance(input_, pd.Series):
            input_ = input_.to_numpy()
        if target is not None and isinstance(target, pd.Series):
            target = target.to_numpy()

        # generate column names if necessary
        if input_columns is None:
            input_columns = self.hparams["input_columns"]
        elif isinstance(input_columns, str):
            input_columns = [input_columns]

        if target is not None or isinstance(input_, pd.DataFrame):
            if target_columns is None:
                target_columns = self.hparams["target_columns"]
            elif isinstance(target_columns, str):
                target_columns = [target_columns]

        # convert inputs to dataframe
        if isinstance(input_, np.ndarray):
            df_dict: dict[str, np.ndarray | pd.Series] = {
                input_columns[i]: input_[:, i]
                for i in range(len(input_columns))
            }
            if target is not None and target_columns is not None:
                if isinstance(target, np.ndarray):
                    for i in range(len(target_columns)):
                        df_dict[target_columns[i]] = target[:, i]
                elif isinstance(target, pd.DataFrame):
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
            df = input_.dropna(how="all", axis="columns")

            if target_columns is not None:
                col_filter = list(input_columns) + [
                    col for col in target_columns if col in df.columns
                ]
                df = df[col_filter]
        else:
            raise TypeError(
                f"Expected np.ndarray or pd.DataFrame, got {type(input_)}"
            )

        return df

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe into the format expected by the model.
        This function should be chaininvoked with the ``read_input`` function,
        and must be called before ``apply_transforms``.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to preprocess.

        Returns
        -------
        pd.DataFrame
            The preprocessed dataframe.
        """
        return df

    def apply_transforms(
        self,
        df: pd.DataFrame,
        skip_target: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize the dataframe into the format expected by the model.
        This function should be chaininvoked with the ``read_input`` function.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to normalize.
        skip_target : bool
            Whether to skip normalizing the target columns.

        Returns
        -------
        pd.DataFrame
            The normalized dataframe.

        Raises
        ------
        sklearn.exceptions.NotFittedError
            If the scaler has not been fitted.
        """
        if self.hparams["remove_polynomial"]:
            df = self._try_transform_polynomial(df)
        if self.hparams["normalize"]:
            df = self._try_scale_df(df, skip_target=skip_target)
        return df

    def transform_input(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        input_columns: str | typing.Sequence[str] | None = None,
        target_columns: str | typing.Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """
        Chains the ``read_input`` and ``preprocess_dataframe``, and
        ``normalize_dataframe`` functions together.

        Parameters
        ----------
        input_ : np.ndarray | pd.Series | pd.DataFrame
            The input data.
        target : np.ndarray | pd.Series | pd.DataFrame | None
            The target data.
        timestamp : np.ndarray | pd.Series | pd.DataFrame | None
            The timestamps of the data.
        input_columns : str | typing.Sequence[str] | None
            The names of the input columns.
        target_columns : str | typing.Sequence[str] | None
            The names of the target columns.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.

        Raises
        ------
        TypeError
            If the input is not a ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame``.
        sklearn.exceptions.NotFittedError
            If the normalizers are not yet fitted.
            This is caused by calling ``transform_input`` before ``prepare_data``,
            or using a datamodule that has previously not been trained on.
        """
        df = self.read_input(
            input_,
            target,
            timestamp=timestamp,
            input_columns=input_columns,
            target_columns=target_columns,
        )
        df = self.preprocess_dataframe(df)
        skip_target = not all(
            [col in df.columns for col in self.hparams["target_columns"]]
        )
        df = self.apply_transforms(df, skip_target=skip_target)
        return df

    def make_dataset(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        predict: bool = False,
    ) -> TimeSeriesDataset:
        df = self.transform_input(
            input_=input_,
            target=target,
            timestamp=timestamp,
        )

        return self._make_dataset_from_df(df, predict=predict)

    def make_dataloader(
        self,
        input_: np.ndarray | pd.Series | pd.DataFrame,
        target: np.ndarray | pd.Series | pd.DataFrame | None = None,
        timestamp: np.ndarray | pd.Series | pd.DataFrame | None = None,
        predict: bool = False,
        **kwargs: typing.Any,
    ) -> torch.utils.data.DataLoader:
        dataset = self.make_dataset(
            input_,
            target,
            timestamp=timestamp,
            predict=predict,
        )

        default_kwargs = {
            "batch_size": self.hparams["batch_size"] if not predict else 1,
            "num_workers": self.hparams["num_workers"],
            "shuffle": not predict,
        }
        default_kwargs.update(kwargs)

        return torch.utils.data.DataLoader(
            dataset,
            **default_kwargs,
        )

    @property
    def train_dataset(self) -> TimeSeriesDataset:
        if self._train_df is None or len(self._train_df) == 0:
            raise ValueError("No training data available.")

        input_data = [
            df[self.hparams["input_columns"]].to_numpy()
            for df in self._train_df
        ]
        target_data = [
            df[self.hparams["target_columns"]].to_numpy()
            for df in self._train_df
        ]

        return TimeSeriesDataset(
            input_data=input_data,
            seq_len=self.hparams["seq_len"],
            target_data=target_data,
            stride=self.hparams["stride"],
            predict=False,
            input_normalizer=self._input_normalizer,
            target_normalizer=self._target_normalizer,
        )

    @property
    def val_dataset(
        self,
    ) -> TimeSeriesDataset | typing.Sequence[TimeSeriesDataset]:
        if self._val_df is None or len(self._val_df) == 0:
            raise ValueError("No validation data available.")

        datasets = [
            self._make_dataset_from_df(df, predict=True) for df in self._val_df
        ]

        return datasets[0] if len(datasets) == 1 else datasets

    def _make_dataset_from_df(
        self, df: pd.DataFrame, predict: bool = False
    ) -> TimeSeriesDataset:
        if all([col in df.columns for col in self.hparams["target_columns"]]):
            target_data = df[self.hparams["target_columns"]].to_numpy()
        else:
            target_data = None

        return TimeSeriesDataset(
            input_data=df[self.hparams["input_columns"]].to_numpy(),
            seq_len=self.hparams["seq_len"],
            target_data=target_data,
            stride=self.hparams["stride"],
            predict=predict,
            min_seq_len=self.hparams["min_seq_len"] if not predict else None,
            randomize_seq_len=self.hparams["randomize_seq_len"]
            if not predict
            else False,
            input_normalizer=self._input_normalizer,
            target_normalizer=self._target_normalizer,
        )

    def train_dataloader(
        self,
    ) -> (
        torch.utils.data.DataLoader
        | typing.Sequence[torch.utils.data.DataLoader]
    ):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=True,
            num_workers=self.hparams["num_workers"],
        )

    def val_dataloader(
        self,
    ) -> (
        torch.utils.data.DataLoader
        | typing.Sequence[torch.utils.data.DataLoader]
    ):
        make_dataloader = functools.partial(
            torch.utils.data.DataLoader,
            batch_size=1,
            num_workers=self.hparams["num_workers"],
            shuffle=False,
        )
        if isinstance(self.val_dataset, TimeSeriesDataset):
            return make_dataloader(self.val_dataset)
        else:
            return [make_dataloader(ds) for ds in self.val_dataset]

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
                multi_file_stem = path.join(dir_, f"{name}_{{}}.parquet")
                # find all files with the same stem
                i = 0
                while True:
                    p = Path(multi_file_stem.format(i))
                    if p.exists():
                        paths.append(p)
                    else:
                        break

                return paths

    def _load_and_preprocess_data(self, pth: str) -> pd.DataFrame:
        """
        Loads and preprocesses data dataframes.

        The dataframes are purposely *not* concatenated to keep
        distinct data from different sources separate.
        The data will be concatenated in :class:`HysteresisDataset`
        after data has been split and samples created using the sliding window technique.

        :return: List of processed dataframes.
        """

        log.info(f"Reading data from {pth}.")
        df = pd.read_parquet(pth)

        return self.read_input(df)

    def _try_scalers_fitted(self) -> None:
        if (
            self._input_normalizer is not None
            and self._target_normalizer is not None
        ):
            try:
                sklearn.utils.validation.check_is_fitted(
                    self._input_normalizer
                )
                sklearn.utils.validation.check_is_fitted(
                    self._target_normalizer
                )
            except sklearn.exceptions.NotFittedError:
                raise sklearn.exceptions.NotFittedError(
                    "Scalers are not fitted yet. "
                    "You should call prepare_data() first."
                )
        else:
            log.debug("No scalers to check.")

    def _fit_scalers(self, df: pd.DataFrame) -> None:
        if self._input_normalizer is None or self._target_normalizer is None:
            log.debug("No scalers found, skipping fitting.")
            return

        self._input_normalizer.fit(
            torch.from_numpy(df[self.hparams["input_columns"]].to_numpy())
        )
        self._target_normalizer.fit(
            torch.from_numpy(df[self.hparams["target_columns"]].to_numpy())
        )

    def _try_scale_df(
        self, df: pd.DataFrame, skip_target: bool = False
    ) -> pd.DataFrame:
        self._try_scalers_fitted()

        if self._input_normalizer is None or (
            not skip_target and self._target_normalizer is None
        ):
            log.debug("No scalers found, skipping scaling.")
            return df

        input_col_names = self.hparams["input_columns"]

        out = pd.DataFrame()
        out[input_col_names] = self._input_normalizer.transform(
            torch.from_numpy(df[input_col_names].to_numpy())
        )
        if not skip_target:
            assert self._target_normalizer is not None
            target_col_names = self.hparams["target_columns"]
            out[target_col_names] = self._target_normalizer.transform(
                torch.from_numpy(df[target_col_names].to_numpy())
            )

        return out

    def _try_fit_polynomial_transform(self, df: pd.DataFrame) -> None:
        if self._polynomial_transform is not None:
            try:
                for transform in self._polynomial_transform.values():
                    sklearn.utils.validation.check_is_fitted(transform)
                log.info("Polynomial transform already fitted.")
            except sklearn.exceptions.NotFittedError:
                log.info("Polynomial transform not fitted yet.")

                if len(self.hparams["input_columns"]) > 1:
                    raise ValueError(
                        "Polynomial transform can only be used with a single input column."
                    )

                base_columns = [
                    o
                    for o in self.hparams["target_columns"]
                    if not o.endswith("_dot")
                ]
                deriv_columns = [
                    o
                    for o in self.hparams["target_columns"]
                    if o.endswith("_dot")
                ]

                for col in base_columns:
                    log.info(f"Fitting polynomial transform for {col}.")
                    self._polynomial_transform[col].fit(
                        df[self.hparams["input_columns"][0]].to_numpy(),
                        df[col].to_numpy(),
                    )

                for col in base_columns:
                    if col + "_dot" in deriv_columns:
                        log.info(
                            f"Fitting polynomial transform for {col}_dot."
                        )
                        self._polynomial_transform[
                            col + "_dot"
                        ] = self._polynomial_transform[col].get_derivative()

    def _try_transform_polynomial(self, df: pd.DataFrame) -> pd.DataFrame:
        self._try_fit_polynomial_transform(df)

        if self._polynomial_transform is None:
            log.debug("No polynomial transform found, skipping transform.")
            return df

        input_col_names = self.hparams["input_columns"]

        if len(input_col_names) > 1:
            raise ValueError(
                "Polynomial transform can only be used with a single input column."
            )

        out = pd.DataFrame()
        for col in self.hparams["target_columns"]:
            out[col] = (
                self._polynomial_transform[col]
                .transform(
                    df[input_col_names[0]].to_numpy(),
                    df[col].to_numpy(),
                )
                .numpy()
            )

        return out
