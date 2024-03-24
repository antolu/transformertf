from __future__ import annotations

import functools
import logging
import os.path as path
import tempfile
import typing
from pathlib import Path
import warnings

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from transformertf.data.dataset import TimeSeriesDataset
from .._downsample import downsample
from ..transform import (
    BaseTransform,
    StandardScaler,
    TransformCollection,
    TransformType,
)
from ..dataset import AbstractTimeSeriesDataset

if typing.TYPE_CHECKING:
    from ...config import BaseConfig

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="DataModuleBase")

TIME = "time_ms"


class DataModuleBase(L.LightningDataModule):
    """
    Abstract base class for all data modules, handles the bulk transformations
    of data, but does not construct the datasets

    Don't forget to call :meth:`save_hyperparameters` in your
    subclass constructor.
    """

    _input_transforms: dict[str, TransformCollection]
    _target_transform: TransformCollection

    def __init__(
        self,
        train_df: pd.DataFrame | list[pd.DataFrame] | None,
        val_df: pd.DataFrame | list[pd.DataFrame] | None,
        known_covariates_cols: str | typing.Sequence[str] | None = None,
        target_col: str | None = None,
        past_covariates_cols: str | typing.Sequence[str] | None = None,
        static_cont_covariates_cols: str | typing.Sequence[str] | None = None,
        static_cat_covariates_cols: str | typing.Sequence[str] | None = None,
        normalize: bool = True,
        downsample: int = 1,
        downsample_method: typing.Literal[
            "interval", "average", "convolve"
        ] = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        input_columns: str | typing.Sequence[str] | None = None,  # deprecated
        target_column: str | None = None,  # deprecated
    ):
        """ "
        Parameters
        ----------
        train_df : pd.DataFrame | list[pd.DataFrame] | None
            The training data. This may be None only when the DataModule
            is recreated from a checkpoint using the ``load_from_checkpoint`` method.
        val_df : pd.DataFrame | list[pd.DataFrame] | None
            The validation data. This may be None only when the DataModule
            is recreated from a checkpoint using the ``load_from_checkpoint`` method.
        known_covariates_cols : str | typing.Sequence[str]
            The names of the covariate columns, these should be known
            in the past and future.
        target_col : str
            The name of the target column. This is the column that
            the model will predict, and is not included in the covariate columns.
        past_covariates_cols : str | typing.Sequence[str] | None
            The names of the covariate columns that are only known in the past.
        static_cont_covariates_cols : str | typing.Sequence[str] | None
            The names of the continuous covariate columns that are static, i.e.
            they do not change over time. The values of these columns will be
            normalized using standard scaling if ``normalize`` is set to True.
        static_cat_covariates_cols : str | typing.Sequence[str] | None
            The names of the categorical covariate columns that are static, i.e.
            they do not change over time. The types of the columns do not matter,
            as they will be one-hot encoded.
        normalize : bool
            Normalize the data using standard scaling.
        downsample : int
            Downsample the data by this factor.
        downsample_method : typing.Literal["interval", "average", "convolve"]
            The method to use for downsampling.
        target_depends_on : str | None
            The name of the column that the target depends on, if a multivariate
            transform is used, supplied in the ``extra_transforms`` parameter.
        extra_transforms : dict[str, list[BaseTransform]] | None
            Extra transforms to apply to the data. The keys are the column names,
            and the values are lists of transforms to apply.
        batch_size : int
            The batch size to use for the dataloaders.
        num_workers : int
            The number of workers to use for the dataloaders. By default, this is 0.
        dtype : str
            The datatype to use for the data. By default, this is ``float32``.
        input_columns : str | typing.Sequence[str] | None
            The names of the input columns. This is deprecated and should not be used.
            Use ``known_covariate_cols`` instead.
        target_column : str | None
            The name of the target column. This is deprecated and should not be used.
            Use ``target_col`` instead.
        """
        super().__init__()
        self.save_hyperparameters(
            ignore=["train_df", "val_df"],
        )
        if input_columns is not None:
            known_covariates_cols = _to_list(input_columns)
            warnings.warn(
                "input_columns is deprecated, use known_covariate_cols instead."
            )
        elif known_covariates_cols is None:
            raise ValueError("known_covariate_columns must be set.")

        if target_column is not None:
            target_col = target_column
            warnings.warn(
                "target_column is deprecated, use target_col instead."
            )
        elif target_col is None:
            raise ValueError("target_col must be set.")

        known_covariates_cols = _to_list(known_covariates_cols)
        past_covariates_cols = _to_list(past_covariates_cols or [])
        static_cont_covariates_cols = _to_list(
            static_cont_covariates_cols or []
        )
        static_cat_covariates_cols = _to_list(static_cat_covariates_cols or [])

        # manually set the changed hyperparameters
        self.hparams["known_covariates_cols"] = known_covariates_cols
        self.hparams["past_covariates_cols"] = past_covariates_cols
        self.hparams["static_cont_covariates_cols"] = (
            static_cont_covariates_cols
        )
        self.hparams["static_cat_covariates_cols"] = static_cat_covariates_cols

        self._create_transforms()

        if train_df is None and val_df is None:
            self._raw_train_df: list[pd.DataFrame] = []
            self._raw_val_df: list[pd.DataFrame] = []
        elif train_df is None and val_df is not None:
            raise ValueError("val_df must be None if train_df is None.")
        elif train_df is not None and val_df is None:
            raise ValueError("train_df must be None if val_df is None.")
        else:
            self._raw_train_df = _to_list(train_df)
            self._raw_val_df = _to_list(val_df)

        # these will be set by prepare_data
        self._train_df: list[pd.DataFrame] = []
        self._val_df: list[pd.DataFrame] = []

        self._tmp_dir = tempfile.TemporaryDirectory()

    """ Override the following in subclasses """

    def _make_dataset_from_arrays(
        self,
        known_covariates_data: np.ndarray,
        target_data: np.ndarray | None = None,
        past_covariates_data: np.ndarray | None = None,
        static_cont_covariates_data: np.ndarray | None = None,
        static_cat_covariates_data: np.ndarray | None = None,
        predict: bool = False,
    ) -> AbstractTimeSeriesDataset:
        """
        This function should be overridden by subclasses to create a dataset
        from numpy arrays.

        Parameters
        ----------
        known_covariates_data : np.ndarray
            Contiuous covariate data that is known in the past and future.
            This excludes the target column and any past covariate data that
            is only known in the past.
        past_covariates_data : np.ndarray | None
            Continuous covariate data that is only known in the past.
        target_data : np.ndarray | None
            The target data. This should only be provided if ``predict`` is False.
        static_cont_covariate_data : np.ndarray | None
            Static continuous covariate data. This should only be provided if
            the static continuous covariates are set.
        static_cat_covariate_data : np.ndarray | None
            Static categorical covariate data. This should only be provided if
            the static categorical covariates are set.
        predict : bool
            Whether the model is predicting or not.

        Returns
        -------
        AbstractTimeSeriesDataset
            The created dataset.
        """
        raise NotImplementedError

    @classmethod
    def parse_config_kwargs(
        cls, config: BaseConfig, **kwargs: typing.Any
    ) -> dict[str, typing.Any]:
        """
        To be overridden by subclasses to parse other config subclasses.
        """

        default_kwargs = dict(
            normalize=config.normalize,
            downsample=config.downsample,
            downsample_method=config.downsample_method,
            target_depends_on=config.target_depends_on,
            extra_transforms=config.extra_transforms,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
        )

        default_kwargs.update(kwargs)

        return default_kwargs

    """ End override """

    def _create_transforms(self) -> None:
        """
        Instantiates the transforms to be used by the datamodule.
        """
        normalize = self.hparams["normalize"]

        # input transforms
        input_cols = self.hparams["known_covariates_cols"]
        input_cols += self.hparams["past_covariates_cols"]
        input_cols += self.hparams["static_cont_covariates_cols"]
        input_transforms: dict[str, list[BaseTransform]]
        input_transforms = {col: [] for col in input_cols}

        if self.hparams["extra_transforms"] is not None:
            for col, transforms in self.hparams["extra_transforms"].items():
                if col == self.hparams["target_col"]:
                    continue
                if col not in input_transforms:
                    raise ValueError(
                        f"Unknown column {col} in extra_transforms."
                    )
                input_transforms[col].extend(transforms)

        # normalize covariates
        if normalize:
            for input_col in input_cols:
                input_transforms[input_col].append(
                    StandardScaler(num_features_=1)
                )

        self._input_transforms = {
            col: TransformCollection(transforms)
            for col, transforms in input_transforms.items()
        }

        target_transform = []
        if self.hparams["extra_transforms"] is not None:
            if self.hparams["target_col"] in self.hparams["extra_transforms"]:
                extra_transforms = self.hparams["extra_transforms"][
                    self.hparams["target_col"]
                ]

                # check if any transforms are XY
                for transform in extra_transforms:
                    if transform.transform_type == TransformType.XY:
                        if self.hparams["target_depends_on"] is None:
                            raise ValueError(
                                "Transforms that depend on other columns "
                                "require target_depends_on to be set."
                            )
                target_transform.extend(extra_transforms)
        if normalize:
            target_transform.append(StandardScaler(num_features_=1))

        self._target_transform = TransformCollection(
            target_transform,
            transform_type=(
                TransformType.XY
                if self.hparams["target_depends_on"] is not None
                else TransformType.X
            ),
        )

    @classmethod
    def from_dataframe(
        cls: typing.Type[SameType],
        config: BaseConfig,
        train_df: pd.DataFrame | list[pd.DataFrame],
        val_df: pd.DataFrame | list[pd.DataFrame],
        known_covariates_cols: str | typing.Sequence[str] | None = None,
        target_col: str | None = None,
        **kwargs: typing.Any,
    ) -> SameType:
        """
        Create a datamodule instance from dataframes and a config.
        This function does not need to be overridden by subclasses.

        Parameters
        ----------
        config : BaseConfig
            The config or config subclass instance to instanitate the datamodule with.
        train_df : pd.DataFrame | list[pd.DataFrame]
            Dataframes containing the training data.
        val_df : pd.DataFrame | list[pd.DataFrame]
            Dataframes containing the validation data.
        input_columns : str | typing.Sequence[str]
            The names of the input columns.
        target_column : str
            The name of the target column.
        kwargs : typing.Any
            Additional keyword arguments to pass to the datamodule constructor.
            These will override the values specified in the config.

        Returns
        -------
        SameType
            The created datamodule instance.
        """
        train_df = _to_list(train_df)
        val_df = _to_list(val_df)

        if known_covariates_cols is None:
            if "input_columns" in kwargs:
                known_covariates_cols = _to_list(kwargs.pop("input_columns"))
                warnings.warn(
                    "input_columns is deprecated, use known_covariate_cols instead."
                )
            elif config.input_columns is not None:
                known_covariates_cols = _to_list(config.input_columns)
                warnings.warn(
                    "input_columns is deprecated, use known_covariate_cols instead."
                )
            elif config.known_covariates_cols is not None:
                known_covariates_cols = _to_list(config.known_covariates_cols)
            else:
                raise ValueError(
                    "Either known_covariate_columns or "
                    "config.input_columns must be set."
                )

        if target_col is None:
            if "target_column" in kwargs:
                target_col = kwargs.pop("target_column")
                warnings.warn(
                    "target_column is deprecated, use target_col instead."
                )
            elif config.target_col is not None:
                target_col = config.target_col
            elif config.target_column is not None:
                warnings.warn(
                    "target_column is deprecated, use target_col instead."
                )
                target_col = config.target_column
            else:
                raise ValueError(
                    "Either target_column or config.target_column must be set."
                )

        return cls(
            train_df=train_df,
            val_df=val_df,
            known_covariates_cols=known_covariates_cols,
            target_col=target_col,
            **cls.parse_config_kwargs(config, **kwargs),
        )

    @classmethod
    def from_parquet(
        cls: typing.Type[SameType],
        config: BaseConfig,
        train_dataset: str | typing.Sequence[str] | None = None,
        val_dataset: str | typing.Sequence[str] | None = None,
        known_covariates_cols: str | typing.Sequence[str] | None = None,
        target_col: str | None = None,
        **kwargs: typing.Any,
    ) -> SameType:
        """
        Create a datamodule instance from a dataset on disk and a config.
        This function does not need to be overridden by subclasses.

        Parameters
        ----------
        config : BaseConfig
            The config or config subclass instance to instanitate the datamodule with.
        train_dataset : str | typing.Sequence[str] | None
            Path to the training dataset, one path per dataframe. If None, the
            path is taken from the config.
        val_dataset : str | typing.Sequence[str] | None
            Path to the validation dataset, one path per dataframe. If None, the
            path is taken from the config.
        kwargs : typing.Any
            Additional keyword arguments to pass to the datamodule constructor.
            These will override the values specified in the config.

        Returns
        -------
        SameType
            The datamodule created instance.
        """
        if train_dataset is None and val_dataset is None:
            if config.train_dataset is None and config.val_dataset is None:
                raise ValueError(
                    "train_dataset and val_dataset must not be None."
                )
            train_dataset = config.train_dataset
            val_dataset = config.val_dataset
        elif train_dataset is None and val_dataset is not None:
            raise ValueError(
                "val_dataset must be None if train_dataset is None."
            )
        elif train_dataset is not None and val_dataset is None:
            raise ValueError(
                "train_dataset must be None if val_dataset is None."
            )

        assert train_dataset is not None
        assert val_dataset is not None

        train_dataset = _to_list(train_dataset)
        val_dataset = _to_list(val_dataset)

        train_df = [pd.read_parquet(pth) for pth in train_dataset]
        val_df = [pd.read_parquet(pth) for pth in val_dataset]

        return cls.from_dataframe(
            config,
            train_df,
            val_df,
            known_covariates_cols=known_covariates_cols,
            target_col=target_col,
            **kwargs,
        )

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
        train_df = list(map(self.preprocess_dataframe, self._raw_train_df))
        val_df = list(map(self.preprocess_dataframe, self._raw_val_df))

        if not self._scalers_fitted():
            self._fit_transforms(pd.concat(train_df))

        self._train_df = list(map(self.apply_transforms, train_df))
        self._val_df = list(map(self.apply_transforms, val_df))

        if save:
            self.save_data(self._tmp_dir.name)

    def setup(
        self,
        stage: (
            typing.Literal["fit", "train", "val", "test", "predict"] | None
        ) = None,
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
        df = downsample(
            df,
            downsample=self.hparams["downsample"],
            method=self.hparams["downsample_method"],
        )
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
        if not self._scalers_fitted():
            raise RuntimeError("Scalers have not been fitted yet. ")

        out = pd.DataFrame(df)
        input_cols = self.hparams["known_covariates_cols"]
        input_cols += self.hparams["past_covariates_cols"]
        input_cols += self.hparams["static_cont_covariates_cols"]
        for col in input_cols:
            out[col] = self._input_transforms[col].transform(
                torch.from_numpy(df[col].to_numpy())
            )

        if skip_target:
            return out

        target_col = self.hparams["target_col"]
        if self.hparams["target_depends_on"] is not None:
            out[target_col] = self._target_transform.transform(
                torch.from_numpy(
                    df[self.hparams["target_depends_on"]].to_numpy()
                ),
                torch.from_numpy(df[target_col].to_numpy()),
            )
        else:
            out[target_col] = self._target_transform.transform(
                torch.tensor([]),
                torch.from_numpy(df[target_col].to_numpy()),
            )

        return out

    def transform_input(
        self,
        input_: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
    ) -> pd.DataFrame:
        """
        Chains the ``read_input`` and ``preprocess_dataframe``, and
        ``normalize_dataframe`` functions together.

        Parameters
        ----------
        input_ : pd.DataFrame
            The input data.
        timestamp : np.ndarray | pd.Series | None
            The timestamps of the data.

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
        skip_target = self.hparams["target_col"] not in input_.columns
        input_cols = self.hparams["known_covariates_cols"]
        input_cols += self.hparams["past_covariates_cols"]
        input_cols += self.hparams["static_cont_covariates_cols"]
        df = self.read_input(
            input_,
            timestamp=timestamp,
            input_columns=input_cols,
            target_column=(
                self.hparams["target_col"] if not skip_target else None
            ),
        )
        df = self.preprocess_dataframe(df)

        df = self.apply_transforms(df, skip_target=skip_target)

        return df

    def _make_dataset_from_df(
        self, df: pd.DataFrame, predict: bool = False
    ) -> torch.utils.data.Dataset:
        target_data: np.ndarray | None = None
        if self.hparams["target_col"] in df.columns:
            target_data = df[self.hparams["target_col"]].to_numpy()

        return self._make_dataset_from_arrays(
            known_covariates_data=df[
                self.hparams["known_covariates_cols"]
            ].to_numpy(),
            target_data=target_data,
            past_covariates_data=df[
                self.hparams["past_covariates_cols"]
            ].to_numpy(),
            static_cont_covariates_data=df[
                self.hparams["static_cont_covariates_cols"]
            ].to_numpy(),
            static_cat_covariates_data=df[
                self.hparams["static_cat_covariates_cols"]
            ].to_numpy(),
            predict=predict,
        )

    def make_dataset(
        self,
        df: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
        predict: bool = False,
    ) -> TimeSeriesDataset:
        df = self.transform_input(
            input_=df,
            timestamp=timestamp,
        )

        return self._make_dataset_from_df(df, predict=predict)

    def make_dataloader(
        self,
        input_: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
        predict: bool = False,
        **kwargs: typing.Any,
    ) -> torch.utils.data.DataLoader:
        dataset = self.make_dataset(
            input_,
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
    def train_dataset(self) -> torch.utils.data.Dataset:
        if self._train_df is None or len(self._train_df) == 0:
            raise ValueError("No training data available.")

        known_covariate_data = np.concatenate(
            [
                df[self.hparams["known_covariates_cols"]].to_numpy()
                for df in self._train_df
            ]
        )
        past_covariate_data = np.concatenate(
            [
                df[self.hparams["past_covariates_cols"]].to_numpy()
                for df in self._train_df
            ]
        )
        target_data = np.concatenate(
            [
                df[self.hparams["target_col"]].to_numpy()
                for df in self._train_df
            ]
        )
        static_cont_covariate_data = np.concatenate(
            [
                df[self.hparams["static_cont_covariates_cols"]].to_numpy()
                for df in self._train_df
            ]
        )
        static_cat_covariate_data = np.concatenate(
            [
                df[self.hparams["static_cat_covariates_cols"]].to_numpy()
                for df in self._train_df
            ]
        )

        return self._make_dataset_from_arrays(
            known_covariates_data=known_covariate_data,
            past_covariates_data=past_covariate_data,
            target_data=target_data,
            static_cont_covariates_data=static_cont_covariate_data,
            static_cat_covariates_data=static_cat_covariate_data,
            predict=False,
        )

    @property
    def val_dataset(
        self,
    ) -> torch.utils.data.Dataset | typing.Sequence[torch.utils.data.Dataset]:
        if self._val_df is None or len(self._val_df) == 0:
            raise ValueError("No validation data available.")

        datasets = [
            self._make_dataset_from_df(df, predict=True) for df in self._val_df
        ]

        return datasets[0] if len(datasets) == 1 else datasets

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
        if len(self._val_df) == 1:
            return make_dataloader(self.val_dataset)  # type: ignore[arg-type]
        else:
            return [make_dataloader(ds) for ds in self.val_dataset]  # type: ignore[arg-type]

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

    def state_dict(self) -> dict[str, typing.Any]:
        state = super().state_dict()
        if self._input_transforms is not None:
            state["input_transforms"] = {
                col: transform.state_dict()
                for col, transform in self._input_transforms.items()
            }
        if self._target_transform is not None:
            state["target_transform"] = self._target_transform.state_dict()

        return state

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        if "input_transforms" in state:
            for col, transform in self._input_transforms.items():
                if col not in state["input_transforms"]:
                    log.warning(f"Could not find state for {col}.")
                transform.load_state_dict(state["input_transforms"][col])

            state.pop("input_transforms")

        if "target_transform" in state:
            self._target_transform.load_state_dict(state["target_transform"])

            state.pop("target_transform")

        super().load_state_dict(state)

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

    def _scalers_fitted(self) -> bool:
        fitted = all(
            transform.__sklearn_is_fitted__()
            for transform in self._input_transforms.values()
        )

        fitted &= self._target_transform.__sklearn_is_fitted__()

        return fitted

    def _fit_transforms(self, df: pd.DataFrame) -> None:
        input_cols = self.hparams["known_covariates_cols"]
        input_cols += self.hparams["past_covariates_cols"]
        input_cols += self.hparams["static_cont_covariates_cols"]
        for col in input_cols:
            log.info(f"Fitting input scaler for {col}.")
            self._input_transforms[col].fit(
                torch.from_numpy(df[col].to_numpy())
            )

        if self.hparams["target_depends_on"] is not None:
            self._target_transform.fit(
                torch.from_numpy(
                    df[self.hparams["target_depends_on"]].to_numpy()
                ),
                torch.from_numpy(df[self.hparams["target_col"]].to_numpy()),
            )
        else:
            self._target_transform.fit(
                torch.from_numpy(df[self.hparams["target_col"]].to_numpy()),
                torch.tensor([]),
            )

    def get_transforms(
        self,
    ) -> tuple[dict[str, TransformCollection], TransformCollection]:
        return (
            self._input_transforms,
            self._target_transform,
        )

    @property
    def input_transforms(self) -> dict[str, TransformCollection]:
        return self._input_transforms

    @property
    def target_transform(self) -> TransformCollection:
        return self._target_transform

    @staticmethod
    def read_input(
        df: pd.DataFrame,
        input_columns: str | typing.Sequence[str],
        target_column: str | None = None,
        timestamp: np.ndarray | pd.Series | str | None = None,
    ) -> pd.DataFrame:
        """
        Transforms the input data into a dataframe with the specified columns.

        If the inputs and targets are numpy arrays, the columns will be named
        ``input_0``, ``input_1``, ..., ``input_n`` and ``target_0``,
        ``target_1``, ..., ``target_m``, if ``input_columns`` and
        ``

        Parameters
        ----------
        input_ : pd.DataFrame
            The input data.
        timestamp : np.ndarray | pd.Series | pd.DataFrame | str | None
            The timestamps of the data.
        """
        input_columns = _to_list(input_columns)

        df = df.dropna(how="all", axis="columns")

        col_filter = input_columns
        if target_column is not None:
            col_filter = col_filter + _to_list(target_column)

        df = df[col_filter]

        if timestamp is not None:
            if isinstance(timestamp, str):
                df[TIME] = df[timestamp]
            elif isinstance(timestamp, (np.ndarray, pd.Series)):
                if len(timestamp) != len(df):
                    raise ValueError(
                        "The length of the timestamp must match the length of the data."
                        f"Got {len(timestamp)} timestamps and {len(df)} rows."
                    )
                df[TIME] = timestamp
        else:
            df[TIME] = np.arange(len(df))

        return df


T = typing.TypeVar("T")


def _to_list(x: T | typing.Sequence[T]) -> list[T]:
    if isinstance(x, typing.Sequence) and not isinstance(
        x, (str, pd.Series, np.ndarray, torch.Tensor, pd.DataFrame)
    ):
        return list(x)
    else:
        return typing.cast(list[T], [x])
