from __future__ import annotations

import dataclasses
import functools
import logging
import pathlib
import shutil
import sys
import tempfile
import typing
from os import path
from pathlib import Path

import lightning as L
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from .._downsample import DOWNSAMPLE_METHODS, downsample
from ..dataset import AbstractTimeSeriesDataset
from ..transform import (
    BaseTransform,
    StandardScaler,
    TransformCollection,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType_co = typing.TypeVar("SameType_co", bound="DataModuleBase", covariant=True)

T = typing.TypeVar("T")
TIME = "time_ms"


TIME_PREFIX = "__time__"
KNOWN_COV_CONT_PREFIX = "__known_continuous__"
PAST_KNOWN_COV_PREFIX = "__past_known_continuous__"
TARGET_PREFIX = "__target__"


@dataclasses.dataclass
class TmpDir:
    """
    Utility class to handle temporary directories for when distributed
    training is used, and a datamodule is created on each worker.
    """

    name: str

    def cleanup(self) -> None:
        shutil.rmtree(self.name)


class TmpDirType(typing.Protocol):
    """Abstraction to represent a temporary directory."""

    name: str

    def cleanup(self) -> None: ...


class Covariate(typing.NamedTuple):
    name: str
    col: str


class DataModuleBase(L.LightningDataModule):
    """
    Abstract base class for all data modules, handles the bulk transformations
    of data, but does not construct the datasets

    Don't forget to call :meth:`save_hyperparameters` in your
    subclass constructor.
    """

    _input_transforms: torch.nn.ModuleDict[str, TransformCollection]
    _target_transform: TransformCollection
    _tmp_dir: TmpDirType

    _raw_train_df: list[pd.DataFrame]
    _raw_val_df: list[pd.DataFrame]

    def __init__(
        self,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        known_past_covariates: str | typing.Sequence[str] | None = None,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,  # noqa: FBT001, FBT002
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        *,
        shuffle: bool = True,
        distributed_sampler: bool = False,
    ):
        """
        Initializes the datamodule.

        Parameters
        ----------
        known_covariates : str | typing.Sequence[str]
            The columns in the data that are known in the past and future. These
            columns are used as input to the model. If the input to the model
            follows encoder-decoder architecture, these columns are used as the
            encoder *and* decoder input.
        target_covariate : str
            The column in the data that is the target of the model. This column
            is used as the target in the model, and if the input to the model
            follows encoder-decoder architecture, this column is used as the
            decoder input.
        known_past_covariates : str | typing.Sequence[str] | None
            The columns in the data that are known in the past. These columns
            are used as input to the model, but only as the encoder input. This
            column should not be used for normal sequence-to-sequence models.
        train_df_paths : str | list[str] | None
            The path to the training data. If a list is provided, the dataframes
            are concatenated. If None, the training data is not loaded, which is
            useful for when loading the datamodule from a checkpoint.
        val_df_paths : str | list[str] | None
            The path to the validation data. If a list is provided, multiple validation
            datasets are created, and the :meth:`val_dataloader` method returns a list
            of dataloaders. If None, the validation data is not loaded, which is useful
            for when loading the datamodule from a checkpoint.
        normalize : bool
            Whether to normalize the data. If True, the data is normalized using
            a standard scaler. If False, the data is not normalized.
        downsample : int
            The factor to downsample the data by. If 1, the data is not downsampled.
        downsample_method : typing.Literal["interval", "average", "convolve"]
            The method to use for downsampling the data. The options are "interval",
            "median", and "convolve".
        target_depends_on : str | None
            The column that the target depends on if additional transforms are provided in the
            ``extra_transforms`` parameter, and are of type
            :class:`BaseTransform.TransformType.XY`, where the target depends on
            another column. If None, the target does not depend on any other column.
        extra_transforms : dict[str, list[BaseTransform]] | None
            Additional transforms to apply to the data. The dictionary should have the column
            name as the key, and a list of transforms as the value. If None, no additional
            transforms are applied. By default, the data is normalized using a standard scaler.
            If additional transforms are provided, the data is normalized after the additional
            transforms are applied.
        batch_size : int
            The batch size to use for the dataloaders.
        num_workers : int
             Number of workers to use for the dataloaders.
        dtype : str
            The data type to use for the data. This is passed to the datasets which convert
            the data to this type when samples are created.
        shuffle : bool
            Whether to shuffle the data. If True, the data is shuffled before creating the
            training dataloader. If False, the data is not shuffled.
            Validation / test dataloaders are never shuffled.
        distributed_sampler : bool
            Whether to use a distributed sampler for the dataloaders. If True, the dataloader
            is wrapped in a :class:`torch.utils.data.distributed.DistributedSampler`, which must
            be used by distributed training strategies.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["extra_transforms"])

        known_covariates = _to_list(known_covariates)
        known_past_covariates = (
            _to_list(known_past_covariates) if known_past_covariates else []
        )

        self.hparams["known_covariates"] = known_covariates
        self.hparams["known_past_covariates"] = known_past_covariates

        self._extra_transforms_source = extra_transforms or {}
        self._create_transforms()

        self._train_df_pths = _or_empty(train_df_paths)

        self._val_df_pths = _or_empty(val_df_paths)

        # these will be set by prepare_data
        self._train_df: list[pd.DataFrame] = []
        self._val_df: list[pd.DataFrame] = []

        self._init_tmpdir()

    """ Override the following in subclasses """

    def _make_dataset_from_df(
        self, df: pd.DataFrame | list[pd.DataFrame], *, predict: bool = False
    ) -> AbstractTimeSeriesDataset:
        raise NotImplementedError

    """ End override """

    @property
    def num_past_known_covariates(self) -> int:
        """
        The number of past known covariates.

        Returns
        -------
        int
            The number of past known covariates.
        """
        return (
            len(self.hparams["known_covariates"])
            + len(self.hparams["known_past_covariates"])
            + 1
        )

    @property
    def num_future_known_covariates(self) -> int:
        """
        The number of future known covariates.

        Returns
        -------
        int
            The number of future known covariates.
        """
        return len(self.hparams["known_covariates"]) + 1

    @property
    def num_static_real_features(self) -> int:
        """
        The number of static real features.

        Returns
        -------
        int
            The number of static real features.
        """
        return 0

    @override  # type: ignore[misc]
    def prepare_data(self) -> None:
        """
        Loads and preprocesses data dataframes.

        The dataframes are loaded from parquet files, and then preprocessed
        and normalized. The dataframes are then saved to parquet files in the temporary
        directory, which are then loaded in the :meth:`setup` method, on all the nodes.

        The dataframes are purposely *not* concatenated to keep
        distinct data from different sources separate.
        The data will be concatenated in subclasses of :class:`AbstractTimeSeriesDataset`
        after data has been split and samples created using the sliding window technique.

        Parameters
        ----------
        save : bool
            Whether to save the dataframes to parquet files.

        """
        super().prepare_data()
        # load all data into memory and then apply transforms
        train_pths = [Path(pth).expanduser() for pth in self._train_df_pths]
        val_pths = [Path(pth).expanduser() for pth in self._val_df_pths]

        train_df = list(map(read_dataframe, train_pths))
        val_df = list(map(read_dataframe, val_pths))

        parse_dataframe = functools.partial(
            self.parse_dataframe,
            input_columns=self.hparams["known_covariates"],
            past_known_columns=self.hparams.get("known_past_covariates"),
            target_column=self.hparams["target_covariate"],
        )

        train_df = list(
            map(
                parse_dataframe,
                train_df,
            )
        )
        val_df = list(
            map(
                parse_dataframe,
                val_df,
            )
        )

        self._raw_train_df = train_df
        self._raw_val_df = val_df

        train_df = list(map(self.preprocess_dataframe, train_df))
        val_df = list(map(self.preprocess_dataframe, val_df))

        if not self._scalers_fitted():
            self._fit_transforms(pd.concat(train_df))

        self._train_df = list(map(self.apply_transforms, train_df))
        self._val_df = list(map(self.apply_transforms, val_df))

        save_data(self._train_df, "train", self._tmp_dir.name)
        save_data(self._val_df, "val", self._tmp_dir.name)

    @override  # type: ignore[misc]
    def setup(
        self,
        stage: typing.Literal["fit", "train", "val", "test", "predict"] | None = None,
    ) -> None:
        """
        Sets up the data for training, validation or testing.
        Loads the preprocessed data from the temporary directory.

        Parameters
        ----------
        stage : typing.Literal["fit", "train", "val", "test", "predict"] | None
            The stage to setup for. If None, all stages are setup.
        """

        def load_parquet(
            name: typing.Literal["train", "val", "test", "predict"],
            dir_: str,
        ) -> list[pd.DataFrame]:
            """Convenience function to load data from parquet files."""
            paths = tmp_data_paths(None, name, dir_)
            dfs = []
            for pth in paths:
                df = pd.read_parquet(pth)
                dfs.append(df)
            return dfs

        if stage is None or stage in {"train", "fit"}:
            self._train_df = load_parquet("train", self._tmp_dir.name)
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "val":
            self._val_df = load_parquet("val", self._tmp_dir.name)
        elif stage == "test":
            msg = (
                "Datamodule does not support using the test set.\n"
                "Use the 'make_dataset' or 'make_dataloader' methods instead."
            )
            raise NotImplementedError(msg)
        elif stage == "predict":
            msg = (
                "Datamodule does not support using the predict set.\n"
                "Use the 'make_dataset' or 'make_dataloader' methods instead."
            )
            raise NotImplementedError(msg)
        else:
            msg = f"Unknown stage {stage}."
            raise ValueError(msg)

    def teardown(self, stage: str) -> None:
        """
        Cleans up the temporary directory.

        Parameters
        ----------
        stage : str
            The stage to teardown.
        """
        if self._tmp_dir is None:
            return

        if self.hparams.get("distributed_sampler"):
            self._tmp_dir.cleanup()
        else:
            try:
                shutil.rmtree(self._tmp_dir.name)
            except OSError:
                log.exception(
                    f"Failed to remove temporary directory {self._tmp_dir.name}."
                )

    @property
    def train_dataset(self) -> AbstractTimeSeriesDataset:
        """
        The training dataset. This is a concatenation of all the training dataframes
        that have been loaded and preprocessed.

        Returns
        -------
        AbstractTimeSeriesDataset
            The training dataset.
        """
        if self._train_df is None or len(self._train_df) == 0:
            msg = "No training data available."
            raise ValueError(msg)

        return self._make_dataset_from_df(self._train_df, predict=False)

    @property
    def val_dataset(
        self,
    ) -> AbstractTimeSeriesDataset | list[AbstractTimeSeriesDataset]:
        """
        The validation dataset(s). If more than one dataframe was provided in the
        constructor, this method returns a list of datasets, one for each dataframe.
        Otherwise, it returns a single dataset.

        Returns
        -------
        AbstractTimeSeriesDataset | list[AbstractTimeSeriesDataset]
            The validation dataset(s).
        """
        if self._val_df is None or len(self._val_df) == 0:
            msg = "No validation data available."
            raise ValueError(msg)

        datasets = [self._make_dataset_from_df(df, predict=True) for df in self._val_df]

        return datasets[0] if len(datasets) == 1 else datasets

    @override  # type: ignore[misc]
    def train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader:
        """
        Returns the training dataloader. If the datamodule is used in distributed
        training, the dataloader is wrapped in a :class:`torch.utils.data.distributed.DistributedSampler`.

        Returns
        -------
        torch.utils.data.DataLoader
            The training dataloader.
        """
        sampler: torch.utils.data.Sampler | None = None
        if self.hparams["distributed_sampler"]:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                shuffle=self.hparams["shuffle"],
                drop_last=True,
            )
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=sampler is None and self.hparams["shuffle"],
            num_workers=self.hparams["num_workers"],
            sampler=sampler,
        )

    @override  # type: ignore[misc]
    def val_dataloader(
        self,
    ) -> torch.utils.data.DataLoader | typing.Sequence[torch.utils.data.DataLoader]:
        """
        Returns the validation dataloader(s). If the datamodule is used in distributed
        training, the dataloaders are wrapped in a :class:`torch.utils.data.distributed.DistributedSampler`.

        Returns
        -------
        torch.utils.data.DataLoader | typing.Sequence[torch.utils.data.DataLoader]
            The validation dataloader(s).
        """
        if self._val_df is None or len(self._val_df) == 0:
            msg = "No validation data available."
            raise ValueError(msg)

        def make_sampler(
            ds: torch.utils.data.Dataset,
        ) -> torch.utils.data.Sampler | None:
            if self.hparams["distributed_sampler"]:
                return torch.utils.data.distributed.DistributedSampler(
                    ds,
                    shuffle=False,
                    drop_last=False,
                )
            return None

        def make_dataloader(
            ds: torch.utils.data.Dataset,
        ) -> torch.utils.data.DataLoader:
            return torch.utils.data.DataLoader(
                ds,
                batch_size=1,
                num_workers=self.hparams["num_workers"],
                shuffle=False,
                sampler=make_sampler(ds),
            )

        if len(self._val_df) == 1:
            return make_dataloader(self.val_dataset)  # type: ignore[arg-type]
        return [make_dataloader(ds) for ds in self.val_dataset]  # type: ignore[arg-type]

    def get_transforms(
        self,
    ) -> tuple[dict[str, TransformCollection], TransformCollection]:
        """
        Returns the input and target transforms used by the datamodule.

        Returns
        -------
        tuple[dict[str, TransformCollection], TransformCollection]
            The input and target transforms.
        """
        return typing.cast(
            tuple[dict[str, TransformCollection], TransformCollection],
            (
                {
                    col: self._input_transforms[col]
                    for col in self.hparams["known_covariates"]
                },
                self._target_transform,
            ),
        )

    @property
    def input_transforms(self) -> dict[str, TransformCollection]:
        """
        The input transforms used by the datamodule.

        Returns
        -------
        dict[str, TransformCollection]
            The input transforms.
        """
        return typing.cast(
            dict[str, TransformCollection],
            {
                cov.name: self._input_transforms[cov.name]
                for cov in self.known_covariates + self.known_past_covariates
            },
        )

    @property
    def target_transform(self) -> TransformCollection:
        """
        The target transform used by the datamodule.

        Returns
        -------
        TransformCollection
            The target transform.
        """
        return self._target_transform

    def transform_input(
        self,
        df: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
    ) -> pd.DataFrame:
        """
        Chains the :meth:`_parse_dataframe` and :meth:`preprocess_dataframe``, and
        :meth:`apply_transforms` functions together.

        This in principle applies downsampling, preprocessing and normalization
        to the input data. The result of this function is a dataframe that is ready to be used
        with the :meth:`make_dataset` or :meth:`make_dataloader` functions
        to create a dataset or dataloader.

        Parameters
        ----------
        df : pd.DataFrame
            The input data. This dataframe should contain the columns specified
            in the datamodule.
        timestamp : np.ndarray | pd.Series | None
            The timestamps of the data.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe, ready to be used with the dataset or dataloader.
        Raises
        ------
        TypeError
            If the input is not a ``np.ndarray``, ``pd.Series``, or ``pd.DataFrame``.
        sklearn.exceptions.NotFittedError
            If the normalizers are not yet fitted.
            This is caused by calling ``transform_input`` before ``prepare_data``,
            or using a datamodule that has previously not been trained on.
        """
        skip_target = self.hparams["target_covariate"] not in df.columns
        df = self.parse_dataframe(
            df,
            timestamp=timestamp,
            input_columns=[
                cov.name for cov in self.known_covariates + self.known_past_covariates
            ],
            target_column=(self.target_covariate.name if not skip_target else None),
        )
        df = self.preprocess_dataframe(df)

        return self.apply_transforms(df, skip_target=skip_target)

    @staticmethod
    def parse_dataframe(
        df: pd.DataFrame,
        input_columns: str | typing.Sequence[str],
        past_known_columns: str | typing.Sequence[str] | None = None,
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
        df : pd.DataFrame
            The input data.
        timestamp : np.ndarray | pd.Series | pd.DataFrame | str | None
            The timestamps of the data.
        input_columns : str | typing.Sequence[str]
            The columns to use as input.
        past_known_columns : str | typing.Sequence[str] | None
            The columns to use as known past covariates.
        target_column : str | None
            The column to use as target. If not provided,
            the target column is not included in the output.
        timestamp : np.ndarray | pd.Series | str | None
            The timestamps of the data.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        df = df.dropna(how="all", axis="columns")

        col_filter = _to_list(input_columns)  # type: ignore[arg-type]
        if target_column is not None:
            col_filter += _to_list(target_column)

        df = df[col_filter]

        if timestamp is not None:
            time: np.ndarray | pd.Series
            if isinstance(timestamp, str):
                time = df[timestamp]
            elif isinstance(timestamp, np.ndarray | pd.Series):
                if len(timestamp) != len(df):
                    msg = (
                        "The length of the timestamp must match the length of the data."
                        f"Got {len(timestamp)} timestamps and {len(df)} rows."
                    )
                    raise ValueError(msg)
                time = timestamp
            else:
                msg = (
                    f"Unknown type {type(timestamp)} for timestamp, "
                    "expected str, np.ndarray, or pd.Series."
                )
                raise TypeError(msg)

            out = pd.DataFrame({TIME_PREFIX: time})
        else:
            # out = pd.DataFrame({TIME_PREFIX: np.arange(len(df))})
            out = pd.DataFrame()

        for col in input_columns:
            out[known_cov_col(col)] = df[col]
        if past_known_columns is not None:
            for col in past_known_columns:
                out[past_known_cov_col(col)] = df[col]
        if target_column is not None:
            for col in _to_list(target_column):
                out[target_col(col)] = df[col]

        return out

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe into the format expected by the model.
        This function should be chaininvoked with the ``read_input`` function,
        and must be called before ``apply_transforms``.

        This function extracts the known past and future covariates from the dataframe,
        and adds them as separate columns to the dataframe. The columns are named
        using the prefixes ``__past_known_continuous_`` and ``__future_known_continuous_``.

        This function can be overridden in subclasses to add additional
        preprocessing steps.

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
        *,
        skip_target: bool = False,
    ) -> pd.DataFrame:
        """
        Normalize the dataframe into the format expected by the model.
        This function should be chaininvoked with the :meth:`parse_dataframe` function.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to normalize.
        skip_target : bool
            Whether to skip transforming the target column. This is useful
            if the target column is not present in the input data.

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
            msg = "Scalers have not been fitted yet. "
            raise RuntimeError(msg)

        # apply downsampling prior to transforms
        df = downsample(
            df,
            downsample=self.hparams["downsample"],
            method=self.hparams["downsample_method"],
        )

        out = pd.DataFrame(df)
        for cov in self.known_covariates + self.known_past_covariates:
            out[cov.col] = self._input_transforms[cov.name].transform(
                torch.from_numpy(df[cov.col].to_numpy())
            )

        if skip_target:
            return out

        if self.target_depends_on is not None:
            out[self.target_covariate.col] = self._target_transform.transform(
                torch.from_numpy(df[self.target_depends_on.col].to_numpy()),
                torch.from_numpy(df[self.target_covariate.col].to_numpy()),
            )
        else:
            out[self.target_covariate.col] = self._target_transform.transform(
                torch.from_numpy(df[self.target_covariate.col].to_numpy()),
            )

        return out

    def make_dataloader(
        self,
        df: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
        *,
        predict: bool = False,
        **kwargs: typing.Any,
    ) -> torch.utils.data.DataLoader:
        """
        Creates a dataloader from an input dataframe. The resulting dataloader
        is the same that would be returned by the :meth:`train_dataloader` or
        :meth:`val_dataloader` methods. This method is useful for creating
        dataloaders for data that is not part of the training or validation sets,
        for instance during prediction.

        The method uses the :meth:`make_dataset` method to create the dataset, which
        applies the transformations same as the training and validation sets.

        Parameters
        ----------
        df : pd.DataFrame
            Input data. Must contain the columns specified in the original datamodule.
        timestamp : np.ndarray | pd.Series | str | None
            The timestamps of each row in the input data. If None, the row index is used.
        predict : bool
            Whether the dataloader is used for prediction. If True, the batch size is set to 1
            and the data is not shuffled.
        kwargs : typing.Any
            Additional keyword arguments to pass to the dataloader.

        Returns
        -------
        torch.utils.data.DataLoader
            A standard PyTorch dataloader wrapping the created dataset.
        """
        dataset = self.make_dataset(
            df,
            timestamp=timestamp,
            predict=predict,
        )

        default_kwargs = {
            "batch_size": self.hparams["batch_size"] if not predict else 1,
            "num_workers": self.hparams["num_workers"],
            "shuffle": not predict and self.hparams["shuffle"],
        }
        default_kwargs.update(kwargs)

        return torch.utils.data.DataLoader(
            dataset,
            **default_kwargs,
        )

    def make_dataset(
        self,
        df: pd.DataFrame,
        timestamp: np.ndarray | pd.Series | str | None = None,
        *,
        predict: bool = False,
    ) -> AbstractTimeSeriesDataset:
        df = self.transform_input(
            df=df,
            timestamp=timestamp,
        )

        return self._make_dataset_from_df(df, predict=predict)

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

    def _create_transforms(self) -> None:
        """
        Instantiates the transforms to be used by the datamodule.
        """
        # input transforms
        input_transforms: dict[str, list[BaseTransform]]
        input_transforms = {
            cov.name: [] for cov in self.known_covariates + self.known_past_covariates
        }

        for col, transforms in self._extra_transforms_source.items():
            if col == self.hparams["target_covariate"]:
                continue
            if col not in input_transforms:
                msg = f"Unknown column {col} in extra_transforms."
                raise ValueError(msg)
            input_transforms[col].extend(transforms)

        if self.hparams["normalize"]:
            for cov in self.known_covariates + self.known_past_covariates:
                input_transforms[cov.name].append(StandardScaler(num_features_=1))

        self._input_transforms = torch.nn.ModuleDict({
            col: TransformCollection(transforms)
            for col, transforms in input_transforms.items()
        })

        target_transform = []
        if self.hparams["target_covariate"] in self._extra_transforms_source:
            target_transform.extend(
                self._extra_transforms_source[self.hparams["target_covariate"]]
            )
        if self.hparams["normalize"]:
            target_transform.append(StandardScaler(num_features_=1))

        self._target_transform = TransformCollection(target_transform)

        if (
            self.target_depends_on is not None
            and self._target_transform.transform_type != BaseTransform.TransformType.XY
        ):
            msg = (
                "The target depends on another column, but the target transform "
                f"does not support this. Got {self._target_transform.transform_type}."
            )
            raise ValueError(msg)
        if (
            self.target_depends_on is None
            and self._target_transform.transform_type == BaseTransform.TransformType.XY
        ):
            msg = (
                "The target does not depend on another column, but the target transform "
                f"does. Got {self._target_transform.transform_type}."
            )
            raise ValueError(msg)

    def _scalers_fitted(self) -> bool:
        fitted = all(
            transform.__sklearn_is_fitted__()
            for transform in self._input_transforms.values()
        )

        fitted &= self._target_transform.__sklearn_is_fitted__()

        return fitted

    def _fit_transforms(self, df: pd.DataFrame) -> None:
        for cov in self.known_covariates + self.known_past_covariates:
            log.info(f"Fitting input scaler for {cov.name}.")
            self._input_transforms[cov.name].fit(
                torch.from_numpy(df[cov.col].to_numpy())
            )

        if self.target_depends_on is not None:
            self._target_transform.fit(
                torch.from_numpy(df[self.target_depends_on.col].to_numpy()),
                torch.from_numpy(df[self.target_covariate.col].to_numpy()),
            )
        else:
            self._target_transform.fit(
                torch.from_numpy(df[self.target_covariate.col].to_numpy()),
            )

    @property
    def known_covariates(self) -> list[Covariate]:
        return [
            Covariate(col, known_cov_col(col))
            for col in self.hparams["known_covariates"]
        ]

    @property
    def known_past_covariates(self) -> list[Covariate]:
        return [
            Covariate(col, past_known_cov_col(col))
            for col in (self.hparams["known_past_covariates"] or [])
        ]

    @property
    def target_covariate(self) -> Covariate:
        return Covariate(
            self.hparams["target_covariate"],
            target_col(self.hparams["target_covariate"]),
        )

    @property
    def target_depends_on(self) -> Covariate | None:
        if self.hparams["target_depends_on"] is None:
            return None

        cov_str = self.hparams["target_depends_on"]

        if cov_str in self.hparams["known_covariates"]:
            return Covariate(cov_str, known_cov_col(cov_str))

        if cov_str in self.hparams["known_past_covariates"]:
            return Covariate(cov_str, past_known_cov_col(cov_str))

        msg = (
            f"Unknown column {cov_str} in target_depends_on that "
            "is not in known_covariates or known_past_covariates."
        )
        raise ValueError(msg)

    def _init_tmpdir(self) -> TmpDirType:
        if self.hparams.get("distributed_sampler"):
            self._tmp_dir = TmpDir("/tmp/tmp_datamodule/")
            Path(self._tmp_dir.name).mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory()

        return self._tmp_dir


def _to_list(x: T | typing.Sequence[T]) -> list[T]:
    if isinstance(x, typing.Sequence) and not isinstance(
        x, str | pd.Series | np.ndarray | torch.Tensor | pd.DataFrame
    ):
        return list(x)
    return typing.cast(list[T], [x])


def _or_empty(x: T | typing.Sequence[T] | None) -> list[T]:
    if x is None:
        return []
    return _to_list(x)


def known_cov_col(name: str) -> str:
    return f"{KNOWN_COV_CONT_PREFIX}{name}"


def past_known_cov_col(name: str) -> str:
    return f"{PAST_KNOWN_COV_PREFIX}{name}"


def time_col(name: str) -> str:
    return f"{TIME_PREFIX}{name}"


def target_col(name: str) -> str:
    return f"{TARGET_PREFIX}{name}"


EXT2READER: dict[str, typing.Callable[[typing.Any], pd.DataFrame]] = {
    ".parquet": pd.read_parquet,
    ".csv": pd.read_csv,
    ".tsv": pd.read_csv,
    ".xlsx": pd.read_excel,
    ".xls": pd.read_excel,
    ".json": pd.read_json,
}


def read_dataframe(pth: pathlib.Path) -> pd.DataFrame:
    """
    Reads a dataframe in different formats.

    The function will automatically determine the format of the file
    based on the file extension.

    Parameters
    ----------
    pth : pathlib.Path
        The path to the parquet file.

    Returns
    -------
    pd.DataFrame
        The dataframe.
    """
    extension = pth.suffix
    if extension not in EXT2READER:
        msg = f"Unknown file extension {extension}."
        raise ValueError(msg)
    return EXT2READER[extension](pth)


def save_data(
    dfs: list[pd.DataFrame] | None,
    name: typing.Literal["train", "val", "test", "predict"],
    save_dir: str,
) -> None:
    """
    Saves the data to the model directory.
    """
    if dfs is None:
        return

    paths = tmp_data_paths(dfs, name, save_dir)
    for i, df in enumerate(dfs):
        if len(df) == 0:
            continue
        df.reset_index(drop=True).to_parquet(paths[i])


def tmp_data_paths(
    dfs: list[pd.DataFrame] | None,
    name: typing.Literal["train", "val", "test", "predict"],
    dir_: str,
) -> list[Path]:
    """
    Returns a list of paths to the data files.
    If the data is not saved yet, the paths are generated,
    otherwise the paths are searched for in the model directory.

    Parameters
    ----------
    dfs : list[pd.DataFrame] | None
        The dataframes to save. If None, the paths are generated.
    name : typing.Literal["train", "val", "test", "predict"]
        The name of the data set.
    dir_ : str
        The directory to save the data files in.

    Returns
    -------
    list[Path]
        Paths to where the data is saved or should be loaded from.
    """
    if dfs is not None and len(dfs) > 0:  # for saving
        if len(dfs) == 1 or isinstance(dfs, pd.DataFrame):
            return [Path(dir_) / f"{name}.parquet"]
        return [Path(dir_) / f"{name}_{i}.parquet" for i in range(len(dfs))]
    # for loading
    # try to find the data files
    single_df_path = Path(dir_) / f"{name}.parquet"
    if single_df_path.exists():
        return [single_df_path]
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
        i += 1

    return paths
