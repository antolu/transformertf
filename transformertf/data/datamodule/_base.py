from __future__ import annotations

import contextlib
import dataclasses
import functools
import importlib
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

from .._covariates import (
    TIME_PREFIX,
    Covariate,
    known_cov_col,
    past_known_cov_col,
    target_col,
)
from .._downsample import DOWNSAMPLE_METHODS, downsample
from .._dtype import VALID_DTYPES
from .._transform_builder import TransformBuilder
from ..dataset import AbstractTimeSeriesDataset
from ..transform import (
    BaseTransform,
    TransformCollection,
)

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType_co = typing.TypeVar("SameType_co", bound="DataModuleBase", covariant=True)

T = typing.TypeVar("T")
TIME = "time_ms"


@dataclasses.dataclass
class TmpDir:
    """
    Temporary directory manager for distributed training.

    This class handles temporary directories that are created during
    distributed training when a datamodule is instantiated on each worker.
    It provides safe cleanup that suppresses common filesystem errors
    that can occur in multi-process environments.

    Parameters
    ----------
    name : str
        Path to the temporary directory.

    Methods
    -------
    cleanup()
        Remove the temporary directory and all its contents.

    Notes
    -----
    The cleanup method suppresses FileNotFoundError, PermissionError,
    and OSError exceptions that commonly occur when multiple processes
    attempt to clean up the same directory simultaneously.

    Examples
    --------
    >>> tmpdir = TmpDir("/tmp/datamodule_worker_1/")
    >>> # ... use the directory ...
    >>> tmpdir.cleanup()  # Safe cleanup
    """

    name: str

    def cleanup(self) -> None:
        """
        Remove the temporary directory and all its contents.

        Safely handles common filesystem errors that occur during
        distributed cleanup operations.
        """
        with contextlib.suppress(FileNotFoundError, PermissionError, OSError):
            shutil.rmtree(self.name)


class TmpDirType(typing.Protocol):
    """Abstraction to represent a temporary directory."""

    name: str

    def cleanup(self) -> None: ...


class ExtraTransformsHparams(typing.TypedDict):
    """
    Type definition for the extra transforms hyperparameters.
    """

    module: str
    name: str
    kwargs: dict[str, typing.Any]


class DataModuleBase(L.LightningDataModule):
    """
    Abstract base class for all data modules in transformertf.

    This class handles the bulk transformations of time series data, including
    normalization, downsampling, and custom transforms, but does not construct
    the actual datasets (which is left to concrete subclasses). It provides
    a standardized interface for data preprocessing, loading, and management
    across different time series modeling architectures.

    The class integrates with PyTorch Lightning's data module system and
    supports distributed training with automatic temporary directory management.

    Parameters
    ----------
    known_covariates : str or sequence of str
        Column names in the data that are known in both past and future.
        These columns are used as input features to the model. For
        encoder-decoder architectures, these serve as both encoder and
        decoder inputs.
    target_covariate : str
        Column name for the target variable that the model should predict.
        For encoder-decoder architectures, this also serves as decoder input.
    known_past_covariates : str or sequence of str, optional
        Column names that are known only in the past (not future).
        These are used only as encoder inputs in encoder-decoder models.
        Default is None.
    train_df_paths : str or list of str, optional
        Path(s) to training data files. Multiple paths are concatenated.
        If None, training data is not loaded (useful for checkpoint loading).
        Default is None.
    val_df_paths : str or list of str, optional
        Path(s) to validation data files. Multiple paths create separate
        validation datasets. If None, validation data is not loaded.
        Default is None.
    normalize : bool, optional
        Whether to apply standard normalization to the data using
        :class:`~transformertf.data.transform.StandardScaler`.
        Default is True.
    downsample : int, optional
        Factor by which to downsample the data. A value of 1 means no
        downsampling. Default is 1.
    downsample_method : {"interval", "average", "convolve"}, optional
        Method for downsampling. "interval" takes every nth sample,
        "average" computes moving averages, "convolve" applies convolution.
        Default is "interval".
    target_depends_on : str, optional
        Column name that the target depends on when using XY-type transforms
        where the target transformation requires another input column.
        Default is None.
    extra_transforms : dict of {str: list of BaseTransform}, optional
        Additional transforms to apply to specific columns. Keys are column
        names, values are lists of :class:`~transformertf.data.transform.BaseTransform`
        instances. Applied before normalization. Default is None.
    batch_size : int, optional
        Batch size for training dataloaders. Default is 128.
    val_batch_size : int, optional
        Batch size for validation dataloaders. Default is 1.
    num_workers : int, optional
        Number of worker processes for data loading. Default is 0.
    dtype : {"float32", "float64", "int32", "int64"}, optional
        Data type for tensors created by datasets. Default is "float32".
    shuffle : bool, optional
        Whether to shuffle training data. Validation data is never shuffled.
        Default is True.
    distributed : bool or "auto", optional
        Whether to use distributed sampling for multi-GPU training.
        "auto" enables distributed sampling if multiple GPUs are available.
        Default is "auto".

    Attributes
    ----------
    hparams : dict
        Hyperparameters saved by Lightning's save_hyperparameters().
    _transforms : torch.nn.ModuleDict
        Dictionary mapping column names to their transform collections.
    _tmp_dir : TmpDirType
        Temporary directory for storing preprocessed data during distributed training.
    _train_df : list of pd.DataFrame
        Preprocessed training dataframes.
    _val_df : list of pd.DataFrame
        Preprocessed validation dataframes.
    _raw_train_df : list of pd.DataFrame
        Raw training dataframes before preprocessing.
    _raw_val_df : list of pd.DataFrame
        Raw validation dataframes before preprocessing.

    Notes
    -----
    This is an abstract base class. Concrete subclasses must implement:

    - :meth:`_make_dataset_from_df`: Creates dataset instances from dataframes

    The class follows this data processing pipeline:

    1. **Loading**: Read data from files (parquet, CSV, Excel, etc.)
    2. **Parsing**: Extract and rename columns according to covariate types
    3. **Preprocessing**: Apply downsampling and other preprocessing steps
    4. **Transformation**: Apply normalization and custom transforms
    5. **Dataset Creation**: Convert to PyTorch datasets (subclass-specific)
    6. **DataLoader Creation**: Wrap datasets in PyTorch dataloaders

    The class automatically handles:

    - Multiple file formats (parquet, CSV, Excel, JSON)
    - Distributed training with temporary directory management
    - Transform serialization for checkpointing
    - Automatic GPU detection for distributed sampling

    Examples
    --------
    Subclasses typically follow this pattern:

    >>> class MyDataModule(DataModuleBase):
    ...     def __init__(self, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.save_hyperparameters()
    ...
    ...     def _make_dataset_from_df(self, df, predict=False):
    ...         return MyDataset(df, predict=predict)

    Usage with custom transforms:

    >>> from transformertf.data.transform import LogTransform
    >>> extra_transforms = {
    ...     "price": [LogTransform()],
    ...     "volume": [LogTransform()]
    ... }
    >>> dm = MyDataModule(
    ...     known_covariates=["price", "volume"],
    ...     target_covariate="target",
    ...     extra_transforms=extra_transforms
    ... )

    See Also
    --------
    TimeSeriesDataModule : For basic time series forecasting
    EncoderDecoderDataModule : For encoder-decoder architectures
    TransformerDataModule : For transformer-based models
    """

    _transforms: torch.nn.ModuleDict[str, TransformCollection]
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
        normalize: bool = True,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        val_batch_size: int = 1,
        num_workers: int = 0,
        dtype: VALID_DTYPES = "float32",
        *,
        shuffle: bool = True,
        distributed: bool | typing.Literal["auto"] = "auto",
        _legacy_target_in_future_covariates: bool = False,
    ):
        """
        Initialize the data module.

        See the class docstring for detailed parameter descriptions.
        All parameters are stored as hyperparameters and can be accessed
        via ``self.hparams``.

        Note
        ----
        Concrete subclasses must call :meth:`save_hyperparameters` in their
        constructor to ensure proper serialization.
        """
        super().__init__()
        self.save_hyperparameters(ignore=["extra_transforms"])

        # Initialize components in organized way
        self._init_covariates(known_covariates, known_past_covariates)
        self._init_file_paths(train_df_paths, val_df_paths)
        self._init_transforms(extra_transforms)
        self._init_data_containers()
        self._init_tmpdir()

    def _init_covariates(
        self,
        known_covariates: str | typing.Sequence[str],
        known_past_covariates: str | typing.Sequence[str] | None,
    ) -> None:
        """Initialize covariate configuration."""
        self.hparams["known_covariates"] = _to_list(known_covariates)
        self.hparams["known_past_covariates"] = (
            _to_list(known_past_covariates) if known_past_covariates else []
        )

    def _init_file_paths(
        self,
        train_df_paths: str | list[str] | None,
        val_df_paths: str | list[str] | None,
    ) -> None:
        """Initialize file paths for data loading."""
        self._train_df_pths = _or_empty(train_df_paths)
        self._val_df_pths = _or_empty(val_df_paths)

    def _init_transforms(
        self, extra_transforms: dict[str, list[BaseTransform]] | None
    ) -> None:
        """Initialize transform system."""
        self._extra_transforms_source: typing.Mapping[
            str, list[BaseTransform] | list[ExtraTransformsHparams]
        ] = extra_transforms or {}
        self._patch_extra_transforms_load()
        self._create_transforms()
        self._patch_extra_transforms_hparams()

    def _init_data_containers(self) -> None:
        """Initialize data containers that will be populated by prepare_data."""
        self._train_df: list[pd.DataFrame] = []
        self._val_df: list[pd.DataFrame] = []

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
            + (1 if self.hparams.get("add_target_to_past", True) else 0)
            + (1 if self.hparams.get("time_column") else 0)
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
        base_count = len(self.hparams["known_covariates"]) + (
            1 if self.hparams.get("time_column") else 0
        )
        # Legacy compatibility: old checkpoints expected target to be included
        if self.hparams.get("_legacy_target_in_future_covariates", False):
            base_count += 1
        return base_count

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

    @override
    def prepare_data(self) -> None:  # type: ignore[misc]
        """
        Loads and preprocesses data dataframes.

        The dataframes are loaded from parquet files, and then preprocessed
        and normalized. The dataframes are then saved to parquet files in the temporary
        directory, which are then loaded in the :meth:`setup` method, on all the nodes.

        The dataframes are purposely *not* concatenated to keep
        distinct data from different sources separate.
        The data will be concatenated in subclasses of :class:`AbstractTimeSeriesDataset`
        after data has been split and samples created using the sliding window technique.
        """
        super().prepare_data()

        # Load data
        train_df, val_df = self._load_raw_data()

        # Parse and preprocess
        train_df = self._parse_and_preprocess_data(train_df)
        val_df = self._parse_and_preprocess_data(val_df)

        # Store raw data for reference
        self._raw_train_df = train_df
        self._raw_val_df = val_df

        # Fit and apply transforms
        self._fit_and_apply_transforms(train_df, val_df)

        # Save processed data
        self._save_processed_data()

    def _load_raw_data(self) -> tuple[list[pd.DataFrame], list[pd.DataFrame]]:
        """Load raw data from file paths."""
        train_paths = [Path(p).expanduser() for p in self._train_df_pths]
        val_paths = [Path(p).expanduser() for p in self._val_df_pths]

        return (
            list(map(read_dataframe, train_paths)),
            list(map(read_dataframe, val_paths)),
        )

    def _parse_and_preprocess_data(self, dfs: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """Parse and preprocess data frames."""
        parse_fn = functools.partial(
            self.parse_dataframe,
            input_columns=self.hparams["known_covariates"],
            past_known_columns=self.hparams.get("known_past_covariates"),
            target_column=self.hparams["target_covariate"],
            timestamp=self.hparams.get("time_column"),
        )

        parsed_dfs = list(map(parse_fn, dfs))
        return list(map(self.preprocess_dataframe, parsed_dfs))

    def _fit_and_apply_transforms(
        self, train_df: list[pd.DataFrame], val_df: list[pd.DataFrame]
    ) -> None:
        """Fit transforms on training data and apply to both sets."""
        if not self._scalers_fitted():
            self._fit_transforms(train_df)

        self._train_df = list(map(self.apply_transforms, train_df))
        self._val_df = list(map(self.apply_transforms, val_df))

    def _save_processed_data(self) -> None:
        """Save processed data to temporary directory."""
        save_data(self._train_df, "train", self._tmp_dir.name)
        save_data(self._val_df, "val", self._tmp_dir.name)

        if self.distributed_sampler:
            self._save_tmp_state()

    @override
    def setup(  # type: ignore[misc]
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
        if self.distributed_sampler:
            self._load_tmp_state()

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

    def _save_tmp_state(self) -> None:
        state_dict = self.state_dict()

        with open(path.join(self._tmp_dir.name, "state_dict.pt"), "wb") as f:
            torch.save(state_dict, f)

    def _load_tmp_state(self) -> None:
        with open(path.join(self._tmp_dir.name, "state_dict.pt"), "rb") as f:
            state_dict = torch.load(f)

        self.load_state_dict(state_dict)

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

        if not self.hparams.get("distributed_sampler"):
            self._tmp_dir.cleanup()
        else:
            if not pathlib.Path(self._tmp_dir.name).exists():
                return
            try:
                shutil.rmtree(self._tmp_dir.name)
            except (FileNotFoundError, PermissionError, OSError):
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

    @override
    def train_dataloader(  # type: ignore[misc]
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
        if self.distributed_sampler:
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
            pin_memory=True,
            multiprocessing_context="forkserver"
            if self.hparams["num_workers"] > 0
            else None,
            persistent_workers=self.hparams["num_workers"] > 0,
            collate_fn=self.collate_fn,
        )

    @override
    def val_dataloader(  # type: ignore[misc]
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
            if self.distributed_sampler:
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
                batch_size=self.hparams["val_batch_size"],
                num_workers=self.hparams["num_workers"],
                shuffle=False,
                sampler=make_sampler(ds),
                pin_memory=True,
                multiprocessing_context="forkserver"
                if self.hparams["num_workers"] > 0
                else None,
                persistent_workers=self.hparams["num_workers"] > 0,
                collate_fn=self.collate_fn,
            )

        if len(self._val_df) == 1:
            return make_dataloader(self.val_dataset)  # type: ignore[arg-type]
        return [make_dataloader(ds) for ds in self.val_dataset]  # type: ignore[arg-type]

    @property
    def transforms(self) -> dict[str, TransformCollection]:
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
                cov.name: self._transforms[cov.name]
                for cov in self.known_covariates
                + self.known_past_covariates
                + [self.target_covariate]
                + (
                    [Covariate(TIME_PREFIX, TIME_PREFIX)]
                    if self.hparams.get("time_column")
                    else []
                )
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
        return self._transforms[self.target_covariate.name]

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
            timestamp=timestamp
            if timestamp is not None
            else self.hparams.get("time_column"),
            input_columns=[cov.name for cov in self.known_covariates],
            past_known_columns=[cov.name for cov in self.known_past_covariates],
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

            if (
                pd.api.types.is_datetime64_any_dtype(time.dtype)
                or pd.api.types.is_timedelta64_dtype(time.dtype)
                or pd.api.types.is_string_dtype(time.dtype)
            ):
                time = pd.to_numeric(time)

            time = np.array(time, dtype=float)
            out = pd.DataFrame({TIME_PREFIX: time})
        else:
            # out = pd.DataFrame({TIME_PREFIX: np.arange(len(df))})
            out = pd.DataFrame()

        for col in input_columns:
            out[known_cov_col(col)] = df[col].to_numpy()
        if past_known_columns is not None:
            for col in past_known_columns:
                out[past_known_cov_col(col)] = df[col].to_numpy()
        if target_column is not None:
            for col in _to_list(target_column):
                out[target_col(col)] = df[col].to_numpy()

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
        # apply downsampling prior to transforms
        return downsample(
            df,
            downsample=self.hparams["downsample"],
            method=self.hparams["downsample_method"],
        )

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

        out = pd.DataFrame(df)
        for cov in self.known_covariates + self.known_past_covariates:
            out[cov.col] = self._transforms[cov.name].transform(
                torch.from_numpy(df[cov.col].to_numpy())
            )

        if skip_target:
            return out

        def pd2torch(x: pd.Series) -> torch.Tensor:
            return torch.from_numpy(x.to_numpy())

        target_transform = self._transforms[self.target_covariate.name]
        if self.target_depends_on is not None:
            out[self.target_covariate.col] = target_transform.transform(
                pd2torch(df[self.target_depends_on.col]),
                pd2torch(df[self.target_covariate.col]),
            )
        else:
            out[self.target_covariate.col] = target_transform.transform(
                pd2torch(df[self.target_covariate.col]),
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
            "persistent_workers": self.hparams["num_workers"] > 0,
            "pin_memory": True,
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
        if self._transforms is not None:
            state["transforms"] = {
                col: transform.state_dict()
                for col, transform in self._transforms.items()
            }

        return state

    def load_state_dict(self, state: dict[str, typing.Any]) -> None:
        if "transforms" in state:
            for col, transform in self._transforms.items():
                if col not in state["transforms"]:
                    log.warning(f"Could not find state for {col}.")
                    continue
                transform.load_state_dict(state["transforms"][col])

            state.pop("transforms")

        # handle old state dicts
        elif "input_transform" in state or "target_transform" in state:
            state_dict = None
            if "input_transform" in state:
                state_dict = state["input_transform"]
                state.pop("input_transform")
            if "target_transform" in state:
                if state_dict is None:
                    state_dict = state["target_transform"]
                else:
                    state_dict[self.target_covariate.name] = state["target_transform"]
                state.pop("target_transform")

            if state_dict is not None:
                self._transforms.load_state_dict(state_dict)

        super().load_state_dict(state)

    def _create_transforms(self) -> None:
        """
        Instantiates the transforms to be used by the datamodule using TransformBuilder.
        """
        builder = TransformBuilder()

        # Add covariate transforms
        covariate_names = [
            cov.name for cov in self.known_covariates + self.known_past_covariates
        ]
        if covariate_names:
            builder.add_covariate_transforms(
                covariate_names=covariate_names,
                extra_transforms=self._extra_transforms_source,
                normalize=self.hparams["normalize"],
            )

        # Add target transforms
        builder.add_target_transforms(
            target_name=self.target_covariate.name,
            extra_transforms=self._extra_transforms_source,
            normalize=self.hparams["normalize"],
            depends_on=self.hparams.get("target_depends_on"),
        )

        # Build transforms with validation
        self._transforms = builder.build()

    def _patch_extra_transforms_load(self) -> None:
        extra_transforms_loaded: dict[str, list[BaseTransform]] = {}
        for colname, extra_transforms in self._extra_transforms_source.items():
            extra_transforms_loaded[colname] = [
                transform
                if isinstance(transform, BaseTransform)
                else instantiate_transform(**transform)
                for transform in extra_transforms
            ]

        self._extra_transforms_source = extra_transforms_loaded

    def _patch_extra_transforms_hparams(self) -> None:
        """
        Patches the hparams with extra transforms to ensure that the transforms
        are serializable. This function translates transforms into strings, with the
        init kwargs taken from the transform's state_dict. No positional arguments are
        supported.
        """
        if len(self._extra_transforms_source) == 0:  # no extra transforms
            return

        covariate_name: str
        transforms: list[BaseTransform] | list[ExtraTransformsHparams]
        extra_transforms_patched: dict[str, list[ExtraTransformsHparams]] = {}
        for covariate_name, transforms in self._extra_transforms_source.items():
            transforms_patched: list[ExtraTransformsHparams] = []
            for transform in transforms:
                if not isinstance(transform, BaseTransform):  # already patched
                    continue

                state_dict = transform.state_dict()
                cls = transform.__class__
                module = cls.__module__

                kwargs = {k: v for k, v in state_dict.items() if k.endswith("_")}

                # save where to import it from and how to instantiate it
                transforms_patched.append({
                    "module": module,
                    "name": cls.__name__,
                    "kwargs": kwargs,
                })

            extra_transforms_patched[covariate_name] = transforms_patched

        self.hparams["extra_transforms"] = extra_transforms_patched

    def _scalers_fitted(self) -> bool:
        return all(
            transform.__sklearn_is_fitted__() for transform in self._transforms.values()
        )

    def _fit_transforms(self, dfs: list[pd.DataFrame]) -> None:
        df = pd.concat(dfs)

        for cov in self.known_covariates + self.known_past_covariates:
            log.info(f"Fitting input scaler for {cov.name}.")
            self._transforms[cov.name].fit(torch.from_numpy(df[cov.col].to_numpy()))

        if self.target_depends_on is not None:
            self._transforms[self.target_covariate.name].fit(
                torch.from_numpy(df[self.target_depends_on.col].to_numpy()),
                torch.from_numpy(df[self.target_covariate.col].to_numpy()),
            )
        else:
            self._transforms[self.target_covariate.name].fit(
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

    @property
    def distributed_sampler(self) -> bool:
        return (
            self.hparams["distributed"]
            if self.hparams["distributed"] != "auto"
            else (torch.cuda.is_available() and torch.cuda.device_count() > 1)
            if "distributed" in self.hparams
            else False
        )

    @staticmethod
    def collate_fn() -> typing.Callable[
        list[dict[str, torch.Tensor]], dict[str, torch.Tensor]
    ]:
        """
        Returns a collate function that can be used with the dataloader.

        Subclasses can override this method to provide a custom collate function.
        """
        return torch.utils.data.default_collate

    def _init_tmpdir(self) -> TmpDirType:
        if self.distributed_sampler:
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
    Read a dataframe from various file formats.

    Automatically determines the file format based on the file extension
    and uses the appropriate pandas reader function. Supports common data
    formats used in time series analysis.

    Parameters
    ----------
    pth : pathlib.Path
        Path to the data file. Supported extensions: .parquet, .csv, .tsv,
        .xlsx, .xls, .json.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.

    Raises
    ------
    ValueError
        If the file extension is not supported.

    Examples
    --------
    >>> import pathlib
    >>> df = read_dataframe(pathlib.Path("data/timeseries.parquet"))
    >>> df = read_dataframe(pathlib.Path("data/features.csv"))

    Notes
    -----
    Supported file formats and their corresponding pandas readers:

    - .parquet → pd.read_parquet
    - .csv → pd.read_csv
    - .tsv → pd.read_csv
    - .xlsx/.xls → pd.read_excel
    - .json → pd.read_json
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


def instantiate_transform(
    module: str,
    name: str,
    kwargs: dict[str, str | dict[str, typing.Any]],
) -> BaseTransform:
    """
    Instantiates a transform from a module and class name.

    Parameters
    ----------
    module : str
        The module to import the class from.
    name : str
        The name of the class to instantiate.
    kwargs : dict[str, str | dict[str, typing.Any]]
        The keyword arguments to pass to the class constructor
    """
    cls = getattr(importlib.import_module(module), name)
    return cls(**kwargs)
