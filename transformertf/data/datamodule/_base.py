from __future__ import annotations

import dataclasses
import logging
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

from transformertf.data.dataset import AbstractTimeSeriesDataset

from .._downsample import downsample
from ..transform import (
    BaseTransform,
    StandardScaler,
    TransformCollection,
    TransformType,
)

if sys.version_info < (3, 12):
    from typing_extensions import override
else:
    from typing import override

__all__ = ["DataModuleBase"]

log = logging.getLogger(__name__)

if typing.TYPE_CHECKING:
    SameType_co = typing.TypeVar("SameType_co", bound="DataModuleBase", covariant=True)

TIME = "time_ms"


@dataclasses.dataclass
class TmpDir:
    """
    Utility class to handle temporary directories for when distributed
    training is used, and a datamodule is created on each worker.
    """

    name: str


class TmpDirType(typing.Protocol):
    name: str


class DataModuleBase(L.LightningDataModule):
    """
    Abstract base class for all data modules, handles the bulk transformations
    of data, but does not construct the datasets

    Don't forget to call :meth:`save_hyperparameters` in your
    subclass constructor.
    """

    _input_transforms: torch.nn.ModuleDict[str, TransformCollection]
    _target_transform: TransformCollection

    def __init__(
        self,
        input_columns: str | typing.Sequence[str],
        target_column: str,
        known_past_columns: str | typing.Sequence[str] | None = None,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,  # noqa: FBT001, FBT002
        downsample: int = 1,
        downsample_method: typing.Literal[
            "interval", "average", "convolve"
        ] = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: str = "float32",
        *,
        distributed_sampler: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["extra_transforms"])

        input_columns = _to_list(input_columns)
        known_past_columns = _to_list(known_past_columns) if known_past_columns else []

        self.hparams["input_columns"] = input_columns
        self.hparams["known_past_columns"] = known_past_columns

        self._extra_transforms_source = extra_transforms or {}

        self._create_transforms()

        self._train_df_pths = (
            _to_list(train_df_paths) if train_df_paths is not None else []
        )
        self._val_df_pths = _to_list(val_df_paths) if val_df_paths is not None else []

        # these will be set by prepare_data
        self._train_df: list[pd.DataFrame] = []
        self._val_df: list[pd.DataFrame] = []

        self._tmp_dir: TmpDirType
        if distributed_sampler:
            self._tmp_dir = TmpDir("/tmp/tmp_datamodule/")
            pth = Path(self._tmp_dir.name)

            pth.mkdir(parents=True, exist_ok=True)
        else:
            self._tmp_dir = tempfile.TemporaryDirectory()

    """ Override the following in subclasses """

    def _make_dataset_from_arrays(
        self,
        input_data: np.ndarray,
        known_past_data: np.ndarray | None = None,
        target_data: np.ndarray | None = None,
        *,
        predict: bool = False,
    ) -> AbstractTimeSeriesDataset:
        raise NotImplementedError

    """ End override """

    @override  # type: ignore[misc]
    def prepare_data(self, *, save: bool = True) -> None:
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
        super().prepare_data()
        # load all data into memory and then apply transforms
        train_pths = [Path(pth).expanduser() for pth in self._train_df_pths]
        val_pths = [Path(pth).expanduser() for pth in self._val_df_pths]

        train_df = list(map(pd.read_parquet, train_pths))
        val_df = list(map(pd.read_parquet, val_pths))

        self._raw_train_df = train_df
        self._raw_val_df = val_df

        train_df = list(map(self.preprocess_dataframe, train_df))
        val_df = list(map(self.preprocess_dataframe, val_df))

        if not self._scalers_fitted():
            self._fit_transforms(pd.concat(train_df))

        self._train_df = list(map(self.apply_transforms, train_df))
        self._val_df = list(map(self.apply_transforms, val_df))

        if save:
            save_data(self._train_df, "train", self._tmp_dir.name)
            save_data(self._val_df, "val", self._tmp_dir.name)

    @override  # type: ignore[misc]
    def setup(
        self,
        stage: typing.Literal["fit", "train", "val", "test", "predict"] | None = None,
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

    @property
    def train_dataset(self) -> torch.utils.data.Dataset:
        if self._train_df is None or len(self._train_df) == 0:
            msg = "No training data available."
            raise ValueError(msg)

        input_data = np.concatenate([
            df[self.hparams["input_columns"]].to_numpy() for df in self._train_df
        ])
        known_past_data = (
            np.concatenate([
                df[self.hparams["known_past_columns"]].to_numpy()
                for df in self._train_df
            ])
            if self.hparams["known_past_columns"] is not None
            else None
        )
        target_data = np.concatenate([
            df[self.hparams["target_column"]].to_numpy() for df in self._train_df
        ])

        return self._make_dataset_from_arrays(
            input_data, known_past_data, target_data, predict=False
        )

    @property
    def val_dataset(
        self,
    ) -> torch.utils.data.Dataset | typing.Sequence[torch.utils.data.Dataset]:
        if self._val_df is None or len(self._val_df) == 0:
            msg = "No validation data available."
            raise ValueError(msg)

        datasets = [self._make_dataset_from_df(df, predict=True) for df in self._val_df]

        return datasets[0] if len(datasets) == 1 else datasets

    @override  # type: ignore[misc]
    def train_dataloader(
        self,
    ) -> torch.utils.data.DataLoader | typing.Sequence[torch.utils.data.DataLoader]:
        sampler: torch.utils.data.Sampler | None = None
        if self.hparams["distributed_sampler"]:
            sampler = torch.utils.data.distributed.DistributedSampler(
                self.train_dataset,
                shuffle=True,
                drop_last=True,
            )
        else:
            sampler = None

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.hparams["batch_size"],
            shuffle=sampler is None,
            num_workers=self.hparams["num_workers"],
            sampler=sampler,
        )

    @override  # type: ignore[misc]
    def val_dataloader(
        self,
    ) -> torch.utils.data.DataLoader | typing.Sequence[torch.utils.data.DataLoader]:
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
        return (
            {col: self._input_transforms[col] for col in self.hparams["input_columns"]},
            self._target_transform,
        )

    @property
    def input_transforms(self) -> dict[str, TransformCollection]:
        return {
            col: self._input_transforms[col] for col in self.hparams["input_columns"]
        }

    @property
    def target_transform(self) -> TransformCollection:
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
        skip_target = self.hparams["target_column"] not in df.columns
        df = self.parse_dataframe(
            df,
            timestamp=timestamp,
            input_columns=self.hparams["input_columns"],
            target_column=(self.hparams["target_column"] if not skip_target else None),
        )
        df = self.preprocess_dataframe(df)

        return self.apply_transforms(df, skip_target=skip_target)

    @staticmethod
    def parse_dataframe(
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
        df : pd.DataFrame
            The input data.
        timestamp : np.ndarray | pd.Series | pd.DataFrame | str | None
            The timestamps of the data.
        input_columns : str | typing.Sequence[str]
            The columns to use as input.
        target_column : str | None
            The column to use as target.
        timestamp : np.ndarray | pd.Series | str | None
            The timestamps of the data.

        Returns
        -------
        pd.DataFrame
            The transformed dataframe.
        """
        input_columns = _to_list(input_columns)

        df = df.dropna(how="all", axis="columns")

        col_filter = input_columns
        if target_column is not None:
            col_filter += _to_list(target_column)

        df = df[col_filter]

        if timestamp is not None:
            if isinstance(timestamp, str):
                df[TIME] = df[timestamp]
            elif isinstance(timestamp, np.ndarray | pd.Series):
                if len(timestamp) != len(df):
                    msg = (
                        "The length of the timestamp must match the length of the data."
                        f"Got {len(timestamp)} timestamps and {len(df)} rows."
                    )
                    raise ValueError(msg)
                df[TIME] = timestamp
        else:
            df[TIME] = np.arange(len(df))

        return df

    def preprocess_dataframe(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe into the format expected by the model.
        This function should be chaininvoked with the ``read_input`` function,
        and must be called before ``apply_transforms``.

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
        return downsample(  # type: ignore[return-value]
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
        for col in self.hparams["input_columns"]:
            out[col] = self._input_transforms[col].transform(
                torch.from_numpy(df[col].to_numpy())
            )

        if skip_target:
            return out

        if self.hparams["target_depends_on"] is not None:
            out[self.hparams["target_column"]] = self._target_transform.transform(
                torch.from_numpy(df[self.hparams["target_depends_on"]].to_numpy()),
                torch.from_numpy(df[self.hparams["target_column"]].to_numpy()),
            )
        else:
            out[self.hparams["target_column"]] = self._target_transform.transform(
                torch.tensor([]),
                torch.from_numpy(df[self.hparams["target_column"]].to_numpy()),
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
            "shuffle": not predict,
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

    def _make_dataset_from_df(
        self, df: pd.DataFrame, *, predict: bool = False
    ) -> AbstractTimeSeriesDataset:
        target_data: np.ndarray | None = None
        if self.hparams["target_column"] in df.columns:
            target_data = df[self.hparams["target_column"]].to_numpy()

        return self._make_dataset_from_arrays(
            input_data=df[self.hparams["input_columns"]].to_numpy(),
            known_past_data=df[self.hparams["known_past_columns"]].to_numpy()
            if self.hparams["known_past_columns"] is not None
            else None,
            target_data=target_data,
            predict=predict,
        )

    def _create_transforms(self) -> None:
        """
        Instantiates the transforms to be used by the datamodule.
        """
        normalize = self.hparams["normalize"]

        # input transforms
        input_transforms: dict[str, list[BaseTransform]]
        input_transforms = {col: [] for col in self.hparams["input_columns"]}
        input_transforms |= {col: [] for col in self.hparams["known_past_columns"]}

        for col, transforms in self._extra_transforms_source.items():
            if col == self.hparams["target_column"]:
                continue
            if col not in input_transforms:
                msg = f"Unknown column {col} in extra_transforms."
                raise ValueError(msg)
            input_transforms[col].extend(transforms)

        for input_col in self.hparams["input_columns"]:
            if normalize:
                input_transforms[input_col].append(StandardScaler(num_features_=1))

        self._input_transforms = torch.nn.ModuleDict({
            col: TransformCollection(transforms)
            for col, transforms in input_transforms.items()
        })

        target_transform = []
        if self.hparams["target_column"] in self._extra_transforms_source:
            target_transform.extend(
                self._extra_transforms_source[self.hparams["target_column"]]
            )
        if normalize:
            target_transform.append(StandardScaler(num_features_=1))

        self._target_transform = TransformCollection(
            target_transform,
            transform_type=TransformType.XY,
        )

    def _scalers_fitted(self) -> bool:
        fitted = all(
            transform.__sklearn_is_fitted__()
            for transform in self._input_transforms.values()
        )

        fitted &= self._target_transform.__sklearn_is_fitted__()

        return fitted

    def _fit_transforms(self, df: pd.DataFrame) -> None:
        for col in self.hparams["input_columns"]:
            log.info(f"Fitting input scaler for {col}.")
            self._input_transforms[col].fit(torch.from_numpy(df[col].to_numpy()))

        if self.hparams["target_depends_on"] is not None:
            self._target_transform.fit(
                torch.from_numpy(df[self.hparams["target_depends_on"]].to_numpy()),
                torch.from_numpy(df[self.hparams["target_column"]].to_numpy()),
            )
        else:
            self._target_transform.fit(
                torch.tensor([]),
                torch.from_numpy(df[self.hparams["target_column"]].to_numpy()),
            )


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


T = typing.TypeVar("T")


def _to_list(x: T | typing.Sequence[T]) -> list[T]:
    if isinstance(x, typing.Sequence) and not isinstance(
        x, str | pd.Series | np.ndarray | torch.Tensor | pd.DataFrame
    ):
        return list(x)
    return typing.cast(list[T], [x])
