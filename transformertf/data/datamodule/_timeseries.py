from __future__ import annotations

import typing

import pandas as pd

from transformertf.data.dataset import TimeSeriesDataset

from .._covariates import TARGET_PREFIX
from .._dataset_factory import DatasetFactory
from .._dtype import VALID_DTYPES
from ._base import DataModuleBase, _to_list

if typing.TYPE_CHECKING:
    from .._downsample import DOWNSAMPLE_METHODS
    from ..transform import BaseTransform


class TimeSeriesDataModule(DataModuleBase):
    """
    Data module for sequence-to-sequence time series modeling.

    This data module is designed for time series forecasting tasks where models
    map a sequence of input covariates to a target covariate using a sliding
    window approach. The input-output relationship is:

    Input: [batch_size, seq_len, n_covariates] → Output: [batch_size, seq_len, 1]

    This is suitable for models like LSTM, GRU, and basic transformer architectures
    that perform sequence-to-sequence prediction without explicit encoder-decoder
    separation.

    Parameters
    ----------
    known_covariates : str or sequence of str
        Column names for input features that are known throughout the sequence.
        These form the input tensor with shape [batch_size, seq_len, n_covariates].
    target_covariate : str
        Column name for the target variable to predict. Forms the output tensor
        with shape [batch_size, seq_len, 1].
    train_df_paths : str or list of str, optional
        Path(s) to training data files. See :class:`DataModuleBase` for details.
    val_df_paths : str or list of str, optional
        Path(s) to validation data files. See :class:`DataModuleBase` for details.
    normalize : bool, optional
        Whether to apply normalization. Default is True.
    seq_len : int, optional
        Length of each sequence sample. Default is 200.
    min_seq_len : int, optional
        Minimum sequence length when using randomized lengths. If None,
        uses ``seq_len``. Default is None.
    randomize_seq_len : bool, optional
        Whether to randomize sequence lengths between ``min_seq_len`` and
        ``seq_len`` during training. Disabled during validation/prediction.
        Default is False.
    stride : int, optional
        Step size for sliding window when creating samples. Default is 1.
    downsample : int, optional
        Downsampling factor. See :class:`DataModuleBase` for details.
    downsample_method : {"interval", "average", "convolve"}, optional
        Downsampling method. See :class:`DataModuleBase` for details.
    target_depends_on : str, optional
        Column dependency for target transforms. See :class:`DataModuleBase`.
    extra_transforms : dict, optional
        Additional transforms. See :class:`DataModuleBase` for details.
    batch_size : int, optional
        Training batch size. Default is 128.
    num_workers : int, optional
        Number of data loading workers. Default is 0.
    dtype : str, optional
        Data type for tensors. Default is "float32".
    shuffle : bool, optional
        Whether to shuffle training data. Default is True.
    distributed : bool or "auto", optional
        Distributed training configuration. Default is "auto".

    Attributes
    ----------
    seq_len : int
        Sequence length for each sample, accessible via property.
    num_past_known_covariates : int
        Number of input covariates, accessible via property.

    Notes
    -----
    This class does not support ``known_past_covariates`` as it's designed for
    sequence-to-sequence models where all covariates are available throughout
    the sequence.

    The sliding window approach creates overlapping samples:
    - Window size: ``seq_len``
    - Step size: ``stride``
    - Randomization: ``randomize_seq_len`` (training only)

    Examples
    --------
    Basic usage for LSTM forecasting:

    >>> dm = TimeSeriesDataModule(
    ...     known_covariates=["temperature", "humidity", "pressure"],
    ...     target_covariate="demand",
    ...     train_df_paths="data/train.parquet",
    ...     val_df_paths="data/val.parquet",
    ...     seq_len=168,  # 1 week with hourly data
    ...     stride=24,    # 1 day stride
    ...     batch_size=32
    ... )

    With sequence length randomization:

    >>> dm = TimeSeriesDataModule(
    ...     known_covariates=["feature1", "feature2"],
    ...     target_covariate="target",
    ...     seq_len=100,
    ...     min_seq_len=50,
    ...     randomize_seq_len=True,
    ...     train_df_paths="data/train.parquet"
    ... )

    See Also
    --------
    DataModuleBase : Base class with common functionality
    TimeSeriesDataset : Underlying dataset implementation
    EncoderDecoderDataModule : For encoder-decoder architectures
    TransformerDataModule : For transformer-based models
    """

    def __init__(
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,
        seq_len: int = 200,
        min_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        target_depends_on: str | None = None,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: VALID_DTYPES = "float32",
        shuffle: bool = True,
        distributed: bool | typing.Literal["auto"] = "auto",
    ):
        """
        Initialize the time series data module.

        See the class docstring for detailed parameter descriptions.
        """
        super().__init__(
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            known_covariates=known_covariates,
            target_covariate=target_covariate,
            known_past_covariates=None,
            normalize=normalize,
            downsample=downsample,
            downsample_method=downsample_method,
            target_depends_on=target_depends_on,
            extra_transforms=extra_transforms,
            batch_size=batch_size,
            num_workers=num_workers,
            dtype=dtype,
            shuffle=shuffle,
            distributed=distributed,
        )

        self.save_hyperparameters(ignore=["extra_transforms"])

        self.hparams["known_covariates"] = _to_list(self.hparams["known_covariates"])

    def _make_dataset_from_df(
        self, df: pd.DataFrame | list[pd.DataFrame], *, predict: bool = False
    ) -> TimeSeriesDataset:
        if len(self.known_past_covariates) > 0:
            msg = "known_past_covariates is not used in this class."
            raise NotImplementedError(msg)

        # Prepare data with correct column prefixes for the factory
        def _prepare_data(data_df: pd.DataFrame) -> pd.DataFrame:
            # Create a copy to avoid modifying original data
            prepared_df = data_df.copy()

            # Keep only known covariates and target columns
            input_cols = [cov.col for cov in self.known_covariates]
            target_cols = [
                col for col in prepared_df.columns if col.startswith(TARGET_PREFIX)
            ]
            all_cols = input_cols + target_cols

            return prepared_df[all_cols]

        if isinstance(df, pd.DataFrame):
            prepared_data = _prepare_data(df)
        else:
            prepared_data = [_prepare_data(d) for d in df]

        return DatasetFactory.create_timeseries_dataset(
            data=prepared_data,
            seq_len=self.hparams["seq_len"],
            min_seq_len=self.hparams["min_seq_len"] if not predict else None,
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            stride=self.hparams["stride"],
            predict=predict,
            transforms=self.transforms,
            dtype=self.hparams["dtype"],
        )

    @property
    def seq_len(self) -> int:
        """
        Returns the sample sequence length. This is used by LightningCLI
        to link arguments to the model.

        Returns
        -------
        int
            Sample sequence length
        """
        return self.hparams["seq_len"]

    @property
    def num_past_known_covariates(self) -> int:
        """
        Returns the number of past known covariates. This is used by LightningCLI
        to link arguments to the model.

        Returns
        -------
        int
            Number of past known covariates
        """
        return len(self.hparams["known_covariates"])
