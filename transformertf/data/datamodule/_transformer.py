from __future__ import annotations

import functools
import sys
import typing

import numba
import numba.typed
import numpy as np
import pandas as pd
import torch

from .._dataset_factory import DatasetFactory
from .._downsample import DOWNSAMPLE_METHODS
from .._dtype import VALID_DTYPES, get_dtype
from .._sample_generator import EncoderDecoderTargetSample
from .._transform_builder import TransformBuilder
from .._window_generator import WindowGenerator
from ..dataset import EncoderDecoderDataset
from ..dataset._encoder_decoder_predict import EncoderDecoderPredictDataset
from ..transform import (
    BaseTransform,
    DeltaTransform,
    TransformCollection,
)
from ._base import TIME_PREFIX as TIME
from ._base import DataModuleBase, _to_list

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override

if typing.TYPE_CHECKING:
    from ..transform import BaseTransform

# Import sequence utilities for collate function
from ...utils.sequence import align_encoder_sequences


class TransformerDataModule(DataModuleBase):
    """
    Base data module for transformer-based time series models.

    This class provides the foundation for transformer architectures that use
    encoder-decoder patterns with separate context and target sequences. It
    handles temporal features, sequence length management, and noise injection
    for robust training.

    Unlike :class:`TimeSeriesDataModule`, this class supports both encoder-only
    and encoder-decoder architectures with separate context and target sequence
    lengths. It also provides advanced features like temporal encoding, sequence
    length randomization, and noise injection.

    Parameters
    ----------
    known_covariates : str or sequence of str
        Column names for features known in both past and future contexts.
        Used as input to both encoder and decoder in encoder-decoder models.
    target_covariate : str
        Column name for the target variable to predict.
    known_past_covariates : str or sequence of str, optional
        Column names for features known only in the past context.
        Used only as encoder input in encoder-decoder models.
    train_df_paths : str or list of str, optional
        Path(s) to training data files. See :class:`DataModuleBase`.
    val_df_paths : str or list of str, optional
        Path(s) to validation data files. See :class:`DataModuleBase`.
    normalize : bool, optional
        Whether to apply normalization. Default is True.
    ctxt_seq_len : int, optional
        Length of the context (encoder) sequence. Default is 500.
    tgt_seq_len : int, optional
        Length of the target (decoder) sequence. Default is 300.
    min_ctxt_seq_len : int, optional
        Minimum context sequence length for randomization. If None, uses
        ``ctxt_seq_len``. Default is None.
    min_tgt_seq_len : int, optional
        Minimum target sequence length for randomization. If None, uses
        ``tgt_seq_len``. Default is None.
    randomize_seq_len : bool, optional
        Whether to randomize sequence lengths during training. Default is False.
    stride : int, optional
        Step size for sliding window sample creation. Default is 1.
    downsample : int, optional
        Downsampling factor. See :class:`DataModuleBase`.
    downsample_method : str, optional
        Downsampling method. See :class:`DataModuleBase`.
    noise_std : float, optional
        Standard deviation of Gaussian noise added to input data during
        training for robustness. Default is 0.0 (no noise).
    target_depends_on : str, optional
        Column dependency for target transforms. See :class:`DataModuleBase`.
    time_column : str, optional
        Column name containing timestamps. If provided, temporal features
        are added to the model inputs. Default is None.
    time_format : {"relative", "absolute", "relative_legacy"}, optional
        Format for temporal features:
        - "relative": Time differences (Δt) normalized with MinMaxScaler [-1, 1]
        - "absolute": Absolute time normalized with MinMaxScaler [-1, 1]
        - "relative_legacy": Time differences with StandardScaler (deprecated)
        Default is "absolute".
    add_target_to_past : bool, optional
        Whether to add target values to past context in encoder-decoder models.
        Default is True.
    extra_transforms : dict, optional
        Additional transforms. Use "__time__" key for temporal transforms.
        See :class:`DataModuleBase`.
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
    encoder_alignment : {"left", "right"}, optional
        Encoder sequence alignment strategy. "left" produces standard left-aligned sequences
        (data at start, padding at end) for PyTorch compatibility. "right" produces
        right-aligned sequences (data at end, padding at start) for TFT-style models.
        Default is "left".

    Attributes
    ----------
    ctxt_seq_len : int
        Context sequence length, accessible via property.
    tgt_seq_len : int
        Target sequence length, accessible via property.

    Notes
    -----
    This is a base class for transformer-based models. Concrete implementations
    include:

    - :class:`EncoderDecoderDataModule`: Full encoder-decoder architecture
    - :class:`EncoderDataModule`: Encoder-only architecture

    **Temporal Features**:

    When ``time_column`` is provided, temporal features are automatically added:

    - **Relative format**: Computes time differences (Δt) relative to sequence start
    - **Absolute format**: Uses absolute timestamps normalized per batch
    - **Legacy format**: Deprecated relative format with StandardScaler

    **Sequence Length Management**:

    The class supports flexible sequence length handling:

    - Fixed lengths: ``ctxt_seq_len`` and ``tgt_seq_len``
    - Randomized lengths: Between min/max values during training
    - Validation/prediction: Always uses maximum lengths

    **Noise Injection**:

    Training robustness can be improved by adding Gaussian noise to inputs
    via the ``noise_std`` parameter. Noise is disabled during validation/prediction.

    Examples
    --------
    Basic transformer data module setup:

    >>> dm = TransformerDataModule(
    ...     known_covariates=["feature1", "feature2"],
    ...     target_covariate="target",
    ...     ctxt_seq_len=200,
    ...     tgt_seq_len=50,
    ...     time_column="timestamp",
    ...     time_format="relative"
    ... )

    With temporal transforms and noise injection:

    >>> from transformertf.data.transform import LogTransform
    >>> dm = TransformerDataModule(
    ...     known_covariates=["price"],
    ...     target_covariate="demand",
    ...     ctxt_seq_len=168,
    ...     tgt_seq_len=24,
    ...     time_column="datetime",
    ...     noise_std=0.01,
    ...     extra_transforms={"__time__": [LogTransform()]},
    ...     randomize_seq_len=True,
    ...     min_ctxt_seq_len=100,
    ...     min_tgt_seq_len=12
    ... )

    See Also
    --------
    DataModuleBase : Base class with common functionality
    EncoderDecoderDataModule : For full encoder-decoder models
    EncoderDataModule : For encoder-only models
    TimeSeriesDataModule : For sequence-to-sequence models
    """

    def __init__(
        self,
        *,
        known_covariates: str | typing.Sequence[str],
        target_covariate: str,
        known_past_covariates: str | typing.Sequence[str] | None = None,
        train_df_paths: str | list[str] | None = None,
        val_df_paths: str | list[str] | None = None,
        normalize: bool = True,
        ctxt_seq_len: int = 500,
        tgt_seq_len: int = 300,
        min_ctxt_seq_len: int | None = None,
        min_tgt_seq_len: int | None = None,
        randomize_seq_len: bool = False,
        stride: int = 1,
        downsample: int = 1,
        downsample_method: DOWNSAMPLE_METHODS = "interval",
        noise_std: float = 0.0,
        target_depends_on: str | None = None,
        time_column: str | None = None,
        time_format: typing.Literal[
            "relative", "absolute", "relative_legacy"
        ] = "absolute",
        add_target_to_past: bool = True,
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
        batch_size: int = 128,
        num_workers: int = 0,
        dtype: VALID_DTYPES = "float32",
        shuffle: bool = True,
        distributed: bool | typing.Literal["auto"] = "auto",
        encoder_alignment: typing.Literal["left", "right"] = "left",
        mask_encoder_features: list[str] | None = None,
        _legacy_target_in_future_covariates: bool = False,
    ):
        """
        Initialize the transformer data module.

        See the class docstring for detailed parameter descriptions.
        """
        super().__init__(
            train_df_paths=train_df_paths,
            val_df_paths=val_df_paths,
            known_covariates=known_covariates,
            target_covariate=target_covariate,
            known_past_covariates=known_past_covariates,
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
            _legacy_target_in_future_covariates=_legacy_target_in_future_covariates,
        )

        self.save_hyperparameters(ignore=["extra_transforms"])

        self.hparams["known_covariates"] = _to_list(self.hparams["known_covariates"])
        self.hparams["known_past_covariates"] = (
            _to_list(self.hparams["known_past_covariates"])
            if self.hparams["known_past_covariates"] is not None
            else []
        )

    def _create_transforms(self) -> None:
        """
        Create transforms using TransformBuilder, including time transforms.
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

        # Add time transforms if time column is specified
        if self.hparams["time_column"] is not None:
            builder.add_time_transforms(
                time_format=self.hparams["time_format"],
                time_column=TIME,
                extra_transforms=self._extra_transforms_source.get(TIME),
                normalize=self.hparams["normalize"],
            )

        # Build transforms with validation
        self._transforms = builder.build()

    def _fit_transforms(self, dfs: list[pd.DataFrame]) -> None:
        super()._fit_transforms(dfs)

        if self.hparams["time_column"] is not None:
            if self.hparams["time_format"] in ["relative", "relative_legacy"]:
                self._fit_relative_time(
                    dfs, self._transforms[TIME], stride=self.hparams["stride"]
                )
            elif self.hparams["time_format"] == "absolute":
                self._fit_absolute_time(dfs, self._transforms[TIME])

    @staticmethod
    def _fit_relative_time(
        dfs: list[pd.DataFrame], transform: BaseTransform, stride: int = 1
    ) -> None:
        """
        Stride is a form of downsamle, since samples are taken at `stride` intervals,
        so we need to fit the scalers taking this into account. The relative time
        uses dt (relative time difference), so we only need to find the largest
        dt possible to fit a MinMaxScaler (since dt is assumed to be always positive).
        """
        for df in dfs:
            for start in range(stride):
                time = df[TIME].iloc[start::stride].to_numpy(dtype=float)
                if isinstance(transform, TransformCollection):
                    # apply the transforms iteratively. Later transforms must be fitted
                    # on already transformerd data. For the delta transform, we need to
                    # shave off the first element, which is set to zero by the transform.
                    for t in transform:
                        time = t.fit_transform(time)
                        if isinstance(t, DeltaTransform):
                            time = time[1:]
                else:
                    transform.fit(time)

    def _fit_absolute_time(
        self, dfs: list[pd.DataFrame], transform: BaseTransform
    ) -> None:
        """
        Fits the absolute time scaler to the data. Since absolute time is monotonically
        increasing, the scaling transform must be fitted on samples directly, so we
        create sliding window samples to fit the largest time axis possible.
        """
        wgs = [
            WindowGenerator(
                num_points=len(df),
                window_size=self.hparams["ctxt_seq_len"] + self.hparams["tgt_seq_len"],
                stride=1,
                zero_pad=False,
            )
            for df in dfs
        ]

        dts = []
        for df, wg in zip(dfs, wgs, strict=True):
            for start in range(self.hparams["stride"]):
                time = df[TIME].to_numpy(dtype=float)[start :: self.hparams["stride"]]

                dt = _calc_fast_absolute_dt(time, numba.typed.List(wg))
                dts.append(dt)

        transform.fit(np.concatenate(dts).flatten())

    @property
    def ctxt_seq_len(self) -> int:
        """Exposes context sequence length for LightningCLI"""
        return self.hparams["ctxt_seq_len"]

    @property
    def tgt_seq_len(self) -> int:
        """Exposes context sequence length for LightningCLI"""
        return self.hparams["tgt_seq_len"]


@numba.njit
def _calc_fast_absolute_dt(time: np.ndarray, slices: list[slice]) -> np.ndarray:
    """
    Calculates the absolute time difference between the start of the slice and the
    rest of the slice. This is a fast implementation using Numba.
    """
    dt = np.zeros(len(slices))
    for i, sl in enumerate(slices):
        dt[i] = np.max(np.abs(time[sl] - time[sl.start]))
    return dt


class EncoderDecoderDataModule(TransformerDataModule):
    """
    Data module for encoder-decoder transformer architectures.

    This class implements the full encoder-decoder pattern where the encoder
    processes past context and the decoder generates future predictions. It
    creates samples with separate encoder and decoder sequences, supporting
    models like Temporal Fusion Transformer (TFT) and other attention-based
    encoder-decoder architectures.

    The data module creates samples with the following structure:
    - **Encoder input**: Past context with known covariates and past-only covariates
    - **Decoder input**: Future context with known covariates and target history
    - **Target output**: Future target values to predict

    Parameters
    ----------
    Inherits all parameters from :class:`TransformerDataModule`.

    Attributes
    ----------
    Inherits all attributes from :class:`TransformerDataModule`.

    Notes
    -----
    This class extends :class:`TransformerDataModule` to create
    :class:`~transformertf.data.dataset.EncoderDecoderDataset` instances.
    The key differences from the base class:

    - **Encoder sequence**: Contains past context with all available features
    - **Decoder sequence**: Contains future context with known covariates only
    - **Target sequence**: Future target values aligned with decoder sequence
    - **Temporal features**: Added to both encoder and decoder if time_column is provided
    - **Noise injection**: Applied to encoder inputs only during training

    **Sample Structure**:

    Each sample contains:
    - ``encoder_input``: [ctxt_seq_len, n_encoder_features]
    - ``decoder_input``: [tgt_seq_len, n_decoder_features]
    - ``target``: [tgt_seq_len, 1]
    - ``encoder_lengths``: Actual encoder sequence length (if randomized)
    - ``decoder_lengths``: Actual decoder sequence length (if randomized)
    - ``encoder_mask``: Padding mask for encoder (if needed)
    - ``decoder_mask``: Padding mask for decoder (if needed)

    **Collate Function**:

    The class provides a specialized collate function that handles variable-length
    sequences by:
    - Trimming encoder sequences from the beginning
    - Trimming decoder sequences from the end
    - Maintaining alignment between decoder input and target output

    Examples
    --------
    Basic encoder-decoder setup:

    >>> dm = EncoderDecoderDataModule(
    ...     known_covariates=["temperature", "humidity"],
    ...     known_past_covariates=["holiday_flag"],
    ...     target_covariate="demand",
    ...     ctxt_seq_len=168,  # 1 week context
    ...     tgt_seq_len=24,    # 1 day prediction
    ...     time_column="datetime",
    ...     train_df_paths="data/train.parquet"
    ... )

    With sequence length randomization:

    >>> dm = EncoderDecoderDataModule(
    ...     known_covariates=["price", "volume"],
    ...     target_covariate="returns",
    ...     ctxt_seq_len=100,
    ...     tgt_seq_len=20,
    ...     min_ctxt_seq_len=50,
    ...     min_tgt_seq_len=10,
    ...     randomize_seq_len=True,
    ...     noise_std=0.001,
    ...     train_df_paths="data/financial.parquet"
    ... )

    See Also
    --------
    TransformerDataModule : Base class for transformer data modules
    EncoderDecoderDataset : Underlying dataset implementation
    EncoderDataModule : For encoder-only architectures
    """

    @override
    def _make_dataset_from_df(
        self,
        df: pd.DataFrame | list[pd.DataFrame],
        *,
        predict: bool = False,
    ) -> EncoderDecoderDataset:
        """Create encoder-decoder dataset using DatasetFactory with standardized column prefixes."""
        time_format: typing.Literal["relative", "absolute"] = (
            "relative"
            if self.hparams["time_format"] in {"relative", "relative_legacy"}
            else "absolute"
        )

        # Prepare data with correct column prefixes for the factory
        def _prepare_dataframe(data_df: pd.DataFrame) -> pd.DataFrame:
            cols = [
                cov.col
                for cov in self.known_covariates
                + self.known_past_covariates
                + [self.target_covariate]
            ]
            try:
                prepared_df = data_df[cols].copy()
            except KeyError as e:
                missing_cols = set(cols) - set(data_df.columns)
                msg = f"Missing required columns in the DataFrame: {missing_cols}"
                raise KeyError(msg) from e

            # Add time data with __time__ prefix if configured
            if self.hparams["time_column"] and TIME in data_df.columns:
                prepared_df[TIME] = data_df[TIME]

            return prepared_df

        if isinstance(df, pd.DataFrame):
            prepared_data = _prepare_dataframe(df)
        else:
            prepared_data = [_prepare_dataframe(d) for d in df]

        return DatasetFactory.create_encoder_decoder_dataset(
            data=prepared_data,
            ctx_seq_len=self.hparams["ctxt_seq_len"],
            tgt_seq_len=self.hparams["tgt_seq_len"],
            min_ctx_seq_len=self.hparams["min_ctxt_seq_len"],
            min_tgt_seq_len=self.hparams["min_tgt_seq_len"],
            randomize_seq_len=(
                self.hparams["randomize_seq_len"] if not predict else False
            ),
            stride=self.hparams["stride"],
            predict=predict,
            transforms=self.transforms,
            dtype=self.hparams["dtype"],
            time_format=time_format,
            noise_std=self.hparams["noise_std"] if not predict else 0.0,
            add_target_to_past=self.hparams["add_target_to_past"],
            masked_encoder_features=self.hparams.get("mask_encoder_features"),
        )

    def collate_fn(  # type: ignore[override]
        self,
    ) -> typing.Callable[
        [list[EncoderDecoderTargetSample]], EncoderDecoderTargetSample
    ]:
        """
        Returns a collate function for encoder-decoder samples with variable lengths.

        Returns
        -------
        callable
            A collate function that takes a list of EncoderDecoderTargetSample and returns
            a collated batch with all samples padded/trimmed to consistent lengths.
        """
        return functools.partial(
            EncoderDecoderDataModule._collate_impl,
            encoder_alignment=self.hparams["encoder_alignment"],
        )

    @staticmethod
    def _collate_impl(
        samples: list[EncoderDecoderTargetSample],
        *,
        encoder_alignment: typing.Literal["left", "right"],
    ) -> EncoderDecoderTargetSample:
        """
        Collate function for encoder-decoder samples with variable lengths.

        This function handles batching of encoder-decoder samples that may have
        different sequence lengths due to randomization. The sequence alignment
        strategy is controlled by the encoder_alignment parameter:

        - "left": Encoder sequences left-aligned (data at start, LSTM-style, PyTorch compatible)
        - "right": Encoder sequences right-aligned (data at end, TFT-style)

        Parameters
        ----------
        samples : list of EncoderDecoderTargetSample
            List of samples to collate into a batch.
        encoder_alignment : {"left", "right"}
            Sequence alignment strategy.

        Returns
        -------
        EncoderDecoderTargetSample
            Collated batch with all samples padded/trimmed to consistent lengths.
            Contains batched tensors for encoder_input, decoder_input, target,
            and optional encoder_lengths, decoder_lengths, encoder_mask, decoder_mask.

        Notes
        -----
        The collation behavior is explicit and deterministic:
        - "left" alignment: Encoder sequences left-aligned (data at start, padding at end)
        - "right" alignment: Encoder sequences right-aligned (data at end, padding at start)
        - Decoder sequences are always trimmed from the end (keeping early predictions)
        - Maximum lengths are determined by the longest sequences in the batch
        """
        if all("encoder_lengths" in sample for sample in samples):
            max_enc_len = max(sample["encoder_lengths"] for sample in samples)
        else:
            max_enc_len = samples[0]["encoder_input"].size(1)

        max_enc_len = int(max_enc_len)

        if all("decoder_lengths" in sample for sample in samples):
            max_tgt_len = max(sample["decoder_lengths"] for sample in samples)
        else:
            max_tgt_len = samples[0]["decoder_input"].size(1)

        assert max_tgt_len > 0

        max_tgt_len = int(max_tgt_len)

        cut_samples = []
        for sample in samples:
            # All alignments use the same trimming initially
            # Left alignment will be applied after collation if needed
            cut_sample = {
                "encoder_input": sample["encoder_input"][-max_enc_len:],
                "decoder_input": sample["decoder_input"][:max_tgt_len],
                "target": sample["target"][:max_tgt_len],
            }

            if "encoder_lengths" in sample:
                cut_sample["encoder_lengths"] = sample["encoder_lengths"]
            if "decoder_lengths" in sample:
                cut_sample["decoder_lengths"] = sample["decoder_lengths"]

            if "encoder_mask" in sample:
                cut_sample["encoder_mask"] = sample["encoder_mask"][-max_enc_len:]

            if "decoder_mask" in sample:
                cut_sample["decoder_mask"] = sample["decoder_mask"][:max_tgt_len]

            cut_samples.append(cut_sample)

        batch = typing.cast(
            EncoderDecoderTargetSample[torch.Tensor],
            torch.utils.data.dataloader.default_collate(cut_samples),
        )

        # default is right alignment from the Dataset.
        if (
            encoder_alignment == "left"
            and "encoder_lengths" in batch
            and batch["encoder_lengths"] is not None
        ):
            # Check if we actually have variable lengths that would benefit from alignment
            encoder_lengths = (
                batch["encoder_lengths"].squeeze(-1)
                if batch["encoder_lengths"].dim() > 1
                else batch["encoder_lengths"]
            )
            if not (encoder_lengths == encoder_lengths[0]).all():
                # Variable lengths detected - align encoder sequences to right-aligned format
                batch["encoder_input"] = align_encoder_sequences(
                    batch["encoder_input"],
                    encoder_lengths,
                    max_enc_len,
                    alignment="left",
                )
                if "encoder_mask" in batch:
                    batch["encoder_mask"] = align_encoder_sequences(
                        batch["encoder_mask"],
                        encoder_lengths,
                        max_enc_len,
                        alignment="left",
                    )

        return batch

    def create_prediction_dataset(
        self,
        past_covariates: pd.DataFrame,
        future_covariates: pd.DataFrame,
        past_target: pd.DataFrame | np.ndarray | pd.Series,
        *,
        context_length: int | None = None,
        prediction_length: int | None = None,
        apply_transforms: bool = True,
    ) -> EncoderDecoderPredictDataset:
        """
        Create a prediction dataset configured with this DataModule's settings.

        This method creates an EncoderDecoderPredictDataset using the same configuration
        as the DataModule (transforms, column mappings, etc.) to ensure consistency
        between training and prediction.

        Parameters
        ----------
        past_covariates : pd.DataFrame
            Past covariates to be used for prediction. Should contain the same
            columns as specified in the DataModule's known_covariates.
        future_covariates : pd.DataFrame
            Future covariates to be used for prediction. Should contain the same
            columns as specified in the DataModule's known_covariates.
        past_target : pd.DataFrame | np.ndarray | pd.Series
            Past target values to be used for prediction context.
        context_length : int, optional
            Length of the context window. If None, uses the DataModule's ctxt_seq_len.
        prediction_length : int, optional
            Length of the prediction window. If None, uses the DataModule's tgt_seq_len.
        apply_transforms : bool, default=True
            Whether to apply the DataModule's transforms to the prediction data.

        Returns
        -------
        EncoderDecoderPredictDataset
            Configured prediction dataset ready for inference.

        Examples
        --------
        >>> # Configure DataModule
        >>> dm = EncoderDecoderDataModule(
        ...     known_covariates=["temp", "humidity"],
        ...     target_covariate="demand",
        ...     ctxt_seq_len=168,
        ...     tgt_seq_len=24,
        ...     train_df_paths="train.parquet"
        ... )
        >>> dm.setup("fit")  # Fit transforms
        >>>
        >>> # Create prediction dataset
        >>> pred_dataset = dm.create_prediction_dataset(
        ...     past_covariates=past_data[["temp", "humidity"]],
        ...     future_covariates=future_data[["temp", "humidity"]],
        ...     past_target=past_data["demand"]
        ... )
        >>>
        >>> # Use for prediction
        >>> sample = pred_dataset[0]
        >>> predictions = model.predict_step(sample, 0)
        """
        # Use DataModule settings as defaults
        context_length = context_length or self.ctxt_seq_len
        prediction_length = prediction_length or self.tgt_seq_len

        # Extract input columns from DataModule configuration
        input_columns = list(self.hparams["known_covariates"] or [])
        known_past_columns = list(self.hparams.get("known_past_covariates") or [])
        target_column = self.hparams["target_covariate"]

        # Use fitted transforms if available and requested
        transforms = self.transforms if apply_transforms else None

        # Ensure dtype is a torch.dtype, not string
        dtype = self.hparams.get("dtype", torch.float32)
        if isinstance(dtype, str):
            dtype = get_dtype(dtype)

        return EncoderDecoderPredictDataset(
            past_covariates=past_covariates,
            future_covariates=future_covariates,
            past_target=past_target,
            context_length=context_length,
            prediction_length=prediction_length,
            input_columns=input_columns,
            target_column=target_column,
            known_past_columns=known_past_columns if known_past_columns else None,
            transforms=transforms,
            time_format=self.hparams.get("time_format", "relative"),
            apply_transforms=apply_transforms,
            dtype=dtype,
        )
