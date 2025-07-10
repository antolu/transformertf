"""
Tests for DatasetFactory class.

This module tests the factory methods for creating different types of datasets
with consistent APIs and explicit parameters.
"""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from transformertf.data._dataset_factory import DatasetFactory
from transformertf.data.dataset import (
    EncoderDataset,
    EncoderDecoderDataset,
    TimeSeriesDataset,
)
from transformertf.data.transform import StandardScaler


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """Create a sample DataFrame for testing."""
    return pd.DataFrame({
        "__future_known_continuous_feature1": [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
        ],
        "__future_known_continuous_feature2": [
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            13.0,
        ],
        "__past_known_continuous_feature3": [
            0.5,
            1.5,
            2.5,
            3.5,
            4.5,
            5.5,
            6.5,
            7.5,
            8.5,
            9.5,
            10.5,
            11.5,
        ],
        "__target": [
            10.0,
            20.0,
            30.0,
            40.0,
            50.0,
            60.0,
            70.0,
            80.0,
            90.0,
            100.0,
            110.0,
            120.0,
        ],
        "__time__": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
    })


class TestDatasetFactory:
    """Test cases for DatasetFactory class."""

    @pytest.fixture
    def timeseries_dataframe(self) -> pd.DataFrame:
        """Create a sample DataFrame for time series testing."""
        return pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "feature2": [2.0, 3.0, 4.0, 5.0, 6.0],
            "__target": [10.0, 20.0, 30.0, 40.0, 50.0],
        })

    @pytest.fixture
    def sample_transforms(self) -> dict[str, torch.nn.Module]:
        """Create sample transforms for testing."""
        return {
            "feature1": StandardScaler(num_features_=1),
            "feature2": StandardScaler(num_features_=1),
            "__target": StandardScaler(num_features_=1),
        }

    def test_create_timeseries_dataset_basic(
        self, timeseries_dataframe: pd.DataFrame
    ) -> None:
        """Test creating basic time series dataset."""
        dataset = DatasetFactory.create_timeseries_dataset(
            data=timeseries_dataframe,
            seq_len=3,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset.seq_len == 3
        assert len(dataset) > 0

    def test_create_timeseries_dataset_with_list(
        self, timeseries_dataframe: pd.DataFrame
    ) -> None:
        """Test creating time series dataset with list of DataFrames."""
        data_list = [timeseries_dataframe, timeseries_dataframe]

        dataset = DatasetFactory.create_timeseries_dataset(
            data=data_list,
            seq_len=3,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset.seq_len == 3
        assert len(dataset) > 0

    def test_create_timeseries_dataset_predict_mode(
        self, timeseries_dataframe: pd.DataFrame
    ) -> None:
        """Test creating time series dataset in predict mode."""
        # Remove target column for prediction
        predict_df = timeseries_dataframe.drop(columns=["__target"])

        dataset = DatasetFactory.create_timeseries_dataset(
            data=predict_df,
            seq_len=3,
            stride=1,
            predict=True,
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset.seq_len == 3

    def test_create_timeseries_dataset_with_transforms(
        self,
        timeseries_dataframe: pd.DataFrame,
        sample_transforms: dict[str, torch.nn.Module],
    ) -> None:
        """Test creating time series dataset with transforms."""
        dataset = DatasetFactory.create_timeseries_dataset(
            data=timeseries_dataframe,
            seq_len=3,
            stride=1,
            predict=False,
            transforms=sample_transforms,
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset._transforms == sample_transforms

    def test_create_timeseries_dataset_randomized_length(
        self, timeseries_dataframe: pd.DataFrame
    ) -> None:
        """Test creating time series dataset with randomized lengths."""
        dataset = DatasetFactory.create_timeseries_dataset(
            data=timeseries_dataframe,
            seq_len=4,
            min_seq_len=2,
            randomize_seq_len=True,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset.seq_len == 4
        assert dataset._min_seq_len == 2
        assert dataset._randomize_seq_len is True

    def test_create_timeseries_dataset_custom_dtype(
        self, timeseries_dataframe: pd.DataFrame
    ) -> None:
        """Test creating time series dataset with custom dtype."""
        dataset = DatasetFactory.create_timeseries_dataset(
            data=timeseries_dataframe,
            seq_len=3,
            stride=1,
            predict=False,
            dtype="float64",
        )

        assert isinstance(dataset, TimeSeriesDataset)
        assert dataset._dtype == "float64"

    def test_create_encoder_decoder_dataset_basic(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating basic encoder-decoder dataset."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset.ctxt_seq_len == 3
        assert dataset.tgt_seq_len == 2
        assert len(dataset) > 0

    def test_create_encoder_decoder_dataset_with_list(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset with list of DataFrames."""
        data_list = [sample_dataframe, sample_dataframe]

        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=data_list,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert len(dataset) > 0

    def test_create_encoder_decoder_dataset_predict_mode(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset in predict mode."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=True,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset._predict is True

    def test_create_encoder_decoder_dataset_with_transforms(
        self,
        sample_dataframe: pd.DataFrame,
        sample_transforms: dict[str, torch.nn.Module],
    ) -> None:
        """Test creating encoder-decoder dataset with transforms."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
            transforms=sample_transforms,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset._transforms == sample_transforms

    def test_create_encoder_decoder_dataset_randomized_length(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset with randomized lengths."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=4,
            tgt_seq_len=3,
            min_ctx_seq_len=2,
            min_tgt_seq_len=1,
            randomize_seq_len=True,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset.ctxt_seq_len == 4
        assert dataset.tgt_seq_len == 3
        assert dataset._min_ctxt_seq_len == 2
        assert dataset._min_tgt_seq_len == 1
        assert dataset._randomize_seq_len is True

    def test_create_encoder_decoder_dataset_with_noise(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset with noise injection."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
            noise_std=0.1,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset._noise_std == 0.1

    def test_create_encoder_decoder_dataset_time_formats(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset with different time formats."""
        for time_format in ["relative", "absolute"]:
            dataset = DatasetFactory.create_encoder_decoder_dataset(
                data=sample_dataframe,
                ctx_seq_len=3,
                tgt_seq_len=2,
                stride=1,
                predict=False,
                time_format=time_format,
            )

            assert isinstance(dataset, EncoderDecoderDataset)
            assert dataset._time_format == time_format

    def test_create_encoder_decoder_dataset_add_target_to_past(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder-decoder dataset with add_target_to_past option."""
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
            add_target_to_past=False,
        )

        assert isinstance(dataset, EncoderDecoderDataset)
        assert dataset._add_target_to_past is False

    def test_create_encoder_dataset_basic(self, sample_dataframe: pd.DataFrame) -> None:
        """Test creating basic encoder dataset."""
        dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDataset)
        assert dataset.ctxt_seq_len == 3
        assert dataset.tgt_seq_len == 2
        assert len(dataset) > 0

    def test_create_encoder_dataset_with_list(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder dataset with list of DataFrames."""
        data_list = [sample_dataframe, sample_dataframe]

        dataset = DatasetFactory.create_encoder_dataset(
            data=data_list,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDataset)
        assert len(dataset) > 0

    def test_create_encoder_dataset_predict_mode(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder dataset in predict mode."""
        dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=True,
        )

        assert isinstance(dataset, EncoderDataset)
        assert dataset._predict is True

    def test_create_encoder_dataset_with_transforms(
        self,
        sample_dataframe: pd.DataFrame,
        sample_transforms: dict[str, torch.nn.Module],
    ) -> None:
        """Test creating encoder dataset with transforms."""
        dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
            transforms=sample_transforms,
        )

        assert isinstance(dataset, EncoderDataset)
        assert dataset._transforms == sample_transforms

    def test_create_encoder_dataset_randomized_length(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder dataset with randomized lengths."""
        dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=4,
            tgt_seq_len=3,
            min_ctx_seq_len=2,
            min_tgt_seq_len=1,
            randomize_seq_len=True,
            stride=1,
            predict=False,
        )

        assert isinstance(dataset, EncoderDataset)
        assert dataset.ctxt_seq_len == 4
        assert dataset.tgt_seq_len == 3
        assert dataset._min_ctxt_seq_len == 2
        assert dataset._min_tgt_seq_len == 1
        assert dataset._randomize_seq_len is True

    def test_create_encoder_dataset_custom_dtype(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test creating encoder dataset with custom dtype."""
        dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
            dtype="float64",
        )

        assert isinstance(dataset, EncoderDataset)
        assert dataset._dtype == "float64"


class TestDatasetFactoryExtractColumns:
    """Test cases for column extraction utility."""

    def test_extract_columns_with_prefix(self) -> None:
        """Test extracting columns with specific prefix."""
        from transformertf.data._dataset_factory import _extract_columns

        df = pd.DataFrame({
            "__future_known_continuous_feature1": [1, 2, 3],
            "__future_known_continuous_feature2": [4, 5, 6],
            "__past_known_continuous_feature3": [7, 8, 9],
            "__target": [10, 11, 12],
        })

        future_cols = _extract_columns(df, "__future_known_continuous_")
        assert list(future_cols.columns) == [
            "__future_known_continuous_feature1",
            "__future_known_continuous_feature2",
        ]

        past_cols = _extract_columns(df, "__past_known_continuous_")
        assert list(past_cols.columns) == ["__past_known_continuous_feature3"]

        target_cols = _extract_columns(df, "__target")
        assert list(target_cols.columns) == ["__target"]

    def test_extract_columns_no_match(self) -> None:
        """Test extracting columns when no match found."""
        from transformertf.data._dataset_factory import _extract_columns

        df = pd.DataFrame({
            "feature1": [1, 2, 3],
            "feature2": [4, 5, 6],
        })

        result = _extract_columns(df, "__nonexistent_")
        assert result.empty
        assert len(result.columns) == 0

    def test_extract_columns_empty_dataframe(self) -> None:
        """Test extracting columns from empty DataFrame."""
        from transformertf.data._dataset_factory import _extract_columns

        df = pd.DataFrame()
        result = _extract_columns(df, "__any_prefix_")
        assert result.empty


class TestDatasetFactoryIntegration:
    """Integration tests for DatasetFactory."""

    def test_factory_methods_return_correct_types(
        self, sample_dataframe: pd.DataFrame
    ) -> None:
        """Test that factory methods return correct dataset types."""
        # Test all factory methods return correct types
        timeseries_dataset = DatasetFactory.create_timeseries_dataset(
            data=sample_dataframe,
            seq_len=3,
            stride=1,
            predict=False,
        )

        encoder_decoder_dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        encoder_dataset = DatasetFactory.create_encoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,
            tgt_seq_len=2,
            stride=1,
            predict=False,
        )

        assert isinstance(timeseries_dataset, TimeSeriesDataset)
        assert isinstance(encoder_decoder_dataset, EncoderDecoderDataset)
        assert isinstance(encoder_dataset, EncoderDataset)

        # Ensure they're different types
        assert type(timeseries_dataset) != type(encoder_decoder_dataset)
        assert type(encoder_decoder_dataset) != type(encoder_dataset)
        assert type(encoder_dataset) != type(timeseries_dataset)

    def test_factory_parameter_passing(self, sample_dataframe: pd.DataFrame) -> None:
        """Test that factory methods properly pass parameters to datasets."""
        # Create dataset with specific parameters (reduced window sizes for stride=2)
        dataset = DatasetFactory.create_encoder_decoder_dataset(
            data=sample_dataframe,
            ctx_seq_len=3,  # Reduced from 5 to fit strided data (12 -> 6 points with stride=2)
            tgt_seq_len=2,  # Reduced from 3 to fit strided data
            min_ctx_seq_len=2,
            min_tgt_seq_len=1,
            randomize_seq_len=True,
            stride=2,
            predict=False,
            time_format="relative",
            noise_std=0.05,
            add_target_to_past=False,
            dtype="float64",
        )

        # Verify parameters were passed correctly
        assert dataset.ctxt_seq_len == 3
        assert dataset.tgt_seq_len == 2
        assert dataset._min_ctxt_seq_len == 2
        assert dataset._min_tgt_seq_len == 1
        assert dataset._randomize_seq_len is True
        assert dataset._stride == 2
        assert dataset._predict is False
        assert dataset._time_format == "relative"
        assert dataset._noise_std == 0.05
        assert dataset._add_target_to_past is False
        assert dataset._dtype == "float64"

    def test_factory_with_empty_dataframe(self) -> None:
        """Test factory methods with empty DataFrame."""
        empty_df = pd.DataFrame()

        # Empty DataFrame should raise ValueError since window size cannot exceed data length
        with pytest.raises(ValueError, match="Input window size.*must be less than"):
            DatasetFactory.create_timeseries_dataset(
                data=empty_df,
                seq_len=3,
                stride=1,
                predict=True,
            )
