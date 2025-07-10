"""
Tests for TransformBuilder class and CommonTransformPatterns.

This module tests the fluent API for building transform collections,
including validation, error handling, and common patterns.
"""

from __future__ import annotations

import pytest
import torch

from transformertf.data._transform_builder import (
    CommonTransformPatterns,
    TransformBuilder,
)
from transformertf.data.transform import (
    BaseTransform,
    DeltaTransform,
    LogTransform,
    MaxScaler,
    StandardScaler,
    TransformCollection,
)


class TestTransformBuilder:
    """Test cases for TransformBuilder class."""

    def test_init(self) -> None:
        """Test TransformBuilder initialization."""
        builder = TransformBuilder()
        assert builder._transforms == {}
        assert builder._target_dependencies == {}
        assert builder._validation_rules == []

    def test_add_covariate_transforms_basic(self) -> None:
        """Test adding basic covariate transforms."""
        builder = TransformBuilder()
        result = builder.add_covariate_transforms(
            covariate_names=["temp", "pressure"],
            extra_transforms={},
            normalize=True,
        )

        # Should return self for method chaining
        assert result is builder

        # Should have transforms for both covariates
        assert "temp" in builder._transforms
        assert "pressure" in builder._transforms

        # Each should have a StandardScaler
        assert len(builder._transforms["temp"]) == 1
        assert isinstance(builder._transforms["temp"][0], StandardScaler)
        assert len(builder._transforms["pressure"]) == 1
        assert isinstance(builder._transforms["pressure"][0], StandardScaler)

    def test_add_covariate_transforms_with_extra(self) -> None:
        """Test adding covariate transforms with extra transforms."""
        extra_transforms = {
            "temp": [LogTransform()],
            "pressure": [DeltaTransform()],
        }

        builder = TransformBuilder()
        builder.add_covariate_transforms(
            covariate_names=["temp", "pressure"],
            extra_transforms=extra_transforms,
            normalize=True,
        )

        # Temp should have LogTransform + StandardScaler
        assert len(builder._transforms["temp"]) == 2
        assert isinstance(builder._transforms["temp"][0], LogTransform)
        assert isinstance(builder._transforms["temp"][1], StandardScaler)

        # Pressure should have DeltaTransform + StandardScaler
        assert len(builder._transforms["pressure"]) == 2
        assert isinstance(builder._transforms["pressure"][0], DeltaTransform)
        assert isinstance(builder._transforms["pressure"][1], StandardScaler)

    def test_add_covariate_transforms_no_normalize(self) -> None:
        """Test adding covariate transforms without normalization."""
        builder = TransformBuilder()
        builder.add_covariate_transforms(
            covariate_names=["temp"],
            extra_transforms={},
            normalize=False,
        )

        # Should have no transforms
        assert len(builder._transforms["temp"]) == 0

    def test_add_covariate_transforms_custom_normalizer(self) -> None:
        """Test adding covariate transforms with custom normalizer."""
        builder = TransformBuilder()
        builder.add_covariate_transforms(
            covariate_names=["temp"],
            extra_transforms={},
            normalize=True,
            normalize_type=MaxScaler,
        )

        # Should have MaxScaler instead of StandardScaler
        assert len(builder._transforms["temp"]) == 1
        assert isinstance(builder._transforms["temp"][0], MaxScaler)

    def test_add_target_transforms_basic(self) -> None:
        """Test adding basic target transforms."""
        builder = TransformBuilder()
        result = builder.add_target_transforms(
            target_name="demand",
            extra_transforms={},
            normalize=True,
        )

        # Should return self for method chaining
        assert result is builder

        # Should have transforms for target
        assert "demand" in builder._transforms
        assert len(builder._transforms["demand"]) == 1
        assert isinstance(builder._transforms["demand"][0], StandardScaler)

    def test_add_target_transforms_with_dependency(self) -> None:
        """Test adding target transforms with dependency."""
        builder = TransformBuilder()
        builder.add_target_transforms(
            target_name="demand",
            extra_transforms={},
            normalize=True,
            depends_on="supply",
        )

        # Should track dependency
        assert builder._target_dependencies["demand"] == "supply"

    def test_add_time_transforms_relative(self) -> None:
        """Test adding relative time transforms."""
        builder = TransformBuilder()
        result = builder.add_time_transforms(
            time_format="relative",
            time_column="__time__",
        )

        # Should return self for method chaining
        assert result is builder

        # Should have DeltaTransform + MaxScaler
        assert "__time__" in builder._transforms
        assert len(builder._transforms["__time__"]) == 2
        assert isinstance(builder._transforms["__time__"][0], DeltaTransform)
        assert isinstance(builder._transforms["__time__"][1], MaxScaler)

    def test_add_time_transforms_absolute(self) -> None:
        """Test adding absolute time transforms."""
        builder = TransformBuilder()
        builder.add_time_transforms(
            time_format="absolute",
            time_column="__time__",
        )

        # Should have only MaxScaler
        assert "__time__" in builder._transforms
        assert len(builder._transforms["__time__"]) == 1
        assert isinstance(builder._transforms["__time__"][0], MaxScaler)

    def test_add_time_transforms_relative_legacy(self) -> None:
        """Test adding relative legacy time transforms."""
        builder = TransformBuilder()
        builder.add_time_transforms(
            time_format="relative_legacy",
            time_column="__time__",
        )

        # Should have DeltaTransform + StandardScaler
        assert "__time__" in builder._transforms
        assert len(builder._transforms["__time__"]) == 2
        assert isinstance(builder._transforms["__time__"][0], DeltaTransform)
        assert isinstance(builder._transforms["__time__"][1], StandardScaler)

    def test_add_time_transforms_invalid_format(self) -> None:
        """Test adding time transforms with invalid format."""
        builder = TransformBuilder()

        with pytest.raises(ValueError, match="Unknown time format"):
            builder.add_time_transforms(
                time_format="invalid",  # type: ignore
                time_column="__time__",
            )

    def test_add_time_transforms_with_extra(self) -> None:
        """Test adding time transforms with extra transforms."""
        extra_transforms = [LogTransform()]

        builder = TransformBuilder()
        builder.add_time_transforms(
            time_format="relative",
            time_column="__time__",
            extra_transforms=extra_transforms,
        )

        # Should have DeltaTransform + MaxScaler + LogTransform
        assert len(builder._transforms["__time__"]) == 3
        assert isinstance(builder._transforms["__time__"][0], DeltaTransform)
        assert isinstance(builder._transforms["__time__"][1], MaxScaler)
        assert isinstance(builder._transforms["__time__"][2], LogTransform)

    def test_add_conditional_transforms_true(self) -> None:
        """Test adding conditional transforms when condition is True."""
        builder = TransformBuilder()
        transforms = {"temp": [LogTransform()]}

        result = builder.add_conditional_transforms(
            condition=True,
            transforms=transforms,
        )

        # Should return self for method chaining
        assert result is builder

        # Should have added transforms
        assert "temp" in builder._transforms
        assert len(builder._transforms["temp"]) == 1
        assert isinstance(builder._transforms["temp"][0], LogTransform)

    def test_add_conditional_transforms_false(self) -> None:
        """Test adding conditional transforms when condition is False."""
        builder = TransformBuilder()
        transforms = {"temp": [LogTransform()]}

        builder.add_conditional_transforms(
            condition=False,
            transforms=transforms,
        )

        # Should not have added transforms
        assert builder._transforms == {}

    def test_add_conditional_transforms_extend(self) -> None:
        """Test that conditional transforms extend existing transforms."""
        builder = TransformBuilder()
        # Add initial transform
        builder.add_covariate_transforms(
            covariate_names=["temp"],
            extra_transforms={},
            normalize=True,
        )

        # Add conditional transform
        builder.add_conditional_transforms(
            condition=True,
            transforms={"temp": [LogTransform()]},
        )

        # Should have both transforms
        assert len(builder._transforms["temp"]) == 2
        assert isinstance(builder._transforms["temp"][0], StandardScaler)
        assert isinstance(builder._transforms["temp"][1], LogTransform)

    def test_add_validation_rule(self) -> None:
        """Test adding custom validation rule."""

        def custom_rule(transforms, dependencies):
            if "temp" in transforms and "pressure" not in transforms:
                msg = "Temperature requires pressure"
                raise ValueError(msg)

        builder = TransformBuilder()
        result = builder.add_validation_rule(custom_rule)

        # Should return self for method chaining
        assert result is builder

        # Should have added rule
        assert len(builder._validation_rules) == 1
        assert builder._validation_rules[0] is custom_rule

    def test_build_basic(self) -> None:
        """Test building transforms."""
        builder = TransformBuilder()
        builder.add_covariate_transforms(
            covariate_names=["temp", "pressure"],
            extra_transforms={},
            normalize=True,
        )

        result = builder.build()

        # Should return ModuleDict
        assert isinstance(result, torch.nn.ModuleDict)
        assert "temp" in result
        assert "pressure" in result

        # Each should be a TransformCollection
        assert isinstance(result["temp"], TransformCollection)
        assert isinstance(result["pressure"], TransformCollection)

    def test_build_with_validation_error(self) -> None:
        """Test building transforms with validation error."""

        def failing_rule(transforms, dependencies):
            msg = "Custom validation error"
            raise ValueError(msg)

        builder = TransformBuilder()
        builder.add_validation_rule(failing_rule)
        builder.add_covariate_transforms(
            covariate_names=["temp"],
            extra_transforms={},
            normalize=True,
        )

        with pytest.raises(
            ValueError, match="Building transforms.*Custom validation error"
        ):
            builder.build()

    def test_build_with_target_dependency_validation(self) -> None:
        """Test building transforms with target dependency validation."""

        # Create a transform that doesn't support XY type
        class NonXYTransform(BaseTransform):
            _transform_type = BaseTransform.TransformType.X

            def fit(self, x, y=None):
                return self

            def transform(self, x, y=None):
                return x

        builder = TransformBuilder()
        builder.add_target_transforms(
            target_name="demand",
            extra_transforms={"demand": [NonXYTransform()]},
            normalize=False,
            depends_on="supply",
        )

        with pytest.raises(ValueError, match="doesn't support XY type"):
            builder.build()

    def test_method_chaining(self) -> None:
        """Test that all methods support method chaining."""
        builder = TransformBuilder()

        # Should be able to chain all methods
        result = (
            builder.add_covariate_transforms(["temp"], {}, normalize=True)
            .add_target_transforms("demand", {}, normalize=True)
            .add_time_transforms("relative")
            .add_conditional_transforms(True, {"extra": [LogTransform()]})
            .add_validation_rule(lambda t, d: None)
            .build()
        )

        assert isinstance(result, torch.nn.ModuleDict)
        assert "temp" in result
        assert "demand" in result
        assert "__time__" in result
        assert "extra" in result


class TestCommonTransformPatterns:
    """Test cases for CommonTransformPatterns class."""

    def test_standard_normalization(self) -> None:
        """Test standard normalization pattern."""
        builder = TransformBuilder()
        result = CommonTransformPatterns.standard_normalization(
            builder,
            covariate_names=["temp", "pressure"],
        )

        # Should return builder for chaining
        assert result is builder

        # Should have standard normalizers
        transforms = builder.build()
        assert "temp" in transforms
        assert "pressure" in transforms

        # Should use StandardScaler
        assert isinstance(transforms["temp"][0], StandardScaler)
        assert isinstance(transforms["pressure"][0], StandardScaler)

    def test_standard_normalization_with_extra(self) -> None:
        """Test standard normalization with extra transforms."""
        builder = TransformBuilder()
        extra_transforms = {"temp": [LogTransform()]}

        CommonTransformPatterns.standard_normalization(
            builder,
            covariate_names=["temp"],
            extra_transforms=extra_transforms,
        )

        transforms = builder.build()
        # Should have LogTransform + StandardScaler
        assert len(transforms["temp"]) == 2
        assert isinstance(transforms["temp"][0], LogTransform)
        assert isinstance(transforms["temp"][1], StandardScaler)

    def test_max_normalization(self) -> None:
        """Test max normalization pattern."""
        builder = TransformBuilder()
        result = CommonTransformPatterns.max_normalization(
            builder,
            covariate_names=["temp", "pressure"],
        )

        # Should return builder for chaining
        assert result is builder

        # Should have max normalizers
        transforms = builder.build()
        assert "temp" in transforms
        assert "pressure" in transforms

        # Should use MaxScaler
        assert isinstance(transforms["temp"][0], MaxScaler)
        assert isinstance(transforms["pressure"][0], MaxScaler)

    def test_relative_time(self) -> None:
        """Test relative time pattern."""
        builder = TransformBuilder()
        result = CommonTransformPatterns.relative_time(builder)

        # Should return builder for chaining
        assert result is builder

        # Should have relative time transforms
        transforms = builder.build()
        assert "__time__" in transforms

        # Should have DeltaTransform + MaxScaler
        assert len(transforms["__time__"]) == 2
        assert isinstance(transforms["__time__"][0], DeltaTransform)
        assert isinstance(transforms["__time__"][1], MaxScaler)

    def test_relative_time_custom_column(self) -> None:
        """Test relative time with custom column name."""
        builder = TransformBuilder()
        CommonTransformPatterns.relative_time(
            builder,
            time_column="custom_time",
        )

        transforms = builder.build()
        assert "custom_time" in transforms
        assert "__time__" not in transforms

    def test_relative_time_with_extra(self) -> None:
        """Test relative time with extra transforms."""
        builder = TransformBuilder()
        extra_transforms = [LogTransform()]

        CommonTransformPatterns.relative_time(
            builder,
            extra_transforms=extra_transforms,
        )

        transforms = builder.build()
        # Should have DeltaTransform + MaxScaler + LogTransform
        assert len(transforms["__time__"]) == 3
        assert isinstance(transforms["__time__"][0], DeltaTransform)
        assert isinstance(transforms["__time__"][1], MaxScaler)
        assert isinstance(transforms["__time__"][2], LogTransform)

    def test_absolute_time(self) -> None:
        """Test absolute time pattern."""
        builder = TransformBuilder()
        result = CommonTransformPatterns.absolute_time(builder)

        # Should return builder for chaining
        assert result is builder

        # Should have absolute time transforms
        transforms = builder.build()
        assert "__time__" in transforms

        # Should have only MaxScaler
        assert len(transforms["__time__"]) == 1
        assert isinstance(transforms["__time__"][0], MaxScaler)

    def test_pattern_chaining(self) -> None:
        """Test that patterns can be chained together."""
        builder = TransformBuilder()

        # Should be able to chain patterns
        transforms = (
            CommonTransformPatterns.standard_normalization(
                builder, ["temp", "pressure"]
            )
            .add_covariate_transforms(
                ["volume"], {}, normalize=True, normalize_type=MaxScaler
            )
            .add_time_transforms("relative")
            .build()
        )

        assert "temp" in transforms
        assert "pressure" in transforms
        assert "volume" in transforms
        assert "__time__" in transforms

        # Should have correct normalizers
        assert isinstance(transforms["temp"][0], StandardScaler)
        assert isinstance(transforms["volume"][0], MaxScaler)
        assert isinstance(transforms["__time__"][0], DeltaTransform)


class TestTransformBuilderIntegration:
    """Integration tests for TransformBuilder."""

    def test_complex_transform_setup(self) -> None:
        """Test building a complex transform setup."""

        def validate_sensor_pairs(transforms, dependencies):
            sensors = [k for k in transforms if k.startswith("sensor")]
            if len(sensors) % 2 != 0:
                msg = "Sensors must be paired"
                raise ValueError(msg)

        builder = TransformBuilder()
        extra_transforms = {
            "sensor1": [DeltaTransform()],
            "sensor2": [LogTransform()],
            "target": [LogTransform()],
        }

        transforms = (
            builder.add_validation_rule(validate_sensor_pairs)
            .add_covariate_transforms(
                ["sensor1", "sensor2"],
                extra_transforms,
                normalize=True,
            )
            .add_target_transforms(
                "target",
                extra_transforms,
                normalize=True,
                # Remove depends_on to avoid XY validation issues in test
            )
            .add_time_transforms("relative")
            .build()
        )

        # Should have all components
        assert "sensor1" in transforms
        assert "sensor2" in transforms
        assert "target" in transforms
        assert "__time__" in transforms

        # Should have correct transform chains
        assert len(transforms["sensor1"]) == 2  # Delta + Standard
        assert len(transforms["sensor2"]) == 2  # Log + Standard
        assert len(transforms["target"]) == 2  # Log + Standard
        assert len(transforms["__time__"]) == 2  # Delta + Max

    def test_empty_builder(self) -> None:
        """Test building with empty builder."""
        builder = TransformBuilder()
        transforms = builder.build()

        assert isinstance(transforms, torch.nn.ModuleDict)
        assert len(transforms) == 0

    def test_builder_reuse(self) -> None:
        """Test that builder can be reused after build."""
        builder = TransformBuilder()
        builder.add_covariate_transforms(["temp"], {}, normalize=True)

        # Build first time
        transforms1 = builder.build()
        assert "temp" in transforms1

        # Add more transforms and build again
        builder.add_covariate_transforms(["pressure"], {}, normalize=True)
        transforms2 = builder.build()

        # Should have both transforms
        assert "temp" in transforms2
        assert "pressure" in transforms2
