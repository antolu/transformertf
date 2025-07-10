"""
Transform Builder for managing transform creation in a fluent, composable way.

This module provides the TransformBuilder class which uses the builder pattern
to create and validate transforms for data modules. It separates transform
creation logic from data module initialization, improving maintainability
and testability.

:author: Anton Lu (anton.lu@cern.ch)
"""

from __future__ import annotations

import logging
import typing
from typing import Literal

import torch

from ._covariates import TIME_PREFIX
from .transform import (
    BaseTransform,
    DeltaTransform,
    MaxScaler,
    StandardScaler,
    TransformCollection,
)

__all__ = ["CommonTransformPatterns", "TransformBuilder"]

log = logging.getLogger(__name__)


class TransformBuilder:
    """
    Builder for creating transform collections with validation.

    This class provides a fluent API for building transform collections
    used by data modules. It handles transform creation, dependency
    tracking, and validation in a composable way.

    Examples
    --------
    Basic usage:

    >>> builder = TransformBuilder()
    >>> transforms = (builder
    ...     .add_covariate_transforms(["temp", "pressure"], {}, normalize=True)
    ...     .add_target_transforms("demand", {}, normalize=True)
    ...     .build())

    With time transforms:

    >>> transforms = (TransformBuilder()
    ...     .add_covariate_transforms(["feature1"], {}, normalize=True)
    ...     .add_target_transforms("target", {}, normalize=False)
    ...     .add_time_transforms("relative")
    ...     .build())

    With custom validation:

    >>> def validate_required_pairs(transforms, deps):
    ...     if "voltage" in transforms and "current" not in transforms:
    ...         raise ValueError("Voltage requires current measurement")
    ...
    >>> transforms = (TransformBuilder()
    ...     .add_validation_rule(validate_required_pairs)
    ...     .add_covariate_transforms(["voltage", "current"], {})
    ...     .build())
    """

    def __init__(self):
        self._transforms: dict[str, list[BaseTransform]] = {}
        self._target_dependencies: dict[str, str] = {}
        self._validation_rules: list[typing.Callable] = []

    def add_covariate_transforms(
        self,
        covariate_names: list[str],
        extra_transforms: dict[str, list[BaseTransform]],
        normalize: bool = True,
        normalize_type: type[BaseTransform] = StandardScaler,
    ) -> TransformBuilder:
        """
        Add transforms for input covariates.

        Parameters
        ----------
        covariate_names : list[str]
            Names of covariates to add transforms for.
        extra_transforms : dict[str, list[BaseTransform]]
            Extra transforms to apply before normalization.
        normalize : bool, optional
            Whether to add normalization transform. Default is True.
        normalize_type : type[BaseTransform], optional
            Type of normalization transform to use. Default is StandardScaler.

        Returns
        -------
        TransformBuilder
            Self for method chaining.
        """
        for name in covariate_names:
            transforms = []

            # Add extra transforms first (applied before normalization)
            if name in extra_transforms:
                transforms.extend(extra_transforms[name])

            # Add normalization
            if normalize:
                transforms.append(normalize_type(num_features_=1))

            self._transforms[name] = transforms

        return self

    def add_target_transforms(
        self,
        target_name: str,
        extra_transforms: dict[str, list[BaseTransform]],
        normalize: bool = True,
        depends_on: str | None = None,
        normalize_type: type[BaseTransform] = StandardScaler,
    ) -> TransformBuilder:
        """
        Add transforms for target variable.

        Parameters
        ----------
        target_name : str
            Name of target variable.
        extra_transforms : dict[str, list[BaseTransform]]
            Extra transforms to apply before normalization.
        normalize : bool, optional
            Whether to add normalization transform. Default is True.
        depends_on : str | None, optional
            Name of covariate this target depends on for XY transforms.
        normalize_type : type[BaseTransform], optional
            Type of normalization transform to use. Default is StandardScaler.

        Returns
        -------
        TransformBuilder
            Self for method chaining.
        """
        transforms = []

        if target_name in extra_transforms:
            transforms.extend(extra_transforms[target_name])

        if normalize:
            transforms.append(normalize_type(num_features_=1))

        self._transforms[target_name] = transforms

        # Track dependencies for validation
        if depends_on:
            self._target_dependencies[target_name] = depends_on

        return self

    def add_time_transforms(
        self,
        time_format: Literal["relative", "absolute", "relative_legacy"],
        time_column: str = TIME_PREFIX,
        extra_transforms: list[BaseTransform] | None = None,
    ) -> TransformBuilder:
        """
        Add time-specific transforms.

        Parameters
        ----------
        time_format : {"relative", "absolute", "relative_legacy"}
            Format for time transforms:
            - "relative": Delta + MaxScaler
            - "absolute": MaxScaler only
            - "relative_legacy": Delta + StandardScaler
        time_column : str, optional
            Name of time column. Default is TIME_PREFIX.
        extra_transforms : list[BaseTransform] | None, optional
            Additional transforms to apply after format transforms.

        Returns
        -------
        TransformBuilder
            Self for method chaining.
        """
        transforms = []

        # Add format-specific transforms
        if time_format == "relative":
            transforms.extend([DeltaTransform(), MaxScaler(num_features_=1)])
        elif time_format == "relative_legacy":
            transforms.extend([DeltaTransform(), StandardScaler(num_features_=1)])
        elif time_format == "absolute":
            transforms.append(MaxScaler(num_features_=1))
        else:
            msg = (
                f"Unknown time format '{time_format}'. "
                "Expected 'relative', 'absolute', or 'relative_legacy'."
            )
            raise ValueError(msg)

        # Add extra time transforms
        if extra_transforms:
            transforms.extend(extra_transforms)

        if transforms:  # Only add if we have transforms
            self._transforms[time_column] = transforms

        return self

    def add_conditional_transforms(
        self, condition: bool, transforms: dict[str, list[BaseTransform]]
    ) -> TransformBuilder:
        """
        Add transforms conditionally.

        Parameters
        ----------
        condition : bool
            Whether to add the transforms.
        transforms : dict[str, list[BaseTransform]]
            Transforms to add if condition is True.

        Returns
        -------
        TransformBuilder
            Self for method chaining.
        """
        if condition:
            for name, transform_list in transforms.items():
                self._transforms.setdefault(name, []).extend(transform_list)
        return self

    def add_validation_rule(
        self,
        rule: typing.Callable[[dict[str, list[BaseTransform]], dict[str, str]], None],
    ) -> TransformBuilder:
        """
        Add custom validation rule.

        Parameters
        ----------
        rule : callable
            Validation function that takes (transforms, dependencies) and
            raises ValueError if validation fails.

        Returns
        -------
        TransformBuilder
            Self for method chaining.
        """
        self._validation_rules.append(rule)
        return self

    def build(self) -> torch.nn.ModuleDict[str, TransformCollection]:
        """
        Build transforms with validation.

        Returns
        -------
        torch.nn.ModuleDict[str, TransformCollection]
            Built and validated transform collections.

        Raises
        ------
        ValueError
            If validation fails.
        """
        try:
            # Run all validation rules
            for rule in self._validation_rules:
                rule(self._transforms, self._target_dependencies)

            # Default validation
            self._validate_target_dependencies()

            # Build the final collection
            return torch.nn.ModuleDict({
                name: (
                    transforms
                    if isinstance(transforms, TransformCollection)
                    else TransformCollection(transforms)
                )
                for name, transforms in self._transforms.items()
            })

        except Exception as e:
            # Provide context about what was being built
            context = (
                f"Building transforms with covariates: {list(self._transforms.keys())}"
            )
            msg = f"{context}. Error: {e}"
            raise ValueError(msg) from e

    def _validate_target_dependencies(self) -> None:
        """Validate target transform dependencies."""
        for target_name, depends_on in self._target_dependencies.items():
            if target_name not in self._transforms:
                continue

            transform_collection = TransformCollection(self._transforms[target_name])

            if transform_collection.transform_type != BaseTransform.TransformType.XY:
                msg = (
                    f"Target '{target_name}' depends on '{depends_on}' but transform "
                    f"doesn't support XY type. Got: {transform_collection.transform_type}"
                )
                raise ValueError(msg)


class CommonTransformPatterns:
    """
    Common transform patterns for reuse across data modules.

    This class provides static methods for common transform configurations
    that can be applied to TransformBuilder instances.
    """

    @staticmethod
    def standard_normalization(
        builder: TransformBuilder,
        covariate_names: list[str],
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
    ) -> TransformBuilder:
        """
        Add standard normalization (StandardScaler) for covariates.

        Parameters
        ----------
        builder : TransformBuilder
            Builder to add transforms to.
        covariate_names : list[str]
            Names of covariates to normalize.
        extra_transforms : dict[str, list[BaseTransform]] | None, optional
            Additional transforms to apply before normalization.

        Returns
        -------
        TransformBuilder
            Builder with standard normalization added.
        """
        return builder.add_covariate_transforms(
            covariate_names,
            extra_transforms or {},
            normalize=True,
            normalize_type=StandardScaler,
        )

    @staticmethod
    def max_normalization(
        builder: TransformBuilder,
        covariate_names: list[str],
        extra_transforms: dict[str, list[BaseTransform]] | None = None,
    ) -> TransformBuilder:
        """
        Add max normalization (MaxScaler) for covariates.

        Parameters
        ----------
        builder : TransformBuilder
            Builder to add transforms to.
        covariate_names : list[str]
            Names of covariates to normalize.
        extra_transforms : dict[str, list[BaseTransform]] | None, optional
            Additional transforms to apply before normalization.

        Returns
        -------
        TransformBuilder
            Builder with max normalization added.
        """
        return builder.add_covariate_transforms(
            covariate_names,
            extra_transforms or {},
            normalize=True,
            normalize_type=MaxScaler,
        )

    @staticmethod
    def relative_time(
        builder: TransformBuilder,
        time_column: str = TIME_PREFIX,
        extra_transforms: list[BaseTransform] | None = None,
    ) -> TransformBuilder:
        """
        Add relative time transforms (Delta + MaxScaler).

        Parameters
        ----------
        builder : TransformBuilder
            Builder to add transforms to.
        time_column : str, optional
            Time column name. Default is TIME_PREFIX.
        extra_transforms : list[BaseTransform] | None, optional
            Additional transforms to apply.

        Returns
        -------
        TransformBuilder
            Builder with relative time transforms added.
        """
        return builder.add_time_transforms(
            "relative", time_column=time_column, extra_transforms=extra_transforms
        )

    @staticmethod
    def absolute_time(
        builder: TransformBuilder,
        time_column: str = TIME_PREFIX,
        extra_transforms: list[BaseTransform] | None = None,
    ) -> TransformBuilder:
        """
        Add absolute time transforms (MaxScaler only).

        Parameters
        ----------
        builder : TransformBuilder
            Builder to add transforms to.
        time_column : str, optional
            Time column name. Default is TIME_PREFIX.
        extra_transforms : list[BaseTransform] | None, optional
            Additional transforms to apply.

        Returns
        -------
        TransformBuilder
            Builder with absolute time transforms added.
        """
        return builder.add_time_transforms(
            "absolute", time_column=time_column, extra_transforms=extra_transforms
        )
