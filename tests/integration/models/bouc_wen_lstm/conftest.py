from __future__ import annotations

import typing

import pytest

from transformertf.data import TimeSeriesDataModule
from transformertf.models.bwlstm import BWLSTM3


@pytest.fixture(scope="module")
def bouc_wen_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df_paths": df_path,
        "val_df_paths": df_path,
        "num_workers": 0,
        "target_depends_on": current_key,
        "known_covariates": current_key,
        "target_covariate": field_key,
        "seq_len": 100,
    }


@pytest.fixture(scope="module")
def bouc_wen_module_config() -> dict[str, typing.Any]:
    return {
        "seq_len": 100,
        "num_layers": 1,
        "hidden_dim": 10,
        "hidden_dim_fc": 16,
    }


@pytest.fixture()
def bouc_wen_module(bouc_wen_module_config: dict[str, typing.Any]) -> BWLSTM3:
    return BWLSTM3(**bouc_wen_module_config)


@pytest.fixture()
def bouc_wen_datamodule(
    bouc_wen_datamodule_config: dict[str, typing.Any],
) -> TimeSeriesDataModule:
    return TimeSeriesDataModule(**bouc_wen_datamodule_config)
