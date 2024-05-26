from __future__ import annotations

import typing

import pytest

from transformertf.models.phylstm import PhyLSTM, PhyLSTMDataModule


@pytest.fixture(scope="module")
def phylstm_datamodule_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df": df_path,
        "val_df": df_path,
        "num_workers": 0,
        "target_depends_on": current_key,
        "input_columns": current_key,
        "target_column": field_key,
        "seq_len": 100,
    }


@pytest.fixture(scope="module")
def phylstm_module_config() -> dict[str, typing.Any]:
    return {
        "seq_len": 100,
        "num_layers": 1,
        "hidden_dim": 10,
        "hidden_dim_fc": 16,
    }


@pytest.fixture()
def phylstm_module(phylstm_module_config: dict[str, typing.Any]) -> PhyLSTM:
    return PhyLSTM(**phylstm_module_config)


@pytest.fixture()
def phylstm_datamodule(
    phylstm_datamodule_config: dict[str, typing.Any],
) -> PhyLSTMDataModule:
    return PhyLSTMDataModule(**phylstm_datamodule_config)
