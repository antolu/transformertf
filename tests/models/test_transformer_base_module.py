from __future__ import annotations

import typing

import pytest

from transformertf.models import TransformerModuleBase


@pytest.fixture(scope="module")
def transformer_base_config(
    df_path: str, current_key: str, field_key: str
) -> dict[str, typing.Any]:
    return {
        "train_df": df_path,
        "val_df": df_path,
        "input_columns": [current_key],
        "target_column": field_key,
    }


def test_create_base_transformer_module_from_config() -> None:
    base_module = TransformerModuleBase()
    assert base_module is not None
