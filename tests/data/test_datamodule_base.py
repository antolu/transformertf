from __future__ import annotations

from transformertf.data import DataModuleBase


def test_datamodule_base_create() -> None:
    dm = DataModuleBase(input_columns=["a"], target_columns=["b"])
    assert dm is not None
