from __future__ import annotations

from transformerft.data import DataModuleBase


def test_datamodule_base_create() -> None:
    dm = DataModuleBase()
    assert dm is not None
