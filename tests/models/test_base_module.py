from __future__ import annotations

from transformertf.models import LightningModuleBase


def test_create_base_module_from_config() -> None:
    base_module = LightningModuleBase()
    assert base_module is not None
