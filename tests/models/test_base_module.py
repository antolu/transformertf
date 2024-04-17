from __future__ import annotations

import pytest

from transformertf.config import BaseConfig
from transformertf.models import LightningModuleBase


@pytest.fixture()
def config() -> BaseConfig:
    return BaseConfig()


def test_create_base_module_from_config(config: BaseConfig) -> None:
    base_module = LightningModuleBase.from_config(config, criterion=None)
    assert base_module is not None
