from __future__ import annotations

import pytest

from transformertf.config import TransformerBaseConfig
from transformertf.models import TransformerModuleBase


@pytest.fixture()
def config() -> TransformerBaseConfig:
    return TransformerBaseConfig()


def test_create_base_transformer_module_from_config(
    config: TransformerBaseConfig,
) -> None:
    base_module = TransformerModuleBase.from_config(config)
    assert base_module is not None


def test_create_base_transformer_module_criterions(
    config: TransformerBaseConfig,
) -> None:
    for crit in ("mse", "mae", "huber"):
        module = TransformerModuleBase.from_config(config, loss_fn=crit)
        assert module.criterion is not None


def test_create_base_transformer_module_quantile(
    config: TransformerBaseConfig,
) -> None:
    module = TransformerModuleBase.from_config(
        config, loss_fn="quantile", quantiles=[0.5]
    )
    assert module.criterion is not None
