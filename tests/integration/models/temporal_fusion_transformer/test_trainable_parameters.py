from __future__ import annotations

import typing

from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer


def test_tft_trainable_parameters(tft_module_config: dict[str, typing.Any]) -> None:
    trainable_parameters = ["env_vs", "dec_vs"]
    tft_module_config = tft_module_config | {
        "trainable_parameters": trainable_parameters,
    }
    tft_module = TemporalFusionTransformer(**tft_module_config)

    parameters = list(tft_module.parameters())
    assert len(parameters) == 13


def test_tft_trainable_parameters_empty(
    tft_module_config: dict[str, typing.Any],
) -> None:
    tft_module = TemporalFusionTransformer(**tft_module_config)

    parameters = list(tft_module.parameters())
    assert len(parameters) == 163
