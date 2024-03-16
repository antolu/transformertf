from __future__ import annotations


def test_tft_module_alias() -> None:
    from transformertf.models.temporal_fusion_transformer import (
        TemporalFusionTransformer,
    )
    from transformertf.models.tft import TemporalFusionTransformer as TFT

    assert TemporalFusionTransformer is TFT
