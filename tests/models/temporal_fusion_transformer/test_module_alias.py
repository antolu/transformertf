from __future__ import annotations


def test_tft_module_alias() -> None:
    from transformertf.models.temporal_fusion_transformer import (  # noqa: PLC0415
        TemporalFusionTransformer,
    )
    from transformertf.models.tft import TFT  # noqa: PLC0415

    assert TemporalFusionTransformer is TFT
