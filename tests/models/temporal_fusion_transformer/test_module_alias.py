from __future__ import annotations


def test_tft_module_alias() -> None:
    from transformertf.models.temporal_fusion_transformer import (  # noqa: PLC0415
        TemporalFusionTransformerModel,
    )
    from transformertf.models.tft import (  # noqa: PLC0415
        TemporalFusionTransformerModel as TFT,
    )

    assert TemporalFusionTransformerModel is TFT
