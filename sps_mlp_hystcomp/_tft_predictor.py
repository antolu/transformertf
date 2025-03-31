from __future__ import annotations

import os
import sys

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.temporal_fusion_transformer import TemporalFusionTransformer

from ._encoder_decoder_base import EncoderDecoderPredictor

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class TFTPredictor(EncoderDecoderPredictor):
    _module: TemporalFusionTransformer

    ALLOW_SHORTER_CONTEXT = False
    ALLOW_LONGER_PREDICTION = False
    ZERO_PAD_TARGETS = True

    @override
    def _load_checkpoint_impl(self, checkpoint_path: str | os.PathLike) -> None:
        self._module = TemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self._datamodule = EncoderDecoderDataModule.load_from_checkpoint(
            checkpoint_path,
        )
