from __future__ import annotations

import os
import sys

from transformertf.data import EncoderDecoderDataModule
from transformertf.models.pf_tft import PFTemporalFusionTransformer

from ._encoder_decoder_base import EncoderDecoderPredictor

if sys.version_info >= (3, 12):
    from typing import override
else:
    from typing_extensions import override


class PFTFTPredictor(EncoderDecoderPredictor):
    _module: PFTemporalFusionTransformer

    ALLOW_SHORTER_CONTEXT = True
    ALLOW_LONGER_PREDICTION = True

    @override
    def _load_checkpoint_impl(self, checkpoint_path: str | os.PathLike) -> None:
        self._module = PFTemporalFusionTransformer.load_from_checkpoint(checkpoint_path)
        self._datamodule = EncoderDecoderDataModule.load_from_checkpoint(
            checkpoint_path,
        )
