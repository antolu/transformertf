from __future__ import annotations

from transformertf.models.phytsmixer import PhyTSMixer


def test_create_module() -> None:
    PhyTSMixer(num_features=1)
