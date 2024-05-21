from __future__ import annotations

from transformertf.models.phytsmixer import PhyTSMixer, PhyTSMixerConfig


def test_create_module() -> None:
    PhyTSMixer.from_config(PhyTSMixerConfig())
