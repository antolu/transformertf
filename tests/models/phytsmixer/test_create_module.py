from __future__ import annotations


from transformertf.models.phytsmixer import PhyTSMixerConfig, PhyTSMixerModule


def test_create_module() -> None:
    PhyTSMixerModule.from_config(PhyTSMixerConfig())
