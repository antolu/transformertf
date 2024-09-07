from __future__ import annotations

import pathlib

import numpy as np
import pytest
from hystcomp_utils.cycle_data import CycleData, unflatten_cycle_data


@pytest.fixture(scope="module")
def buffers() -> list[list[CycleData]]:
    buffers = []
    for buffer_file in (pathlib.Path(__file__).parent / "sample_data").glob("*.npy"):
        buffer = np.load(buffer_file, allow_pickle=True)[()]
        buffers.append(unflatten_cycle_data(buffer))

    return buffers
