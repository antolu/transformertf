import random
from pathlib import Path

import numpy
import pandas as pd
import pytest
import torch


@pytest.fixture(scope="session")
def random_seed() -> int:
    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    return seed


DF_PATH = str(Path(__file__).parent / "sample_data.parquet")
CURRENT = "I_meas_A"
FIELD = "B_meas_T"
FIELD_DOT = f"{FIELD}_dot"


@pytest.fixture(scope="session")
def df() -> pd.DataFrame:
    df = pd.read_parquet(DF_PATH)
    df = df.dropna()
    df = df.reset_index(drop=True)
    df[FIELD_DOT] = numpy.gradient(df[FIELD].values)  # type: ignore

    return df
