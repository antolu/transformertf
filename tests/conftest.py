import random
from pathlib import Path

import numpy
import pandas as pd
import pytest
import torch

from transformertf.data.datamodule._base import (
    known_cov_col,
    target_col,
)


@pytest.fixture(scope="session")
def random_seed() -> int:
    seed = 0
    random.seed(seed)
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
    df[FIELD_DOT] = numpy.gradient(df[FIELD].to_numpy())

    df[known_cov_col(CURRENT)] = df[CURRENT]
    df[target_col(FIELD)] = df[FIELD]

    return df


@pytest.fixture(scope="session")
def df_path() -> str:
    return DF_PATH


@pytest.fixture(scope="session")
def current_key() -> str:
    return CURRENT


@pytest.fixture(scope="session")
def field_key() -> str:
    return FIELD


@pytest.fixture(scope="session")
def time_key() -> str:
    return "time_ms"


@pytest.fixture(scope="session")
def current_cov_key() -> str:
    return known_cov_col(CURRENT)


@pytest.fixture(scope="session")
def field_cov_key() -> str:
    return target_col(FIELD)


@pytest.fixture(scope="session")
def tmp_dir(tmpdir_factory: pytest.TempdirFactory) -> Path:
    return Path(tmpdir_factory.mktemp("tmp"))
