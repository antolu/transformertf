import numpy
import random
import torch

import pytest


@pytest.fixture(scope="session")
def random_seed() -> int:
    seed = 0
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)

    return seed
