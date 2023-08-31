from __future__ import annotations
from ..conftest import CURRENT, FIELD
import pandas as pd
from sklearn.utils.validation import check_is_fitted
import torch

from transformertf.data import PolynomialTransform


def test_polynomial_transform_fit(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=1000)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)


def test_polynomial_transform_transform(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=1000)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)

    transformed = transform.transform(df[CURRENT].values, df[FIELD].values)

    assert transformed.shape == (len(df),)

    import matplotlib.pyplot as plt

    plt.plot(df[CURRENT].values, df[FIELD].values)
    plt.plot(
        df[CURRENT].values,
        transform(torch.from_numpy(df[CURRENT].values).detach()),
    )
    plt.twinx()
    plt.plot(df[CURRENT].values, transformed)
    plt.show()


def test_polynomial_transform_inverse_transform(df: pd.DataFrame) -> None:
    transform = PolynomialTransform(degree=2, num_iterations=1000)

    transform.fit(df[CURRENT].values, df[FIELD].values)

    check_is_fitted(transform)

    transformed = transform.transform(df[CURRENT].values, df[FIELD].values)

    assert transformed.shape == (len(df),)

    inverse_transformed = transform.inverse_transform(
        df[CURRENT].values, transformed
    )

    assert inverse_transformed.shape == (len(df),)
