import torch

from transformertf.data import RunningNormalizer


def test_running_normalizer_single_feature() -> None:
    normalizer = RunningNormalizer(num_features_=1)

    x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    x_transformed = normalizer.fit_transform(x)

    target = torch.tensor(
        [-1.2649109, -0.6324555, 0.0000000, 0.6324555, 1.2649109]
    )

    assert torch.allclose(x_transformed, target)
    assert normalizer.n_samples_seen_ == 5
    assert torch.allclose(normalizer.center_, torch.tensor([3.0]))
    assert torch.allclose(normalizer.scale_, torch.tensor([1.5811390]))
    assert torch.allclose(
        normalizer.get_parameters(), torch.tensor([[3.0, 1.5811390]])
    )

    x_inverse_transformed = normalizer.inverse_transform(x_transformed)
    assert torch.allclose(x_inverse_transformed, x)


def test_running_normalizer_multi_feature() -> None:
    normalizer = RunningNormalizer(num_features_=2)

    # add batch dimension
    x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]])[
        None, ...
    ]

    x_transformed = normalizer.fit_transform(x)

    target = torch.tensor(
        [
            [
                [-1.1619, -1.1619],
                [-0.3873, -0.3873],
                [0.3873, 0.3873],
                [1.1619, 1.1619],
            ]
        ]
    )

    assert torch.allclose(x_transformed, target)
    assert normalizer.n_samples_seen_ == 1
    assert torch.allclose(normalizer.center_, torch.tensor([2.5, 3.5]))
    assert torch.allclose(
        normalizer.scale_, torch.tensor([1.2909945, 1.2909945])
    )
    assert torch.allclose(
        normalizer.get_parameters(),
        torch.tensor([[2.5, 1.2909945], [3.5, 1.2909945]]),
    )

    x_inverse_transformed = normalizer.inverse_transform(x_transformed)
    assert torch.allclose(x_inverse_transformed, x)
