from __future__ import annotations

from transformertf.models.pete import PETEModel


def test_create_pete() -> None:
    model = PETEModel(
        seq_len=100,
        num_features=4,
        n_dim_selection=64,
        n_dim_model=300,
        n_heads=4,
        n_layers=2,
        n_layers_encoded=2,
        dropout=0.1,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(num_params)
