from __future__ import annotations

import torch

from ._grn import GatedResidualNetwork


class VariableSelection(torch.nn.Module):
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 8,
        n_dim_model: int = 300,
        context_size: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_dim_model = n_dim_model
        self.context_size = context_size
        self.dropout = torch.nn.Dropout(dropout)

        self.prescalers = torch.nn.ModuleList([
            torch.nn.Linear(1, hidden_dim) for _ in range(n_features)
        ])

        if self.n_features > 1:
            if self.context_size is not None:
                self.flattened_grn = GatedResidualNetwork(
                    input_dim=n_features * hidden_dim,
                    hidden_dim=min(self.n_dim_model, self.n_features),
                    output_dim=self.n_features,
                    context_dim=self.context_size,
                    dropout=dropout,
                    projection="interpolate",
                )
            else:
                self.flattened_grn = GatedResidualNetwork(
                    input_dim=n_features * hidden_dim,
                    hidden_dim=min(self.n_dim_model, self.n_features),
                    output_dim=self.n_features,
                    dropout=dropout,
                    projection="interpolate",
                )
        else:
            self.flattened_grn = torch.nn.Identity()

        self.single_grn = torch.nn.ModuleList([
            GatedResidualNetwork(
                input_dim=hidden_dim,
                hidden_dim=min(hidden_dim, self.n_dim_model),
                output_dim=self.n_dim_model,
                dropout=dropout,
                projection="interpolate",
            )
            for i in range(n_features)
        ])

    def forward(
        self, x: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """

        Parameters
        ----------
        x : torch.Tensor
            [batch_size, seq, n_feat]
        context : torch.Tensor, optional

        Returns
        -------

        """
        if x.shape[-1] != self.n_features:
            msg = (
                f"Input tensor must have {self.n_features} features, "
                f"but got {x.shape[-1]} features instead."
            )
            raise ValueError(msg)

        if self.context_size is None and context is not None:
            msg = "Context tensor is not expected, but got a context tensor."
            raise ValueError(msg)

        if self.n_features == 1:
            # if there is only one feature, we don't need to do
            # any variable selection (i.e. unit weights).
            x = self.prescalers[0](x)
            x = self.single_grn[0](x)

            if x.ndim == 3:
                sparse_weights = torch.ones(*x.shape[0:2], 1, 1, device=x.device)
            else:
                sparse_weights = torch.ones(x.shape[0], 1, 1, device=x.device)

            return x, sparse_weights

        # more than one feature
        outputs_l = []
        weights_l = []

        for i in range(self.n_features):
            x_i = self.prescalers[i](x[..., i : i + 1])
            weights_l.append(x_i)

            x_i = self.single_grn[i](x_i)
            outputs_l.append(x_i)

        outputs = torch.stack(outputs_l, dim=-1)
        weights = torch.cat(weights_l, dim=-1)
        weights = self.flattened_grn(weights, context)
        weights = torch.nn.functional.softmax(weights, dim=-1).unsqueeze(-2)

        outputs *= weights
        outputs = outputs.sum(dim=-1)

        return outputs, weights
