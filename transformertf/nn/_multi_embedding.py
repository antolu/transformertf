from __future__ import annotations

import typing
import torch


class MultiEmbedding(torch.nn.Module):
    def __init__(self, embedding_sizes: typing.Sequence[tuple[int, int]]):
        """
        Multi-embedding layer

        Parameters
        ----------
        embedding_sizes: typing.Sequence[tuple[int, int]]
            list of tuples with number of categories and embedding size
            for each categorical variable
        """
        super().__init__()
        self.embedding_sizes = list(embedding_sizes)

        embeddings = torch.nn.ModuleList()
        for num_categories, embedding_size in embedding_sizes:
            embeddings.append(
                torch.nn.Embedding(num_categories, embedding_size)
            )

        self.embeddings = embeddings

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass

        Parameters
        ----------
        x : torch.Tensor
            tensor with categorical variables to embed.
            Shape: [batch_size, time, num_categoricals]

        Returns
        -------
        torch.Tensor
            tensor with embedded categorical variables.
            Shape: [batch_size, time, embedding_size, num_categoricals]
        """
        if x.shape[-1] != len(self.embeddings):
            raise ValueError(
                "Number of categorical variables does not match "
                "number of embeddings"
            )
        out = []
        for i, embedding in enumerate(self.embeddings):
            out.append(embedding(x[..., i]))

        return torch.stack(out, dim=-1)
