from __future__ import annotations

import typing
import torch
import einops

from ...utils import get_activation


@typing.overload
def get_norm_layer(
    norm_type: typing.Literal["batch"], **norm_kwargs: typing.Any
) -> torch.nn.BatchNorm2d:
    ...


@typing.overload
def get_norm_layer(
    norm_type: typing.Literal["layer"], **norm_kwargs: typing.Any
) -> torch.nn.LayerNorm:
    ...


def get_norm_layer(
    norm_type: typing.Literal["batch", "layer"], **norm_kwargs: typing.Any
) -> BatchNorm2D | torch.nn.LayerNorm:
    if norm_type == "batch":
        return BatchNorm2D(**norm_kwargs)
    elif norm_type == "layer":
        return torch.nn.LayerNorm(**norm_kwargs)
    else:
        raise ValueError(f"norm must be 'batch' or 'layer', not {norm_type}")


class TemporalProjection(torch.nn.Linear):
    ...


class BatchNorm2D(torch.nn.Module):
    """
    Implementation of 2D batch normalization for 3D data by
    adding an extra dimension.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bn = torch.nn.BatchNorm2d(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "b l f -> b 1 l f")
        x = self.bn(x)
        x = einops.rearrange(x, "b 1 l f -> b l f")

        return x


class TimeMixer(torch.nn.Module):
    """
    Time-mixing MLPs model temporal patterns in time series. They consist
    of a fully-connected layer followed by an activation function and dropout.
    They transpose the input to apply the fully-connected layers along the
    time domain and shared by features. We employ a single-layer MLP, as
    demonstrated in Sec.3, where a simple linear model already proves to be a
    strong model for learning complex temporal patterns.
    """

    def __init__(
        self,
        num_features: int,
        dropout: float = 0.2,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = get_activation(activation)

        self.norm = get_norm_layer(norm, num_features=1)
        self.fc = torch.nn.Linear(num_features, num_features)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        1. Add an extra dimension to the input tensor to be able to
        apply the 2D normalization layer.
        2. Apply normalization.
        3. Remove the extra dimension.
        4. Apply the fully connected layer of the temporal MLP.
        5. Apply the activation function.
        6. Apply dropout.
        7. Add the residual connection.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = self.norm(inputs)

        x = self.fc(x)
        x = self.activation(x)
        x = self.dropout(x)
        res = x + inputs

        return res


class FeatureMixer(torch.nn.Module):
    """
    Feature-mixing MLPs are shared by time steps and serve to leverage
    covariate information. Similar to Transformer-based models, we consider
    two-layer MLPs to learn complex feature transformations.
    """

    def __init__(
        self,
        num_features: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
        out_num_features: int | None = None,
    ):
        """

        Parameters
        ----------
        num_features
        norm
        activation
        dropout
        fc_dim
        """
        super().__init__()

        self.dropout = torch.nn.Dropout(dropout)
        self.activation = get_activation(activation)

        self.norm = get_norm_layer(norm, num_features=1)
        self.fc1 = torch.nn.Linear(num_features, fc_dim)
        self.fc2 = torch.nn.Linear(fc_dim, out_num_features or num_features)

        if out_num_features is not None:
            self.fc3 = torch.nn.Linear(num_features, out_num_features)
        else:
            self.fc3 = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """

        1. Add an extra dimension to the input tensor to be able to
        apply the 2D normalization layer.
        2. Apply normalization.
        3. Remove the extra dimension.
        4. Apply the fully connected layer of the feature MLP.
        5. Apply the activation function.
        6. Apply dropout.
        7. Add the residual connection.

        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = self.norm(inputs)

        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        if self.fc3 is None:
            x = x + inputs
        else:
            x = x + self.fc3(inputs)

        return x


class ConditionalFeatureMixer(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        num_static_features: int,
        fc_dim: int = 512,
        dropout: float = 0.2,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
        out_num_features: int | None = None,
    ):
        """

        Parameters
        ----------
        num_features
        norm
        activation
        dropout
        fc_dim
        """
        super().__init__()

        self.fr = FeatureMixer(
            num_features=num_static_features,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features or num_features,
        )

        self.fm = FeatureMixer(
            num_features=(out_num_features or num_features) + num_features,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features,
        )

    def forward(
        self, x: torch.Tensor, static_features: torch.Tensor
    ) -> torch.Tensor:
        """

        1. Add an extra dimension to the input tensor to be able to
        apply the 2D normalization layer.
        2. Apply normalization.
        3. Remove the extra dimension.
        4. Apply the fully connected layer of the feature MLP.
        5. Apply the activation function.
        6. Apply dropout.
        7. Add the residual connection.

        Parameters
        ----------
        x

        Returns
        -------

        """
        v = self.fr(
            einops.repeat(static_features, "b s -> b l s", l=x.shape[1])
        )

        x = self.fm(torch.cat([x, v], dim=-1))

        return x


class MixerBlock(torch.nn.Module):
    """
    Residual Block for TSMixer, containing a time-mixing MLP and a feature-mixing
    MLP.
    """

    def __init__(
        self,
        num_features: int,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
        fc_dim: int = 512,
        dropout: float = 0.2,
        out_num_features: int | None = None,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of features in the input, i.e. the number of channels.
        norm : typing.Literal["batch", "layer"]
            Type of normalization to use. Either "batch" or "layer",
            corresponding to `torch.nn.BatchNorm2d` and `torch.nn.LayerNorm`
        activation : typing.Literal["relu", "gelu"]
            Type of activation to use. Either "relu" or "gelu", corresponding
            to `torch.nn.ReLU` and `torch.nn.GELU`
        dropout : float
            Dropout probability to use. Must be between 0 and 1.
        fc_dim : int
            Dimension of the fully-connected layers.
        """
        super().__init__()

        self.tm = TimeMixer(
            num_features=num_features,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.fm = FeatureMixer(
            num_features=num_features,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = self.tm(inputs)
        x = self.fm(x)

        return x


class ConditionalMixerBlock(torch.nn.Module):
    """
    Residual Block for TSMixer, containing a time-mixing MLP and a feature-mixing
    MLP.
    """

    def __init__(
        self,
        num_features: int,
        num_static_features: int,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
        fc_dim: int = 512,
        dropout: float = 0.2,
        out_num_features: int | None = None,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of features in the input, i.e. the number of channels.
        norm : typing.Literal["batch", "layer"]
            Type of normalization to use. Either "batch" or "layer",
            corresponding to `torch.nn.BatchNorm2d` and `torch.nn.LayerNorm`
        activation : typing.Literal["relu", "gelu"]
            Type of activation to use. Either "relu" or "gelu", corresponding
            to `torch.nn.ReLU` and `torch.nn.GELU`
        dropout : float
            Dropout probability to use. Must be between 0 and 1.
        fc_dim : int
            Dimension of the fully-connected layers.
        """
        super().__init__()

        self.tm = TimeMixer(
            num_features=num_features,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.fm = ConditionalFeatureMixer(
            num_features=num_features,
            num_static_features=num_static_features,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features or num_features,
        )

    def forward(
        self, inputs: torch.Tensor, static_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        inputs : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = self.tm(inputs)
        x = self.fm(x, static_features)

        return x
