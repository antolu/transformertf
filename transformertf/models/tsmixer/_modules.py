from __future__ import annotations

import typing

import einops
import torch

from ...utils import get_activation


@typing.overload
def get_norm_layer(
    norm_type: typing.Literal["batch"],
    num_features: int | None,
    normalized_shape: tuple[int, ...] | None,
    **kwargs: typing.Any,
) -> torch.nn.BatchNorm2d: ...


@typing.overload
def get_norm_layer(
    norm_type: typing.Literal["layer"],
    num_features: int | None,
    normalized_shape: tuple[int, ...] | None,
    **kwargs: typing.Any,
) -> torch.nn.LayerNorm: ...


def get_norm_layer(
    norm_type: typing.Literal["batch", "layer"],
    num_features: int | None = None,
    normalized_shape: tuple[int, ...] | None = None,
    **kwargs: typing.Any,
) -> BatchNorm2D | torch.nn.LayerNorm:
    if norm_type == "batch":
        if num_features is None:
            msg = "num_features must be provided"
            raise ValueError(msg)
        return BatchNorm2D(num_features)
    if norm_type == "layer":
        if normalized_shape is None:
            msg = "normalized_shape must be provided"
            raise ValueError(msg)
        return torch.nn.LayerNorm(normalized_shape)  # type: ignore
    msg = f"norm must be 'batch' or 'layer', not {norm_type}"
    raise ValueError(msg)


class FeatureProjection(torch.nn.Linear): ...


class TemporalProjection(torch.nn.Module):
    activation: torch.nn.Module | None

    def __init__(
        self,
        input_len: int,
        output_len: int,
        activation: typing.Literal["relu", "gelu"] | None = "relu",
        dropout: float = 0.2,
    ):
        """
        Project past covariates to future covariates length.

        Parameters
        ----------
        input_len : int
            Length of the input sequence.
        output_len : int
            Length of the output sequence.
        activation : typing.Literal["relu", "gelu"]
            Type of activation to use. Either "relu" or "gelu", corresponding
            to `torch.nn.ReLU` and `torch.nn.GELU`
        dropout : float
            Dropout probability to use. Must be between 0 and 1.
        """
        super().__init__()

        self.fc = torch.nn.Linear(input_len, output_len)
        self.activation = None if activation is None else get_activation(activation)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = einops.rearrange(x, "b l f -> b f l")
        x = self.fc(x)
        x = einops.rearrange(x, "b f l -> b l f")
        x = self.activation(x) if self.activation is not None else x
        return self.dropout(x)


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
        return einops.rearrange(x, "b 1 l f -> b l f")


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
        input_len: int,
        num_features: int,
        dropout: float = 0.2,
        norm: typing.Literal["batch", "layer"] = "batch",
        activation: typing.Literal["relu", "gelu"] = "relu",
    ):
        super().__init__()

        self.temporal_linear = TemporalProjection(
            input_len=input_len,
            output_len=input_len,
            activation=activation,
            dropout=dropout,
        )

        self.norm = get_norm_layer(
            norm, num_features=1, normalized_shape=(input_len, num_features)
        )

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
        inputs : torch.Tensor
            Input tensor of shape [batch_size, seq_len, num_features]

        Returns
        -------
        torch.Tensor
            Output tensor of shape [batch_size, seq_len, num_features]
        """
        x = self.temporal_linear(inputs)
        res = x + inputs
        return self.norm(res)


class FeatureMixer(torch.nn.Module):
    """
    Feature-mixing MLPs are shared by time steps and serve to leverage
    covariate information. Similar to Transformer-based models, we consider
    two-layer MLPs to learn complex feature transformations.
    """

    def __init__(
        self,
        input_len: int,
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

        self.norm = get_norm_layer(
            norm,
            num_features=1,
            normalized_shape=(input_len, out_num_features or num_features),
        )
        self.fc1 = torch.nn.Linear(num_features, fc_dim)
        self.fc2 = torch.nn.Linear(fc_dim, out_num_features or num_features)

        # reduce dimensionality of residual connection
        if out_num_features is not None and out_num_features != num_features:
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
        residual = self.fc3(inputs) if self.fc3 is not None else inputs

        x = self.fc1(inputs)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)

        x += residual
        return self.norm(x)


class ConditionalFeatureMixer(torch.nn.Module):
    fr: FeatureMixer | None

    def __init__(
        self,
        input_len: int,
        num_features: int,
        num_static_features: int,
        hidden_dim: int | None = None,
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

        self.num_static_features = num_static_features

        hidden_dim = hidden_dim or num_features

        if num_static_features > 0:
            self.fr = FeatureMixer(
                input_len=input_len,
                num_features=num_static_features,
                fc_dim=fc_dim,
                dropout=dropout,
                norm=norm,
                activation=activation,
                out_num_features=hidden_dim,
            )

            cfm_num_features = num_features + hidden_dim
        else:
            self.fr = None

            cfm_num_features = num_features

        self.fm = FeatureMixer(
            input_len=input_len,
            num_features=cfm_num_features,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features,
        )

    def forward(
        self, x: torch.Tensor, static_features: torch.Tensor | None = None
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
        if self.fr is None:
            return self.fm(x)

        if static_features is None:
            msg = "static_features must be provided"
            raise ValueError(msg)

        v = self.fr(einops.repeat(static_features, "b s -> b l s", l=x.shape[1]))

        return self.fm(torch.cat([x, v], dim=-1))


class MixerBlock(torch.nn.Module):
    """
    Residual Block for TSMixer, containing a time-mixing MLP and a feature-mixing
    MLP.
    """

    def __init__(
        self,
        input_len: int,
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
            input_len=input_len,
            num_features=num_features,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.fm = FeatureMixer(
            input_len=input_len,
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
        return self.fm(x)


class ConditionalMixerBlock(torch.nn.Module):
    """
    Residual Block for TSMixer, containing a time-mixing MLP and a feature-mixing
    MLP.
    """

    def __init__(
        self,
        input_len: int,
        num_features: int,
        num_static_features: int,
        hidden_dim: int | None = None,
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
            input_len=input_len,
            num_features=num_features,
            dropout=dropout,
            norm=norm,
            activation=activation,
        )
        self.fm = ConditionalFeatureMixer(
            input_len=input_len,
            num_features=num_features,
            num_static_features=num_static_features,
            hidden_dim=hidden_dim,
            fc_dim=fc_dim,
            dropout=dropout,
            norm=norm,
            activation=activation,
            out_num_features=out_num_features or num_features,
        )

    def forward(
        self, inputs: torch.Tensor, static_features: torch.Tensor | None = None
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
        return self.fm(x, static_features)
