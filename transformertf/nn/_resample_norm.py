from __future__ import annotations

import einops
import torch


def interpolate(
    x: torch.Tensor, output_size: int, mask: torch.Tensor | None = None
) -> torch.Tensor:
    if x.device.type == "mps":
        x = x.to("cpu")
        upsampled = torch.nn.functional.interpolate(
            x.unsqueeze(1), output_size, mode="linear", align_corners=True
        ).squeeze(1)
        return upsampled.to("mps")

    upsampled = torch.nn.functional.interpolate(
        x.unsqueeze(1), output_size, mode="linear", align_corners=True
    ).squeeze(1)

    if mask is not None:
        upsampled = upsampled * torch.nn.functional.sigmoid(mask.unsqueeze(0)) * 2.0
    return upsampled


class TimeDistributedInterpolation(torch.nn.Module):
    def __init__(
        self,
        output_dim: int,
        *,
        batch_first: bool = False,
        trainable: bool = False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.batch_first = batch_first
        self.trainable = trainable

        self.mask: torch.nn.Parameter | None = None
        if self.trainable:
            self.mask = torch.nn.Parameter(
                torch.zeros(self.output_dim, dtype=torch.float)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim <= 2:
            return interpolate(x, self.output_dim, self.mask)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)

        # do the same with einops
        x_reshape = einops.rearrange(x, "s t c -> (s t) c")

        y = interpolate(x_reshape, self.output_dim, self.mask)

        # We have to reshape Y
        if self.batch_first:
            # y = y.contiguous().view(
            #     x.size(0), -1, y.size(-1)
            # )  # (samples, timesteps, output_size)
            return einops.rearrange(y, "(s t) c -> s t c", s=x.size(0), t=x.size(1))
        # y = y.view(
        #     -1, x.size(1), y.size(-1)
        # )  # (timesteps, samples, output_size)
        return einops.rearrange(y, "t s c -> s t c", s=x.size(0), t=x.size(1))


class ResampleNorm(torch.nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int | None = None,
        *,
        trainable_add: bool = True,
    ):
        """
        ResampleNorm layer, which is just a linear layer followed by layer normalization,
        but...

        Parameters
        ----------
        input_dim : int
            Input dimension
        output_dim : int, default=None
            Output dimension. If None, the output dimension will be the same as the input dimension
        trainable_add : bool, default=True
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim or input_dim
        self.trainable_add = trainable_add

        if self.input_dim != self.output_dim:
            self.resample = TimeDistributedInterpolation(
                output_dim=self.output_dim,
                batch_first=True,
                trainable=False,
            )
        else:
            self.resample = torch.nn.Identity()

        if self.trainable_add:
            self.mask = torch.nn.Parameter(
                torch.zeros(self.output_dim, dtype=torch.float)
            )
        else:
            self.mask = torch.tensor(0.0)

        self.norm = torch.nn.LayerNorm(self.output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_dim != self.output_dim:
            x = self.resample(x)

        if self.trainable_add:
            x = x * torch.nn.functional.sigmoid(self.mask) * 2.0

        return self.norm(x)
