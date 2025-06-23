from __future__ import annotations

import torch


def get_output_padding(
    input_length: int,
    output_length: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int = 1,
) -> int:
    # Formula from PyTorch docs:
    # L_out = (L_in - 1) * stride - 2*padding + dilation*(kernel_size-1) + output_padding + 1
    expected = (
        (input_length - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + 1
    )
    return output_length - expected


def dynamic_same_pad_1d(
    x: torch.Tensor, kernel_size: int, dilation: int = 1
) -> torch.Tensor:
    total_pad = dilation * (kernel_size - 1)
    pad_left = total_pad // 2
    pad_right = total_pad - pad_left
    return torch.nn.functional.pad(x, (pad_left, pad_right))


class PaddedConv1d(torch.nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Calculate padding
        total_pad = self.kernel_size[0] - 1
        pad_left = total_pad // 2
        pad_right = total_pad - pad_left
        x = torch.nn.functional.pad(x, (pad_left, pad_right))
        return super().forward(x)


class DynamicConvTranspose1d(torch.nn.ConvTranspose1d):
    def forward(
        self, x: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        if target_length is not None:
            output_padding = get_output_padding(
                x.shape[-1],
                target_length,
                self.kernel_size[0],
                self.stride[0],
                self.padding[0],
                self.dilation[0],
            )
            output_padding = max(
                0, output_padding
            )  # Ensure non-negative output padding
            out = torch.nn.functional.conv_transpose1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                output_padding,
                self.groups,
                self.dilation,
            )
            if out.shape[-1] > target_length:
                out = out[..., :target_length]
            return out
        return super().forward(x)


# gather the convolutions into blocks
class DownsampleBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        downsample: int = 2,
    ):
        super().__init__()
        self.conv1 = PaddedConv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.bn1 = torch.nn.BatchNorm1d(in_channels)
        self.conv2 = PaddedConv1d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
        )
        self.bn2 = torch.nn.BatchNorm1d(in_channels * 2)
        # self.conv3 = PaddedConv1d(
        #     in_channels=n_channels * 2,
        #     out_channels=n_channels * 4,
        #     kernel_size=3,
        #     stride=1,
        #     padding=0,
        # )
        # self.bn3 = torch.nn.BatchNorm1d(n_channels * 4)
        self.conv4 = PaddedConv1d(
            in_channels=in_channels * 2,
            out_channels=in_channels * 2,
            kernel_size=3,
            stride=downsample,
            padding=0,
        )
        self.bn4 = torch.nn.BatchNorm1d(in_channels * 2)
        self.conv5 = PaddedConv1d(
            in_channels=in_channels * 2,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
        )

        self.layers = torch.nn.Sequential(
            self.conv1,
            self.bn1,
            torch.nn.LeakyReLU(),
            self.conv2,
            self.bn2,
            torch.nn.LeakyReLU(),
            # self.conv3,
            # self.bn3,
            torch.nn.LeakyReLU(),
            self.conv4,
            self.bn4,
            torch.nn.LeakyReLU(),
            self.conv5,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class UpsampleBlock(torch.nn.Module):
    def __init__(self, n_channels: int, kernel_size: int = 3, upsample: int = 2):
        super().__init__()
        self.conv1 = DynamicConvTranspose1d(
            in_channels=n_channels,
            out_channels=n_channels * 4,
            kernel_size=1,
            stride=1,
            padding=0,
        )
        self.bn1 = torch.nn.BatchNorm1d(n_channels * 4)
        self.conv2 = DynamicConvTranspose1d(
            in_channels=n_channels * 4,
            out_channels=n_channels * 4,
            kernel_size=kernel_size,
            stride=upsample,
            padding=kernel_size // 2,
        )
        self.bn2 = torch.nn.BatchNorm1d(n_channels * 4)
        self.conv3 = DynamicConvTranspose1d(
            in_channels=n_channels * 4,
            out_channels=n_channels * 2,
            kernel_size=3,
            stride=1,
            padding=kernel_size // 2,
        )
        self.bn3 = torch.nn.BatchNorm1d(n_channels * 2)
        self.conv4 = DynamicConvTranspose1d(
            in_channels=n_channels * 2,
            out_channels=n_channels,
            kernel_size=3,
            stride=upsample,
            padding=kernel_size // 2,
        )
        self.bn4 = torch.nn.BatchNorm1d(n_channels)
        self.conv5 = DynamicConvTranspose1d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=3,
            stride=1,
            padding=3 // 3,
        )

        self.activation = torch.nn.LeakyReLU()

    def forward(
        self, x: torch.Tensor, target_length: int | None = None
    ) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.conv2(x, target_length=target_length)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.conv3(x, target_length=target_length)
        x = self.bn3(x)
        x = self.activation(x)
        x = self.conv4(x, target_length=target_length)
        x = self.bn4(x)
        x = self.activation(x)
        return self.conv5(x, target_length=target_length)
