from __future__ import annotations

import typing

import numpy as np
import torch
from torch.nn import Module

if typing.TYPE_CHECKING:
    from .hysteresis import HysteresisMagnet


class TorchAccelerator(Module):
    def __init__(
        self,
        elements: typing.Sequence[HysteresisMagnet],
        allow_duplicates: bool = False,
    ):
        Module.__init__(self)

        # check to make sure no duplicate names exist
        names = []
        for ele in elements:
            if ele.name not in names or allow_duplicates:
                names += [ele.name]
            else:
                msg = f"duplicate name {ele.name} found but not allowed"
                raise RuntimeError(msg)

        self.elements = {element.name: element for element in elements}

        for ele in self.elements.values():
            self.add_module(ele.name, ele)

    def calculate_transport(self) -> torch.Tensor:
        M_i = torch.eye(6)
        M = [M_i]
        for ele in self.elements.values():
            M += [torch.matmul(ele(), M[-1])]
        return torch.cat([m.unsqueeze(0) for m in M], dim=0)

    @staticmethod
    def propagate_beam(M: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Propagate beam given a initial beam matrix R, and a beam transport
        matrix/matricies M
        Parameters
        ----------
        M : torch.Tensor
            Transport matrix of form n x 6 x 6 where n is the number of beam
            matricies to calculate (not necessarily in order)
        R : torch.Tensor
            Initial beam matrix

        """
        if M.shape[-2:] != torch.Size([6, 6]):
            msg = "last dims of transport matrix must be 6 x 6"
            raise RuntimeError(msg)

        return torch.matmul(M, torch.matmul(R, torch.transpose(M, -2, -1)))

    def forward(self, R: torch.Tensor, full: bool = True) -> torch.Tensor:
        """
        Calculate the beam matrix. If full is True (default) the beam matrix at the
        end of the beamline is returned. If False the beam matrix after every element
        is returned.

        Parameters
        ----------
        R
        full

        Returns
        -------

        """
        M = self.calculate_transport()
        R_f = self.propagate_beam(M, R)
        if full:
            return R_f[-1]
        return R_f


def rot(alpha: float) -> torch.Tensor:
    M = torch.eye(6)

    C = torch.cos(torch.tensor(alpha))
    S = torch.sin(torch.tensor(alpha))
    M[0, 0] = C
    M[0, 2] = S
    M[1, 1] = C
    M[1, 3] = S
    M[2, 0] = -S
    M[2, 2] = C
    M[3, 1] = -S
    M[3, 3] = C

    return M


class TorchQuad(Module):
    def __init__(self, name: str, length: torch.Tensor, K1: torch.Tensor | None):
        Module.__init__(self)
        self.register_parameter("L", torch.nn.parameter.Parameter(length))
        self.register_parameter("K1", torch.nn.parameter.Parameter(K1))

        self.name = name

    def forward(self) -> torch.Tensor:
        return self.get_matrix(self.K1)

    def get_matrix(self, K1: torch.Tensor) -> torch.Tensor:
        # add small deviation if K1 is zero
        K1 = torch.where(torch.abs(K1) < 1e-6, K1 + 1e-6, K1)

        M = torch.empty(*K1.shape, 6, 6)
        M[..., :, :] = torch.eye(6)

        mag_K1 = torch.where(K1 < 0.0, -K1, K1)

        k = torch.sqrt(mag_K1)

        kl = self.L * k
        M[..., 0, 0] = torch.cos(kl)
        M[..., 0, 1] = torch.sin(kl) / k
        M[..., 1, 0] = -k * torch.sin(kl)
        M[..., 1, 1] = torch.cos(kl)

        M[..., 2, 2] = torch.cosh(kl)
        M[..., 2, 3] = torch.sinh(kl) / k
        M[..., 3, 2] = k * torch.sinh(kl)
        M[..., 3, 3] = torch.cosh(kl)

        M_rot = torch.matmul(torch.matmul(rot(-np.pi / 2), M), rot(np.pi / 2))

        # expand array to make >0 comparison, don't try this at home kids
        K1_test = (
            K1.unsqueeze(-1)
            .repeat_interleave(6, dim=-1)
            .unsqueeze(-1)
            .repeat_interleave(6, dim=-1)
        )
        return M * (K1_test >= 0.0).float() + M_rot * (K1_test < 0.0).float()


class TorchDrift(Module):
    def __init__(self, name: str, length: torch.Tensor, fixed: bool = True):
        Module.__init__(self)
        self.name = name
        if fixed:
            self.register_buffer("L", torch.nn.parameter.Parameter(length))
        else:
            self.register_parameter("L", torch.nn.parameter.Parameter(length))

    def forward(self) -> torch.Tensor:
        M = torch.eye(6)
        M[0, 1] = self.L
        M[2, 3] = self.L

        return M
