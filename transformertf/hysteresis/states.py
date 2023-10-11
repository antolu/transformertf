from __future__ import annotations

import torch
import typing


def sweep_up(
    h: float, mesh: torch.Tensor, initial_state: torch.Tensor, T: float = 1e-2
) -> torch.Tensor:
    return torch.minimum(
        initial_state + switch(h, mesh[:, 1], T), torch.ones_like(mesh[:, 1])
    )


def sweep_left(
    h: float, mesh: torch.Tensor, initial_state: torch.Tensor, T: float = 1e-2
) -> torch.Tensor:
    return torch.maximum(
        initial_state - switch(mesh[:, 0], h, T),
        torch.ones_like(mesh[:, 0]) * -1.0,
    )


def switch(h: float, mesh: torch.Tensor, T: float = 1e-4) -> torch.Tensor:
    # note that + T is needed to satisfy boundary conditions (creating a bit of delay
    # before the flip starts happening
    return 1.0 + torch.tanh((h - mesh - 0 * T) / abs(T))


def get_current(
    current_state: torch.Tensor | None,
    current_field: torch.Tensor | None,
    n_mesh_points: int,
    **tkwargs: typing.Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    # list of hysteresis states with initial state set
    if current_state is None:
        initial_state = torch.ones(n_mesh_points, **tkwargs) * -1.0
        initial_field = torch.zeros(1)
    else:
        if not isinstance(current_field, torch.Tensor):
            raise ValueError("need to specify current field if state is given")
        if current_state.shape[-1] != n_mesh_points:
            raise ValueError("curret state must match number of mesh points")
        initial_state = current_state
        initial_field = current_field

    return initial_state, initial_field


def predict_batched_state(
    h: torch.Tensor,
    mesh_points: torch.Tensor,
    current_state: torch.Tensor | None = None,
    current_field: torch.Tensor | None = None,
    tkwargs: dict[str, typing.Any] | None = None,
    temp: float = 1e-3,
) -> torch.Tensor:
    """
    Speed up optimization by calculating a batched future state given a batch of h
    values.

    Parameters
    ----------
    temp
    tkwargs
    h
    mesh_points
    current_state
    current_field

    Returns
    -------

    """

    h = h.unsqueeze(-1)
    n_mesh_points = mesh_points.shape[0]
    tkwargs = tkwargs or {}

    state, field = get_current(
        current_state, current_field, n_mesh_points, **tkwargs
    )

    result = torch.where(
        torch.greater_equal(h - field, torch.zeros(1).to(h)),
        sweep_up(h, mesh_points, state, temp),
        sweep_left(h, mesh_points, state, temp),
    )
    return result


def get_states(
    h: torch.Tensor,
    mesh_points: torch.Tensor,
    current_state: torch.Tensor | None = None,
    current_field: torch.Tensor | None = None,
    tkwargs: dict[str, typing.Any] | None = None,
    temp: float = 1e-3,
) -> torch.Tensor:
    """
    Returns magnetic hysteresis state as an m x n x n tensor, where
    m is the number of distinct applied magnetic fields. The
    states are initially entirely off, and then updated per
    time step depending on both the most recent applied magnetic
    field and prior inputs (i.e. the "history" of the states tensor).

    For each time step, the state matrix is either "swept up" or
    "swept left" based on how the state matrix corresponds to like
    elements in the meshgrid; the meshgrid contains alpha, beta
    coordinates which serve as thresholds for the hysterion state to
    "flip".

    This calculation can be expensive, so we skip recalcuation until if h !=
    current_h

    See: https://www.wolframcloud.com/objects/demonstrations
    /TheDiscretePreisachModelOfHysteresis-source.nb

    Parameters
    ----------
    current_field
    current_state
    temp
    tkwargs
    mesh_points
    h : torch.Tensor,
        The applied magnetic field H_1:t={H_1, ... ,H_t}, where
        t represents each time step.

    Raises
    ------
    ValueError
        If n is negative.
    """
    # verify the inputs are in the normalized region within some machine epsilon
    epsilon = 1e-6
    if torch.any(torch.less(h + epsilon, torch.zeros(1))) or torch.any(
        torch.greater(h - epsilon, torch.ones(1))
    ):
        raise RuntimeError("applied values are outside of the unit domain")

    assert len(h.shape) == 1
    n_mesh_points = mesh_points.shape[0]
    tkwargs = tkwargs or {}

    # list of hysteresis states with initial state set
    initial_state, initial_field = get_current(
        current_state, current_field, n_mesh_points, **tkwargs
    )

    states = []

    # loop through the states
    for i in range(len(h)):
        if i == 0:
            # handle initial case
            if h[0] > initial_field:
                states += [sweep_up(h[i], mesh_points, initial_state, temp)]
            elif h[0] < initial_field:
                states += [sweep_left(h[i], mesh_points, initial_state, temp)]
            else:
                states += [initial_state]

        elif h[i] > h[i - 1]:
            # if the new applied field is greater than the old one, sweep up to
            # new applied field
            states += [sweep_up(h[i], mesh_points, states[i - 1], temp)]
        elif h[i] < h[i - 1]:
            # if the new applied field is less than the old one, sweep left to
            # new applied field
            states += [sweep_left(h[i], mesh_points, states[i - 1], temp)]
        else:
            states += [states[i - 1]]

    # concatenate states into one tensor
    total_states = torch.cat([ele.unsqueeze(0) for ele in states])
    return total_states
