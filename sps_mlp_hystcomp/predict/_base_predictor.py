from __future__ import annotations

import abc
import enum
import logging
import os
import threading
import typing

import lightning as L
import numpy as np
import numpy.typing as npt
import pandas as pd
import pybind11_rdp
import torch
from hystcomp_utils.cycle_data import CycleData
from transformertf.data.datamodule import DataModuleBase
from transformertf.models import LightningModuleBase
from transformertf.utils import signal

if typing.TYPE_CHECKING:
    SameType = typing.TypeVar("SameType", bound="Predictor")


TIME_COLNAME = "__time__"
I_PROG_COLNAME = "I_meas_A_filtered"
I_PROG_DOT_COLNAME = "I_meas_A_filtered_dot"
TARGET_PAST_COLNAME = "B_meas_T_filtered_"
TARGET_COLNAME = "B_meas_T_filtered"


class PredictionCovariates(enum.StrEnum):
    TIME = "__time__"
    CURRENT = "past_current"
    CURRENT_DOT = "past_current_dot"
    PAST_TARGET_ = "past_target_"
    TARGET = "past_target"


T_Module_co = typing.TypeVar("T_Module_co", bound=LightningModuleBase, covariant=True)
T_DataModule_co = typing.TypeVar(
    "T_DataModule_co", bound=DataModuleBase, covariant=True
)


log = logging.getLogger(__name__)


class NoModelLoadedError(Exception):
    pass


class NoInitialStateError(Exception):
    pass


class PredictorHooks:
    def on_before_predict(self) -> None:
        pass

    def on_after_predict(self) -> None:
        pass

    def on_after_load_checkpoint(self) -> None:
        pass


class PredictorABC(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def _set_initial_state_impl(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Set the initial state of the model. The method should be implemented by
        subclasses.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_impl(
        self,
        future_covariates: pd.DataFrame,
        *args: typing.Any,
        save_state: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        """
        Predict the next value. The method should be implemented by
        subclasses and update the internal state of the model for the
        next prediction.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _predict_cycle_impl(
        self,
        cycle: CycleData,
        *args: typing.Any,
        save_state: bool = True,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        """
        Predict the next value. The method should be implemented by
        subclasses and update the internal state of the model for the
        next prediction.

        Parameters
        ----------
        cycle : CycleData
            Cycle to use for prediction
        save_state : bool
            Whether to save the state of the model after making the
            prediction, by default True. This is required when making an autoregressive
            prediction.
        use_programmed_current : bool
            Whether to use the programmed current, by default True. If false,
            measured current will be used instead.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def _load_checkpoint_impl(self, checkpoint_path: str | os.PathLike) -> None:
        """
        Load a checkpoint from disk.

        Parameters
        ----------
        checkpoint_path : str | os.PathLike
            Path to the checkpoint file.
        """
        raise NotImplementedError


class PredictorUtils:
    @staticmethod
    def buffer_to_covariates(
        buffer: list[CycleData],
        *,
        use_programmed_current: bool = True,
        add_target: bool = True,
        rdp: typing.Literal[False] | float = False,
    ) -> pd.DataFrame:
        """
        Convert a buffer of cycles to covariates for the model.

        Parameters
        ----------
        buffer : list[CycleData]
            List of CycleData objects.
        use_programmed_current : bool
            Whether to use the programmed current, by default True.
            If False, measured current will be used instead.
        add_target : bool
            Whether to add the target to the covariates, by default True.
        rdp : bool or float
            Whether to use adaptive downsampling, by default False.
            If true, the programmed current will not be interpolated if
            use_programmed_current is True. If False, the programmed current
            will be used as collocation points to sample the measured current.
            If a float, the value will be used as the epsilon parameter for
            the Ramer-Douglas-Peucker algorithm.

        Returns
        -------
        pd.DataFrame
            DataFrame with the covariates. The columns are:
            - "__time__": time array
            - "past_current": past current
            - "past_current_dot": derivative of the past current
            - "past_target_": past target
            - "past_target": past target
        """
        if len(buffer) == 0:
            msg = "Buffer must contain at least one cycle."
            raise ValueError(msg)

        if use_programmed_current:
            covariates = make_prog_base_covariates(
                buffer,
                interpolate=not isinstance(rdp, float),
                rdp_eps=0.0 if rdp is False else rdp,
            )
        else:
            covariates = make_meas_base_covariates(buffer, rdp=rdp)

        if add_target:
            if all(cycle.field_meas is not None for cycle in buffer):
                b_meas = np.concatenate(
                    [cycle.field_meas.flatten() for cycle in buffer]
                )
                b_meas = filter_bmeas(b_meas)

                if use_programmed_current:
                    time = covariates[TIME_COLNAME].to_numpy()
                    time_b = np.arange(len(b_meas)) / 1e3

                    b_meas = np.interp(time, time_b, b_meas)

                covariates[PredictionCovariates.TARGET] = b_meas
                covariates[PredictionCovariates.PAST_TARGET_] = b_meas
            else:
                msg = "Buffer must contain field measurements to add target."
                raise ValueError(msg)

        return rename_columns(covariates)

    @staticmethod
    def chain_programs(
        *programs: tuple[np.ndarray, np.ndarray] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Chain multiple LSA programs, shaped (2, N), into a single program.

        This requires shifting the time of each program so that the end of one
        program is the start of the next program.
        """
        if len(programs) < 2:
            msg = "Must provide at least two programs to chain."
            raise ValueError(msg)

        program = [programs[0]]

        for p in programs[1:]:
            # shift the time of the program so that the end of the previous program
            # is the start of the current program
            p = (p[0] + program[-1][0][-1], p[1])
            program.append((p[0][1:], p[1][1:]))

        return (
            np.concatenate([p[0] for p in program], axis=-1).astype(np.float64),
            np.concatenate([p[1] for p in program], axis=-1).astype(np.float64),
        )

    @staticmethod
    def interpolate_program(
        program: tuple[np.ndarray, np.ndarray] | np.ndarray,
        fs: float = 1e3,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Interpolate a program to a new time array.
        """
        time_s, value = program
        time_interp = np.arange(time_s[0], time_s[-1], 1 / fs)

        value_interp = np.interp(time_interp, time_s, value)

        return time_interp, value_interp


class Predictor(
    PredictorHooks,
    PredictorUtils,
    PredictorABC,
    typing.Generic[T_Module_co, T_DataModule_co],
    metaclass=abc.ABCMeta,
):
    """
    Base for all predictors. All predictors should inherit from this class.

    The predictor can load checkpoints from disk, with the
    :meth:`from_checkpoint` class method, or it can be instantiated with a
    model and datamodule directly. After instantiation, the predictor can
    be loaded with a new model and datamodule using the :meth:`load_model`
    or :meth:`load_from_checkpoint` methods.

    The predictor uses lightning.Fabric to reconfigure the model with the
    given device, and it uses the :func:`transformertf.predict.predict`
    function to make predictions.

    The class is thread-safe, and it can be used in a multi-threaded
    environment. The :attr:`busy` attribute can be used to check if the
    predictor is currently busy making a prediction, and the
    :meth:`set_busy` method can be used to set the busy state of the
    predictor.

    Parameters
    ----------
    device : "cpu" | "cuda" | "auto"
        Device to use for prediction, by default "auto".
    """

    state: typing.Any | None

    def __init__(
        self,
        device: typing.Literal["cpu", "cuda", "auto"] = "auto",
    ) -> None:
        self._module: T_Module_co | None = None
        self._datamodule: T_DataModule_co | None = None

        self._fabric = L.fabric.Fabric(accelerator=device)
        self._device = device

        self.lock = threading.Lock()
        self._busy = False
        self._busy_cv = threading.Condition(self.lock)

    @property
    def device(self) -> typing.Literal["cpu", "cuda", "auto"]:
        """
        Device to use for prediction.

        Returns
        -------
        torch.device
            Device to use for prediction.
        """
        return self._device

    @device.setter
    def device(self, value: typing.Literal["cpu", "cuda", "auto"]) -> None:
        """
        Set the device to use for prediction.

        If the predictor is currently busy making a prediction, this method
        will wait for the predictor to become available before changing the
        device. Changing the device requires re-instantiating the fabric and
        reconfiguring the model with the new device.

        Parameters
        ----------
        value : torch.device | str
            Device to use for prediction.

        Returns
        -------
        None
        """
        if isinstance(value, str):
            value = torch.device(value)

        if value != self._device:
            if self.busy:
                log.warning(
                    "Predictor is busy, waiting for it to become available "
                    "before changing device."
                )
                self._busy_cv.wait()

            self._device = value
            self._fabric = L.Fabric(accelerator=str(self._device))

            self._reconfigure_module()

    @property
    def busy(self) -> bool:
        """
        Whether the predictor is currently busy making a prediction.

        Returns
        -------
        bool
            True if the predictor is currently busy making a prediction,
            False otherwise.
        """
        with self.lock:
            return self._busy

    @busy.setter
    def busy(self, value: bool) -> None:
        """
        Set the busy state of the predictor.

        Parameters
        ----------
        value : bool

        Returns
        -------
        None
        """
        with self.lock:
            self._busy = value

            if not value:
                self._busy_cv.notify_all()

    def set_busy(self, value: bool) -> None:  # noqa: FBT001
        """
        Set the busy state of the predictor. Callable version of the
        :attr:`busy` setter.

        Parameters
        ----------
        value : bool
            Busy state to set.

        Returns
        -------
        None
        """
        self.busy = value

    def _reconfigure_module(self) -> None:
        if self._module is not None:
            self._module = self._fabric.setup(self._module)
            self._module.eval()  # type: ignore[union-attr]

    def set_initial_state(self, *args: typing.Any, **kwargs: typing.Any) -> None:
        """
        Set the initial state of the model.

        Parameters
        ----------
        *args
            Positional arguments.
        **kwargs
            Keyword arguments.

        Returns
        -------
        None
        """
        if self._module is None:
            msg = "No model loaded."
            raise NoModelLoadedError(msg)

        self._set_initial_state_impl(*args, **kwargs)

    def reset_state(self) -> None:
        """
        Reset the state of the model. By default, this method sets the state to None.
        """
        self.state = None

    def set_cycled_initial_state(
        self,
        cycles: list[CycleData],
        *args: typing.Any,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> None:
        """
        Set the initial state of the model. This method should be
        implemented by subclasses.

        Parameters
        ----------
        cycles : list[CycleData]
            Cycles to use for setting the initial state. Not all cycles may be used
            if the context length is shorter than the number of cycles.
        *args
            Positional arguments.
        use_programmed_current : bool
            Whether to use the programmed current, by default True. If false, measured
            current will be used instead.
        **kwargs
            Keyword arguments.

        Returns
        -------
        None
        """
        past_covariates = Predictor.buffer_to_covariates(
            cycles,
            use_programmed_current=use_programmed_current,
        )
        past_targets = past_covariates.pop(TARGET_COLNAME).to_numpy()

        self.set_initial_state(
            *args,
            past_covariates=past_covariates,  # type: ignore[misc]
            past_targets=past_targets,  # type: ignore[misc]
            **kwargs,
        )

    def predict(
        self,
        future_covariates: pd.DataFrame,
        *args: typing.Any,
        save_state: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        """
        Predict the next value.

        Parameters
        ----------
        future_covariates : pd.DataFrame
            Future covariates to use for prediction
        *args
            Positional arguments.
        save_state : bool
            Whether to save the state of the model after making the
            prediction, by default True.
        **kwargs
            Keyword arguments.

        Returns
        -------
        np.ndarray
            Prediction of shape [n_points]
        """
        if self._module is None:
            msg = "No model loaded."
            raise NoModelLoadedError(msg)

        self.on_before_predict()
        prediction = self._predict_impl(
            future_covariates, *args, save_state=save_state, **kwargs
        )
        self.on_after_predict()

        return prediction

    def predict_cycle(
        self,
        cycle: CycleData,
        *args: typing.Any,
        save_state: bool = True,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        """
        Predict the next value. The method should be implemented by
        subclasses and update the internal state of the model for the
        next prediction.

        Parameters
        ----------
        cycle : CycleData
            Cycle to use for prediction
        save_state : bool
            Whether to save the state of the model after making the
            prediction, by default True.
        use_programmed_current : bool
            Whether to use the programmed current, by default True.
            If False, measured current will be used instead.
            The current will be filtered before being used for prediction.

        Returns
        -------
        np.ndarray
            Prediction of shape [2, n_points], where n_points is the
            number of points in the prediction. The first row is the
            time axis, and the second row is the prediction.

        Raises
        ------
        RuntimeError
            If the initial state has not been set.
        """
        return self._predict_cycle_impl(
            cycle,
            *args,
            save_state=save_state,
            use_programmed_current=use_programmed_current,
            **kwargs,
        )

    def predict_last_cycle(
        self,
        cycle_data: list[CycleData],
        *args: typing.Any,
        save_state: bool = True,
        autoregressive: bool = False,
        use_programmed_current: bool = True,
        **kwargs: typing.Any,
    ) -> npt.NDArray[np.float64]:
        """
        Predict the field for the last cycle in the given data.

        Uses the :meth:`set_cycled_initial_state` method to set the initial
        state of the model, and then uses the :meth:`predict_cycle` method
        to make the prediction.

        Parameters
        ----------
        cycle_data : list[CycleData]
            The data to predict the field for.
        save_state : bool
            Whether to save the state of the model after making the
            prediction, by default True.
        autoregressive : bool
            Whether to use the autoregressive mode, by default False.
            If True, the predictor will use the saved state of the model
            to make the prediction. If False, the predictor will reset
            the state of the model before making the prediction.
        use_programmed_current : bool
            Whether to use the programmed current, by default True.
            If False, measured current will be used instead.
            The current will be filtered before being used for prediction.

        Returns
        -------
        np.ndarray
            The predicted field of the last cycle. Does not necessarily have the same
            length as the measured field, and may have to be interpolated to match the
            length of the measurements.

        Raises
        ------
        ValueError
            If not all data has input current set.
        """
        if autoregressive or self.state is None:
            self.set_cycled_initial_state(
                cycles=cycle_data[:-1],
                use_programmed_current=use_programmed_current,
            )

        return self.predict_cycle(
            cycle_data[-1],
            save_state=save_state,
            use_programmed_current=use_programmed_current,
            **kwargs,
        )

    @property
    def hparams(self) -> dict[str, typing.Any]:
        """
        Hyperparameters of the model.

        Returns
        -------
        dict[str, Any]
            Hyperparameters of the model.

        Raises
        ------
        NoModelLoadedError
            If no model is loaded.
        """
        if self._module is None or self._datamodule is None:
            msg = "No model loaded."
            raise NoModelLoadedError(msg)

        return dict(self._module.hparams) | dict(self._datamodule.hparams)

    @classmethod
    def load_from_checkpoint(
        cls: type[SameType],
        checkpoint_path: str | os.PathLike,
        device: typing.Literal["cpu", "cuda", "auto"] = "auto",
    ) -> SameType:
        """
        Load a predictor from a checkpoint file.

        Parameters
        ----------
        checkpoint_path : str | os.PathLike
            Path to the checkpoint file.
        device : "cpu" | "cuda" | "auto"
            Device to use for prediction, by default "auto".

        Returns
        -------
        Predictor
            Predictor loaded from the checkpoint file.
        """
        predictor = cls(device=device)
        predictor.load_checkpoint(checkpoint_path)

        return predictor

    def load_checkpoint(
        self,
        checkpoint_path: str | os.PathLike,
    ) -> None:
        """
        Load a checkpoint from disk and run the :meth:`on_after_load_checkpoint`
        method.

        Parameters
        ----------
        checkpoint_path : str | os.PathLike
            Path to the checkpoint file.

        Returns
        -------
        None
        """
        self._load_checkpoint_impl(checkpoint_path)
        self._reconfigure_module()
        self.on_after_load_checkpoint()


def rename_columns(
    df: pd.DataFrame, old_to_new: dict[str, str] | None = None
) -> pd.DataFrame:
    """
    Rename columns in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to rename columns in.
    old_to_new : dict[str, str]
        Dictionary mapping old column names to new column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with renamed columns.
    """
    if old_to_new is None:
        old_to_new = {
            PredictionCovariates.TIME: TIME_COLNAME,
            PredictionCovariates.CURRENT: I_PROG_COLNAME,
            PredictionCovariates.CURRENT_DOT: I_PROG_DOT_COLNAME,
            PredictionCovariates.TARGET: TARGET_COLNAME,
            PredictionCovariates.PAST_TARGET_: TARGET_PAST_COLNAME,
        }
    return df.rename(columns=old_to_new)


def make_prog_base_covariates(
    buffers: list[CycleData],
    *,
    interpolate: bool = True,
    rdp_eps: float = 0.0,
) -> pd.DataFrame:
    """
    Make the covariates for the base model.

    Parameters
    ----------
    buffers : list[CycleData]
        List of CycleData objects.
    interpolate : bool
        Whether to interpolate the programmed current to a new time array,
        by default True.

    Returns
    -------
    pd.DataFrame
        DataFrame with the covariates.
    """
    if len(buffers) == 0:
        msg = "Buffer must contain at least one cycle."
        raise ValueError(msg)
    i_prog_2d: (
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]
        | npt.NDArray[np.float64]
    )
    if len(buffers) == 1:
        i_prog_2d = apply_rdp(prog_to_s(buffers[0].current_prog), epsilon=rdp_eps)
    else:
        i_prog_2d = Predictor.chain_programs(
            *[
                apply_rdp(prog_to_s(cycle.current_prog), epsilon=rdp_eps)
                for cycle in buffers
            ]
        )

    if interpolate:
        t_prog, i_prog = Predictor.interpolate_program(i_prog_2d, fs=1e3)
    else:
        t_prog, i_prog = i_prog_2d

    # NB: we are using the programmed current, which is noise-free
    i_prog_dot = np.gradient(i_prog, t_prog * 1e3)

    return pd.DataFrame(
        {
            TIME_COLNAME: t_prog,
            PredictionCovariates.CURRENT: i_prog,
            PredictionCovariates.CURRENT_DOT: i_prog_dot,
        }
    )


def apply_rdp(
    array: npt.NDArray[np.float64],  # [2, N]
    epsilon: float,
) -> npt.NDArray[np.float64]:
    if epsilon == 0.0:
        return array

    new_array = pybind11_rdp.rdp(array.T, epsilon=epsilon).T
    # add edge points if they were removed
    if new_array[0, 0] != array[0, 0]:
        new_array = np.concatenate([array[:, :1], new_array], axis=1)
    if new_array[0, -1] != array[0, -1]:
        new_array = np.concatenate([new_array, array[:, -1:]], axis=1)
    msg = f"RDP: {array.shape[1]} -> {new_array.shape[1]}"

    log.info(msg)
    return new_array


def prog_to_s(
    prog: npt.NDArray[np.float64],  # [2, N]
) -> npt.NDArray[np.float64]:
    """
    Convert a program to seconds.

    Parameters
    ----------
    prog : np.ndarray
        Program to convert. The first row is the time array, and the second
        row is the value array.

    Returns
    -------
    np.ndarray
        Program converted to seconds.
    """
    return np.vstack((prog[0] / 1e3, prog[1]))


def make_meas_base_covariates(
    buffers: list[CycleData],
    *,
    rdp: typing.Literal[False] | float = False,
) -> pd.DataFrame:
    """
    Make the covariates for the base model.

    Parameters
    ----------
    buffers : list[CycleData]
        List of CycleData objects.
    rdp : float
        Whether to use adaptive downsampling, by default False if 0.0.
        If non-zero, the programmed current will not be interpolated if
        use_programmed_current is True. If zero, the programmed current
        will be used as collocation points to sample the measured current.

    Returns
    -------
    pd.DataFrame
        DataFrame with the covariates.
    """
    if len(buffers) == 0:
        msg = "Buffer must contain at least one cycle."
        raise ValueError(msg)

    i_meas = np.concatenate([cycle.current_input.flatten() for cycle in buffers])
    i_meas_filtered = filter_imeas(i_meas)
    i_meas_filtered_dot = np.gradient(i_meas_filtered, 1.0)

    time_max = sum(cycle.num_samples for cycle in buffers)
    time = np.arange(time_max) / 1e3

    if len(time) != len(i_meas):
        msg = f"Wrong array lengths: {len(time)} != {len(i_meas)}"
        log.error(msg)

    if isinstance(rdp, float):
        prog_covariates = make_prog_base_covariates(
            buffers, interpolate=False, rdp_eps=rdp
        )
        time_prog = prog_covariates[TIME_COLNAME].to_numpy()

        i_meas_filtered = np.interp(time_prog, time, i_meas_filtered)
        i_meas_filtered_dot = np.interp(time_prog, time, i_meas_filtered_dot)
        time = time_prog

    return pd.DataFrame(
        {
            TIME_COLNAME: time,
            PredictionCovariates.CURRENT: i_meas_filtered,
            PredictionCovariates.CURRENT_DOT: i_meas_filtered_dot,
        }
    )


def filter_imeas(
    i_meas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Filter the measured current.

    Parameters
    ----------
    i_meas : np.ndarray
        Measured current.

    Returns
    -------
    np.ndarray
        Filtered measured current.
    """
    i_meas = signal.butter_lowpass_filter(i_meas, cutoff=80, fs=1e3, order=1)
    return signal.mean_filter(i_meas, window_size=151, stride=1, threshold=0.06 / 2)


def filter_bmeas(
    b_meas: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Filter the measured field.

    Parameters
    ----------
    b_meas : np.ndarray
        Measured field.

    Returns
    -------
    np.ndarray
        Filtered measured field.
    """
    b_meas = signal.butter_lowpass_filter(b_meas, cutoff=80, fs=1e3, order=1)
    return signal.mean_filter(b_meas, window_size=151, stride=1, threshold=1.5e-5 / 2)
