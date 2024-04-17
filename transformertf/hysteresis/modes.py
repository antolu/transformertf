from __future__ import annotations

from torch.nn import Module

FITTING = 0
REGRESSION = 1
NEXT = 2
FUTURE = 3
CURRENT = 4


class ModeModule(Module):
    _mode = FITTING

    @property
    def mode(self) -> int:
        return self._mode

    @mode.setter
    def mode(self, value: int) -> None:
        assert value in {REGRESSION, NEXT, FUTURE, FITTING, CURRENT}
        self._mode = value

        # if mode is FITTING set module to training.py
        if value == FITTING:
            self.train()
        else:
            self.eval()

        for ele in self.children():
            if isinstance(ele, ModeModule):
                ele.mode = value

    def fitting(self) -> None:
        self.mode = FITTING

    def regression(self) -> None:
        self.mode = REGRESSION

    def next(self) -> None:
        self.mode = NEXT

    def future(self) -> None:
        self.mode = FUTURE

    def current(self) -> None:
        self.mode = CURRENT
