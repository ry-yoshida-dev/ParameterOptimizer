from dataclasses import dataclass
from typing import cast

from optuna import Trial
from optuna.distributions import BaseDistribution, IntDistribution

from ..spec import ParameterSpec, parameter_spec_registry
from ..value import ParameterValue


@parameter_spec_registry.register("int")
@dataclass(frozen=True, kw_only=True)
class IntSpec(ParameterSpec):
    """
    Parameter taking an integer within a range.

    Attributes
    ----------
    low : int
        Lower bound, inclusive.
    high : int
        Upper bound, inclusive.
    is_log : bool
        Whether the range is explored on a logarithmic scale.
    step : int | None
        Distance between the values, or None for every integer. A grid search requires it. Optuna lowers ``high`` to the
        last value it reaches.

    Raises
    ------
    TypeError
        If a bound is not an int; a bool is not accepted.
    ValueError
        If the range is empty, a logarithmic range starts below one or has a step, or the step is not positive.
    """

    low: int
    high: int
    is_log: bool = False
    step: int | None = None

    def __post_init__(self) -> None:
        for bound in (cast(object, self.low), cast(object, self.high)):
            if type(bound) is not int:
                raise TypeError(f"int bounds must be int. got {type(bound).__name__}: {bound}")
        if self.low >= self.high:
            raise ValueError(f"int requires low < high. got low={self.low}, high={self.high}")
        if self.is_log and self.low < 1:
            raise ValueError(f"A logarithmic int requires a low of at least 1. got {self.low}")
        if self.step is not None:
            if self.is_log:
                raise ValueError("A logarithmic int cannot have a step.")
            if self.step <= 0:
                raise ValueError(f"int step must be positive. got {self.step}")

    @property
    def grid_values(self) -> list[ParameterValue] | None:
        """
        Values a grid search tries.

        Returns
        -------
        list[ParameterValue] | None
            Every step of the range, or None without a step.
        """
        if self.step is None:
            return None
        return list(range(self.low, self.high + 1, self.step))

    @property
    def probe_values(self) -> list[ParameterValue]:
        """
        Values that are checked against the target dataclass before any trial runs.

        Returns
        -------
        list[ParameterValue]
            Both bounds.
        """
        return [self.low, self._distribution.high]

    def to_optuna_distribution(self) -> BaseDistribution:
        """
        Convert to the distribution Optuna records for this parameter.

        Returns
        -------
        BaseDistribution
            Equivalent Optuna distribution.
        """
        return self._distribution

    def suggest(self, trial: Trial, name: str) -> ParameterValue:
        """
        Ask a trial for an integer within the range.

        Parameters
        ----------
        trial : Trial
            Trial choosing the value.
        name : str
            Parameter name recorded in the study.

        Returns
        -------
        ParameterValue
            Chosen integer.
        """
        return trial.suggest_int(name, self.low, self.high, step=self._step, log=self.is_log)

    @property
    def _step(self) -> int:
        return 1 if self.step is None else self.step

    @property
    def _distribution(self) -> IntDistribution:
        return IntDistribution(self.low, self.high, log=self.is_log, step=self._step)
