import math
from dataclasses import dataclass
from typing import cast

from optuna import Trial
from optuna.distributions import BaseDistribution, FloatDistribution

from ..spec import ParameterSpec, parameter_spec_registry
from ..value import ParameterValue


@parameter_spec_registry.register("float")
@dataclass(frozen=True, kw_only=True)
class FloatSpec(ParameterSpec):
    """
    Parameter taking a float within a range.

    Attributes
    ----------
    low : float
        Lower bound, inclusive.
    high : float
        Upper bound, inclusive.
    is_log : bool
        Whether the range is explored on a logarithmic scale.
    step : float | None
        Distance between the values, or None for a continuous range. Optuna lowers ``high`` to the last value it
        reaches.

    Raises
    ------
    TypeError
        If a bound is not a float.
    ValueError
        If a bound is not finite, the range is empty, a logarithmic range does not start above zero or has a step, or
        the step is not positive and finite.
    """

    low: float
    high: float
    is_log: bool = False
    step: float | None = None

    def __post_init__(self) -> None:
        for bound in (cast(object, self.low), cast(object, self.high)):
            if not isinstance(bound, float):
                raise TypeError(f"float bounds must be float. got {type(bound).__name__}: {bound}")
            if not math.isfinite(bound):
                raise ValueError(f"float bounds must be finite. got {bound}")
        if self.low >= self.high:
            raise ValueError(f"float requires low < high. got low={self.low}, high={self.high}")
        if self.is_log and self.low <= 0.0:
            raise ValueError(f"A logarithmic float requires a positive low. got {self.low}")
        if self.step is not None:
            if self.is_log:
                raise ValueError("A logarithmic float cannot have a step.")
            if not math.isfinite(self.step) or self.step <= 0.0:
                raise ValueError(f"float step must be positive and finite. got {self.step}")

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
        high: float = self._distribution.high
        value_count: int = round((high - self.low) / self.step) + 1
        return [min(self.low + index * self.step, high) for index in range(value_count)]

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
        Ask a trial for a float within the range.

        Parameters
        ----------
        trial : Trial
            Trial choosing the value.
        name : str
            Parameter name recorded in the study.

        Returns
        -------
        ParameterValue
            Chosen float.
        """
        return trial.suggest_float(name, self.low, self.high, log=self.is_log, step=self.step)

    @property
    def _distribution(self) -> FloatDistribution:
        return FloatDistribution(self.low, self.high, log=self.is_log, step=self.step)
