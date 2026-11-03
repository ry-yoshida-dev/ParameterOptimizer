import math
from dataclasses import dataclass
from typing import cast

from optuna import Trial
from optuna.distributions import BaseDistribution, FloatDistribution

from ..spec import ParameterSpec
from ..value import ParameterValue


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

    Raises
    ------
    TypeError
        If a bound is not a float.
    ValueError
        If a bound is not finite, the range is empty, or a logarithmic range does not start above zero.
    """

    low: float
    high: float
    is_log: bool = False

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

    @property
    def probe_values(self) -> list[ParameterValue]:
        """
        Values that are checked against the target dataclass before any trial runs.

        Returns
        -------
        list[ParameterValue]
            Both bounds.
        """
        return [self.low, self.high]

    def to_optuna_distribution(self) -> BaseDistribution:
        """
        Convert to the distribution Optuna records for this parameter.

        Returns
        -------
        BaseDistribution
            Equivalent Optuna distribution.
        """
        return FloatDistribution(self.low, self.high, log=self.is_log)

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
        return trial.suggest_float(name, self.low, self.high, log=self.is_log)
