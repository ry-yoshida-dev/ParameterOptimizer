from dataclasses import dataclass
from typing import cast

from optuna import Trial
from optuna.distributions import BaseDistribution, IntDistribution

from ..spec import ParameterSpec
from ..value import ParameterValue


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

    Raises
    ------
    TypeError
        If a bound is not an int; a bool is not accepted.
    ValueError
        If the range is empty, or a logarithmic range starts below one.
    """

    low: int
    high: int
    is_log: bool = False

    def __post_init__(self) -> None:
        for bound in (cast(object, self.low), cast(object, self.high)):
            if type(bound) is not int:
                raise TypeError(f"int bounds must be int. got {type(bound).__name__}: {bound}")
        if self.low >= self.high:
            raise ValueError(f"int requires low < high. got low={self.low}, high={self.high}")
        if self.is_log and self.low < 1:
            raise ValueError(f"A logarithmic int requires a low of at least 1. got {self.low}")

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
        return IntDistribution(self.low, self.high, log=self.is_log)

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
        return trial.suggest_int(name, self.low, self.high, log=self.is_log)
