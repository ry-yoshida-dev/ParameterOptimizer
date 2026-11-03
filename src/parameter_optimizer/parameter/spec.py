from abc import ABC, abstractmethod

from optuna import Trial
from optuna.distributions import BaseDistribution

from .value import ParameterValue


class ParameterSpec(ABC):
    """Search definition of one parameter."""

    @property
    @abstractmethod
    def probe_values(self) -> list[ParameterValue]:
        """
        Values that are checked against the target dataclass before any trial runs.

        Returns
        -------
        list[ParameterValue]
            Every candidate, or both bounds of a range.
        """

    @abstractmethod
    def to_optuna_distribution(self) -> BaseDistribution:
        """
        Convert to the distribution Optuna records for this parameter.

        Returns
        -------
        BaseDistribution
            Equivalent Optuna distribution.
        """

    @abstractmethod
    def suggest(self, trial: Trial, name: str) -> ParameterValue:
        """
        Ask a trial for a value of this parameter.

        Parameters
        ----------
        trial : Trial
            Trial choosing the value.
        name : str
            Parameter name recorded in the study.

        Returns
        -------
        ParameterValue
            Chosen value.
        """
