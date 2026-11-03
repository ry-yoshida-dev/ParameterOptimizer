from abc import ABC, abstractmethod
from typing import ClassVar

from optuna import Trial
from optuna.distributions import BaseDistribution

from ..registry import Registry
from .value import ParameterValue


class ParameterSpec(ABC):
    """
    Search definition of one parameter.

    Attributes
    ----------
    is_numeric : ClassVar[bool]
        Whether the values are ordered numbers, as numeric search methods require.
    """

    is_numeric: ClassVar[bool] = True

    @property
    def grid_values(self) -> list[ParameterValue] | None:
        """
        Values a grid search tries.

        Returns
        -------
        list[ParameterValue] | None
            Every value, or None if this parameter cannot be enumerated.
        """
        return None

    @property
    @abstractmethod
    def probe_values(self) -> list[ParameterValue]:
        """
        Values that are checked against the target before any trial runs.

        Returns
        -------
        list[ParameterValue]
            Values that are representative of this parameter; the first one must not depend on anything but the spec.
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

        An implementation must record one Optuna parameter under ``name`` with ``to_optuna_distribution``, and may
        return a transformation of it.

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


parameter_spec_registry: Registry[ParameterSpec] = Registry(base_class=ParameterSpec, key="distribution")
