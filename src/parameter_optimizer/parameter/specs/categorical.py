from dataclasses import dataclass
from typing import ClassVar, cast

from optuna import Trial
from optuna.distributions import BaseDistribution, CategoricalDistribution

from ..spec import ParameterSpec, parameter_spec_registry
from ..value import ParameterValue


@parameter_spec_registry.register("categorical")
@dataclass(frozen=True, kw_only=True)
class CategoricalSpec(ParameterSpec):
    """
    Parameter taking one of the listed candidates.

    Attributes
    ----------
    candidates : list[ParameterValue]
        Values to choose from.

    Raises
    ------
    TypeError
        If ``candidates`` is not a list of scalars.
    ValueError
        If the candidates are empty or duplicated.
    """

    is_numeric: ClassVar[bool] = False

    candidates: list[ParameterValue]

    def __post_init__(self) -> None:
        raw_candidates: object = cast(object, self.candidates)
        if not isinstance(raw_candidates, list):
            raise TypeError(f"candidates must be a list. got {type(raw_candidates).__name__}")
        if not self.candidates:
            raise ValueError("categorical requires at least one candidate.")
        for candidate in cast("list[object]", self.candidates):
            if not isinstance(candidate, bool | int | float | str | None):
                raise TypeError(f"Candidates must be scalars. got {type(candidate).__name__}: {candidate}")
        if len(set(self.candidates)) != len(self.candidates):
            raise ValueError(f"categorical candidates must be unique. got {self.candidates}")

    @property
    def grid_values(self) -> list[ParameterValue] | None:
        """
        Values a grid search tries.

        Returns
        -------
        list[ParameterValue] | None
            Every candidate.
        """
        return list(self.candidates)

    @property
    def probe_values(self) -> list[ParameterValue]:
        """
        Values that are checked against the target dataclass before any trial runs.

        Returns
        -------
        list[ParameterValue]
            Every candidate.
        """
        return list(self.candidates)

    def to_optuna_distribution(self) -> BaseDistribution:
        """
        Convert to the distribution Optuna records for this parameter.

        Returns
        -------
        BaseDistribution
            Equivalent Optuna distribution.
        """
        return CategoricalDistribution(self.candidates)

    def suggest(self, trial: Trial, name: str) -> ParameterValue:
        """
        Ask a trial for one of the candidates.

        Parameters
        ----------
        trial : Trial
            Trial choosing the value.
        name : str
            Parameter name recorded in the study.

        Returns
        -------
        ParameterValue
            Chosen candidate.
        """
        return trial.suggest_categorical(name, self.candidates)
