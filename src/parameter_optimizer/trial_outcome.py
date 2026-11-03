from dataclasses import dataclass

from omegaconf import DictConfig

from .parameter import ParameterValue


@dataclass(frozen=True, kw_only=True)
class TrialOutcome[ConfigT]:
    """
    One of the best trials of an optimization.

    Attributes
    ----------
    config : ConfigT
        Object built from the parameters.
    resolved_config : DictConfig
        Base configuration with the parameters written in; usable as a normal configuration.
    values : list[float]
        Objective values, one per direction.
    parameters : dict[str, ParameterValue]
        Values keyed by parameter path.
    """

    config: ConfigT
    resolved_config: DictConfig
    values: list[float]
    parameters: dict[str, ParameterValue]

    @property
    def value(self) -> float:
        """
        Objective value of a single-objective optimization.

        Returns
        -------
        float
            The only objective value.

        Raises
        ------
        ValueError
            If there are several objectives.
        """
        if len(self.values) != 1:
            raise ValueError(f"Expected a single objective value. got {self.values}")
        return self.values[0]
