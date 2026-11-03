from dataclasses import dataclass

from omegaconf import DictConfig
from optuna import Study

from .parameter import ParameterValue


@dataclass(frozen=True, kw_only=True)
class OptimizationResult[ConfigT]:
    """
    Outcome of an optimization.

    Attributes
    ----------
    best_config : ConfigT
        Dataclass built from the best parameters.
    resolved_config : DictConfig
        Base configuration with the best parameters written in; usable as a normal configuration.
    best_value : float
        Objective value of the best trial.
    best_parameters : dict[str, ParameterValue]
        Best values keyed by parameter path.
    study : Study
        Underlying study, for analysis and visualization.
    """

    best_config: ConfigT
    resolved_config: DictConfig
    best_value: float
    best_parameters: dict[str, ParameterValue]
    study: Study
