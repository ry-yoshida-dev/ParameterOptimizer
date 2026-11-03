from typing import Protocol

from omegaconf import DictConfig


class ConfigBuilder[ConfigT](Protocol):
    """Turns a plain configuration into the object an objective receives."""

    def __call__(self, config: DictConfig, /) -> ConfigT:
        """
        Build the object of one set of values.

        Parameters
        ----------
        config : DictConfig
            Plain configuration with no search definition or interpolation left.

        Returns
        -------
        ConfigT
            Object handed to the objective.

        Raises
        ------
        TypeError
            If a value has the wrong type; construction fails at once.
        ValueError
            If the values are rejected; construction tries the other parameters' values before it fails.
        """
        ...
