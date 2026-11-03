from dataclass_initializer import DataclassInitializer
from omegaconf import DictConfig


class DataclassBuilder[ConfigT]:
    """
    Builds a dataclass the same strict way a normal configuration is built.

    Parameters
    ----------
    config_class : type[ConfigT]
        Dataclass to build.
    """

    def __init__(self, config_class: type[ConfigT]) -> None:
        self._config_class: type[ConfigT] = config_class

    def __call__(self, config: DictConfig, /) -> ConfigT:
        """
        Build the dataclass.

        Parameters
        ----------
        config : DictConfig
            Plain configuration of the dataclass.

        Returns
        -------
        ConfigT
            New dataclass.

        Raises
        ------
        TypeError
            If a value has the wrong type for its field.
        ValueError
            If a key is unknown or a value is rejected by the dataclass.
        """
        return DataclassInitializer.build(self._config_class, config)
