from enum import Enum

from dataclass_initializer import DataclassInitializer

from .spec import ParameterSpec
from .specs import CategoricalSpec, FloatSpec, IntSpec


class ParameterDistribution(Enum):
    """
    Kind of values one parameter can take, written as ``distribution`` in a search space.

    Attributes
    ----------
    CATEGORICAL : str
        One of the listed candidates, written with ``candidates``.
    FLOAT : str
        A float within a range, written with ``low``, ``high`` and optionally ``is_log``.
    INT : str
        An integer within a range, written with ``low``, ``high`` and optionally ``is_log``.
    """

    CATEGORICAL = "categorical"
    FLOAT = "float"
    INT = "int"

    def build_spec(self, fields: dict[str, object]) -> ParameterSpec:
        """
        Create the search definition from the other keys of a search space entry.

        Parameters
        ----------
        fields : dict[str, object]
            Entry without its ``distribution`` key.

        Returns
        -------
        ParameterSpec
            Validated search definition.

        Raises
        ------
        TypeError
            If a field has the wrong type.
        ValueError
            If a field is missing, unknown or rejected by the search definition.
        """
        match self:
            case ParameterDistribution.CATEGORICAL:
                return DataclassInitializer.build(CategoricalSpec, fields)
            case ParameterDistribution.FLOAT:
                return DataclassInitializer.build(FloatSpec, fields)
            case ParameterDistribution.INT:
                return DataclassInitializer.build(IntSpec, fields)
