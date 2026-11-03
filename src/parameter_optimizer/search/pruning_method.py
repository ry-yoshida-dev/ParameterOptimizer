from abc import ABC, abstractmethod
from typing import ClassVar

from optuna.pruners import BasePruner

from ..registry import Registry


class PruningMethod(ABC):
    """
    Rule stopping unpromising trials early from the values reported by the objective.

    It only decides how long a trial runs, so it combines with any ``SearchMethod``. Without reported values, a trial
    runs to completion and ``ParameterOptimizer`` warns. Decorating a dataclass implementation with
    ``pruning_method_registry.register`` makes it selectable from a configuration.

    Attributes
    ----------
    is_stopping_trials : ClassVar[bool]
        Whether this rule can stop a trial, which requires a single objective and reported values.
    """

    is_stopping_trials: ClassVar[bool] = True

    @abstractmethod
    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """


pruning_method_registry: Registry[PruningMethod] = Registry(base_class=PruningMethod, key="name")
