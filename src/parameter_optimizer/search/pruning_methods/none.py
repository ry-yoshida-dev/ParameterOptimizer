from dataclasses import dataclass
from typing import ClassVar

from optuna.pruners import BasePruner, NopPruner

from ..pruning_method import PruningMethod, pruning_method_registry


@pruning_method_registry.register("none")
@dataclass(frozen=True, kw_only=True)
class NoPruning(PruningMethod):
    """Every trial runs to completion."""

    is_stopping_trials: ClassVar[bool] = False

    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        return NopPruner()
