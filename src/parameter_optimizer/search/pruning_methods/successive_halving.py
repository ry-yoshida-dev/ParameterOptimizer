from dataclasses import dataclass

from optuna.pruners import BasePruner, SuccessiveHalvingPruner

from ..pruning_method import PruningMethod, pruning_method_registry


@pruning_method_registry.register("successive_halving")
@dataclass(frozen=True, kw_only=True)
class SuccessiveHalvingPruning(PruningMethod):
    """
    Repeatedly keep only the better part of the trials and give them more steps.

    Attributes
    ----------
    min_resource : int | None
        Steps of the first round; None lets Optuna choose.
    reduction_factor : int
        Inverse of the part kept after each round.
    min_early_stopping_rate : int
        Rounds skipped before the first check.
    """

    min_resource: int | None = None
    reduction_factor: int = 4
    min_early_stopping_rate: int = 0

    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        return SuccessiveHalvingPruner(
            min_resource="auto" if self.min_resource is None else self.min_resource,
            reduction_factor=self.reduction_factor,
            min_early_stopping_rate=self.min_early_stopping_rate,
        )
