from dataclasses import dataclass

from optuna.pruners import BasePruner, HyperbandPruner

from ..pruning_method import PruningMethod, pruning_method_registry


@pruning_method_registry.register("hyperband")
@dataclass(frozen=True, kw_only=True)
class HyperbandPruning(PruningMethod):
    """
    Run successive halving with several budgets, so that the rate of elimination need not be chosen.

    Attributes
    ----------
    min_resource : int
        Fewest steps a trial receives.
    max_resource : int | None
        Most steps a trial receives; None lets Optuna choose.
    reduction_factor : int
        Inverse of the part kept after each round.
    """

    min_resource: int = 1
    max_resource: int | None = None
    reduction_factor: int = 3

    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        return HyperbandPruner(
            min_resource=self.min_resource,
            max_resource="auto" if self.max_resource is None else self.max_resource,
            reduction_factor=self.reduction_factor,
        )
