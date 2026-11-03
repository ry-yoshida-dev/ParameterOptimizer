from enum import Enum

from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner, SuccessiveHalvingPruner


class PruningMethod(Enum):
    """
    Rule stopping unpromising trials early from the values reported by the objective.

    It only decides how long a trial runs, so it combines with any ``SearchMethod``. Without reported values, a trial
    runs to completion and ``ParameterOptimizer`` warns.

    Attributes
    ----------
    NONE : str
        Every trial runs to completion.
    MEDIAN : str
        Stop a trial that falls behind the median of earlier trials at the same step.
    SUCCESSIVE_HALVING : str
        Repeatedly keep only the better part of the trials and give them more steps.
    HYPERBAND : str
        Run successive halving with several budgets, so that the rate of elimination need not be chosen.
    """

    NONE = "none"
    MEDIAN = "median"
    SUCCESSIVE_HALVING = "successive_halving"
    HYPERBAND = "hyperband"

    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        match self:
            case PruningMethod.NONE:
                return NopPruner()
            case PruningMethod.MEDIAN:
                return MedianPruner()
            case PruningMethod.SUCCESSIVE_HALVING:
                return SuccessiveHalvingPruner()
            case PruningMethod.HYPERBAND:
                return HyperbandPruner()
