from dataclasses import dataclass

from optuna.pruners import BasePruner, MedianPruner

from ..pruning_method import PruningMethod, pruning_method_registry


@pruning_method_registry.register("median")
@dataclass(frozen=True, kw_only=True)
class MedianPruning(PruningMethod):
    """
    Stop a trial that falls behind the median of earlier trials at the same step.

    Attributes
    ----------
    startup_trial_count : int
        Trials that always run to completion.
    warmup_step_count : int
        Steps of each trial that are never pruned.
    step_interval : int
        Steps between two checks.
    """

    startup_trial_count: int = 5
    warmup_step_count: int = 0
    step_interval: int = 1

    def build(self) -> BasePruner:
        """
        Create the pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        return MedianPruner(self.startup_trial_count, self.warmup_step_count, self.step_interval)
