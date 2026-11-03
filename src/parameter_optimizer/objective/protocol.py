from typing import Protocol

from .reporter import TrialReporter


class Objective[ConfigT](Protocol):
    """Evaluates one configuration; expensive resources should be cached by the implementation between calls."""

    def __call__(self, config: ConfigT, reporter: TrialReporter, /) -> float:
        """
        Score a configuration.

        Parameters
        ----------
        config : ConfigT
            Dataclass built for this trial.
        reporter : TrialReporter
            Receives intermediate values; may be ignored when pruning is not used.

        Returns
        -------
        float
            Objective value.
        """
        ...
