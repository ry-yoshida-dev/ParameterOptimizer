from dataclasses import dataclass

from optuna import Study

from .trial_outcome import TrialOutcome


@dataclass(frozen=True, kw_only=True)
class OptimizationResult[ConfigT]:
    """
    Outcome of an optimization.

    Attributes
    ----------
    best_trials : list[TrialOutcome[ConfigT]]
        The best trial, or the Pareto front with several objectives.
    study : Study
        Underlying study, for analysis and visualization.
    """

    best_trials: list[TrialOutcome[ConfigT]]
    study: Study

    @property
    def best(self) -> TrialOutcome[ConfigT]:
        """
        The only best trial, which a single-objective optimization always has.

        Returns
        -------
        TrialOutcome[ConfigT]
            Best trial.

        Raises
        ------
        ValueError
            If the Pareto front holds several trials; choose among ``best_trials``.
        """
        if len(self.best_trials) != 1:
            raise ValueError(f"The Pareto front holds {len(self.best_trials)} trials; choose among best_trials.")
        return self.best_trials[0]
