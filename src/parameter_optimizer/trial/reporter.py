from optuna import Trial, TrialPruned


class TrialReporter:
    """
    Lets an objective report intermediate values so that the pruner can stop the trial early.

    Parameters
    ----------
    trial : Trial
        Trial being evaluated.
    """

    def __init__(self, trial: Trial) -> None:
        self._trial: Trial = trial
        self._is_reported: bool = False

    @property
    def is_reported(self) -> bool:
        """
        Whether the objective has reported an intermediate value.

        Returns
        -------
        bool
            True after the first call of ``report``.
        """
        return self._is_reported

    def report(self, value: float, step: int) -> None:
        """
        Record an intermediate objective value.

        Parameters
        ----------
        value : float
            Objective value at this step.
        step : int
            Progress of the evaluation, such as an epoch.

        Raises
        ------
        TrialPruned
            If the pruner decides to stop this trial; let it propagate out of the objective.
        """
        self._is_reported = True
        self._trial.report(value, step)
        if self._trial.should_prune():
            raise TrialPruned(f"Pruned at step {step} with value {value}.")
