from collections.abc import Mapping, Sequence

from optuna import Study
from optuna.samplers import GridSampler
from optuna.trial import FrozenTrial, TrialState

from ..parameter import ParameterValue


class ResumableGridSampler(GridSampler):
    """
    Grid sampler that retries, on resume, the points whose trials failed.

    ``GridSampler`` counts a failed trial as visited, so a resumed grid would finish without ever evaluating that point.

    Parameters
    ----------
    candidates : Mapping[str, list[ParameterValue]]
        Candidates keyed by parameter path.
    seed : int
        Seed of the grid order; it must stay the same across the runs of a study.
    """

    def __init__(self, candidates: Mapping[str, list[ParameterValue]], seed: int) -> None:
        super().__init__(dict(candidates), seed=seed)
        self._candidates: dict[str, list[ParameterValue]] = dict(candidates)

    def enqueue_failed_points(self, study: Study) -> None:
        """
        Queue each point of this grid that failed and has neither succeeded nor been queued since.

        Parameters
        ----------
        study : Study
            Study being resumed.
        """
        trials: list[FrozenTrial] = study.get_trials(deepcopy=False)
        settled_points: set[tuple[tuple[str, ParameterValue], ...]] = {
            self._point_key(self._point_of(trial))
            for trial in trials
            if trial.state in (TrialState.COMPLETE, TrialState.PRUNED, TrialState.WAITING)
        }
        failed_points: dict[tuple[tuple[str, ParameterValue], ...], dict[str, ParameterValue]] = {
            self._point_key(trial.params): trial.params
            for trial in trials
            if trial.state is TrialState.FAIL and self._is_in_grid(trial.params)
        }
        for key, point in failed_points.items():
            if key not in settled_points:
                study.enqueue_trial(point)

    def is_finished(self, study: Study) -> bool:
        """
        Whether no point remains to be evaluated.

        Parameters
        ----------
        study : Study
            Study being resumed.

        Returns
        -------
        bool
            True if the grid is exhausted and no retry is queued.
        """
        return self.is_exhausted(study) and not self._has_queued_trial(study)

    def after_trial(
        self,
        study: Study,
        trial: FrozenTrial,
        state: TrialState,
        values: Sequence[float] | None,
    ) -> None:
        """
        Stop the study once the grid is exhausted and no retry is queued.

        Parameters
        ----------
        study : Study
            Study being optimized.
        trial : FrozenTrial
            Trial that just finished.
        state : TrialState
            Resulting state of the trial.
        values : Sequence[float] | None
            Resulting objective values of the trial.
        """
        if self._has_queued_trial(study):
            return
        # A retried trial has no grid id, which GridSampler.after_trial requires when one point is left.
        if "grid_id" in trial.system_attrs:
            super().after_trial(study, trial, state, values)
        elif self.is_exhausted(study):
            study.stop()

    def _is_in_grid(self, point: Mapping[str, ParameterValue]) -> bool:
        return set(point) == set(self._candidates) and all(
            value in self._candidates[path] for path, value in point.items()
        )

    @staticmethod
    def _point_of(trial: FrozenTrial) -> dict[str, ParameterValue]:
        # A queued trial holds its values in a system attribute until it starts.
        fixed_params: dict[str, ParameterValue] | None = trial.system_attrs.get("fixed_params")
        return trial.params if fixed_params is None else fixed_params

    @staticmethod
    def _point_key(point: Mapping[str, ParameterValue]) -> tuple[tuple[str, ParameterValue], ...]:
        return tuple(sorted(point.items(), key=lambda item: item[0]))

    @staticmethod
    def _has_queued_trial(study: Study) -> bool:
        return bool(study.get_trials(deepcopy=False, states=(TrialState.WAITING,)))
