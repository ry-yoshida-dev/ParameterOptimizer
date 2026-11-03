import hashlib
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence

import optuna
from omegaconf import DictConfig, OmegaConf
from optuna import Study, Trial
from optuna.distributions import BaseDistribution
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState

from .builder import ConfigBuilder
from .objective import Objective, TrialReporter
from .parameter import ParameterValue
from .result import OptimizationResult
from .search import SearchSpace
from .settings import OptimizerSettings
from .trial_outcome import TrialOutcome


class ParameterOptimizer[ConfigT]:
    """
    Tunes anything built from a configuration by overlaying a search space on its base configuration.

    Every candidate and bound is checked at construction by calling ``config_builder``, so mistakes in the search space
    fail before any trial runs. A value passes if it builds next to any tried values of the other parameters.

    Parameters
    ----------
    config_builder : ConfigBuilder[ConfigT]
        Builds the object of each trial, such as ``DataclassBuilder(ModelConfig)``.
    base_config : DictConfig
        Normal configuration of the object.
    search_space : DictConfig
        Same hierarchy as ``base_config``, containing only the keys to tune, each as a
        definition with a ``distribution`` key.
    settings : OptimizerSettings
        How the optimization runs.

    Raises
    ------
    ModuleNotFoundError
        If an optional dependency of the search method is not installed.
    TypeError
        If a probed value has the wrong type for its field.
    ValueError
        If the search space is invalid, a key is unknown to ``config_builder``, a probed value is rejected by it next to
        every tried combination, or the search method does not support the parameters or the number of objectives.
    """

    _BASE_CONFIG_DIGEST_ATTRIBUTE: str = "parameter_optimizer.base_config_digest"
    _VALUES_ATTRIBUTE: str = "parameter_optimizer.values"

    def __init__(
        self,
        *,
        config_builder: ConfigBuilder[ConfigT],
        base_config: DictConfig,
        search_space: DictConfig,
        settings: OptimizerSettings,
    ) -> None:
        self._config_builder: ConfigBuilder[ConfigT] = config_builder
        self._base_config: DictConfig = base_config
        self._search_space: SearchSpace = SearchSpace.from_config(search_space)
        self._settings: OptimizerSettings = settings
        self._settings.search_method.validate(self._search_space, len(self._settings.directions))
        if settings.is_probing_enabled:
            for probe_group in self._search_space.iter_probe_groups():
                self._require_buildable(probe_group)

    def optimize(
        self,
        objective: Objective[ConfigT],
        caught_exceptions: tuple[type[Exception], ...] = (),
        callbacks: Sequence[Callable[[Study, FrozenTrial], None]] = (),
    ) -> OptimizationResult[ConfigT]:
        """
        Run the trials and return the best configurations.

        Parameters
        ----------
        objective : Objective[ConfigT]
            Scores each configuration.
        caught_exceptions : tuple[type[Exception], ...]
            Exceptions that mark only the current trial as failed; any other exception stops the run.
        callbacks : Sequence[Callable[[Study, FrozenTrial], None]]
            Called after each trial, for logging or for stopping the study.

        Returns
        -------
        OptimizationResult[ConfigT]
            Best configurations, their scores and the study.

        Raises
        ------
        RuntimeError
            If no trial completed.
        ValueError
            If a resumed study has other directions, another search space or another base configuration.
        """
        study: Study = optuna.create_study(
            storage=self._settings.storage,
            pruner=self._settings.pruning_method.build(),
            study_name=self._settings.study_name,
            directions=[direction.to_study_direction() for direction in self._settings.directions],
            load_if_exists=self._settings.storage is not None,
        )
        self._validate_resumed_study(study)
        seed: int | None = self._settings.seed_after(len(study.get_trials(deepcopy=False)))
        if not self._settings.search_method.prepare(study, self._search_space, seed):
            return self._summarize(study)

        def evaluate(trial: Trial) -> float | Sequence[float]:
            values: dict[str, ParameterValue] = self._search_space.suggest(trial)
            # A parameter specification may transform what Optuna records, so the recorded parameters cannot rebuild it.
            trial.set_user_attr(self._VALUES_ATTRIBUTE, values)
            config: ConfigT = self._build_config(values)
            reporter: TrialReporter = TrialReporter(trial)
            objective_value: float | Sequence[float] = objective(config, reporter)
            self._warn_if_pruning_is_ineffective(reporter)
            return objective_value

        study.optimize(
            evaluate,
            n_trials=self._settings.n_trials,
            timeout=self._settings.timeout_seconds,
            n_jobs=self._settings.n_jobs,
            catch=caught_exceptions,
            callbacks=callbacks,
        )
        return self._summarize(study)

    def _validate_resumed_study(self, study: Study) -> None:
        # Optuna keeps the stored directions and accepts other ranges without complaint, which would silently optimize
        # the opposite way or mix incomparable trials.
        expected_directions: list[StudyDirection] = [
            direction.to_study_direction() for direction in self._settings.directions
        ]
        if study.directions != expected_directions:
            raise ValueError(f"Study {study.study_name} has directions {[item.name for item in study.directions]}.")
        expected_distributions: dict[str, BaseDistribution] = self._search_space.optuna_distributions
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)):
            if trial.distributions != expected_distributions:
                raise ValueError(
                    f"Study {study.study_name} has trials of another search space: {trial.distributions}. "
                    + f"Expected {expected_distributions}."
                )
        # Trials of another base configuration would be reported as scores of a configuration that never ran them.
        digest: str = self._base_config_digest()
        stored_digest: str | None = study.user_attrs.get(self._BASE_CONFIG_DIGEST_ATTRIBUTE)
        if stored_digest != digest and (stored_digest is not None or study.get_trials(deepcopy=False)):
            raise ValueError(
                f"Study {study.study_name} was started with another base configuration or search space. "
                + "List the keys that may change in resume_ignored_paths."
            )
        study.set_user_attr(self._BASE_CONFIG_DIGEST_ATTRIBUTE, digest)

    def _base_config_digest(self) -> str:
        # The tuned keys are overwritten by every trial, so their base values must not count as a change.
        fixed_config: DictConfig = self._search_space.apply(self._base_config, self._search_space.first_values)
        for path in self._settings.resume_ignored_paths:
            OmegaConf.update(fixed_config, path, None, merge=False, force_add=True)
        return hashlib.sha256(OmegaConf.to_yaml(fixed_config, sort_keys=True).encode()).hexdigest()

    def _warn_if_pruning_is_ineffective(self, reporter: TrialReporter) -> None:
        if self._settings.pruning_method.is_stopping_trials and not reporter.is_reported:
            warnings.warn(
                f"pruning_method {type(self._settings.pruning_method).__name__} has no effect "
                + "because the objective did not call reporter.report.",
                UserWarning,
                stacklevel=2,
            )

    def _require_buildable(self, probe_group: Iterable[Mapping[str, ParameterValue]]) -> None:
        first_error: ValueError | None = None
        for values in probe_group:
            try:
                self._build_config(values)
                return
            except ValueError as error:
                first_error = first_error or error
        raise ValueError(
            f"Every tried combination was rejected: {first_error}. If only values inside the bounds can build, "
            + "set is_probing_enabled to False."
        ) from first_error

    def _build_config(self, values: Mapping[str, ParameterValue]) -> ConfigT:
        return self._config_builder(self._search_space.apply(self._base_config, values))

    def _summarize(self, study: Study) -> OptimizationResult[ConfigT]:
        if not study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            raise RuntimeError(f"No trial of study {study.study_name} completed.")
        # With one objective, best_trials would hold every tie.
        best_trials: list[FrozenTrial] = [study.best_trial] if len(study.directions) == 1 else study.best_trials
        return OptimizationResult(best_trials=[self._to_outcome(trial) for trial in best_trials], study=study)

    def _to_outcome(self, trial: FrozenTrial) -> TrialOutcome[ConfigT]:
        parameters: dict[str, ParameterValue] = dict(trial.user_attrs.get(self._VALUES_ATTRIBUTE, trial.params))
        return TrialOutcome(
            config=self._build_config(parameters),
            resolved_config=self._search_space.apply(self._base_config, parameters),
            values=list(trial.values),
            parameters=parameters,
        )
