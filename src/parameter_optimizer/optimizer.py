import warnings
from collections.abc import Mapping

import optuna
from dataclass_initializer import DataclassInitializer
from omegaconf import DictConfig
from optuna import Study, Trial
from optuna.distributions import BaseDistribution
from optuna.samplers import BaseSampler, GridSampler
from optuna.study import StudyDirection
from optuna.trial import TrialState

from .parameter import ParameterValue
from .search import SearchSpace
from .study import OptimizationResult, OptimizerSettings
from .trial import Objective, PruningMethod, TrialReporter


class ParameterOptimizer[ConfigT]:
    """
    Tunes a dataclass by overlaying a search space on its base configuration.

    Every candidate is checked at construction by building ``config_class``, so mistakes in the search space fail
    before any trial runs.

    Parameters
    ----------
    config_class : type[ConfigT]
        Dataclass to build for each trial.
    base_config : DictConfig
        Normal configuration of ``config_class``.
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
        If the search space is invalid, a key is unknown to ``config_class``, a probed value is rejected by it, or the
        search method does not support the parameters.
    """

    def __init__(
        self,
        *,
        config_class: type[ConfigT],
        base_config: DictConfig,
        search_space: DictConfig,
        settings: OptimizerSettings,
    ) -> None:
        self._config_class: type[ConfigT] = config_class
        self._base_config: DictConfig = base_config
        self._search_space: SearchSpace = SearchSpace.from_config(search_space)
        self._settings: OptimizerSettings = settings
        self._settings.search_method.validate(self._search_space)
        for probe in self._search_space.iter_probes():
            self._build_config(probe)

    def optimize(
        self,
        objective: Objective[ConfigT],
        caught_exceptions: tuple[type[Exception], ...] = (),
    ) -> OptimizationResult[ConfigT]:
        """
        Run the trials and return the best configuration.

        Parameters
        ----------
        objective : Objective[ConfigT]
            Scores each configuration.
        caught_exceptions : tuple[type[Exception], ...]
            Exceptions that mark only the current trial as failed; any other exception stops the run.

        Returns
        -------
        OptimizationResult[ConfigT]
            Best configuration, its score and the study.

        Raises
        ------
        RuntimeError
            If no trial completed.
        ValueError
            If a resumed study has another direction or another search space.
        """
        study: Study = optuna.create_study(
            storage=self._settings.storage_url,
            pruner=self._settings.pruning_method.build(),
            study_name=self._settings.study_name,
            direction=self._settings.direction.to_study_direction(),
            load_if_exists=self._settings.storage_url is not None,
        )
        self._validate_resumed_study(study)
        sampler: BaseSampler = self._settings.search_method.build_sampler(self._search_space, self._next_seed(study))
        study.sampler = sampler
        # A finished grid would otherwise run one more trial, because Optuna only stops after sampling.
        if isinstance(sampler, GridSampler) and sampler.is_exhausted(study):
            return self._summarize(study)

        def evaluate(trial: Trial) -> float:
            config: ConfigT = self._build_config(self._search_space.suggest(trial))
            reporter: TrialReporter = TrialReporter(trial)
            objective_value: float = objective(config, reporter)
            self._warn_if_pruning_is_ineffective(reporter)
            return objective_value

        study.optimize(
            evaluate,
            n_trials=self._settings.n_trials,
            timeout=self._settings.timeout_seconds,
            catch=caught_exceptions,
        )
        return self._summarize(study)

    def _validate_resumed_study(self, study: Study) -> None:
        # Optuna keeps the stored direction and accepts other ranges without complaint, which would silently optimize
        # the opposite way or mix incomparable trials.
        expected_direction: StudyDirection = self._settings.direction.to_study_direction()
        if study.direction is not expected_direction:
            raise ValueError(f"Study {study.study_name} has direction {study.direction.name}.")
        expected_distributions: dict[str, BaseDistribution] = self._search_space.optuna_distributions
        for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE, TrialState.PRUNED)):
            if trial.distributions != expected_distributions:
                raise ValueError(
                    f"Study {study.study_name} has trials of another search space: {trial.distributions}. "
                    + f"Expected {expected_distributions}."
                )

    def _next_seed(self, study: Study) -> int | None:
        # The same seed on a resumed study would repeat the values of the earlier run.
        if self._settings.seed is None:
            return None
        return self._settings.seed + len(study.get_trials(deepcopy=False))

    def _warn_if_pruning_is_ineffective(self, reporter: TrialReporter) -> None:
        if self._settings.pruning_method is not PruningMethod.NONE and not reporter.is_reported:
            warnings.warn(
                f"pruning_method {self._settings.pruning_method.value} has no effect "
                + "because the objective did not call reporter.report.",
                UserWarning,
                stacklevel=2,
            )

    def _build_config(self, values: Mapping[str, ParameterValue]) -> ConfigT:
        resolved_config: DictConfig = self._search_space.apply(self._base_config, values)
        return DataclassInitializer.build(self._config_class, resolved_config)

    def _summarize(self, study: Study) -> OptimizationResult[ConfigT]:
        if not study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,)):
            raise RuntimeError(f"No trial of study {study.study_name} completed.")
        best_parameters: dict[str, ParameterValue] = dict(study.best_trial.params)
        return OptimizationResult(
            best_config=self._build_config(best_parameters),
            resolved_config=self._search_space.apply(self._base_config, best_parameters),
            best_value=study.best_value,
            best_parameters=best_parameters,
            study=study,
        )
