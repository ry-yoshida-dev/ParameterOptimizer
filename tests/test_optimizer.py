import time
from pathlib import Path

import optuna
import pytest
from omegaconf import DictConfig, OmegaConf
from optuna import Study, TrialPruned
from optuna.distributions import BaseDistribution, FloatDistribution, IntDistribution
from optuna.pruners import BasePruner, HyperbandPruner, MedianPruner, NopPruner, SuccessiveHalvingPruner
from optuna.storages import InMemoryStorage
from optuna.trial import FrozenTrial, TrialState
from sample.mode import SampleMode
from sample.model_config import SampleModelConfig
from sample.range_config import SampleRangeConfig

from parameter_optimizer import (
    CmaEsSearch,
    DataclassBuilder,
    Direction,
    GaussianProcessSearch,
    GeneticSearch,
    GridSearch,
    HyperbandPruning,
    MedianPruning,
    NoPruning,
    OptimizationResult,
    OptimizerSettings,
    ParameterOptimizer,
    PruningMethod,
    QuasiMonteCarloSearch,
    RandomSearch,
    SearchMethod,
    SuccessiveHalvingPruning,
    TpeSearch,
    TrialReporter,
)


def _config(content: dict[str, object]) -> DictConfig:
    config = OmegaConf.create(content)
    assert isinstance(config, DictConfig)
    return config


def _base_config() -> DictConfig:
    return _config({"name": "base", "solver": {"learning_rate": 0.5, "momentum": 0.9}})


def _optimizer(
    search_space: dict[str, object],
    settings: OptimizerSettings,
) -> ParameterOptimizer[SampleModelConfig]:
    return ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=_base_config(),
        search_space=_config(search_space),
        settings=settings,
    )


def _sampled_settings(n_trials: int = 1, seed: int | None = None) -> OptimizerSettings:
    return OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=n_trials,
        seed=seed,
    )


def _distance_to_target(config: SampleModelConfig, reporter: TrialReporter) -> float:
    return (config.solver.learning_rate - 0.3) ** 2


def test_grid_tries_every_combination_and_stops_by_itself() -> None:
    seen: list[tuple[SampleMode, float, float | None]] = []

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        seen.append((config.mode, config.solver.learning_rate, config.solver.momentum))
        return _distance_to_target(config, reporter)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {
            "mode": {"distribution": "categorical", "candidates": ["fast", "slow"]},
            "solver": {
                "learning_rate": {"distribution": "categorical", "candidates": [0.1, 0.3, 0.5]},
                "momentum": {"distribution": "categorical", "candidates": [None, 0.9]},
            },
        },
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch(), seed=0),
    ).optimize(objective)

    assert len(seen) == 12
    assert len(set(seen)) == 12
    assert result.best.config.solver.learning_rate == 0.3
    assert result.best.value == 0.0


def test_grid_rejects_numeric_parameter_at_construction() -> None:
    with pytest.raises(ValueError, match="grid requires categorical"):
        _optimizer(
            {"depth": {"distribution": "int", "low": 1, "high": 3}},
            OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
        )


def test_sampled_search_finds_optimum_and_keeps_untuned_values() -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}},
        _sampled_settings(n_trials=60, seed=0),
    ).optimize(_distance_to_target)

    assert result.best.config.solver.learning_rate == pytest.approx(0.3, abs=0.05)
    assert result.best.config.solver.momentum == 0.9
    assert result.best.config.name == "base"


def test_maximize_prefers_larger_value() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}},
        OptimizerSettings(directions=[Direction.MAXIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert result.best.config.depth == 3


def test_same_seed_reproduces_trials() -> None:
    def run() -> list[dict[str, object]]:
        result: OptimizationResult[SampleModelConfig] = _optimizer(
            {"solver": {"learning_rate": {"distribution": "float", "low": 0.001, "high": 1.0, "is_log": True}}},
            _sampled_settings(n_trials=15, seed=7),
        ).optimize(_distance_to_target)
        return [trial.params for trial in result.study.trials]

    assert run() == run()


def test_log_int_stays_within_bounds() -> None:
    depths: list[int] = []

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        depths.append(config.depth)
        return float(config.depth)

    _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 1024, "is_log": True}},
        _sampled_settings(n_trials=20, seed=0),
    ).optimize(objective)
    assert all(1 <= depth <= 1024 for depth in depths)


def test_resolved_config_is_a_normal_configuration() -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {
            "mode": {"distribution": "categorical", "candidates": ["slow"]},
            "solver": {"momentum": {"distribution": "categorical", "candidates": [None]}},
        },
        _sampled_settings(),
    ).optimize(_distance_to_target)

    assert OmegaConf.to_container(result.best.resolved_config) == {
        "name": "base",
        "solver": {"learning_rate": 0.5, "momentum": None},
        "mode": "slow",
    }
    assert result.best.config.mode is SampleMode.SLOW
    assert result.best.parameters == {"mode": "slow", "solver.momentum": None}


@pytest.mark.parametrize(
    "search_space, error_type",
    [
        ({"solver": {"learning_rat": {"distribution": "categorical", "candidates": [0.1]}}}, ValueError),
        ({"unknown": {"distribution": "categorical", "candidates": [0.1]}}, ValueError),
        ({"solver": {"learning_rate": {"distribution": "categorical", "candidates": [0.5, 1]}}}, TypeError),
        ({"solver": {"momentum": {"distribution": "categorical", "candidates": [0.5, 1]}}}, ValueError),
        ({"solver": {"learning_rate": {"distribution": "int", "low": 0, "high": 1}}}, TypeError),
        ({"solver": {"learning_rate": {"distribution": "float", "low": 0.5, "high": 1.5}}}, ValueError),
    ],
)
def test_construction_rejects_mistakes_before_any_trial(
    search_space: dict[str, object],
    error_type: type[Exception],
) -> None:
    with pytest.raises(error_type):
        _optimizer(search_space, _sampled_settings())


def test_objective_error_stops_the_run_by_default() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        raise RuntimeError("broken objective")

    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _sampled_settings(n_trials=3),
    )
    with pytest.raises(RuntimeError, match="broken objective"):
        optimizer.optimize(objective)


def test_caught_exception_fails_only_the_trial() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        if config.depth == 1:
            raise ArithmeticError("flaky")
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "categorical", "candidates": [1, 2]}},
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective, caught_exceptions=(ArithmeticError,))

    assert [trial.state for trial in result.study.trials].count(TrialState.FAIL) == 1
    assert result.best.config.depth == 2


def test_no_completed_trial_raises() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        raise ArithmeticError("always")

    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _sampled_settings(n_trials=2),
    )
    with pytest.raises(RuntimeError, match="No trial"):
        optimizer.optimize(objective, caught_exceptions=(ArithmeticError,))


@pytest.mark.parametrize(
    "pruning_method",
    [MedianPruning(), SuccessiveHalvingPruning(), HyperbandPruning()],
)
def test_reporter_lets_pruning_method_stop_bad_trials(pruning_method: PruningMethod) -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        for step in range(30):
            reporter.report(float(config.depth), step)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 100}},
        OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=TpeSearch(),
            pruning_method=pruning_method,
            n_trials=40,
            seed=0,
        ),
    ).optimize(objective)

    assert any(trial.state is TrialState.PRUNED for trial in result.study.trials)


def _pruning_settings(pruning_method: PruningMethod) -> OptimizerSettings:
    return OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        pruning_method=pruning_method,
        n_trials=2,
    )


def test_warns_when_pruning_has_nothing_to_act_on() -> None:
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _pruning_settings(MedianPruning()),
    )
    with pytest.warns(UserWarning, match="pruning_method MedianPruning has no effect"):
        optimizer.optimize(_distance_to_target)


@pytest.mark.filterwarnings("error::UserWarning")
def test_does_not_warn_when_pruning_is_usable_or_disabled() -> None:
    def reporting_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        reporter.report(1.0, 0)
        return 1.0

    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    _optimizer(search_space, _pruning_settings(MedianPruning())).optimize(reporting_objective)
    _optimizer(search_space, _pruning_settings(NoPruning())).optimize(_distance_to_target)


def test_storage_resumes_existing_study(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=3,
        seed=0,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len(result.study.trials) == 6


@pytest.mark.parametrize("search_method", [RandomSearch(), TpeSearch(), QuasiMonteCarloSearch()])
def test_resuming_with_a_seed_tries_new_values(search_method: SearchMethod, tmp_path: Path) -> None:
    if search_method is QuasiMonteCarloSearch():
        pytest.importorskip("scipy")
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=search_method,
        n_trials=5,
        seed=0,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len({trial.params["solver.learning_rate"] for trial in result.study.trials}) == 10


def test_resuming_a_finished_grid_runs_no_trial(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=GridSearch(),
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_forbidden_objective)
    assert len(result.study.trials) == 3


def test_resuming_an_interrupted_grid_runs_the_rest(tmp_path: Path) -> None:
    def settings(n_trials: int | None, seed: int | None) -> OptimizerSettings:
        return OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=GridSearch(),
            n_trials=n_trials,
            seed=seed,
            study_name="resume",
            storage=f"sqlite:///{tmp_path / 'study.db'}",
        )

    search_space: dict[str, object] = {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}}
    _optimizer(search_space, settings(1, 0)).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings(None, 1)).optimize(
        _distance_to_target
    )
    assert sorted(trial.params["depth"] for trial in result.study.trials) == [1, 2, 3]


def _forbidden_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
    raise AssertionError("No trial should run.")


@pytest.mark.parametrize(
    "other_search_space",
    [
        {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}},
        {"depth": {"distribution": "int", "low": 1, "high": 9}},
        {"depth": {"distribution": "int", "low": 1, "high": 3, "is_log": True}},
        {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}},
    ],
)
def test_resuming_with_another_space_fails_before_any_trial(
    other_search_space: dict[str, object],
    tmp_path: Path,
) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=1,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    _optimizer({"depth": {"distribution": "int", "low": 1, "high": 3}}, settings).optimize(_distance_to_target)
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(other_search_space, settings)
    with pytest.raises(ValueError, match="another search space"):
        optimizer.optimize(_forbidden_objective)


def test_resuming_with_another_direction_fails_before_any_trial(tmp_path: Path) -> None:
    storage_url: str = f"sqlite:///{tmp_path / 'study.db'}"
    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    _optimizer(
        search_space,
        OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=TpeSearch(),
            n_trials=1,
            study_name="resume",
            storage=storage_url,
        ),
    ).optimize(_distance_to_target)
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        search_space,
        OptimizerSettings(
            directions=[Direction.MAXIMIZE],
            search_method=TpeSearch(),
            n_trials=1,
            study_name="resume",
            storage=storage_url,
        ),
    )
    with pytest.raises(ValueError, match="has directions"):
        optimizer.optimize(_forbidden_objective)


@pytest.mark.parametrize(
    "search_method, required_module",
    [
        (RandomSearch(), None),
        (QuasiMonteCarloSearch(), "scipy"),
        (TpeSearch(), None),
        (GaussianProcessSearch(), "torch"),
        (CmaEsSearch(), "cmaes"),
        (GeneticSearch(), None),
    ],
)
def test_every_sampled_method_runs(search_method: SearchMethod, required_module: str | None) -> None:
    if required_module is not None:
        pytest.importorskip(required_module)
    search_space: dict[str, object] = {
        "depth": {"distribution": "int", "low": 1, "high": 8, "is_log": True},
        "solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}},
    }
    if not isinstance(search_method, CmaEsSearch):
        search_space["mode"] = {"distribution": "categorical", "candidates": ["fast", "slow"]}
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        search_space,
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=search_method, n_trials=12, seed=0),
    ).optimize(_distance_to_target)
    assert len(result.study.trials) == 12


def _range_optimizer(high_bound: int, settings: OptimizerSettings) -> ParameterOptimizer[SampleRangeConfig]:
    return ParameterOptimizer(
        config_builder=DataclassBuilder(SampleRangeConfig),
        base_config=_config({"low": 1, "high": 2}),
        search_space=_config(
            {
                "low": {"distribution": "int", "low": 1, "high": 5},
                "high": {"distribution": "int", "low": 1, "high": high_bound},
            }
        ),
        settings=settings,
    )


def test_constraint_across_parameters_passes_construction() -> None:
    _range_optimizer(5, _sampled_settings())


def test_value_rejected_next_to_every_other_value_fails_construction() -> None:
    with pytest.raises(ValueError, match="Every tried combination was rejected"):
        _range_optimizer(9, _sampled_settings())


def test_disabled_probing_defers_checks_to_the_trials() -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=20,
        seed=0,
        is_probing_enabled=False,
    )

    def objective(config: SampleRangeConfig, reporter: TrialReporter) -> float:
        return float(config.high - config.low)

    result: OptimizationResult[SampleRangeConfig] = _range_optimizer(9, settings).optimize(
        objective,
        caught_exceptions=(ValueError,),
    )
    assert result.best.value == 0.0
    assert any(trial.state is TrialState.FAIL for trial in result.study.trials)


def test_part_of_a_configuration_tunes_values_seen_through_interpolation() -> None:
    root_config: DictConfig = _config(
        {
            "model": {
                "name": "${model.mode}",
                "mode": "fast",
                "solver": {"learning_rate": 0.5, "momentum": 0.9},
            }
        }
    )
    names: list[str] = []

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        names.append(config.name)
        return 0.0 if config.mode is SampleMode.SLOW else 1.0

    result: OptimizationResult[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=root_config.model,
        search_space=_config({"mode": {"distribution": "categorical", "candidates": ["fast", "slow"]}}),
        settings=OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)

    assert sorted(names) == ["fast", "slow"]
    assert result.best.config.name == "slow"
    assert result.best.resolved_config.name == "slow"


def _base_config_settings(storage_url: str) -> OptimizerSettings:
    return OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=1,
        study_name="resume",
        storage=storage_url,
    )


def test_resuming_with_another_base_config_fails_before_any_trial(tmp_path: Path) -> None:
    settings: OptimizerSettings = _base_config_settings(f"sqlite:///{tmp_path / 'study.db'}")
    search_space: DictConfig = _config({"depth": {"distribution": "int", "low": 1, "high": 3}})
    ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=_base_config(),
        search_space=search_space,
        settings=settings,
    ).optimize(_distance_to_target)
    other_base_config: DictConfig = _base_config()
    other_base_config.solver.momentum = 0.1
    optimizer: ParameterOptimizer[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=other_base_config,
        search_space=search_space,
        settings=settings,
    )
    with pytest.raises(ValueError, match="another base configuration"):
        optimizer.optimize(_forbidden_objective)


def test_resuming_ignores_base_values_of_tuned_keys(tmp_path: Path) -> None:
    settings: OptimizerSettings = _base_config_settings(f"sqlite:///{tmp_path / 'study.db'}")
    search_space: DictConfig = _config({"solver": {"momentum": {"distribution": "float", "low": 0.0, "high": 1.0}}})
    other_base_config: DictConfig = _base_config()
    other_base_config.solver.momentum = 0.1
    results: list[OptimizationResult[SampleModelConfig]] = [
        ParameterOptimizer(
            config_builder=DataclassBuilder(SampleModelConfig),
            base_config=base_config,
            search_space=search_space,
            settings=settings,
        ).optimize(_distance_to_target)
        for base_config in (_base_config(), other_base_config)
    ]
    assert len(results[-1].study.trials) == 2


def test_resuming_a_study_without_base_config_record_fails(tmp_path: Path) -> None:
    storage_url: str = f"sqlite:///{tmp_path / 'study.db'}"
    settings: OptimizerSettings = _base_config_settings(storage_url)
    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    study: Study = optuna.create_study(study_name="resume", storage=storage_url)
    study.optimize(lambda trial: float(trial.suggest_int("depth", 1, 3)), n_trials=1)
    with pytest.raises(ValueError, match="another base configuration"):
        _optimizer(search_space, settings).optimize(_forbidden_objective)


def test_resuming_with_another_space_fails_with_only_pruned_trials(tmp_path: Path) -> None:
    def pruned_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        raise TrialPruned()

    settings: OptimizerSettings = _base_config_settings(f"sqlite:///{tmp_path / 'study.db'}")
    with pytest.raises(RuntimeError, match="No trial"):
        _optimizer({"depth": {"distribution": "int", "low": 1, "high": 3}}, settings).optimize(pruned_objective)
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 9}},
        settings,
    )
    with pytest.raises(ValueError, match="another search space"):
        optimizer.optimize(_forbidden_objective)


def _grid_settings(storage_url: str, n_trials: int | None = None) -> OptimizerSettings:
    return OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=GridSearch(),
        n_trials=n_trials,
        study_name="resume",
        storage=storage_url,
    )


_DEPTH_GRID: dict[str, object] = {"depth": {"distribution": "categorical", "candidates": [1, 2, 3, 4]}}


def _depth_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
    return float(config.depth)


def _completed_depths(result: OptimizationResult[SampleModelConfig]) -> list[int]:
    return sorted(trial.params["depth"] for trial in result.study.trials if trial.state is TrialState.COMPLETE)


def test_resuming_a_grid_retries_the_point_that_stopped_the_run(tmp_path: Path) -> None:
    settings: OptimizerSettings = _grid_settings(f"sqlite:///{tmp_path / 'study.db'}")
    evaluated_depths: list[int] = []

    def crashing_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        evaluated_depths.append(config.depth)
        if len(evaluated_depths) == 2:
            raise MemoryError("out of memory")
        return float(config.depth)

    with pytest.raises(MemoryError):
        _optimizer(_DEPTH_GRID, settings).optimize(crashing_objective)
    result: OptimizationResult[SampleModelConfig] = _optimizer(_DEPTH_GRID, settings).optimize(_depth_objective)
    assert _completed_depths(result) == [1, 2, 3, 4]
    assert result.best.config.depth == 1


def test_resuming_a_finished_grid_retries_every_failed_point_once(tmp_path: Path) -> None:
    settings: OptimizerSettings = _grid_settings(f"sqlite:///{tmp_path / 'study.db'}")

    def flaky_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        if config.depth < 3:
            raise ArithmeticError("flaky")
        return float(config.depth)

    first_result: OptimizationResult[SampleModelConfig] = _optimizer(_DEPTH_GRID, settings).optimize(
        flaky_objective,
        caught_exceptions=(ArithmeticError,),
    )
    assert _completed_depths(first_result) == [3, 4]

    evaluated_depths: list[int] = []

    def recording_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        evaluated_depths.append(config.depth)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(_DEPTH_GRID, settings).optimize(recording_objective)
    assert sorted(evaluated_depths) == [1, 2]
    assert _completed_depths(result) == [1, 2, 3, 4]

    _optimizer(_DEPTH_GRID, settings).optimize(_forbidden_objective)


def test_grid_retry_failing_again_ends_the_run(tmp_path: Path) -> None:
    settings: OptimizerSettings = _grid_settings(f"sqlite:///{tmp_path / 'study.db'}")

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        if config.depth == 1:
            raise ArithmeticError("always")
        return float(config.depth)

    for expected_trials in (4, 5):
        result: OptimizationResult[SampleModelConfig] = _optimizer(_DEPTH_GRID, settings).optimize(
            objective,
            caught_exceptions=(ArithmeticError,),
        )
        assert len(result.study.trials) == expected_trials


def test_grid_retry_limited_by_n_trials_stays_queued(tmp_path: Path) -> None:
    storage_url: str = f"sqlite:///{tmp_path / 'study.db'}"

    def failing_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        if config.depth < 3:
            raise ArithmeticError("flaky")
        return float(config.depth)

    _optimizer(_DEPTH_GRID, _grid_settings(storage_url)).optimize(
        failing_objective,
        caught_exceptions=(ArithmeticError,),
    )
    _optimizer(_DEPTH_GRID, _grid_settings(storage_url, n_trials=1)).optimize(_depth_objective)
    result: OptimizationResult[SampleModelConfig] = _optimizer(_DEPTH_GRID, _grid_settings(storage_url)).optimize(
        _depth_objective
    )
    assert _completed_depths(result) == [1, 2, 3, 4]
    assert len(result.study.trials) == 6


@pytest.mark.parametrize("first_trials", [1, 2, 3, 4, 5, 6, 7])
def test_resuming_a_grid_with_a_new_seed_evaluates_each_point_once(first_trials: int, tmp_path: Path) -> None:
    def settings(n_trials: int | None, seed: int) -> OptimizerSettings:
        return OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=GridSearch(),
            n_trials=n_trials,
            seed=seed,
            study_name="resume",
            storage=f"sqlite:///{tmp_path / 'study.db'}",
        )

    search_space: dict[str, object] = {
        "depth": {"distribution": "categorical", "candidates": [1, 2, 3, 4, 5, 6, 7, 8]},
    }
    _optimizer(search_space, settings(first_trials, 0)).optimize(_depth_objective)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings(None, 123)).optimize(
        _depth_objective
    )
    assert _completed_depths(result) == [1, 2, 3, 4, 5, 6, 7, 8]


@pytest.mark.parametrize(
    "search_method, required_module",
    [
        (RandomSearch(), None),
        (QuasiMonteCarloSearch(), "scipy"),
        (TpeSearch(), None),
        (GaussianProcessSearch(), "torch"),
        (CmaEsSearch(), "cmaes"),
        (GeneticSearch(), None),
    ],
)
def test_every_sampled_method_follows_the_seed(search_method: SearchMethod, required_module: str | None) -> None:
    if required_module is not None:
        pytest.importorskip(required_module)

    def run(seed: int) -> list[dict[str, object]]:
        result: OptimizationResult[SampleModelConfig] = _optimizer(
            {
                "depth": {"distribution": "int", "low": 1, "high": 8},
                "solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}},
            },
            OptimizerSettings(directions=[Direction.MINIMIZE], search_method=search_method, n_trials=12, seed=seed),
        ).optimize(_distance_to_target)
        return [trial.params for trial in result.study.trials]

    assert run(7) == run(7)
    assert run(7) != run(8)


@pytest.mark.parametrize("is_log", [False, True])
def test_is_log_reaches_the_sampling(is_log: bool) -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {
            "depth": {"distribution": "int", "low": 1, "high": 8, "is_log": is_log},
            "solver": {"learning_rate": {"distribution": "float", "low": 0.1, "high": 0.9, "is_log": is_log}},
        },
        _sampled_settings(),
    ).optimize(_distance_to_target)
    depth_distribution: BaseDistribution = result.study.trials[0].distributions["depth"]
    learning_rate_distribution: BaseDistribution = result.study.trials[0].distributions["solver.learning_rate"]
    assert isinstance(depth_distribution, IntDistribution)
    assert isinstance(learning_rate_distribution, FloatDistribution)
    assert depth_distribution.log is is_log
    assert learning_rate_distribution.log is is_log


@pytest.mark.parametrize(
    "pruning_method, pruner_type",
    [
        (NoPruning(), NopPruner),
        (MedianPruning(), MedianPruner),
        (SuccessiveHalvingPruning(), SuccessiveHalvingPruner),
        (HyperbandPruning(), HyperbandPruner),
    ],
)
def test_pruning_method_reaches_the_study(pruning_method: PruningMethod, pruner_type: type[BasePruner]) -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        reporter.report(1.0, 0)
        return 1.0

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _pruning_settings(pruning_method),
    ).optimize(objective)
    assert isinstance(result.study.pruner, pruner_type)


def test_timeout_stops_starting_new_trials() -> None:
    def slow_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        time.sleep(0.02)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=TpeSearch(),
            n_trials=100,
            timeout_seconds=0.1,
        ),
    ).optimize(slow_objective)
    assert 1 <= len(result.study.trials) < 100


def test_resuming_with_logarithmic_float_keeps_the_study(tmp_path: Path) -> None:
    settings: OptimizerSettings = _base_config_settings(f"sqlite:///{tmp_path / 'study.db'}")
    search_space: dict[str, object] = {
        "solver": {"learning_rate": {"distribution": "float", "low": 0.1, "high": 0.9, "is_log": True}}
    }
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len(result.study.trials) == 2


def test_grid_does_not_retry_failed_points_outside_of_it(tmp_path: Path) -> None:
    settings: OptimizerSettings = _grid_settings(f"sqlite:///{tmp_path / 'study.db'}")
    base_config: DictConfig = _base_config()
    base_config.mode = "fast"
    base_config.depth = 1

    def failing_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        raise ArithmeticError("always")

    with pytest.raises(RuntimeError, match="No trial"):
        ParameterOptimizer(
            config_builder=DataclassBuilder(SampleModelConfig),
            base_config=base_config,
            search_space=_config({"depth": {"distribution": "categorical", "candidates": [1, 2]}}),
            settings=settings,
        ).optimize(failing_objective, caught_exceptions=(ArithmeticError,))

    evaluated: list[tuple[SampleMode, int]] = []

    def recording_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        evaluated.append((config.mode, config.depth))
        return float(config.depth)

    ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=base_config,
        search_space=_config({"mode": {"distribution": "categorical", "candidates": ["fast", "slow"]}}),
        settings=settings,
    ).optimize(recording_objective)
    assert sorted(evaluated, key=lambda item: item[0].value) == [(SampleMode.FAST, 1), (SampleMode.SLOW, 1)]


def test_several_objectives_return_the_pareto_front() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> tuple[float, float]:
        return float(config.depth), -float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        _DEPTH_GRID,
        OptimizerSettings(directions=[Direction.MINIMIZE, Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert sorted(outcome.config.depth for outcome in result.best_trials) == [1, 2, 3, 4]
    assert all(outcome.values == [outcome.config.depth, -outcome.config.depth] for outcome in result.best_trials)
    with pytest.raises(ValueError, match="choose among best_trials"):
        _ = result.best
    with pytest.raises(ValueError, match="single objective"):
        _ = result.best_trials[0].value


def test_several_objectives_without_a_tradeoff_have_one_best_trial() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> tuple[float, float]:
        return float(config.depth), float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        _DEPTH_GRID,
        OptimizerSettings(directions=[Direction.MINIMIZE, Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert result.best.config.depth == 1


def test_ties_of_a_single_objective_still_have_one_best_trial() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        return 0.0

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        _DEPTH_GRID,
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert len(result.best_trials) == 1
    assert result.best.value == 0.0


@pytest.mark.parametrize(
    "search_method, required_module",
    [
        (RandomSearch(), None),
        (QuasiMonteCarloSearch(), "scipy"),
        (TpeSearch(), None),
        (GaussianProcessSearch(), "torch"),
        (GeneticSearch(), None),
    ],
)
def test_sampled_methods_handle_several_objectives(search_method: SearchMethod, required_module: str | None) -> None:
    if required_module is not None:
        pytest.importorskip(required_module)

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> tuple[float, float]:
        return config.solver.learning_rate, 1.0 - config.solver.learning_rate

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}},
        OptimizerSettings(
            directions=[Direction.MINIMIZE, Direction.MINIMIZE],
            search_method=search_method,
            n_trials=12,
            seed=0,
        ),
    ).optimize(objective)
    assert result.best_trials


def test_method_without_several_objectives_fails_at_construction() -> None:
    pytest.importorskip("cmaes")
    with pytest.raises(ValueError, match="does not support several objectives"):
        _optimizer(
            {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}},
            OptimizerSettings(
                directions=[Direction.MINIMIZE, Direction.MINIMIZE],
                search_method=CmaEsSearch(),
                n_trials=1,
            ),
        )


def test_resuming_with_another_number_of_objectives_fails(tmp_path: Path) -> None:
    storage: str = f"sqlite:///{tmp_path / 'study.db'}"

    def settings(directions: list[Direction]) -> OptimizerSettings:
        return OptimizerSettings(
            directions=directions,
            search_method=GridSearch(),
            study_name="resume",
            storage=storage,
        )

    _optimizer(_DEPTH_GRID, settings([Direction.MINIMIZE])).optimize(_depth_objective)
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        _DEPTH_GRID,
        settings([Direction.MINIMIZE, Direction.MINIMIZE]),
    )
    with pytest.raises(ValueError, match="has directions"):
        optimizer.optimize(_forbidden_objective)


def test_any_builder_can_produce_the_object_of_a_trial() -> None:
    def build_tuple(config: DictConfig) -> tuple[int, str]:
        return config.depth, config.name

    def objective(config: tuple[int, str], reporter: TrialReporter) -> float:
        return float(config[0])

    result: OptimizationResult[tuple[int, str]] = ParameterOptimizer(
        config_builder=build_tuple,
        base_config=_config({"name": "plain", "depth": 0}),
        search_space=_config(_DEPTH_GRID),
        settings=OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert result.best.config == (1, "plain")


def test_part_of_a_configuration_merged_with_overrides_keeps_them() -> None:
    root_config: DictConfig = _config({"model": {"name": "root", "solver": {"learning_rate": 0.5, "momentum": 0.9}}})
    base_config = OmegaConf.merge(root_config.model, {"name": "merged"})
    assert isinstance(base_config, DictConfig)
    names: list[str] = []

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        names.append(config.name)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=base_config,
        search_space=_config(_DEPTH_GRID),
        settings=OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()),
    ).optimize(objective)
    assert set(names) == {"merged"}
    assert result.best.resolved_config.name == "merged"


def test_resuming_with_the_largest_seed_wraps_around(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=RandomSearch(),
        n_trials=1,
        seed=2**32 - 1,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len(result.study.trials) == 2


def test_resume_ignored_paths_may_change_between_runs(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=1,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        resume_ignored_paths=["name"],
    )
    search_space: DictConfig = _config({"depth": {"distribution": "int", "low": 1, "high": 3}})
    results: list[OptimizationResult[SampleModelConfig]] = [
        ParameterOptimizer(
            config_builder=DataclassBuilder(SampleModelConfig),
            base_config=_config({"name": name, "solver": {"learning_rate": 0.5, "momentum": 0.9}}),
            search_space=search_space,
            settings=settings,
        ).optimize(_distance_to_target)
        for name in ("first", "second")
    ]
    assert len(results[-1].study.trials) == 2
    assert results[-1].best.config.name == "second"


def test_other_paths_still_fail_with_resume_ignored_paths(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=1,
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
        resume_ignored_paths=["name"],
    )
    search_space: DictConfig = _config({"depth": {"distribution": "int", "low": 1, "high": 3}})
    ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=_base_config(),
        search_space=search_space,
        settings=settings,
    ).optimize(_distance_to_target)
    other_base_config: DictConfig = _base_config()
    other_base_config.solver.momentum = 0.1
    optimizer: ParameterOptimizer[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=other_base_config,
        search_space=search_space,
        settings=settings,
    )
    with pytest.raises(ValueError, match="resume_ignored_paths"):
        optimizer.optimize(_forbidden_objective)


def test_grid_tries_every_step_of_numeric_parameters_and_retries_them(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=GridSearch(),
        study_name="resume",
        storage=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {
        "depth": {"distribution": "int", "low": 1, "high": 5, "step": 2},
        "solver": {"learning_rate": {"distribution": "float", "low": 0.1, "high": 0.3, "step": 0.1}},
    }

    def failing_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        if config.depth == 3:
            raise ArithmeticError("flaky")
        return float(config.depth)

    first_result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(
        failing_objective,
        caught_exceptions=(ArithmeticError,),
    )
    assert len(first_result.study.trials) == 9
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_depth_objective)
    completed: list[FrozenTrial] = [trial for trial in result.study.trials if trial.state is TrialState.COMPLETE]
    assert len(completed) == 9
    assert sorted({trial.params["depth"] for trial in completed}) == [1, 3, 5]


def test_stepped_sampling_stays_on_the_steps() -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 2, "high": 10, "step": 4}},
        _sampled_settings(n_trials=20, seed=0),
    ).optimize(_distance_to_target)
    assert {trial.params["depth"] for trial in result.study.trials} <= {2, 6, 10}


def test_callbacks_see_every_trial() -> None:
    seen: list[int] = []

    def record(study: Study, trial: FrozenTrial) -> None:
        seen.append(trial.number)

    _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _sampled_settings(n_trials=3),
    ).optimize(_distance_to_target, callbacks=[record])
    assert seen == [0, 1, 2]


def test_parallel_jobs_run_every_trial() -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        _DEPTH_GRID,
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch(), n_jobs=2),
    ).optimize(_depth_objective)
    # Parallel workers of a grid may evaluate its last points twice.
    assert set(_completed_depths(result)) == {1, 2, 3, 4}


def test_storage_object_resumes_a_study() -> None:
    settings: OptimizerSettings = OptimizerSettings(
        directions=[Direction.MINIMIZE],
        search_method=TpeSearch(),
        n_trials=1,
        study_name="resume",
        storage=InMemoryStorage(),
    )
    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len(result.study.trials) == 2


@pytest.mark.parametrize(
    "pruning_method",
    [
        MedianPruning(startup_trial_count=0),
        SuccessiveHalvingPruning(min_resource=1, reduction_factor=2),
        HyperbandPruning(max_resource=4),
    ],
)
def test_options_of_a_pruning_method_build(pruning_method: PruningMethod) -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        reporter.report(float(config.depth), 0)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        OptimizerSettings(
            directions=[Direction.MINIMIZE],
            search_method=RandomSearch(),
            pruning_method=pruning_method,
            n_trials=1,
        ),
    ).optimize(objective)
    assert result.best.config.depth in (1, 2, 3)


@pytest.mark.parametrize(
    "search_method, required_module",
    [
        (QuasiMonteCarloSearch(sequence="halton", is_scrambled=True), "scipy"),
        (TpeSearch(startup_trial_count=2, candidate_count=4, is_multivariate=True, is_grouped=True), None),
        (GaussianProcessSearch(startup_trial_count=2, is_deterministic_objective=True), "torch"),
        (CmaEsSearch(population_size=4, is_separable=True), "cmaes"),
        (GeneticSearch(population_size=4, mutation_probability=0.1), None),
    ],
)
def test_options_of_a_search_method_reach_the_run(search_method: SearchMethod, required_module: str | None) -> None:
    if required_module is not None:
        pytest.importorskip(required_module)
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"solver": {"learning_rate": {"distribution": "float", "low": 0.1, "high": 0.9}}},
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=search_method, n_trials=8, seed=0),
    ).optimize(_distance_to_target)
    assert len(result.study.trials) == 8
