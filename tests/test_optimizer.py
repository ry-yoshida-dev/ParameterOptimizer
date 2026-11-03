from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf
from optuna.trial import TrialState
from sample.mode import SampleMode
from sample.model_config import SampleModelConfig

from parameter_optimizer import (
    Direction,
    OptimizationResult,
    OptimizerSettings,
    ParameterOptimizer,
    PruningMethod,
    SearchMethod,
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
        config_class=SampleModelConfig,
        base_config=_base_config(),
        search_space=_config(search_space),
        settings=settings,
    )


def _sampled_settings(n_trials: int = 1, seed: int | None = None) -> OptimizerSettings:
    return OptimizerSettings(
        direction=Direction.MINIMIZE,
        search_method=SearchMethod.TPE,
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
        OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.GRID, seed=0),
    ).optimize(objective)

    assert len(seen) == 12
    assert len(set(seen)) == 12
    assert result.best_config.solver.learning_rate == 0.3
    assert result.best_value == 0.0


def test_grid_rejects_numeric_parameter_at_construction() -> None:
    with pytest.raises(ValueError, match="grid requires categorical"):
        _optimizer(
            {"depth": {"distribution": "int", "low": 1, "high": 3}},
            OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.GRID),
        )


def test_sampled_search_finds_optimum_and_keeps_untuned_values() -> None:
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}},
        _sampled_settings(n_trials=60, seed=0),
    ).optimize(_distance_to_target)

    assert result.best_config.solver.learning_rate == pytest.approx(0.3, abs=0.05)
    assert result.best_config.solver.momentum == 0.9
    assert result.best_config.name == "base"


def test_maximize_prefers_larger_value() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}},
        OptimizerSettings(direction=Direction.MAXIMIZE, search_method=SearchMethod.GRID),
    ).optimize(objective)
    assert result.best_config.depth == 3


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

    assert OmegaConf.to_container(result.resolved_config) == {
        "name": "base",
        "solver": {"learning_rate": 0.5, "momentum": None},
        "mode": "slow",
    }
    assert result.best_config.mode is SampleMode.SLOW
    assert result.best_parameters == {"mode": "slow", "solver.momentum": None}


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
        OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.GRID),
    ).optimize(objective, caught_exceptions=(ArithmeticError,))

    assert [trial.state for trial in result.study.trials].count(TrialState.FAIL) == 1
    assert result.best_config.depth == 2


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
    [PruningMethod.MEDIAN, PruningMethod.SUCCESSIVE_HALVING, PruningMethod.HYPERBAND],
)
def test_reporter_lets_pruning_method_stop_bad_trials(pruning_method: PruningMethod) -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        for step in range(30):
            reporter.report(float(config.depth), step)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 100}},
        OptimizerSettings(
            direction=Direction.MINIMIZE,
            search_method=SearchMethod.TPE,
            pruning_method=pruning_method,
            n_trials=40,
            seed=0,
        ),
    ).optimize(objective)

    assert any(trial.state is TrialState.PRUNED for trial in result.study.trials)


def _pruning_settings(pruning_method: PruningMethod) -> OptimizerSettings:
    return OptimizerSettings(
        direction=Direction.MINIMIZE,
        search_method=SearchMethod.TPE,
        pruning_method=pruning_method,
        n_trials=2,
    )


def test_warns_when_pruning_has_nothing_to_act_on() -> None:
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        {"depth": {"distribution": "int", "low": 1, "high": 3}},
        _pruning_settings(PruningMethod.MEDIAN),
    )
    with pytest.warns(UserWarning, match="pruning_method median has no effect"):
        optimizer.optimize(_distance_to_target)


@pytest.mark.filterwarnings("error::UserWarning")
def test_does_not_warn_when_pruning_is_usable_or_disabled() -> None:
    def reporting_objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        reporter.report(1.0, 0)
        return 1.0

    search_space: dict[str, object] = {"depth": {"distribution": "int", "low": 1, "high": 3}}
    _optimizer(search_space, _pruning_settings(PruningMethod.MEDIAN)).optimize(reporting_objective)
    _optimizer(search_space, _pruning_settings(PruningMethod.NONE)).optimize(_distance_to_target)


def test_storage_resumes_existing_study(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        direction=Direction.MINIMIZE,
        search_method=SearchMethod.TPE,
        n_trials=3,
        seed=0,
        study_name="resume",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len(result.study.trials) == 6


@pytest.mark.parametrize("search_method", [SearchMethod.RANDOM, SearchMethod.TPE, SearchMethod.QUASI_MONTE_CARLO])
def test_resuming_with_a_seed_tries_new_values(search_method: SearchMethod, tmp_path: Path) -> None:
    if search_method is SearchMethod.QUASI_MONTE_CARLO:
        pytest.importorskip("scipy")
    settings: OptimizerSettings = OptimizerSettings(
        direction=Direction.MINIMIZE,
        search_method=search_method,
        n_trials=5,
        seed=0,
        study_name="resume",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_distance_to_target)
    assert len({trial.params["solver.learning_rate"] for trial in result.study.trials}) == 10


def test_resuming_a_finished_grid_runs_no_trial(tmp_path: Path) -> None:
    settings: OptimizerSettings = OptimizerSettings(
        direction=Direction.MINIMIZE,
        search_method=SearchMethod.GRID,
        study_name="resume",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
    )
    search_space: dict[str, object] = {"depth": {"distribution": "categorical", "candidates": [1, 2, 3]}}
    _optimizer(search_space, settings).optimize(_distance_to_target)
    result: OptimizationResult[SampleModelConfig] = _optimizer(search_space, settings).optimize(_forbidden_objective)
    assert len(result.study.trials) == 3


def test_resuming_an_interrupted_grid_runs_the_rest(tmp_path: Path) -> None:
    def settings(n_trials: int | None, seed: int | None) -> OptimizerSettings:
        return OptimizerSettings(
            direction=Direction.MINIMIZE,
            search_method=SearchMethod.GRID,
            n_trials=n_trials,
            seed=seed,
            study_name="resume",
            storage_url=f"sqlite:///{tmp_path / 'study.db'}",
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
        direction=Direction.MINIMIZE,
        search_method=SearchMethod.TPE,
        n_trials=1,
        study_name="resume",
        storage_url=f"sqlite:///{tmp_path / 'study.db'}",
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
            direction=Direction.MINIMIZE,
            search_method=SearchMethod.TPE,
            n_trials=1,
            study_name="resume",
            storage_url=storage_url,
        ),
    ).optimize(_distance_to_target)
    optimizer: ParameterOptimizer[SampleModelConfig] = _optimizer(
        search_space,
        OptimizerSettings(
            direction=Direction.MAXIMIZE,
            search_method=SearchMethod.TPE,
            n_trials=1,
            study_name="resume",
            storage_url=storage_url,
        ),
    )
    with pytest.raises(ValueError, match="has direction MINIMIZE"):
        optimizer.optimize(_forbidden_objective)


@pytest.mark.parametrize(
    "search_method, required_module",
    [
        (SearchMethod.RANDOM, None),
        (SearchMethod.QUASI_MONTE_CARLO, "scipy"),
        (SearchMethod.TPE, None),
        (SearchMethod.GAUSSIAN_PROCESS, "torch"),
        (SearchMethod.CMA_ES, "cmaes"),
        (SearchMethod.GENETIC, None),
    ],
)
def test_every_sampled_method_runs(search_method: SearchMethod, required_module: str | None) -> None:
    if required_module is not None:
        pytest.importorskip(required_module)
    search_space: dict[str, object] = {
        "depth": {"distribution": "int", "low": 1, "high": 8, "is_log": True},
        "solver": {"learning_rate": {"distribution": "float", "low": 0.0, "high": 1.0}},
    }
    if search_method is not SearchMethod.CMA_ES:
        search_space["mode"] = {"distribution": "categorical", "candidates": ["fast", "slow"]}
    result: OptimizationResult[SampleModelConfig] = _optimizer(
        search_space,
        OptimizerSettings(direction=Direction.MINIMIZE, search_method=search_method, n_trials=12, seed=0),
    ).optimize(_distance_to_target)
    assert len(result.study.trials) == 12
