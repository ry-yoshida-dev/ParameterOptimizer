import pytest
from dataclass_initializer import DataclassInitializer

from parameter_optimizer import Direction, OptimizerSettings, PruningMethod, SearchMethod


def test_builds_from_configuration() -> None:
    settings: OptimizerSettings = DataclassInitializer.build(
        OptimizerSettings,
        {
            "direction": "maximize",
            "search_method": "gaussian_process",
            "pruning_method": "successive_halving",
            "n_trials": 10,
            "seed": 1,
        },
    )
    assert settings.direction is Direction.MAXIMIZE
    assert settings.search_method is SearchMethod.GAUSSIAN_PROCESS
    assert settings.pruning_method is PruningMethod.SUCCESSIVE_HALVING


@pytest.mark.parametrize(
    "overrides",
    [
        {"n_trials": 0},
        {"timeout_seconds": 0.0},
        {"timeout_seconds": float("nan")},
        {"timeout_seconds": float("inf")},
        {"n_trials": 1, "seed": -1},
        {"n_trials": 1, "seed": 2**32},
        {"n_trials": 1, "storage_url": "sqlite:///study.db"},
    ],
)
def test_rejects_invalid_values(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        DataclassInitializer.build(OptimizerSettings, {"direction": "minimize", "search_method": "tpe", **overrides})


def test_only_grid_may_omit_trial_limits() -> None:
    assert OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.GRID).n_trials is None
    with pytest.raises(ValueError, match="tpe requires n_trials or timeout_seconds"):
        OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.TPE)
    assert OptimizerSettings(direction=Direction.MINIMIZE, search_method=SearchMethod.TPE, timeout_seconds=1.0)
