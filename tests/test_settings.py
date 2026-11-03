import pytest
from omegaconf import DictConfig, OmegaConf

from parameter_optimizer import (
    Direction,
    GridSearch,
    MedianPruning,
    NoPruning,
    OptimizerSettings,
    SuccessiveHalvingPruning,
    TpeSearch,
)


def test_builds_from_configuration() -> None:
    settings: OptimizerSettings = OptimizerSettings.from_config(
        {
            "directions": ["maximize"],
            "search_method": "tpe",
            "pruning_method": "successive_halving",
            "n_trials": 10,
            "seed": 1,
        },
    )
    assert settings.directions == [Direction.MAXIMIZE]
    assert settings.search_method == TpeSearch()
    assert settings.pruning_method == SuccessiveHalvingPruning()


def test_builds_options_of_the_implementations_from_a_dict_config() -> None:
    config = OmegaConf.create(
        {
            "directions": ["minimize", "maximize"],
            "search_method": {"name": "tpe", "startup_trial_count": 3, "is_multivariate": True},
            "n_trials": 10,
            "resume_ignored_paths": ["output_directory"],
        }
    )
    assert isinstance(config, DictConfig)
    settings: OptimizerSettings = OptimizerSettings.from_config(config)
    assert settings.search_method == TpeSearch(startup_trial_count=3, is_multivariate=True)
    assert settings.pruning_method == NoPruning()
    assert settings.resume_ignored_paths == ["output_directory"]


def test_builds_options_of_a_pruning_method() -> None:
    settings: OptimizerSettings = OptimizerSettings.from_config(
        {
            "directions": ["minimize"],
            "search_method": "grid",
            "pruning_method": {"name": "median", "warmup_step_count": 2},
        },
    )
    assert settings.pruning_method == MedianPruning(warmup_step_count=2)


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
        OptimizerSettings.from_config({"directions": ["minimize"], "search_method": "tpe", **overrides})


def test_only_grid_may_omit_trial_limits() -> None:
    assert OptimizerSettings(directions=[Direction.MINIMIZE], search_method=GridSearch()).n_trials is None
    with pytest.raises(ValueError, match="TpeSearch requires n_trials or timeout_seconds"):
        OptimizerSettings(directions=[Direction.MINIMIZE], search_method=TpeSearch())
    assert OptimizerSettings(directions=[Direction.MINIMIZE], search_method=TpeSearch(), timeout_seconds=1.0)
