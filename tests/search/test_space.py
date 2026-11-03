import pytest
from omegaconf import DictConfig, OmegaConf

from parameter_optimizer import CategoricalSpec, FloatSpec, ParameterValue, SearchSpace


def _config(content: dict[str, object]) -> DictConfig:
    config = OmegaConf.create(content)
    assert isinstance(config, DictConfig)
    return config


def test_collects_nested_definitions_as_dotted_paths() -> None:
    space: SearchSpace = SearchSpace.from_config(
        _config(
            {
                "mode": {"distribution": "categorical", "candidates": ["fast", "slow"]},
                "solver": {"learning_rate": {"distribution": "float", "low": 0.1, "high": 0.2}},
            }
        )
    )
    assert space.categorical_candidates == {"mode": ["fast", "slow"]}
    assert space.numeric_paths == ["solver.learning_rate"]


@pytest.mark.parametrize(
    "content",
    [
        {},
        {"depth": 3},
        {"solver": {}},
        {"depth": {"distribution": "categorical", "values": [1]}},
        {"depth": {"distribution": "categorical", "low": 0, "high": 1}},
        {"depth": {"distribution": "grid", "candidates": [1]}},
        {"depth": {"distribution": "tpe", "low": 1, "high": 2}},
        {"depth": {"method": "categorical", "candidates": [1]}},
        {"depth": {"distribution": "unknown", "value": [1]}},
        {"depth": {"distribution": "int", "low": 1.0, "high": 2.0}},
    ],
)
def test_rejects_invalid_search_config(content: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        SearchSpace.from_config(_config(content))


def test_reports_parameter_path_of_invalid_definition() -> None:
    with pytest.raises(ValueError, match=r"solver\.learning_rate"):
        SearchSpace.from_config(_config({"solver": {"learning_rate": {"distribution": "float", "low": 0, "high": 1}}}))


def test_field_named_distribution_can_be_tuned() -> None:
    space: SearchSpace = SearchSpace.from_config(
        _config({"distribution": {"distribution": "categorical", "candidates": ["normal", "uniform"]}})
    )
    assert space.categorical_candidates == {"distribution": ["normal", "uniform"]}


def test_probes_cover_every_value_of_every_parameter() -> None:
    space: SearchSpace = SearchSpace(
        {
            "a": FloatSpec(low=0.0, high=1.0),
            "b": CategoricalSpec(candidates=["x", "y", None]),
        }
    )
    probes: list[dict[str, ParameterValue]] = list(space.iter_probes())
    assert {probe["a"] for probe in probes} == {0.0, 1.0}
    assert {probe["b"] for probe in probes} == {"x", "y", None}


def test_apply_writes_values_without_modifying_base() -> None:
    base_config: DictConfig = _config({"name": "base", "solver": {"learning_rate": 0.5}, "alias": "${name}"})
    space: SearchSpace = SearchSpace({"solver.learning_rate": CategoricalSpec(candidates=[0.1])})
    resolved_config: DictConfig = space.apply(base_config, {"solver.learning_rate": 0.1})
    assert resolved_config.solver.learning_rate == 0.1
    assert resolved_config.alias == "base"
    assert base_config.solver.learning_rate == 0.5


def test_apply_replaces_interpolated_source() -> None:
    base_config: DictConfig = _config({"name": "base", "alias": "${name}"})
    space: SearchSpace = SearchSpace({"name": CategoricalSpec(candidates=["other"])})
    assert space.apply(base_config, {"name": "other"}).alias == "other"


def test_apply_requires_every_parameter() -> None:
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[1])})
    with pytest.raises(ValueError, match="Expected values"):
        space.apply(_config({}), {})
