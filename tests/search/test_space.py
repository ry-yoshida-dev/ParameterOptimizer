import copy
from collections.abc import Iterator

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


def test_probe_groups_cover_every_value_next_to_every_other_value() -> None:
    space: SearchSpace = SearchSpace(
        {
            "a": FloatSpec(low=0.0, high=1.0),
            "b": CategoricalSpec(candidates=["x", "y", None]),
        }
    )
    groups: list[list[dict[str, ParameterValue]]] = [list(group) for group in space.iter_probe_groups()]
    assert len(groups) == 5
    assert groups[1] == [{"a": 1.0, "b": "x"}, {"a": 1.0, "b": "y"}, {"a": 1.0, "b": None}]
    assert groups[4] == [{"a": 0.0, "b": None}, {"a": 1.0, "b": None}]
    assert all(group[0].keys() == space.first_values.keys() for group in groups)


def test_probe_groups_are_bounded() -> None:
    space: SearchSpace = SearchSpace({str(index): CategoricalSpec(candidates=[0, 1]) for index in range(20)})
    group: list[dict[str, ParameterValue]] = list(next(space.iter_probe_groups()))
    assert len(group) == 1024


def test_probe_groups_can_be_read_after_being_collected() -> None:
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[1, 2]), "b": CategoricalSpec(candidates=[3])})
    groups: list[Iterator[dict[str, ParameterValue]]] = list(space.iter_probe_groups())
    assert [list(group) for group in groups][:2] == [[{"a": 1, "b": 3}], [{"a": 2, "b": 3}]]


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


def test_apply_to_part_of_a_configuration_updates_absolute_interpolations() -> None:
    root_config: DictConfig = _config(
        {"shared": 1, "model": {"a": 0, "b": "${model.a}", "c": "${shared}", "d": "${.a}"}}
    )
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    resolved_config: DictConfig = space.apply(root_config.model, {"a": 5})
    assert OmegaConf.to_container(resolved_config) == {"a": 5, "b": 5, "c": 1, "d": 5}
    assert root_config.model.a == 0


def test_apply_keeps_values_merged_into_part_of_a_configuration() -> None:
    root_config: DictConfig = _config({"model": {"a": 0, "b": 1, "c": "${model.b}"}})
    merged_config = OmegaConf.merge(root_config.model, {"b": 99})
    assert isinstance(merged_config, DictConfig)
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    assert OmegaConf.to_container(space.apply(merged_config, {"a": 5})) == {"a": 5, "b": 99, "c": 99}
    assert root_config.model.b == 1


def test_apply_keeps_values_edited_in_a_copy_of_part_of_a_configuration() -> None:
    root_config: DictConfig = _config({"group": {"model": {"a": 0, "b": 1}}})
    copied_group: DictConfig = copy.deepcopy(root_config.group)
    copied_group.model.b = 77
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    assert OmegaConf.to_container(space.apply(copied_group.model, {"a": 5})) == {"a": 5, "b": 77}


def test_apply_to_a_copy_whose_key_left_the_configuration() -> None:
    root_config: DictConfig = _config({"model": {"a": 0, "b": 1}})
    copied_config: DictConfig = copy.deepcopy(root_config.model)
    del root_config["model"]
    OmegaConf.set_struct(root_config, True)
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    assert OmegaConf.to_container(space.apply(copied_config, {"a": 5})) == {"a": 5, "b": 1}


def test_apply_to_a_list_element_of_a_configuration() -> None:
    root_config: DictConfig = _config({"models": [{"a": 0, "b": "${models[0].a}"}]})
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    assert OmegaConf.to_container(space.apply(root_config.models[0], {"a": 5})) == {"a": 5, "b": 5}


def test_apply_to_a_key_containing_a_dot() -> None:
    root_config: DictConfig = _config({"model.v1": {"a": 0}})
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[5])})
    assert OmegaConf.to_container(space.apply(root_config["model.v1"], {"a": 5})) == {"a": 5}


@pytest.mark.parametrize("is_part", [False, True])
def test_apply_accepts_read_only_configuration(is_part: bool) -> None:
    root_config: DictConfig = _config({"model": {"a": 0}})
    base_config: DictConfig = root_config.model if is_part else root_config
    OmegaConf.set_readonly(base_config, True)
    path: str = "a" if is_part else "model.a"
    space: SearchSpace = SearchSpace({path: CategoricalSpec(candidates=[5])})
    assert OmegaConf.select(space.apply(base_config, {path: 5}), path) == 5
    assert OmegaConf.is_readonly(base_config)


def test_apply_requires_every_parameter() -> None:
    space: SearchSpace = SearchSpace({"a": CategoricalSpec(candidates=[1])})
    with pytest.raises(ValueError, match="Expected values"):
        space.apply(_config({}), {})
