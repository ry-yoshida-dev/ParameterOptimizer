import pytest
from dataclass_initializer import DataclassInitializer

from parameter_optimizer import CategoricalSpec, FloatSpec, IntSpec, ParameterSpec, parameter_spec_registry


@pytest.mark.parametrize(
    "spec, probe_values",
    [
        (CategoricalSpec(candidates=[None, 0.5, "text", True]), [None, 0.5, "text", True]),
        (FloatSpec(low=-1.0, high=1.0), [-1.0, 1.0]),
        (FloatSpec(low=0.001, high=1.0, is_log=True), [0.001, 1.0]),
        (IntSpec(low=0, high=8), [0, 8]),
        (IntSpec(low=1, high=1024, is_log=True), [1, 1024]),
    ],
)
def test_probes_every_candidate_or_both_bounds(spec: ParameterSpec, probe_values: list[object]) -> None:
    assert spec.probe_values == probe_values


@pytest.mark.parametrize(
    "candidates, error_type",
    [
        ([], ValueError),
        (["a", "a"], ValueError),
        (0.5, TypeError),
        ([[1.0, 2.0]], TypeError),
        ([{"a": 1.0}], TypeError),
    ],
)
def test_categorical_rejects_invalid_candidates(candidates: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        DataclassInitializer.build(CategoricalSpec, {"candidates": candidates})


@pytest.mark.parametrize(
    "low, high, is_log, error_type",
    [
        (1.0, 0.0, False, ValueError),
        (1.0, 1.0, False, ValueError),
        (0, 1, False, TypeError),
        (0.0, float("inf"), False, ValueError),
        (float("nan"), 1.0, False, ValueError),
        (0.0, 1.0, True, ValueError),
    ],
)
def test_float_rejects_invalid_range(low: float, high: float, is_log: bool, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        FloatSpec(low=low, high=high, is_log=is_log)


@pytest.mark.parametrize(
    "low, high, is_log, error_type",
    [
        (2, 1, False, ValueError),
        (1, 1, False, ValueError),
        (1.0, 2.0, False, TypeError),
        (False, True, False, TypeError),
        (0, 8, True, ValueError),
    ],
)
def test_int_rejects_invalid_range(low: int, high: int, is_log: bool, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        IntSpec(low=low, high=high, is_log=is_log)


@pytest.mark.parametrize(
    "definition, spec",
    [
        ({"distribution": "categorical", "candidates": ["a"]}, CategoricalSpec(candidates=["a"])),
        (
            {"distribution": "float", "low": 0.1, "high": 1.0, "is_log": True},
            FloatSpec(low=0.1, high=1.0, is_log=True),
        ),
        ({"distribution": "int", "low": 1, "high": 2}, IntSpec(low=1, high=2)),
        ({"distribution": "int", "low": 1, "high": 5, "step": 2}, IntSpec(low=1, high=5, step=2)),
        ({"distribution": "float", "low": 0.0, "high": 1.0, "step": 0.5}, FloatSpec(low=0.0, high=1.0, step=0.5)),
    ],
)
def test_registry_builds_spec(definition: dict[str, object], spec: ParameterSpec) -> None:
    assert parameter_spec_registry.build(definition) == spec


@pytest.mark.parametrize(
    "definition",
    [
        {"distribution": "categorical", "low": 0.0, "high": 1.0},
        {"distribution": "float", "candidates": [0.0, 1.0]},
        {"distribution": "int", "low": 1},
        {"distribution": "unknown"},
    ],
)
def test_registry_rejects_fields_of_another_kind(definition: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        parameter_spec_registry.build(definition)


@pytest.mark.parametrize(
    "spec_class, fields",
    [
        (FloatSpec, {"low": 0.1, "high": 1.0, "is_log": True, "step": 0.1}),
        (FloatSpec, {"low": 0.0, "high": 1.0, "step": 0.0}),
        (FloatSpec, {"low": 0.0, "high": 1.0, "step": float("inf")}),
        (IntSpec, {"low": 1, "high": 4, "is_log": True, "step": 2}),
        (IntSpec, {"low": 1, "high": 4, "step": 0}),
    ],
)
def test_rejects_invalid_step(spec_class: type[ParameterSpec], fields: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        spec_class(**fields)


def test_probe_stops_at_the_last_value_a_step_reaches() -> None:
    assert IntSpec(low=0, high=7, step=3).probe_values == [0, 6]
