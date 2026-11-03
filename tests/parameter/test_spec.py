import pytest
from dataclass_initializer import DataclassInitializer

from parameter_optimizer import CategoricalSpec, FloatSpec, IntSpec, ParameterDistribution, ParameterSpec


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
        (1.0, 2.0, False, TypeError),
        (False, True, False, TypeError),
        (0, 8, True, ValueError),
    ],
)
def test_int_rejects_invalid_range(low: int, high: int, is_log: bool, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        IntSpec(low=low, high=high, is_log=is_log)


@pytest.mark.parametrize(
    "distribution, fields, spec",
    [
        (ParameterDistribution.CATEGORICAL, {"candidates": ["a"]}, CategoricalSpec(candidates=["a"])),
        (
            ParameterDistribution.FLOAT,
            {"low": 0.1, "high": 1.0, "is_log": True},
            FloatSpec(low=0.1, high=1.0, is_log=True),
        ),
        (ParameterDistribution.INT, {"low": 1, "high": 2}, IntSpec(low=1, high=2)),
    ],
)
def test_distribution_builds_spec(
    distribution: ParameterDistribution,
    fields: dict[str, object],
    spec: ParameterSpec,
) -> None:
    assert distribution.build_spec(fields) == spec


@pytest.mark.parametrize(
    "distribution, fields",
    [
        (ParameterDistribution.CATEGORICAL, {"low": 0.0, "high": 1.0}),
        (ParameterDistribution.FLOAT, {"candidates": [0.0, 1.0]}),
        (ParameterDistribution.INT, {"low": 1}),
    ],
)
def test_distribution_rejects_fields_of_another_kind(
    distribution: ParameterDistribution,
    fields: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        distribution.build_spec(fields)
