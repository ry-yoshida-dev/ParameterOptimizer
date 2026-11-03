import pytest
from omegaconf import DictConfig, OmegaConf

from parameter_optimizer import (
    CmaEsSearch,
    GaussianProcessSearch,
    GeneticSearch,
    GridSearch,
    QuasiMonteCarloSearch,
    RandomSearch,
    SearchMethod,
    SearchSpace,
    TpeSearch,
)


def _space(content: dict[str, object]) -> SearchSpace:
    config = OmegaConf.create(content)
    assert isinstance(config, DictConfig)
    return SearchSpace.from_config(config)


def _find_any_module(module: str) -> str:
    return module


def _find_no_module(module: str) -> None:
    return None


_NUMERIC: dict[str, object] = {"a": {"distribution": "float", "low": 0.0, "high": 1.0}}
_CATEGORICAL: dict[str, object] = {"b": {"distribution": "categorical", "candidates": [1, 2]}}


def test_grid_rejects_numeric_parameter() -> None:
    with pytest.raises(ValueError, match=r"grid requires categorical parameters or a step. got \['a'\]"):
        GridSearch().validate(_space({**_NUMERIC, **_CATEGORICAL}), 1)


def test_cma_es_rejects_categorical_parameter() -> None:
    pytest.importorskip("cmaes")
    with pytest.raises(ValueError, match=r"cma_es does not support categorical parameters: \['b'\]"):
        CmaEsSearch().validate(_space({**_NUMERIC, **_CATEGORICAL}), 1)


@pytest.mark.parametrize(
    "method",
    [
        RandomSearch(),
        QuasiMonteCarloSearch(),
        TpeSearch(),
        GaussianProcessSearch(),
        GeneticSearch(),
    ],
)
def test_other_methods_accept_mixed_parameters(method: SearchMethod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("parameter_optimizer.search.method.find_spec", _find_any_module)
    method.validate(_space({**_NUMERIC, **_CATEGORICAL}), 1)


@pytest.mark.parametrize(
    "method, extra",
    [
        (QuasiMonteCarloSearch(), "quasi-monte-carlo"),
        (GaussianProcessSearch(), "gaussian-process"),
        (CmaEsSearch(), "cma-es"),
    ],
)
def test_reports_missing_optional_dependency(
    method: SearchMethod,
    extra: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("parameter_optimizer.search.method.find_spec", _find_no_module)
    with pytest.raises(ModuleNotFoundError, match=f"Install the {extra} extra"):
        method.validate(_space(_NUMERIC), 1)
