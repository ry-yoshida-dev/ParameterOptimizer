import pytest
from omegaconf import DictConfig, OmegaConf

from parameter_optimizer import SearchMethod, SearchSpace


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
    with pytest.raises(ValueError, match=r"grid requires categorical parameters. got \['a'\]"):
        SearchMethod.GRID.validate(_space({**_NUMERIC, **_CATEGORICAL}))


def test_cma_es_rejects_categorical_parameter() -> None:
    pytest.importorskip("cmaes")
    with pytest.raises(ValueError, match=r"cma_es does not support categorical parameters: \['b'\]"):
        SearchMethod.CMA_ES.validate(_space({**_NUMERIC, **_CATEGORICAL}))


@pytest.mark.parametrize(
    "method",
    [
        SearchMethod.RANDOM,
        SearchMethod.QUASI_MONTE_CARLO,
        SearchMethod.TPE,
        SearchMethod.GAUSSIAN_PROCESS,
        SearchMethod.GENETIC,
    ],
)
def test_other_methods_accept_mixed_parameters(method: SearchMethod, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("parameter_optimizer.search.method.find_spec", _find_any_module)
    method.validate(_space({**_NUMERIC, **_CATEGORICAL}))


@pytest.mark.parametrize(
    "method, extra",
    [
        (SearchMethod.QUASI_MONTE_CARLO, "quasi-monte-carlo"),
        (SearchMethod.GAUSSIAN_PROCESS, "gaussian-process"),
        (SearchMethod.CMA_ES, "cma-es"),
    ],
)
def test_reports_missing_optional_dependency(
    method: SearchMethod,
    extra: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("parameter_optimizer.search.method.find_spec", _find_no_module)
    with pytest.raises(ModuleNotFoundError, match=f"Install the {extra} extra"):
        method.validate(_space(_NUMERIC))
