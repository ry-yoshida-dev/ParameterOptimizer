from dataclasses import dataclass

import pytest
from omegaconf import DictConfig, OmegaConf
from optuna import Trial
from optuna.distributions import BaseDistribution, IntDistribution
from optuna.pruners import BasePruner, NopPruner
from optuna.samplers import BaseSampler, RandomSampler
from sample.model_config import SampleModelConfig

from parameter_optimizer import (
    DataclassBuilder,
    OptimizationResult,
    OptimizerSettings,
    ParameterOptimizer,
    ParameterSpec,
    ParameterValue,
    PruningMethod,
    Registry,
    SearchMethod,
    SearchSpace,
    TrialReporter,
    parameter_spec_registry,
    pruning_method_registry,
    search_method_registry,
)


@search_method_registry.register("sample_search")
@dataclass(frozen=True, kw_only=True)
class SampleSearch(SearchMethod):
    """Search method defined outside of the package, with an option of its own."""

    label: str = "default"

    def build_sampler(self, search_space: SearchSpace, seed: int | None) -> BaseSampler:
        """
        Create a random sampler.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Random seed.

        Returns
        -------
        BaseSampler
            New sampler.
        """
        return RandomSampler(seed=seed)


@pruning_method_registry.register("sample_pruning")
@dataclass(frozen=True, kw_only=True)
class SamplePruning(PruningMethod):
    """Pruning rule defined outside of the package."""

    def build(self) -> BasePruner:
        """
        Create a pruner.

        Returns
        -------
        BasePruner
            New pruner.
        """
        return NopPruner()


@parameter_spec_registry.register("sample_power_of_two")
@dataclass(frozen=True, kw_only=True)
class SamplePowerOfTwoSpec(ParameterSpec):
    """Distribution defined outside of the package, recording an exponent and returning its power of two."""

    max_exponent: int

    @property
    def probe_values(self) -> list[ParameterValue]:
        """
        Both bounds.

        Returns
        -------
        list[ParameterValue]
            Smallest and largest value.
        """
        return [1, 2**self.max_exponent]

    def to_optuna_distribution(self) -> BaseDistribution:
        """
        Convert to the distribution of the exponent.

        Returns
        -------
        BaseDistribution
            Equivalent Optuna distribution.
        """
        return IntDistribution(0, self.max_exponent)

    def suggest(self, trial: Trial, name: str) -> ParameterValue:
        """
        Ask a trial for an exponent and return the power.

        Parameters
        ----------
        trial : Trial
            Trial choosing the value.
        name : str
            Parameter name recorded in the study.

        Returns
        -------
        ParameterValue
            Chosen power of two.
        """
        return 1 << trial.suggest_int(name, 0, self.max_exponent)


def _config(content: dict[str, object]) -> DictConfig:
    config = OmegaConf.create(content)
    assert isinstance(config, DictConfig)
    return config


def test_custom_implementations_are_selectable_from_a_configuration() -> None:
    settings: OptimizerSettings = OptimizerSettings.from_config(
        {
            "directions": ["minimize"],
            "search_method": {"name": "sample_search", "label": "custom"},
            "pruning_method": "sample_pruning",
            "n_trials": 5,
            "seed": 0,
        },
    )
    assert settings.search_method == SampleSearch(label="custom")

    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        reporter.report(0.0, 0)
        assert config.depth in (1, 2, 4, 8)
        return float(config.depth)

    result: OptimizationResult[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=_config({"name": "base", "solver": {"learning_rate": 0.5, "momentum": 0.9}}),
        search_space=_config({"depth": {"distribution": "sample_power_of_two", "max_exponent": 3}}),
        settings=settings,
    ).optimize(objective)
    assert len(result.study.trials) == 5
    assert result.best.parameters == {"depth": result.best.config.depth}


def test_values_of_a_transforming_spec_reach_the_result() -> None:
    def objective(config: SampleModelConfig, reporter: TrialReporter) -> float:
        return -float(config.depth)

    result: OptimizationResult[SampleModelConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(SampleModelConfig),
        base_config=_config({"name": "base", "solver": {"learning_rate": 0.5, "momentum": 0.9}}),
        search_space=_config({"depth": {"distribution": "sample_power_of_two", "max_exponent": 3}}),
        settings=OptimizerSettings.from_config(
            {"directions": ["minimize"], "search_method": "sample_search", "n_trials": 30, "seed": 0}
        ),
    ).optimize(objective)
    assert result.best.config.depth == 8
    assert result.best.parameters == {"depth": 8}
    assert result.best.resolved_config.depth == 8
    assert result.study.best_trial.params == {"depth": 3}


def test_a_taken_name_cannot_be_registered_again() -> None:
    with pytest.raises(ValueError, match="already registered"):
        search_method_registry.register("sample_search")(SampleSearch)


def test_only_dataclass_implementations_of_the_interface_can_be_registered() -> None:
    class PlainSearch(SearchMethod):
        def build_sampler(self, search_space: SearchSpace, seed: int | None) -> BaseSampler:
            return RandomSampler(seed=seed)

    with pytest.raises(TypeError, match="must be a dataclass"):
        search_method_registry.register("sample_plain")(PlainSearch)
    with pytest.raises(TypeError, match="deriving from SearchMethod"):
        search_method_registry.register("sample_pruning")(SamplePruning)


def test_registry_works_for_any_interface() -> None:
    @dataclass(frozen=True, kw_only=True)
    class Shape:
        pass

    @dataclass(frozen=True, kw_only=True)
    class Circle(Shape):
        radius: float = 1.0

    registry: Registry[Shape] = Registry(base_class=Shape, key="kind")
    registry.register("circle")(Circle)
    assert registry.build("circle") == Circle()
    assert registry.build({"kind": "circle", "radius": 2.0}) == Circle(radius=2.0)


@pytest.mark.parametrize(
    "definition, error_type",
    [
        ("unknown", ValueError),
        ({"radius": 1.0}, ValueError),
        ({"name": 3}, ValueError),
        (3, TypeError),
    ],
)
def test_registry_rejects_invalid_definitions(definition: object, error_type: type[Exception]) -> None:
    with pytest.raises(error_type):
        search_method_registry.build(definition)
