from dataclasses import dataclass
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from parameter_optimizer import (
    DataclassBuilder,
    OptimizationResult,
    OptimizerSettings,
    ParameterOptimizer,
    TrialReporter,
)


@dataclass(frozen=True, kw_only=True)
class QuadraticConfig:
    """
    Parameters of a function to minimize.

    Attributes
    ----------
    name : str
        Name of the function.
    center : float
        Input being tuned.
    scale : float
        Multiplier being tuned.
    """

    name: str
    center: float
    scale: float


def evaluate(config: QuadraticConfig, reporter: TrialReporter) -> float:
    """
    Score a configuration; the optimum is ``center == 2.0`` with ``scale == 1.0``.

    Parameters
    ----------
    config : QuadraticConfig
        Configuration to score.
    reporter : TrialReporter
        Unused, as this objective has no intermediate steps.

    Returns
    -------
    float
        Value to minimize.
    """
    return config.scale * (config.center - 2.0) ** 2 + config.scale


def main() -> None:
    """Tune the function and print the best configuration as YAML."""
    directory: Path = Path(__file__).parent
    base_config = OmegaConf.load(directory / "base.yaml")
    search_space = OmegaConf.load(directory / "search_space.yaml")
    settings_config = OmegaConf.load(directory / "settings.yaml")
    assert isinstance(base_config, DictConfig)
    assert isinstance(search_space, DictConfig)
    assert isinstance(settings_config, DictConfig)

    optimizer: ParameterOptimizer[QuadraticConfig] = ParameterOptimizer(
        config_builder=DataclassBuilder(QuadraticConfig),
        base_config=base_config.model,
        search_space=search_space.model,
        settings=OptimizerSettings.from_config(settings_config),
    )
    result: OptimizationResult[QuadraticConfig] = optimizer.optimize(evaluate)
    print(f"best value: {result.best.value:.4f}")
    print(OmegaConf.to_yaml(result.best.resolved_config))


if __name__ == "__main__":
    main()
