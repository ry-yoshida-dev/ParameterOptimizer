# ParameterOptimizer

## Overview

`parameter_optimizer` tunes anything built from a `DictConfig`. It takes the normal configuration, a search space with
the same hierarchy naming only the keys to tune, and the optimizer settings. Each trial goes through the same build as
production, by default the strict [DataclassInitializer](https://github.com/ry-yoshida-dev/DataclassInitializer), and
every candidate is checked before the first trial. The best values come back as a normal, resolved configuration.
[Optuna](https://optuna.org) performs the search, and distributions, search methods and pruning rules can be extended
through registries.

**What each parameter can take**: `distribution` in the search space.

| `distribution` | Other keys | Values |
| -------------- | ---------- | ------ |
| `categorical` | `candidates` | One of the candidates |
| `float` | `low`, `high`, optional `is_log` or `step` | A float on a linear or logarithmic scale, or on steps |
| `int` | `low`, `high`, optional `is_log` or `step` | An integer on a linear or logarithmic scale, or on steps |

**Which combinations are tried**: `search_method` in the settings.

| `search_method` | Class | Options | Use when |
| --------------- | ----- | ------- | -------- |
| `grid` | `GridSearch` | | The combinations are few; requires `categorical` parameters or a `step`, and `n_trials` may be omitted |
| `random` | `RandomSearch` | | A baseline is needed |
| `quasi_monte_carlo` | `QuasiMonteCarloSearch` | `sequence`, `is_scrambled` | The space should be covered evenly |
| `tpe` | `TpeSearch` | `startup_trial_count`, `candidate_count`, `is_multivariate`, `is_grouped`, `is_constant_liar` | A robust default is needed, especially with mixed parameters |
| `gaussian_process` | `GaussianProcessSearch` | `startup_trial_count`, `is_deterministic_objective` | Each trial is expensive and the trials are few |
| `cma_es` | `CmaEsSearch` | `startup_trial_count`, `population_size`, `is_separable` | The parameters are numeric and the trials are many; rejects `categorical` parameters and several objectives |
| `genetic` | `GeneticSearch` | `population_size`, `mutation_probability`, `crossover_probability`, `swapping_probability` | The trials are many and the parameters are mixed |

**How long each trial runs**: `pruning_method` in the settings, combined with any search method. The objective must
call `reporter.report` for pruning to take effect.

| `pruning_method` | Class | Options | Behavior |
| ---------------- | ----- | ------- | -------- |
| `none` | `NoPruning` | | Every trial runs to completion |
| `median` | `MedianPruning` | `startup_trial_count`, `warmup_step_count`, `step_interval` | Stop a trial behind the median of earlier trials at the same step |
| `successive_halving` | `SuccessiveHalvingPruning` | `min_resource`, `reduction_factor`, `min_early_stopping_rate` | Repeatedly keep only the better part of the trials and give them more steps |
| `hyperband` | `HyperbandPruning` | `min_resource`, `max_resource`, `reduction_factor` | Run successive halving with several budgets |

## Installation

```bash
uv sync --extra dev
```

The `gaussian-process`, `cma-es` and `quasi-monte-carlo` extras enable the search methods of the same name.

## Usage

```yaml
# base.yaml: normal configuration
name: resnet
solver:
  learning_rate: 0.01
  momentum: 0.9
```

```yaml
# search_space.yaml: only the keys to tune
solver:
  learning_rate: {distribution: float, low: 0.0001, high: 0.1, is_log: true}
  momentum: {distribution: categorical, candidates: [null, 0.9, 0.99]}
```

```python
from omegaconf import OmegaConf
from parameter_optimizer import (
    DataclassBuilder,
    Direction,
    OptimizerSettings,
    ParameterOptimizer,
    TpeSearch,
    TrialReporter,
)


def objective(config: ModelConfig, reporter: TrialReporter) -> float:
    return evaluate(config)


optimizer = ParameterOptimizer(
    config_builder=DataclassBuilder(ModelConfig),
    base_config=OmegaConf.load("base.yaml"),
    search_space=OmegaConf.load("search_space.yaml"),
    settings=OptimizerSettings(directions=[Direction.MAXIMIZE], search_method=TpeSearch(), n_trials=100, seed=42),
)
result = optimizer.optimize(objective)
OmegaConf.save(result.best.resolved_config, "best.yaml")
```

See the [example](examples/README.md) for a complete run with settings read from YAML by `OptimizerSettings.from_config`.
