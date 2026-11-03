# ParameterOptimizer

## Overview

`parameter_optimizer` tunes any dataclass built from a `DictConfig`. It takes the normal configuration, a second
configuration naming only the keys to change, and the optimizer settings. The target dataclass stays unaware of the
optimizer, and each trial goes through the same [DataclassInitializer](https://github.com/ry-yoshida-dev/DataclassInitializer)
build as production. [Optuna](https://optuna.org) performs the search.

- **Same shape as the configuration**: a search space mirrors the hierarchy of the base configuration, with
  `{distribution: ..., ...}` at each key to tune.
- **Mistakes fail before the first trial**: every candidate and bound is built at construction, so unknown keys, wrong
  types (an `int` for a `float` field) and values rejected by `__post_init__` never cost a long run.
- **The result is a normal configuration**: `resolved_config` has the best values written in and its interpolations
  resolved, so it can be saved as YAML and used on its own.

See the [package documentation](src/parameter_optimizer/README.md) for the components and the
[example](examples/README.md) for a complete run.

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
from parameter_optimizer import Direction, OptimizerSettings, ParameterOptimizer, SearchMethod, TrialReporter


def objective(config: ModelConfig, reporter: TrialReporter) -> float:
    return evaluate(config)


optimizer = ParameterOptimizer(
    config_class=ModelConfig,
    base_config=OmegaConf.load("base.yaml"),
    search_space=OmegaConf.load("search_space.yaml"),
    settings=OptimizerSettings(direction=Direction.MAXIMIZE, search_method=SearchMethod.TPE, n_trials=100, seed=42),
)
result = optimizer.optimize(objective)
OmegaConf.save(result.resolved_config, "best.yaml")
```

Three independent choices define a run.

**What each parameter can take**: `distribution` in the search space.

| `distribution` | Other keys | Values |
| -------------- | ---------- | ------ |
| `categorical` | `candidates` | One of the candidates |
| `float` | `low`, `high`, optional `is_log` | A float on a linear or logarithmic scale |
| `int` | `low`, `high`, optional `is_log` | An integer on a linear or logarithmic scale |

**Which combinations are tried**: `search_method` in the settings.

| `search_method` | Approach | Use when |
| --------------- | -------- | -------- |
| `grid` | Exhaustive | The combinations are few; requires only `categorical` parameters, and `n_trials` may be omitted |
| `random` | Random | A baseline is needed |
| `quasi_monte_carlo` | Low-discrepancy sequence | The space should be covered evenly |
| `tpe` | Bayesian optimization | A robust default is needed, especially with mixed parameters |
| `gaussian_process` | Bayesian optimization | Each trial is expensive and the trials are few |
| `cma_es` | Evolution strategy | The parameters are numeric and the trials are many; rejects `categorical` parameters |
| `genetic` | Genetic algorithm (NSGA-II) | The trials are many and the parameters are mixed |

**How long each trial runs**: `pruning_method` in the settings, combined with any search method. The objective must
call `reporter.report`, otherwise every trial runs to completion and a warning is issued.

| `pruning_method` | Behavior |
| ---------------- | -------- |
| `none` | Every trial runs to completion |
| `median` | Stop a trial behind the median of earlier trials at the same step |
| `successive_halving` | Repeatedly keep only the better part of the trials and give them more steps |
| `hyperband` | Run successive halving with several budgets |

## Rules

- Only values that a normal configuration can build can be tuned. `DataclassInitializer` is strict, so a `float`
  field needs `1.0`, not `1`.
- A mapping whose `distribution` key holds a string is always a search definition, so a nested dataclass with such a
  field cannot be tuned through its own fields.
- The checks at construction try each candidate and bound next to the candidates and bounds of the other parameters,
  and fail only if every tried combination raises `ValueError`. A combination rejected by a constraint across
  parameters stops the run when it is sampled, unless its exception is listed in `caught_exceptions`.
- Set `is_probing_enabled` to `False` for a constraint that no candidate or bound satisfies, such as a divisibility that
  only values inside the bounds meet. Other mistakes then surface at the first trial that samples them.
- Any exception from the objective stops the run unless it is listed in `caught_exceptions`.
- Parameters that are only valid together, such as a model family and its weights, belong in separate base
  configurations and studies.
- A study resumed from `storage_url` must have the same direction, search space (including bounds and candidates) and
  base configuration (except the base values of the tuned keys), otherwise the run fails before its first trial; use a
  new `study_name` when they change. The search method may change.
- The base configuration is compared after resolving its interpolations, so a key that varies between runs, such as an
  output directory or a value from `${now:}`, stops a resume.
- A resumed grid search evaluates each failed point again, including those failed through `caught_exceptions`.
- With a `seed`, a resumed run continues with new values: the seed is offset by the number of existing trials, so a
  run is reproducible as long as the study holds the same trials.
