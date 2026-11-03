# ParameterOptimizer

## Overview

`parameter_optimizer` tunes anything built from a `DictConfig`. It takes the normal configuration, a second
configuration naming only the keys to change, and the optimizer settings. The target stays unaware of the optimizer, and
each trial goes through the same build as production, by default the strict
[DataclassInitializer](https://github.com/ry-yoshida-dev/DataclassInitializer). [Optuna](https://optuna.org) performs
the search.

- **Same shape as the configuration**: a search space mirrors the hierarchy of the base configuration, with
  `{distribution: ..., ...}` at each key to tune.
- **Mistakes fail before the first trial**: every candidate and bound is built at construction, so unknown keys, wrong
  types (an `int` for a `float` field) and values rejected by `__post_init__` never cost a long run.
- **The result is a normal configuration**: `resolved_config` has the best values written in and its interpolations
  resolved, so it can be saved as YAML and used on its own.
- **Every choice is open**: distributions, search methods and pruning rules are interfaces with registries, so your own
  implementation is selectable from YAML next to the built-in ones.

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

Three independent choices define a run.

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
call `reporter.report`, otherwise every trial runs to completion and a warning is issued.

| `pruning_method` | Class | Options | Behavior |
| ---------------- | ----- | ------- | -------- |
| `none` | `NoPruning` | | Every trial runs to completion |
| `median` | `MedianPruning` | `startup_trial_count`, `warmup_step_count`, `step_interval` | Stop a trial behind the median of earlier trials at the same step |
| `successive_halving` | `SuccessiveHalvingPruning` | `min_resource`, `reduction_factor`, `min_early_stopping_rate` | Repeatedly keep only the better part of the trials and give them more steps |
| `hyperband` | `HyperbandPruning` | `min_resource`, `max_resource`, `reduction_factor` | Run successive halving with several budgets |

Settings can be read from YAML with `OptimizerSettings.from_config`, writing each implementation as its name or as a
mapping of `name` and its options:

```yaml
directions: [maximize]
search_method: {name: tpe, is_multivariate: true}
pruning_method: hyperband
n_trials: 100
seed: 42
n_jobs: 4
study_name: resnet
storage: sqlite:///study.db
resume_ignored_paths: [output_directory]
```

### Several objectives

List one direction per objective and return one value per direction. `result.best_trials` then holds the Pareto front,
and `result.best` works only if the front has one trial.

```python
settings = OptimizerSettings(
    directions=[Direction.MAXIMIZE, Direction.MINIMIZE], search_method=TpeSearch(), n_trials=100
)
result = ParameterOptimizer(...).optimize(lambda config, reporter: (accuracy(config), latency(config)))
for outcome in result.best_trials:
    print(outcome.values, outcome.parameters)
```

### Extending

Decorate a dataclass with the registry of its interface; the name is then usable in YAML, and its fields are its
options.

```python
from dataclasses import dataclass
from optuna.samplers import BaseSampler, BruteForceSampler
from parameter_optimizer import SearchMethod, SearchSpace, search_method_registry


@search_method_registry.register("brute_force")
@dataclass(frozen=True, kw_only=True)
class BruteForceSearch(SearchMethod):
    is_avoiding_premature_stop: bool = False

    def build_sampler(self, search_space: SearchSpace, seed: int | None) -> BaseSampler:
        return BruteForceSampler(seed=seed, avoid_premature_stop=self.is_avoiding_premature_stop)
```

| Interface | Registry | Written in |
| --------- | -------- | ---------- |
| `ParameterSpec` | `parameter_spec_registry` | `distribution` of a search space entry |
| `SearchMethod` | `search_method_registry` | `search_method` of the settings |
| `PruningMethod` | `pruning_method_registry` | `pruning_method` of the settings |

The module defining an implementation must be imported before the configuration is read. A target that is not a
dataclass needs only another `config_builder`: any callable from a `DictConfig` to the object. `optimize` also takes
`callbacks`, and `storage` also takes any Optuna storage object.

## Rules

- Only values that a normal configuration can build can be tuned. `DataclassInitializer` is strict, so a `float`
  field needs `1.0`, not `1`. Likewise, `ParameterValue` is a scalar, so a list cannot be a candidate.
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
- A study resumed from `storage` must have the same directions, search space (including bounds and candidates) and
  base configuration (except the base values of the tuned keys), otherwise the run fails before its first trial; use a
  new `study_name` when they change. The search method may change.
- The base configuration is compared after resolving its interpolations, so a key that varies between runs, such as an
  output directory or a value from `${now:}`, stops a resume unless it is listed in `resume_ignored_paths`.
- A resumed grid search evaluates each failed point again, including those failed through `caught_exceptions`.
- With a `seed`, a resumed run continues with new values: the seed is offset by the number of existing trials, so a
  run is reproducible as long as the study holds the same trials.
- Optuna cannot prune with several objectives, so a pruning rule other than `none` is rejected with them.
- `n_jobs` runs trials in threads, so the objective must be thread-safe.
