# parameter_optimizer

## Overview

Overlays a search space on a base `DictConfig`, builds the target dataclass for each Optuna trial and returns the best
configuration. Optuna types are used internally; an objective only sees the dataclass and a `TrialReporter`. Everything
public is imported from `parameter_optimizer` itself.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [optimizer.py](optimizer.py) | `ParameterOptimizer`: validates the inputs, runs the study and summarizes it |
| [parameter/](parameter/README.md) | Definition of one parameter |
| [search/](search/README.md) | What is explored and by which algorithm |
| [trial/](trial/README.md) | Evaluation of one configuration and its early stopping |
| [study/](study/README.md) | Settings and outcome of a whole run |

## Examples

`OptimizerSettings` can be built from configuration as well:

```python
settings = DataclassInitializer.build(OptimizerSettings, OmegaConf.load("settings.yaml"))
```

```yaml
direction: maximize
search_method: gaussian_process
pruning_method: hyperband
n_trials: 100
seed: 42
study_name: resnet
storage_url: sqlite:///study.db
```
