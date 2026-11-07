# parameter_optimizer

## Overview

Overlays a search space on a base `DictConfig`, builds the target for each Optuna trial and returns the best
configurations. Optuna types are used internally; an objective only sees the target and a `TrialReporter`. Everything
public is imported from `parameter_optimizer` itself.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [optimizer.py](optimizer.py) | `ParameterOptimizer`: validates the inputs, runs the study and summarizes it |
| [settings.py](settings.py) | `OptimizerSettings`: input of an optimization |
| [result.py](result.py) | `OptimizationResult`: output of an optimization |
| [trial_outcome.py](trial_outcome.py) | `TrialOutcome`: one of the best trials |
| [registry.py](registry.py) | `Registry`: named implementations of an interface, selectable from a configuration |
| [builder/](builder/README.md) | How a plain configuration becomes the target |
| [objective/](objective/README.md) | The function being optimized |
| [parameter/](parameter/README.md) | Definition of one parameter |
| [search/](search/README.md) | What is explored and which methods explore it |
