# search

## Overview

What is explored and which methods explore it. Search methods and pruning rules are interfaces with registries, so a new
one is selectable from a configuration as soon as it is registered.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [space.py](space.py) | `SearchSpace`: reads definitions from a config tree and writes values into a config |
| [method.py](method.py) | `SearchMethod`: interface of the algorithm deciding which combinations are tried, and `search_method_registry`: its registry |
| [methods/](methods/README.md) | Concrete search methods |
| [pruning_method.py](pruning_method.py) | `PruningMethod`: interface of the rule stopping unpromising trials early, and `pruning_method_registry`: its registry |
| [pruning_methods/](pruning_methods/README.md) | Concrete pruning rules |
| [resumable_grid_sampler.py](resumable_grid_sampler.py) | `ResumableGridSampler`: grid sampler retrying failed points on resume |
