# search

## Overview

What is explored and which methods explore it.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [space.py](space.py) | `SearchSpace`: reads definitions from a config tree and writes values into a config |
| [method.py](method.py) | `SearchMethod`: algorithm deciding which combinations are tried |
| [pruning_method.py](pruning_method.py) | `PruningMethod`: rule stopping unpromising trials early |
| [resumable_grid_sampler.py](resumable_grid_sampler.py) | `ResumableGridSampler`: grid sampler retrying failed points on resume |
