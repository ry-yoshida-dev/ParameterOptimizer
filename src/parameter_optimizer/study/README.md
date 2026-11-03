# study

## Overview

Settings and outcome of a whole run.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [settings.py](settings.py) | `OptimizerSettings`: direction, search and pruning methods, trial limits, seed and storage |
| [direction.py](direction.py) | `Direction`: minimize or maximize |
| [result.py](result.py) | `OptimizationResult`: best dataclass, resolved configuration, score and study |
