# trial

## Overview

Evaluation of one configuration and its early stopping.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [objective.py](objective.py) | `Objective`: protocol of the function being optimized |
| [reporter.py](reporter.py) | `TrialReporter`: receives intermediate values and triggers pruning |
| [pruning_method.py](pruning_method.py) | `PruningMethod`: rule stopping unpromising trials early |
