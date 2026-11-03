# pruning_methods

## Overview

Concrete pruning rules, implementing `PruningMethod` and registered under the name written in a configuration.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [none.py](none.py) | `NoPruning`: every trial runs to completion |
| [median.py](median.py) | `MedianPruning`: stops a trial behind the median of earlier trials |
| [successive_halving.py](successive_halving.py) | `SuccessiveHalvingPruning`: keeps the better part of the trials |
| [hyperband.py](hyperband.py) | `HyperbandPruning`: successive halving with several budgets |
