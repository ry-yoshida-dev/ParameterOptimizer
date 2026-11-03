# objective

## Overview

The function being optimized: its signature, the direction of its value and how it reports intermediate values.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [protocol.py](protocol.py) | `Objective`: protocol of the function being optimized |
| [direction.py](direction.py) | `Direction`: minimize or maximize |
| [reporter.py](reporter.py) | `TrialReporter`: receives intermediate values and triggers pruning |
