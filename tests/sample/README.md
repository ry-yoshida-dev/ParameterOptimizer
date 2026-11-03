# sample

## Overview

Target dataclass used by the tests.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [range_config.py](range_config.py) | `SampleRangeConfig`: target dataclass with a constraint across its fields |
| [model_config.py](model_config.py) | Target dataclass used by the tests |
| [solver_config.py](solver_config.py) | Nested dataclass with a `__post_init__` constraint |
| [mode.py](mode.py) | Enum field of the target dataclass |
