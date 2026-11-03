# tests

## Overview

Tests of the public behavior of `parameter_optimizer`, run with `pytest`, laid out like the package. The cases of search
methods with optional dependencies are skipped when their extras are not installed.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [test_optimizer.py](test_optimizer.py) | End-to-end optimization, early failures, pruning, resuming and every search method |
| [test_settings.py](test_settings.py) | Building and validating `OptimizerSettings` |
| [test_extension.py](test_extension.py) | Implementations defined outside of the package, and `Registry` |
| [test_example.py](test_example.py) | Running the example |
| [parameter/](parameter/README.md) | Tests of parameter definitions |
| [search/](search/README.md) | Tests of search spaces and search methods |
| [sample/](sample/README.md) | Target dataclass used by the tests |
