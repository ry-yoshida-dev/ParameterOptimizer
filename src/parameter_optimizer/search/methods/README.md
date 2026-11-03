# methods

## Overview

Concrete search methods, implementing `SearchMethod` and registered under the name written in a configuration.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [grid.py](grid.py) | `GridSearch`: every combination, retrying failed points on resume |
| [random.py](random.py) | `RandomSearch`: independent random combinations |
| [quasi_monte_carlo.py](quasi_monte_carlo.py) | `QuasiMonteCarloSearch`: low-discrepancy sequence |
| [tpe.py](tpe.py) | `TpeSearch`: Bayesian optimization with a Tree-structured Parzen Estimator |
| [gaussian_process.py](gaussian_process.py) | `GaussianProcessSearch`: Bayesian optimization with a Gaussian process |
| [cma_es.py](cma_es.py) | `CmaEsSearch`: evolution strategy |
| [genetic.py](genetic.py) | `GeneticSearch`: genetic algorithm (NSGA-II) |
