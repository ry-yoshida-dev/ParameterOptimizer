# examples

## Overview

A complete run driven by YAML files: run `python examples/quadratic.py` to tune a small function and print the best
configuration as YAML.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [quadratic.py](quadratic.py) | Target dataclass, objective and optimizer setup |
| [base.yaml](base.yaml) | Normal configuration |
| [search_space.yaml](search_space.yaml) | Keys to tune |
| [settings.yaml](settings.yaml) | Optimizer settings |
