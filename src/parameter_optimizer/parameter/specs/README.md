# specs

## Overview

Concrete search definitions of one parameter, implementing `ParameterSpec`.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [categorical.py](categorical.py) | `CategoricalSpec`: one of the listed candidates |
| [float.py](float.py) | `FloatSpec`: a float within a range, optionally on a logarithmic scale |
| [int.py](int.py) | `IntSpec`: an integer within a range, optionally on a logarithmic scale |
