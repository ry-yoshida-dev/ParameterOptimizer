# parameter

## Overview

Definition of one parameter, written as `{distribution: ..., ...}` in a search space.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [spec.py](spec.py) | `ParameterSpec`: interface of a search definition |
| [specs/](specs/README.md) | Concrete search definitions |
| [distribution.py](distribution.py) | `ParameterDistribution`: builds a search definition from a search space entry |
| [value.py](value.py) | `ParameterValue`: scalar types a parameter can take |
