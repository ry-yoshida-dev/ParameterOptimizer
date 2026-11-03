# parameter

## Overview

Definition of one parameter, written as `{distribution: ..., ...}` in a search space. Decorating a `ParameterSpec`
dataclass with `parameter_spec_registry.register` adds a distribution.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [spec.py](spec.py) | `ParameterSpec`: interface of a search definition, and `parameter_spec_registry`: its registry |
| [specs/](specs/README.md) | Concrete search definitions |
| [value.py](value.py) | `ParameterValue`: scalar types a parameter can take |
