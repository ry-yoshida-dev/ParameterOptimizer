# builder

## Overview

How a plain configuration becomes the object an objective receives.

## Components

| Component | Responsibility |
| --------- | -------------- |
| [protocol.py](protocol.py) | `ConfigBuilder`: protocol of any builder |
| [dataclass.py](dataclass.py) | `DataclassBuilder`: builds a dataclass with `DataclassInitializer` |
