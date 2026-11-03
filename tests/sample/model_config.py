from dataclasses import dataclass

from sample.mode import SampleMode
from sample.solver_config import SampleSolverConfig


@dataclass(frozen=True, kw_only=True)
class SampleModelConfig:
    name: str
    solver: SampleSolverConfig
    mode: SampleMode = SampleMode.FAST
    depth: int = 1
