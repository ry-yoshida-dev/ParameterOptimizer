from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SampleSolverConfig:
    learning_rate: float
    momentum: float | None = None

    def __post_init__(self) -> None:
        if self.learning_rate > 1.0:
            raise ValueError(f"learning_rate must not exceed 1.0. got {self.learning_rate}")
