from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class SampleRangeConfig:
    """Target dataclass with a constraint across its fields."""

    low: int
    high: int

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"low must not exceed high. got low={self.low}, high={self.high}")
        if self.high > 5:
            raise ValueError(f"high must not exceed 5. got {self.high}")
