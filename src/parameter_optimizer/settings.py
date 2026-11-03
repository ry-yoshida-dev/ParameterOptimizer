import math
from dataclasses import dataclass

from .objective import Direction
from .search import PruningMethod, SearchMethod


@dataclass(frozen=True, kw_only=True)
class OptimizerSettings:
    """
    How an optimization runs, independent of what is being tuned.

    Attributes
    ----------
    direction : Direction
        Whether the objective is minimized or maximized.
    search_method : SearchMethod
        Algorithm deciding which parameter combinations are tried.
    n_trials : int | None
        Maximum number of trials in this run; None means no limit, which only ``GRID`` allows without a timeout.
    timeout_seconds : float | None
        Time after which no new trial starts; None means no limit.
    pruning_method : PruningMethod
        Rule stopping unpromising trials early.
    seed : int | None
        Search seed; None means a non-reproducible run. ``GRID`` ignores it, as its order is fixed.
    study_name : str | None
        Name of the study, required with ``storage_url``.
    storage_url : str | None
        Database URL such as ``sqlite:///study.db``; an existing study of the same name is resumed.
    is_probing_enabled : bool
        Whether candidates and bounds are checked at construction. Disable it only for a constraint across parameters
        that no candidate or bound satisfies, as other mistakes then surface at the first trial that samples them.

    Raises
    ------
    ValueError
        If ``n_trials`` is not positive, ``timeout_seconds`` is not positive and finite, ``seed`` is outside what Optuna
        accepts, ``storage_url`` lacks a ``study_name``, or a method other than ``GRID`` has neither ``n_trials`` nor
        ``timeout_seconds``.
    """

    direction: Direction
    search_method: SearchMethod
    n_trials: int | None = None
    timeout_seconds: float | None = None
    pruning_method: PruningMethod = PruningMethod.NONE
    seed: int | None = None
    study_name: str | None = None
    storage_url: str | None = None
    is_probing_enabled: bool = True

    def __post_init__(self) -> None:
        if self.n_trials is not None and self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive. got {self.n_trials}")
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0
        ):
            raise ValueError(f"timeout_seconds must be positive and finite. got {self.timeout_seconds}")
        if self.seed is not None and not 0 <= self.seed < 2**32:
            raise ValueError(f"seed must be between 0 and 2**32 - 1. got {self.seed}")
        if self.storage_url is not None and self.study_name is None:
            raise ValueError("study_name is required with storage_url, otherwise the study cannot be resumed.")
        is_unbounded: bool = self.n_trials is None and self.timeout_seconds is None
        if is_unbounded and self.search_method is not SearchMethod.GRID:
            raise ValueError(f"{self.search_method.value} requires n_trials or timeout_seconds.")
