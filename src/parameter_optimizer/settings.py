import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, cast

from dataclass_initializer import DataclassInitializer
from omegaconf import DictConfig, OmegaConf
from optuna.storages import BaseStorage

from .objective import Direction
from .search import NoPruning, PruningMethod, SearchMethod, pruning_method_registry, search_method_registry


@dataclass(frozen=True, kw_only=True)
class OptimizerSettings:
    """
    How an optimization runs, independent of what is being tuned.

    Attributes
    ----------
    directions : list[Direction]
        Whether each objective is minimized or maximized.
    search_method : SearchMethod
        Algorithm deciding which parameter combinations are tried.
    n_trials : int | None
        Maximum number of trials in this run; None means no limit, which requires a timeout unless the search method
        ends by itself.
    timeout_seconds : float | None
        Time after which no new trial starts; None means no limit.
    pruning_method : PruningMethod
        Rule stopping unpromising trials early.
    seed : int | None
        Search seed; None means a non-reproducible run. A search method with a fixed order ignores it.
    n_jobs : int
        Trials running in parallel threads; -1 uses one thread per CPU.
    study_name : str | None
        Name of the study, required with ``storage``.
    storage : str | BaseStorage | None
        Database URL such as ``sqlite:///study.db``, or any Optuna storage; an existing study of the same name is
        resumed.
    resume_ignored_paths : list[str]
        Dotted paths of the base configuration that may change between the runs of a study, such as an output
        directory. They must not affect the objective.
    is_probing_enabled : bool
        Whether candidates and bounds are checked at construction. Disable it only for a constraint across parameters
        that no candidate or bound satisfies, as other mistakes then surface at the first trial that samples them.

    Raises
    ------
    ValueError
        If ``directions`` is empty, ``n_trials`` is not positive, ``timeout_seconds`` is not positive and finite,
        ``seed`` is outside what Optuna accepts, ``n_jobs`` is neither positive nor -1, ``storage`` lacks a
        ``study_name``, a pruning rule is used with several objectives, or a search method that does not end by itself
        has neither ``n_trials`` nor ``timeout_seconds``.
    """

    _SEED_RANGE: ClassVar[int] = 2**32

    directions: list[Direction]
    search_method: SearchMethod
    n_trials: int | None = None
    timeout_seconds: float | None = None
    pruning_method: PruningMethod = NoPruning()
    seed: int | None = None
    n_jobs: int = 1
    study_name: str | None = None
    storage: str | BaseStorage | None = None
    resume_ignored_paths: list[str] = field(default_factory=list[str])
    is_probing_enabled: bool = True

    def __post_init__(self) -> None:
        if not self.directions:
            raise ValueError("directions must hold at least one direction.")
        if self.n_trials is not None and self.n_trials <= 0:
            raise ValueError(f"n_trials must be positive. got {self.n_trials}")
        if self.timeout_seconds is not None and (
            not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0.0
        ):
            raise ValueError(f"timeout_seconds must be positive and finite. got {self.timeout_seconds}")
        if self.seed is not None and not 0 <= self.seed < self._SEED_RANGE:
            raise ValueError(f"seed must be between 0 and 2**32 - 1. got {self.seed}")
        if self.n_jobs <= 0 and self.n_jobs != -1:
            raise ValueError(f"n_jobs must be positive or -1. got {self.n_jobs}")
        if self.storage is not None and self.study_name is None:
            raise ValueError("study_name is required with storage, otherwise the study cannot be resumed.")
        if len(self.directions) > 1 and self.pruning_method.is_stopping_trials:
            raise ValueError("Optuna cannot prune trials with several objectives.")
        is_unbounded: bool = self.n_trials is None and self.timeout_seconds is None
        if is_unbounded and not self.search_method.is_self_terminating:
            raise ValueError(f"{type(self.search_method).__name__} requires n_trials or timeout_seconds.")

    @classmethod
    def from_config(cls, config: DictConfig | Mapping[str, object]) -> "OptimizerSettings":
        """
        Read settings from a configuration, selecting implementations through the registries.

        Parameters
        ----------
        config : DictConfig | Mapping[str, object]
            Fields of the settings. ``directions`` lists names, and ``search_method`` and ``pruning_method`` are each a
            name or a mapping of ``name`` and the fields of the implementation.

        Returns
        -------
        OptimizerSettings
            Validated settings.

        Raises
        ------
        TypeError
            If a field has the wrong type.
        ValueError
            If a field is missing, unknown or rejected.
        """
        fields: dict[str, object] = (
            cast("dict[str, object]", OmegaConf.to_container(config, resolve=True))
            if isinstance(config, DictConfig)
            else dict(config)
        )
        if "directions" in fields:
            fields["directions"] = [Direction(name) for name in cast("list[str]", fields["directions"])]
        if "search_method" in fields:
            fields["search_method"] = search_method_registry.build(fields["search_method"])
        if "pruning_method" in fields:
            fields["pruning_method"] = pruning_method_registry.build(fields["pruning_method"])
        return DataclassInitializer.build(cls, fields)

    def seed_after(self, trial_count: int) -> int | None:
        """
        Seed of a run that continues a study, because the same seed would repeat the values of the earlier run.

        Parameters
        ----------
        trial_count : int
            Number of trials the study already holds.

        Returns
        -------
        int | None
            Seed offset by ``trial_count``, or None for a non-reproducible run.
        """
        return None if self.seed is None else (self.seed + trial_count) % self._SEED_RANGE
