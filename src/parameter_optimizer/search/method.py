from abc import ABC, abstractmethod
from importlib.util import find_spec
from typing import ClassVar

from optuna import Study
from optuna.samplers import BaseSampler

from ..registry import Registry
from .space import SearchSpace


class SearchMethod(ABC):
    """
    Algorithm deciding which parameter combinations are tried.

    Decorating a dataclass implementation with ``search_method_registry.register`` makes it selectable from a
    configuration.

    Attributes
    ----------
    required_modules : ClassVar[tuple[str, ...]]
        Optional modules this method needs.
    extra : ClassVar[str | None]
        Extra of this package that installs ``required_modules``.
    is_self_terminating : ClassVar[bool]
        Whether this method ends by itself, so that a run needs neither ``n_trials`` nor ``timeout_seconds``.
    is_multi_objective_supported : ClassVar[bool]
        Whether this method handles several objectives.
    """

    required_modules: ClassVar[tuple[str, ...]] = ()
    extra: ClassVar[str | None] = None
    is_self_terminating: ClassVar[bool] = False
    is_multi_objective_supported: ClassVar[bool] = True

    def validate(self, search_space: SearchSpace, objective_count: int) -> None:
        """
        Check that this method can run and really explores a search space.

        Optuna would silently replace an unsupported parameter with random sampling, and would report a missing
        optional dependency only at the first trial.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        objective_count : int
            Number of objectives being optimized.

        Raises
        ------
        ModuleNotFoundError
            If an optional dependency of this method is not installed.
        ValueError
            If this method does not support the parameters of ``search_space`` or ``objective_count``.
        """
        missing_modules: list[str] = [module for module in self.required_modules if find_spec(module) is None]
        if missing_modules:
            hint: str = "Install them." if self.extra is None else f"Install the {self.extra} extra."
            raise ModuleNotFoundError(f"{type(self).__name__} requires {missing_modules}. {hint}")
        if objective_count > 1 and not self.is_multi_objective_supported:
            raise ValueError(f"{type(self).__name__} does not support several objectives.")

    def prepare(self, study: Study, search_space: SearchSpace, seed: int | None) -> bool:
        """
        Attach the sampler of this method to a study that may hold trials of an earlier run.

        Parameters
        ----------
        study : Study
            Study about to run.
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Random seed, or None for a non-reproducible run.

        Returns
        -------
        bool
            Whether any trial remains to be run.
        """
        study.sampler = self.build_sampler(search_space, seed)
        return True

    @abstractmethod
    def build_sampler(self, search_space: SearchSpace, seed: int | None) -> BaseSampler:
        """
        Create the sampler of this method.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Random seed, or None for a non-reproducible run.

        Returns
        -------
        BaseSampler
            New sampler.
        """


search_method_registry: Registry[SearchMethod] = Registry(base_class=SearchMethod, key="name")
