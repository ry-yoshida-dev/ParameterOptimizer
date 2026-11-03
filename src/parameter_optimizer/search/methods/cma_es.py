from dataclasses import dataclass
from typing import ClassVar

from optuna.samplers import BaseSampler, CmaEsSampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("cma_es")
@dataclass(frozen=True, kw_only=True)
class CmaEsSearch(SearchMethod):
    """
    Evolution strategy for many trials; rejects non-numeric parameters and several objectives.

    Attributes
    ----------
    startup_trial_count : int
        Random trials before the strategy is used.
    population_size : int | None
        Size of each generation; None follows Optuna.
    is_separable : bool
        Whether only the diagonal of the covariance is adapted, which scales to many parameters.
    """

    required_modules: ClassVar[tuple[str, ...]] = ("cmaes",)
    extra: ClassVar[str | None] = "cma-es"
    is_multi_objective_supported: ClassVar[bool] = False

    startup_trial_count: int = 1
    population_size: int | None = None
    is_separable: bool = False

    def validate(self, search_space: SearchSpace, objective_count: int) -> None:
        """
        Check that every parameter is numeric.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        objective_count : int
            Number of objectives being optimized.

        Raises
        ------
        ValueError
            If a parameter is not numeric or there are several objectives.
        """
        super().validate(search_space, objective_count)
        if search_space.non_numeric_paths:
            raise ValueError(f"cma_es does not support categorical parameters: {search_space.non_numeric_paths}")

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
        return CmaEsSampler(
            n_startup_trials=self.startup_trial_count,
            popsize=self.population_size,
            use_separable_cma=self.is_separable,
            seed=seed,
        )
