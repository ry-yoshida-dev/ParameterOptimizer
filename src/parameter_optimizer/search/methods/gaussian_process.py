from dataclasses import dataclass
from typing import ClassVar

from optuna.samplers import BaseSampler, GPSampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("gaussian_process")
@dataclass(frozen=True, kw_only=True)
class GaussianProcessSearch(SearchMethod):
    """
    Bayesian optimization with a Gaussian process; suits few trials of an expensive objective.

    Attributes
    ----------
    startup_trial_count : int
        Random trials before the process is used.
    is_deterministic_objective : bool
        Whether the objective returns the same value for the same parameters.
    """

    required_modules: ClassVar[tuple[str, ...]] = ("scipy", "torch")
    extra: ClassVar[str | None] = "gaussian-process"

    startup_trial_count: int = 10
    is_deterministic_objective: bool = False

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
        return GPSampler(
            n_startup_trials=self.startup_trial_count,
            deterministic_objective=self.is_deterministic_objective,
            seed=seed,
        )
