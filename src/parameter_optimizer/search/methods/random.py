from dataclasses import dataclass

from optuna.samplers import BaseSampler, RandomSampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("random")
@dataclass(frozen=True, kw_only=True)
class RandomSearch(SearchMethod):
    """Choose each combination independently at random; a useful baseline."""

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
        return RandomSampler(seed=seed)
