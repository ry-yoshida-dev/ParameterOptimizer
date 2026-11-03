from dataclasses import dataclass

from optuna.samplers import BaseSampler, NSGAIISampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("genetic")
@dataclass(frozen=True, kw_only=True)
class GeneticSearch(SearchMethod):
    """
    Genetic algorithm (NSGA-II) that also handles ``categorical`` parameters; needs many trials.

    Attributes
    ----------
    population_size : int
        Size of each generation.
    mutation_probability : float | None
        Probability of mutating a parameter; None follows Optuna.
    crossover_probability : float
        Probability of crossing two parents.
    swapping_probability : float
        Probability of swapping each parameter during a crossover.
    """

    population_size: int = 50
    mutation_probability: float | None = None
    crossover_probability: float = 0.9
    swapping_probability: float = 0.5

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
        return NSGAIISampler(
            population_size=self.population_size,
            mutation_prob=self.mutation_probability,
            crossover_prob=self.crossover_probability,
            swapping_prob=self.swapping_probability,
            seed=seed,
        )
