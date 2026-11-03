from enum import Enum
from importlib.util import find_spec

from optuna.samplers import (
    BaseSampler,
    CmaEsSampler,
    GPSampler,
    GridSampler,
    NSGAIISampler,
    QMCSampler,
    RandomSampler,
    TPESampler,
)

from .space import SearchSpace

_GRID_SHUFFLE_SEED: int = 0


class SearchMethod(Enum):
    """
    Algorithm deciding which parameter combinations are tried.

    Attributes
    ----------
    GRID : str
        Try every combination of the candidates; requires only ``categorical`` parameters.
    RANDOM : str
        Choose each combination independently at random; a useful baseline.
    QUASI_MONTE_CARLO : str
        Cover the space evenly with a low-discrepancy sequence.
    TPE : str
        Bayesian optimization with a Tree-structured Parzen Estimator; a robust default for mixed parameters.
    GAUSSIAN_PROCESS : str
        Bayesian optimization with a Gaussian process; suits few trials of an expensive objective.
    CMA_ES : str
        Evolution strategy for many trials; rejects ``categorical`` parameters.
    GENETIC : str
        Genetic algorithm (NSGA-II) that also handles ``categorical`` parameters; needs many trials.
    """

    GRID = "grid"
    RANDOM = "random"
    QUASI_MONTE_CARLO = "quasi_monte_carlo"
    TPE = "tpe"
    GAUSSIAN_PROCESS = "gaussian_process"
    CMA_ES = "cma_es"
    GENETIC = "genetic"

    def validate(self, search_space: SearchSpace) -> None:
        """
        Check that this method can run and really explores a search space.

        Optuna would silently replace an unsupported parameter with random sampling, and would report a missing
        optional dependency only at the first trial.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.

        Raises
        ------
        ModuleNotFoundError
            If an optional dependency of this method is not installed.
        ValueError
            If this method does not support the parameters of ``search_space``.
        """
        missing_modules: list[str] = [module for module in self._required_modules() if find_spec(module) is None]
        if missing_modules:
            extra: str = self.value.replace("_", "-")
            raise ModuleNotFoundError(f"{self.value} requires {missing_modules}. Install the {extra} extra.")
        match self:
            case SearchMethod.GRID:
                if search_space.numeric_paths:
                    raise ValueError(f"grid requires categorical parameters. got {search_space.numeric_paths}")
            case SearchMethod.CMA_ES:
                categorical_paths: list[str] = list(search_space.categorical_candidates)
                if categorical_paths:
                    raise ValueError(f"cma_es does not support categorical parameters: {categorical_paths}")
            case (
                SearchMethod.RANDOM
                | SearchMethod.QUASI_MONTE_CARLO
                | SearchMethod.TPE
                | SearchMethod.GAUSSIAN_PROCESS
                | SearchMethod.GENETIC
            ):
                pass

    def build_sampler(self, search_space: SearchSpace, seed: int | None) -> BaseSampler:
        """
        Create the sampler of this method.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Random seed, or None for a non-reproducible run; ``GRID`` ignores it.

        Returns
        -------
        BaseSampler
            New sampler.
        """
        match self:
            case SearchMethod.GRID:
                # Stored trials refer to positions in the shuffled grid, so another seed would misread them on resume.
                return GridSampler(search_space.categorical_candidates, seed=_GRID_SHUFFLE_SEED)
            case SearchMethod.RANDOM:
                return RandomSampler(seed=seed)
            case SearchMethod.QUASI_MONTE_CARLO:
                return QMCSampler(seed=seed)
            case SearchMethod.TPE:
                return TPESampler(seed=seed)
            case SearchMethod.GAUSSIAN_PROCESS:
                return GPSampler(seed=seed)
            case SearchMethod.CMA_ES:
                return CmaEsSampler(seed=seed)
            case SearchMethod.GENETIC:
                return NSGAIISampler(seed=seed)

    def _required_modules(self) -> tuple[str, ...]:
        match self:
            case SearchMethod.QUASI_MONTE_CARLO:
                return ("scipy",)
            case SearchMethod.GAUSSIAN_PROCESS:
                return ("scipy", "torch")
            case SearchMethod.CMA_ES:
                return ("cmaes",)
            case SearchMethod.GRID | SearchMethod.RANDOM | SearchMethod.TPE | SearchMethod.GENETIC:
                return ()
