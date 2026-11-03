from dataclasses import dataclass
from typing import ClassVar

from optuna.samplers import BaseSampler, QMCSampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("quasi_monte_carlo")
@dataclass(frozen=True, kw_only=True)
class QuasiMonteCarloSearch(SearchMethod):
    """
    Cover the space evenly with a low-discrepancy sequence.

    Attributes
    ----------
    sequence : str
        Sequence that Optuna accepts as ``qmc_type``, such as ``sobol`` or ``halton``.
    is_scrambled : bool
        Whether the sequence is scrambled.
    """

    required_modules: ClassVar[tuple[str, ...]] = ("scipy",)
    extra: ClassVar[str | None] = "quasi-monte-carlo"

    sequence: str = "sobol"
    is_scrambled: bool = False

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
        return QMCSampler(qmc_type=self.sequence, scramble=self.is_scrambled, seed=seed)
