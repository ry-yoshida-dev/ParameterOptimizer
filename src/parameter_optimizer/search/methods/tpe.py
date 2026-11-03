from dataclasses import dataclass

from optuna.samplers import BaseSampler, TPESampler

from ..method import SearchMethod, search_method_registry
from ..space import SearchSpace


@search_method_registry.register("tpe")
@dataclass(frozen=True, kw_only=True)
class TpeSearch(SearchMethod):
    """
    Bayesian optimization with a Tree-structured Parzen Estimator; a robust default for mixed parameters.

    Attributes
    ----------
    startup_trial_count : int
        Random trials before the estimator is used.
    candidate_count : int
        Candidates the estimator compares for each trial.
    is_multivariate : bool | None
        Whether parameters are modeled jointly; None follows Optuna.
    is_grouped : bool
        Whether parameters that appear together are modeled as a group; requires ``is_multivariate``.
    is_constant_liar : bool
        Whether running trials count as bad values, which keeps parallel workers apart.
    """

    startup_trial_count: int = 10
    candidate_count: int = 24
    is_multivariate: bool | None = None
    is_grouped: bool = False
    is_constant_liar: bool = True

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
        return TPESampler(
            n_startup_trials=self.startup_trial_count,
            n_ei_candidates=self.candidate_count,
            multivariate=self.is_multivariate,
            group=self.is_grouped,
            constant_liar=self.is_constant_liar,
            seed=seed,
        )
