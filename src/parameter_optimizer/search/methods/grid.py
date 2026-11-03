from dataclasses import dataclass
from typing import ClassVar

from optuna import Study

from ..method import SearchMethod, search_method_registry
from ..resumable_grid_sampler import ResumableGridSampler
from ..space import SearchSpace

_SHUFFLE_SEED: int = 0


@search_method_registry.register("grid")
@dataclass(frozen=True, kw_only=True)
class GridSearch(SearchMethod):
    """Try every combination; requires parameters that can be enumerated, and retries failed points on resume."""

    is_self_terminating: ClassVar[bool] = True

    def validate(self, search_space: SearchSpace, objective_count: int) -> None:
        """
        Check that every parameter can be enumerated.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        objective_count : int
            Number of objectives being optimized.

        Raises
        ------
        ValueError
            If a parameter is neither categorical nor has a step.
        """
        super().validate(search_space, objective_count)
        if search_space.non_enumerable_paths:
            raise ValueError(f"grid requires categorical parameters or a step. got {search_space.non_enumerable_paths}")

    def prepare(self, study: Study, search_space: SearchSpace, seed: int | None) -> bool:
        """
        Attach the sampler and queue the failed points of an earlier run.

        Parameters
        ----------
        study : Study
            Study about to run.
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Ignored, as the order of a grid is fixed.

        Returns
        -------
        bool
            Whether any point remains; a finished grid would otherwise run one more trial, because Optuna only stops
            after sampling.
        """
        sampler: ResumableGridSampler = self.build_sampler(search_space, seed)
        study.sampler = sampler
        sampler.enqueue_failed_points(study)
        return not sampler.is_finished(study)

    def build_sampler(self, search_space: SearchSpace, seed: int | None) -> ResumableGridSampler:
        """
        Create the sampler of this method.

        Parameters
        ----------
        search_space : SearchSpace
            Search space to explore.
        seed : int | None
            Ignored, as the order of a grid is fixed.

        Returns
        -------
        ResumableGridSampler
            New sampler.
        """
        # Stored trials refer to positions in the shuffled grid, so another seed would misread them on resume.
        return ResumableGridSampler(search_space.grid_candidates, seed=_SHUFFLE_SEED)
