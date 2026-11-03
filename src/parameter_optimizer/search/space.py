import copy
from collections.abc import Iterator, Mapping
from typing import cast

from omegaconf import DictConfig, OmegaConf
from optuna import Trial
from optuna.distributions import BaseDistribution

from ..parameter import CategoricalSpec, ParameterDistribution, ParameterSpec, ParameterValue


class SearchSpace:
    """
    Parameters to explore, keyed by their dotted path in the configuration.

    Parameters
    ----------
    specs : Mapping[str, ParameterSpec]
        Search definition of each parameter path.

    Raises
    ------
    ValueError
        If ``specs`` is empty.
    """

    _DISTRIBUTION_KEY: str = "distribution"

    def __init__(self, specs: Mapping[str, ParameterSpec]) -> None:
        if not specs:
            raise ValueError("The search space must define at least one parameter.")
        self._specs: dict[str, ParameterSpec] = dict(specs)

    @classmethod
    def from_config(cls, search_config: DictConfig) -> "SearchSpace":
        """
        Read a search space written in the same hierarchy as the base configuration.

        Parameters
        ----------
        search_config : DictConfig
            Tree whose leaves are all definitions with a ``distribution`` key.

        Returns
        -------
        SearchSpace
            Search space with one parameter per definition.

        Raises
        ------
        ValueError
            If a leaf is not a valid definition.
        """
        tree: object = OmegaConf.to_container(search_config, resolve=True)
        specs: dict[str, ParameterSpec] = {}
        cls._collect_specs(tree, "", specs)
        return cls(specs)

    @property
    def categorical_candidates(self) -> dict[str, list[ParameterValue]]:
        """
        Candidates of the parameters chosen among listed values, which form the grid of a grid search.

        Returns
        -------
        dict[str, list[ParameterValue]]
            Candidates keyed by parameter path.
        """
        return {path: list(spec.candidates) for path, spec in self._specs.items() if isinstance(spec, CategoricalSpec)}

    @property
    def numeric_paths(self) -> list[str]:
        """
        Paths of the parameters chosen within a range.

        Returns
        -------
        list[str]
            Parameter paths using any distribution other than ``categorical``.
        """
        return [path for path, spec in self._specs.items() if not isinstance(spec, CategoricalSpec)]

    @property
    def optuna_distributions(self) -> dict[str, BaseDistribution]:
        """
        Distributions Optuna records for the trials of this space.

        Returns
        -------
        dict[str, BaseDistribution]
            Optuna distributions keyed by parameter path.
        """
        return {path: spec.to_optuna_distribution() for path, spec in self._specs.items()}

    def iter_probes(self) -> Iterator[dict[str, ParameterValue]]:
        """
        Yield assignments that together cover every probe value of every parameter.

        Parameters not under test stay at their first value, so constraints that depend on a specific
        combination of parameters are not covered.

        Yields
        ------
        dict[str, ParameterValue]
            Value of every parameter.
        """
        first_values: dict[str, ParameterValue] = {path: spec.probe_values[0] for path, spec in self._specs.items()}
        for path, spec in self._specs.items():
            for value in spec.probe_values:
                yield {**first_values, path: value}

    def suggest(self, trial: Trial) -> dict[str, ParameterValue]:
        """
        Ask a trial for a value of every parameter.

        Parameters
        ----------
        trial : Trial
            Trial choosing the values.

        Returns
        -------
        dict[str, ParameterValue]
            Chosen values keyed by parameter path.
        """
        return {path: spec.suggest(trial, path) for path, spec in self._specs.items()}

    def apply(self, base_config: DictConfig, values: Mapping[str, ParameterValue]) -> DictConfig:
        """
        Write values into a copy of a configuration.

        Paths absent from ``base_config`` are created, leaving unknown keys for the dataclass builder to reject.

        Parameters
        ----------
        base_config : DictConfig
            Configuration to start from; it is not modified.
        values : Mapping[str, ParameterValue]
            Value of every parameter.

        Returns
        -------
        DictConfig
            Plain configuration with no search definition left.

        Raises
        ------
        ValueError
            If ``values`` does not cover exactly the parameters of this space.
        """
        if set(values) != set(self._specs):
            raise ValueError(f"Expected values for {sorted(self._specs)}. got {sorted(values)}")
        resolved_config: DictConfig = copy.deepcopy(base_config)
        for path, value in values.items():
            OmegaConf.update(resolved_config, path, value, merge=False, force_add=True)
        return resolved_config

    @classmethod
    def _collect_specs(cls, node: object, path: str, specs: dict[str, ParameterSpec]) -> None:
        if not isinstance(node, dict) or not node:
            location: str = path or "<root>"
            raise ValueError(f"{location}: Expected a mapping of parameters to definitions. got {node}")
        mapping: dict[str, object] = {str(key): value for key, value in cast("dict[object, object]", node).items()}
        if isinstance(mapping.get(cls._DISTRIBUTION_KEY), str):
            specs[path] = cls._build_spec(mapping, path)
            return
        for key, child in mapping.items():
            cls._collect_specs(child, f"{path}.{key}" if path else key, specs)

    @classmethod
    def _build_spec(cls, definition: dict[str, object], path: str) -> ParameterSpec:
        fields: dict[str, object] = dict(definition)
        try:
            return ParameterDistribution(fields.pop(cls._DISTRIBUTION_KEY)).build_spec(fields)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}: {error}") from error
