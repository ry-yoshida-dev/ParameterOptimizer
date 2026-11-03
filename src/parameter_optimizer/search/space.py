import copy
from collections.abc import Iterator, Mapping
from itertools import islice, product
from typing import cast

from omegaconf import Container, DictConfig, OmegaConf, open_dict, read_write
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
    _MAX_PROBE_COMBINATIONS: int = 1024

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

    @property
    def first_values(self) -> dict[str, ParameterValue]:
        """
        First probe value of every parameter, which stays the same as long as the search space does.

        Returns
        -------
        dict[str, ParameterValue]
            Values keyed by parameter path.
        """
        return {path: spec.probe_values[0] for path, spec in self._specs.items()}

    def iter_probe_groups(self) -> Iterator[Iterator[dict[str, ParameterValue]]]:
        """
        Yield one group of assignments per probe value of every parameter.

        A probe value is usable if any assignment of its group is, because a constraint across parameters can reject
        it next to some values of the other parameters only. Each group holds a bounded number of combinations.

        Yields
        ------
        Iterator[dict[str, ParameterValue]]
            Assignments of every parameter that share the probe value, starting from ``first_values``.
        """
        for path, spec in self._specs.items():
            for value in spec.probe_values:
                yield self._iter_probe_group(path, value)

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
        Interpolations are resolved after writing, so they see the new values and the result stands on its own even if
        ``base_config`` is a part of a larger configuration.

        Parameters
        ----------
        base_config : DictConfig
            Configuration to start from; it is not modified.
        values : Mapping[str, ParameterValue]
            Value of every parameter.

        Returns
        -------
        DictConfig
            Plain configuration with no search definition or interpolation left.

        Raises
        ------
        ValueError
            If ``values`` does not cover exactly the parameters of this space.
        """
        if set(values) != set(self._specs):
            raise ValueError(f"Expected values for {sorted(self._specs)}. got {sorted(values)}")
        node: DictConfig = self._copy_within_tree(base_config)
        OmegaConf.set_readonly(node, False)
        for path, value in values.items():
            OmegaConf.update(node, path, value, merge=False, force_add=True)
        return cast(DictConfig, OmegaConf.create(OmegaConf.to_container(node, resolve=True)))

    def _iter_probe_group(self, path: str, value: ParameterValue) -> Iterator[dict[str, ParameterValue]]:
        other_paths: list[str] = [other_path for other_path in self._specs if other_path != path]
        other_values: list[list[ParameterValue]] = [self._specs[other].probe_values for other in other_paths]
        for combination in islice(product(*other_values), self._MAX_PROBE_COMBINATIONS):
            yield {path: value, **dict(zip(other_paths, combination, strict=True))}

    @staticmethod
    def _copy_within_tree(config: DictConfig) -> DictConfig:
        # An absolute interpolation in a copy of only ``config`` would still read the original tree instead of the new
        # values. The tree may hold something else at the position of ``config``, because ``OmegaConf.merge`` and
        # ``copy.deepcopy`` keep the parent, so each node is written into a copy of its parent rather than looked up.
        # OmegaConf has no public access to the parents of a node.
        keys: list[str | int] = []
        node: Container = config
        copied_node: object = copy.deepcopy(config)
        while (parent := node._get_parent_container()) is not None:  # pyright: ignore[reportPrivateUsage]
            key: str | int = cast("str | int", node._key())  # pyright: ignore[reportPrivateUsage]
            copied_parent: Container = copy.deepcopy(parent)
            with read_write(copied_parent), open_dict(copied_parent):
                cast("dict[str | int, object]", copied_parent)[key] = copied_node
            keys.append(key)
            node = parent
            copied_node = copied_parent
        for key in reversed(keys):
            copied_node = cast("dict[str | int, object]", copied_node)[key]
        return cast(DictConfig, copied_node)

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
