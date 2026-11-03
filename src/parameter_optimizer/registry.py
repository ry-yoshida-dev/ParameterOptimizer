from collections.abc import Callable, Mapping
from dataclasses import is_dataclass
from typing import cast

from dataclass_initializer import DataclassInitializer


class Registry[BaseT]:
    """
    Named dataclass implementations of an interface, so that a configuration can select any of them.

    Parameters
    ----------
    base_class : type[object]
        Interface every implementation must derive from.
    key : str
        Key holding the name of the implementation in a definition.
    """

    def __init__(self, *, base_class: type[object], key: str) -> None:
        self._base_class: type[object] = base_class
        self._key: str = key
        self._implementations: dict[str, type[BaseT]] = {}

    @property
    def key(self) -> str:
        """
        Key holding the name of the implementation in a definition.

        Returns
        -------
        str
            Name of the key.
        """
        return self._key

    def register[ImplementationT](self, name: str) -> Callable[[type[ImplementationT]], type[ImplementationT]]:
        """
        Create a class decorator that makes a dataclass selectable under a name.

        Parameters
        ----------
        name : str
            Name written in a configuration.

        Returns
        -------
        Callable[[type[ImplementationT]], type[ImplementationT]]
            Decorator returning the class unchanged. It raises ``TypeError`` if the class is not a dataclass deriving
            from the interface, and ``ValueError`` if the name is taken.
        """

        def decorator(implementation: type[ImplementationT]) -> type[ImplementationT]:
            if not is_dataclass(implementation) or not issubclass(implementation, self._base_class):
                raise TypeError(
                    f"{implementation.__name__} must be a dataclass deriving from {self._base_class.__name__}."
                )
            if name in self._implementations:
                raise ValueError(f"{self._key} {name} is already registered.")
            self._implementations[name] = cast("type[BaseT]", implementation)
            return implementation

        return decorator

    def build(self, definition: object) -> BaseT:
        """
        Create the implementation a definition selects.

        Parameters
        ----------
        definition : object
            Name of the implementation, or a mapping of its name under ``key`` and its fields.

        Returns
        -------
        BaseT
            New implementation.

        Raises
        ------
        TypeError
            If ``definition`` is neither a name nor a mapping, or a field has the wrong type.
        ValueError
            If the name is missing or unknown, or a field is missing, unknown or rejected by the implementation.
        """
        fields: dict[str, object]
        if isinstance(definition, str):
            fields = {self._key: definition}
        elif isinstance(definition, Mapping):
            fields = {str(key): value for key, value in cast("Mapping[object, object]", definition).items()}
        else:
            raise TypeError(f"Expected a name or a mapping for {self._key}. got {type(definition).__name__}")
        name: object = fields.pop(self._key, None)
        if not isinstance(name, str) or name not in self._implementations:
            raise ValueError(f"Unknown {self._key} {name}. Registered: {sorted(self._implementations)}")
        return DataclassInitializer.build(self._implementations[name], fields)
