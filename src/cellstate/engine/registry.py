"""Small registries that make unsupported model capabilities explicit."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Generic, TypeVar

from cellstate.errors import UnsupportedModalityError

ModelT = TypeVar("ModelT")


class ModelRegistry(Generic[ModelT]):
    def __init__(self, entries: Iterable[tuple[str, ModelT]] = ()) -> None:
        self._entries: dict[str, ModelT] = {}
        for key, model in entries:
            self.register(key, model)

    def register(self, key: str, model: ModelT) -> None:
        normalized = key.casefold()
        if normalized in self._entries:
            raise ValueError(f"a model is already registered for {key!r}")
        self._entries[normalized] = model

    def get(self, key: str) -> ModelT:
        try:
            return self._entries[key.casefold()]
        except KeyError as error:
            raise UnsupportedModalityError(f"no model is registered for {key!r}") from error

    def supports(self, key: str) -> bool:
        return key.casefold() in self._entries

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(sorted(self._entries))
