from typing import Callable, Dict, Generic, TypeVar

T = TypeVar("T")

class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, T] = {}

    def register(self, name: str, *, overwrite: bool = False) -> Callable[[T], T]:
        def deco(obj: T) -> T:
            if not overwrite and name in self._items:
                raise KeyError(f"{self.kind} '{name}' already registered")
            self._items[name] = obj
            return obj
        return deco

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ValueError(f"Unknown {self.kind} '{name}'. Known: {sorted(self._items)}")

    def known(self):
        return tuple(sorted(self._items))

# One registry per component family
LOADERS = Registry("loader")
PREPROCESS = Registry("preprocess")
ALGORITHM = Registry("algorithm")
METRIC = Registry("metric")
RETRIEVER = Registry("retriever")
