import importlib, pkgutil
from typing import Any
from .registry import LOADERS, PREPROCESS, ALGORITHM, METRIC, RETRIEVER

PKG = {
    "loader": "recstack.backend.datasets",
    "preprocess": "recstack.backend.preprocess",
    "algorithm": "recstack.backend.algorithm",
    "metric": "recstack.backend.metric",
    "retriever": "recstack.backend.search"
}

_loaded_families:set[str] = set()
def _load_family(family:str) -> None:
    """Import all modules in the family package once, so they self-register."""
    if family in _loaded_families:
        return
    pkg_name = PKG[family]
    pkg = importlib.import_module(pkg_name)
    if hasattr(pkg, "__path__"):  # it's a package (not a plain module)
        for m in pkgutil.walk_packages(pkg.__path__, prefix=pkg_name + "."):
            importlib.import_module(m.name)
    _loaded_families.add(family)

def create_loader(name:str, **opts:Any):
    _load_family("loader")
    return LOADERS.get(name)(**opts)
def create_preprocess(name:str, **opts:Any):
    _load_family("preprocess")
    return PREPROCESS.get(name)(**opts)
def create_algorithm(name:str, **opts:Any):
    _load_family("algorithm")
    return ALGORITHM.get(name)(**opts)
def create_metric(name:str, **opts:Any):
    _load_family("metric")
    return METRIC.get(name)(**opts)
def create_retriever(name:str, **opts:Any):
    _load_family("retriever")
    return RETRIEVER.get(name)(**opts)