from __future__ import annotations
from typing import Protocol, runtime_checkable, Iterator, Dict, TypedDict
import pandas as pd

class SplitTables(TypedDict):
    interactions: pd.DataFrame
    user_features: pd.DataFrame
    item_features: pd.DataFrame

@runtime_checkable
class RecsysDataset(Protocol):
    """
    Behavioral contract for dataset adapters used in the recsys pipeline.
    Any class with this shape is accepted.
    """

    # lifecycle / IO
    def load(self) -> None: ...
    def iterate(self) -> Iterator[dict]: ...

    # tabular accessors
    def interactions(self) -> pd.DataFrame: ...
    def user_features(self) -> pd.DataFrame: ...
    def item_features(self) -> pd.DataFrame: ...

    # splitting
    def split(self, strategy:str, **kwargs) -> Dict[str, SplitTables]: ...
