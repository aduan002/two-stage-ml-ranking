from __future__ import annotations
from typing import Iterator, Dict, List
from pathlib import Path
import pandas as pd

from ..registry import LOADERS

@LOADERS.register("movielens_loader")
class MovieLensLoader:
    def __init__(self, folder_path:str, binarize_threshold:float|None = None, **kwargs):
        self.folder_path = Path(folder_path)
        self._binarize = binarize_threshold

        self._users:pd.DataFrame|None = None
        self._items:pd.DataFrame|None = None
        self._interactions:pd.DataFrame|None = None
        self._splits:Dict[str, pd.DataFrame]|None = None

        self._defaults = {
            "per_user_holdout":{
                "holdout_per_user": 5
            },
            "random":{
                "split_size": 0.8,
                "shuffle": True,
                "seed": 1337
            }
        }


    def _read(self, file_name:str, names:List[str]) -> pd.DataFrame:
        path = Path(self.folder_path, file_name)
        return pd.read_csv(path, sep="::", engine="python", header=None, names=names, encoding="latin-1")
    
    def load(self) -> None:
        users = self._read("users.dat", ["user_id", "gender", "age", "occupation", "zip_code"])
        movies = self._read("movies.dat", ["movie_id", "title", "genres"])
        ratings = self._read("ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])

        self._users = users
        self._items = movies
        self._interactions = ratings

        self._interactions["rating"] = self._interactions["rating"] >= self._binarize if self._binarize else self._interactions["rating"]
        self._interactions["timestamp"] = pd.to_datetime(self._interactions["timestamp"], unit="s")

    def iterate(self) -> Iterator[dict]:
        itx = self.interactions()
        for _, row in itx.iterrows():
            yield row.to_dict()

    def interactions(self) -> pd.DataFrame:
        if self._interactions is None:
            raise RuntimeError("Call load() first.")
        return self._interactions
    
    def user_features(self) -> pd.DataFrame:
        if self._users is None:
            raise RuntimeError("Call load() first.")
        return self._users

    def item_features(self) -> pd.DataFrame:
        if self._items is None:
            raise RuntimeError("Call load() first.")
        return self._items
    

    def _filter_feature_tables(self, itx_split:pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        uids = itx_split["user_id"].unique()
        iids = itx_split["movie_id"].unique()
        u_feats = self.user_features()[self.user_features()["user_id"].isin(uids)].reset_index(drop=True)
        i_feats = self.item_features()[self.item_features()["movie_id"].isin(iids)].reset_index(drop=True)
        return u_feats, i_feats
    
    def split(self, strategy:str, **kwargs) -> Dict[str, pd.DataFrame]:
        kwargs = {**self._defaults[strategy], **kwargs}

        if strategy == "per_user_holdout":
            # Sort by time per user and reserve the last N interactions per user as validation
            interactions = self.interactions().sort_values(["user_id", "timestamp"])

            def split_group(g):
                n = len(g)
                if n <= kwargs["holdout_per_user"]:
                    g = g.copy()
                    g["split"] = "train"
                    return g
                g = g.copy()
                g["split"] = "train"
                g.loc[g.index[-kwargs["holdout_per_user"]:], "split"] = "val"
                return g

            interactions = interactions.groupby("user_id", group_keys=False).apply(split_group)
            train_interactions = interactions[interactions["split"] == "train"].drop(columns=["split"])
            val_interactions = interactions[interactions["split"] == "val"].drop(columns=["split"])

        elif strategy == "random":
            interactions = self.interactions().sample(frac=1.0, random_state=kwargs["seed"]) if kwargs["shuffle"] else self.interactions()

            # Random global split
            n_train = int(kwargs["split_size"] * len(interactions))
            train_interactions = interactions.head(n_train)
            val_interactions   = interactions.tail(len(interactions) - n_train)

        else:
            raise ValueError(f"Unknown split_strategy: {strategy}")

        u_train, i_train = self._filter_feature_tables(train_interactions)
        u_valid, i_valid = self._filter_feature_tables(val_interactions)
        return {
            "train": {
                "interactions": train_interactions, 
                "user_features": u_train,
                "item_features": i_train,
                "labels": train_interactions["rating"]
            },
            "val": {
                "interactions": val_interactions, 
                "user_features": u_valid,
                "item_features": i_valid,
                "labels": val_interactions["rating"]
            }
        }


