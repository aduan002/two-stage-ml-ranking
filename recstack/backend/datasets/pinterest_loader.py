from __future__ import annotations
from typing import Iterator, Dict, List
from pathlib import Path
import pandas as pd
import json

from ..registry import LOADERS


# These urls failed...
BAD = {
    "a434737b4d17124b7a2e5658469c7209",
    "e20d854044b7909142cd54f3da082116",
    "09e363c112056e3a6e57993063e9c5d8",
    "6f5602b21e3c5d1a2434181a66040f50",
    "6fe931f3535d50ba30969497668dcb42",
    "ee221161e2efa1424bb20f283021d6f3",
}

# https://cseweb.ucsd.edu/~jmcauley/datasets.html#pinterest
@LOADERS.register("pinterest_loader")
class PinterestLoader:
    def __init__(self, folder_path:str, **kwargs):
        self.folder_path = Path(folder_path)

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

    def convert_to_url(self, signature:str) -> str:
        prefix = 'http://i.pinimg.com/400x/%s/%s/%s/%s.jpg'
        return prefix % (signature[0:2], signature[2:4], signature[4:6], signature)
    
    def add_url_column(self, df:pd.DataFrame, id_col:str, out_col:str = "image_url") -> pd.DataFrame:
        df = df.copy()
        df[out_col] = df[id_col].astype("string").apply(self.convert_to_url)
        return df


    def _read(self, file_name:str) -> dict:
        path = Path(self.folder_path, file_name)
        if path.suffix == ".json":
            with open(path, "r") as file:
                return json.load(file)
        elif path.suffix == ".jsonl":
            records = []
            with open(path, "r") as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))

            return records
        
        raise ValueError(f"Unsupported file extension: {path.suffix}")
    
    def load(self) -> None:
        items = self._read("items.jsonl")
        items_categories = self._read("items-cat.json")

        items_df = pd.DataFrame(items) 
        items_df = items_df.rename(columns={"scene": "scene_id", "product": "product_id"})
        # NOTE: This is only because the downloads failed...
        items_df = items_df[~items_df["product_id"].isin(BAD) & ~items_df["scene_id"].isin(BAD)]

        self._users = items_df[["scene_id"]]
        self._users = self.add_url_column(self._users, "scene_id", "scene_url")

        self._items = items_df[["product_id"]]
        self._items["category"] = items_df["product_id"].map(items_categories).fillna("unknown")
        self._items = self.add_url_column(self._items, "product_id", "product_url")

        self._interactions = items_df[["scene_id", "product_id", "bbox"]]

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
        uids = itx_split["scene_id"].unique()
        iids = itx_split["product_id"].unique()
        u_feats = self.user_features()[self.user_features()["scene_id"].isin(uids)].reset_index(drop=True)
        i_feats = self.item_features()[self.item_features()["product_id"].isin(iids)].reset_index(drop=True)
        return u_feats, i_feats
    
    def split(self, strategy:str, **kwargs) -> Dict[str, pd.DataFrame]:
        kwargs = {**self._defaults[strategy], **kwargs}

        if strategy == "per_user_holdout":
            # Sort by time per user and reserve the last N interactions per user as validation
            interactions = self.interactions().sort_values(["scene_id"])

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

            interactions = interactions.groupby("scene_id", group_keys=False).apply(split_group)
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
                "labels": pd.Series([1] * len(train_interactions))
            },
            "val": {
                "interactions": val_interactions, 
                "user_features": u_valid,
                "item_features": i_valid,
                "labels": pd.Series([1] * len(val_interactions))
            }
        }


