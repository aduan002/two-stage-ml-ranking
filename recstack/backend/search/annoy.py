from annoy import AnnoyIndex
import numpy as np
from typing import Optional, Sequence, List, Tuple
from pathlib import Path
import json

from ..registry import RETRIEVER

@RETRIEVER.register("annoy_retriever")
class AnnoyRetrieval:
    def __init__(self, folder_path:str, distance_metric:str, embed_dim:int, n_trees:int, include_distances:bool=False, n_jobs:int=-1, **kwargs):
        folder_path = Path(folder_path)
        folder_path.mkdir(parents=True, exist_ok=True)
        self.index_path = folder_path / "annoy" / "annoy.ann"
        self.distance_metric = distance_metric

        self.embed_dim = embed_dim
        self.n_trees = n_trees
        self.include_distances = include_distances
        self.n_jobs = n_jobs

        self.k = kwargs.get('k', None)

        self.reset()

    def reset(self):
        self._size = 0
        self.idx_item_ids = {}
        self.item_ids_row_embeds = {}
        self.embeddings = []
        self.retriever = AnnoyIndex(f=self.embed_dim, metric=self.distance_metric)
        self._built = False


    def add(self, ids:Sequence[str], embeddings:np.ndarray) -> None:
        # embeddings shape (N, embed_dim)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.embed_dim:
            raise ValueError(f"Embeddings must have shape (N, {self.embed_dim})")

        if self._built:
            raise RuntimeError("Index is already built. Call reset(...) first")
        
        for i_id, v_embed in zip(ids, embeddings):
            self.retriever.add_item(self._size, v_embed)
            self.idx_item_ids[self._size] = i_id
            
            self.item_ids_row_embeds[i_id] = len(self.embeddings)
            self.embeddings.append(v_embed)
                        
            self._size += 1

    def build(self, embeddings:Optional[np.ndarray] = None) -> None:
        if embeddings is not None:
            self.add(embeddings)

        self.retriever.build(self.n_trees, n_jobs=self.n_jobs)
        self._built = True

    def search(self, query:np.ndarray, k:Optional[int]=None) -> Tuple[List, List]:
        if not self._built:
            raise RuntimeError("Index not built. Call build(...) or load(...) first")
        
        if query.shape != (self.embed_dim,):
            raise ValueError(f"Query must have shape ({self.embed_dim},)")
        
        k = self.k if k is None else k
        if k is None:
            raise ValueError("k must be provided either at init(...) or in search(...).")

        if self.include_distances:
            indices, distances = self.retriever.get_nns_by_vector(vector=query, n=k, search_k=-1, include_distances=True)
            return self.get_item_ids_by_indices(indices), distances

        indices = self.retriever.get_nns_by_vector(vector=query, n=k, search_k=-1, include_distances=False)
        return self.get_item_ids_by_indices(indices), []
    
    def get_embeddings(self, item_ids:list) -> np.array:
        rows = np.fromiter((self.item_ids_row_embeds[str(i)] for i in item_ids), dtype=np.int64)
        return self.embeddings[rows]


    def get_item_ids_by_indices(self, indices:Sequence[int]) -> np.ndarray:
        return [self.idx_item_ids[str(idx)] for idx in indices]

    def save(self) -> None: 
        item_ids_path = self.index_path.with_suffix(".idx_to_ids.json")
        row_embeddings_path = self.index_path.with_suffix(".ids_to_row.json")
        embeddings_path = self.index_path.with_suffix(".embeddings.npy")

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        ok = self.retriever.save(str(self.index_path))
        if not ok:
            raise IOError(f"Failed to save Annoy index to {self.index_path}")
        
        with open(item_ids_path, "w") as file:
            json.dump(self.idx_item_ids, file)
        with open(row_embeddings_path, "w") as file:
            json.dump(self.item_ids_row_embeds, file)

        embeddings = np.stack(self.embeddings, axis=0) # [N, D]
        np.save(embeddings_path, embeddings)


    def load(self) -> None:
        item_ids_path = self.index_path.with_suffix(".idx_to_ids.json")
        row_embeddings_path = self.index_path.with_suffix(".ids_to_row.json")
        embeddings_path = self.index_path.with_suffix(".embeddings.npy")

        ok = self.retriever.load(str(self.index_path))
        if not ok:
            raise IOError(f"Failed to load Annoy index from {self.index_path}")
        
        with open(item_ids_path, "r") as file:
            self.idx_item_ids = json.load(file)
        with open(row_embeddings_path, "r") as file:
            self.item_ids_row_embeds = json.load(file)
        self.embeddings = np.load(embeddings_path)

        self._size = self.retriever.get_n_items()
        self._built = True