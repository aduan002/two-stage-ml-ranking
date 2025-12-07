import pandas as pd
import xxhash
import numpy as np
from pathlib import Path
import json

class IndexNamespaces:
    """Manage disjoint index ranges inside the wide EmbeddingBag table."""
    def __init__(self):
        self.offsets = {}
        self.sizes = {}
        self.total_bins = 0

    def add_vocab(self, name:str, size:int):
        offset = self.total_bins
        self.offsets[name] = offset
        self.sizes[name] = size
        self.total_bins += size
        return offset

    def add_hashed(self, name:str, num_bins:int):
        offset = self.total_bins
        self.offsets[name] = offset
        self.sizes[name] = num_bins
        self.total_bins += num_bins
        return offset

class GenericPreprocess:
    def __init__(self, specs:dict):
        self.ns = IndexNamespaces()
        self.sep = {b['col']: b.get('sep','|') for b in specs.get('bags',[])}
        self.bags = [b['col'] for b in specs.get('bags',[])]
        self.embeds = specs.get('embeds', [])
        self.primary_key = specs['primary_key']
        self.namespace_name = specs['namespace_name']
        self.seed = specs.get('hash_seed', 1337)
        self.bucket_factor = {b['col']: b.get('bucket_factor', 32) for b in specs.get('bags',[])}
        self.bucket_sizes = {}

        self._fitted = False
        self.specs = specs

    def _row_tokens(self, val:list[str]|str, col:str, dedup:bool=True):
        if isinstance(val, list):
            toks = [str(t) for t in val]
        else:
            s = "" if val is None or (isinstance(val, float) and np.isnan(val)) else str(val)
            toks = s.split(self.sep[col])

        toks = [t.strip() for t in toks if t and t.strip() != ""]
        if dedup:
            # preserve order while de-duping
            seen = set()
            toks = [x for x in toks if not (x in seen or seen.add(x))]
        return [t.casefold() for t in toks]


    def fit(self, X:pd.DataFrame, y=None):
        for col in self.bags:
            s = pd.Series(X[col], copy=False)
            tokens = s.map(lambda v: self._row_tokens(v, col))
            flat = pd.Series([t for lst in tokens for t in lst], dtype="string")
            uniq = int(flat.nunique()) if len(flat) else 0
            est = max(1, uniq * self.bucket_factor[col])
            num_bins = 1 << (est - 1).bit_length()
            self.bucket_sizes[col] = num_bins

            if col == self.primary_key:
                continue
            self.ns.add_hashed(f"{self.namespace_name}.{col}", num_bins)

        self._fitted = True
        return self

    def _hash(self, s:str):
        return xxhash.xxh64(s.encode(), seed=self.seed).intdigest()

    def _bags_to_indices(self, col:str, X:pd.DataFrame):
        base = self.ns.offsets[f"{self.namespace_name}.{col}"]
        bins = self.bucket_sizes[col]

        tokens_list = X[col].map(lambda v: self._row_tokens(v, col))
        all_toks = [t for lst in tokens_list for t in lst]
        hashed = np.fromiter(((self._hash(t) % bins) for t in all_toks), dtype=np.int64)

        idx = base + hashed
        lengths = np.fromiter((len(lst) for lst in tokens_list), dtype=np.int64, count=len(tokens_list))

        offsets = np.zeros(len(lengths) + 1, dtype=np.int64)
        np.cumsum(lengths, out=offsets[1:])

        return idx, offsets

    def transform(self, X:pd.DataFrame):
        assert self._fitted

        bags_out = {c: self._bags_to_indices(c, X) for c in self.bags if c != self.primary_key}
        embeds_out = {}  # images
        for e in self.embeds:
            col, kind = e['col'], e.get('kind', 'image')
            # pass through URLs/paths
            embeds_out[col] = X[col].tolist() # kind is not really used right now, but it could be.

        entity_ids = X[self.primary_key].to_numpy()
        id_to_row = {int(k) if isinstance(k, (int, np.integer)) else k: i for i, k in enumerate(entity_ids)}

        primary_ids = np.fromiter(
            (xxhash.xxh64((str(int(v)) if isinstance(v, (int, np.integer)) else str(v)).encode(),
                        seed=self.seed).intdigest() % self.bucket_factor[self.primary_key]
            for v in pd.Series(X[self.primary_key], copy=False)),
            dtype=np.int64, count=len(X)
        )

        return {
            "namespace": self.namespace_name,
            "ids": {self.primary_key: primary_ids},                 # [ids]
            "bags": bags_out,                           # {col: (idx, offsets)}
            "embeds": embeds_out,                       # {col: [url|path|text]}
            "entity_ids": entity_ids,
            "id_to_row": id_to_row,
            "total_bins": self.ns.total_bins,
        }
    
    def fit_transform(self, X:pd.DataFrame, y=None):
        return self.fit(X, y).transform(X)

    def namespaces_to_dict(self):
        return {
            "offsets": {k:int(v) for k, v in self.ns.offsets.items()},
            "sizes": {k:int(v) for k, v in self.ns.sizes.items()},
            "total_bins": int(self.ns.total_bins),
        }
    def namespaces_from_dict(self, d:dict):
        ns = IndexNamespaces()
        ns.offsets = {k:int(v) for k, v in d["offsets"].items()}
        ns.sizes   = {k:int(v) for k, v in d["sizes"].items()}
        ns.total_bins = int(d["total_bins"])
        return ns
    
    def to_dict(self):
        return {
            "__version__": "1.0.0",
            "params": {
                "specs": self.specs
            },
            "namespaces": self.namespaces_to_dict(),
            "bucket_sizes": self.bucket_sizes
        }
    
    @classmethod
    def from_dict(cls, data:dict):
        params = data["params"]
        obj = cls(
           specs=params["specs"]
        )

        # namespaces
        obj.ns = obj.namespaces_from_dict(data["namespaces"])
        obj.bucket_sizes = data["bucket_sizes"]

        obj._fitted = True
        return obj
    
    def save(self, folder:str, filename:str = "preprocess.json"):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / filename
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, folder:str, filename: str = "preprocess.json"):
        path = Path(folder) / filename
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
    

    def get_bags(self):
        return [bag for bag in self.bags if bag != self.primary_key]
    def get_embeds(self):
        return [embed_cols["col"] for embed_cols in self.embeds if embed_cols["col"] != self.primary_key]
    
    def get_size_id_features(self):
        return self.bucket_sizes[self.primary_key]
    
    def get_num_bag_features(self):
        return len(self.get_bags())
    def get_num_image_features(self):
        return len(self.get_embeds())
    
    def get_total_bins(self):
        return self.ns.total_bins
    def get_primary_key(self):
        return self.primary_key
