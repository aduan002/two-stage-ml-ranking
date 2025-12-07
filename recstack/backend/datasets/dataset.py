import io
import os
from dataclasses import dataclass
from typing import Optional, Callable, Any
from pathlib import Path
import requests
from torchvision import transforms

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

def get_image_preprocess(name:str):
        if name == "clip-ViT-B/32":
            import torchvision
            # CLIP-like preprocessing
            return torchvision.transforms.Compose([
                torchvision.transforms.Resize(224),
                torchvision.transforms.CenterCrop(224),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                                                 std=(0.26862954, 0.26130258, 0.27577711))
            ])
        elif name == "vit-b-32-imagenet":
            from torchvision.models import ViT_B_32_Weights
            w = ViT_B_32_Weights.IMAGENET1K_V1
            return w.transforms()
        else:
            raise ValueError(f"Unknown preprocess: {name}")

class ImageCache:
    def __init__(self, folder_path:str, timeout:float = 5.0):
        self.folder = Path(folder_path)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def _key(self, key:str) -> Path:
        return self.folder / f"{key}.jpg"

    def get_from_url_or_path(self, key:str, url:str) -> Image.Image:
        if url.startswith("http://") or url.startswith("https://"):
            path = self._key(key)
            if path.exists():
                with Image.open(path) as im:
                    return im.convert("RGB")

            r = requests.get(url, timeout=self.timeout)
            r.raise_for_status()
            img = Image.open(io.BytesIO(r.content)).convert("RGB")
            img.save(path, format="JPEG")
            return img

        # Otherwise, assume it's a local path for serving
        return self.get_from_path(url)

    def get_from_path(self, path_str:str) -> Image.Image:
        with Image.open(path_str) as im:
            return im.convert("RGB")

@dataclass
class GenericDatasetConfig:
    compound_primary_key: tuple[str, str] = ("user_id", "item_id")

    # Image pipeline
    user_image_cache_dir: Optional[str] = None
    item_image_cache_dir: Optional[str] = None
    user_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None
    item_image_transform: Optional[Callable[[Image.Image], torch.Tensor]] = None
    image_timeout: float = 5.0

class GenericPairDataset(Dataset):
    """
    Dataset-agnostic pair dataset for retrieval/rerank:
    - user_feats/item_feats are the dicts from UniversalPreprocess.transform()
    - interactions has columns [u_col, i_col]
    """
    def __init__(self, user_feats:dict, item_feats:dict, interactions:pd.DataFrame, compound_primary_key:tuple[str, str],  **kwargs):
        self.compound_primary_key = compound_primary_key
        self.user_feats = user_feats
        self.item_feats = item_feats

        cache_path = Path(kwargs.get("cache_dir", os.path.join(".cache", "_".join(compound_primary_key))))
        timeout = kwargs.get("timeout", 5.0)
        tf = get_image_preprocess(kwargs.get("image_preprocess", "vit-b-32-imagenet"))
        cfg = GenericDatasetConfig(compound_primary_key=compound_primary_key, 
            user_image_cache_dir=cache_path / "user", item_image_cache_dir=cache_path / "item",
            user_image_transform=tf, item_image_transform=tf,
            image_timeout=timeout)
        self.cfg = cfg

        u_col, i_col = cfg.compound_primary_key
        user_ids = interactions[u_col].to_numpy()
        item_ids = interactions[i_col].to_numpy()

        # Ensure id_to_row keys match lookup type
        u_map = {int(k) if isinstance(k, (int, np.integer)) else k: v for k, v in user_feats["id_to_row"].items() }
        i_map = {int(k) if isinstance(k, (int, np.integer)) else k: v for k, v in item_feats["id_to_row"].items() }

        # Vectorized keep mask
        u_keep = np.fromiter(((int(k) if isinstance(k, (int, np.integer)) else k) in u_map for k in user_ids), dtype=bool, count=len(user_ids))
        i_keep = np.fromiter(((int(k) if isinstance(k, (int, np.integer)) else k) in i_map for k in item_ids), dtype=bool, count=len(item_ids))
        keep = u_keep & i_keep
        self.user_ids = user_ids[keep]
        self.item_ids = item_ids[keep]

        self.u_map = u_map
        self.i_map = i_map

        # Image setup
        self.u_cache = ImageCache(cfg.user_image_cache_dir, cfg.image_timeout) if cfg.user_image_cache_dir else None
        self.i_cache = ImageCache(cfg.item_image_cache_dir, cfg.image_timeout) if cfg.item_image_cache_dir else None

        # Embed column lists
        self.user_embed_cols = list(user_feats.get("embeds", {}).keys())
        self.item_embed_cols = list(item_feats.get("embeds", {}).keys())

        # Transforms
        self.u_tf = cfg.user_image_transform
        self.i_tf = cfg.item_image_transform

        # Make it reusable by retriever and reranker
        self.labels = kwargs.get("labels", None)

    def __len__(self) -> int:
        return len(self.user_ids)

    def _slice_bag_row(self, bags_dict:dict[str, tuple[np.ndarray, np.ndarray]], row:int):
        out = {}
        for col, (idx, offsets) in bags_dict.items():
            start = int(offsets[row])
            end = int(offsets[row + 1])

            seg_idx = idx[start:end]
            seg_offsets = np.array([0, end - start], dtype=np.int64)  # include_last_offset=True
            out[col] = (seg_idx, seg_offsets)
        return out

    def _build_images(self, side:str, row:int, key_val:Any):
        imgs = {}
        if side == "user":
            if not self.user_embed_cols: return imgs

            embeds = self.user_feats["embeds"]
            tf, cache = self.u_tf, self.u_cache
            for col in self.user_embed_cols:
                url_or_path = embeds[col][row]
                if cache is not None:
                    img = cache.get_from_url_or_path(str(key_val), url_or_path)
                else:
                    continue

                if tf: 
                    img = tf(img)

                imgs[col] = img
        else:
            if not self.item_embed_cols: return imgs

            embeds = self.item_feats["embeds"]
            tf, cache = self.i_tf, self.i_cache
            for col in self.item_embed_cols:
                url_or_path = embeds[col][row]
                if cache is not None:
                    img = cache.get_from_url_or_path(str(key_val), url_or_path)
                else:
                    continue
                if tf: 
                    img = tf(img)

                imgs[col] = img
        return imgs

    def __getitem__(self, i: int):
        u_id = self.user_ids[i]
        i_id = self.item_ids[i]

        u_row = self.u_map[int(u_id) if isinstance(u_id, (int, np.integer)) else u_id]
        i_row = self.i_map[int(i_id) if isinstance(i_id, (int, np.integer)) else i_id]

        u_bags = self._slice_bag_row(self.user_feats["bags"], u_row)
        i_bags = self._slice_bag_row(self.item_feats["bags"], i_row)

        u_imgs = self._build_images("user", u_row, u_id)
        i_imgs = self._build_images("item", i_row, i_id)

        u_id_idx = int(self.user_feats["ids"][self.compound_primary_key[0]][u_row])
        i_id_idx = int(self.item_feats["ids"][self.compound_primary_key[1]][i_row])

        return {
            "user_id": u_id, "item_id": i_id,
            "user_bags": u_bags, "item_bags": i_bags,
            "user_images": u_imgs, "item_images": i_imgs,
            "user_id_index": u_id_idx, "item_id_index": i_id_idx,
            "labels": self.labels.iloc[i] if self.labels is not None else None
        }
