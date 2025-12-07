from collections import defaultdict
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

def _stack_bags(field_order:list[str], batch_bags:list[dict[str, tuple[np.ndarray, np.ndarray]]]):
    idx_bag = defaultdict(list)
    len_bag = defaultdict(list)

    for bags in batch_bags:
        for col in field_order:
            if col in bags:
                idx, offsets = bags[col]
            else:
                idx = np.empty(0, dtype=np.int64)
                offsets = np.array([0], dtype=np.int64)

            t_idx = torch.from_numpy(idx) if isinstance(idx, np.ndarray) else torch.tensor(idx, dtype=torch.long)
            idx_bag[col].append(t_idx)
            len_bag[col].append(torch.tensor([int(offsets[-1])], dtype=torch.long))

    out = {}
    for col in field_order:
        if len(idx_bag[col]) == 0: # never appeared in the batch
            out[col] = (torch.empty(0, dtype=torch.long), torch.zeros(1, dtype=torch.long))
            continue

        idx_cat = torch.cat(idx_bag[col], dim=0) 
        lengths = torch.cat(len_bag[col], dim=0)
        offsets = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(lengths, dim=0)], dim=0)
        out[col] = (idx_cat, offsets)
    return out  # dict[col]: (idx, offsets)


def _stack_images(embed_cols:list[str], batch_images: Optional[list[dict[str, torch.Tensor]]]):
    if not embed_cols:
        return {}
    if not batch_images:
        return {c: None for c in embed_cols}

    per_col = defaultdict(list)
    for record in batch_images:
        for c in embed_cols:
            img = record.get(c, None)
            if img is not None:
                per_col[c].append(img)

    out = {}
    for c in embed_cols:
        imgs = per_col.get(c, [])
        out[c] = torch.stack(imgs, dim=0) if len(imgs) > 0 else None # [B, C, H, W]
    return out  # dict[col]: torch.Tensor or None

def _stack_ids(batch_id_indices:list[Optional[int]]):
    if not batch_id_indices or batch_id_indices[0] is None:
        return None
    return torch.as_tensor(batch_id_indices, dtype=torch.long)  # [B]

@dataclass
class SideBatch:
    bags: dict[str, tuple[torch.Tensor, torch.Tensor]]  # col: (idx, offsets)
    images: dict[str, Optional[torch.Tensor]]           # col: [B,C,H,W] or None
    bag_order: list[str]
    image_order: list[Optional[str]]
    ids: Optional[torch.Tensor] = None                  # [B] or None


    def to(self, device: torch.device) -> "SideBatch":
        bags = {k: (v[0].to(device), v[1].to(device)) for k, v in self.bags.items()}
        imgs = {k: (v.to(device) if v is not None else None) for k, v in self.images.items()}
        ids  = self.ids.to(device) if self.ids is not None else None
        return SideBatch(bags=bags, images=imgs, bag_order=self.bag_order, image_order=self.image_order, ids=ids)
    
@dataclass
class RetrievalBatch:
    user: SideBatch
    item: SideBatch
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    def to(self, device):
        return RetrievalBatch(
            user=self.user.to(device),
            item=self.item.to(device),
            user_ids=self.user_ids,
            item_ids=self.item_ids,
        )

@dataclass
class RerankBatch:
    user: SideBatch
    item: SideBatch
    user_ids: torch.Tensor
    item_ids: torch.Tensor
    labels: torch.Tensor
    def to(self, device):
        return RerankBatch(
            user=self.user.to(device),
            item=self.item.to(device),
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            labels=self.labels.to(device)
        )
    
class SideCollator:
    def __init__(self, field_order:list[str], embed_cols:list[str]):
        self.field_order = field_order
        self.embed_cols = embed_cols

    def __call__(self, batch_bags:list[dict], batch_images:Optional[list[dict]] = None, batch_ids:Optional[list[int]] = None) -> SideBatch:
        bags = _stack_bags(self.field_order, batch_bags)
        images = _stack_images(self.embed_cols, batch_images)
        ids = _stack_ids(batch_ids)
        return SideBatch(bags=bags, images=images, bag_order=self.field_order, image_order=self.embed_cols, ids=ids)
    
def collate_retrieval(batch, user_collator:SideCollator, item_collator:SideCollator):
    user_bags_list = [s["user_bags"] for s in batch]
    item_bags_list = [s["item_bags"] for s in batch]

    user_images_list = [s.get("user_images", {}) for s in batch]
    item_images_list = [s.get("item_images", {}) for s in batch]

    user_ids_list = [s["user_id_index"] for s in batch]
    item_ids_list = [s["item_id_index"] for s in batch]

    user_side = user_collator(user_bags_list, user_images_list, user_ids_list)
    item_side = item_collator(item_bags_list, item_images_list, item_ids_list)

    user_ids = [s["user_id"] for s in batch]
    item_ids = [s["item_id"] for s in batch]

    return RetrievalBatch(user_side, item_side, user_ids, item_ids)


def collate_rerank(batch, user_collator:SideCollator, item_collator:SideCollator):
    user_bags_list = [s["user_bags"] for s in batch]
    item_bags_list = [s["item_bags"] for s in batch]

    user_images_list = [s.get("user_images", {}) for s in batch]
    item_images_list = [s.get("item_images", {}) for s in batch]

    user_ids_list = [s["user_id_index"] for s in batch]
    item_ids_list = [s["item_id_index"] for s in batch]

    user_side = user_collator(user_bags_list, user_images_list, user_ids_list)
    item_side = item_collator(item_bags_list, item_images_list, item_ids_list)

    user_ids = [s["user_id"] for s in batch]
    item_ids = [s["item_id"] for s in batch]

    label = torch.as_tensor([s["labels"] for s in batch], dtype=torch.float32)

    return RerankBatch(user_side, item_side, user_ids, item_ids, label)


