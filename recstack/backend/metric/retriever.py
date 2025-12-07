import statistics as stats
from typing import Tuple
import numpy as np
import torch
from typing import Optional

from .metric import Metric
from ..registry import METRIC

@METRIC.register("retriever_metrics")
class Retrieval(Metric):
    def __init__(self, ks: Tuple[int, ...] = (1, 5, 10), **kwargs):
        self.ks = ks
        self.reset()

    def reset(self):
        self.recall_scores = {k: [] for k in self.ks}
        self.mrr_scores = []

    @torch.no_grad()
    def update(self,sims: torch.Tensor, pos_mask:Optional[torch.Tensor] = None, user_ids:Optional[torch.Tensor] = None):
        B = sims.size(0)
        device = sims.device

        # Fallback mask
        if pos_mask is None:
            if user_ids is not None:
                pos_mask = (user_ids.view(-1,1) == user_ids.view(1,-1))
            else:
                pos_mask = torch.eye(B, dtype=torch.bool, device=device)
        else:
            # ensure correct dtype and device
            pos_mask = pos_mask.to(device=device, dtype=torch.bool)

        # Top-K indices per row
        for k in self.ks:
            topk = sims.topk(k, dim=1).indices            # [B, k]
            hit_any = pos_mask.gather(1, topk).any(dim=1) # [B] bool
            recall_k = hit_any.float().mean().item()
            self.recall_scores[k].append(recall_k)

        # MRR: rank of first positive in the sorted order per row
        order = sims.argsort(dim=1, descending=True)      # [B, B]
        pos_in_order = pos_mask.gather(1, order)          # [B, B] bool
        # argmax returns the first index of the max (1) along dim=1; guaranteed ≥1 positive if diagonal is included
        first_pos_rank = pos_in_order.to(torch.int8).argmax(dim=1) + 1  # [B], 1-based
        mrr = (1.0 / first_pos_rank.float()).mean().item()
        self.mrr_scores.append(mrr)

    def mean(self):
        out = {f"recall@{k}": stats.mean(v) for k, v in self.recall_scores.items() if v}
        if self.mrr_scores:
            out["mrr"] = stats.mean(self.mrr_scores)
        return out

    def stdev(self):
        out = {f"recall@{k}": stats.pstdev(v) for k, v in self.recall_scores.items() if v}
        if self.mrr_scores:
            out["mrr"] = stats.pstdev(self.mrr_scores)
        return out

    @torch.no_grad()
    def __call__(self, sims: torch.Tensor, pos_mask:Optional[torch.Tensor] = None, user_ids:Optional[torch.Tensor] = None):
        B = sims.size(0)
        device = sims.device
        if pos_mask is None:
            if user_ids is not None:
                pos_mask = (user_ids.view(-1,1) == user_ids.view(1,-1))
            else:
                pos_mask = torch.eye(B, dtype=torch.bool, device=device)
        else:
            pos_mask = pos_mask.to(device=device, dtype=torch.bool)

        out = {}
        for k in self.ks:
            topk = sims.topk(k, dim=1).indices
            hit_any = pos_mask.gather(1, topk).any(dim=1).float().mean().item()
            out[f"recall@{k}"] = float(hit_any)

        order = sims.argsort(dim=1, descending=True)
        pos_in_order = pos_mask.gather(1, order)
        first_pos_rank = pos_in_order.to(torch.int8).argmax(dim=1) + 1
        out["mrr"] = float((1.0 / first_pos_rank.float()).mean().item())
        return out
