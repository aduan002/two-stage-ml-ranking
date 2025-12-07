from typing import List
import torch
import torch.nn.functional as F

from ..registry import ALGORITHM


_ACTIVATION_FUNCTIONS = {
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
}

@ALGORITHM.register("uv_reranker")
class UVDeep(torch.nn.Module):
    def __init__(self, embed_dim:int, architecture_config:dict, use_score:bool=True, normalize:bool=True):
        super(UVDeep, self).__init__()

        self.normalize = normalize
        self.use_score = use_score

        self.architecture = []
        in_dim = 4 * embed_dim + 2 + (1 if use_score else 0)
        for out_dim, activation_function in zip(architecture_config["out_features"], architecture_config["activation_functions"]):
            self.architecture.append(torch.nn.Linear(in_dim, out_dim))
            self.architecture.append(_ACTIVATION_FUNCTIONS[activation_function])

            in_dim = out_dim
        
        self.architecture.append(torch.nn.Linear(in_dim, 1))
        self.architecture = torch.nn.Sequential(*self.architecture)
        self.log_tau = torch.nn.Parameter(torch.zeros(())) # scale for cos/dot

    def l2_norm(self, x, dim=1, eps=1e-8):
        return F.normalize(x, p=2, dim=dim, eps=eps)
    
    def forward(self, u:torch.Tensor, v:torch.Tensor, retriever_scores:torch.Tensor=None):
        if self.normalize:
            u = self.l2_norm(u)
            v = self.l2_norm(v)

        uv = torch.cat([u, v, u * v, torch.abs(u - v)], dim=-1)

        tau = torch.exp(self.log_tau) + 1e-6
        dot = ((u*v).sum(-1, keepdim=True))
        denominator = (self.l2_norm(u) * self.l2_norm(v)).sum(-1, keepdim=True)
        cos = dot / denominator.clamp(1e-8)

        dot /= tau
        cos /= tau

        parts = [uv, cos, dot]
        if self.use_score and retriever_scores is not None:
            s = retriever_scores.view(-1,1)
            s = (s - s.mean(0, keepdim=True)) / (s.std(0, keepdim=True) + 1e-6)
            parts.append(s)

        x = torch.cat(parts, dim=-1)
        return self.architecture(x)



        








