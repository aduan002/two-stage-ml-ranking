from typing import List
import torch
import torch.nn.functional as F
import torchvision
from torchvision.models import ViT_B_32_Weights

from .collate import SideBatch
from ..registry import ALGORITHM


_ACTIVATION_FUNCTIONS = {
    "relu": torch.nn.ReLU(),
    "tanh": torch.nn.Tanh(),
    "sigmoid": torch.nn.Sigmoid(),
}


class Bag(torch.nn.Module):
    def __init__(self, total_bins:int, embed_dim:int, mode:str='mean', bias:bool = True):
        super(Bag, self).__init__()
        self.bag = torch.nn.EmbeddingBag(total_bins, embed_dim, mode=mode, include_last_offset=True)

    def forward(self, idx, offsets):
        x = self.bag(idx, offsets)  # [batch, embed_dim]
        return x
    
class Tower(torch.nn.Module):
    def __init__(self, size_id_features:int, num_bag_features:int, num_image_features:int, total_bins:int, ids_embed_dim:int, bag_embed_dim:int, image_embed_dim:int, 
            architecture_config:dict, dropout_p:float=0.1, normalize:bool=True,**kwargs):
        super(Tower, self).__init__()
        self.normalize = normalize

        self.bag_model = None
        self.img_model = None
        self.id_model = None

        img_output_dim = 0
        if total_bins > 0:
            self.bag_model = Bag(total_bins=total_bins, embed_dim=bag_embed_dim)
        if num_image_features > 0:
            weights = ViT_B_32_Weights.IMAGENET1K_V1
            self.img_model = torchvision.models.vit_b_32(weights=weights)
            self.img_model.heads = torch.nn.Identity() # Want 768-dim image features not 1000-dim class boundaries

            img_output_dim = self.img_model.hidden_dim
            for p in self.img_model.parameters():
                p.requires_grad = False

            self.img_proj = torch.nn.Linear(img_output_dim, image_embed_dim) if num_image_features else None
            self.img_model.eval()
        if size_id_features > 0:
            self.id_model = torch.nn.Embedding(size_id_features, ids_embed_dim)


        in_dim = ids_embed_dim + bag_embed_dim * num_bag_features + num_image_features * image_embed_dim
        self.architecture = [torch.nn.LayerNorm(in_dim), torch.nn.Dropout(dropout_p)]
        for out_dim, activation_function in zip(architecture_config["out_features"], architecture_config["activation_functions"]):
            self.architecture.append(torch.nn.Linear(in_dim, out_dim))
            self.architecture.append(_ACTIVATION_FUNCTIONS[activation_function])

            in_dim = out_dim
        self.architecture = torch.nn.Sequential(*self.architecture)

        self.out_dim = out_dim

    def l2_norm(self, x, dim=1, eps=1e-8):
        return F.normalize(x, p=2, dim=dim, eps=eps)
    
    def get_output_dim(self):
        return self.out_dim

    def forward(self, side:SideBatch):
        bags_embeds = None
        imgs_embeds = None
        id_embeds = None

        if self.id_model:
            id_embeds = self.id_model(side.ids)

        if self.bag_model:
            bag_vecs = [self.bag_model(*side.bags[col]) for col in side.bag_order if col in side.bags]
            bags_embeds = torch.cat(bag_vecs, dim=-1) if len(bag_vecs) > 1 else bag_vecs[0] # [B, bag_embed_dim * num_bag_features]

        if self.img_model:
            img_vecs = []
            with torch.set_grad_enabled(self.training and any(p.requires_grad for p in self.img_model.parameters())):
                for col in side.image_order:
                    imgs = side.images.get(col)
                    if imgs is None: 
                        continue

                    v = self.img_model(imgs) # [B, head_features]
                    v = self.img_proj(v) # [B, image_embed_dim]
                    img_vecs.append(v)

            imgs_embeds = torch.cat(img_vecs, dim=-1) if len(img_vecs) > 1 else img_vecs[0] # [B, head_features * num_image_features]

        vecs = [v for v in [id_embeds, bags_embeds, imgs_embeds] if v is not None]
        if not vecs:
            raise ValueError("All id, bag, and image inputs are empty.")
        
        # [B, id_embed_dim + bag_embed_dim * num_bag_features + image_embed_dim * num_image_features]
        x = torch.cat(vecs, dim=-1)
        x = self.architecture(x)
        return self.l2_norm(x) if self.normalize else x

@ALGORITHM.register("user_tower")
class UserTower(Tower):
    def __init__(self, size_id_features:int, num_bag_features:int, num_image_features:int, total_bins:int, ids_embed_dim:int, bag_embed_dim:int, image_embed_dim:int, architecture_config:dict, dropout_p:float=0.1, normalize:bool = True, **kwargs):
        super().__init__(size_id_features, num_bag_features, num_image_features, total_bins, ids_embed_dim, bag_embed_dim, image_embed_dim, architecture_config, dropout_p, normalize, **kwargs)
@ALGORITHM.register("item_tower")
class ItemTower(Tower):
    def __init__(self, size_id_features:int, num_bag_features:int, num_image_features:int, total_bins:int, ids_embed_dim:int, bag_embed_dim:int, image_embed_dim:int, architecture_config:dict, dropout_p:float=0.1, normalize:bool = True, **kwargs):
        super().__init__(size_id_features, num_bag_features, num_image_features, total_bins, ids_embed_dim, bag_embed_dim, image_embed_dim, architecture_config, dropout_p, normalize, **kwargs)




        








