import argparse
import yaml
import os
import numpy as np
import pandas as pd
from functools import partial

import torch
from torch.utils.data import DataLoader
from safetensors.torch import load_file

from recstack.backend.preprocess.preprocess import GenericPreprocess
from recstack.backend.algorithm.collate import SideCollator
from recstack.backend.datasets.dataset import GenericPairDataset
from recstack.backend.factory import create_loader, create_preprocess, create_algorithm, create_retriever


def collate_items_only(batch, item_collator:SideCollator):
    item_bags_list = [s["item_bags"] for s in batch]
    item_images_list = [s.get("item_images", {}) for s in batch]
    item_ids_list = [s["item_id_index"] for s in batch]

    item_side = item_collator(item_bags_list, item_images_list, item_ids_list)

    raw_item_ids = [s["item_id"].item() if hasattr(s, "item") else s["item_id"] for s in batch]

    return raw_item_ids, item_side

def collate_fn(item_collator):
    return partial(collate_items_only, item_collator=item_collator)

def main(config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    folder_path = config["SAVE"]["folder_path"]
    
    data_cfg = config["DATA"]
    dataset_name = data_cfg["data_name"]
    dataset_cfg = data_cfg[data_cfg["data_name"]]
    dataloader = create_loader(dataset_name + "_loader", **dataset_cfg)
    dataloader.load()

    item_prep = GenericPreprocess.load(os.path.join(folder_path, dataset_name, "item_preprocess"))
    
    item_total_bins = item_prep.get_total_bins()
    size_id_item_feats = item_prep.get_size_id_features()
    num_bag_item_feats = item_prep.get_num_bag_features()
    num_image_item_feats = item_prep.get_num_image_features()

    algorithm_cfg = config["RETRIEVER"]
    item_algorithm_name = algorithm_cfg["item_algorithm_name"]

    algorithm_cfg[item_algorithm_name].update(
        {
            "size_id_features": size_id_item_feats,
            "num_bag_features": num_bag_item_feats,
            "num_image_features": num_image_item_feats,
            "total_bins": item_total_bins  
        }
    )
    item_model = create_algorithm(item_algorithm_name, **algorithm_cfg[item_algorithm_name])
    item_state_dict = load_file(os.path.join(folder_path, dataset_name, item_algorithm_name, "best.safetensors"))
    item_model.load_state_dict(item_state_dict)
    item_model.to(device)
    item_model.eval()

    retriever_cfg = config["RETRIEVAL"]
    retriever_name = retriever_cfg["retriever_name"]
    retriever = create_retriever(retriever_name, **retriever_cfg[retriever_name])

    item_features = dataloader.item_features()
    item_feats = item_prep.transform(item_features)

    # Fake interactions with dummy user 
    # Just to reuse GenericPairDataset as is for consistency across tower train, reranker train, index build, api deployment, etc.
    DUMMY_USER_ID = 0
    interactions = pd.DataFrame({
        "user_id": np.full(len(item_feats["id_to_row"]), DUMMY_USER_ID),
        item_prep.get_primary_key(): list(item_feats["id_to_row"].keys())
    })
    user_feats = {
        "id_to_row": {DUMMY_USER_ID: 0}, "bags": {}, "embeds": {},
        "ids": {"user_id": np.array([0], dtype=np.int64)}
    }

    dataset = GenericPairDataset(user_feats, item_feats, interactions, compound_primary_key=("user_id", item_prep.get_primary_key()),
        **data_cfg)
    
    item_field_order = item_prep.get_bags()
    item_embed_cols = item_prep.get_embeds()
    item_collator = SideCollator(item_field_order, item_embed_cols)


    loader = DataLoader(dataset, batch_size=config["TRAIN"]["batch_size"], 
        shuffle=False, 
        num_workers=config["TRAIN"]["num_workers"], 
        collate_fn=collate_fn(item_collator))

    with torch.inference_mode():
        for raw_item_ids, item_side in loader:
            item_side = item_side.to(device)

            item_embeds = item_model(item_side)
            retriever.add(raw_item_ids, item_embeds.cpu().numpy())

    retriever.build()
    retriever.save()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "2_retriever_build_index.py"
    )
    parser.add_argument("-c", "--config", required=True)

    args = vars(parser.parse_args())

    config_path = args["config"] 
    with open(config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(e)

    main(config)