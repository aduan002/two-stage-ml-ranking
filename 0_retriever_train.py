import argparse
import yaml
import os
import json
from functools import partial
import numpy as np
from torchvision import transforms

import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file

from recstack.backend.datasets.dataset import GenericPairDataset
from recstack.backend.preprocess.preprocess import GenericPreprocess
from recstack.backend.algorithm.collate import SideCollator, collate_retrieval
from recstack.backend.factory import create_loader, create_algorithm, create_metric

def to_cpu_state_dict(state_dict:dict[str, torch.Tensor]):
    # make sure tensors are saved from CPU for portability
    return {k: v.detach().cpu() for k, v in state_dict.items()}

def save_best_safetensors(model, path:str, epoch:int, metric_name:str, metric_value:float):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cpu_state_dict = to_cpu_state_dict(model.state_dict())

    metadata_path = os.path.splitext(path)[0] + ".json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "epoch": int(epoch),
            "metric_name": metric_name,
            "metric_value": float(metric_value),
        }, f, indent=2)
    save_file(cpu_state_dict, path)

def make_side_collator_from(preprocess):
    field_order = list(preprocess.get_bags())                   # bag columns (order from spec)
    embed_cols  = list(preprocess.get_embeds())                 # image/text columns
    return SideCollator(field_order, embed_cols)

def make_loaders(config:dict):
    data_cfg = config["DATA"]

    dataset_name = data_cfg["data_name"]
    dataset_cfg = data_cfg[dataset_name]
    data_loader = create_loader(dataset_name + "_loader", **dataset_cfg)
    data_loader.load()

    preprocess_cfg = config["PREPROCESS"]
    user_prep = GenericPreprocess(preprocess_cfg["user"])
    item_prep = GenericPreprocess(preprocess_cfg["item"])

    train_val_data = data_loader.split(dataset_cfg["split_strategy"], **dataset_cfg[dataset_cfg["split_strategy"]])
    
    user_train_feats = user_prep.fit_transform(train_val_data["train"]["user_features"])
    user_val_feats = user_prep.transform(train_val_data["val"]["user_features"])

    item_train_feats = item_prep.fit_transform(train_val_data["train"]["item_features"])
    item_val_feats = item_prep.transform(train_val_data["val"]["item_features"])

    interactions_train_feats = train_val_data["train"]["interactions"]
    interactions_val_feats = train_val_data["val"]["interactions"]

    user_prep.save(os.path.join(config["SAVE"]["folder_path"], data_cfg["data_name"], "user_preprocess"))
    item_prep.save(os.path.join(config["SAVE"]["folder_path"], data_cfg["data_name"], "item_preprocess"))

    compound_primary_key = (user_prep.get_primary_key(), item_prep.get_primary_key())
    train_dataset_params = {
        "user_feats": user_train_feats, "item_feats": item_train_feats, "interactions": interactions_train_feats, "compound_primary_key": compound_primary_key,
    }
    train_dataset_params.update(data_cfg)
    val_dataset_params = {
        "user_feats": user_val_feats, "item_feats": item_val_feats, "interactions": interactions_val_feats, "compound_primary_key": compound_primary_key,
    }
    train_dataset_params.update(data_cfg)
    
    train_dataset = GenericPairDataset(**train_dataset_params)
    val_dataset = GenericPairDataset(**val_dataset_params)

    user_collator = make_side_collator_from(user_prep)
    item_collator = make_side_collator_from(item_prep)
    collate_fn = partial(collate_retrieval, user_collator=user_collator, item_collator=item_collator)

    train_loader = DataLoader(train_dataset, batch_size=config["TRAIN"]["batch_size"], 
                              shuffle=True, 
                              num_workers=config["TRAIN"]["num_workers"], 
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config["TRAIN"]["batch_size"], 
                            shuffle=False, 
                            num_workers=config["TRAIN"]["num_workers"], 
                            collate_fn=collate_fn)

    return train_loader, val_loader, \
        (user_prep.get_total_bins(), item_prep.get_total_bins()), \
        (user_prep.get_size_id_features(), item_prep.get_size_id_features()), \
        (user_prep.get_num_bag_features(), item_prep.get_num_bag_features()), \
        (user_prep.get_num_image_features(), item_prep.get_num_image_features())


def train_epoch(user_model, item_model, loader, optimizer, metric, tau, device):
    user_model.train()
    item_model.train()

    loss_sum, total = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        user_batch, item_batch, user_ids, item_ids = batch.user, batch.item, batch.user_ids, batch.item_ids
        user_embedding = user_model(user_batch)
        item_embedding = item_model(item_batch)

        _, user_inv = np.unique(np.asarray(user_ids), return_inverse=True)
        user_labels = torch.from_numpy(user_inv).to(device)  # shape [B], dtype int64

        sims = (user_embedding @ item_embedding.T) / tau
        pos_mask = (user_labels.view(-1,1) == user_labels.view(1,-1))

        logp = sims.log_softmax(dim=1)
        # For each user, maximize the total softmax probability mass assigned to all items belonging to that same user, and ignore the rest.
        loss = -(torch.logsumexp(logp.masked_fill(~pos_mask, -float('inf')), dim=1)).mean()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metric.update(sims)

        loss_sum += loss.item() * sims.size(0)
        total += sims.size(0)

    return loss_sum / total

@torch.no_grad()
def evaluate(user_model, item_model, loader, metric, tau, device):
    user_model.eval()
    item_model.eval()

    loss_sum, total = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        user_batch, item_batch, user_ids, item_ids = batch.user, batch.item, batch.user_ids, batch.item_ids
        user_embedding = user_model(user_batch)
        item_embedding = item_model(item_batch)

        _, user_inv = np.unique(np.asarray(user_ids), return_inverse=True)
        user_labels = torch.from_numpy(user_inv).to(device)  # shape [B], dtype int64

        sims = (user_embedding @ item_embedding.T) / tau
        pos_mask = (user_labels.view(-1,1) == user_labels.view(1,-1))

        logp = sims.log_softmax(dim=1)
        # For each user, maximize the total softmax probability mass assigned to all items belonging to that same user, and ignore the rest.
        loss = -(torch.logsumexp(logp.masked_fill(~pos_mask, -float('inf')), dim=1)).mean()

        metric.update(sims, pos_mask=pos_mask)

        loss_sum += loss.item() * sims.size(0)
        total += sims.size(0)

    return loss_sum / total

def main(config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, (user_total_bins, item_total_bins), (size_id_user_feats, size_id_item_feats), (num_bag_user_feats, num_bag_item_feats), \
        (num_image_user_feats, num_image_item_feats)  = make_loaders(config)

    algorithm_cfg = config["RETRIEVER"]

    user_algorithm_name = algorithm_cfg["user_algorithm_name"]
    item_algorithm_name = algorithm_cfg["item_algorithm_name"]

    algorithm_cfg[user_algorithm_name].update(
        {
            "size_id_features": size_id_user_feats,
            "num_bag_features": num_bag_user_feats,
            "num_image_features": num_image_user_feats,
            "total_bins": user_total_bins            
        }
    )
    user_model = create_algorithm(user_algorithm_name, **algorithm_cfg[user_algorithm_name])
    user_model.to(device)

    algorithm_cfg[item_algorithm_name].update(
        {
            "size_id_features": size_id_item_feats,
            "num_bag_features": num_bag_item_feats,
            "num_image_features": num_image_item_feats,
            "total_bins": item_total_bins            
        }
    )
    item_model = create_algorithm(item_algorithm_name, **algorithm_cfg[item_algorithm_name])
    item_model.to(device)


    train_cfg = config["TRAIN"]

    user_params = list(user_model.parameters())
    item_params = list(item_model.parameters())
    optimizer = torch.optim.Adam(params=user_params + item_params, lr=train_cfg["learning_rate"])

    train_metric = create_metric(config["METRIC"]["metric_name"], **config["METRIC"])
    val_metric = create_metric(config["METRIC"]["metric_name"], **config["METRIC"])
    def format_metric(metric, prefix:str=""):
        means = metric.mean()
        stdevs = metric.stdev()
        parts = []
        for name in means:
            mean_val = means[name]
            std_val = stdevs[name]
            parts.append(f"{prefix} {name}: {mean_val:.4f} ± {std_val:.4f}")
        return " | ".join(parts)

    benchmark_metric_name = config["METRIC"]["benchmark_metric_name"]
    best_benchmark_metrics_mean = None
    best_val = float("-inf")
    for epoch in range(1, train_cfg["epochs"] + 1): 
        train_loss = train_epoch(user_model, item_model, train_loader, optimizer, train_metric, train_cfg["tau"], device)
        val_loss = evaluate(user_model, item_model, val_loader, val_metric, train_cfg["tau"], device)
        
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | " + 
            format_metric(train_metric, "train") + " | " + format_metric(val_metric, "val"))
        
        mean_metric_dict = val_metric.mean()
        current_val = mean_metric_dict[benchmark_metric_name]

        if best_val < current_val:
            best_benchmark_metrics_mean = mean_metric_dict
            best_val = current_val
            save_best_safetensors(
                user_model,
                path=os.path.join(config["SAVE"]["folder_path"], config["DATA"]["data_name"], config["RETRIEVER"]["user_algorithm_name"], "best.safetensors"),
                epoch=epoch,
                metric_name=benchmark_metric_name,
                metric_value=current_val,
            )
            save_best_safetensors(
                item_model,
                path=os.path.join(config["SAVE"]["folder_path"], config["DATA"]["data_name"], config["RETRIEVER"]["item_algorithm_name"], "best.safetensors"),
                epoch=epoch,
                metric_name=benchmark_metric_name,
                metric_value=current_val,
            )
        
        train_metric.reset()
        val_metric.reset()

    print(f"Best validation {benchmark_metric_name} mean: {best_benchmark_metrics_mean}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "0_retriever_train.py"
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