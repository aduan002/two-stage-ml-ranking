import argparse
import yaml
import os
import json
from functools import partial


import torch
from torch.utils.data import DataLoader
from safetensors.torch import save_file, load_file

from recstack.backend.datasets.dataset import GenericPairDataset
from recstack.backend.preprocess.preprocess import GenericPreprocess
from recstack.backend.algorithm.collate import SideCollator, collate_rerank
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
        "labels": train_val_data["train"]["labels"]
    }
    train_dataset_params.update(data_cfg)
    val_dataset_params = {
        "user_feats": user_val_feats, "item_feats": item_val_feats, "interactions": interactions_val_feats, "compound_primary_key": compound_primary_key,
        "labels": train_val_data["val"]["labels"]
    }
    train_dataset_params.update(data_cfg)
    
    train_dataset = GenericPairDataset(**train_dataset_params)
    val_dataset = GenericPairDataset(**val_dataset_params)

    user_collator = make_side_collator_from(user_prep)
    item_collator = make_side_collator_from(item_prep)
    collate_fn = partial(collate_rerank, user_collator=user_collator, item_collator=item_collator)

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


def _inbatch_neg_sampling(u:torch.Tensor, v:torch.Tensor, tau:float, neg_k:int=4, hard_frac:float=0.5):
    """
    u, v: [B, D]
    Returns paired tensors for reranker:
      u_pairs, v_pairs: [B*(1+K), D]
      r_scores:         [B*(1+K), 1]  (retriever sims)
      y:                [B*(1+K), 1]  (labels: 1 for pos, 0 for negs)
    """
    device = u.device
    B = u.size(0)
    sims = (u @ v.T) / tau # [B,B]

    # mask out diagonal for negative selection
    mask_pos = torch.eye(B, device=device, dtype=torch.bool)
    sims_off = sims.masked_fill(mask_pos, float('-inf'))

    hard_k = int(neg_k * hard_frac)
    rand_k = neg_k - hard_k

    # hard negatives: top-k sims per row
    hard_idx = torch.topk(sims_off, k=min(hard_k, max(B-1, 0)), dim=1).indices if hard_k > 0 \
               else torch.empty(B, 0, dtype=torch.long, device=device)

    # random negatives per row, excluding pos and already chosen hards
    if rand_k > 0:
        all_cols = torch.arange(B, device=device).expand(B, B)
        avoid = torch.zeros_like(mask_pos)
        if hard_k > 0:
            avoid.scatter_(1, hard_idx, True)
        pool_mask = (~mask_pos) & (~avoid)

        rand_idx_list = []
        for i in range(B):
            cand = all_cols[i, pool_mask[i]]           # [M_i]
            sel = cand[torch.randint(0, cand.numel(), (rand_k,), device=device)]
            rand_idx_list.append(sel)
        rand_idx = torch.stack(rand_idx_list, dim=0)   # [B, rand_k]
    else:
        rand_idx = torch.empty(B, 0, dtype=torch.long, device=device)

    neg_idx = torch.cat([hard_idx, rand_idx], dim=1)   # [B, neg_k]

    # Build (user, item) pairs
    pos_item_idx = torch.arange(B, device=device)
    all_item_idx = torch.cat([pos_item_idx.view(B,1), neg_idx], dim=1)     # [B,1+K]
    all_user_idx = torch.arange(B, device=device).view(B,1).expand(B, 1+neg_k)

    u_pairs = u[all_user_idx.reshape(-1)]                                     # [B*(1+K), D]
    v_pairs = v[all_item_idx.reshape(-1)]                                     # [B*(1+K), D]
    r_scores = sims[all_user_idx, all_item_idx].reshape(-1,1)                 # [B*(1+K), 1]

    y = torch.zeros(B, 1+neg_k, device=device, dtype=torch.float32)
    y[:, 0] = 1.0
    y = y.reshape(-1, 1)
    return u_pairs, v_pairs, r_scores, y

def train_epoch(user_model, item_model, reranker_model, loss_fn, loader, optimizer, metric, tau, device):
    user_model.train()
    item_model.train()

    loss_sum, total = 0.0, 0

    for batch in loader:
        batch = batch.to(device)
        user_batch, item_batch, user_ids, item_ids, labels = batch.user, batch.item, batch.user_ids, batch.item_ids, batch.labels
        with torch.no_grad():
            user_embedding = user_model(user_batch)
            item_embedding = item_model(item_batch)

        all_pos = bool(torch.all(labels > 0.5).item())
        if all_pos:
            # implicit-only batch
            user_embedding, item_embedding, scores, labels = _inbatch_neg_sampling(user_embedding, item_embedding, tau)
        else:
            sims = (user_embedding @ item_embedding.T) / tau
            scores = torch.diag(sims, diagonal=0)

        logits = reranker_model(user_embedding, item_embedding, scores)
        loss = loss_fn(logits, labels.view(-1,1))

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        metric.update(labels, logits)

        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)

    return loss_sum / total

@torch.no_grad()
def evaluate(user_model, item_model, reranker_model, loss_fn, loader, metric, tau, device):
    user_model.eval()
    item_model.eval()

    loss_sum, total = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        user_batch, item_batch, user_ids, item_ids, labels = batch.user, batch.item, batch.user_ids, batch.item_ids, batch.labels
        with torch.no_grad():
            user_embedding = user_model(user_batch)
            item_embedding = item_model(item_batch)

        all_pos = bool(torch.all(labels > 0.5).item())
        if all_pos:
            # implicit-only batch
            user_embedding, item_embedding, scores, labels = _inbatch_neg_sampling(user_embedding, item_embedding, tau)
        else:
            sims = (user_embedding @ item_embedding.T) / tau
            scores = torch.diag(sims, diagonal=0)

        logits = reranker_model(user_embedding, item_embedding, scores)
        loss = loss_fn(logits, labels.view(-1,1))

        metric.update(labels, logits)

        loss_sum += loss.item() * labels.size(0)
        total += labels.size(0)

    return loss_sum / total

def main(config:dict):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, (user_total_bins, item_total_bins), (size_id_user_feats, size_id_item_feats), (num_bag_user_feats, num_bag_item_feats), \
        (num_image_user_feats, num_image_item_feats) = make_loaders(config)

    # Load USER and ITEM towers
    retriever_cfg = config["RETRIEVER"]
    data_name = config["DATA"]["data_name"]
    folder_path = config["SAVE"]["folder_path"]

    user_algorithm_name = retriever_cfg["user_algorithm_name"]
    item_algorithm_name = retriever_cfg["item_algorithm_name"]

    retriever_cfg[user_algorithm_name].update(
        {
            "size_id_features": size_id_user_feats,
            "num_bag_features": num_bag_user_feats,
            "num_image_features": num_image_user_feats,
            "total_bins": user_total_bins            
        }
    )
    user_model = create_algorithm(user_algorithm_name, **retriever_cfg[user_algorithm_name])
    user_state_dict = load_file(os.path.join(folder_path, data_name, user_algorithm_name, "best.safetensors"))
    user_model.load_state_dict(user_state_dict)
    user_model.to(device)
    user_model.eval()

    retriever_cfg[item_algorithm_name].update(
        {
            "size_id_features": size_id_item_feats,
            "num_bag_features": num_bag_item_feats,
            "num_image_features": num_image_item_feats,
            "total_bins": item_total_bins            
        }
    )
    item_model = create_algorithm(item_algorithm_name, **retriever_cfg[item_algorithm_name])
    item_state_dict = load_file(os.path.join(folder_path, data_name, item_algorithm_name, "best.safetensors"))
    item_model.load_state_dict(item_state_dict)
    item_model.to(device)
    item_model.eval()

    for p in user_model.parameters(): p.requires_grad_(False)
    for p in item_model.parameters(): p.requires_grad_(False)


    reranker_cfg = config["RERANKER"]
    reranker_algorithm_name = reranker_cfg["algorithm_name"]
    reranker_cfg[reranker_algorithm_name].update(
        {
            "embed_dim": user_model.get_output_dim()
        }
    )
    reranker_model = create_algorithm(reranker_algorithm_name, **reranker_cfg[reranker_algorithm_name])
    reranker_model.to(device)

    train_cfg = config["TRAIN"]

    optimizer = torch.optim.Adam(params=reranker_model.parameters(), lr=train_cfg["learning_rate"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

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
        train_loss = train_epoch(user_model, item_model, reranker_model, loss_fn, train_loader, optimizer, train_metric, train_cfg["tau"], device)
        val_loss = evaluate(user_model, item_model, reranker_model, loss_fn, val_loader, val_metric, train_cfg["tau"], device)
        
        print(f"Epoch {epoch:03d} | train loss {train_loss:.4f} | val loss {val_loss:.4f} | " + 
            format_metric(train_metric, "train") + " | " + format_metric(val_metric, "val"))
        
        mean_metric_dict = val_metric.mean()
        current_val = mean_metric_dict[benchmark_metric_name]

        if best_val < current_val:
            best_benchmark_metrics_mean = mean_metric_dict
            best_val = current_val
            save_best_safetensors(
                reranker_model,
                path=os.path.join(config["SAVE"]["folder_path"], config["DATA"]["data_name"], config["RERANKER"]["algorithm_name"], "best.safetensors"),
                epoch=epoch,
                metric_name=benchmark_metric_name,
                metric_value=current_val,
            )
        
        train_metric.reset()
        val_metric.reset()

    print(f"Best validation {benchmark_metric_name} mean: {best_benchmark_metrics_mean}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
         prog = "1_reranker_train.py"
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