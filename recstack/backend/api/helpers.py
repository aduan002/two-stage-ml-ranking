import os
import time
import threading
from collections import OrderedDict
from typing import Dict
import logging
import json

import torch
from safetensors.torch import load_file

from ..preprocess.preprocess import GenericPreprocess
from ..factory import create_loader, create_algorithm, create_retriever

def now_ms() -> int:
    return int(time.time() * 1000)

class InferencePipeline:
    def __init__(self, item_mapping, user_preprocess, user_model, retriever, reranker, bytes_estimate:int, tau:float):
        self.item_mapping = item_mapping
        self.user_preprocess = user_preprocess
        self.user_model = user_model

        self.retriever = retriever
        self.reranker = reranker
        self.loaded_at = time.time()
        self.last_used = self.loaded_at
        self.bytes_estimate = bytes_estimate
        self.tau = tau

    def get_item_mapping(self): return self.item_mapping
    def get_user_preprocess(self): return self.user_preprocess
    def get_user_model(self): return self.user_model
    def get_retriever(self): return self.retriever
    def get_reranker(self): return self.reranker
    def get_tau(self): return self.tau

class InferenceStore:
    def __init__(self, device:str, max_loaded:int=3, ttl_s:int=None, gpu_mem_cap_mb:int=None):
        self.device = device
        self.max_loaded = max_loaded
        self.ttl_s = ttl_s
        self.gpu_mem_cap_mb = gpu_mem_cap_mb
        self.lock = threading.Lock()
        self.lru = OrderedDict()

    def get_device(self): return self.device
    
    def evict_if_needed(self):
        # Time-to-live eviction
        if self.ttl_s:
            cutoff = time.time() - self.ttl_s
            for k, p in list(self.lru.items()):
                if p.last_used < cutoff:
                    self.lru.pop(k, None)
        
        # Count-based eviction
        while len(self.lru) > self.max_loaded:
            self.lru.popitem(last=False) # FIFO

    def gpu_mem_ok(self, bytes_to_add:int, overhead:float=0.15, reserve_mb:int=512):
        """
        If there's likely enough GPU memory to load a new model of size `bytes_to_add` with a safety `overhead`, 
        and a safety headroom of `reserve_mb`, return True
        """
        def _bytes(num_mb:int):
            return num_mb * 1024 * 1024
        
        if self.gpu_mem_cap_mb is None or not str(self.device).startswith("cuda"):
            return True
        if not torch.cuda.is_available():
            return True
        
        device = torch.device(self.device)

        # Free this process's cache before measuring
        torch.cuda.empty_cache()

        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        need_bytes = int(bytes_to_add * (1 + overhead))

        # Enforce config memory cap and actual free memory cap
        cap_ok = (total_bytes - free_bytes + need_bytes) <= _bytes(self.gpu_mem_cap_mb)
        headroom_ok = (free_bytes - _bytes(reserve_mb)) >= need_bytes

        return cap_ok and headroom_ok
    

    def get_preprocess_and_model(self, folder_path:str, data_name:str, preprocess_name:str, pipeline_cfg:dict, pipeline_key:str):
        prep = GenericPreprocess.load(os.path.join(folder_path, data_name, preprocess_name))
        total_bins = prep.get_total_bins()
        size_id_feats = prep.get_size_id_features()
        num_bag_feats = prep.get_num_bag_features()
        num_image_feats = prep.get_num_image_features()

        algorithm_cfg = pipeline_cfg["retriever"]
        algorithm_name = algorithm_cfg[pipeline_key]

        algorithm_cfg[algorithm_name].update(
            {
                "size_id_features": size_id_feats,
                "num_bag_features": num_bag_feats,
                "num_image_features": num_image_feats,
                "total_bins": total_bins
            }
        )
        model = create_algorithm(algorithm_name, **algorithm_cfg[algorithm_name])
        state_dict = load_file(os.path.join(folder_path, data_name, algorithm_name, "best.safetensors"))
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return prep, model

    
    def get(self, data_name:str, pipeline_name:str, version:str, pipeline_cfg:Dict) -> InferencePipeline:
        key = (data_name, pipeline_name, version)

        with self.lock:
            if key in self.lru:
                p = self.lru.pop(key)
                p.last_used = time.time()
                self.lru[key] = p
                return p
        
        folder_path = pipeline_cfg["folder_path"]
        dataloader = create_loader(data_name + "_loader", **pipeline_cfg[data_name])
        dataloader.load()
        item_features = dataloader.item_features()
        item_features = item_features.drop_duplicates()
        item_mapping = item_features.set_index(pipeline_cfg[data_name]["item_primary_key"]).to_dict(orient='index')

        user_prep, user_model = self.get_preprocess_and_model(folder_path, data_name, "user_preprocess", pipeline_cfg, "user_algorithm_name")
        #item_prep, item_model = self.get_preprocess_and_model(folder_path, data_name, "item_preprocess", pipeline_cfg, "item_algorithm_name")

        retriever_cfg = pipeline_cfg["retrieval"]
        retriever_name = retriever_cfg["retriever_name"]
        retriever = create_retriever(retriever_name, **retriever_cfg[retriever_name])
        retriever.load()

        reranker_cfg = pipeline_cfg["reranker"]
        reranker_name = reranker_cfg["algorithm_name"]
        reranker_cfg[reranker_name].update(
            {
                "embed_dim": user_model.get_output_dim()
            }
        )
        reranker_model = create_algorithm(reranker_name, **reranker_cfg[reranker_name])
        reranker_model_state_dict = load_file(os.path.join(folder_path, data_name, reranker_name, "best.safetensors"))
        reranker_model.load_state_dict(reranker_model_state_dict)
        reranker_model.to(self.device)
        reranker_model.eval()

        bytes_estimate = sum(p.numel() * p.element_size() for p in user_model.parameters())

        with self.lock:
            if not self.gpu_mem_ok(bytes_estimate):
                while len(self.lru) and not self.gpu_mem_ok(bytes_estimate):
                    self.lru.popitem(last=False)
                if not self.gpu_mem_ok(bytes_estimate):
                    raise RuntimeError("GPU memory cap reached. Cannot load model.")
                
            self.lru[key] = InferencePipeline(item_mapping, user_prep, user_model, retriever, reranker_model, bytes_estimate, pipeline_cfg["retriever"]["tau"])
            self.evict_if_needed()
            return self.lru[key]
        
    def preload(self, items:list[dict]):
        for it in items:
            try:
                self.get(it["data_name"], it["pipeline_name"], it["version"], it["pipeline_cfg"])
            except Exception as e:
                print(f'[preload] failed {it["data_name"]}-{it["pipeline_name"]}-{it["version"]}: {e}')

            
class JsonFormatter(logging.Formatter):
    def format(self, record:logging.LogRecord) -> str:
        base = {"level": record.levelname, "msg": record.getMessage()}

        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            base.update(extra)

        return json.dumps(base)