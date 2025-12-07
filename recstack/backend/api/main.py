import logging
from contextlib import asynccontextmanager
import numpy as np
import uvicorn
import argparse
import os
from torch.utils.data import DataLoader
import pandas as pd
from typing import Any, Union
from pathlib import Path
from fastapi.staticfiles import StaticFiles

import torch
from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from uuid import uuid4
import shutil

from ..algorithm.collate import SideCollator
from ..datasets.dataset import GenericPairDataset

from .schema import InferenceRequest, InferenceResponse, SchemaRequest, FeedbackRequest, UploadImageResponse, \
    validate_rows_against_schema, load_feature_schema
from .helpers import JsonFormatter, InferenceStore, now_ms
from .settings import load_settings_from_yaml, to_dict

INFERENCE_STORE = None

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("API")
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

app = FastAPI(title="RecStack API", version="1.0.0")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR), html=False), name="uploads")


@asynccontextmanager
async def lifespan(app:FastAPI):
    global INFERENCE_STORE

    config = app.state.settings_preloaded

    serve_config = config["SERVE"]
    serve_cache_config = serve_config["cache"]

    INFERENCE_STORE = InferenceStore(
        device=serve_config["device"],
        max_loaded=serve_cache_config["max_loaded_models"],
        ttl_s=serve_cache_config["ttl_s"],
        gpu_mem_cap_mb=serve_cache_config["gpu_mem_cap_mb"]
    )

    INFERENCE_STORE.preload(serve_config["preload"])
    app.state.ready = True

    yield


app.router.lifespan_context = lifespan

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.get("/readyz")
def readyz(request:Request):
    return {"ready": bool(getattr(request.app.state, "ready", False))}


def collate_users_only(batch, user_collator:SideCollator):
    user_bags_list = [s["user_bags"] for s in batch]
    user_images_list = [s.get("user_images", {}) for s in batch]
    user_ids_list = [s["user_id_index"] for s in batch]

    user_side = user_collator(user_bags_list, user_images_list, user_ids_list)

    raw_user_ids = [s["user_id"].item() if hasattr(s["user_id"], 'item') else s["user_id"] for s in batch]

    return raw_user_ids, user_side

def predict_batch(user_prep, user_model, user_data, retriever, reranker_model, device:str="cuda:0",
        batch_size:int = 64, num_workers:int = 4, tau:float=0.7, dataset_kwargs:dict = {}) -> list[tuple[list,list]]:
    user_model.eval()
    preds_all = []

    user_feats = user_prep.transform(user_data)

    # Fake interactions with dummy item 
    DUMMY_ITEM_ID = 0
    interactions = pd.DataFrame({
        user_prep.get_primary_key(): list(user_feats["id_to_row"].keys()),
        "item_id": np.full(len(user_feats["id_to_row"]), DUMMY_ITEM_ID)
    })
    item_feats = {
        "id_to_row": {DUMMY_ITEM_ID: 0}, "bags": {}, "embeds": {},
        "ids": {"item_id": np.array([0], dtype=np.int64)}
    }

    user_field_order = user_prep.get_bags()
    user_embed_cols = user_prep.get_embeds()
    user_collator = SideCollator(user_field_order, user_embed_cols)

    dataset = GenericPairDataset(user_feats, item_feats, interactions, compound_primary_key=(user_prep.get_primary_key(), "item_id"),
        **dataset_kwargs)
    loader = DataLoader(dataset, batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda b: collate_users_only(b, user_collator))

    with torch.inference_mode():
        for raw_user_ids, user_side in loader:
            user_side = user_side.to(device)
            user_embeddings = user_model(user_side)

            for q in user_embeddings:
                item_ids, distances = retriever.search(q)
                item_embeddings = torch.from_numpy(retriever.get_embeddings(item_ids))
                item_embeddings = item_embeddings.to(device)

                retriever_scores = (q @ item_embeddings.T) / tau
                q_expanded = q.expand(item_embeddings.size(0), -1)
                reranker_scores = reranker_model(q_expanded, item_embeddings, retriever_scores).squeeze(-1)
                reranker_scores = reranker_scores.detach().cpu().numpy().tolist()

                preds_all.append((item_ids, reranker_scores))

    return preds_all

@app.get("/recommend", response_model=InferenceResponse)
async def query(request:InferenceRequest):
    if not getattr(app.state, "ready", False):
        raise HTTPException(503, "Service not ready")
    
    serve_config = app.state.settings_preloaded["SERVE"]
    
    t0  = now_ms()
    request = validate_rows_against_schema(request=request, folder_path=serve_config["schema_folder"])
    try:
        data = request.to_dataframe()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid rows payload: {e}")
    
    keys = {"data_name": request.dataset, "pipeline_name": request.pipeline_name, "version": request.version}
    matches = [
        d for d in serve_config["preload"]
        if all(item in d.items() for item in keys.items())
    ]
    if not matches:
        raise HTTPException(status_code=400, detail=f"Invalid dataset-pipeline-version payload")
    
    inference_pipeline = INFERENCE_STORE.get(**matches[0])
    item_mapping = inference_pipeline.get_item_mapping()
    user_preprocess = inference_pipeline.get_user_preprocess()
    user_model = inference_pipeline.get_user_model()
    retriever = inference_pipeline.get_retriever()
    reranker_model = inference_pipeline.get_reranker()
    tau = inference_pipeline.get_tau()
    # List[(item_ids, scores)]
    preds = predict_batch(user_preprocess, user_model, data, retriever, reranker_model, INFERENCE_STORE.get_device(), serve_config["batch_size"], serve_config["num_workers"], tau)
    item_ids, scores = zip(*preds)
    item_ids, scores = np.array(item_ids), np.array(scores)
    scores = 1 / (1 + np.exp(-scores)) # reranker scores are logits

    idx_part = np.argpartition(-scores, serve_config["top_k"] - 1, axis=-1)[:, :serve_config["top_k"]] # fast partial sort
    scores_part = np.take_along_axis(scores, idx_part, axis=-1)             
    order = np.argsort(-scores_part, axis=-1)                                 
    topk_idx = np.take_along_axis(idx_part, order, axis=-1)       

    item_ids = np.take_along_axis(item_ids, topk_idx, axis=-1) # [B, k]
    scores = np.take_along_axis(scores, topk_idx, axis=-1)     # [B, k]

    ranks = scores.argsort().argsort() + 1
    ranks = [(len(score) - rank + 1).tolist() for score, rank in zip(scores, ranks)]

    item_ids = item_ids.tolist()
    scores = scores.tolist()

    return InferenceResponse(
        item_ids=item_ids,
        scores=scores,
        ranks=ranks,
        metadata=[[item_mapping[i] for i in sublist] for sublist in item_ids],
        latency_ms=now_ms() - t0
    )

@app.get("/schema", response_model=dict[str, Union[dict[str, Any], bool]])
async def schema(request:SchemaRequest):
    serve_config = app.state.settings_preloaded["SERVE"]

    return load_feature_schema(
        data_name=request.dataset,
        version=request.version or "1.0.0",
        folder_path=serve_config["schema_folder"],
        pipeline_name=request.pipeline_name
    )

@app.post("/feedback", response_model=dict[str, Any])
async def feedback(request:FeedbackRequest):
    print(f"Received: {request.dataset}, {request.item_id}, {request.rating}")

    response = {**request.model_dump(), "ok": True}
    return response

@app.post("/upload_image", response_model=UploadImageResponse)
async def upload_image(request:Request, dataset:str = Form("pinterest"), file:UploadFile = File(...)):
    ext = Path(file.filename).suffix or ".jpg"
    scene_id = f"upload_{uuid4().hex}"
    filename = f"{scene_id}{ext}"

    file_path = UPLOAD_DIR / filename
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    scene_url = str(file_path.resolve())

    return UploadImageResponse(
        scene_id=scene_id,
        scene_url=scene_url,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="app.py", description="WideDeep API")
    parser.add_argument("-c", "--config", help="Path to YAML config", required=False)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)

    args = vars(parser.parse_args())

    config_path = args["config"] or os.getenv("APP_CONFIG")
    settings_obj = load_settings_from_yaml(config_path) 

    app.state.settings_preloaded = to_dict(settings_obj)

    uvicorn.run(app, host=args["host"], port=args["port"])
    

