from __future__ import annotations
from dataclasses import asdict, is_dataclass
from pydantic import BaseModel, Field, ValidationError, ConfigDict, field_validator, model_validator
from typing import Any, Dict, List, Optional
import os
import yaml


class PreloadItem(BaseModel):
    data_name:str
    pipeline_name:str # "base"
    version:str
    pipeline_cfg:Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

class CacheConfig(BaseModel):
    max_loaded_models:int = Field(default=3, ge=1)
    ttl_s:Optional[int] = Field(default=1800, ge=1)
    gpu_mem_cap_mb:Optional[int] = Field(default=None, ge=1)

    model_config = ConfigDict(extra="forbid")

# ---- SERVE block ----
class ServeConfig(BaseModel):
    device:str = "cuda:0"                  
    schema_folder:str = Field(default="artifacts")
    batch_size:int = Field(default=512, ge=1)
    num_workers:int = Field(default=4, ge=0)
    top_k:int = Field(default=5, ge=1)
    preload:List[PreloadItem] = Field(default_factory=list)
    cache:CacheConfig = Field(default_factory=CacheConfig)


# --------------------
# Root Validation
# --------------------
class AppSettings(BaseModel):
    serve:ServeConfig = Field(alias="SERVE")


def load_settings_from_yaml(path: str) -> AppSettings:
    if not path:
        raise SystemExit("No config path provided. Use --config or set APP_CONFIG.")
    if not os.path.exists(path):
        raise SystemExit(f"Config file not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"Failed to parse YAML: {e}") from e

    try:
        return AppSettings(**raw)
    except ValidationError as e:
        raise SystemExit(f"Config validation error:\n{e}") from e
    
def to_dict(obj):
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "model_dump"):   # Pydantic v2
        return obj.model_dump(by_alias=True)
    if hasattr(obj, "dict"):         # Pydantic v1
        return obj.dict()
    if isinstance(obj, dict):
        return obj
    return obj  # fallback
