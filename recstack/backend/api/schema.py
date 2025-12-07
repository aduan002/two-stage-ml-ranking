from __future__ import annotations
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field
from fastapi import UploadFile
from functools import lru_cache
import json, os
import pandas as pd

@lru_cache(maxsize=128)
def load_feature_schema(data_name:str, version:str, folder_path:str, pipeline_name:Optional[str]):
    """
    Locate and load preprocess/schema.json for a (dataset, version).
    """
    schema_path = os.path.join(folder_path, data_name)
    with open(os.path.join(schema_path, "schema.json"), "r", encoding="utf-8") as f:
        return json.load(f)
    raise FileNotFoundError(f"schema.json not found in {schema_path}")


# Map schema dtype strings → Python isinstance checks
_DTYPE_CHECKS = {
    "int": (int,),
    "float": (int, float),
    "number": (int, float),
    "str": (str,),
    "bool": (bool,),
}

def is_type_ok(value:Any, want:str):
    """JSON-friendly dtype check with a special case: don't accept True/False for int."""
    if value is None:
        return False
    t = _DTYPE_CHECKS.get(want)
    if t is None:
        return True
    if want in ("int",) and isinstance(value, bool):
        return False
    return isinstance(value, t)

def as_type(value:Any, target:str):
    """Coerce a single JSON value to target type."""
    if target == "str":
        if value is None:
            raise ValueError("Cannot coerce None to str")
        return str(value)
    if target == "int":
        if value is None or (isinstance(value, bool)):
            raise ValueError("Cannot coerce to int")
        return int(value)
    if target in ("float", "number"):
        if value is None or (isinstance(value, bool)):
            raise ValueError("Cannot coerce to float")
        return float(value)
    if target == "bool":
        if isinstance(value, bool):
            return value
        
        if isinstance(value, (int, float)):
            return value != 0
        if isinstance(value, str):
            if value.lower() in ("true", "1", "yes", "y", "t"):
                return True
            if value.lower() in ("false", "0", "no", "n", "f"):
                return False
        raise ValueError("cannot coerce to bool")
    
    # Unknown target
    return value

class SchemaRequest(BaseModel):
    dataset:Literal["movielens", "pinterest"]
    pipeline_name:Literal["base"] = "base"
    version:Optional[str] = Field(default="1.0.0")

class InferenceRequest(BaseModel):
    dataset:Literal["movielens", "pinterest"]
    pipeline_name:Literal["base"] = "base"
    version:Optional[str] = Field(default="1.0.0")
    rows:Dict[str, List[Any]]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)
    
class FeedbackRequest(BaseModel):
    dataset:Literal["movielens", "pinterest"]
    item_id:str
    rating:float
    
class InferenceResponse(BaseModel):
    item_ids: List[List[int|str]]
    scores: List[List[float]]
    ranks: List[List[int]]
    metadata: List[List[Dict[str, str]]]
    latency_ms:float

class UploadImageResponse(BaseModel):
    scene_id:str
    scene_url:str
    
def validate_rows_against_schema(request:InferenceRequest, folder_path:str):
        req_schema = load_feature_schema(
            data_name=request.dataset,
            version=request.version or "1.0.0",
            folder_path=folder_path,
            pipeline_name=request.pipeline_name
        )

        required:Dict[str, Any] = req_schema.get("required", {})
        optional:Dict[str, Any] = req_schema.get("optional", {})
        allow_extra:bool = bool(req_schema.get("allow_extra", False))
        coercions:Dict[str, str] = req_schema.get("coercions", {})

        # Required keys present
        missing = [k for k in required.keys() if k not in request.rows]
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        # Extra keys 
        extra = [k for k in request.rows.keys() if k not in required and k not in optional]
        if extra and not allow_extra:
            raise ValueError(f"Unexpected feature(s): {extra}")

        # Equal list lengths
        lengths = {len(v) for v in request.rows.values()}
        if len(lengths) > 1:
            raise ValueError(f"All feature lists must have the same length. Got lengths {sorted(lengths)}")

        # Dtype checks (required cannot contain None. Optional may contain None but checked otherwise)
        def _acceptable_types(decl):
            if isinstance(decl, list):
                return [str(x) for x in decl]
            return [str(decl)]

        # Required columns
        for name, decl in required.items():
            values = request.rows.get(name, [])
            wants = _acceptable_types(decl)
            for i, v in enumerate(values):
                if v is None:
                    raise ValueError(f"Column '{name}' has null at position {i}, but it's required")
                if not any(is_type_ok(v, want) for want in wants):
                    raise ValueError(f"Column '{name}' value at idx {i}={v!r} violates dtypes {wants}")

        # Optional columns
        for name, decl in optional.items():
            if name not in request.rows:
                continue
            values = request.rows[name]
            wants = _acceptable_types(decl)
            for i, v in enumerate(values):
                if v is None:
                    continue
                if not any(is_type_ok(v, want) for want in wants):
                    raise ValueError(f"Column '{name}' value at idx {i}={v!r} violates dtypes {wants}")

        # Apply per-column coercions in-place
        for col, target in coercions.items():
            if col in request.rows:
                try:
                    request.rows[col] = [as_type(v, target) for v in request.rows[col]]
                except ValueError as e:
                    raise ValueError(f"Coercion failed for column '{col}': {e}") from e

        return request



