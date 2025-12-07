import json
import math
from typing import Any, Dict, List, Optional, Union
import threading

import pandas as pd
import requests
import streamlit as st

# =========================
# Backend calls
# =========================
def auth_headers(api_key:Optional[str]) -> Dict[str, str]:
    h = {"Accept": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def call_backend(endpoint:str, method:str, payload:Dict[str, Any]) -> Dict[str, Any]:
    url = f"{BASE_URL}/{endpoint}"
    headers = auth_headers(API_KEY)
    if method == "GET":
        r = requests.get(url, json=payload, headers=headers, timeout=TIMEOUT)
    elif method == "POST":
        r = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    if r.status_code != 200:
        try:
            raise RuntimeError(f"{r.status_code}: {r.json()}")
        except Exception:
            raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()

def get_recommend(payload:Dict[str, Any]) -> Dict[str, Any]:
   return call_backend("recommend", "GET", payload)

def get_schema(payload:Dict[str, Any]) -> Dict[str, Any]:
    return call_backend("schema", "GET", payload)

def post_feedback(payload:Dict[str, Any]) -> Dict[str, Any]:
    return call_backend("feedback", "POST", payload)

def post_feedback_async(dataset:str, item_id:str, label:int):
    def worker():
        try:
            post_feedback({"dataset": dataset, "item_id": item_id, "rating": label})
        except Exception as e:
            pass
    threading.Thread(target=worker, daemon=True).start()


def upload_image(file) -> Dict[str, Any]:
    url = f"{BASE_URL}/upload_image"
    headers = auth_headers(API_KEY)
    
    data = {"dataset": "pinterest"}
    files = {
        "file": (file.name, file.getvalue(), file.type or "application/octet-stream")
    }

    r = requests.post(url, data=data, files=files, headers=headers, timeout=TIMEOUT)
    if r.status_code != 200:
        try:
            raise RuntimeError(f"{r.status_code}: {r.json()}")
        except Exception:
            raise RuntimeError(f"{r.status_code}: {r.text}")
    return r.json()



    

# =========================
# Page and Sidebar
# =========================
st.set_page_config(page_title="RecStack UI", page_icon=":brain:", layout="wide")
st.title("🧠 Recommendation UI")

st.sidebar.title("API")
BASE_URL = st.sidebar.text_input("Base URL", st.secrets.get("BASE_URL", "http://localhost:8000")).rstrip("/")
API_KEY  = st.sidebar.text_input("API key", st.secrets.get("API_KEY", ""), type="password")
TIMEOUT  = st.sidebar.number_input("HTTP timeout (s)", 5, 60, 10)
PAGE_SZ  = st.sidebar.slider("Cards per page", 3, 24, 9, step=3)

DATASETS = ["movielens", "pinterest"]
UPLOAD_IMG_DATASETS = ["pinterest"]
CARD_IMAGE_WIDTH = 260

SCHEMAS:Dict[str, Dict[str, Any]] = st.session_state.setdefault("schemas", {})
FEEDBACK:Dict[tuple[str,str], int] = st.session_state.setdefault("feedback", {})

st.session_state.setdefault("upload_meta", None)
st.session_state.setdefault("upload_key", None)

# =========================
# Dataset And Pipeline controls
# =========================
c1, c2, c3 = st.columns([1.2, 1, 1])
with c1:
    dataset = st.selectbox("Dataset", DATASETS)
with c2:
    pipeline_name = st.text_input("Pipeline", value="base")
with c3:
    version = st.text_input("Version", value="1.0.0")

# =========================
# Schema
# =========================
if dataset not in SCHEMAS.keys():
    SCHEMAS[dataset] = get_schema({"dataset": dataset, "pipeline_name": pipeline_name, "version": version})

schema = SCHEMAS[dataset]
st.caption(schema.get("notes", ""))


# =========================
# Utilities
# =========================
def ensure_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else [x]

def coerce(x: Any, typ: str) -> Any:
    if x is None: return None
    try:
        if typ == "str": return str(x)
        if typ == "int": return int(x)
        if typ == "float": return float(x)
        if typ == "bool": return bool(x)
    except Exception:
        return x
    return x

def equal_length_error(rows:Dict[str, Any]) -> Optional[str]:
    lens = [len(v) for v in rows.values() if isinstance(v, list)]
    if not lens:
        return None
    return None if len(set(lens)) == 1 else f"All lists must be equal length, got: {lens}"

# =========================
# Dynamic inputs from schema
# =========================
def render_field(name:str, type_spec:Union[str, List[str]], required:bool):
    label = f"{name}{' *' if required else ''}"
    if isinstance(type_spec, list):
        return st.text_input(label)
    if type_spec in ("int", "float"):
        return st.number_input(label, value=0 if type_spec=="int" else 0.0, step=1 if type_spec=="int" else 1.0)
    if type_spec == "bool":
        return st.checkbox(label, value=False)
    return st.text_input(label)

st.subheader("Inputs")
rows_single: Dict[str, Any] = {}
with st.container(border=True):
    st.markdown("**Required**")
    for k, t in schema.get("required", {}).items():
        if dataset in UPLOAD_IMG_DATASETS:
            st.text_input(f"{k} (will be filled from uploaded image)", disabled=True, key=f"disabled-{k}")
            rows_single[k] = None  # will be overridden after upload
        else:
            rows_single[k] = render_field(k, t, True)

if schema.get("optional"):
    with st.container(border=True):
        st.markdown("**Optional**")
        for k, t in schema.get("optional", {}).items():
            rows_single[k] = render_field(k, t, False)

extra_obj = {}
if schema.get("allow_extra", False):
    with st.expander("➕ Extra fields (JSON object)"):
        raw = st.text_area("Additional key/values (applied to each row)", placeholder='{"foo":"bar"}')
        if raw.strip():
            try:
                extra_obj = json.loads(raw)
                if not isinstance(extra_obj, dict):
                    st.error("Extra must be a JSON object.")
                    extra_obj = {}
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                extra_obj = {}

if dataset in UPLOAD_IMG_DATASETS:
    st.subheader("Image")
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png", "webp"],
        key=f"{dataset}-upload"
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Query image", width="stretch")

        file_key = f"{uploaded_file.name}-{uploaded_file.size}"

        cached_key = st.session_state.get("upload_key")
        cached_response = st.session_state.get("upload_meta")

        if cached_key != file_key or cached_response is None:
            with st.spinner("Uploading image to backend..."):
                try:
                    response = upload_image(uploaded_file)
                except Exception as e:
                    st.error(f"Failed to upload image: {e}")
                else:
                    rows_single = response.copy()
                    st.session_state["upload_key"] = file_key
                    st.session_state["upload_meta"] = response
                    st.success("Image uploaded and linked to this request.")
        else:
            # Reuse cached response
            response = cached_response

        if response is not None:
           rows_single = response.copy()

    else:
        # No file selected, clear cached upload
        st.session_state["upload_key"] = None
        st.session_state["upload_meta"] = None

# =========================
# Build rows payload
# =========================
def build_rows() -> Dict[str, List[Any]]:
    rows = {k: ensure_list(v) for k, v in rows_single.items()}
    if extra_obj:
        for k, v in extra_obj.items():
            rows[k] = ensure_list(v)

    for k, tgt in schema.get("coercions", {}).items():
        if k in rows:
            rows[k] = [coerce(x, tgt) for x in rows[k]]

    for k in schema.get("required", {}):
        if k not in rows:
            rows[k] = []
    return rows

rows = build_rows()
err = equal_length_error(rows)
if err:
    st.error(err)

with st.expander("Request preview (JSON)"):
    st.json({
        "dataset": dataset,
        "pipeline_name": pipeline_name,
        "version": version,
        "rows": rows
    }, expanded=False)

# =========================
# Response handling
# =========================
def normalize_row(
    item_ids: List[Union[int, str]],
    scores: List[float],
    ranks: List[int],
    metas: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    n = min(len(item_ids), len(scores), len(ranks), len(metas))
    items = []
    for i in range(n):
        items.append({
            "id": item_ids[i],
            "score": scores[i],
            "rank": ranks[i],
            "metadata": metas[i] or {}
        })
    # sort by provided rank if it exists
    items.sort(key=lambda x: x.get("rank", 1_000_000))
    return items

def unpack_response(data:Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns:
      {
        "latency_ms": float,
        "batch_size": int,
        "row_items": List[List[Dict]]]  # one list per input row
      }
    """
    item_ids = data.get("item_ids", [])
    scores = data.get("scores", [])
    ranks = data.get("ranks", [])
    metas = data.get("metadata", [])
    latency = data.get("latency_ms", None)

    batch_size = min(len(item_ids), len(scores), len(ranks), len(metas))
    row_items: List[List[Dict[str, Any]]] = []
    for b in range(batch_size):
        row_items.append(
            normalize_row(item_ids[b], scores[b], ranks[b], metas[b])
        )
    return {"latency_ms": latency, "batch_size": batch_size, "row_items": row_items}

# =========================
# Rendering
# =========================
def toggle_feedback(dataset: str, item_id: str, click_label:int):
    k = (dataset, item_id)
    feeback = FEEDBACK.get(k, None)

    if feeback == click_label:
        new_val = None          
    else:
        new_val = click_label  

    FEEDBACK[k] = new_val       

    if new_val is not None:     
        post_feedback_async(dataset, item_id, new_val)


@st.fragment
def card_fragment(item:Dict[str, Any]):
    item_id = str(item.get("id"))
    k = (dataset, item_id)

    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        with h1:
            st.markdown(f"### {item_id}")
        with h2:
            r = item.get("rank")
            s = item.get("score")
            if r is not None:
                st.caption(f"Rank: {r}")
            if s is not None:
                try:
                    st.caption(f"score: {float(s):.3f}")
                except Exception:
                    st.caption(f"score: {s}")

        meta = item.get("metadata") or {}
        img_url = None

        if isinstance(meta, dict):
            for key in meta.keys():
                if key.endswith("_url"):
                    img_url = meta[key]
                    break

        if img_url:
            st.image(img_url, width=CARD_IMAGE_WIDTH)

        if meta:
            with st.expander(f"Details"):
                st.json(meta)

        left_spacer, down_col, up_col = st.columns([6, 1, 1])
        feedback = st.session_state["feedback"].get(k)

        down_type = "primary" if feedback == 0 else "secondary"
        up_type   = "primary" if feedback == 1 else "secondary"

        with down_col:
            st.button("👎", key=f"down-{dataset}-{item_id}", width="stretch", type=down_type, on_click=toggle_feedback, args=(dataset, item_id, 0))
        with up_col:
            st.button("👍", key=f"up-{dataset}-{item_id}", width="stretch", type=up_type, on_click=toggle_feedback, args=(dataset, item_id, 1))

def render_grid(items:List[Dict[str, Any]], page:int, page_size:int):
    total = len(items)
    if total == 0:
        st.info("No items.")
        return
    start, end = page*page_size, min(page*page_size+page_size, total)
    page_items = items[start:end]
    ncols = 3
    rows = math.ceil(len(page_items)/ncols)
    idx = 0
    for _ in range(rows):
        cols = st.columns(3)
        for c in cols:
            if idx < len(page_items):
                with c:
                    # NOTE: Using fragments to solve annoying jitter bug on first like/dislike click after get recommendations
                    card_fragment(page_items[idx])  
                    idx += 1
                    
    p1, p2, p3 = st.columns([1,2,1])
    with p1:
        if st.button("⬅️ Prev", disabled=(start==0)):
            st.session_state["page"] = max(0, st.session_state.get("page",0)-1)
            st.rerun()
    with p2:
        st.markdown(f"<div style='text-align:center;'>Page <b>{start//page_size+1}</b> / <b>{math.ceil(total/page_size)}</b> • Showing {start+1}–{end} / {total}</div>", unsafe_allow_html=True)
    with p3:
        if st.button("Next ➡️", disabled=(end>=total)):
            st.session_state["page"] = st.session_state.get("page",0)+1
            st.rerun()

if "page" not in st.session_state: st.session_state["page"] = 0

# =========================
# Submit
# =========================
clicked = st.button("Get recommendations", type="primary", width="stretch")

if clicked:
    missing = [k for k in schema["required"].keys() if rows.get(k, [None])[0] in (None, "")]
    if missing:
        st.error(f"Missing required fields: {missing}")

    if err:
        st.stop()

    payload = {
        "dataset": dataset,
        "pipeline_name": pipeline_name,
        "version": version,
        "rows": rows
    }

    with st.spinner("Calling /recommend ..."):
        try:
            data = get_recommend(payload)
        except Exception as e:
            st.error(str(e))
            st.stop()

    parsed = unpack_response(data)
    batch_size = parsed["batch_size"]
    latency_ms = parsed["latency_ms"]
    row_items  = parsed["row_items"]

    if latency_ms is not None:
        st.toast(f"✅ Received results • latency: {latency_ms:.1f} ms")
    else:
        st.toast(f"✅ Received results")

    row_idx = 0
    items = row_items[row_idx] if batch_size > 0 else []

    # Store in session for pagination
    st.session_state["data"] = data
    st.session_state["items"] = items                
    st.session_state["page"] = 0    
    for it in st.session_state["items"]:
        k = (dataset, str(it.get("id")))
        FEEDBACK.setdefault(k, None)                

items = st.session_state.get("items", [])
has_results = bool(items)

if has_results:
    with st.expander("📄 Raw response"):
        st.json(st.session_state.get("data", {}))
    with st.expander("📊 Items table"):
        if items:
            # flatten metadata columns into json strings for compactness
            df = pd.DataFrame(items)
            if "metadata" in df.columns:
                df["metadata"] = df["metadata"].apply(lambda d: json.dumps(d, ensure_ascii=False))
            st.dataframe(df, hide_index=True, width="stretch")


    render_grid(items, st.session_state["page"], PAGE_SZ)
else:
    st.caption("Fill the form, then click **Get recommendations**.")
