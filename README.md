# RecStack – Two-Stage Recommendation System (Retriever + Reranker)

A production-style recommendation pipeline with:

- **Two-tower retriever** (user / item towers, ANN index)
- **Deep Learning (DL) reranker** on top of retriever scores
- **Generic feature pipeline** that works for:
  - **Tabular data** (e.g. MovieLens)
  - **Image-based data** (e.g. Pinterest-style)
- **FastAPI backend** for online inference
- **Streamlit UI** for interactive exploration and feedback

---

## High-level Overview

This project implements a **two-stage recommender system**:

1. **Retriever (tower models)**
   - User and item are encoded by separate neural networks.
   - Training objective: maximize similarity between a user and their interacted items.
   - Retrieval uses an ANN index over item embeddings.

2. **Reranker**
   - Takes:
     - User embedding  
     - Candidate item embedding  
     - Retriever similarity score
   - Learns to more finely distinguish relevant vs. non-relevant items (BCE loss).

3. **Serving**
   - **FastAPI** backend exposes:
     - `GET /recommend` – retrieve recommendations
     - `GET /schema` – get expected input schema
     - `POST /feedback` – log user feedback
     - `POST /upload_image` – upload an image query (for Pinterest-like data)
   - **Streamlit** front-end:
     - Dynamic form built from `/schema`
     - Image upload for Pinterest
     - Renders recommendations as cards with images + feedback buttons

The same pipeline works for **MovieLens (tabular)** and **Pinterest-like (image)** datasets by plugging in different configs + preprocessors.

![High-level architecture diagram](/assets/recstack_diagram.png)
---

## Architecture

### 1. Data & Preprocessing

- Generic preprocessing using `GenericPreprocess`:
  - Supports:
    - Vocabulary columns (e.g. `user_id`, `gender`, `age`, `occupation`, `movie_id`, `title`)
    - Hashed columns (e.g. `zip_code`, `genres`)
  - Tracks:
    - Primary keys
    - Bag features
    - Embedding columns (tabular and/or image)
    - ID feature sizes + total bins for embedding tables
- Saves preprocessing artifacts per dataset. Example:
  - `artifacts/<dataset_name>/user_preprocess/`
  - `artifacts/<dataset_name>/item_preprocess/`

### 2. Retriever Training

`0_retriever_train.py`:

- Builds `DataLoader`s using `GenericPairDataset`.
- Creates user / item models via a factory.
- Loss:
  - In-batch retrieval loss with temperature `tau`:
    - Compute similarity matrix between user and item embeddings.
    - For each user, maximize total softmax probability assigned to all their positive items.
- Metrics:
  - Generic retrieval metrics (e.g. `recall@k`).
- Saves **best checkpoints** as `safetensors` + a small JSON metadata file. Example:
  - `artifacts/<data_name>/<user_algorithm_name>/best.safetensors`
  - `artifacts/<data_name>/<item_algorithm_name>/best.safetensors`

### 3. Reranker Training

`1_reranker_train.py`:

- Reloads **frozen** user and item towers from best retriever checkpoints.
- Supports two label shapes:
  - Explicit labels (0/1, ratings, etc.)
  - Fully positive batches (implicit feedback).
- For fully positive batches, uses `_inbatch_neg_sampling`:
  - Generates hard + random negatives from the batch.
- Reranker model consumes:
  - User embedding
  - Item embedding
  - Retriever similarity score
- Trains with `BCEWithLogitsLoss`.
- Tracks metrics and saves best reranker checkpoint. Example:
  - `artifacts/<data_name>/<reranker_algorithm_name>/best.safetensors`

### 4. Index Building

`2_retriever_build_index.py`:

- Loads item tower + item preprocess.
- Computes embeddings for **all items**.
- Adds them to an ANN retriever.

### 5. API (FastAPI)

`recstack/backend/api/main.py`:

- Lifespan context initializes a global `InferenceStore`:
  - Caches loaded pipelines (user tower, item index, reranker, preprocessors).
  - Supports:
    - Device management (CPU / GPU)
    - Max loaded models
    - TTL
    - GPU memory cap
- Endpoints:

| Method | Endpoint        | Description                                         |
|--------|------------------|-----------------------------------------------------|
| GET    | `/healthz`       | Basic health check.                                 |
| GET    | `/readyz`        | Readiness check (pipeline loaded).                  |
| GET    | `/schema`        | Returns feature schema for a dataset/pipeline/version. |
| GET    | `/recommend`     | Main inference endpoint.                            |
| POST   | `/feedback`      | Accepts feedback for item recommendations.          |
| POST   | `/upload_image`  | Handles image upload, stores file, returns `scene_id` and URL. |


Prediction flow:

1. Validate incoming rows against schema.
2. Transform into user features via `GenericPreprocess`.
3. Build a dummy `GenericPairDataset` (fake item for batching).
4. Encode users → embeddings.
5. For each user:
   - Use retriever to get candidate item IDs + distances.
   - Pull item embeddings from retriever.
   - Compute retriever similarity (`q @ item_embeddings.T / tau`).
   - Rerank using reranker model.
6. Apply sigmoid to logits, perform top-k selection.
7. Return:
   - `item_ids`, `scores`, `ranks`
   - `metadata` (from item mapping)
   - `latency_ms`

### 6. Streamlit UI

`recstack/frontend/app.py`:

- Left sidebar:
  - Backend base URL, API key, timeout, page size.
- Main controls:
  - Dataset selection (e.g. `movielens`, `pinterest`).
  - Pipeline name and version.
- Fetches schema from `/schema` and dynamically renders:
  - Required fields
  - Optional fields
  - Optional extra JSON fields
- For image datasets:
  - File uploader sends file to `/upload_image`.
  - Backend responds with `scene_id` and `scene_url`, which are injected into the request rows.
- Calls `/recommend` and displays:
  - Raw JSON response.
  - Flattened table view.
  - Recommendation cards (with:
    - image preview when a `_url` field is present
    - rank, score
    - metadata in expander
    - 👍 / 👎 feedback buttons).

### Streamlit UI (Demo)

The Streamlit app is designed as a lightweight **engineering validation tool**
rather than a production UI. It demonstrates:

- Dynamic schema-driven inputs
- Image upload for the Pinterest pipeline
- Online ANN retrieval + deep reranking
- Real-time recommendation display
- Explicit user feedback logging (👍 / 👎)

The goal is to validate the **end-to-end serving pipeline**, not to showcase UI/UX design.

<img src="https://github.com/user-attachments/assets/6afc908d-f962-4a1b-bd15-7a54092a849e" alt="Pinterest demo">

---

## Project Structure


```text
recstack/
├── recstack/
│   ├── backend/
│   │   ├── algorithm/
│   │   ├── api/                     # FastAPI
│   │   ├── datasets/
│   │   ├── metric/
│   │   ├── preprocess/
│   │   ├── search/
│   │   ├── factory.py
│   │   └── registry.py
│   ├── frontend/
│   │   │   ├── app.py                # Streamlit
│   └── ...
├── cfg/
│   ├── api.yaml
│   ├── reranker_*.yaml
│   └── retrieval_*.yaml
├── artifacts/
│   └── ...                           # model checkpoints, preprocessors, indexes
├── 0_retriever_train.py
├── 1_reranker_train.py
├── 2_retriever_build_index.py
└── README.md
```

## How to Run Locally

### Clone the Repository
```bash
git clone https://github.com/aduan002/RecStack.git
cd RecStack
```

### Install dependencies
```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Download Datasets
- Download a dataset e.g. Movielens or Pinterest.
- Update files in `cfg/` accordingly.

### Training

#### Train Retriever (User + Item towers)
```bash
python 0_retriever_train.py -c cfg/retrieval_movielens.yaml
```
This will:
- Preprocess data (fit + transform).
- Train user / item towers.
- Save:
    - Preprocessors
    - Best user / item checkpoints (safetensors)
    - Metrics logs.

### Train Reranker
```bash
python 1_reranker_train.py -c cfg/reranker_movielens.yaml
```

This will:
- Reload best retriever towers.
- Freeze them.
- Train reranker on:
    - Explicit labels OR
    - Implicit-only batches with in-batch negatives.

### Build Retrieval Index
```bash
python 2_retriever_build_index.py -c cfg/reranker_movielens.yaml
```
This will:
- Compute item embeddings for all items.
- Build an ANN index.
- Save the index in `folder_path` (per dataset).

### Running the API
```bash
python -m recstack.backend.api.main -c cfg/api.yaml
```

### Running the Streamlit UI
```bash
cd recstack/frontend 
streamlit run app.py
```
UI displays:
- ID, rank, score
- Image preview (if available)
- Metadata in a collapsible section
- Feedback buttons (👍 / 👎)
- Raw JSON response & a dataframe view.


## Design Trade-offs & Limitations

- Current reranker is trained offline; online learning is not yet implemented.
- Feedback is logged but not yet used for continuous retraining.
- ANN index is rebuilt in batch mode.
- The Streamlit UI is intentionally minimal and intended only for engineering validation.
