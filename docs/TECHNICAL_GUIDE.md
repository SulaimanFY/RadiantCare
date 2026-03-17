# Technical Guide — RadiantCare

This document explains the architecture and implementation of RadiantCare in detail. The backend code (`main.py`, `rag.py`, `gradcam.py`) is **thoroughly commented line-by-line** to make every step understandable and reproducible. The project applies **MLOps** practices (model serving, containerization, observability, reproducibility, deployment) so the pipeline from training to inference is consistent and deployable.

---

## Table of contents

1. [Overview](#1-overview)
2. [Model (DenseNet-121)](#2-model-densenet-121)
3. [API architecture](#3-api-architecture)
4. [Grad-CAM](#4-grad-cam)
5. [RAG (Retrieval-Augmented Generation)](#5-rag-retrieval-augmented-generation)
6. [Frontend](#6-frontend)
7. [Docker & deployment](#7-docker--deployment)
8. [MLOps](#8-mlops)

---

## 1. Overview

RadiantCare is an end-to-end chest X-ray analysis system:

```
User uploads X-ray image
        │
        ▼
┌──────────────────┐
│   FastAPI (API)   │
│                  │
│  1. Preprocess   │  Resize 224×224, normalize (ImageNet stats)
│  2. Inference    │  DenseNet-121 → 14 probabilities
│  3. Grad-CAM     │  Heatmap for top finding
│  4. LLM Report   │  OpenAI Chat API + RAG context
│                  │
└──────────────────┘
        │
        ▼
Frontend displays: predictions table, heatmap, report, chat
```

---

## 2. Model (DenseNet-121)

### Architecture

```
Input image (3, 224, 224)
        │
        ▼
┌─────────────────────┐
│  DenseNet-121        │  Pretrained on ImageNet (1.2M images, 1000 classes)
│  backbone.features   │  Convolutional layers → (1024, 7, 7) feature maps
│  backbone.classifier │  Replaced by Identity() → outputs (1024,) vector
└─────────────────────┘
        │
        ▼  (1024-dim feature vector)
┌─────────────────────┐
│  Custom head         │
│  Dropout(0.3)        │  Regularization: randomly zeros 30% of features
│  Linear(1024, 14)    │  Maps to 14 raw logits (one per pathology)
└─────────────────────┘
        │
        ▼  (14 logits)
   torch.sigmoid()  →  14 probabilities in [0, 1]
```

### Why DenseNet-121?

- Widely used in medical imaging (CheXpert, CheXNet papers).
- Dense connections (each layer receives features from ALL previous layers) enable strong feature reuse with fewer parameters.
- Pretrained ImageNet weights provide good low-level features (edges, textures) that transfer well to medical images.

### Training (notebook.ipynb)

- **Dataset**: De-identified chest X-rays with 14 thoracic condition labels (Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices), following the CheXpert label schema.
- **Split**: Multilabel stratified split (train / val / test) to preserve label distribution.
- **Loss**: BCEWithLogitsLoss with per-class positive weights (handles class imbalance).
- **Optimizer**: AdamW with ReduceLROnPlateau scheduler.
- **Augmentation**: RandomHorizontalFlip, RandomRotation(10), ColorJitter.
- **Training**: Mixed precision (torch.amp) when GPU is available, early stopping.
- **Checkpoint**: the best model is saved as `models/best_model.pth` (not versioned in git; can be replaced by any compatible checkpoint pointed to by `MODEL_PATH`).

### Model card (intended use, limitations, ethics)

The following aligns with the spirit of [Mitchell et al., 2019 — "Model Cards for Model Reporting"](https://arxiv.org/abs/1810.03993).

**Model details**

| Field        | Value                                                                 |
| ------------ | --------------------------------------------------------------------- |
| Architecture | DenseNet-121 (pretrained ImageNet backbone + custom head)             |
| Head         | Dropout(0.3) → Linear(1024, 14)                                       |
| Output       | 14 independent probabilities (multilabel, sigmoid activation)        |
| Dataset      | De-identified multi-institution chest X-ray dataset (~150k+ images) with a CheXpert-style 14-label schema |
| Labels       | 14 — Atelectasis, Cardiomegaly, Consolidation, Edema, Enlarged Cardiomediastinum, Fracture, Lung Lesion, Lung Opacity, No Finding, Pleural Effusion, Pleural Other, Pneumonia, Pneumothorax, Support Devices |
| Framework    | PyTorch 2.x                                                           |
| License      | Educational / research use                                           |

**Intended use**

- **Primary**: Decision-support tool for radiology education and demonstration.
- **Users**: Clinicians, researchers, students exploring AI-assisted chest X-ray interpretation.
- **NOT intended for**: Clinical diagnosis, treatment decisions, or any real-world medical use without validation.

**Limitations**

- **Distribution**: Trained on a single aggregated dataset. Performance may degrade on X-rays from other institutions, scanners, or protocols.
- **No clinical validation**: The model has not been validated in a clinical trial or real-world setting.
- **Dataset bias**: The underlying hospital datasets may carry demographic and equipment biases that propagate to the model.
- **Threshold**: Default 0.5 is generic; per-class optimal thresholds were not tuned.
- **Uncertainty**: Point probabilities only; no calibration or confidence intervals.

**Metrics**

- Primary metric: AUROC (per-label and macro-average) on a stratified held-out test split from the same dataset. See `notebooks/notebook.ipynb` for full evaluation.

**Results (from notebook)**

Held-out test set (stratified split from the training dataset, never seen during training):

| Metric           | Value   |
| ---------------- | ------- |
| Test loss        | 0.5494  |
| Test F1-macro    | 0.6769  |
| Test Hamming     | 0.1606  |
| Classification report (macro avg) | precision 0.60, recall 0.84, F1 0.68 |

Per-label AUROC and full classification report are in `notebooks/notebook.ipynb`.

**Explainability**

- **Grad-CAM** heatmaps are generated for the top predicted class (see [§ 4 Grad-CAM](#4-grad-cam) and `api/gradcam.py`).

**Ethical considerations**

- AI-assisted radiology must be used alongside — not instead of — trained clinicians.
- Patients should be informed when AI is used as part of their care.
- Model outputs carry the biases present in the training data.

---

## 3. API architecture

### Framework: FastAPI

FastAPI is a modern Python web framework that provides:
- Automatic request/response validation via Pydantic models.
- Auto-generated API docs (Swagger UI at /docs).
- Async support (though our endpoints are sync since PyTorch inference is CPU-bound).
- Easy file upload handling.

### Endpoints

#### `GET /health`
Returns model and RAG status. The frontend calls this on page load.

#### `POST /predict`
1. Read uploaded image → PIL Image (RGB).
2. Apply eval_transforms: Resize(224) → ToTensor → Normalize.
3. Forward pass through the model → 14 logits → sigmoid → 14 probabilities.
4. Compare each probability to the threshold → positive/negative.
5. Compute Grad-CAM for the top positive class.
6. Return predictions + Grad-CAM as JSON.

#### `POST /full-report`
Same as /predict, plus:
1. Parse clinical context (JSON string from form field).
2. Build an LLM prompt with predictions + context + RAG chunks.
3. Call OpenAI Chat API → get report text.
4. Return predictions + Grad-CAM + report + disclaimer.

#### `POST /report`
Like /full-report but without image upload. Takes predictions as JSON input and generates only the report.

#### `POST /chat`
Stateless conversational endpoint:
1. Client sends: new message + predictions + report + clinical context + conversation history.
2. Server builds the full message list for OpenAI (system prompt + context + history + user message).
3. RAG chunks are retrieved and injected as context.
4. Returns the assistant's reply.

### CORS

The frontend (localhost:5173 in dev, or a different domain in prod) makes requests to the API. Browsers block cross-origin requests by default. CORSMiddleware tells the browser to allow them.

### Model loading

The model is loaded once at import time (when uvicorn starts). This avoids loading the ~100MB model on every request. If loading fails, the error is stored and reported via /health.

---

## 4. Grad-CAM

### What it shows

A heatmap overlay on the original X-ray, where:
- **Red/orange** = regions that strongly influenced the prediction (the model "looked here").
- **Blue/green** = regions with little influence.

### Algorithm

1. **Hook into the last convolutional layer** (`model.backbone.features`).
   - DenseNet-121's features block outputs (batch, 1024, 7, 7) feature maps.
2. **Forward pass**: compute logits normally. The hook saves the activations.
3. **Backward pass**: backpropagate the target class score. The hook saves the gradients.
4. **Weights**: global-average-pool the gradients → one importance weight per channel (1024 weights).
5. **Weighted sum**: multiply each feature map by its weight and sum → (7, 7) heatmap.
6. **ReLU**: remove negative values (we only want positive influence).
7. **Normalize** to [0, 1].
8. **Resize** the 7×7 heatmap to the original image size.
9. **Colormap** (jet): blue → green → yellow → red.
10. **Blend** with the original image (45% heatmap + 55% original).

### Class selection

`pick_gradcam_class()` chooses which class to visualize:
- If any class is above the threshold → pick the highest-probability positive.
- If no class is positive → pick the overall highest probability.

---

## 5. RAG (Retrieval-Augmented Generation)

### Why RAG?

The LLM (GPT-4.1-mini) has general medical knowledge, but we want it to reference our specific pathology protocols. RAG injects relevant excerpts from our document corpus into the prompt, so the LLM can cite specific guidelines.

### How it works

```
pathologies/
├── Atelectasis.pdf
├── Cardiomegaly.pdf
├── Consolidation.txt
├── ...
```

**At startup (build_rag_index):**
1. Read all .txt and .pdf files from `pathologies/`.
2. Split each document into overlapping chunks of ~600 characters.
3. Embed each chunk using OpenAI's `text-embedding-3-small` → a 1536-dim vector.
4. Store chunks and embeddings in memory.

**At query time (get_rag_context):**
1. Embed the query (e.g., predictions + clinical context).
2. Compute cosine similarity between the query embedding and all chunk embeddings.
3. Return the top 5 most similar chunks.
4. These chunks are added to the LLM prompt as "Relevant excerpts from protocol/pathology documents."

### Cosine similarity

```
cosine_sim(a, b) = dot(a, b) / (|a| × |b|)
```

Values range from -1 (opposite) to 1 (identical direction). We normalize all vectors to unit length, so similarity = dot product.

---

## 6. Frontend

### Stack

- **React 19** + **TypeScript** — component-based UI with type safety.
- **Tailwind CSS v4** — utility-first CSS (no separate CSS files needed).
- **Vite** — fast build tool with hot module replacement (HMR).
- **lucide-react** — icon library.
- **react-markdown** — render Markdown from the LLM report.

### Components

| Component            | Purpose                                                  |
|----------------------|----------------------------------------------------------|
| `Layout`             | Navbar (with API status indicator) + footer              |
| `ImageUpload`        | Drag & drop / browse file input with image preview       |
| `ClinicalContextForm`| Collapsible form for age, sex, notes                     |
| `PredictionsTable`   | Sorted probability bars with alert icons                 |
| `GradCamViewer`      | Toggle between original image and Grad-CAM overlay       |
| `ReportDisplay`      | Rendered Markdown report + disclaimer banner             |
| `ChatPanel`          | Chat interface with message bubbles and loading state    |

### API client

`src/api/client.ts` wraps all fetch calls:
- `fetchHealth()` → GET /api/health
- `fetchPredict(file)` → POST /api/predict (FormData)
- `fetchFullReport(file, ctx)` → POST /api/full-report (FormData)
- `fetchChat(payload)` → POST /api/chat (JSON)

### Proxy

In development, Vite proxies `/api/*` → `http://localhost:8000/*` (configured in `vite.config.ts`). In production (Docker), nginx does the same proxy.

---

## 7. Docker & deployment

### Architecture

```
┌──────────────────────────────┐
│         User's browser        │
│   http://localhost:3000       │
└──────────────┬───────────────┘
               │
        ┌──────▼──────┐
        │   nginx      │  (frontend container)
        │   port 80    │
        │              │
        │  /api/*  ────┼──→  api:8000  (API container)
        │  /*      ────┼──→  static files (React build)
        └─────────────┘
```

### Dockerfile.api

- Base: `python:3.11-slim`
- Installs requirements.txt, copies API code + model + pathologies docs.
- Runs: `uvicorn api.main:app --host 0.0.0.0 --port 8000`

### Dockerfile.frontend

- Multi-stage build:
  - Stage 1 (`node:20-alpine`): `npm ci` + `npm run build` → produces `dist/`.
  - Stage 2 (`nginx:alpine`): copies `dist/` into nginx, adds custom `nginx.conf`.
- The nginx config proxies `/api/` to the API container and serves the SPA.

### docker-compose.yml

Defines two services (`api` and `frontend`), links them, loads `.env` for the API.

### Deployment options

1. **Local demo**: `docker compose up --build` — everything at localhost.
2. **Cloud (AWS/GCP/Azure)**: push Docker images to a registry (ECR, Docker Hub), deploy on ECS/Fargate, EC2, or Cloud Run.
3. **Hybrid**: frontend on GitHub Pages / Vercel, API on Render / Railway / Fly.io (set `VITE_API_URL` at build time).

For a step-by-step deployment procedure (e.g. Render, Railway, or AWS), see **[docs/DEPLOYMENT.md](DEPLOYMENT.md)**.

---

## 8. MLOps

This section summarizes the **MLOps** aspects of RadiantCare: how the model is served, how the environment is controlled, and how the system can be deployed and monitored. These practices are what make the project reproducible and production-ready in spirit, even when used as a portfolio or demo.

### Model serving

- The trained PyTorch model is **served via a REST API** (FastAPI). Clients send an image; the API runs preprocessing, inference, and optional Grad-CAM, then returns JSON.
- The model is **loaded once at startup** (no per-request load), and the same preprocessing (resize, normalize) used in training is applied at inference so behavior is consistent.
- Endpoints are versioned in responses (`model_version: "v1"`), so clients can track which model produced the output.
- The **model card** (intended use, limitations, metrics, ethics) is part of this guide; see [§ 2 Model (DenseNet-121) — Model card](#model-card-intended-use-limitations-ethics).

### Containerization

- **Docker** is used for both the API and the frontend:
  - **API**: `Dockerfile.api` — Python 3.11, dependencies, API code, model checkpoint, and pathology documents. Single process running uvicorn.
  - **Frontend**: `Dockerfile.frontend` — multi-stage build (Node for build, nginx for serving static assets and proxying `/api` to the API).
- **docker-compose** orchestrates both services and injects environment variables from `.env`. One command (`docker compose up --build`) runs the full stack locally.

Containerization ensures the same runtime (Python version, system libs, Node build) everywhere, reducing “works on my machine” issues and simplifying deployment to any host that runs Docker.

### Configuration and secrets

- **No secrets or environment-specific paths are hardcoded.** All of the following are read from the environment (typically via a `.env` file):
  - `MODEL_PATH` — path to the trained checkpoint
  - `PREDICTION_THRESHOLD`
  - `OPENAI_API_KEY`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE`
- `.env.example` documents required variables; the real `.env` is gitignored. In deployment, the host or orchestrator provides these variables.

### Observability

- A **health endpoint** (`GET /health`) reports:
  - Whether the model loaded successfully (and the error message if not)
  - Whether the RAG index is ready (and the error if not)
  - Device (CPU/CUDA), model version, number of labels
  - Number of RAG documents and chunks loaded
- **Inference latency** is measured per request (`time.perf_counter()` around the model forward pass) and returned in the response as `inference_time_ms`. This is displayed in the frontend next to the results.
- The frontend calls `/health` on load to show “API Ready” or “API Offline.”

### Reproducibility

- **Training**: The notebook defines data split, preprocessing, architecture, and training loop. The same `eval_transforms` and model architecture are replicated in the API so inference matches training conditions.
- **Dependencies**: `requirements.txt` pins Python packages; Docker freezes the base image and install step. Anyone can rebuild the API image and get the same environment.
- **Artifacts**: The model checkpoint (`best_model.pth`) is loaded from a configurable path; the same file produced by the notebook is used in the API.

### Deployment

- The application is **deployable as-is** using the Docker images and `docker-compose` (or by running the API and frontend images on a cloud provider). Deployment is intended for **demo or portfolio** use (e.g. to share a live link or take screenshots), not as a permanent production service.
- The exact steps (choice of host, registry, env vars, domain) are documented in **[docs/DEPLOYMENT.md](DEPLOYMENT.md)** so the procedure can be repeated or adapted (e.g. for interviews or future projects).
