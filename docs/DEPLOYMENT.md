# Deployment — RadiantCare

This document describes how to run and deploy RadiantCare using **Docker**. The app is intended for **demo or portfolio** use (e.g. to share a live link or capture screenshots), not as a long-term production service.

---

## Prerequisites

- **Docker** and **Docker Compose** installed ([Docker Desktop](https://docs.docker.com/desktop/install/) with WSL2 integration if you use WSL).
- A trained model checkpoint at `models/best_model.pth` (or set `MODEL_PATH` in `.env`).
- An **OpenAI API key** (for report generation, chat, and RAG embeddings).

### Before you build (important)

- **Model file**: The API image is built with `COPY models/ ./models/`. If `models/` is empty or does not contain `best_model.pth`, the container will start but the API will fail to load the model (reported on `/health`). Either place `best_model.pth` in `models/` before running `docker compose build`, or after deployment mount the model at runtime and set `MODEL_PATH` to that path inside the container.
- **RAG documents**: The image also copies `pathologies/`. If that folder is empty, RAG will report "No documents found" and reports/chat will run without RAG context. Add at least one `.txt` or `.pdf` in `pathologies/` before building if you want RAG.

To obtain a compatible checkpoint, either:

- train a model with `notebooks/notebook.ipynb` and export the best weights as `models/best_model.pth`, or  
- provide your own `.pth` file that matches the `ChestXrayClassifier` architecture and update `MODEL_PATH` accordingly.

---

## 1. Local run with Docker (recommended)

From the project root, with a `.env` file present (copy from `.env.example`, add `OPENAI_API_KEY`):

```bash
docker compose up --build
```

- **Frontend**: http://localhost:3000  
- **API**: http://localhost:8000  
- **API docs**: http://localhost:8000/docs  

The frontend container (nginx) proxies `/api/*` to the API container. No need to run Python or Node locally.

To stop:

```bash
docker compose down
```

To force a full rebuild of the API image (e.g. after changing `api/` or `requirements.txt`):

```bash
docker compose build --no-cache api
docker compose up
```

---

## 2. Deploying to a cloud host

To run RadiantCare on a remote server (Render, Railway, Fly.io, AWS, etc.), use the same Docker images.

### 2.1 Build and push images

Build the images and push them to a registry (Docker Hub, GitHub Container Registry, AWS ECR, etc.):

```bash
# Example: tag and push to Docker Hub
docker compose build
docker tag radiantcare-api your-username/radiantcare-api:latest
docker tag radiantcare-frontend your-username/radiantcare-frontend:latest
docker push your-username/radiantcare-api:latest
docker push your-username/radiantcare-frontend:latest
```

Replace `your-username` and the registry URL according to your provider.

### 2.2 Run the API

On your host (VM, container service, etc.):

- Run the API container with **env vars** set: `MODEL_PATH`, `OPENAI_API_KEY`, `PREDICTION_THRESHOLD`, `OPENAI_MODEL`, `OPENAI_TEMPERATURE` (see `.env.example`).
- Ensure the **model file** (`best_model.pth`) and optionally the **pathologies** folder are available (e.g. mounted or baked into the image).
- Expose the API on a public URL (e.g. `https://radiantcare-api.yourdomain.com`).

### 2.3 Run the frontend

- **Option A — Same host**: Run the frontend container and configure nginx (or your reverse proxy) so that `/api` is proxied to the API URL. The frontend is built with relative `/api` calls, so as long as the same origin serves both the SPA and proxies `/api` to the API, it works.
- **Option B — Separate host**: If the frontend is served from another domain (e.g. Vercel, GitHub Pages), set at **build time** the API base URL (e.g. `VITE_API_URL`) so the frontend calls the correct API, and enable CORS on the API for that origin.

### 2.4 Optional: domain and HTTPS

- Attach a custom domain to the frontend and/or API.
- Use the provider's SSL/TLS (e.g. Let's Encrypt) so the app is served over HTTPS.

---

## 3. Procedure checklist (to fill after your deployment)

Once you have deployed, you can document your exact steps here for reproducibility:

1. **Registry**: Where you pushed the images (e.g. Docker Hub).
2. **API host**: Where the API runs (service name, region, URL).
3. **Model**: How the checkpoint is provided (baked in image or volume) and `MODEL_PATH` if different.
4. **Frontend host**: Where the frontend runs and how `/api` is proxied (or the `VITE_API_URL` used).
5. **Env vars**: Which variables you set on the API service (including `CORS_ORIGINS` if you restrict to a domain).
6. **Domain/HTTPS**: Custom domain and SSL if configured.

---

## Reference: run without Docker (local development)

For development without Docker, see the **README** → **Quick start (local development)** (venv + uvicorn for the API, `npm run dev` for the frontend). Docker is not required for that workflow.
