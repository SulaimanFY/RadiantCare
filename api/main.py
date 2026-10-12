"""
RadiantCare API — main application file.

This FastAPI application exposes endpoints that let a frontend:
  1. Upload a chest X-ray image and get predictions for 14 pathologies (/predict)
  2. Get predictions + an AI-written report in one call (/full-report)
  3. Generate a report from existing predictions (/report)
  4. Chat with an LLM assistant about the case (/chat)

Every prediction response also includes a Grad-CAM heatmap showing which
regions of the image most influenced the model's top prediction.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from __future__ import annotations          # allow "X | Y" type hints on older Python

import io                                   # for reading uploaded file bytes in memory
import json                                 # for parsing clinical context JSON string
import logging                              # for logging warnings (e.g. a failed Grad-CAM)
import os                                   # for reading environment variables and file paths
import time                                 # for measuring inference latency
from typing import List, Literal, Optional  # type annotations

# Module-level logger: messages appear in the uvicorn console.
logger = logging.getLogger(__name__)

import numpy as np                          # numerical arrays (model probabilities, thresholds)
import torch                                # PyTorch: tensors, inference, GPU support
import torch.nn as nn                       # neural network layers (Linear, Dropout, etc.)
from dotenv import load_dotenv              # reads .env file into os.environ
from fastapi import (                       # FastAPI framework
    FastAPI,
    File,                                   # marks a parameter as file upload
    Form,                                   # marks a parameter as form field
    HTTPException,                          # raises HTTP errors (400, 503, etc.)
    UploadFile,                             # type for uploaded files
)
from fastapi.middleware.cors import CORSMiddleware  # allows cross-origin requests (frontend → API)
from openai import OpenAI                   # OpenAI Python client for Chat API
from PIL import Image                       # image loading and manipulation
from pydantic import BaseModel              # data validation and serialization for request/response
from torchvision import models, transforms  # pretrained models (DenseNet) and image transforms

from api import rag                         # our RAG module: loads pathology docs, embeds, retrieves
from api.gradcam import (                   # our Grad-CAM module: heatmap generation
    generate_gradcam_b64,                   # compute Grad-CAM and return base64 image
    pick_gradcam_class,                     # choose which class to visualize
)


# ---------------------------------------------------------------------------
# Startup: load .env and build RAG index
# ---------------------------------------------------------------------------

# load_dotenv() reads key=value pairs from the .env file at the project root
# and sets them as environment variables so os.getenv() can find them.
# This must happen BEFORE we use any env vars (MODEL_PATH, OPENAI_API_KEY, etc.)
load_dotenv()

# At startup, load all documents from pathologies/ folder, split them into chunks,
# embed each chunk using OpenAI embeddings, and store everything in memory.
# Later, when a user asks a question, we search these embeddings for relevant context.
# This is called Retrieval-Augmented Generation (RAG).
rag.build_rag_index()


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

PROJECT_NAME = "RadiantCare API"
MODEL_VERSION = "v1"  # should bump this when retrain and deploy a new model

# BASE_DIR = the project root (one level up from api/)
# __file__           → /path/to/RadiantCare/api/main.py
# dirname(__file__)  → /path/to/RadiantCare/api/
# dirname(dirname()) → /path/to/RadiantCare/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the .pth model checkpoint; overridable via MODEL_PATH in .env
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# The 14 pathology labels, in the EXACT same order as the training notebook.
# The model outputs 14 logits; logit[0] corresponds to "Atelectasis", etc.
# If you change the order here, predictions will be wrong.
LABEL_COLS: List[str] = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]

# DenseNet-121 expects 224×224 images, normalized with ImageNet stats
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]  # per-channel mean from ImageNet
IMAGENET_STD = [0.229, 0.224, 0.225]   # per-channel std from ImageNet

# Defaults for OpenAI Chat API; overridable in .env
OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"   # which chat model to use
OPENAI_TEMPERATURE_DEFAULT = 0.2         # 0 = deterministic, 1 = creative


# ---------------------------------------------------------------------------
# Pydantic models (request/response schemas)
# ---------------------------------------------------------------------------
# Pydantic models define the shape of JSON data going in and out of the API.
# FastAPI automatically validates incoming data against these schemas,
# returns clear errors if fields are missing/wrong, and generates API docs.

class Prediction(BaseModel):
    """One row of model output: pathology name, probability, and positive/negative flag."""
    label: str          # e.g. "Pneumonia"
    probability: float  # e.g. 0.87 (between 0 and 1)
    positive: bool      # True if probability >= threshold (e.g. 0.5)


class PredictResponse(BaseModel):
    """JSON returned by POST /predict."""
    model_version: str                      # e.g. "v1"
    predictions: List[Prediction]           # one per pathology (14 items)
    inference_time_ms: Optional[float] = None  # how long inference took (milliseconds)
    grad_cam_image: Optional[str] = None    # base64-encoded PNG of the heatmap overlay
    grad_cam_label: Optional[str] = None    # which pathology the heatmap shows (e.g. "Pneumonia")


class ClinicalContext(BaseModel):
    """Optional info the clinician can provide alongside the image."""
    free_text: Optional[str] = None    # e.g. "Persistent cough for 3 weeks"
    age: Optional[int] = None          # e.g. 58
    sex: Optional[str] = None          # "male", "female", or "other"
    other_info: Optional[str] = None   # anything else


class ReportRequest(BaseModel):
    """JSON body for POST /report (predictions-only report, no image upload)."""
    predictions: List[Prediction]
    clinical_context: Optional[ClinicalContext] = None


class ReportResponse(BaseModel):
    """JSON returned by POST /report."""
    report: str       # LLM-generated text
    disclaimer: str   # legal/safety disclaimer


class ChatMessage(BaseModel):
    """One message in a conversation turn (used for chat history)."""
    role: Literal["user", "assistant"]  # who said it
    content: str                        # what was said


class ChatRequest(BaseModel):
    """JSON body for POST /chat."""
    message: str                                       # the new user question
    predictions: Optional[List[Prediction]] = None     # current predictions (for context)
    report: Optional[str] = None                       # current report (for context)
    clinical_context: Optional[ClinicalContext] = None # clinical info (for context)
    history: Optional[List[ChatMessage]] = None        # previous messages in this conversation


class ChatResponse(BaseModel):
    """JSON returned by POST /chat."""
    answer: str       # LLM reply
    disclaimer: str   # legal/safety disclaimer


class FullReportResponse(BaseModel):
    """JSON returned by POST /full-report (predictions + report + Grad-CAM in one call)."""
    predictions: List[Prediction]
    report: str
    disclaimer: str
    inference_time_ms: Optional[float] = None  # model inference time (milliseconds)
    grad_cam_image: Optional[str] = None
    grad_cam_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------
# This class MUST match exactly what was used during training (in notebook.ipynb).
# If one change anything here (layer names, sizes), loading the checkpoint will fail.

class ChestXrayClassifier(nn.Module):
    """
    DenseNet-121 fine-tuned for multilabel chest X-ray classification.

    Architecture:
      1. backbone = DenseNet-121 pretrained on ImageNet (feature extractor)
         - The original classifier head is replaced by nn.Identity()
           so backbone(x) outputs a 1024-dimensional feature vector.
      2. head = Sequential(Dropout → Linear)
         - Dropout(0.3) for regularization (randomly zeros 30% of features during training)
         - Linear(1024 → 14) maps features to one logit per pathology
         - No sigmoid here: we apply sigmoid later (BCEWithLogitsLoss during training,
           torch.sigmoid during inference)
    """

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        # Load DenseNet-121 with pretrained ImageNet weights
        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)

        # backbone.classifier is originally Linear(1024, 1000) for ImageNet's 1000 classes
        # We save the input size (1024) then replace the classifier with Identity()
        # so that backbone(x) returns the 1024-dim feature vector instead of 1000 logits
        in_features = backbone.classifier.in_features  # = 1024
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        # Our custom classification head for 14 chest pathologies
        self.head = nn.Sequential(
            nn.Dropout(dropout),                   # randomly drop 30% of features (regularization)
            nn.Linear(in_features, num_classes),   # 1024 → 14 (one logit per pathology)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 3, 224, 224) — batch of RGB images
        features = self.backbone(x)   # → (batch_size, 1024)
        return self.head(features)    # → (batch_size, 14) raw logits (not probabilities yet)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

# Automatically use GPU if available (much faster), otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# These transforms are applied to every image before feeding it to the model.
# They must match what was used during training (same size, same normalization).
# No data augmentation here because we are in evaluation mode, not training.
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),    # resize to 224×224
    transforms.ToTensor(),                      # PIL Image → Tensor, scales pixels to [0, 1]
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),  # normalize with ImageNet stats
])


def load_model() -> ChestXrayClassifier:
    """
    Load the trained model from a .pth file and return it ready for inference.

    Steps:
      1. Read MODEL_PATH from .env (or use default models/best_model.pth)
      2. Create an empty ChestXrayClassifier with the right architecture
      3. Load the saved weights (state_dict) from disk
      4. Copy those weights into the model
      5. Move model to GPU/CPU and set to eval mode (disables Dropout)
    """
    # Step 1: get the path to the .pth file
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    # Step 2: create the model (random weights at this point)
    model = ChestXrayClassifier(num_classes=len(LABEL_COLS))

    # Step 3: load the checkpoint from disk
    # map_location=device ensures weights go to the right device (CPU or GPU)
    state = torch.load(model_path, map_location=device)

    # The checkpoint can be either:
    #   a) a raw state_dict: {"backbone.features.conv0.weight": tensor, ...}
    #   b) a dict with metadata: {"state_dict": {...}, "epoch": 10, "loss": 0.3}
    # We handle both cases:
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # Step 4: load the saved weights into our model
    # strict=True (default): will error if any key is missing or unexpected
    model.load_state_dict(state)

    # Step 5: move to device and set eval mode
    model.to(device)
    model.eval()  # turns off Dropout and BatchNorm training behavior
    return model


# Load the model immediately when the server starts (import time).
# This way the model is already in memory when the first request arrives.
# If loading fails (file not found, architecture mismatch, etc.), we catch
# the error and store it — the /health endpoint will report it.
try:
    model = load_model()
    MODEL_LOADED = True
except Exception as exc:
    model = None  # type: ignore[assignment]
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(exc)
else:
    MODEL_LOAD_ERROR = None


# ---------------------------------------------------------------------------
# Create the FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title=PROJECT_NAME)

# CORS (Cross-Origin Resource Sharing) middleware:
# The frontend runs on localhost:5173 (Vite dev server) while the API runs on
# localhost:8000. Browsers block cross-origin requests by default for security.
# This middleware tells the browser which origins may call the API.
# CORS_ORIGINS in .env is a comma-separated list of allowed origins; the default
# "*" allows any origin (fine locally; restrict it to your frontend in production).
_cors_origins = os.getenv("CORS_ORIGINS", "*").strip()
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],       # allow GET, POST, PUT, DELETE, etc.
    allow_headers=["*"],       # allow any HTTP headers
)


# ---------------------------------------------------------------------------
# Disclaimer (shown with every report/chat response)
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This system is a research / educational tool and does not necessarily provide a reliable medical "
    "diagnosis. Results must be interpreted by a qualified clinician."
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _read_image_file(file: UploadFile) -> Image.Image:
    """
    Read an uploaded file and return a PIL Image in RGB mode.
    Raises HTTP 400 if no file or file is not a valid image.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        contents = file.file.read()                         # read raw bytes from upload
        image = Image.open(io.BytesIO(contents))            # parse bytes as an image
        return image.convert("RGB")                         # ensure 3 channels (some X-rays are grayscale)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")


def _run_inference(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """
    Run the model on a single PIL image.

    Returns:
      - probs: numpy array of shape (14,) with values in [0, 1]
      - threshold: float from .env or default 0.5
      - inference_ms: how long inference took in milliseconds
    """
    # Apply the same transforms used during training: resize, to tensor, normalize
    # (3, 224, 224) → add a batch dimension → (1, 3, 224, 224) → move to GPU if available
    tensor = eval_transforms(image).unsqueeze(0).to(device)

    # Measure inference time (preprocessing is excluded, only the model forward pass)
    start = time.perf_counter()

    # torch.no_grad() disables gradient computation — faster and uses less memory
    # (we don't need gradients during inference, only during training)
    with torch.no_grad():
        logits = model(tensor)                           # shape: (1, 14) raw logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]   # shape: (14,) probabilities in [0,1]

    inference_ms = (time.perf_counter() - start) * 1000  # convert seconds → milliseconds

    # Read the threshold from .env (default 0.5): above threshold = "positive" finding
    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    return probs, threshold, inference_ms


def _build_predictions(probs: np.ndarray, threshold: float) -> List[Prediction]:
    """Convert the raw probability array into a list of Prediction objects."""
    return [
        Prediction(
            label=label,                          # pathology name
            probability=float(prob),              # probability value
            positive=bool(prob >= threshold),      # is it above the decision threshold?
        )
        for label, prob in zip(LABEL_COLS, probs)  # pair each label with its probability
    ]


def _compute_gradcam(
    image: Image.Image, probs: np.ndarray, threshold: float
) -> tuple[str | None, str | None]:
    """
    Generate a Grad-CAM heatmap for the most relevant pathology.

    1. pick_gradcam_class: choose which class to visualize
       (highest-probability positive, or overall highest if none are positive)
    2. generate_gradcam_b64: compute Grad-CAM and return the overlay as base64

    Returns (base64_image_string, label_name) or (None, None) on any error.
    """
    try:
        class_idx = pick_gradcam_class(probs, threshold)
        if class_idx is None:
            return None, None
        # We need a NEW tensor (not the one from inference) because Grad-CAM needs gradients
        tensor = eval_transforms(image).unsqueeze(0)
        b64 = generate_gradcam_b64(model, tensor, image, class_idx, device)
        return b64, LABEL_COLS[class_idx]
    except Exception as e:
        # The heatmap is optional: log the error and return the predictions without it.
        logger.warning("Grad-CAM failed: %s", e, exc_info=True)
        return None, None


# ---------------------------------------------------------------------------
# LLM helpers (OpenAI Chat API)
# ---------------------------------------------------------------------------

def _format_predictions_for_prompt(predictions: List[Prediction]) -> str:
    """Turn the predictions list into a readable text block for the LLM prompt."""
    lines = []
    for p in predictions:
        status = "positive" if p.positive else "negative"
        lines.append(f"- {p.label}: {status} (probability={p.probability:.2f})")
    return "\n".join(lines)


def _format_clinical_context_for_prompt(ctx: Optional[ClinicalContext]) -> str:
    """Turn clinical context fields into a readable text block for the LLM prompt."""
    if ctx is None:
        return "No clinical context provided."
    parts = []
    if ctx.age is not None:
        parts.append(f"Age: {ctx.age}")
    if ctx.sex:
        parts.append(f"Sex: {ctx.sex}")
    if ctx.other_info:
        parts.append(f"Other structured info: {ctx.other_info}")
    if ctx.free_text:
        parts.append(f"Free text: {ctx.free_text}")
    if not parts:
        return "No clinical context provided."
    return "\n".join(parts)


def _get_openai_client() -> OpenAI:
    """Create an OpenAI client using the API key from .env. Raises 503 if key is missing."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not configured on the server.",
        )
    return OpenAI(api_key=api_key)


def _call_openai_chat(messages: list[dict[str, str]]) -> str:
    """
    Send a conversation (list of messages) to the OpenAI Chat API and return the reply.

    The messages list follows the OpenAI format:
      [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]

    Raises HTTP 502 if the API call fails (network error, quota exceeded, etc.).
    """
    client = _get_openai_client()
    model_name = os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)
    temperature = float(os.getenv("OPENAI_TEMPERATURE", str(OPENAI_TEMPERATURE_DEFAULT)))

    try:
        response = client.chat.completions.create(
            model=model_name,          # e.g. "gpt-4.1-mini"
            messages=messages,         # the full conversation
            temperature=temperature,   # controls randomness (0 = deterministic, 1 = creative)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502, detail=f"Error while calling OpenAI API: {exc}"
        )

    # response.choices[0].message.content is the assistant's reply text
    return response.choices[0].message.content or ""


def _generate_report_text(
    predictions: List[Prediction],
    clinical_context: Optional[ClinicalContext] = None,
) -> str:
    """
    Build a prompt from predictions + clinical context + RAG, send to OpenAI, return report.

    Steps:
      1. Format predictions and clinical context as text
      2. Query the RAG index to find relevant document chunks (from pathologies/)
      3. Build a system prompt (instructions for the LLM) and a user prompt (data)
      4. Call OpenAI Chat API and return the generated report
    """
    # Step 1: format our data as text
    predictions_text = _format_predictions_for_prompt(predictions)
    context_text = _format_clinical_context_for_prompt(clinical_context)

    # Step 2: RAG — find relevant chunks from pathology documents
    # We combine predictions + context into a query string, then search for similar chunks
    rag_query = f"Chest X-ray findings:\n{predictions_text}\n\nClinical context:\n{context_text}"
    rag_context = rag.get_rag_context(rag_query)  # returns "" if RAG is not ready

    # Step 3: build the prompt messages
    # System message: tells the LLM who it is and how to respond
    system_prompt = (
        "You are an assistant for radiologists and clinicians who use you as a "
        "decision-support partner. They are the treating professionals; give them "
        "direct, actionable input. You receive chest X-ray model predictions and "
        "optional clinical context. Do the following:\n"
        "- Summarize the main findings clearly.\n"
        "- Highlight potential red flags and differentials.\n"
        "- Suggest concrete next steps (further imaging, workup, treatment options) "
        "as appropriate.\n"
        "Format the report in Markdown with three short sections, each starting with a "
        "### heading: Findings, Red flags and differentials, Next steps. Use one to three "
        "sentences or short - bullet points per section, about 120 words in total, in "
        "clear clinical language."
    )

    # User message: the actual data (predictions, context, RAG excerpts)
    user_content = (
        f"Model predictions:\n{predictions_text}\n\n"
        f"Clinical context:\n{context_text}\n\n"
    )
    if rag_context:
        user_content += (
            f"Relevant excerpts from protocol/pathology documents:\n{rag_context}\n\n"
        )
    user_content += "Write a short report for the clinician."

    # Step 4: send to OpenAI and return the reply
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    return _call_openai_chat(messages)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root() -> dict:
    """Root endpoint: returns basic API info. Useful for quick "is it alive?" checks."""
    return {
        "name": PROJECT_NAME,
        "model_version": MODEL_VERSION,
        "num_labels": len(LABEL_COLS),
        "labels": LABEL_COLS,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health() -> dict:
    """
    Health check: reports whether the model and RAG index loaded successfully.
    The frontend calls this on page load to show "API Ready" or "API Offline".
    """
    return {
        "model_loaded": MODEL_LOADED,
        "model_version": MODEL_VERSION,
        "device": str(device),
        "num_labels": len(LABEL_COLS),
        "error": MODEL_LOAD_ERROR,
        "rag_ready": rag.is_rag_ready(),
        "rag_documents": rag.get_num_documents(),
        "rag_chunks": rag.get_num_chunks(),
        "rag_error": rag.get_rag_error(),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(file: UploadFile = File(...)) -> PredictResponse:
    """
    Upload a chest X-ray image → get predictions + Grad-CAM heatmap.

    Flow:
      1. Read the uploaded image file
      2. Run inference: image → 14 probabilities
      3. Build Prediction objects (label, probability, positive/negative)
      4. Compute Grad-CAM for the top finding
      5. Return everything as JSON
    """
    if not MODEL_LOADED or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    image = _read_image_file(file)                         # Step 1
    probs, threshold, inference_ms = _run_inference(image)  # Step 2
    predictions = _build_predictions(probs, threshold)      # Step 3
    cam_b64, cam_label = _compute_gradcam(image, probs, threshold)  # Step 4

    return PredictResponse(                                 # Step 5
        model_version=MODEL_VERSION,
        predictions=predictions,
        inference_time_ms=round(inference_ms, 1),
        grad_cam_image=cam_b64,
        grad_cam_label=cam_label,
    )


@app.post("/full-report", response_model=FullReportResponse)
def full_report(
    file: UploadFile = File(...),
    clinical_context: Optional[str] = Form(None),
) -> FullReportResponse:
    """
    Upload image + optional clinical context → predictions + AI report + Grad-CAM.

    This is the "one-click" endpoint: the frontend sends the image and optional
    clinical context (as a JSON string in a form field), and gets back everything
    needed to display results (predictions, heatmap, report, disclaimer).

    Flow:
      1. Read image, run inference, build predictions, compute Grad-CAM
      2. Parse the clinical context JSON (if provided)
      3. Generate the report (predictions + context + RAG → LLM)
      4. Return everything
    """
    if not MODEL_LOADED or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    # Step 1: same as /predict
    image = _read_image_file(file)
    probs, threshold, inference_ms = _run_inference(image)
    predictions = _build_predictions(probs, threshold)
    cam_b64, cam_label = _compute_gradcam(image, probs, threshold)

    # Step 2: parse clinical context
    # The frontend sends it as a JSON string in a form field (because the request
    # is multipart/form-data due to the file upload, not application/json)
    ctx: Optional[ClinicalContext] = None
    if clinical_context and clinical_context.strip():
        try:
            data = json.loads(clinical_context)
            ctx = ClinicalContext(
                free_text=data.get("free_text"),
                age=data.get("age"),
                sex=data.get("sex"),
                other_info=data.get("other_info"),
            )
        except Exception:
            pass  # if parsing fails, just proceed without context

    # Step 3: generate report via LLM
    report_text = _generate_report_text(predictions, ctx)

    # Step 4: return everything
    return FullReportResponse(
        predictions=predictions,
        report=report_text,
        disclaimer=DISCLAIMER,
        inference_time_ms=round(inference_ms, 1),
        grad_cam_image=cam_b64,
        grad_cam_label=cam_label,
    )


@app.post("/report", response_model=ReportResponse)
def generate_report(payload: ReportRequest) -> ReportResponse:
    """
    Generate a report from existing predictions (no image upload needed).

    Use case: the frontend already called /predict and has predictions;
    now the user wants a report without re-uploading the image.
    The request body is JSON (not form-data) since there's no file.
    """
    if not payload.predictions:
        raise HTTPException(
            status_code=400, detail="At least one prediction is required."
        )
    report_text = _generate_report_text(payload.predictions, payload.clinical_context)
    return ReportResponse(report=report_text, disclaimer=DISCLAIMER)


@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(payload: ChatRequest) -> ChatResponse:
    """
    Free-form chat with the AI assistant about the current case.

    The server is STATELESS: it does not remember previous messages.
    The frontend must send the full conversation history + context each time.

    Flow:
      1. Build a RAG query from the user message (+ predictions if available)
      2. Retrieve relevant document chunks
      3. Build the messages list for OpenAI:
         - system prompt (who the LLM is)
         - context block (predictions, clinical context, previous report, RAG)
         - conversation history (previous user/assistant messages)
         - the new user message
      4. Call OpenAI and return the reply
    """
    if not payload.message:
        raise HTTPException(status_code=400, detail="message is required.")

    # Step 1-2: RAG retrieval
    rag_query = payload.message
    if payload.predictions:
        # Prepend predictions to the query so RAG can find more relevant chunks
        rag_query = _format_predictions_for_prompt(payload.predictions) + "\n\n" + rag_query
    rag_context = rag.get_rag_context(rag_query)

    # Step 3: build the messages list
    system_prompt = (
        "You are an AI assistant for clinicians interpreting chest X-ray model "
        "outputs. They use you as a partner. Provide clear differential diagnoses, "
        "next steps, and treatment or workup recommendations as appropriate. "
        "Use direct clinical language; the clinician will make the final decision. "
        "Answer the question first, in one or two sentences with the key point in "
        "**bold**, then add detail only if it helps. Format your reply in Markdown: "
        "use ### for short section headings, - for bullet lists, and separate "
        "paragraphs with a blank line. Keep it under about 150 words unless the "
        "clinician asks for more."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Add context as a second system message (predictions, clinical context, report, RAG)
    context_parts: list[str] = []
    if payload.predictions:
        context_parts.append(
            "Model predictions:\n" + _format_predictions_for_prompt(payload.predictions)
        )
    if payload.clinical_context:
        context_parts.append(
            "Clinical context:\n" + _format_clinical_context_for_prompt(payload.clinical_context)
        )
    if payload.report:
        context_parts.append(f"Previous report:\n{payload.report}")
    if rag_context:
        context_parts.append(
            "Relevant excerpts from protocol/pathology documents:\n" + rag_context
        )
    if context_parts:
        messages.append({"role": "system", "content": "\n\n".join(context_parts)})

    # Add conversation history (previous turns)
    if payload.history:
        for turn in payload.history:
            messages.append({"role": turn.role, "content": turn.content})

    # Add the new user message
    messages.append({"role": "user", "content": payload.message})

    # Step 4: call OpenAI and return
    answer = _call_openai_chat(messages)
    return ChatResponse(answer=answer, disclaimer=DISCLAIMER)
