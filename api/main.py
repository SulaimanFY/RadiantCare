"""
RadiantCare API — main application file.

FastAPI app that exposes:
  /predict      — image → 14-class predictions + Grad-CAM
  /full-report  — image + context → predictions + Grad-CAM + LLM report
  /report       — predictions → LLM report (no image)
  /chat         — follow-up Q&A with the LLM
  /health       — model & RAG status
"""

from __future__ import annotations

import io
import json
import logging
import os
import time
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from dotenv import load_dotenv
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel
from torchvision import models, transforms

from api import rag
from api.gradcam import generate_gradcam_b64, pick_gradcam_class


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

load_dotenv()
rag.build_rag_index()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PROJECT_NAME = "RadiantCare API"
MODEL_VERSION = "v1"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pth")

# Must match training order exactly — logit[i] ↔ LABEL_COLS[i]
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

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

OPENAI_MODEL_DEFAULT = "gpt-4.1-mini"
OPENAI_TEMPERATURE_DEFAULT = 0.2


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class Prediction(BaseModel):
    label: str
    probability: float
    positive: bool


class PredictResponse(BaseModel):
    model_version: str
    predictions: List[Prediction]
    inference_time_ms: Optional[float] = None
    grad_cam_image: Optional[str] = None
    grad_cam_label: Optional[str] = None


class ClinicalContext(BaseModel):
    free_text: Optional[str] = None
    age: Optional[int] = None
    sex: Optional[str] = None
    other_info: Optional[str] = None


class ReportRequest(BaseModel):
    predictions: List[Prediction]
    clinical_context: Optional[ClinicalContext] = None


class ReportResponse(BaseModel):
    report: str
    disclaimer: str


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str
    predictions: Optional[List[Prediction]] = None
    report: Optional[str] = None
    clinical_context: Optional[ClinicalContext] = None
    history: Optional[List[ChatMessage]] = None


class ChatResponse(BaseModel):
    answer: str
    disclaimer: str


class FullReportResponse(BaseModel):
    predictions: List[Prediction]
    report: str
    disclaimer: str
    inference_time_ms: Optional[float] = None
    grad_cam_image: Optional[str] = None
    grad_cam_label: Optional[str] = None


# ---------------------------------------------------------------------------
# Model architecture — must match the training notebook exactly
# ---------------------------------------------------------------------------

class ChestXrayClassifier(nn.Module):
    """DenseNet-121 backbone (ImageNet pretrained) + Dropout → Linear(1024, 14)."""

    def __init__(self, num_classes: int, dropout: float = 0.3):
        super().__init__()

        backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        in_features = backbone.classifier.in_features  # 1024
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)   # (batch, 1024)
        return self.head(features)    # (batch, 14) raw logits


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same transforms as training (resize + normalize, no augmentation)
eval_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def load_model() -> ChestXrayClassifier:
    """Load the .pth checkpoint and return the model in eval mode on `device`."""
    model_path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at '{model_path}'")

    model = ChestXrayClassifier(num_classes=len(LABEL_COLS))

    state = torch.load(model_path, map_location=device)
    # Handle both raw state_dict and {state_dict: ..., epoch: ...} formats
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# Load at import time so the model is ready when the first request arrives
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
# FastAPI app + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title=PROJECT_NAME)

_cors_origins = os.getenv("CORS_ORIGINS", "*").strip()
allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] if _cors_origins != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Disclaimer
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "This system is a research / educational tool and does not necessarily provide a reliable medical "
    "diagnosis. Results must be interpreted by a qualified clinician."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_image_file(file: UploadFile) -> Image.Image:
    """Read an uploaded file → PIL RGB Image.  Raises 400 on failure."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        contents = file.file.read()
        image = Image.open(io.BytesIO(contents))
        return image.convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image file.")


def _run_inference(image: Image.Image) -> tuple[np.ndarray, float, float]:
    """Run model on a PIL image → (probs[14], threshold, inference_ms)."""
    tensor = eval_transforms(image).unsqueeze(0).to(device)

    start = time.perf_counter()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits).cpu().numpy()[0]
    inference_ms = (time.perf_counter() - start) * 1000

    threshold = float(os.getenv("PREDICTION_THRESHOLD", "0.5"))
    return probs, threshold, inference_ms


def _build_predictions(probs: np.ndarray, threshold: float) -> List[Prediction]:
    return [
        Prediction(label=label, probability=float(prob), positive=bool(prob >= threshold))
        for label, prob in zip(LABEL_COLS, probs)
    ]


def _compute_gradcam(
    image: Image.Image, probs: np.ndarray, threshold: float
) -> tuple[str | None, str | None]:
    """Generate Grad-CAM for the top finding → (base64_img, label) or (None, None)."""
    try:
        class_idx = pick_gradcam_class(probs, threshold)
        if class_idx is None:
            return None, None
        tensor = eval_transforms(image).unsqueeze(0)
        b64 = generate_gradcam_b64(model, tensor, image, class_idx, device)
        return b64, LABEL_COLS[class_idx]
    except Exception as e:
        logger.warning("Grad-CAM failed: %s", e, exc_info=True)
        return None, None


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _format_predictions_for_prompt(predictions: List[Prediction]) -> str:
    lines = []
    for p in predictions:
        status = "positive" if p.positive else "negative"
        lines.append(f"- {p.label}: {status} (probability={p.probability:.2f})")
    return "\n".join(lines)


def _format_clinical_context_for_prompt(ctx: Optional[ClinicalContext]) -> str:
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
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured on the server.")
    return OpenAI(api_key=api_key)


def _call_openai_chat(messages: list[dict[str, str]]) -> str:
    """Send messages to the Chat API and return the assistant's reply."""
    client = _get_openai_client()
    model_name = os.getenv("OPENAI_MODEL", OPENAI_MODEL_DEFAULT)
    temperature = float(os.getenv("OPENAI_TEMPERATURE", str(OPENAI_TEMPERATURE_DEFAULT)))

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Error while calling OpenAI API: {exc}")

    return response.choices[0].message.content or ""


def _generate_report_text(
    predictions: List[Prediction],
    clinical_context: Optional[ClinicalContext] = None,
) -> str:
    """Build prompt (predictions + context + RAG) → call OpenAI → return report."""
    predictions_text = _format_predictions_for_prompt(predictions)
    context_text = _format_clinical_context_for_prompt(clinical_context)

    # RAG: find relevant chunks from pathology docs
    rag_query = f"Chest X-ray findings:\n{predictions_text}\n\nClinical context:\n{context_text}"
    rag_context = rag.get_rag_context(rag_query)

    system_prompt = (
        "You are an assistant for radiologists and clinicians who use you as a "
        "decision-support partner. They are the treating professionals; give them "
        "direct, actionable input. You receive chest X-ray model predictions and "
        "optional clinical context. Do the following:\n"
        "- Summarize the main findings clearly.\n"
        "- Highlight potential red flags and differentials.\n"
        "- Suggest concrete next steps (further imaging, workup, treatment options) "
        "as appropriate. Be concise (5-8 sentences) and use clear clinical language."
    )

    user_content = (
        f"Model predictions:\n{predictions_text}\n\n"
        f"Clinical context:\n{context_text}\n\n"
    )
    if rag_context:
        user_content += f"Relevant excerpts from protocol/pathology documents:\n{rag_context}\n\n"
    user_content += "Write a short report for the clinician."

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
    """Basic API info."""
    return {
        "name": PROJECT_NAME,
        "model_version": MODEL_VERSION,
        "num_labels": len(LABEL_COLS),
        "labels": LABEL_COLS,
        "disclaimer": DISCLAIMER,
    }


@app.get("/health")
def health() -> dict:
    """Model + RAG status (called by frontend on page load)."""
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
    """Image → predictions + Grad-CAM."""
    if not MODEL_LOADED or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    image = _read_image_file(file)
    probs, threshold, inference_ms = _run_inference(image)
    predictions = _build_predictions(probs, threshold)
    cam_b64, cam_label = _compute_gradcam(image, probs, threshold)

    return PredictResponse(
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
    """Image + optional context → predictions + Grad-CAM + LLM report."""
    if not MODEL_LOADED or model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")

    image = _read_image_file(file)
    probs, threshold, inference_ms = _run_inference(image)
    predictions = _build_predictions(probs, threshold)
    cam_b64, cam_label = _compute_gradcam(image, probs, threshold)

    # clinical_context arrives as a JSON string (multipart form field)
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
            pass

    report_text = _generate_report_text(predictions, ctx)

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
    """Generate a report from existing predictions (no image upload)."""
    if not payload.predictions:
        raise HTTPException(status_code=400, detail="At least one prediction is required.")
    report_text = _generate_report_text(payload.predictions, payload.clinical_context)
    return ReportResponse(report=report_text, disclaimer=DISCLAIMER)


@app.post("/chat", response_model=ChatResponse)
def chat_with_assistant(payload: ChatRequest) -> ChatResponse:
    """
    Stateless chat — frontend sends full history + context each time.
    """
    if not payload.message:
        raise HTTPException(status_code=400, detail="message is required.")

    # RAG retrieval
    rag_query = payload.message
    if payload.predictions:
        rag_query = _format_predictions_for_prompt(payload.predictions) + "\n\n" + rag_query
    rag_context = rag.get_rag_context(rag_query)

    system_prompt = (
        "You are an AI assistant for clinicians interpreting chest X-ray model "
        "outputs. They use you as a partner. Provide clear differential diagnoses, "
        "next steps, and treatment or workup recommendations as appropriate. "
        "Use direct clinical language; the clinician will make the final decision. "
        "Format your reply in Markdown: use ## for section headings, - for bullet "
        "lists, and separate paragraphs with a blank line so the answer is easy to read."
    )
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Inject available context as a second system message
    context_parts: list[str] = []
    if payload.predictions:
        context_parts.append("Model predictions:\n" + _format_predictions_for_prompt(payload.predictions))
    if payload.clinical_context:
        context_parts.append("Clinical context:\n" + _format_clinical_context_for_prompt(payload.clinical_context))
    if payload.report:
        context_parts.append(f"Previous report:\n{payload.report}")
    if rag_context:
        context_parts.append("Relevant excerpts from protocol/pathology documents:\n" + rag_context)
    if context_parts:
        messages.append({"role": "system", "content": "\n\n".join(context_parts)})

    # Conversation history
    if payload.history:
        for turn in payload.history:
            messages.append({"role": turn.role, "content": turn.content})

    messages.append({"role": "user", "content": payload.message})

    answer = _call_openai_chat(messages)
    return ChatResponse(answer=answer, disclaimer=DISCLAIMER)
