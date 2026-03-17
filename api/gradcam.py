"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for DenseNet-121.

Produces a heatmap overlay showing which image regions drove the model's
prediction for a given class.  Red/orange = high influence, blue = low.

Algorithm:
  1. Forward pass, capturing activations at the last conv block.
  2. Compute gradients of the target class logit w.r.t. those activations.
  3. Global-average-pool the gradients → per-channel weights.
  4. Weighted sum → 2D map, ReLU, normalize, resize, apply jet colormap, blend.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_gradcam(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: int,
    device: torch.device,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Compute Grad-CAM for `class_idx` and return the blended overlay.

    Uses a forward hook to capture activations and torch.autograd.grad
    (instead of backward hooks) to avoid DenseNet in-place ReLU issues.
    """
    model.eval()

    activations: list[torch.Tensor] = []

    # Last conv block of DenseNet-121 → (batch, 1024, 7, 7) for 224×224 input
    target_layer = model.backbone.features

    def _fwd_hook(_module: torch.nn.Module, _inp: tuple, output: torch.Tensor) -> None:
        activations.append(output)

    fh = target_layer.register_forward_hook(_fwd_hook)

    try:
        inp = input_tensor.to(device).requires_grad_(True)
        logits = model(inp)  # (1, 14)

        model.zero_grad()
        target_logit = logits[0, class_idx]
        grads = torch.autograd.grad(
            target_logit,
            activations[0],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        act = activations[0][0].detach()   # (1024, 7, 7)
        grad = grads[0].detach()           # (1024, 7, 7)

        # Per-channel importance via global average pooling of gradients
        weights = grad.mean(dim=(1, 2))    # (1024,)

        # Weighted combination → heatmap
        cam = (weights[:, None, None] * act).sum(dim=0)  # (7, 7)
        cam = F.relu(cam)

        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()

    finally:
        fh.remove()

    # Resize 7×7 heatmap to original image resolution and apply jet colormap
    cam_uint8 = (cam_np * 255).astype(np.uint8)
    cam_pil = Image.fromarray(cam_uint8, mode="L")
    cam_resized = cam_pil.resize(original_image.size, Image.BILINEAR)
    cam_norm = np.array(cam_resized).astype(np.float32) / 255.0

    heatmap_rgba = cm.jet(cam_norm)
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)

    # Blend: overlay = alpha * heatmap + (1-alpha) * original
    orig_np = np.array(original_image.convert("RGB"))
    overlay = (alpha * heatmap_rgb + (1 - alpha) * orig_np).astype(np.uint8)

    return Image.fromarray(overlay)


def gradcam_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URI (data:image/png;base64,…)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_gradcam_b64(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    original_image: Image.Image,
    class_idx: int,
    device: torch.device,
) -> str:
    """Compute Grad-CAM and return the overlay as a base64 data URI."""
    overlay = generate_gradcam(model, input_tensor, original_image, class_idx, device)
    return gradcam_to_base64(overlay)


def pick_gradcam_class(
    probs: np.ndarray,
    threshold: float,
) -> Optional[int]:
    """
    Pick which class (0–13) to visualize.

    If any class is positive (>= threshold), pick the highest-probability one.
    Otherwise fall back to the overall highest probability.
    """
    positives = np.where(probs >= threshold)[0]
    if len(positives) > 0:
        return int(positives[np.argmax(probs[positives])])
    return int(np.argmax(probs))
