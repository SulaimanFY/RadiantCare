"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for DenseNet-121.

What is Grad-CAM?
  It highlights which regions of the input image most influenced the model's
  prediction for a specific class. The output is a heatmap overlaid on the
  original image (red/orange = high influence, blue = low influence).

How it works (simplified):
  1. Do a forward pass through the model.
  2. Record the feature maps (activations) at a target convolutional layer.
  3. Compute the gradient of the target class score w.r.t. those feature maps.
  4. Global-average-pool the gradients to get per-channel importance weights.
  5. Weighted sum of feature maps → 2D heatmap.
  6. Apply ReLU (keep only positive influence), normalize, resize to image size.
  7. Apply a colormap (jet) and blend with the original image.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (no GUI window needed on a server)
import matplotlib.cm as cm  # colormaps (jet, viridis, etc.)
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
    Compute Grad-CAM for a specific class and return the overlay on the original image.

    Args:
        model:          the ChestXrayClassifier (must have .backbone.features)
        input_tensor:   preprocessed image tensor, shape (1, 3, 224, 224)
        original_image: the original PIL image (for overlay, at its original resolution)
        class_idx:      which class (0–13) to compute the heatmap for
        device:         CPU or CUDA
        alpha:          blending factor (0 = only original, 1 = only heatmap)

    Returns:
        PIL Image with the heatmap overlaid on the original image.
    """
    model.eval()

    # We capture the activations at the target layer with a forward hook (a callback
    # PyTorch calls during the forward pass). The gradients are computed afterwards with
    # torch.autograd.grad rather than a backward hook, which avoids errors caused by
    # DenseNet's in-place ReLU operations.
    activations: list[torch.Tensor] = []

    # Target layer: model.backbone.features = the last convolutional block of DenseNet-121.
    # For a 224×224 input, this outputs feature maps of shape (batch, 1024, 7, 7).
    target_layer = model.backbone.features

    # Forward hook: called after the target layer computes its output.
    # We keep the output (the activations / feature maps) attached to the graph,
    # so that gradients can be computed with respect to it.
    def _fwd_hook(_module: torch.nn.Module, _inp: tuple, output: torch.Tensor) -> None:
        activations.append(output)

    # Register the hook (it will be called automatically during the forward pass)
    fh = target_layer.register_forward_hook(_fwd_hook)

    try:
        # --- Forward pass ---
        inp = input_tensor.to(device).requires_grad_(True)  # enable gradient tracking
        logits = model(inp)  # shape: (1, 14)
        # At this point, _fwd_hook has been called and activations[0] is stored.

        # --- Backward pass ---
        # We want the gradient of the target class score w.r.t. the feature maps.
        # zero_grad ensures no leftover gradients from a previous call.
        model.zero_grad()
        target_logit = logits[0, class_idx]  # the score of the target class only
        grads = torch.autograd.grad(         # d(target score) / d(feature maps)
            target_logit,
            activations[0],
            retain_graph=False,
            create_graph=False,
            allow_unused=False,
        )[0]

        # --- Compute the heatmap ---
        act = activations[0][0].detach()   # shape: (1024, 7, 7) — feature maps for first (only) image
        grad = grads[0].detach()           # shape: (1024, 7, 7) — gradients for first (only) image

        # Global average pooling of gradients → one weight per channel
        # This tells us how important each of the 1024 feature map channels is
        weights = grad.mean(dim=(1, 2))  # shape: (1024,)

        # Weighted combination of feature maps
        # weights[:, None, None] broadcasts (1024,) to (1024, 1, 1) to multiply with (1024, 7, 7)
        cam = (weights[:, None, None] * act).sum(dim=0)  # shape: (7, 7)

        # ReLU: keep only positive activations (regions that positively influence the class)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        cam_np = cam.cpu().numpy()  # shape: (7, 7) with values in [0, 1]

    finally:
        # Always remove the hook to avoid memory leaks
        fh.remove()

    # --- Create the overlay image ---

    # Convert the 7×7 heatmap to uint8 (0–255) for resizing
    cam_uint8 = (cam_np * 255).astype(np.uint8)

    # Resize from 7×7 to the original image size (e.g. 2048×2048)
    cam_pil = Image.fromarray(cam_uint8, mode="L")  # grayscale
    cam_resized = cam_pil.resize(original_image.size, Image.BILINEAR)

    # Normalize back to [0, 1] for the colormap
    cam_norm = np.array(cam_resized).astype(np.float32) / 255.0

    # Apply the "jet" colormap: blue (low) → green → yellow → red (high)
    heatmap_rgba = cm.jet(cam_norm)                          # shape: (H, W, 4) RGBA
    heatmap_rgb = (heatmap_rgba[:, :, :3] * 255).astype(np.uint8)  # drop alpha, to uint8

    # Blend: overlay = alpha * heatmap + (1-alpha) * original
    orig_np = np.array(original_image.convert("RGB"))
    overlay = (alpha * heatmap_rgb + (1 - alpha) * orig_np).astype(np.uint8)

    return Image.fromarray(overlay)


def gradcam_to_base64(img: Image.Image) -> str:
    """Encode a PIL Image as a base64 data URI string (data:image/png;base64,...)."""
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
    """Shortcut: compute Grad-CAM and return the overlay as a base64 data URI."""
    overlay = generate_gradcam(model, input_tensor, original_image, class_idx, device)
    return gradcam_to_base64(overlay)


def pick_gradcam_class(
    probs: np.ndarray,
    threshold: float,
) -> Optional[int]:
    """
    Choose which class to generate the Grad-CAM heatmap for.

    Strategy:
      - If any class is above the threshold (positive), pick the one with the highest probability.
      - If no class is positive, pick the overall highest probability class anyway.

    Returns the index (0–13) of the chosen class.
    """
    # Find all classes above threshold
    positives = np.where(probs >= threshold)[0]
    if len(positives) > 0:
        # Among positives, pick the one with the highest probability
        return int(positives[np.argmax(probs[positives])])
    # No positives: pick the overall highest (most suspicious, even if below threshold)
    return int(np.argmax(probs))
