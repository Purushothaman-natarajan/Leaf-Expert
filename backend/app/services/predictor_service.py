"""
Leaf-Expert — PyTorch Predictor Service
Loads a trained model (.pth + _config.json) and runs inference.
Model is cached in-process after first load.
"""
import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from PIL import Image

from app.core.logging import get_logger
from app.services.trainer_service import _build_backbone, get_device

logger = get_logger(__name__)

# ─── In-process model cache ────────────────────────────────────────────────────
_model_cache: dict[str, tuple[nn.Module, dict]] = {}


def _load_model(model_path: str, device: torch.device) -> tuple[nn.Module, dict]:
    """Load model + config; caches by model_path."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    config_path = model_path.replace("_best.pth", "_config.json")
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    model = _build_backbone(config["backbone"], config["num_classes"])
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    _model_cache[model_path] = (model, config)
    logger.info(f"Loaded model '{config['backbone']}' from {model_path}")
    return model, config


def predict(
    image_path: str,
    model_path: str,
    device_pref: str = "auto",
) -> dict:
    """
    Run inference on a single image.

    Returns:
        label, confidence, all_class_probs dict
    """
    device = get_device(device_pref)
    model, config = _load_model(model_path, device)

    image_size = config["image_size"]
    classes = config["classes"]

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

    pred_idx = int(probs.argmax().item())
    label = classes[pred_idx]
    confidence = float(probs[pred_idx].item())
    all_class_probs = {cls: round(float(probs[i].item()), 6) for i, cls in enumerate(classes)}

    logger.info(f"Prediction: {label} ({confidence:.4f})")
    return {
        "label": label,
        "confidence": confidence,
        "all_class_probs": all_class_probs,
    }


def predict_from_pil(
    image: Image.Image,
    model_path: str,
    device_pref: str = "auto",
) -> tuple[dict, nn.Module, dict, torch.Tensor]:
    """
    Predict from a PIL image; also returns model, config, and input tensor
    for use by the explainer service.
    """
    device = get_device(device_pref)
    model, config = _load_model(model_path, device)

    image_size = config["image_size"]
    classes = config["classes"]

    transform = T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu()

    pred_idx = int(probs.argmax().item())
    label = classes[pred_idx]
    confidence = float(probs[pred_idx].item())
    all_class_probs = {cls: round(float(probs[i].item()), 6) for i, cls in enumerate(classes)}

    result = {
        "label": label,
        "confidence": confidence,
        "all_class_probs": all_class_probs,
    }
    return result, model, config, tensor


def clear_cache():
    """Remove all cached models from memory."""
    _model_cache.clear()
    logger.info("Model cache cleared")
