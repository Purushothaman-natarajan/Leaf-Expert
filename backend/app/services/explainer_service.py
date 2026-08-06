"""
Leaf-Expert — XAI Explainer Service
Generates Grad-CAM++ heatmaps (via pytorch-grad-cam) and
LIME superpixel explanations (via lime library).
All outputs returned as base64 PNG strings — fully stateless, no disk I/O.
"""
import base64
import io
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from app.core.logging import get_logger
from app.services.predictor_service import predict_from_pil
from app.services.trainer_service import get_device

logger = get_logger(__name__)


# ─── Helpers ───────────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _find_target_layer(model: nn.Module) -> Optional[nn.Module]:
    """Walk model in reverse to find the last Conv2d layer."""
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv


def _overlay_heatmap(original_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Superimpose a colourised heatmap onto the original image."""
    import matplotlib.cm as cm
    heatmap_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    colourmap = cm.get_cmap("jet")
    coloured = (colourmap(heatmap_norm)[:, :, :3] * 255).astype(np.uint8)
    coloured_img = Image.fromarray(coloured).resize(
        (original_rgb.shape[1], original_rgb.shape[0]), Image.LANCZOS
    )
    original_img = Image.fromarray(original_rgb)
    return Image.blend(original_img, coloured_img, alpha)


# ─── Grad-CAM++ ────────────────────────────────────────────────────────────────

def _generate_gradcam(
    model: nn.Module,
    input_tensor: torch.Tensor,
    original_np: np.ndarray,
    pred_idx: int,
) -> Optional[str]:
    """
    Use pytorch-grad-cam library to generate a Grad-CAM++ heatmap.
    Returns base64 PNG or None if library unavailable.
    """
    try:
        from pytorch_grad_cam import GradCAMPlusPlus
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

        target_layer = _find_target_layer(model)
        if target_layer is None:
            # ViT / MLP-Mixer models — try attention-based CAM
            logger.warning("No Conv2d found; skipping GradCAM++")
            return None

        cam = GradCAMPlusPlus(model=model, target_layers=[target_layer])
        targets = [ClassifierOutputTarget(pred_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # (H, W)

        overlay = _overlay_heatmap(original_np, grayscale_cam)
        return _pil_to_b64(overlay)

    except ImportError:
        logger.warning("pytorch-grad-cam not installed; skipping GradCAM++")
        return None
    except Exception as e:
        logger.warning(f"GradCAM++ failed: {e}")
        return None


# ─── LIME ──────────────────────────────────────────────────────────────────────

def _generate_lime(
    model: nn.Module,
    original_pil: Image.Image,
    config: dict,
    device: torch.device,
    num_samples: int = 100,
    num_features: int = 30,
    segmentation_alg: str = "quickshift",
) -> Optional[str]:
    """
    Generate a LIME superpixel explanation.
    Returns base64 PNG or None if library unavailable.
    """
    try:
        import torchvision.transforms.v2 as T
        from lime.lime_image import LimeImageExplainer, SegmentationAlgorithm

        image_size = config["image_size"]

        transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        def predict_fn(images: np.ndarray) -> np.ndarray:
            batch = []
            for img_np in images:
                pil = Image.fromarray(img_np.astype(np.uint8))
                batch.append(transform(pil))
            batch_tensor = torch.stack(batch).to(device)
            with torch.no_grad():
                logits = model(batch_tensor)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
            return probs

        img_resized = original_pil.resize((image_size, image_size))
        img_np = np.array(img_resized)

        if segmentation_alg == "quickshift":
            seg_fn = SegmentationAlgorithm("quickshift", kernel_size=2, max_dist=100, ratio=0.1)
        else:
            seg_fn = SegmentationAlgorithm("slic", n_segments=50, compactness=10, sigma=1)

        explainer = LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=1,
            hide_color=0,
            num_samples=num_samples,
            num_features=num_features,
            segmentation_fn=seg_fn,
        )
        temp, mask = explanation.get_image_and_mask(
            explanation.top_labels[0],
            positive_only=True,
            num_features=num_features,
            hide_rest=False,
        )
        # Overlay mask on original
        mask_rgb = np.stack([mask * 200, mask * 50, np.zeros_like(mask)], axis=-1).astype(np.uint8)
        mask_img = Image.fromarray(mask_rgb).resize(original_pil.size, Image.LANCZOS)
        orig_rgba = original_pil.convert("RGBA")
        mask_rgba = mask_img.convert("RGBA")
        mask_rgba.putalpha(120)
        blended = Image.alpha_composite(orig_rgba, mask_rgba).convert("RGB")
        return _pil_to_b64(blended)

    except ImportError:
        logger.warning("lime not installed; skipping LIME explanation")
        return None
    except Exception as e:
        logger.warning(f"LIME failed: {e}")
        return None


# ─── Public API ────────────────────────────────────────────────────────────────

def explain(
    image_path: str,
    model_path: str,
    num_lime_samples: int = 100,
    num_lime_features: int = 30,
    segmentation_alg: str = "quickshift",
    device_pref: str = "auto",
) -> dict:
    """
    Run full explanation pipeline: predict + GradCAM++ + LIME.

    Returns a dict with prediction results + base64 PNG explanations.
    """
    device = get_device(device_pref)
    original_pil = Image.open(image_path).convert("RGB")
    original_np = np.array(original_pil)

    # Run prediction (reuses cached model)
    result, model, config, input_tensor = predict_from_pil(original_pil, model_path, device_pref)

    classes = config["classes"]
    pred_idx = classes.index(result["label"])

    # Grad-CAM++
    gradcam_b64 = _generate_gradcam(model, input_tensor, original_np, pred_idx)

    # LIME
    lime_b64 = _generate_lime(
        model, original_pil, config, device,
        num_lime_samples, num_lime_features, segmentation_alg
    )

    return {
        **result,
        "gradcam_b64": gradcam_b64 or "",
        "lime_b64": lime_b64 or "",
        "gradcam_available": gradcam_b64 is not None,
        "lime_available": lime_b64 is not None,
    }
