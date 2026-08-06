"""
Leaf-Expert — API Routes: Prediction & Explanation
"""
import os
import tempfile

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from app.schemas.prediction import PredictionResponse, ExplainResponse
from app.services.predictor_service import predict
from app.services.explainer_service import explain
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/predict", tags=["Prediction & Explanation"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _validate_image(file: UploadFile):
    ext = os.path.splitext(file.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}"
        )


@router.post("/", response_model=PredictionResponse)
async def predict_image(
    image: UploadFile = File(..., description="Leaf image to classify"),
    model_path: str = Form(..., description="Path to trained .pth model file"),
):
    """
    Classify a leaf image using a trained PyTorch model.
    Returns predicted label, confidence score, and per-class probabilities.
    """
    _validate_image(image)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = predict(tmp_path, model_path, device_pref=settings.device)
        return PredictionResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


@router.post("/explain", response_model=ExplainResponse)
async def predict_and_explain(
    image: UploadFile = File(..., description="Leaf image to classify and explain"),
    model_path: str = Form(..., description="Path to trained .pth model file"),
    num_lime_samples: int = Form(100, ge=10, le=1000),
    num_lime_features: int = Form(30, ge=5, le=100),
    segmentation_alg: str = Form("quickshift"),
):
    """
    Classify a leaf image and generate XAI explanations:
    - **Grad-CAM++**: Highlights regions that influenced the prediction
    - **LIME**: Superpixel-based local explanation

    Both explanations are returned as base64-encoded PNG strings.
    """
    _validate_image(image)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await image.read())
        tmp_path = tmp.name

    try:
        result = explain(
            image_path=tmp_path,
            model_path=model_path,
            num_lime_samples=num_lime_samples,
            num_lime_features=num_lime_features,
            segmentation_alg=segmentation_alg,
            device_pref=settings.device,
        )
        return ExplainResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("Explanation failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)
