"""
Leaf-Expert — Pydantic Schemas: Prediction & Explanation
"""
from pydantic import BaseModel, Field
from typing import Optional


class PredictionResponse(BaseModel):
    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    all_class_probs: dict[str, float]


class ExplainRequest(BaseModel):
    num_lime_samples: int = Field(100, ge=10, le=1000, description="LIME perturbation samples (higher = slower but more accurate)")
    num_lime_features: int = Field(30, ge=5, le=100)
    segmentation_alg: str = Field("quickshift", description="LIME segmentation: 'quickshift' or 'slic'")


class ExplainResponse(BaseModel):
    label: str
    confidence: float
    all_class_probs: dict[str, float]
    gradcam_b64: str = Field(..., description="Base64-encoded PNG of Grad-CAM++ heatmap overlaid on original")
    lime_b64: str = Field(..., description="Base64-encoded PNG of LIME superpixel mask overlaid on original")
    gradcam_available: bool = True
    lime_available: bool = True
