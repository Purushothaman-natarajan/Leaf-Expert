"""
Leaf-Expert — Pydantic Schemas: VLM Quick Scan + DataStore
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── VLM Provider ──────────────────────────────────────────────────────────────

class VLMProvider(str, Enum):
    gemini = "gemini"
    openai = "openai"
    claude = "claude"
    ollama = "ollama"


# ─── Disease Analysis (structured output schema) ───────────────────────────────

class LeafScanResult(BaseModel):
    """
    Structured disease analysis returned by the VLM.
    Used as response_schema for Gemini; mirrors OpenAI Structured Outputs format.
    """
    is_plant: bool = Field(..., description="True if the image contains a plant leaf")
    crop_type: str = Field(..., description="Crop species (e.g. 'Tomato', 'Wheat', 'Unknown')")
    is_diseased: bool = Field(..., description="True if disease/deficiency symptoms are visible")
    disease_name: str = Field(
        ..., description="Specific disease name (e.g. 'Tomato Late Blight') or 'Healthy' or 'Unknown'"
    )
    scientific_name: Optional[str] = Field(
        None, description="Scientific name of the pathogen if known"
    )
    confidence: float = Field(
        ..., ge=0.0, le=1.0,
        description="VLM self-estimated confidence (0.0=uncertain, 1.0=very certain)"
    )
    severity: str = Field(
        ..., description="Disease severity: 'none' | 'low' | 'moderate' | 'high' | 'critical'"
    )
    affected_area_percent: int = Field(
        ..., ge=0, le=100,
        description="Estimated % of visible leaf surface showing symptoms"
    )
    symptoms: list[str] = Field(
        default_factory=list,
        description="List of visible symptoms (e.g. ['Brown lesions', 'Yellow halo'])"
    )
    treatment: list[str] = Field(
        default_factory=list,
        description="Actionable treatment steps (specific, no brand names)"
    )
    prevention: list[str] = Field(
        default_factory=list,
        description="Future prevention measures"
    )
    explanation: str = Field(
        ..., description="Detailed natural language explanation of the diagnosis"
    )
    urgency: str = Field(
        ..., description="Action urgency: 'none' | 'low' | 'medium' | 'high'"
    )
    is_safe_to_consume: Optional[bool] = Field(
        None, description="Whether the crop is safe to consume (null if unknown)"
    )


# ─── Scan API ──────────────────────────────────────────────────────────────────

class ScanResponse(BaseModel):
    scan_id: str
    provider_used: str
    model_used: str
    result: LeafScanResult
    processing_time_ms: int


class ProviderInfo(BaseModel):
    id: str
    name: str
    default_model: str
    cost_tier: str          # "free" | "low" | "medium" | "high"
    requires_api_key: bool
    description: str


# ─── DataStore API ─────────────────────────────────────────────────────────────

class DataPointSaveRequest(BaseModel):
    scan_id: str = Field(..., description="scan_id from /vlm/scan response")
    accepted_label: str = Field(
        ..., description="User-accepted or corrected class label (will become folder name in dataset)"
    )
    confirmed: bool = Field(True, description="Whether user manually confirmed/corrected this label")
    notes: Optional[str] = Field(None, description="Optional annotation notes")


class DataPointRecord(BaseModel):
    id: str
    image_url: str          # endpoint to retrieve the stored image
    vlm_provider: str
    vlm_model: str
    disease_name: str       # from VLM (original prediction)
    user_label: str         # accepted/corrected label
    confirmed: bool
    confidence: float
    severity: str
    notes: Optional[str]
    collected_at: str       # ISO datetime
    used_for_training: bool


class DataBankStats(BaseModel):
    total_points: int
    confirmed_points: int
    class_distribution: dict[str, int]   # label → count
    training_threshold: int              # configured minimum per class
    classes_ready: list[str]             # classes that have >= threshold images
    classes_pending: dict[str, int]      # label → how many more needed
    can_train: bool                      # true if at least 2 classes ready


class ExportRequest(BaseModel):
    target_dir: str = Field(..., description="Directory to export the dataset into")
    confirmed_only: bool = Field(True, description="Only export user-confirmed data points")
    label_filter: Optional[list[str]] = Field(None, description="Export only these class labels")
    val_ratio: float = Field(0.15, ge=0.05, le=0.4)
    test_ratio: float = Field(0.15, ge=0.05, le=0.4)


class ExportResponse(BaseModel):
    status: str
    target_dir: str
    exported_counts: dict[str, int]   # class → count
    train_count: int
    val_count: int
    test_count: int
    message: str
