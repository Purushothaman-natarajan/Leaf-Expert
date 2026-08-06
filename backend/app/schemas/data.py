"""
Leaf-Expert — Pydantic Schemas: Data Preparation
"""
from pydantic import BaseModel, Field
from typing import Optional


class DataPrepRequest(BaseModel):
    raw_dataset_path: str = Field(..., description="Absolute path to the raw dataset folder (class subdirs inside)")
    target_folder: str = Field(..., description="Where to save the prepared train/val/test splits")
    image_size: int = Field(224, ge=32, le=1024, description="Resize images to this square dimension")
    batch_size: int = Field(32, ge=1, le=256)
    augment: bool = Field(False, description="Apply training-time augmentation (RandAugment)")
    train_ratio: float = Field(0.70, ge=0.5, le=0.9)
    val_ratio: float = Field(0.15, ge=0.05, le=0.3)


class DataPrepResponse(BaseModel):
    status: str
    train_count: int
    val_count: int
    test_count: int
    classes: list[str]
    target_folder: str
