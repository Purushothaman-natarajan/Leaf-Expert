"""
Leaf-Expert — Pydantic Schemas: Training
"""
from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class BackboneChoice(str, Enum):
    efficientnet_v2_s = "efficientnet_v2_s"
    efficientnet_v2_m = "efficientnet_v2_m"
    resnet50 = "resnet50"
    resnet101 = "resnet101"
    densenet121 = "densenet121"
    mobilenet_v3_large = "mobilenet_v3_large"
    vgg16 = "vgg16"
    vgg19 = "vgg19"
    # timm-backed
    vit_base_patch16_224 = "vit_base_patch16_224"
    convnext_small = "convnext_small"


class OptimizerChoice(str, Enum):
    adamw = "adamw"
    adam = "adam"
    sgd = "sgd"


class TrainRequest(BaseModel):
    data_path: str = Field(..., description="Path to prepared dataset (must contain train/val/test subdirs)")
    model_dir: str = Field(..., description="Directory to save trained model checkpoints")
    log_dir: str = Field(..., description="Directory to save training logs")
    backbones: list[BackboneChoice] = Field(
        default=[BackboneChoice.efficientnet_v2_s],
        description="List of backbone architectures to train"
    )
    image_size: int = Field(224, ge=32, le=512)
    epochs: int = Field(30, ge=1, le=500)
    batch_size: int = Field(32, ge=1, le=256)
    learning_rate: float = Field(1e-3, gt=0)
    optimizer: OptimizerChoice = Field(OptimizerChoice.adamw)
    patience: int = Field(7, ge=1, le=100, description="Early stopping patience")
    unfreeze_layers: int = Field(20, ge=0, description="Number of backbone layers to unfreeze in phase 2")


class TrainStatusResponse(BaseModel):
    job_id: str
    status: str  # "running" | "completed" | "failed" | "not_found"
    current_epoch: Optional[int] = None
    total_epochs: Optional[int] = None
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    best_val_acc: Optional[float] = None
    message: Optional[str] = None


class TrainStartResponse(BaseModel):
    job_id: str
    message: str
    backbones: list[str]
