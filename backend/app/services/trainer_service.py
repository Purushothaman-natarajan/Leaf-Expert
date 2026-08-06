"""
Leaf-Expert — PyTorch Training Service
Full transfer-learning pipeline with two-phase fine-tuning,
mixed-precision, CosineAnnealingLR, and background job support.
"""
import json
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.v2 as T
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Global job registry ───────────────────────────────────────────────────────
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # One training job at a time

# ─── Device resolution ─────────────────────────────────────────────────────────

def get_device(preference: str = "auto") -> torch.device:
    if preference == "cuda" or (preference == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    if preference == "mps" or (preference == "auto" and torch.backends.mps.is_available()):
        return torch.device("mps")
    return torch.device("cpu")


# ─── Backbone registry ─────────────────────────────────────────────────────────

def _build_backbone(name: str, num_classes: int) -> nn.Module:
    """
    Load a pretrained backbone from torchvision or timm and
    replace the classification head with one matching num_classes.
    """
    tv_map = {
        "efficientnet_v2_s": (tvm.efficientnet_v2_s, tvm.EfficientNet_V2_S_Weights.DEFAULT),
        "efficientnet_v2_m": (tvm.efficientnet_v2_m, tvm.EfficientNet_V2_M_Weights.DEFAULT),
        "resnet50":          (tvm.resnet50,           tvm.ResNet50_Weights.DEFAULT),
        "resnet101":         (tvm.resnet101,          tvm.ResNet101_Weights.DEFAULT),
        "densenet121":       (tvm.densenet121,        tvm.DenseNet121_Weights.DEFAULT),
        "mobilenet_v3_large":(tvm.mobilenet_v3_large, tvm.MobileNet_V3_Large_Weights.DEFAULT),
        "vgg16":             (tvm.vgg16,              tvm.VGG16_Weights.DEFAULT),
        "vgg19":             (tvm.vgg19,              tvm.VGG19_Weights.DEFAULT),
    }

    if name in tv_map:
        factory, weights = tv_map[name]
        model = factory(weights=weights)
        # Replace classifier head
        if hasattr(model, "classifier"):
            in_features = (
                model.classifier[-1].in_features
                if hasattr(model.classifier[-1], "in_features")
                else model.classifier[1].in_features
            )
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        elif hasattr(model, "fc"):
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    # Fall back to timm for ViT, ConvNeXt, etc.
    try:
        import timm
        model = timm.create_model(name, pretrained=True, num_classes=num_classes)
        return model
    except Exception as e:
        raise ValueError(f"Unsupported backbone '{name}'. Error: {e}")


def _freeze_backbone(model: nn.Module, backbone_name: str):
    """Freeze all parameters except the final classifier head."""
    head_names = {"classifier", "fc", "head"}
    for name, param in model.named_parameters():
        if not any(h in name for h in head_names):
            param.requires_grad = False


def _unfreeze_last_n(model: nn.Module, n: int):
    """Unfreeze the last n layers for phase-2 fine-tuning."""
    all_params = list(model.named_parameters())
    for name, param in all_params[-n:]:
        param.requires_grad = True


# ─── Dataset / Transforms ──────────────────────────────────────────────────────

def _get_transforms(image_size: int, augment: bool = False):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if augment:
        return T.Compose([
            T.Resize((image_size + 32, image_size + 32)),
            T.RandomCrop(image_size),
            T.RandomHorizontalFlip(),
            T.RandAugment(num_ops=2, magnitude=9),
            T.ToImage(),
            T.ToDtype(torch.float32, scale=True),
            T.Normalize(mean=mean, std=std),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=mean, std=std),
    ])


# ─── Training loop ─────────────────────────────────────────────────────────────

def _train_one_backbone(
    backbone_name: str,
    data_path: str,
    model_dir: str,
    log_dir: str,
    image_size: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    optimizer_name: str,
    patience: int,
    unfreeze_layers: int,
    device: torch.device,
    job: dict,
):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    train_ds = ImageFolder(os.path.join(data_path, "train"), transform=_get_transforms(image_size, augment=True))
    val_ds   = ImageFolder(os.path.join(data_path, "val"),   transform=_get_transforms(image_size, augment=False))

    classes = train_ds.classes
    num_classes = len(classes)
    logger.info(f"[{backbone_name}] Classes: {classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = _build_backbone(backbone_name, num_classes).to(device)
    _freeze_backbone(model, backbone_name)

    # ── Phase 1: Train head only ──────────────────────────────────────────────
    phase1_epochs = max(1, epochs // 3)
    phase2_epochs = epochs - phase1_epochs

    def _make_optimizer(params):
        lr = learning_rate
        if optimizer_name == "adamw":
            return AdamW(params, lr=lr, weight_decay=1e-2)
        elif optimizer_name == "adam":
            return Adam(params, lr=lr)
        else:
            return SGD(params, lr=lr, momentum=0.9, weight_decay=1e-4)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=(device.type == "cuda"))

    best_val_acc = 0.0
    no_improve = 0
    log_lines = []

    def run_epoch(loader, train_mode: bool):
        model.train() if train_mode else model.eval()
        total_loss, correct, total = 0.0, 0, 0
        with torch.set_grad_enabled(train_mode):
            for imgs, lbls in loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                optimizer.zero_grad()
                with autocast(enabled=(device.type == "cuda")):
                    outputs = model(imgs)
                    loss = criterion(outputs, lbls)
                if train_mode:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += loss.item() * imgs.size(0)
                correct    += (outputs.argmax(1) == lbls).sum().item()
                total      += imgs.size(0)
        return total_loss / total, correct / total

    for phase, ep_count in [("phase1", phase1_epochs), ("phase2", phase2_epochs)]:
        if phase == "phase2":
            _unfreeze_last_n(model, unfreeze_layers)
            logger.info(f"[{backbone_name}] Phase 2: unfroze last {unfreeze_layers} layers")

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = _make_optimizer(trainable)
        scheduler = CosineAnnealingLR(optimizer, T_max=ep_count, eta_min=1e-6)

        for ep in range(ep_count):
            train_loss, train_acc = run_epoch(train_loader, True)
            val_loss,   val_acc   = run_epoch(val_loader,   False)
            scheduler.step()

            epoch_num = (phase1_epochs if phase == "phase2" else 0) + ep + 1
            log_lines.append(
                f"Epoch {epoch_num}/{epochs} [{phase}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )
            logger.info(log_lines[-1])

            # Update job status
            job.update({
                "current_epoch": epoch_num,
                "total_epochs": epochs,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 4),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 4),
                "best_val_acc": round(best_val_acc, 4),
            })

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improve = 0
                ckpt = os.path.join(model_dir, f"{backbone_name}_best.pth")
                torch.save(model.state_dict(), ckpt)
                logger.info(f"[{backbone_name}] Saved best model: val_acc={val_acc:.4f}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    logger.info(f"[{backbone_name}] Early stopping at epoch {epoch_num}")
                    break

    # Save config
    config = {
        "backbone": backbone_name,
        "num_classes": num_classes,
        "classes": classes,
        "image_size": image_size,
        "best_val_acc": round(best_val_acc, 4),
    }
    config_path = os.path.join(model_dir, f"{backbone_name}_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    log_path = os.path.join(log_dir, f"{backbone_name}_training.log")
    with open(log_path, "w") as f:
        f.write("\n".join(log_lines))

    logger.info(f"[{backbone_name}] Training complete. Best val acc: {best_val_acc:.4f}")


def _run_job(job_id: str, request: dict, device: torch.device):
    job = _jobs[job_id]
    job["status"] = "running"
    try:
        for backbone in request["backbones"]:
            _train_one_backbone(
                backbone_name=backbone,
                data_path=request["data_path"],
                model_dir=request["model_dir"],
                log_dir=request["log_dir"],
                image_size=request["image_size"],
                epochs=request["epochs"],
                batch_size=request["batch_size"],
                learning_rate=request["learning_rate"],
                optimizer_name=request["optimizer"],
                patience=request["patience"],
                unfreeze_layers=request["unfreeze_layers"],
                device=device,
                job=job,
            )
        job["status"] = "completed"
        job["message"] = "All backbones trained successfully"
    except Exception as e:
        job["status"] = "failed"
        job["message"] = str(e)
        logger.exception(f"Job {job_id} failed")


# ─── Public API ────────────────────────────────────────────────────────────────

def start_training(request: dict, device_pref: str = "auto") -> str:
    """Submit a training job; returns job_id."""
    device = get_device(device_pref)
    logger.info(f"Training on device: {device}")
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {
        "status": "queued",
        "current_epoch": 0,
        "total_epochs": request["epochs"],
        "train_loss": None,
        "train_acc": None,
        "val_loss": None,
        "val_acc": None,
        "best_val_acc": None,
        "message": None,
    }
    _executor.submit(_run_job, job_id, request, device)
    return job_id


def get_job_status(job_id: str) -> dict:
    if job_id not in _jobs:
        return {"job_id": job_id, "status": "not_found"}
    return {"job_id": job_id, **_jobs[job_id]}
