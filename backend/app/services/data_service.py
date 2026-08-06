"""
Leaf-Expert — Data Service
Handles dataset preparation: stratified splitting, optional augmentation,
and saving organised train/val/test folder structures.
PyTorch-native (no TensorFlow dependency).
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from app.core.logging import get_logger

logger = get_logger(__name__)


def _collect_samples(raw_path: str) -> Tuple[list[str], list[str]]:
    """Walk raw dataset dir and collect (filepath, class_label) pairs."""
    all_paths, all_labels = [], []
    raw = Path(raw_path)
    for cls_dir in sorted(raw.iterdir()):
        if not cls_dir.is_dir():
            continue
        for fpath in cls_dir.iterdir():
            if fpath.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                all_paths.append(str(fpath))
                all_labels.append(cls_dir.name)
    return all_paths, all_labels


def _save_split(
    paths: list[str],
    labels: list[str],
    dest_root: Path,
    image_size: int,
    augment: bool = False,
) -> int:
    """Resize and copy images into dest_root/<class>/ folders."""
    dest_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for src, label in tqdm(zip(paths, labels), total=len(paths), desc=str(dest_root.name)):
        cls_dir = dest_root / label
        cls_dir.mkdir(exist_ok=True)
        dest_path = cls_dir / f"{uuid.uuid4().hex}.jpg"
        try:
            img = Image.open(src).convert("RGB").resize((image_size, image_size), Image.LANCZOS)
            img.save(str(dest_path), "JPEG", quality=95)
            count += 1

            if augment:
                # Simple augmentation: horizontal flip
                img_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
                img_flip.save(str(cls_dir / f"{uuid.uuid4().hex}_flip.jpg"), "JPEG", quality=95)
                count += 1
        except Exception as e:
            logger.warning(f"Skipping {src}: {e}")
    return count


def prepare_dataset(
    raw_dataset_path: str,
    target_folder: str,
    image_size: int = 224,
    augment: bool = False,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> dict:
    """
    Split raw dataset into train / val / test and save to target_folder.

    Returns a dict with counts and class list.
    """
    all_paths, all_labels = _collect_samples(raw_dataset_path)
    if not all_paths:
        raise ValueError(f"No images found in {raw_dataset_path}")

    classes = sorted(set(all_labels))
    logger.info(f"Found {len(all_paths)} images across {len(classes)} classes")

    test_ratio = round(1.0 - train_ratio - val_ratio, 4)

    # First split: train vs. (val + test)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio), random_state=42)
    train_idx, rest_idx = next(sss1.split(all_paths, all_labels))

    rest_paths = [all_paths[i] for i in rest_idx]
    rest_labels = [all_labels[i] for i in rest_idx]
    train_paths = [all_paths[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]

    # Second split: val vs. test
    val_frac = val_ratio / (val_ratio + test_ratio)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1 - val_frac), random_state=42)
    val_idx, test_idx = next(sss2.split(rest_paths, rest_labels))

    val_paths = [rest_paths[i] for i in val_idx]
    val_labels = [rest_labels[i] for i in val_idx]
    test_paths = [rest_paths[i] for i in test_idx]
    test_labels = [rest_labels[i] for i in test_idx]

    root = Path(target_folder)

    train_count = _save_split(train_paths, train_labels, root / "train", image_size, augment)
    val_count = _save_split(val_paths, val_labels, root / "val", image_size, augment=False)
    test_count = _save_split(test_paths, test_labels, root / "test", image_size, augment=False)

    logger.info(f"Split complete — train: {train_count}, val: {val_count}, test: {test_count}")

    return {
        "status": "success",
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "classes": classes,
        "target_folder": str(root.resolve()),
    }
