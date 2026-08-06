"""
Leaf-Expert — DataStore Service
SQLite-backed local data collection for the VLM data flywheel.

Stores:
  • Leaf images in data/collected/<uuid>.jpg
  • Structured metadata in data/datastore.db (SQLite)

The collected data can be exported as a standard folder dataset
and fed directly into the existing PyTorch trainer.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite
from PIL import Image

from app.core.logging import get_logger

logger = get_logger(__name__)

# ─── Schema ────────────────────────────────────────────────────────────────────

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS data_points (
    id              TEXT PRIMARY KEY,
    image_path      TEXT NOT NULL,
    image_hash      TEXT NOT NULL,
    vlm_provider    TEXT NOT NULL,
    vlm_model       TEXT NOT NULL,
    vlm_prediction  TEXT NOT NULL,   -- JSON blob (LeafScanResult)
    disease_name    TEXT NOT NULL,   -- from VLM (original prediction)
    user_label      TEXT NOT NULL,   -- accepted / corrected by user
    confirmed       INTEGER NOT NULL DEFAULT 1,
    confidence      REAL NOT NULL,
    severity        TEXT NOT NULL,
    notes           TEXT,
    collected_at    TEXT NOT NULL,
    used_for_training INTEGER NOT NULL DEFAULT 0,
    session_id      TEXT
);
CREATE INDEX IF NOT EXISTS idx_user_label ON data_points(user_label);
CREATE INDEX IF NOT EXISTS idx_confirmed ON data_points(confirmed);
CREATE INDEX IF NOT EXISTS idx_used_for_training ON data_points(used_for_training);
"""


# ─── DataStore class ────────────────────────────────────────────────────────────

class DataStore:
    def __init__(self, db_path: Path, images_dir: Path, training_threshold: int = 30):
        self.db_path = db_path
        self.images_dir = images_dir
        self.training_threshold = training_threshold

    async def init(self):
        """Create tables and directories."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.executescript(CREATE_TABLE_SQL)
            await db.commit()
        logger.info(f"DataStore initialised at {self.db_path}")

    # ─── Helpers ───────────────────────────────────────────────────────────────

    def _hash_image(self, img_path: str) -> str:
        with open(img_path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    async def _hash_exists(self, db: aiosqlite.Connection, img_hash: str) -> bool:
        async with db.execute(
            "SELECT id FROM data_points WHERE image_hash = ?", (img_hash,)
        ) as cur:
            return await cur.fetchone() is not None

    # ─── Save ──────────────────────────────────────────────────────────────────

    async def save(
        self,
        temp_image_path: str,
        vlm_provider: str,
        vlm_model: str,
        vlm_prediction: dict,
        user_label: str,
        confirmed: bool = True,
        notes: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        """
        Persist a scan image + label as a data point.
        Returns the saved record as a dict.
        """
        img_hash = self._hash_image(temp_image_path)

        async with aiosqlite.connect(str(self.db_path)) as db:
            if await self._hash_exists(db, img_hash):
                logger.info(f"Duplicate image hash {img_hash[:8]}… — skipping")
                return {"status": "duplicate", "id": None}

            point_id = uuid.uuid4().hex
            dest_path = self.images_dir / f"{point_id}.jpg"

            # Copy + resize to standard size for storage
            img = Image.open(temp_image_path).convert("RGB")
            img.thumbnail((512, 512), Image.LANCZOS)
            img.save(str(dest_path), "JPEG", quality=92)

            collected_at = datetime.now(timezone.utc).isoformat()
            disease_name = vlm_prediction.get("disease_name", "Unknown")
            confidence = float(vlm_prediction.get("confidence", 0.0))
            severity = vlm_prediction.get("severity", "unknown")

            await db.execute(
                """INSERT INTO data_points
                   (id, image_path, image_hash, vlm_provider, vlm_model, vlm_prediction,
                    disease_name, user_label, confirmed, confidence, severity,
                    notes, collected_at, used_for_training, session_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,0,?)""",
                (
                    point_id, str(dest_path), img_hash,
                    vlm_provider, vlm_model, json.dumps(vlm_prediction),
                    disease_name, user_label, int(confirmed),
                    confidence, severity, notes, collected_at, session_id,
                ),
            )
            await db.commit()
            logger.info(f"Saved data point {point_id} — label={user_label}")

        return {
            "status": "saved",
            "id": point_id,
            "user_label": user_label,
            "image_path": str(dest_path),
        }

    # ─── List ──────────────────────────────────────────────────────────────────

    async def list_points(
        self,
        label: Optional[str] = None,
        confirmed_only: bool = False,
        limit: int = 200,
        offset: int = 0,
    ) -> list[dict]:
        clauses, params = [], []
        if label:
            clauses.append("user_label = ?")
            params.append(label)
        if confirmed_only:
            clauses.append("confirmed = 1")

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT id, image_path, vlm_provider, vlm_model, disease_name, "
            f"user_label, confirmed, confidence, severity, notes, "
            f"collected_at, used_for_training "
            f"FROM data_points {where} "
            f"ORDER BY collected_at DESC LIMIT ? OFFSET ?"
        )
        params.extend([limit, offset])

        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()

        return [dict(r) for r in rows]

    # ─── Stats ─────────────────────────────────────────────────────────────────

    async def get_stats(self) -> dict:
        async with aiosqlite.connect(str(self.db_path)) as db:
            async with db.execute(
                "SELECT COUNT(*) as total, "
                "SUM(CASE WHEN confirmed=1 THEN 1 ELSE 0 END) as confirmed "
                "FROM data_points"
            ) as cur:
                totals = await cur.fetchone()

            async with db.execute(
                "SELECT user_label, COUNT(*) as cnt "
                "FROM data_points WHERE confirmed=1 "
                "GROUP BY user_label ORDER BY cnt DESC"
            ) as cur:
                rows = await cur.fetchall()

        distribution: dict[str, int] = {r[0]: r[1] for r in rows}
        classes_ready = [lbl for lbl, cnt in distribution.items() if cnt >= self.training_threshold]
        classes_pending = {
            lbl: max(0, self.training_threshold - cnt)
            for lbl, cnt in distribution.items()
            if cnt < self.training_threshold
        }

        return {
            "total_points": totals[0] if totals else 0,
            "confirmed_points": totals[1] if totals else 0,
            "class_distribution": distribution,
            "training_threshold": self.training_threshold,
            "classes_ready": classes_ready,
            "classes_pending": classes_pending,
            "can_train": len(classes_ready) >= 2,
        }

    # ─── Delete ────────────────────────────────────────────────────────────────

    async def delete(self, point_id: str) -> bool:
        async with aiosqlite.connect(str(self.db_path)) as db:
            async with db.execute(
                "SELECT image_path FROM data_points WHERE id=?", (point_id,)
            ) as cur:
                row = await cur.fetchone()
            if not row:
                return False
            img_path = Path(row[0])
            if img_path.exists():
                img_path.unlink()
            await db.execute("DELETE FROM data_points WHERE id=?", (point_id,))
            await db.commit()
        logger.info(f"Deleted data point {point_id}")
        return True

    # ─── Export as dataset ─────────────────────────────────────────────────────

    async def export_as_dataset(
        self,
        target_dir: str,
        confirmed_only: bool = True,
        label_filter: Optional[list[str]] = None,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ) -> dict:
        """
        Copy collected images into target_dir/train|val|test/<label>/ structure
        compatible with torchvision.datasets.ImageFolder and the trainer service.
        """
        import random
        from math import floor

        all_points = await self.list_points(confirmed_only=confirmed_only, limit=100_000)
        if label_filter:
            all_points = [p for p in all_points if p["user_label"] in label_filter]

        if not all_points:
            raise ValueError("No data points found to export.")

        # Group by label
        by_label: dict[str, list[dict]] = {}
        for p in all_points:
            by_label.setdefault(p["user_label"], []).append(p)

        exported_counts: dict[str, int] = {}
        train_count = val_count = test_count = 0
        target = Path(target_dir)

        for label, points in by_label.items():
            random.shuffle(points)
            n = len(points)
            n_val = max(1, floor(n * val_ratio))
            n_test = max(1, floor(n * test_ratio))
            n_train = n - n_val - n_test

            splits = {
                "train": points[:n_train],
                "val":   points[n_train:n_train + n_val],
                "test":  points[n_train + n_val:],
            }

            for split_name, split_points in splits.items():
                dest = target / split_name / label
                dest.mkdir(parents=True, exist_ok=True)
                for pt in split_points:
                    src = Path(pt["image_path"])
                    if src.exists():
                        shutil.copy2(str(src), str(dest / src.name))

            exported_counts[label] = n
            train_count += len(splits["train"])
            val_count   += len(splits["val"])
            test_count  += len(splits["test"])

        # Mark exported points as used
        exported_ids = [p["id"] for p in all_points]
        async with aiosqlite.connect(str(self.db_path)) as db:
            placeholders = ",".join("?" * len(exported_ids))
            await db.execute(
                f"UPDATE data_points SET used_for_training=1 WHERE id IN ({placeholders})",
                exported_ids,
            )
            await db.commit()

        logger.info(
            f"Exported {sum(exported_counts.values())} images "
            f"({train_count} train / {val_count} val / {test_count} test) to {target_dir}"
        )
        return {
            "status": "success",
            "target_dir": str(target.resolve()),
            "exported_counts": exported_counts,
            "train_count": train_count,
            "val_count": val_count,
            "test_count": test_count,
            "message": f"Exported {len(by_label)} classes. Ready for training.",
        }

    # ─── Image serving ─────────────────────────────────────────────────────────

    def get_image_path(self, point_id: str) -> Optional[Path]:
        """Return the stored image path for a data point (sync, for file responses)."""
        path = self.images_dir / f"{point_id}.jpg"
        return path if path.exists() else None


# ─── Singleton ─────────────────────────────────────────────────────────────────
# Instantiated in main.py lifespan; imported by routes.
_datastore: Optional[DataStore] = None


def get_datastore() -> DataStore:
    if _datastore is None:
        raise RuntimeError("DataStore not initialised. Call init_datastore() first.")
    return _datastore


async def init_datastore(db_path: Path, images_dir: Path, threshold: int = 30) -> DataStore:
    global _datastore
    _datastore = DataStore(db_path, images_dir, threshold)
    await _datastore.init()
    return _datastore
