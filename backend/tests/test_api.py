"""
Tests for Leaf-Expert backend using FastAPI TestClient + httpx.
"""
import io
import json
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from app.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "torch_version" in data
    assert "cuda_available" in data


def test_data_prepare_missing_path():
    resp = client.post("/data/prepare", json={
        "raw_dataset_path": "/nonexistent/path",
        "target_folder": "/tmp/out",
    })
    assert resp.status_code in (400, 500)


def test_train_start_returns_job_id():
    resp = client.post("/train/start", json={
        "data_path": "/tmp/dataset",
        "model_dir": "/tmp/models",
        "log_dir": "/tmp/logs",
        "backbones": ["efficientnet_v2_s"],
        "epochs": 1,
        "batch_size": 4,
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert len(data["job_id"]) > 0


def test_train_status_not_found():
    resp = client.get("/train/status/nonexistent_job")
    assert resp.status_code == 200
    assert resp.json()["status"] == "not_found"


def test_predict_rejects_invalid_file():
    """Should reject non-image file types."""
    resp = client.post(
        "/predict/",
        data={"model_path": "/tmp/fake.pth"},
        files={"image": ("test.txt", b"not an image", "text/plain")},
    )
    assert resp.status_code == 400


def test_predict_missing_model():
    """Should return 404 when model file doesn't exist."""
    img_bytes = io.BytesIO()
    from PIL import Image
    img = Image.new("RGB", (224, 224), color=(100, 150, 80))
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)

    resp = client.post(
        "/predict/",
        data={"model_path": "/nonexistent/model_best.pth"},
        files={"image": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code in (404, 500)
