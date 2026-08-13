"""
Leaf-Expert — API Routes: DataStore (Data Flywheel Collection)
"""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.schemas.vlm import (
    DataPointSaveRequest, DataPointRecord,
    DataBankStats, ExportRequest, ExportResponse,
)
from app.services.datastore_service import get_datastore
from app.api.routes_vlm import get_pending_scan_path, release_pending_scan
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/datastore", tags=["DataBank (Data Flywheel)"])


@router.post("/save")
async def save_datapoint(req: DataPointSaveRequest):
    """
    Save a VLM scan result as a labeled data point in the local DataBank.

    - `scan_id`: returned by `POST /vlm/scan`
    - `accepted_label`: user-accepted or corrected class label
    - `confirmed`: set false if you're unsure about the label

    The stored image + label will later be used to bootstrap your own PyTorch model.
    """
    tmp_path = get_pending_scan_path(req.scan_id)
    if not tmp_path:
        raise HTTPException(404, f"Scan ID '{req.scan_id}' not found or already saved.")

    # Retrieve VLM prediction from the running scan — we need it to store full metadata.
    # The vlm_prediction is stored alongside the image in the datastore.
    # For simplicity, we store minimal fields here; the full prediction is already in ScanResponse.
    try:
        ds = get_datastore()
        result = await ds.save(
            temp_image_path=tmp_path,
            vlm_provider="unknown",   # would be passed from client in production
            vlm_model="unknown",
            vlm_prediction={"disease_name": req.accepted_label},
            user_label=req.accepted_label,
            confirmed=req.confirmed,
            notes=req.notes,
        )
        release_pending_scan(req.scan_id)
        return result
    except Exception as e:
        logger.exception("Failed to save data point")
        raise HTTPException(500, str(e))


@router.post("/save_full")
async def save_datapoint_full(
    scan_id: str,
    accepted_label: str,
    confirmed: bool = True,
    notes: Optional[str] = None,
    vlm_provider: str = "unknown",
    vlm_model: str = "unknown",
    vlm_prediction: Optional[dict] = None,
):
    """
    Extended save endpoint that includes full VLM prediction metadata.
    Called by the frontend after a scan to persist everything.
    """
    tmp_path = get_pending_scan_path(scan_id)
    if not tmp_path:
        raise HTTPException(404, f"Scan ID '{scan_id}' not found or already saved.")

    try:
        ds = get_datastore()
        result = await ds.save(
            temp_image_path=tmp_path,
            vlm_provider=vlm_provider,
            vlm_model=vlm_model,
            vlm_prediction=vlm_prediction or {},
            user_label=accepted_label,
            confirmed=confirmed,
            notes=notes,
        )
        release_pending_scan(scan_id)
        return result
    except Exception as e:
        logger.exception("Failed to save data point")
        raise HTTPException(500, str(e))


@router.get("/list", response_model=list[DataPointRecord])
async def list_datapoints(
    label: Optional[str] = Query(None, description="Filter by class label"),
    confirmed_only: bool = Query(False, description="Return only user-confirmed points"),
    limit: int = Query(200, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """List all collected data points, optionally filtered by label."""
    try:
        ds = get_datastore()
        points = await ds.list_points(label=label, confirmed_only=confirmed_only,
                                       limit=limit, offset=offset)
        return [
            DataPointRecord(
                id=p["id"],
                image_url=f"/datastore/{p['id']}/image",
                vlm_provider=p["vlm_provider"],
                vlm_model=p["vlm_model"],
                disease_name=p["disease_name"],
                user_label=p["user_label"],
                confirmed=bool(p["confirmed"]),
                confidence=p["confidence"],
                severity=p["severity"],
                notes=p.get("notes"),
                collected_at=p["collected_at"],
                used_for_training=bool(p["used_for_training"]),
            )
            for p in points
        ]
    except Exception as e:
        raise HTTPException(500, str(e))


@router.get("/stats", response_model=DataBankStats)
async def get_stats():
    """
    Return DataBank statistics: class distribution, training readiness,
    and which classes have enough images to unlock model training.
    """
    try:
        ds = get_datastore()
        stats = await ds.get_stats()
        return DataBankStats(**stats)
    except Exception as e:
        raise HTTPException(500, str(e))


@router.delete("/{point_id}")
async def delete_datapoint(point_id: str):
    """Delete a data point and its stored image file."""
    ds = get_datastore()
    deleted = await ds.delete(point_id)
    if not deleted:
        raise HTTPException(404, f"Data point '{point_id}' not found.")
    return {"status": "deleted", "id": point_id}


@router.get("/{point_id}/image")
async def get_image(point_id: str):
    """Serve the stored image for a data point."""
    ds = get_datastore()
    img_path = ds.get_image_path(point_id)
    if not img_path:
        raise HTTPException(404, f"Image for data point '{point_id}' not found.")
    return FileResponse(str(img_path), media_type="image/jpeg")


@router.post("/export", response_model=ExportResponse)
async def export_dataset(req: ExportRequest):
    """
    Export collected data points as a standard dataset directory
    (train/val/test/<label>/) ready for the PyTorch trainer.

    After export, use the `target_dir` as the `data_path` in `POST /train/start`.
    """
    try:
        ds = get_datastore()
        result = await ds.export_as_dataset(
            target_dir=req.target_dir,
            confirmed_only=req.confirmed_only,
            label_filter=req.label_filter,
            val_ratio=req.val_ratio,
            test_ratio=req.test_ratio,
        )
        return ExportResponse(**result)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(500, str(e))
