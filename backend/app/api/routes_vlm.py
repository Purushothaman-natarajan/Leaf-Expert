"""
Leaf-Expert — API Routes: VLM Quick Scan
"""
import os
import tempfile
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, Header

from app.schemas.vlm import ScanResponse, ProviderInfo, VLMProvider
from app.services.vlm_service import scan_leaf, get_providers_info
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/vlm", tags=["VLM Quick Scan"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@router.get("/providers", response_model=list[ProviderInfo])
async def list_providers():
    """
    List all available VLM providers with their default models, cost tiers, and requirements.
    Use this to populate the provider picker in the frontend.
    """
    return get_providers_info()


@router.post("/scan", response_model=ScanResponse)
async def scan_leaf_image(
    image: UploadFile = File(..., description="Leaf image to analyse"),
    provider: VLMProvider = Form(VLMProvider.gemini, description="VLM provider to use"),
    model_name: Optional[str] = Form(None, description="Override model name (leave empty for default)"),
    ollama_host: str = Form("http://localhost:11434", description="Ollama host URL (only for Ollama provider)"),
    x_vlm_api_key: Optional[str] = Header(None, alias="X-VLM-API-Key", description="Provider API key"),
):
    """
    Zero-shot leaf disease analysis using a Vision Language Model.

    **No training data required.** The VLM analyses the image and returns:
    - Disease identification + confidence
    - Severity assessment
    - Visible symptoms
    - Treatment recommendations
    - Detailed explanation

    The returned `scan_id` can be used with `POST /datastore/save` to store
    this result as a labeled training data point.

    **API key**: Pass as `X-VLM-API-Key` header. It is used in-flight and never stored server-side.
    """
    ext = os.path.splitext(image.filename or "")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported format '{ext}'. Allowed: {ALLOWED_EXTENSIONS}")

    contents = await image.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    # Store the temp file path keyed by scan_id so datastore/save can retrieve it
    try:
        scan_id, result, provider_str, model_str, elapsed_ms = scan_leaf(
            image_path=tmp_path,
            provider=provider,
            api_key=x_vlm_api_key,
            model_name=model_name or None,
            ollama_host=ollama_host,
        )
        # Persist temp path in a short-lived registry so /datastore/save can find it
        _pending_scans[scan_id] = tmp_path
        return ScanResponse(
            scan_id=scan_id,
            provider_used=provider_str,
            model_used=model_str,
            result=result,
            processing_time_ms=elapsed_ms,
        )
    except ValueError as e:
        os.unlink(tmp_path)
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        os.unlink(tmp_path)
        raise HTTPException(503, str(e))
    except Exception as e:
        os.unlink(tmp_path)
        logger.exception("VLM scan failed")
        raise HTTPException(500, f"VLM scan failed: {str(e)}")


# ─── In-memory pending scan registry (scan_id → temp file path) ────────────────
# TTL is implicit: files are cleaned up by datastore/save or GC at shutdown.
# For production, replace with Redis or a simple SQLite temp table.
_pending_scans: dict[str, str] = {}


def get_pending_scan_path(scan_id: str) -> Optional[str]:
    return _pending_scans.get(scan_id)


def release_pending_scan(scan_id: str):
    path = _pending_scans.pop(scan_id, None)
    if path and os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass
