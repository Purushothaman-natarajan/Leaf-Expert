"""
Leaf-Expert — API Routes: Training
"""
from fastapi import APIRouter, HTTPException
from app.schemas.training import TrainRequest, TrainStartResponse, TrainStatusResponse
from app.services.trainer_service import start_training, get_job_status
from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/train", tags=["Training"])


@router.post("/start", response_model=TrainStartResponse)
async def start_train_job(req: TrainRequest):
    """
    Start a background training job. Returns a job_id to poll status.
    Supports multiple backbones; they are trained sequentially.
    """
    try:
        job_id = start_training(req.model_dump(), device_pref=settings.device)
        return TrainStartResponse(
            job_id=job_id,
            message=f"Training job queued with {len(req.backbones)} backbone(s)",
            backbones=[b.value for b in req.backbones],
        )
    except Exception as e:
        logger.exception("Failed to start training")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=TrainStatusResponse)
async def get_train_status(job_id: str):
    """
    Poll the status and live metrics of a training job.
    Status values: 'queued' | 'running' | 'completed' | 'failed' | 'not_found'
    """
    status = get_job_status(job_id)
    return TrainStatusResponse(**status)
