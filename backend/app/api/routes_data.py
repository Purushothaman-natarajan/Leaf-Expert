"""
Leaf-Expert — API Routes: Data Preparation
"""
from fastapi import APIRouter, HTTPException
from app.schemas.data import DataPrepRequest, DataPrepResponse
from app.services.data_service import prepare_dataset
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/data", tags=["Data Preparation"])


@router.post("/prepare", response_model=DataPrepResponse)
async def prepare_data(req: DataPrepRequest):
    """
    Split a raw dataset into train/val/test folders with optional augmentation.
    The raw dataset must follow the structure: raw_path/<class_name>/*.jpg
    """
    try:
        result = prepare_dataset(
            raw_dataset_path=req.raw_dataset_path,
            target_folder=req.target_folder,
            image_size=req.image_size,
            augment=req.augment,
            train_ratio=req.train_ratio,
            val_ratio=req.val_ratio,
        )
        return DataPrepResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Data preparation failed")
        raise HTTPException(status_code=500, detail=str(e))
