"""
Leaf-Expert — FastAPI Application Entry Point
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.logging import setup_logging, get_logger
from app.api.routes_data import router as data_router
from app.api.routes_train import router as train_router
from app.api.routes_predict import router as predict_router
from app.api.routes_vlm import router as vlm_router
from app.api.routes_datastore import router as datastore_router
from app.services.datastore_service import init_datastore

setup_logging(debug=settings.debug)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")
    # Create necessary directories
    for d in [settings.data_dir, settings.model_dir, settings.log_dir,
              settings.collected_data_dir]:
        d.mkdir(parents=True, exist_ok=True)
    # Initialise DataStore (creates SQLite tables)
    await init_datastore(
        db_path=settings.datastore_db_path,
        images_dir=settings.collected_data_dir,
        threshold=settings.vlm_training_threshold,
    )
    yield
    logger.info("Shutting down Leaf-Expert API")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Leaf-Expert API — AI-powered plant disease detection with Grad-CAM++ and LIME explanations. "
        "Supports no-code training via EfficientNetV2, ResNet, MobileNet, ViT, and more."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Routers ───────────────────────────────────────────────────────────────────
app.include_router(data_router)
app.include_router(train_router)
app.include_router(predict_router)
app.include_router(vlm_router)
app.include_router(datastore_router)


# ─── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health():
    import torch
    ds_stats = {}
    try:
        from app.services.datastore_service import get_datastore
        s = await get_datastore().get_stats()
        ds_stats = {"databank_total": s["total_points"], "databank_can_train": s["can_train"]}
    except Exception:
        pass
    return {
        "status": "ok",
        "version": settings.app_version,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": settings.device,
        **ds_stats,
    }
