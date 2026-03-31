import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import HealthResponse, VersionResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    db_connected = True
    try:
        await db.execute(text("SELECT 1"))
    except Exception:
        logger.warning("Database health check failed")
        db_connected = False

    model_loaded = request.app.state.predictor.is_loaded
    return HealthResponse(
        status="ok" if (model_loaded and db_connected) else "degraded",
        model_loaded=model_loaded,
        db_connected=db_connected,
    )


@router.get("/version", response_model=VersionResponse)
async def version(request: Request) -> VersionResponse:
    settings = request.app.state.settings
    return VersionResponse(
        model_version=settings.model_version,
        api_version=settings.api_version,
    )
