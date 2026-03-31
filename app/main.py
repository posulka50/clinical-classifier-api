import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from app.config import get_settings
from app.database import close_db, init_db
from app.logging_config import configure_logging
from app.predictor import Predictor
from app.routers import health, predict

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    configure_logging()

    settings = get_settings()
    init_db(settings.database_url)

    predictor = Predictor(settings)
    predictor.load()

    app.state.settings = settings
    app.state.predictor = predictor

    logger.info("startup_complete", extra={"model_version": settings.model_version})
    yield

    await close_db()
    logger.info("shutdown")


app = FastAPI(
    title="Clinical Classifier API",
    version="1.0.0",
    lifespan=lifespan,
)

Instrumentator(
    should_group_status_codes=True,
    should_ignore_untemplated=True,
).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

app.include_router(predict.router, prefix="/api/v1")
app.include_router(health.router, prefix="/api/v1")
