import logging

from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.dependencies import verify_api_key
from app.db_models import PredictionLog
from app.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(verify_api_key)],
)
async def predict(
    request: Request,
    body: PredictRequest,
    db: AsyncSession = Depends(get_db),
) -> PredictResponse:
    predictor = request.app.state.predictor
    result = predictor.predict(body)

    try:
        db.add(
            PredictionLog(
                prediction=result.prediction,
                confidence=result.confidence,
                model_version=result.model_version,
                features=body.features,
            )
        )
        await db.commit()
    except Exception:
        logger.exception("Failed to persist prediction log")
        await db.rollback()

    logger.info(
        "predict",
        extra={"prediction": result.prediction, "confidence": result.confidence},
    )
    return result
