import logging
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn.functional as F

from app.config import Settings
from app.model import MLPClassifier
from app.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent


class Predictor:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model: MLPClassifier | None = None
        self._scaler = None
        self._loaded: bool = False

    def load(self) -> None:
        model_path = _ROOT / self._settings.model_path
        scaler_path = _ROOT / self._settings.scaler_path

        self._model = MLPClassifier()
        self._model.load_state_dict(
            torch.load(model_path, map_location="cpu", weights_only=True)
        )
        self._model.eval()

        self._scaler = joblib.load(scaler_path)
        self._loaded = True
        logger.info("Model loaded from %s", model_path)
        logger.info("Scaler loaded from %s", scaler_path)

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def predict(self, request: PredictRequest) -> PredictResponse:
        if not self._loaded:
            raise RuntimeError("Predictor is not loaded. Call .load() first.")

        features = np.array(request.features, dtype=np.float32).reshape(1, -1)
        scaled = self._scaler.transform(features)
        tensor = torch.tensor(scaled, dtype=torch.float32)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).numpy()

        class_idx = int(probs.argmax())
        classes = self._settings.classes
        label = classes[class_idx]
        confidence = round(float(probs[class_idx]), 4)
        probabilities = {classes[i]: round(float(p), 4) for i, p in enumerate(probs)}

        return PredictResponse(
            prediction=label,
            confidence=confidence,
            probabilities=probabilities,
            model_version=self._settings.model_version,
        )
