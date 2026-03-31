import logging
from pathlib import Path

import joblib
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from app.model import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"


def evaluate() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = joblib.load(SCALER_PATH)
    X_test_scaled = scaler.transform(X_test)

    model = MLPClassifier()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()

    logger.info("Accuracy: %.4f", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, target_names=["benign", "malignant"]))


if __name__ == "__main__":
    evaluate()
