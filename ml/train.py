import logging
from pathlib import Path

import joblib
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from app.model import MLPClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts"
MODEL_PATH = ARTIFACTS_DIR / "model.pt"
SCALER_PATH = ARTIFACTS_DIR / "scaler.pkl"

EPOCHS = 100
BATCH_SIZE = 32
LR = 1e-3


def train() -> None:
    ARTIFACTS_DIR.mkdir(exist_ok=True)

    data = load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)
    logger.info("Scaler saved to %s", SCALER_PATH)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)

    model = MLPClassifier()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if epoch % 20 == 0:
            logger.info("Epoch %d/%d  loss=%.4f", epoch, EPOCHS, total_loss / len(loader))

    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Model saved to %s", MODEL_PATH)

    model.eval()
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        preds = model(X_test_t).argmax(dim=1).numpy()
    accuracy = (preds == y_test).mean()
    logger.info("Test accuracy: %.4f", accuracy)


if __name__ == "__main__":
    train()
