# Clinical Classifier API

REST API for binary classification of clinical samples using a PyTorch MLP model trained on the Breast Cancer Wisconsin dataset. Accepts 30 numeric features, returns a benign/malignant prediction with confidence scores.

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.11 |
| HTTP Framework | FastAPI 0.115 + Uvicorn |
| ML | PyTorch 2.4 + scikit-learn 1.5 |
| Database | PostgreSQL 16 + asyncpg + SQLAlchemy 2.0 |
| Migrations | Alembic |
| Monitoring | Prometheus (prometheus-fastapi-instrumentator) |
| Containerization | Docker |

---

## Running

### Option A — Local (recommended for development)

Requires Python 3.11+ and Docker (for the database).

```bash
# 1. Start only the database
docker-compose up db -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model (creates artifacts/model.pt and artifacts/scaler.pkl)
python -m ml.train

# 4. Start the API
uvicorn app.main:app --reload
```

API: `http://localhost:8000`
Interactive docs: `http://localhost:8000/docs`

---

### Option B — Full Docker Compose

```bash
# Train the model first (artifacts are volume-mounted into the container)
python -m ml.train

docker-compose up --build
```

> **Note:** `requirements.txt` does not include `torch` — the Dockerfile installs the CPU-only wheel separately to keep the image size small (~200 MB vs ~2 GB with CUDA).

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/clinical` | PostgreSQL connection string |
| `API_KEYS` | `["dev-key-change-me"]` | JSON array of valid API keys |

Copy `.env.example` → `.env` and adjust values, or export them in your shell. The `.env` file is gitignored and not loaded automatically — set variables explicitly or pass via `docker-compose.yml`.

---

## API Reference

### POST /api/v1/predict

Requires `X-API-Key` header.

**Request:**
```json
{ "features": [0.1, -0.5, 1.2, ...] }
```

`features` — exactly 30 floats, each in `[-10.0, 10.0]`.

**Response:**
```json
{
  "prediction": "benign",
  "confidence": 0.9823,
  "probabilities": { "benign": 0.9823, "malignant": 0.0177 },
  "model_version": "1.0.0"
}
```

**curl:**
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-key-change-me" \
  -d '{"features": [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]}'
```

---

### GET /api/v1/health

```bash
curl http://localhost:8000/api/v1/health
# {"status":"ok","model_loaded":true,"db_connected":true}
```

### GET /api/v1/version

```bash
curl http://localhost:8000/api/v1/version
# {"model_version":"1.0.0","api_version":"1.0.0"}
```

---

## Model

| Property | Value |
|---|---|
| Architecture | MLP: 30 → 64 → 32 → 2 (ReLU, Dropout 0.3) |
| Dataset | Breast Cancer Wisconsin (569 samples) |
| Train/test split | 80/20, stratified |
| Normalization | StandardScaler (fit on train set) |
| Loss | CrossEntropyLoss |
| Optimizer | Adam (lr=1e-3, 100 epochs, batch=32) |
| Test accuracy | ~97% |

Evaluate after training:

```bash
python -m ml.evaluate
```

---

## Model Versioning

To swap the model without rebuilding the Docker image:

1. Train: `python -m ml.train`
2. Update `config.yaml`:
   ```yaml
   model_version: "2.0.0"
   model_path: "artifacts/model_v2.pt"
   scaler_path: "artifacts/scaler_v2.pkl"
   ```
3. Restart: `docker-compose restart api`

The `./artifacts` directory is volume-mounted, so the container picks up new files immediately.

---

## Tests

Artifacts must exist before running tests.

```bash
python -m ml.train   # if not already done
pytest -v
pytest tests/test_predict.py -v
```

No PostgreSQL needed — the DB dependency is mocked in tests.

---

## Other Commands

```bash
# Lint
ruff check .

# Run Alembic migrations manually
alembic upgrade head
alembic downgrade -1
```
