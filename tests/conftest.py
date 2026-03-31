import os

# Must be set before app is imported so get_settings() picks it up
os.environ["API_KEYS"] = '["test-api-key"]'

from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.database import get_db  # noqa: E402
from app.main import app  # noqa: E402

TEST_API_KEY = "test-api-key"


async def mock_get_db() -> AsyncMock:
    yield AsyncMock()


app.dependency_overrides[get_db] = mock_get_db


@pytest.fixture(scope="session")
def client() -> TestClient:
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def auth_headers() -> dict[str, str]:
    return {"X-API-Key": TEST_API_KEY}
