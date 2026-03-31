from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple, Type

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

_CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


class YamlSettingsSource(PydanticBaseSettingsSource):
    """Load settings from config.yaml, allowing env vars to override."""

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        data = self._load_yaml()
        value = data.get(field_name)
        return value, field_name, isinstance(value, dict)

    @staticmethod
    def _load_yaml() -> dict[str, Any]:
        with open(_CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def __call__(self) -> dict[str, Any]:
        yaml_data = self._load_yaml()
        return {k: v for k, v in yaml_data.items() if k in self.settings_cls.model_fields}


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    model_version: str
    model_path: str
    scaler_path: str
    classes: dict[int, str]
    api_version: str = "1.0.0"

    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/clinical",
        validation_alias="DATABASE_URL",
    )
    api_keys: list[str] = Field(
        default=["dev-key-change-me"],
        validation_alias="API_KEYS",
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        return (init_settings, env_settings, YamlSettingsSource(settings_cls))


@lru_cache
def get_settings() -> Settings:
    return Settings()
