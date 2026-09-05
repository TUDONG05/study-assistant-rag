"""Typed application settings with explicit source precedence."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class ConfigurationError(ValueError):
    """Raised when runtime settings are inconsistent or unsafe."""


class StorageMode(StrEnum):
    """Supported Qdrant deployment modes."""

    LOCAL = "local"
    DEMO = "demo"
    CLOUD = "cloud"


@dataclass(frozen=True, slots=True)
class AppSettings:
    """Validated settings safe to pass through application boundaries."""

    app_name: str = "Study Assistant"
    chat_model: str = "gemini-3.7-flash"
    embedding_model: str = "gemini-embedding-2"
    embedding_dimension: int = 768
    storage_mode: StorageMode = StorageMode.LOCAL
    qdrant_path: Path = Path(".data/qdrant")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = field(default=None, repr=False)
    request_timeout_ms: int = 60_000
    max_upload_mb: int = 25
    max_requests_per_session: int = 40

    def __post_init__(self) -> None:
        if self.embedding_dimension <= 0:
            raise ConfigurationError("EMBEDDING_DIMENSION phải lớn hơn 0.")
        if self.request_timeout_ms < 1_000:
            raise ConfigurationError("REQUEST_TIMEOUT_MS phải từ 1000 ms trở lên.")
        if self.max_upload_mb <= 0 or self.max_requests_per_session <= 0:
            raise ConfigurationError("Các giới hạn tài nguyên phải lớn hơn 0.")
        if self.storage_mode is StorageMode.CLOUD and (
            not self.qdrant_url or not self.qdrant_api_key
        ):
            raise ConfigurationError(
                "Chế độ cloud yêu cầu QDRANT_URL và QDRANT_API_KEY."
            )

    def public_diagnostics(self) -> dict[str, str | int]:
        """Return operational settings without credential material."""

        return {
            "Ứng dụng": self.app_name,
            "Chat model": self.chat_model,
            "Embedding model": self.embedding_model,
            "Embedding dimension": self.embedding_dimension,
            "Storage mode": self.storage_mode.value,
            "Upload limit (MB)": self.max_upload_mb,
            "Session request limit": self.max_requests_per_session,
        }


def load_settings(secrets: Mapping[str, Any] | None = None) -> AppSettings:
    """Load environment variables before Streamlit secrets and safe defaults."""

    source = secrets or {}
    storage_mode_value = _read("STORAGE_MODE", source, StorageMode.LOCAL.value)

    try:
        storage_mode = StorageMode(str(storage_mode_value).strip().lower())
    except ValueError as exc:
        allowed = ", ".join(mode.value for mode in StorageMode)
        raise ConfigurationError(f"STORAGE_MODE phải là một trong: {allowed}.") from exc

    return AppSettings(
        chat_model=str(_read("CHAT_MODEL", source, "gemini-3.7-flash")),
        embedding_model=str(_read("EMBEDDING_MODEL", source, "gemini-embedding-2")),
        embedding_dimension=_read_int("EMBEDDING_DIMENSION", source, 768),
        storage_mode=storage_mode,
        qdrant_path=Path(str(_read("QDRANT_PATH", source, ".data/qdrant"))),
        qdrant_url=_read_optional("QDRANT_URL", source),
        qdrant_api_key=_read_optional("QDRANT_API_KEY", source),
        request_timeout_ms=_read_int("REQUEST_TIMEOUT_MS", source, 60_000),
        max_upload_mb=_read_int("MAX_UPLOAD_MB", source, 25),
        max_requests_per_session=_read_int("MAX_REQUESTS_PER_SESSION", source, 40),
    )


def _read(name: str, secrets: Mapping[str, Any], default: Any) -> Any:
    environment_value = os.getenv(name)
    if environment_value is not None and environment_value.strip():
        return environment_value.strip()

    secret_value = secrets.get(name)
    if secret_value is not None and str(secret_value).strip():
        return secret_value

    return default


def _read_optional(name: str, secrets: Mapping[str, Any]) -> str | None:
    value = _read(name, secrets, None)
    return str(value).strip() if value is not None else None


def _read_int(name: str, secrets: Mapping[str, Any], default: int) -> int:
    value = _read(name, secrets, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(f"{name} phải là số nguyên.") from exc
