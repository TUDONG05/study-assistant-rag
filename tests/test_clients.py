from __future__ import annotations

from pathlib import Path

import pytest

from src.clients import (
    _cached_local_qdrant_client,
    get_gemini_client,
    get_qdrant_client,
)
from src.config import AppSettings, ConfigurationError, StorageMode


def test_empty_gemini_key_is_rejected_before_client_creation() -> None:
    with pytest.raises(ConfigurationError, match="Gemini API key"):
        get_gemini_client("  ", 60_000)


def test_local_qdrant_reuses_client_when_workspace_rotates(tmp_path: Path) -> None:
    settings = AppSettings(
        storage_mode=StorageMode.LOCAL,
        qdrant_path=tmp_path / "qdrant",
    )
    _cached_local_qdrant_client.clear()

    try:
        first_client = get_qdrant_client(settings, "workspace-one")
        second_client = get_qdrant_client(settings, "workspace-two")

        assert first_client is second_client
    finally:
        _cached_local_qdrant_client.clear()
