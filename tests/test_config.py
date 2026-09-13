from __future__ import annotations

import pytest

from src.config import AppSettings, ConfigurationError, StorageMode, load_settings

CONFIG_ENVIRONMENT_VARIABLES = (
    "CHAT_MODEL",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_MODEL",
    "MAX_QUESTION_CHARS",
    "MAX_REQUESTS_PER_SESSION",
    "MAX_UPLOAD_MB",
    "QDRANT_API_KEY",
    "QDRANT_PATH",
    "QDRANT_URL",
    "RETRIEVAL_SCORE_THRESHOLD",
    "RETRIEVAL_TOP_K",
    "REQUEST_TIMEOUT_MS",
    "STORAGE_MODE",
)


@pytest.fixture(autouse=True)
def clear_config_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


def test_load_settings_uses_safe_defaults() -> None:
    settings = load_settings({})

    assert settings.storage_mode is StorageMode.LOCAL
    assert settings.chat_model == "gemini-3.5-flash-lite"
    assert settings.embedding_model == "gemini-embedding-2"
    assert settings.embedding_dimension == 768
    assert settings.max_question_chars == 4_000
    assert settings.retrieval_top_k == 6
    assert settings.retrieval_score_threshold == 0.5


def test_environment_takes_precedence_over_streamlit_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CHAT_MODEL", "model-from-environment")

    settings = load_settings({"CHAT_MODEL": "model-from-secrets"})

    assert settings.chat_model == "model-from-environment"


def test_cloud_mode_requires_both_credentials() -> None:
    with pytest.raises(ConfigurationError, match="QDRANT_URL"):
        load_settings({"STORAGE_MODE": "cloud"})


def test_invalid_integer_has_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MAX_UPLOAD_MB", "many")

    with pytest.raises(ConfigurationError, match="MAX_UPLOAD_MB"):
        load_settings({})


def test_retrieval_settings_are_loaded_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RETRIEVAL_TOP_K", "8")
    monkeypatch.setenv("RETRIEVAL_SCORE_THRESHOLD", "0.42")

    settings = load_settings({})

    assert settings.retrieval_top_k == 8
    assert settings.retrieval_score_threshold == 0.42

    with pytest.raises(ConfigurationError, match="RETRIEVAL_SCORE_THRESHOLD"):
        AppSettings(retrieval_score_threshold=1.1)


def test_public_diagnostics_never_contains_credentials() -> None:
    settings = AppSettings(qdrant_api_key="qdrant-private")

    serialized = repr(settings.public_diagnostics())

    assert "qdrant-private" not in serialized
