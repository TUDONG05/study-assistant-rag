"""Central ownership of Streamlit session keys and reset behavior."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any, cast
from uuid import uuid4

import streamlit as st

from src.config import StorageMode

WORKSPACE_ID = "workspace_id"
LOCAL_WORKSPACE_ID = "local-default"
MESSAGES = "messages"
SELECTED_DOCUMENT_IDS = "selected_document_ids"
RETRIEVAL_STRATEGY = "retrieval_strategy"
REQUEST_COUNT = "request_count"
GEMINI_API_KEY = "gemini_api_key"
ACTIVE_VIEW = "active_view"

SESSION_KEYS = (
    WORKSPACE_ID,
    MESSAGES,
    SELECTED_DOCUMENT_IDS,
    RETRIEVAL_STRATEGY,
    REQUEST_COUNT,
    GEMINI_API_KEY,
    ACTIVE_VIEW,
)

_DEFAULT_FACTORIES: dict[str, Callable[[], Any]] = {
    WORKSPACE_ID: lambda: uuid4().hex,
    MESSAGES: list,
    SELECTED_DOCUMENT_IDS: list,
    RETRIEVAL_STRATEGY: lambda: "advanced",
    REQUEST_COUNT: lambda: 0,
    GEMINI_API_KEY: str,
    ACTIVE_VIEW: lambda: "Tài liệu",
}


def _session() -> MutableMapping[str, Any]:
    return cast(MutableMapping[str, Any], st.session_state)


def initialize_session_state(state: MutableMapping[str, Any] | None = None) -> None:
    """Initialize all known keys without overwriting values across reruns."""

    target = state if state is not None else _session()
    for key, factory in _DEFAULT_FACTORIES.items():
        if key not in target:
            target[key] = factory()


def reset_session_state(state: MutableMapping[str, Any] | None = None) -> None:
    """Remove user-provided content and create a fresh isolated workspace."""

    target = state if state is not None else _session()
    for key in SESSION_KEYS:
        target.pop(key, None)
    initialize_session_state(target)


def configure_workspace(
    storage_mode: StorageMode,
    state: MutableMapping[str, Any] | None = None,
) -> None:
    """Use a stable namespace for persistent single-user local storage."""

    target = state if state is not None else _session()
    if storage_mode is StorageMode.LOCAL:
        target[WORKSPACE_ID] = LOCAL_WORKSPACE_ID


def active_gemini_api_key() -> str | None:
    """Return the current user's session-only BYOK value."""

    byok_value = str(_session().get(GEMINI_API_KEY, "")).strip()
    return byok_value or None


def current_workspace_id() -> str:
    """Return the namespace that every later data operation must filter by."""

    return str(_session()[WORKSPACE_ID])
