from __future__ import annotations

from typing import Any

from src.config import StorageMode
from src.ui.session_state import (
    LOCAL_WORKSPACE_ID,
    MESSAGES,
    REQUEST_COUNT,
    RETRIEVAL_STRATEGY,
    SELECTED_DOCUMENT_IDS,
    WORKSPACE_ID,
    configure_workspace,
    initialize_session_state,
    reset_session_state,
)


def test_initialize_preserves_state_across_reruns() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)
    workspace_id = state[WORKSPACE_ID]
    state[MESSAGES].append({"role": "user", "content": "Xin chào"})

    initialize_session_state(state)

    assert state[WORKSPACE_ID] == workspace_id
    assert state[MESSAGES] == [{"role": "user", "content": "Xin chào"}]
    assert state[RETRIEVAL_STRATEGY] == "dense"


def test_reset_clears_content_and_rotates_workspace() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)
    old_workspace_id = state[WORKSPACE_ID]
    state[MESSAGES].append({"role": "user", "content": "Dữ liệu cũ"})
    state[REQUEST_COUNT] = 9
    state[SELECTED_DOCUMENT_IDS] = ["document-a"]

    reset_session_state(state)

    assert state[WORKSPACE_ID] != old_workspace_id
    assert state[MESSAGES] == []
    assert state[REQUEST_COUNT] == 0
    assert state[SELECTED_DOCUMENT_IDS] == []


def test_local_storage_uses_stable_workspace_across_sessions() -> None:
    first_state: dict[str, Any] = {}
    second_state: dict[str, Any] = {}
    initialize_session_state(first_state)
    initialize_session_state(second_state)

    configure_workspace(StorageMode.LOCAL, first_state)
    configure_workspace(StorageMode.LOCAL, second_state)

    assert first_state[WORKSPACE_ID] == LOCAL_WORKSPACE_ID
    assert second_state[WORKSPACE_ID] == LOCAL_WORKSPACE_ID
