from __future__ import annotations

from typing import Any

from src.ui.session_state import (
    MESSAGES,
    REQUEST_COUNT,
    WORKSPACE_ID,
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


def test_reset_clears_content_and_rotates_workspace() -> None:
    state: dict[str, Any] = {}
    initialize_session_state(state)
    old_workspace_id = state[WORKSPACE_ID]
    state[MESSAGES].append({"role": "user", "content": "Dữ liệu cũ"})
    state[REQUEST_COUNT] = 9

    reset_session_state(state)

    assert state[WORKSPACE_ID] != old_workspace_id
    assert state[MESSAGES] == []
    assert state[REQUEST_COUNT] == 0
