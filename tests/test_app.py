from pathlib import Path

from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parents[1] / "app.py"


def test_app_starts_without_cloud_credentials() -> None:
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    assert not app.exception
    assert any(title.value == "Study Assistant" for title in app.title)
    assert app.subheader[0].value == "Tài liệu"


def test_navigation_renders_each_workspace() -> None:
    expected_headings = {
        "Tài liệu": "Tài liệu",
        "Chat": "Chat có trích dẫn",
        "Đánh giá": "Đánh giá RAG",
        "Công cụ học tập": "Công cụ học tập",
    }
    app = AppTest.from_file(APP_PATH).run(timeout=20)

    for view, heading in expected_headings.items():
        app.radio[0].set_value(view).run(timeout=20)
        assert not app.exception
        assert app.subheader[0].value == heading
