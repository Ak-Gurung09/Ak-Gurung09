"""Smoke-test the Streamlit dashboard headlessly (skipped if streamlit is not installed)."""

import os

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("folium")
from streamlit.testing.v1 import AppTest  # noqa: E402

from changelens.pipeline import run_dry_run  # noqa: E402

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "dashboard.py")


def run_app(out_dir, monkeypatch):
    monkeypatch.setenv("CHANGELENS_OUTPUTS", str(out_dir))
    at = AppTest.from_file(APP, default_timeout=30)
    at.run()
    assert not at.exception, at.exception
    return at


def test_dashboard_renders_synthetic_case_with_warnings(tmp_path, monkeypatch):
    run_dry_run(str(tmp_path))
    at = run_app(tmp_path, monkeypatch)
    assert any("not** an official" in w.value for w in at.warning)
    assert any("SYNTHETIC" in e.value for e in at.error)
    assert any(m.label == "Flagged areas" for m in at.metric)


def test_unsupported_hazard_shows_explanation_not_map(tmp_path, monkeypatch):
    run_dry_run(str(tmp_path))
    at = run_app(tmp_path, monkeypatch)
    at.sidebar.radio[0].set_value("Landslide").run()
    assert not at.exception
    assert any("not available" in h.value for h in at.subheader)


def test_empty_outputs_folder(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch)
    assert any("No results found" in i.value for i in at.info)


def test_offline_map_mode(tmp_path, monkeypatch):
    run_dry_run(str(tmp_path))
    at = run_app(tmp_path, monkeypatch)
    at.sidebar.radio[2].set_value("Offline image").run()
    assert not at.exception
    assert len(at.get("image")) >= 1
