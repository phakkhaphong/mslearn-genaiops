"""Unit tests for cloud evaluation polling and resume helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "evaluators" / "evaluate_agent.py"
)


def load_evaluate_agent():
    """Load evaluate_agent.py by path so tests do not require a package layout."""
    spec = importlib.util.spec_from_file_location("evaluate_agent", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_agent"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


ea = load_evaluate_agent()


class NotFoundError(Exception):
    """Minimal 404 stand-in used by the OpenAI-compatible client."""

    def __init__(self, message="Error code: 404 - Project not found"):
        super().__init__(message)
        self.status_code = 404


def test_is_not_found_detects_status_code_and_message():
    """404 detection should cover both status_code and Foundry error text."""
    assert ea.is_not_found(NotFoundError()) is True
    assert ea.is_not_found(RuntimeError("Project not found")) is True
    assert ea.is_not_found(RuntimeError("quota exceeded")) is False


def test_parse_args_resume_flags():
    """CLI flags should populate eval_id and run_id for resume mode."""
    args = ea.parse_args(
        ["--eval-id", "eval_abc", "--run-id", "evalrun_xyz"]
    )
    assert args.eval_id == "eval_abc"
    assert args.run_id == "evalrun_xyz"


def test_fetch_eval_run_falls_back_to_list_on_404():
    """When retrieve 404s, list() should still return the matching run."""
    run = SimpleNamespace(id="evalrun_xyz", status="in_progress")
    openai_client = MagicMock()
    openai_client.evals.runs.retrieve.side_effect = NotFoundError()
    openai_client.evals.runs.list.return_value = [run]

    result = ea.fetch_eval_run(openai_client, "eval_abc", "evalrun_xyz")

    assert result is run
    openai_client.evals.runs.list.assert_called_once_with(eval_id="eval_abc")


def test_poll_for_results_retries_404_then_completes(monkeypatch):
    """A transient 404 after create must not abort a live cloud run."""
    clock = {"t": 0.0}

    def fake_time():
        return clock["t"]

    def fake_sleep(seconds):
        clock["t"] += seconds

    monkeypatch.setattr(ea.time, "time", fake_time)

    run_queued = SimpleNamespace(id="evalrun_xyz", status="queued")
    run_done = SimpleNamespace(id="evalrun_xyz", status="completed")
    openai_client = MagicMock()
    openai_client.evals.runs.retrieve.side_effect = [
        NotFoundError(),
        run_queued,
        run_done,
    ]
    openai_client.evals.runs.list.return_value = []

    result = ea.poll_for_results(
        SimpleNamespace(id="eval_abc"),
        SimpleNamespace(id="evalrun_xyz"),
        openai_client=openai_client,
        sleep_fn=fake_sleep,
        poll_interval=10,
        not_found_retry_seconds=180,
    )

    assert result.status == "completed"
    assert openai_client.evals.runs.retrieve.call_count == 3


def test_poll_for_results_raises_on_failed_run():
    """A completed-but-failed cloud run should surface as RuntimeError."""
    openai_client = MagicMock()
    openai_client.evals.runs.retrieve.return_value = SimpleNamespace(
        id="evalrun_xyz",
        status="failed",
        error="judge quota exceeded",
    )

    with pytest.raises(RuntimeError, match="judge quota exceeded"):
        ea.poll_for_results(
            SimpleNamespace(id="eval_abc"),
            SimpleNamespace(id="evalrun_xyz"),
            openai_client=openai_client,
            sleep_fn=lambda _seconds: None,
            poll_interval=0,
        )
