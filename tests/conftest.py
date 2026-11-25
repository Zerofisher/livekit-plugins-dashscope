"""Fixtures carried over from the livekit/agents monorepo test suite (the subset these tests use)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest

from livekit.agents import utils


@pytest.fixture
async def job_process() -> AsyncIterator[None]:
    utils.http_context._new_session_ctx()
    yield
    await utils.http_context._close_http_ctx()


def _is_ignorable_task(task: asyncio.Task[Any]) -> bool:
    try:
        coro: Any = task.get_coro()
        # async_generator_athrow tasks are created by the GC when finalizing async generators
        coro_name = getattr(coro, "__qualname__", "") or type(coro).__name__
        if "async_generator_athrow" in coro_name:
            return True
        if "pytest" in getattr(coro, "__module__", ""):
            return True
        for frame in task.get_stack():
            if "pytest" in frame.f_code.co_filename:
                return True
    except Exception:
        pass
    return False


def _format_task(task: asyncio.Task[Any]) -> str:
    coro: Any = task.get_coro()
    coro_name = getattr(coro, "__qualname__", None) or type(coro).__name__
    frame = getattr(coro, "cr_frame", None)
    location = f"{frame.f_code.co_filename}:{frame.f_lineno}" if frame else "no frame available"
    return f"Task {task.get_name()} ({coro_name}) at {location}"


@pytest.fixture(autouse=True)
async def fail_on_leaked_tasks() -> AsyncIterator[None]:
    """Every stream must be closed by the test; a pending task left behind is a leak."""
    tasks_before = set(asyncio.all_tasks())
    yield
    leaked = [
        t
        for t in set(asyncio.all_tasks()) - tasks_before
        if not t.done() and not _is_ignorable_task(t)
    ]
    if leaked:
        pytest.fail("Test leaked tasks:\n\n" + "\n".join(_format_task(t) for t in leaked))
