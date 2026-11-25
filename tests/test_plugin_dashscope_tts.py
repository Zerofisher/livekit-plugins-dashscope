# Copyright 2024 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit.agents import DEFAULT_API_CONNECT_OPTIONS
from livekit.agents.types import USERDATA_TIMED_TRANSCRIPT

pytestmark = pytest.mark.unit

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")

from livekit.plugins.dashscope import TTS  # noqa: E402
from livekit.plugins.dashscope.tts import ChunkedStream, _build_run_task  # noqa: E402

NO_RETRY_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)


def _make_task_started_msg(task_id: str = "test-task") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps({"header": {"event": "task-started", "task_id": task_id}}),
        extra=None,
    )


def _make_task_finished_msg(task_id: str = "test-task") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps({"header": {"event": "task-finished", "task_id": task_id}}),
        extra=None,
    )


def _make_task_failed_msg(code: str = "ERR_001", message: str = "tts failed") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "header": {
                    "event": "task-failed",
                    "error_code": code,
                    "error_message": message,
                }
            }
        ),
        extra=None,
    )


def _make_result_msg(words: list[dict] | None = None) -> aiohttp.WSMessage:
    payload: dict = {}
    if words is not None:
        payload = {
            "payload": {
                "output": {
                    "sentence": {
                        "index": 0,
                        "words": words,
                    }
                }
            }
        }

    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps({"header": {"event": "result-generated"}, **payload}),
        extra=None,
    )


def _make_binary_msg(data: bytes) -> aiohttp.WSMessage:
    return aiohttp.WSMessage(type=aiohttp.WSMsgType.BINARY, data=data, extra=None)


class MockWSConnection:
    def __init__(self, server_messages: list[aiohttp.WSMessage]) -> None:
        self.sent: list[str] = []
        self._messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        for msg in server_messages:
            self._messages.put_nowait(msg)
        self.closed = False
        self.close_code: int | None = None
        self._exception: BaseException | None = None

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def receive(self, timeout: float | None = None) -> aiohttp.WSMessage:
        try:
            return self._messages.get_nowait()
        except asyncio.QueueEmpty:
            await asyncio.sleep(60)
            return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

    async def close(self) -> None:
        self.closed = True

    def exception(self) -> BaseException | None:
        return self._exception


def _create_tts(**kwargs) -> TTS:  # type: ignore[type-arg]
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    return TTS(api_key="test-key", http_session=mock_session, **kwargs)


class TestTTSCapabilities:
    def test_capabilities(self) -> None:
        instance = _create_tts()
        assert instance.capabilities.streaming is True
        assert instance.capabilities.aligned_transcript is False
        assert instance.provider == "dashscope"
        assert instance.model == "cosyvoice-v3-flash"

    def test_word_timestamps_enable_aligned_transcript(self) -> None:
        instance = _create_tts(word_timestamp_enabled=True)
        assert instance.capabilities.aligned_transcript is True


class TestTTSProtocol:
    @pytest.mark.asyncio
    async def test_run_task_format(self) -> None:
        instance = _create_tts(
            model="cosyvoice-v2",
            voice="longxiaochun_v2",
            word_timestamp_enabled=True,
            language="zh",
        )
        stream = ChunkedStream(
            tts=instance,
            input_text="今天天气怎么样？",
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        try:
            msg = _build_run_task(stream._opts, stream._task_id)
            assert msg["header"]["action"] == "run-task"
            assert msg["header"]["streaming"] == "duplex"
            assert msg["payload"]["task"] == "tts"
            assert msg["payload"]["function"] == "SpeechSynthesizer"
            assert msg["payload"]["model"] == "cosyvoice-v2"
            assert msg["payload"]["parameters"]["voice"] == "longxiaochun_v2"
            assert msg["payload"]["parameters"]["format"] == "pcm"
            assert msg["payload"]["parameters"]["word_timestamp_enabled"] is True
            assert msg["payload"]["parameters"]["language_hints"] == ["zh"]
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_synthesize_pcm_audio_and_timestamps(self) -> None:
        instance = _create_tts(word_timestamp_enabled=True)
        audio_200ms = b"\x00" * (instance.sample_rate * 2 // 5)
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_binary_msg(audio_200ms),
                _make_result_msg(
                    [
                        {"text": "你", "begin_time": 0, "end_time": 80},
                        {"text": "好", "begin_time": 80, "end_time": 160},
                    ]
                ),
                _make_task_finished_msg(),
            ]
        )
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            async with instance.synthesize("你好") as stream:
                audio_events = [ev async for ev in stream]

        assert audio_events
        assert sum(ev.frame.duration for ev in audio_events) > 0
        assert audio_events[-1].is_final is True

        timed_words = [
            text
            for ev in audio_events
            for text in ev.frame.userdata.get(USERDATA_TIMED_TRANSCRIPT, [])
        ]
        assert [str(text) for text in timed_words] == ["你", "好"]

        sent_messages = [json.loads(msg) for msg in mock_ws.sent]
        assert sent_messages[0]["header"]["action"] == "run-task"
        assert sent_messages[1]["header"]["action"] == "continue-task"
        assert sent_messages[1]["payload"]["input"]["text"] == "你好"
        assert sent_messages[2]["header"]["action"] == "finish-task"
        assert mock_session.ws_connect.call_args is not None
        assert mock_session.ws_connect.call_args.kwargs["headers"]["Authorization"] == (
            "bearer test-key"
        )


class TestTTSErrorHandling:
    @pytest.mark.asyncio
    async def test_handshake_401_has_actionable_hint(self) -> None:
        from livekit.agents import APIStatusError

        instance = _create_tts()
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=aiohttp.WSServerHandshakeError(
                MagicMock(),
                (),
                status=401,
                message="Invalid response status",
                headers={"x-request-id": "req-123"},
            )
        )

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIStatusError) as excinfo:
                async with instance.synthesize("你好") as stream:
                    async for _ in stream:
                        pass

        assert "401 Unauthorized" in excinfo.value.message
        assert "DASHSCOPE_REGION" in excinfo.value.message
        assert "DASHSCOPE_BASE_URL" in excinfo.value.message
        assert "x-request-id=req-123" in excinfo.value.message

    @pytest.mark.asyncio
    async def test_task_failed_raises_api_error(self) -> None:
        from livekit.agents import APIStatusError

        instance = _create_tts()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_failed_msg(message="bad voice"),
            ]
        )
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIStatusError, match="bad voice"):
                async with instance.synthesize("你好", conn_options=NO_RETRY_OPTIONS) as stream:
                    async for _ in stream:
                        pass

    @pytest.mark.asyncio
    async def test_non_retryable_task_failed_is_not_retried(self) -> None:
        """A rejected voice/parameter fails on the first attempt even with retries allowed."""
        from livekit.agents import APIStatusError

        instance = _create_tts()
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=lambda *a, **kw: MockWSConnection(
                [_make_task_failed_msg(code="InvalidParameter", message="bad voice")]
            )
        )
        retry_options = dataclasses.replace(
            DEFAULT_API_CONNECT_OPTIONS, max_retry=3, retry_interval=0.01
        )

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            with pytest.raises(APIStatusError, match="bad voice") as excinfo:
                async with instance.synthesize("你好", conn_options=retry_options) as stream:
                    async for _ in stream:
                        pass

        assert excinfo.value.status_code == 400
        assert excinfo.value.retryable is False
        assert mock_session.ws_connect.call_count == 1


class TestTTSConnectionPool:
    """One duplex WebSocket runs many tasks: finished tasks return the socket for reuse."""

    @staticmethod
    def _task_messages() -> list[aiohttp.WSMessage]:
        return [
            _make_task_started_msg(),
            _make_binary_msg(b"\x00\x01" * 2205),
            _make_task_finished_msg(),
        ]

    async def _synthesize(self, instance: TTS, session: MagicMock) -> Any:  # type: ignore[type-arg]
        with patch.object(instance, "_ensure_session", return_value=session):
            async with instance.synthesize("你好", conn_options=NO_RETRY_OPTIONS) as stream:
                async for _ in stream:
                    pass
        return stream

    @pytest.mark.asyncio
    async def test_finished_task_socket_is_reused(self) -> None:
        instance = _create_tts()
        ws = MockWSConnection(self._task_messages() + self._task_messages())
        session = MagicMock(spec=aiohttp.ClientSession)
        session.ws_connect = AsyncMock(return_value=ws)

        first = await self._synthesize(instance, session)
        second = await self._synthesize(instance, session)

        assert session.ws_connect.call_count == 1
        assert first._connection_reused is False and second._connection_reused is True
        assert [json.loads(m)["header"]["action"] for m in ws.sent].count("run-task") == 2
        assert not ws.closed
        await instance.aclose()
        assert ws.closed

    @pytest.mark.asyncio
    async def test_failed_task_discards_socket(self) -> None:
        from livekit.agents import APIStatusError

        instance = _create_tts()
        broken = MockWSConnection([_make_task_started_msg(), _make_task_failed_msg()])
        fresh = MockWSConnection(self._task_messages())
        session = MagicMock(spec=aiohttp.ClientSession)
        session.ws_connect = AsyncMock(side_effect=[broken, fresh])

        with pytest.raises(APIStatusError):
            await self._synthesize(instance, session)
        await self._synthesize(instance, session)

        assert session.ws_connect.call_count == 2  # the failed task's socket was never reused
        assert broken.closed

    @pytest.mark.asyncio
    async def test_socket_closed_while_idle_is_skipped(self) -> None:
        instance = _create_tts()
        idle = MockWSConnection(self._task_messages())
        fresh = MockWSConnection(self._task_messages())
        session = MagicMock(spec=aiohttp.ClientSession)
        session.ws_connect = AsyncMock(side_effect=[idle, fresh])

        await self._synthesize(instance, session)
        idle.closed = True  # server closed the pooled socket
        await self._synthesize(instance, session)

        assert session.ws_connect.call_count == 2
