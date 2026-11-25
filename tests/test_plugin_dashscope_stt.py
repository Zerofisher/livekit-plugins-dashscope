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

"""Unit tests for livekit-plugins-dashscope STT plugin.

All tests use mocked WebSocket connections — no real DashScope server is needed.
Tests go through the full stream lifecycle (stt.stream() -> async with -> aclose)
to avoid leaking background tasks, per conftest.py's `fail_on_leaked_tasks`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from livekit import rtc
from livekit.agents import DEFAULT_API_CONNECT_OPTIONS, stt
from livekit.agents.stt import SpeechEventType

pytestmark = pytest.mark.unit

os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")

from livekit.plugins.dashscope import STT, SpeechStream  # noqa: E402

NO_RETRY_OPTIONS = dataclasses.replace(DEFAULT_API_CONNECT_OPTIONS, max_retry=0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_started_msg(task_id: str = "test-task") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "header": {"event": "task-started", "task_id": task_id},
            }
        ),
        extra=None,
    )


def _make_task_finished_msg(task_id: str = "test-task") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "header": {"event": "task-finished", "task_id": task_id},
            }
        ),
        extra=None,
    )


def _make_task_failed_msg(code: str = "ERR_001", message: str = "test error") -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "header": {
                    "event": "task-failed",
                    "error_code": code,
                    "error_message": message,
                },
            }
        ),
        extra=None,
    )


def _make_result_msg(
    text: str,
    sentence_end: bool = False,
    begin_time: int | None = 0,
    end_time: int | None = None,
    words: list[dict] | None = None,  # type: ignore[type-arg]
    heartbeat: bool = False,
    usage: dict | None = None,  # type: ignore[type-arg]
) -> aiohttp.WSMessage:
    sentence: dict = {  # type: ignore[type-arg]
        "text": text,
        "sentence_end": sentence_end,
    }
    if begin_time is not None:
        sentence["begin_time"] = begin_time
    if end_time is not None:
        sentence["end_time"] = end_time
    if words is not None:
        sentence["words"] = words
    if heartbeat:
        sentence["heartbeat"] = True

    payload: dict = {"output": {"sentence": sentence}}  # type: ignore[type-arg]
    if usage is not None:
        payload["usage"] = usage

    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.TEXT,
        data=json.dumps(
            {
                "header": {"event": "result-generated"},
                "payload": payload,
            }
        ),
        extra=None,
    )


def _make_closed_msg() -> aiohttp.WSMessage:
    return aiohttp.WSMessage(
        type=aiohttp.WSMsgType.CLOSED,
        data=None,
        extra=None,
    )


class MockWSConnection:
    """Simulates aiohttp.ClientWebSocketResponse for testing."""

    def __init__(self, server_messages: list[aiohttp.WSMessage]) -> None:
        self.sent: list[bytes | str] = []
        self._messages: asyncio.Queue[aiohttp.WSMessage] = asyncio.Queue()
        for msg in server_messages:
            self._messages.put_nowait(msg)
        self.closed = False
        self.close_code: int | None = None

    async def send_str(self, data: str) -> None:
        self.sent.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent.append(data)

    async def receive(self) -> aiohttp.WSMessage:
        try:
            return self._messages.get_nowait()
        except asyncio.QueueEmpty:
            # Block until close or new message
            await asyncio.sleep(60)
            return aiohttp.WSMessage(type=aiohttp.WSMsgType.CLOSED, data=None, extra=None)

    async def close(self) -> None:
        self.closed = True


def _create_stt(**kwargs) -> STT:  # type: ignore[type-arg]
    """Create an STT instance with a fake session to avoid http_context dependency."""
    mock_session = MagicMock(spec=aiohttp.ClientSession)
    return STT(api_key="test-key", http_session=mock_session, **kwargs)


async def _run_stream_with_mock(
    stt_instance: STT,
    mock_ws: MockWSConnection,
    audio_frames: list[rtc.AudioFrame] | None = None,
    collect_timeout: float = 2.0,
    conn_options: stt.APIConnectOptions | None = None,
) -> list[stt.SpeechEvent]:
    """Run a stream through its full lifecycle with mocked WebSocket, collecting events."""
    events: list[stt.SpeechEvent] = []

    with patch.object(
        stt_instance, "_ensure_session", return_value=MagicMock(spec=aiohttp.ClientSession)
    ) as mock_ensure:
        mock_ensure.return_value.ws_connect = AsyncMock(return_value=mock_ws)

        kwargs: dict = {}  # type: ignore[type-arg]
        if conn_options is not None:
            kwargs["conn_options"] = conn_options
        stream = stt_instance.stream(**kwargs)
        try:
            # Push audio frames if provided
            if audio_frames:
                for frame in audio_frames:
                    stream.push_frame(frame)

            # End input to trigger finish-task
            stream.end_input()

            # Collect events with timeout
            try:
                async with asyncio.timeout(collect_timeout):
                    async for ev in stream:
                        events.append(ev)
            except (TimeoutError, asyncio.TimeoutError):
                pass

        finally:
            await stream.aclose()

    return events


def _make_pcm_frame(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    """Create a silent PCM audio frame."""
    num_samples = int(sample_rate * duration_ms / 1000)
    data = bytes(num_samples * num_channels * 2)  # 16-bit PCM = 2 bytes per sample
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=num_samples,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSTTCapabilities:
    """Tests 1-3: capabilities, provider/model, api_key."""

    def test_capabilities(self) -> None:
        """Test 1: streaming=True, interim_results=True, offline_recognize=False."""
        instance = _create_stt()
        assert instance.capabilities.streaming is True
        assert instance.capabilities.interim_results is True
        assert instance.capabilities.aligned_transcript == "word"
        assert instance.capabilities.offline_recognize is False

    def test_provider_and_model(self) -> None:
        """Test 2: provider=='dashscope', model=='fun-asr-realtime'."""
        instance = _create_stt()
        assert instance.provider == "dashscope"
        assert instance.model == "fun-asr-realtime"

    def test_custom_model(self) -> None:
        """Custom model name is reflected in .model property."""
        instance = _create_stt(model="custom-model")
        assert instance.model == "custom-model"

    def test_api_key_from_env(self) -> None:
        """Test 3: API key read from DASHSCOPE_API_KEY env var."""
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-test-key"}):
            instance = STT(http_session=MagicMock(spec=aiohttp.ClientSession))
            assert instance._opts.api_key == "env-test-key"

    def test_api_key_missing_raises(self) -> None:
        """Missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove the env var entirely
            env = dict(os.environ)
            env.pop("DASHSCOPE_API_KEY", None)
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="API key is required"):
                    STT(http_session=MagicMock(spec=aiohttp.ClientSession))

    def test_base_url_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "env-test-key",
                "DASHSCOPE_BASE_URL": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/",
            },
            clear=True,
        ):
            instance = STT(http_session=MagicMock(spec=aiohttp.ClientSession))
            assert (
                instance._opts.base_url == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/"
            )

    def test_region_from_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "env-test-key",
                "DASHSCOPE_REGION": "singapore",
            },
            clear=True,
        ):
            instance = STT(http_session=MagicMock(spec=aiohttp.ClientSession))
            assert (
                instance._opts.base_url == "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/"
            )

    def test_explicit_base_url_overrides_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "env-test-key",
                "DASHSCOPE_REGION": "singapore",
                "DASHSCOPE_BASE_URL": "wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/",
            },
            clear=True,
        ):
            instance = STT(
                http_session=MagicMock(spec=aiohttp.ClientSession),
                base_url="wss://dashscope.aliyuncs.com/api-ws/v1/inference/",
            )
            assert instance._opts.base_url == "wss://dashscope.aliyuncs.com/api-ws/v1/inference/"

    def test_invalid_region_raises(self) -> None:
        with patch.dict(
            os.environ,
            {
                "DASHSCOPE_API_KEY": "env-test-key",
                "DASHSCOPE_REGION": "moon-base-1",
            },
            clear=True,
        ):
            with pytest.raises(ValueError, match="Unsupported DASHSCOPE_REGION"):
                STT(http_session=MagicMock(spec=aiohttp.ClientSession))


class TestRunTaskFormat:
    """Test 4: run-task JSON structure."""

    @pytest.mark.asyncio
    async def test_run_task_format(self) -> None:
        """Handshake JSON has correct structure: action, task_id, model, parameters."""
        instance = _create_stt()
        stream = SpeechStream(
            stt=instance,
            opts=instance._opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        try:
            msg = stream._build_run_task()

            assert msg["header"]["action"] == "run-task"
            assert msg["header"]["streaming"] == "duplex"
            assert "task_id" in msg["header"]
            assert msg["payload"]["model"] == "fun-asr-realtime"
            assert msg["payload"]["task_group"] == "audio"
            assert msg["payload"]["task"] == "asr"
            assert msg["payload"]["function"] == "recognition"

            params = msg["payload"]["parameters"]
            assert params["format"] == "pcm"
            assert params["sample_rate"] == 16000
            assert params["max_sentence_silence"] == 700
            assert params["heartbeat"] is True
            assert params["language_hints"] == ["zh"]
        finally:
            await stream.aclose()


class TestProtocol:
    """Tests 5-6: audio chunking and finish-task."""

    @pytest.mark.asyncio
    async def test_connect_uses_lowercase_bearer_header(self) -> None:
        instance = _create_stt()
        mock_ws = MockWSConnection([])
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream(conn_options=NO_RETRY_OPTIONS)
            try:
                await stream._connect_ws()  # pyright: ignore[reportPrivateUsage]
                assert mock_session.ws_connect.call_args is not None
                assert (
                    mock_session.ws_connect.call_args.kwargs["headers"]["Authorization"]
                    == "bearer test-key"
                )
            finally:
                await stream.aclose()

    @pytest.mark.asyncio
    async def test_audio_chunking_100ms(self) -> None:
        """Test 5: PCM frames are re-chunked into 3200-byte (100ms) chunks."""
        instance = _create_stt()

        # Server sends: task-started, then task-finished after a brief delay
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        # Push a 200ms frame = 6400 bytes, should be split into 2 x 3200
        frame_200ms = _make_pcm_frame(duration_ms=200)
        await _run_stream_with_mock(instance, mock_ws, audio_frames=[frame_200ms])

        # Check sent bytes: first message is run-task (str), then binary chunks
        binary_msgs = [m for m in mock_ws.sent if isinstance(m, bytes)]
        for chunk in binary_msgs:
            assert len(chunk) == 3200, f"Expected 3200 bytes, got {len(chunk)}"
        assert len(binary_msgs) == 2

    @pytest.mark.asyncio
    async def test_finish_task_on_end_input(self) -> None:
        """Test 6: finish-task JSON is sent when input ends."""
        instance = _create_stt()

        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        await _run_stream_with_mock(instance, mock_ws)

        # Find finish-task in sent messages
        str_msgs = [m for m in mock_ws.sent if isinstance(m, str)]
        finish_msgs = [json.loads(m) for m in str_msgs if "finish-task" in m]
        assert len(finish_msgs) == 1
        assert finish_msgs[0]["header"]["action"] == "finish-task"


class TestEventMapping:
    """Tests 7-12: event mapping from DashScope to LiveKit."""

    @pytest.mark.asyncio
    async def test_interim_event_mapping(self) -> None:
        """Test 7: sentence_end=false -> INTERIM_TRANSCRIPT."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("你好", sentence_end=False, begin_time=100, end_time=500),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        interim_events = [e for e in events if e.type == SpeechEventType.INTERIM_TRANSCRIPT]
        assert len(interim_events) == 1
        assert interim_events[0].alternatives[0].text == "你好"
        assert interim_events[0].alternatives[0].language == "zh"

    @pytest.mark.asyncio
    async def test_final_event_mapping(self) -> None:
        """Test 8: sentence_end=true -> FINAL_TRANSCRIPT + END_OF_SPEECH."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("你好世界", sentence_end=True, begin_time=0, end_time=1500),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        event_types = [e.type for e in events]
        assert SpeechEventType.START_OF_SPEECH in event_types
        assert SpeechEventType.FINAL_TRANSCRIPT in event_types
        assert SpeechEventType.END_OF_SPEECH in event_types

        final = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT][0]
        assert final.alternatives[0].text == "你好世界"

    @pytest.mark.asyncio
    async def test_speech_lifecycle_start_end(self) -> None:
        """Test 9: START -> INTERIM -> FINAL -> END complete lifecycle."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("你", sentence_end=False, begin_time=100),
                _make_result_msg("你好", sentence_end=False, begin_time=100, end_time=300),
                _make_result_msg("你好世界", sentence_end=True, begin_time=100, end_time=1500),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        event_types = [e.type for e in events]
        # Should be: START, INTERIM, INTERIM, FINAL, END_OF_SPEECH
        assert event_types[0] == SpeechEventType.START_OF_SPEECH
        assert event_types[1] == SpeechEventType.INTERIM_TRANSCRIPT
        assert event_types[2] == SpeechEventType.INTERIM_TRANSCRIPT
        assert event_types[3] == SpeechEventType.FINAL_TRANSCRIPT
        # There might be RECOGNITION_USAGE before END_OF_SPEECH if usage is present
        assert SpeechEventType.END_OF_SPEECH in event_types

    @pytest.mark.asyncio
    async def test_heartbeat_ignored(self) -> None:
        """Test 10: heartbeat=true result-generated produces no events."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("", heartbeat=True),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        # No speech events should be emitted for heartbeat
        speech_events = [
            e
            for e in events
            if e.type
            in (
                SpeechEventType.START_OF_SPEECH,
                SpeechEventType.INTERIM_TRANSCRIPT,
                SpeechEventType.FINAL_TRANSCRIPT,
                SpeechEventType.END_OF_SPEECH,
            )
        ]
        assert len(speech_events) == 0

    @pytest.mark.asyncio
    async def test_empty_text_ignored(self) -> None:
        """Test 11: empty text result-generated produces no events."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("", sentence_end=False),
                _make_result_msg("  ", sentence_end=False),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        speech_events = [
            e
            for e in events
            if e.type
            in (
                SpeechEventType.START_OF_SPEECH,
                SpeechEventType.INTERIM_TRANSCRIPT,
                SpeechEventType.FINAL_TRANSCRIPT,
            )
        ]
        assert len(speech_events) == 0

    @pytest.mark.asyncio
    async def test_multiple_sentences(self) -> None:
        """Test 12: two sentences have separate START->END lifecycles, different request_ids."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                # First sentence
                _make_result_msg("你好", sentence_end=False, begin_time=0),
                _make_result_msg("你好世界", sentence_end=True, begin_time=0, end_time=1500),
                # Second sentence
                _make_result_msg("谢谢", sentence_end=False, begin_time=2000),
                _make_result_msg("谢谢你", sentence_end=True, begin_time=2000, end_time=3000),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        start_events = [e for e in events if e.type == SpeechEventType.START_OF_SPEECH]
        end_events = [e for e in events if e.type == SpeechEventType.END_OF_SPEECH]
        final_events = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]

        assert len(start_events) == 2
        assert len(end_events) == 2
        assert len(final_events) == 2

        # Request IDs should be different
        assert final_events[0].request_id != final_events[1].request_id

        assert final_events[0].alternatives[0].text == "你好世界"
        assert final_events[1].alternatives[0].text == "谢谢你"


class TestErrorHandling:
    """Tests 13-16: error conditions."""

    @pytest.mark.asyncio
    async def test_handshake_401_has_actionable_hint(self) -> None:
        from livekit.agents import APIStatusError

        instance = _create_stt()
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
            stream = instance.stream(conn_options=NO_RETRY_OPTIONS)
            try:
                with pytest.raises(APIStatusError) as excinfo:
                    await stream._connect_ws()  # pyright: ignore[reportPrivateUsage]

                assert "401 Unauthorized" in excinfo.value.message
                assert "DASHSCOPE_REGION" in excinfo.value.message
                assert "DASHSCOPE_BASE_URL" in excinfo.value.message
                assert "x-request-id=req-123" in excinfo.value.message
            finally:
                await stream.aclose()

    @pytest.mark.asyncio
    async def test_task_failed_raises_api_error(self) -> None:
        """Test 13: task-failed event raises APIStatusError."""
        from livekit.agents import APIStatusError

        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_failed_msg(code="ERR_001", message="test error"),
                _make_closed_msg(),
            ]
        )

        with pytest.raises(APIStatusError, match="test error"):
            await _run_stream_with_mock(
                instance, mock_ws, collect_timeout=5.0, conn_options=NO_RETRY_OPTIONS
            )

    @pytest.mark.asyncio
    async def test_unexpected_close_raises_api_error(self) -> None:
        """Test 14: connection closes without task-finished -> APIStatusError."""
        from livekit.agents import APIStatusError

        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                # No task-finished before close
                _make_closed_msg(),
            ]
        )

        with pytest.raises(APIStatusError, match="closed unexpectedly"):
            await _run_stream_with_mock(
                instance, mock_ws, collect_timeout=5.0, conn_options=NO_RETRY_OPTIONS
            )

    @pytest.mark.asyncio
    async def test_expected_close_after_task_finished(self) -> None:
        """Test 15: close after task-finished is normal, no error."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        # Should not raise
        await _run_stream_with_mock(instance, mock_ws)
        # No assertion needed — the test passes if no exception is raised

    @pytest.mark.asyncio
    async def test_connection_error_raises(self) -> None:
        """Test 16: connection failure raises APIConnectionError."""
        from livekit.agents import APIConnectionError

        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=aiohttp.ClientConnectorError(
                connection_key=MagicMock(), os_error=OSError("Connection refused")
            )
        )

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream()
            try:
                with pytest.raises(APIConnectionError):
                    stream.end_input()
                    async with asyncio.timeout(5.0):
                        async for _ in stream:
                            pass
            finally:
                await stream.aclose()


class TestTimestamps:
    """Tests 17-18: word timestamps and fallback."""

    @pytest.mark.asyncio
    async def test_word_timestamps(self) -> None:
        """Test 17: words field correctly maps to SpeechData.words."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg(
                    "你好世界",
                    sentence_end=True,
                    begin_time=100,
                    end_time=1500,
                    words=[
                        {"text": "你好", "begin_time": 100, "end_time": 700, "punctuation": ""},
                        {"text": "世界", "begin_time": 700, "end_time": 1500, "punctuation": ""},
                    ],
                ),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        final = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT][0]
        words = final.alternatives[0].words
        assert words is not None
        assert len(words) == 2
        assert str(words[0]) == "你好"
        assert str(words[1]) == "世界"

    @pytest.mark.asyncio
    async def test_interim_timestamp_fallback(self) -> None:
        """Test 18: interim end_time=None falls back to begin_time, no end < start."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg(
                    "你好",
                    sentence_end=False,
                    begin_time=500,
                    end_time=None,
                ),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        interim = [e for e in events if e.type == SpeechEventType.INTERIM_TRANSCRIPT][0]
        alt = interim.alternatives[0]
        # end_time should have fallen back to begin_time, not be 0
        assert alt.end_time >= alt.start_time
        assert alt.start_time == pytest.approx(0.5, abs=0.01)  # 500ms / 1000


class TestUsage:
    """Test 19: recognition usage."""

    @pytest.mark.asyncio
    async def test_recognition_usage(self) -> None:
        """Test 19: usage.duration maps to RECOGNITION_USAGE event."""
        instance = _create_stt()
        mock_ws = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg(
                    "测试",
                    sentence_end=True,
                    begin_time=0,
                    end_time=1000,
                    usage={"duration": 1.5},
                ),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        events = await _run_stream_with_mock(instance, mock_ws)

        usage_events = [e for e in events if e.type == SpeechEventType.RECOGNITION_USAGE]
        assert len(usage_events) == 1
        assert usage_events[0].recognition_usage is not None
        assert usage_events[0].recognition_usage.audio_duration == 1.5


class TestConfig:
    """Tests 20-22: config copy, update_options, aclose."""

    @pytest.mark.asyncio
    async def test_config_copy_per_stream(self) -> None:
        """Test 20: each stream gets an independent config copy."""
        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream1 = instance.stream()
            stream2 = instance.stream(language="en")

            assert stream1._opts.language == "zh"
            assert stream2._opts.language == "en"

            # Modifying one doesn't affect the other
            stream1._opts.max_sentence_silence = 999
            assert stream2._opts.max_sentence_silence == 700

            await stream1.aclose()
            await stream2.aclose()

    @pytest.mark.asyncio
    async def test_update_options_reconnect(self) -> None:
        """Test 21: update_options() triggers _reconnect_event.set()."""
        instance = _create_stt()
        stream = SpeechStream(
            stt=instance,
            opts=instance._opts,
            conn_options=DEFAULT_API_CONNECT_OPTIONS,
        )
        try:
            assert not stream._reconnect_event.is_set()
            stream.update_options(max_sentence_silence=500)
            assert stream._reconnect_event.is_set()
            assert stream._opts.max_sentence_silence == 500
        finally:
            await stream.aclose()

    @pytest.mark.asyncio
    async def test_aclose_cleans_streams(self) -> None:
        """Test 22: stt.aclose() closes all active streams."""
        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream1 = instance.stream()
            stream2 = instance.stream()

            assert len(instance._streams) == 2

            await instance.aclose()

            # Streams should be closed (their tasks cancelled)
            assert stream1._input_ch.closed
            assert stream2._input_ch.closed


class TestFlushSegmentBoundary:
    """Tests for flush() sending finish-task and reconnecting for a new segment."""

    @pytest.mark.asyncio
    async def test_flush_sends_finish_task_and_reconnects(self) -> None:
        """flush() sends finish-task on the current WS, then opens a new connection."""
        instance = _create_stt()

        # First connection: task-started, result, then task-finished (after flush)
        ws1 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第一句", sentence_end=True, begin_time=0, end_time=1000),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        # Second connection: task-started, result, then task-finished (after end_input)
        ws2 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第二句", sentence_end=True, begin_time=0, end_time=800),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        ws_iter = iter([ws1, ws2])

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: next(ws_iter))

        events: list[stt.SpeechEvent] = []

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream()
            try:
                # Push some audio
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                # Flush triggers finish-task + reconnect
                stream.flush()
                # Give time for the reconnect to happen
                await asyncio.sleep(0.3)
                # Push more audio on the new segment
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                stream.end_input()

                try:
                    async with asyncio.timeout(5.0):
                        async for ev in stream:
                            events.append(ev)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            finally:
                await stream.aclose()

        # Should have two FINAL transcripts from two segments
        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 2
        assert finals[0].alternatives[0].text == "第一句"
        assert finals[1].alternatives[0].text == "第二句"

        # ws1 should have received a finish-task
        ws1_str_msgs = [m for m in ws1.sent if isinstance(m, str)]
        ws1_finish = [json.loads(m) for m in ws1_str_msgs if "finish-task" in m]
        assert len(ws1_finish) == 1

        # ws2 should also have received a finish-task (from end_input)
        ws2_str_msgs = [m for m in ws2.sent if isinstance(m, str)]
        ws2_finish = [json.loads(m) for m in ws2_str_msgs if "finish-task" in m]
        assert len(ws2_finish) == 1


class TestReconnectStateReset:
    """Tests for state reset on reconnect (speaking, request_id)."""

    @pytest.mark.asyncio
    async def test_reconnect_resets_speaking_with_end_of_speech(self) -> None:
        """If speaking=True when reconnect happens, END_OF_SPEECH is emitted."""
        instance = _create_stt()

        # First connection: returns an interim (speaking=True) then task-finished
        ws1 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("你好", sentence_end=False, begin_time=0, end_time=500),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        # Second connection: clean
        ws2 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        ws_iter = iter([ws1, ws2])

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: next(ws_iter))

        events: list[stt.SpeechEvent] = []

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream()
            try:
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                stream.flush()
                await asyncio.sleep(0.3)
                stream.end_input()

                try:
                    async with asyncio.timeout(5.0):
                        async for ev in stream:
                            events.append(ev)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            finally:
                await stream.aclose()

        event_types = [e.type for e in events]
        # The first segment had speaking=True (from the interim), so reconnect should
        # emit a synthetic END_OF_SPEECH before the second segment starts.
        end_of_speech_count = event_types.count(SpeechEventType.END_OF_SPEECH)
        assert end_of_speech_count >= 1, (
            "Expected at least one END_OF_SPEECH from reconnect state reset"
        )

    @pytest.mark.asyncio
    async def test_reconnect_gets_new_request_id(self) -> None:
        """Each connection segment gets a fresh request_id."""
        instance = _create_stt()

        ws1 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第一句", sentence_end=True, begin_time=0, end_time=1000),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )
        ws2 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第二句", sentence_end=True, begin_time=0, end_time=800),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )

        ws_iter = iter([ws1, ws2])
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: next(ws_iter))

        events: list[stt.SpeechEvent] = []

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream()
            try:
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                stream.flush()
                await asyncio.sleep(0.3)
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                stream.end_input()

                try:
                    async with asyncio.timeout(5.0):
                        async for ev in stream:
                            events.append(ev)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            finally:
                await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert len(finals) == 2
        # Request IDs should differ between segments
        assert finals[0].request_id != finals[1].request_id


class TestWSHandshakeError:
    """Tests for _connect_ws exception wrapping."""

    @pytest.mark.asyncio
    async def test_handshake_error_becomes_api_status_error(self) -> None:
        """WSServerHandshakeError (e.g. 401) is wrapped as APIStatusError."""
        from livekit.agents import APIStatusError

        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=aiohttp.WSServerHandshakeError(
                request_info=MagicMock(),
                history=(),
                message="Unauthorized",
                status=401,
                headers=MagicMock(),
            )
        )

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream(conn_options=NO_RETRY_OPTIONS)
            try:
                with pytest.raises(APIStatusError, match="handshake failed"):
                    stream.end_input()
                    async with asyncio.timeout(5.0):
                        async for _ in stream:
                            pass
            finally:
                await stream.aclose()

    @pytest.mark.asyncio
    async def test_generic_client_error_becomes_api_connection_error(self) -> None:
        """Generic aiohttp.ClientError is wrapped as APIConnectionError."""
        from livekit.agents import APIConnectionError

        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=aiohttp.ClientError("DNS lookup failed"))

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream(conn_options=NO_RETRY_OPTIONS)
            try:
                with pytest.raises(APIConnectionError):
                    stream.end_input()
                    async with asyncio.timeout(5.0):
                        async for _ in stream:
                            pass
            finally:
                await stream.aclose()

    @pytest.mark.asyncio
    async def test_timeout_becomes_api_timeout_error(self) -> None:
        """asyncio.TimeoutError during connect is wrapped as APITimeoutError."""
        from livekit.agents import APITimeoutError

        instance = _create_stt()

        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream(conn_options=NO_RETRY_OPTIONS)
            try:
                with pytest.raises(APITimeoutError):
                    stream.end_input()
                    async with asyncio.timeout(5.0):
                        async for _ in stream:
                            pass
            finally:
                await stream.aclose()


class TestKeepalive:
    """Tests for the keepalive silence-frame mechanism."""

    @pytest.mark.asyncio
    async def test_keepalive_sends_silence(self) -> None:
        """Keepalive task sends 3200-byte silence frames at regular intervals."""
        instance = _create_stt()

        # Use a mock WS that stays open (blocks on receive) until task-finished
        mock_ws = MockWSConnection([_make_task_started_msg()])

        # Patch the keepalive interval to be very short for testing
        with patch.object(SpeechStream, "_KEEPALIVE_INTERVAL", 0.05):
            mock_session = MagicMock(spec=aiohttp.ClientSession)
            mock_session.ws_connect = AsyncMock(return_value=mock_ws)

            with patch.object(instance, "_ensure_session", return_value=mock_session):
                stream = instance.stream()
                try:
                    # Don't push any audio, just let keepalive run
                    await asyncio.sleep(0.2)  # ~4 keepalive intervals

                    # Now end the stream
                    stream.end_input()

                    # Inject task-finished to let it exit cleanly
                    mock_ws._messages.put_nowait(_make_task_finished_msg())
                    mock_ws._messages.put_nowait(_make_closed_msg())

                    try:
                        async with asyncio.timeout(3.0):
                            async for _ in stream:
                                pass
                    except (TimeoutError, asyncio.TimeoutError):
                        pass
                finally:
                    await stream.aclose()

        # Check that silence frames (3200 bytes of zeros) were sent
        silence_msgs = [m for m in mock_ws.sent if isinstance(m, bytes) and m == b"\x00" * 3200]
        assert len(silence_msgs) >= 2, (
            f"Expected at least 2 keepalive silence frames, got {len(silence_msgs)}"
        )


class TestErrorClassification:
    """task-failed error codes decide whether the SDK retries."""

    @pytest.mark.parametrize(
        ("code", "status", "retryable"),
        [
            ("InvalidApiKey", 401, False),
            ("AccessDenied.Unpurchased", 403, False),
            ("Model.AccessDenied", 403, False),
            ("Arrearage", 403, False),
            ("InvalidParameter", 400, False),
            ("DataInspectionFailed", 400, False),
            ("Throttling.RateQuota", 429, True),
            ("InternalError.Timeout", 500, True),
            ("SomethingNew", -1, True),
        ],
    )
    def test_task_failed_error_classification(
        self, code: str, status: int, retryable: bool
    ) -> None:
        from livekit.plugins.dashscope.common import task_failed_error

        err = task_failed_error(
            {
                "header": {
                    "event": "task-failed",
                    "task_id": "t-1",
                    "error_code": code,
                    "error_message": "boom",
                }
            },
            service_name="DashScope ASR",
        )
        assert err.status_code == status
        assert err.retryable is retryable
        assert err.request_id == "t-1"
        assert code in err.message and "boom" in err.message

    @pytest.mark.asyncio
    async def test_non_retryable_task_failed_is_not_retried(self) -> None:
        """An invalid API key fails on the first attempt even with retries allowed."""
        from livekit.agents import APIStatusError

        instance = _create_stt()
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(
            side_effect=lambda *a, **kw: MockWSConnection(
                [_make_task_failed_msg(code="InvalidApiKey", message="bad key"), _make_closed_msg()]
            )
        )
        retry_options = dataclasses.replace(
            DEFAULT_API_CONNECT_OPTIONS, max_retry=3, retry_interval=0.01
        )

        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream(conn_options=retry_options)
            try:
                with pytest.raises(APIStatusError, match="bad key") as excinfo:
                    async with asyncio.timeout(5.0):
                        async for _ in stream:
                            pass
            finally:
                await stream.aclose()

        assert excinfo.value.retryable is False
        assert mock_session.ws_connect.call_count == 1


class TestTimelineAcrossReconnects:
    @pytest.mark.asyncio
    async def test_flush_reconnect_keeps_timestamps_linear(self) -> None:
        """DashScope restarts its clock for every task; the second segment's timestamps must
        include the time that elapsed in the first one."""
        instance = _create_stt()
        ws1 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第一句", sentence_end=True, begin_time=0, end_time=100),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )
        ws2 = MockWSConnection(
            [
                _make_task_started_msg(),
                _make_result_msg("第二句", sentence_end=True, begin_time=0, end_time=100),
                _make_task_finished_msg(),
                _make_closed_msg(),
            ]
        )
        ws_iter = iter([ws1, ws2])
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(side_effect=lambda *a, **kw: next(ws_iter))

        events: list[stt.SpeechEvent] = []
        with patch.object(instance, "_ensure_session", return_value=mock_session):
            stream = instance.stream()
            try:
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                await asyncio.sleep(0.3)  # time spent in the first segment
                stream.flush()
                await asyncio.sleep(0.1)
                stream.push_frame(_make_pcm_frame(duration_ms=100))
                stream.end_input()
                try:
                    async with asyncio.timeout(5.0):
                        async for ev in stream:
                            events.append(ev)
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            finally:
                await stream.aclose()

        finals = [e for e in events if e.type == SpeechEventType.FINAL_TRANSCRIPT]
        assert [f.alternatives[0].text for f in finals] == ["第一句", "第二句"]
        first, second = (f.alternatives[0] for f in finals)
        assert first.start_time == pytest.approx(0.0, abs=0.05)
        assert second.start_time >= 0.3
        assert second.start_time > first.end_time


class TestKeepaliveWhileStreaming:
    @pytest.mark.asyncio
    async def test_no_silence_injected_while_audio_flows(self) -> None:
        instance = _create_stt()
        mock_ws = MockWSConnection([_make_task_started_msg()])
        mock_session = MagicMock(spec=aiohttp.ClientSession)
        mock_session.ws_connect = AsyncMock(return_value=mock_ws)
        loud = b"\x10\x00" * 1600  # 100ms of non-silent 16 kHz PCM

        with (
            patch.object(SpeechStream, "_KEEPALIVE_INTERVAL", 0.05),
            patch.object(instance, "_ensure_session", return_value=mock_session),
        ):
            stream = instance.stream()
            try:
                for _ in range(15):  # ~0.3s of audio, 6 keepalive intervals
                    stream.push_frame(
                        rtc.AudioFrame(
                            data=loud, sample_rate=16000, num_channels=1, samples_per_channel=1600
                        )
                    )
                    await asyncio.sleep(0.02)
                stream.end_input()
                mock_ws._messages.put_nowait(_make_task_finished_msg())
                mock_ws._messages.put_nowait(_make_closed_msg())
                try:
                    # the mock never answers finish-task; idle keepalive must not resume
                    async with asyncio.timeout(0.5):
                        async for _ in stream:
                            pass
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            finally:
                await stream.aclose()

        finish_at = next(i for i, m in enumerate(mock_ws.sent) if "finish-task" in str(m))
        audio = [m for m in mock_ws.sent if isinstance(m, bytes)]
        assert len(audio) == 15
        assert b"\x00" * 3200 not in audio
        assert not any(isinstance(m, bytes) for m in mock_ws.sent[finish_at:])
