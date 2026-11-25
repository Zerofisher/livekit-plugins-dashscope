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
import contextlib
import dataclasses
import json
import os
import time
import uuid
import weakref
from dataclasses import dataclass
from typing import Any

import aiohttp

from livekit import rtc
from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    LanguageCode,
    stt,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import AudioBuffer, is_given
from livekit.agents.voice.io import TimedString

from .common import format_handshake_error, resolve_base_url, task_failed_error
from .log import logger


@dataclass
class DashScopeSTTOptions:
    api_key: str
    model: str
    base_url: str
    language: str
    sample_rate: int
    format: str
    max_sentence_silence: int
    semantic_punctuation: bool
    heartbeat: bool
    disfluency_removal: bool
    vocabulary_id: str | None
    speech_noise_threshold: float | None


class STT(stt.STT):
    """Alibaba Cloud DashScope Fun-ASR real-time speech-to-text.

    Uses the DashScope WebSocket API with server-side VAD for sentence
    boundary detection, producing INTERIM and FINAL transcripts that
    map naturally to the LiveKit Agents framework event model.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "fun-asr-realtime",
        base_url: str | None = None,
        language: str = "zh",
        max_sentence_silence: int = 700,
        heartbeat: bool = True,
        semantic_punctuation: bool = False,
        disfluency_removal: bool = False,
        vocabulary_id: str | None = None,
        speech_noise_threshold: float | None = None,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=stt.STTCapabilities(
                streaming=True,
                interim_results=True,
                # sentence and word begin/end times, relative to the stream's audio start
                aligned_transcript="word",
                offline_recognize=False,
            )
        )

        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Pass api_key or set DASHSCOPE_API_KEY env var."
            )

        self._opts = DashScopeSTTOptions(
            api_key=api_key,
            model=model,
            base_url=resolve_base_url(base_url),
            language=language,
            sample_rate=16000,
            format="pcm",
            max_sentence_silence=max_sentence_silence,
            semantic_punctuation=semantic_punctuation,
            heartbeat=heartbeat,
            disfluency_removal=disfluency_removal,
            vocabulary_id=vocabulary_id,
            speech_noise_threshold=speech_noise_threshold,
        )
        self._session = http_session
        self._streams: weakref.WeakSet[SpeechStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return self._opts.model

    @property
    def provider(self) -> str:
        return "dashscope"

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()
        return self._session

    async def _recognize_impl(
        self,
        buffer: AudioBuffer,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> stt.SpeechEvent:
        raise NotImplementedError("DashScope STT only supports streaming mode")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SpeechStream:
        opts = dataclasses.replace(self._opts)
        if is_given(language):
            opts.language = language
        stream = SpeechStream(stt=self, opts=opts, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        max_sentence_silence: NotGivenOr[int] = NOT_GIVEN,
        heartbeat: NotGivenOr[bool] = NOT_GIVEN,
        vocabulary_id: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(max_sentence_silence):
            self._opts.max_sentence_silence = max_sentence_silence
        if is_given(heartbeat):
            self._opts.heartbeat = heartbeat
        if is_given(vocabulary_id):
            self._opts.vocabulary_id = vocabulary_id

        for s in self._streams:
            s.update_options(
                language=language,
                max_sentence_silence=max_sentence_silence,
                heartbeat=heartbeat,
                vocabulary_id=vocabulary_id,
            )

    async def aclose(self) -> None:
        """Close all active streams. Does not close the HTTP session (managed by http_context)."""
        streams = list(self._streams)
        if streams:
            await asyncio.gather(*(s.aclose() for s in streams), return_exceptions=True)


class SpeechStream(stt.RecognizeStream):
    """DashScope Fun-ASR streaming speech recognition.

    Implements the DashScope WebSocket protocol:
    run-task -> task-started -> binary PCM + result-generated -> finish-task -> task-finished
    """

    def __init__(
        self,
        *,
        stt: STT,
        opts: DashScopeSTTOptions,
        conn_options: APIConnectOptions,
    ) -> None:
        super().__init__(stt=stt, conn_options=conn_options, sample_rate=16000)
        self._opts = opts
        self._stt_instance = stt
        self._reconnect_event = asyncio.Event()
        self._task_started = asyncio.Event()
        self._task_finished = asyncio.Event()
        self._finish_sent = False
        self._speaking = False
        self._request_id = utils.shortuuid()
        self._task_id = str(uuid.uuid4())
        self._last_audio_sent_at = 0.0  # time.monotonic() of the last frame sent

    def update_options(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        max_sentence_silence: NotGivenOr[int] = NOT_GIVEN,
        heartbeat: NotGivenOr[bool] = NOT_GIVEN,
        vocabulary_id: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        if is_given(language):
            self._opts.language = language
        if is_given(max_sentence_silence):
            self._opts.max_sentence_silence = max_sentence_silence
        if is_given(heartbeat):
            self._opts.heartbeat = heartbeat
        if is_given(vocabulary_id):
            self._opts.vocabulary_id = vocabulary_id
        self._reconnect_event.set()

    async def _run(self) -> None:
        """Outer loop: handles reconnection triggered by update_options() or flush()."""
        ws: aiohttp.ClientWebSocketResponse[bool] | None = None
        send_task: asyncio.Task[None] | None = None
        recv_task: asyncio.Task[None] | None = None
        keepalive_task: asyncio.Task[None] | None = None
        reconnect_task: asyncio.Task[object] | None = None
        wait_started_task: asyncio.Task[object] | None = None
        tasks_group: asyncio.Future[object] | None = None
        segment_started_at: float | None = None

        while True:
            try:
                # Every task restarts DashScope's clock at 0, so timestamps must stay linear
                # across flush() / update_options() reconnects.
                now = time.time()
                if segment_started_at is not None:
                    restart_clock = getattr(self._input_ch, "restart_provider_clock", None)
                    if restart_clock is not None:
                        # livekit-agents >= 1.8.11 (livekit/agents#7795): start_time_offset is
                        # anchored on the first frame each connection reads, and assigning it
                        # declares the session start instead of adding to the offset, so `+=`
                        # would be lost. Re-anchor on the next frame read, as the upstream
                        # plugins do when they reconnect inside _run.
                        restart_clock()
                    else:
                        # older SDKs: carry the elapsed time into the offset, as
                        # RecognizeStream._main_task did across retries
                        self.start_time_offset += now - segment_started_at
                segment_started_at = now

                # Reset state for each connection
                self._task_started.clear()
                self._task_finished.clear()
                self._finish_sent = False
                self._task_id = str(uuid.uuid4())
                # Reset speech state so new segments don't inherit stale state
                # from a mid-utterance disconnect or update_options() reconnect
                if self._speaking:
                    self._speaking = False
                    self._event_ch.send_nowait(
                        stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH)
                    )
                self._request_id = utils.shortuuid()

                ws = await self._connect_ws()

                # 1. Start recv_task first (it's the one that sets _task_started)
                recv_task = asyncio.create_task(self._recv_task(ws))

                # 2. Send run-task message
                await self._send_run_task(ws)

                # 3. Wait for task-started, racing with recv_task in case of early failure
                wait_started_task = asyncio.create_task(self._task_started.wait())
                done, _ = await asyncio.wait(
                    (recv_task, wait_started_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                # If recv_task finished first, it hit an error (task-failed, etc.)
                if recv_task in done:
                    recv_task.result()  # propagate the exception

                # Timeout protection for task-started
                if not wait_started_task.done():
                    await asyncio.wait_for(wait_started_task, timeout=self._conn_options.timeout)

                # 4. After task-started, begin sending audio + keepalive
                send_task = asyncio.create_task(self._send_audio_task(ws))
                keepalive_task = (
                    asyncio.create_task(self._keepalive_task(ws)) if self._opts.heartbeat else None
                )
                reconnect_task = asyncio.create_task(self._reconnect_event.wait())

                tasks_group = asyncio.gather(send_task, recv_task)  # type: ignore[assignment]
                done, _ = await asyncio.wait(  # type: ignore[type-var]
                    (tasks_group, reconnect_task),  # type: ignore[arg-type]
                    return_when=asyncio.FIRST_COMPLETED,
                )

                for t in done:
                    if t is not reconnect_task:
                        t.result()  # propagate APIStatusError

                if reconnect_task not in done:
                    # tasks_group completed — either flush() or end_input()
                    if self._input_ch.closed:
                        break  # end_input: done
                    # flush: loop to start a new segment
                    continue

                self._reconnect_event.clear()

            finally:
                tasks = [
                    t
                    for t in (
                        send_task,
                        recv_task,
                        keepalive_task,
                        reconnect_task,
                        wait_started_task,
                    )
                    if t is not None
                ]
                if tasks:
                    await utils.aio.gracefully_cancel(*tasks)
                if tasks_group is not None:
                    tasks_group.cancel()
                    with contextlib.suppress(asyncio.CancelledError, asyncio.InvalidStateError):
                        tasks_group.exception()
                    tasks_group = None
                if ws is not None:
                    await ws.close()

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse[bool]:
        session = self._stt_instance._ensure_session()
        t0 = time.perf_counter()
        try:
            ws = await asyncio.wait_for(
                session.ws_connect(
                    self._opts.base_url,
                    headers={"Authorization": f"bearer {self._opts.api_key}"},
                ),
                self._conn_options.timeout,
            )
            self._report_connection_acquired(time.perf_counter() - t0, False)
            return ws
        except aiohttp.WSServerHandshakeError as e:
            message, status_code, body = format_handshake_error(
                e,
                base_url=self._opts.base_url,
                service_name="DashScope WebSocket",
            )
            logger.warning(message)
            raise APIStatusError(
                message=message,
                status_code=status_code,
                body=body,
            ) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(f"DashScope connection failed: {self._opts.base_url}") from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("DashScope connection timed out") from e

    # Keepalive: 100ms of silence (3200 bytes of zero PCM at 16kHz mono 16-bit)
    _SILENCE_FRAME: bytes = b"\x00" * 3200
    _KEEPALIVE_INTERVAL: float = 15.0  # seconds between keepalive silence frames

    async def _send_run_task(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        await ws.send_str(json.dumps(self._build_run_task()))

    async def _keepalive_task(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        """Send a silence frame when no audio went out for a while, to prevent the 60s
        server-side idle timeout. Flowing audio keeps the connection alive by itself; silence
        injected into it would cut gaps into the speech and shift DashScope's timestamps."""
        try:
            while not self._finish_sent:  # no audio may follow finish-task
                await asyncio.sleep(self._KEEPALIVE_INTERVAL)
                if self._finish_sent:
                    return
                if time.monotonic() - self._last_audio_sent_at >= self._KEEPALIVE_INTERVAL:
                    await self._send_audio(ws, self._SILENCE_FRAME)
        except (asyncio.CancelledError, ConnectionError):
            return

    async def _send_audio(self, ws: aiohttp.ClientWebSocketResponse[bool], data: bytes) -> None:
        await ws.send_bytes(data)
        self._last_audio_sent_at = time.monotonic()

    async def _send_audio_task(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        samples_100ms = self._opts.sample_rate // 10  # 1600 samples = 3200 bytes
        audio_bstream = utils.audio.AudioByteStream(
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            samples_per_channel=samples_100ms,
        )

        async for data in self._input_ch:
            if isinstance(data, rtc.AudioFrame):
                for frame in audio_bstream.write(data.data.tobytes()):
                    await self._send_audio(ws, frame.data.tobytes())
            elif isinstance(data, self._FlushSentinel):
                # Flush residual audio from the chunking buffer
                for frame in audio_bstream.flush():
                    await self._send_audio(ws, frame.data.tobytes())
                # Send finish-task to end the current recognition segment,
                # then trigger reconnect to start a new segment.
                # This matches the base class contract: flush() = "end of current segment".
                self._finish_sent = True
                await ws.send_str(json.dumps(self._build_finish_task()))
                return  # exit send_task; recv_task will see task-finished and also exit

        # input_ch closed (end_input) -> send finish-task
        self._finish_sent = True
        await ws.send_str(json.dumps(self._build_finish_task()))
        # recv_task will set _task_finished upon receiving task-finished event

    async def _recv_task(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        while True:
            msg = await ws.receive()

            if msg.type in (
                aiohttp.WSMsgType.CLOSED,
                aiohttp.WSMsgType.CLOSE,
                aiohttp.WSMsgType.CLOSING,
            ):
                # Only consider it normal if we already received task-finished
                if self._task_finished.is_set():
                    return
                raise APIStatusError(
                    message="DashScope connection closed unexpectedly",
                    status_code=ws.close_code or -1,
                    body=str(msg.data),
                )

            if msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("DashScope WebSocket error")

            if msg.type != aiohttp.WSMsgType.TEXT:
                continue

            try:
                data = json.loads(msg.data)
            except (json.JSONDecodeError, TypeError):
                logger.warning("DashScope: received non-JSON text message")
                continue

            header = data.get("header", {})
            event = header.get("event", "")

            if event == "task-failed":
                raise task_failed_error(data, service_name="DashScope ASR")

            if event == "task-started":
                self._task_started.set()
                continue

            if event == "task-finished":
                self._task_finished.set()
                return  # normal exit

            if event == "result-generated":
                self._process_result(data)

    def _process_result(self, data: dict[str, Any]) -> None:
        payload = data.get("payload", {})
        output = payload.get("output", {})
        sentence = output.get("sentence", {})

        # Ignore heartbeat keep-alive messages
        if sentence.get("heartbeat"):
            return

        text = sentence.get("text", "").strip()
        if not text:
            return

        sentence_end = sentence.get("sentence_end", False)

        # State machine: ensure START -> (INTERIM)* -> FINAL -> END
        if not self._speaking:
            self._speaking = True
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.START_OF_SPEECH))

        # Timestamp handling with fallback: end_time=None -> use begin_time
        begin_ms = sentence.get("begin_time") or 0
        end_ms = sentence.get("end_time")
        if end_ms is None:
            end_ms = begin_ms

        # Word-level timestamps
        words: list[TimedString] | None = None
        if sentence.get("words"):
            words = []
            for w in sentence["words"]:
                word_begin = w.get("begin_time", 0)
                word_end = w.get("end_time")
                if word_end is None:
                    word_end = word_begin
                words.append(
                    TimedString(
                        text=w.get("text", "") + w.get("punctuation", ""),
                        start_time=(word_begin / 1000) + self.start_time_offset,
                        end_time=(word_end / 1000) + self.start_time_offset,
                    )
                )

        alt = stt.SpeechData(
            language=LanguageCode(self._opts.language),
            text=text,
            start_time=(begin_ms / 1000) + self.start_time_offset,
            end_time=(end_ms / 1000) + self.start_time_offset,
            words=words,
        )

        if not sentence_end:
            # Interim transcript
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[alt],
                )
            )
        else:
            # Final transcript
            self._event_ch.send_nowait(
                stt.SpeechEvent(
                    type=stt.SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._request_id,
                    alternatives=[alt],
                )
            )

            # Emit usage if available
            usage = payload.get("usage")
            if usage:
                self._event_ch.send_nowait(
                    stt.SpeechEvent(
                        type=stt.SpeechEventType.RECOGNITION_USAGE,
                        request_id=self._request_id,
                        recognition_usage=stt.RecognitionUsage(
                            audio_duration=float(usage.get("duration", 0)),
                        ),
                    )
                )

            self._speaking = False
            self._event_ch.send_nowait(stt.SpeechEvent(type=stt.SpeechEventType.END_OF_SPEECH))
            self._request_id = utils.shortuuid()

    def _build_run_task(self) -> dict[str, object]:
        parameters: dict[str, object] = {
            "format": self._opts.format,
            "sample_rate": self._opts.sample_rate,
            "max_sentence_silence": self._opts.max_sentence_silence,
            "semantic_punctuation_enabled": self._opts.semantic_punctuation,
            "heartbeat": self._opts.heartbeat,
            "disfluency_removal_enabled": self._opts.disfluency_removal,
            "language_hints": [self._opts.language],
        }
        if self._opts.vocabulary_id:
            parameters["vocabulary_id"] = self._opts.vocabulary_id
        if self._opts.speech_noise_threshold is not None:
            parameters["speech_noise_threshold"] = self._opts.speech_noise_threshold

        return {
            "header": {
                "action": "run-task",
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": {
                "task_group": "audio",
                "task": "asr",
                "function": "recognition",
                "model": self._opts.model,
                "parameters": parameters,
                "input": {},
            },
        }

    def _build_finish_task(self) -> dict[str, object]:
        return {
            "header": {
                "action": "finish-task",
                "task_id": self._task_id,
                "streaming": "duplex",
            },
            "payload": {"input": {}},
        }
