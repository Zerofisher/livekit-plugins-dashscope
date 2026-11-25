from __future__ import annotations

import asyncio
import dataclasses
import json
import os
import uuid
import weakref
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import aiohttp

from livekit.agents import (
    DEFAULT_API_CONNECT_OPTIONS,
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
    tts,
    utils,
)
from livekit.agents.types import NOT_GIVEN, NotGivenOr
from livekit.agents.utils import is_given
from livekit.agents.voice.io import TimedString

from .common import format_handshake_error, resolve_base_url, task_failed_error
from .log import logger
from .sentences import SentenceTokenizer

if TYPE_CHECKING:
    # aiohttp>=3.14: a `ClientWebSocketResponse[bool]` may also yield TEXT messages as bytes
    from aiohttp.http import WSMessageNoDecodeText

DEFAULT_MODEL = "cosyvoice-v3-flash"
DEFAULT_VOICE = "longanyang"
DEFAULT_FORMAT = "pcm"
DEFAULT_SAMPLE_RATE = 22050

# CosyVoice server drops the connection after 23 s of silence between
# continue-task messages.  We use a generous 60 s as the post-connect
# timeout so that slow LLM responses don't kill the stream, while still
# bounding the wait if the server stops responding entirely.
_POST_CONNECT_TIMEOUT: float = 60.0
# drop a pooled socket that sat unused this long rather than risk the server's idle close
_POOL_IDLE_TIMEOUT: float = 45.0


@dataclass
class DashScopeTTSOptions:
    api_key: str
    model: str
    voice: str
    base_url: str
    format: str
    sample_rate: int
    volume: int
    rate: float
    pitch: float
    enable_ssml: bool
    bit_rate: int | None
    word_timestamp_enabled: bool
    language: str | None
    instruction: str | None


class TTS(tts.TTS):
    """Alibaba Cloud DashScope CosyVoice text-to-speech."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        voice: str = DEFAULT_VOICE,
        base_url: str | None = None,
        format: str = DEFAULT_FORMAT,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        volume: int = 50,
        rate: float = 1.0,
        pitch: float = 1.0,
        enable_ssml: bool = False,
        bit_rate: int | None = None,
        word_timestamp_enabled: bool = False,
        language: str | None = None,
        instruction: str | None = None,
        http_session: aiohttp.ClientSession | None = None,
        sentence_tokenizer: NotGivenOr[tokenize.SentenceTokenizer] = NOT_GIVEN,
    ) -> None:
        api_key = api_key or os.environ.get("DASHSCOPE_API_KEY", "")
        if not api_key:
            raise ValueError(
                "DashScope API key is required. Pass api_key or set DASHSCOPE_API_KEY env var."
            )

        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=True,
                aligned_transcript=word_timestamp_enabled,
            ),
            sample_rate=sample_rate,
            num_channels=1,
        )

        self._opts = DashScopeTTSOptions(
            api_key=api_key,
            model=model,
            voice=voice,
            base_url=resolve_base_url(base_url),
            format=format.lower(),
            sample_rate=sample_rate,
            volume=volume,
            rate=rate,
            pitch=pitch,
            enable_ssml=enable_ssml,
            bit_rate=bit_rate,
            word_timestamp_enabled=word_timestamp_enabled,
            language=language,
            instruction=instruction,
        )
        self._session = http_session
        self._sentence_tokenizer = (
            sentence_tokenizer if is_given(sentence_tokenizer) else SentenceTokenizer()
        )
        self._streams = weakref.WeakSet[SynthesizeStream]()
        # One DashScope duplex WebSocket can run many tasks back to back (run-task ... task-finished),
        # so sockets are pooled: a turn skips the ~0.1s TLS + WebSocket handshake. A socket goes back
        # only after its task finished; anything else (error, interruption) discards it.
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse[bool]](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
            max_session_duration=_POOL_IDLE_TIMEOUT,
            mark_refreshed_on_get=True,
        )

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

    async def _connect_ws(self, timeout: float) -> aiohttp.ClientWebSocketResponse[bool]:
        return await asyncio.wait_for(
            self._ensure_session().ws_connect(
                self._opts.base_url,
                headers={"Authorization": f"bearer {self._opts.api_key}"},
            ),
            timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
        await ws.close()

    @asynccontextmanager
    async def _pooled_ws(
        self, timeout: float
    ) -> AsyncIterator[tuple[aiohttp.ClientWebSocketResponse[bool], float, bool]]:
        """(socket, acquire time, reused). Returned to the pool only on a clean exit."""
        while True:
            ws = await self._pool.get(timeout=timeout)
            if not ws.closed:
                break
            self._pool.remove(ws)  # the server closed it while it sat idle
        acquired, reused = self._pool.last_acquire_time, self._pool.last_connection_reused
        try:
            yield ws, acquired, reused
        except BaseException:
            self._pool.remove(ws)
            raise
        else:
            self._pool.put(ws)

    def prewarm(self) -> None:
        self._pool.prewarm()

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice: NotGivenOr[str] = NOT_GIVEN,
        rate: NotGivenOr[float] = NOT_GIVEN,
        pitch: NotGivenOr[float] = NOT_GIVEN,
        volume: NotGivenOr[int] = NOT_GIVEN,
        instruction: NotGivenOr[str | None] = NOT_GIVEN,
        language: NotGivenOr[str | None] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice):
            self._opts.voice = voice
        if is_given(rate):
            self._opts.rate = rate
        if is_given(pitch):
            self._opts.pitch = pitch
        if is_given(volume):
            self._opts.volume = volume
        if is_given(instruction):
            self._opts.instruction = instruction
        if is_given(language):
            self._opts.language = language

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> ChunkedStream:
        return ChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
    ) -> SynthesizeStream:
        stream = SynthesizeStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._pool.aclose()


class ChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: TTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = dataclasses.replace(tts._opts)
        self._task_id = str(uuid.uuid4())

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        if not self.input_text.strip():
            return

        try:
            async with self._tts._pooled_ws(self._conn_options.timeout) as (ws, acquired, reused):
                self._acquire_time = acquired
                self._connection_reused = reused

                await ws.send_str(json.dumps(_build_run_task(self._opts, self._task_id)))
                await _wait_for_task_started(ws, timeout=_POST_CONNECT_TIMEOUT)

                output_emitter.initialize(
                    request_id=self._task_id,
                    sample_rate=self._opts.sample_rate,
                    num_channels=1,
                    mime_type=_mime_type_from_format(self._opts.format),
                )

                await ws.send_str(json.dumps(_build_continue_task(self._task_id, self.input_text)))
                await ws.send_str(json.dumps(_build_finish_task(self._task_id)))
                await _drain_task_output(ws, output_emitter, timeout=_POST_CONNECT_TIMEOUT)
            output_emitter.flush()
        except aiohttp.WSServerHandshakeError as e:
            message, status_code, body = format_handshake_error(
                e,
                base_url=self._opts.base_url,
                service_name="DashScope TTS WebSocket",
            )
            logger.warning(message)
            raise APIStatusError(message=message, status_code=status_code, body=body) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"DashScope TTS connection failed: {self._opts.base_url}"
            ) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("DashScope TTS connection timed out") from e


class SynthesizeStream(tts.SynthesizeStream):
    def __init__(self, *, tts: TTS, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, conn_options=conn_options)
        self._tts: TTS = tts
        self._opts = dataclasses.replace(tts._opts)

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        task_id = str(uuid.uuid4())
        request_id = utils.shortuuid()

        output_emitter.initialize(
            request_id=request_id,
            sample_rate=self._opts.sample_rate,
            num_channels=1,
            mime_type=_mime_type_from_format(self._opts.format),
            stream=True,
        )

        sent_stream = self._tts._sentence_tokenizer.stream()

        # Event: set once the first continue-task has been sent.
        # _recv_task waits on this before starting its timeout-guarded
        # receive loop so it doesn't self-timeout while the sentence
        # tokenizer is still accumulating the first sentence.
        first_text_sent = asyncio.Event()

        async def _input_task() -> None:
            async for data in self._input_ch:
                if isinstance(data, self._FlushSentinel):
                    sent_stream.flush()
                    continue
                sent_stream.push_text(data)
            sent_stream.end_input()

        async def _sentence_stream_task(ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
            if self._opts.enable_ssml:
                # SSML mode: CosyVoice only allows a single continue-task
                # per session, so we must accumulate all tokenized sentences
                # into one string and send it at the end.
                parts: list[str] = []
                async for ev in sent_stream:
                    self._mark_started()
                    parts.append(ev.token)
                if parts:
                    full_text = " ".join(parts)
                    await ws.send_str(json.dumps(_build_continue_task(task_id, full_text)))
                    first_text_sent.set()
            else:
                async for ev in sent_stream:
                    self._mark_started()
                    await ws.send_str(json.dumps(_build_continue_task(task_id, ev.token)))
                    first_text_sent.set()
            await ws.send_str(json.dumps(_build_finish_task(task_id)))
            # If no text was ever sent (empty input), unblock _recv_task
            # so it can see the task-finished from the server.
            first_text_sent.set()

        async def _recv_task(ws: aiohttp.ClientWebSocketResponse[bool]) -> None:
            # Wait until the first continue-task is sent before entering the
            # timeout-guarded receive loop.  This avoids killing the stream
            # when the LLM / sentence tokenizer takes a while to produce the
            # first sentence.
            await first_text_sent.wait()

            output_emitter.start_segment(segment_id=utils.shortuuid())
            while True:
                msg = await ws.receive(timeout=_POST_CONNECT_TIMEOUT)
                if msg.type == aiohttp.WSMsgType.BINARY:
                    output_emitter.push(bytes(msg.data))
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    payload = _load_text_message(msg)
                    header = payload.get("header", {})
                    event = header.get("event")

                    if event == "result-generated":
                        timed_words = _parse_timed_words(payload)
                        if timed_words:
                            output_emitter.push_timed_transcript(timed_words)
                        continue

                    if event == "task-finished":
                        output_emitter.end_input()
                        return

                    if event == "task-failed":
                        raise task_failed_error(payload, service_name="DashScope TTS")

                    if event == "error":
                        raise APIStatusError(
                            message=payload.get("payload", {}).get(
                                "message", "DashScope TTS error"
                            ),
                            status_code=500,
                            body=json.dumps(payload, ensure_ascii=False),
                        )
                    continue

                _raise_unexpected_ws_message(ws, msg)

        try:
            async with self._tts._pooled_ws(self._conn_options.timeout) as (ws, acquired, reused):
                self._acquire_time = acquired
                self._connection_reused = reused

                await ws.send_str(json.dumps(_build_run_task(self._opts, task_id)))
                await _wait_for_task_started(ws, timeout=_POST_CONNECT_TIMEOUT)

                tasks = [
                    asyncio.create_task(_input_task()),
                    asyncio.create_task(_sentence_stream_task(ws)),
                    asyncio.create_task(_recv_task(ws)),
                ]
                try:
                    await asyncio.gather(*tasks)
                finally:
                    await sent_stream.aclose()
                    await utils.aio.gracefully_cancel(*tasks)
        except aiohttp.WSServerHandshakeError as e:
            message, status_code, body = format_handshake_error(
                e,
                base_url=self._opts.base_url,
                service_name="DashScope TTS WebSocket",
            )
            logger.warning(message)
            raise APIStatusError(message=message, status_code=status_code, body=body) from e
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"DashScope TTS connection failed: {self._opts.base_url}"
            ) from e
        except asyncio.TimeoutError as e:
            raise APITimeoutError("DashScope TTS connection timed out") from e


# --- Module-level helpers ---


def _build_run_task(opts: DashScopeTTSOptions, task_id: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "text_type": "PlainText",
        "voice": opts.voice,
        "format": opts.format,
        "sample_rate": opts.sample_rate,
        "volume": opts.volume,
        "rate": opts.rate,
        "pitch": opts.pitch,
    }
    if opts.enable_ssml:
        parameters["enable_ssml"] = True
    if opts.bit_rate is not None:
        parameters["bit_rate"] = opts.bit_rate
    if opts.word_timestamp_enabled:
        parameters["word_timestamp_enabled"] = True
    if opts.language:
        parameters["language_hints"] = [opts.language]
    if opts.instruction:
        parameters["instruction"] = opts.instruction

    return {
        "header": {
            "action": "run-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {
            "task_group": "audio",
            "task": "tts",
            "function": "SpeechSynthesizer",
            "model": opts.model,
            "parameters": parameters,
            "input": {},
        },
    }


def _build_continue_task(task_id: str, text: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "continue-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {"text": text}},
    }


def _build_finish_task(task_id: str) -> dict[str, Any]:
    return {
        "header": {
            "action": "finish-task",
            "task_id": task_id,
            "streaming": "duplex",
        },
        "payload": {"input": {}},
    }


async def _wait_for_task_started(
    ws: aiohttp.ClientWebSocketResponse[bool],
    *,
    timeout: float = _POST_CONNECT_TIMEOUT,
) -> None:
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type != aiohttp.WSMsgType.TEXT:
            _raise_unexpected_ws_message(ws, msg)

        payload = _load_text_message(msg)
        header = payload.get("header", {})
        event = header.get("event")
        if event == "task-started":
            return
        if event == "task-failed":
            raise task_failed_error(payload, service_name="DashScope TTS")
        if event == "error":
            raise APIStatusError(
                message=payload.get("payload", {}).get("message", "DashScope TTS error"),
                status_code=500,
                body=json.dumps(payload, ensure_ascii=False),
            )


async def _drain_task_output(
    ws: aiohttp.ClientWebSocketResponse[bool],
    output_emitter: tts.AudioEmitter,
    *,
    timeout: float = _POST_CONNECT_TIMEOUT,
) -> None:
    while True:
        msg = await ws.receive(timeout=timeout)
        if msg.type == aiohttp.WSMsgType.BINARY:
            output_emitter.push(bytes(msg.data))
            continue

        if msg.type == aiohttp.WSMsgType.TEXT:
            payload = _load_text_message(msg)
            header = payload.get("header", {})
            event = header.get("event")

            if event == "result-generated":
                timed_words = _parse_timed_words(payload)
                if timed_words:
                    output_emitter.push_timed_transcript(timed_words)
                continue

            if event == "task-finished":
                return

            if event == "task-failed":
                raise task_failed_error(payload, service_name="DashScope TTS")

            if event == "error":
                raise APIStatusError(
                    message=payload.get("payload", {}).get("message", "DashScope TTS error"),
                    status_code=500,
                    body=json.dumps(payload, ensure_ascii=False),
                )
            continue

        _raise_unexpected_ws_message(ws, msg)


def _raise_unexpected_ws_message(
    ws: aiohttp.ClientWebSocketResponse[bool],
    msg: WSMessageNoDecodeText,
) -> None:
    if msg.type in (
        aiohttp.WSMsgType.CLOSE,
        aiohttp.WSMsgType.CLOSED,
        aiohttp.WSMsgType.CLOSING,
    ):
        raise APIStatusError(
            message="DashScope TTS connection closed unexpectedly",
            status_code=ws.close_code or -1,
            body=str(msg.data),
        )

    if msg.type == aiohttp.WSMsgType.ERROR:
        raise APIConnectionError("DashScope TTS websocket error") from ws.exception()

    raise APIStatusError(
        message=f"Unexpected DashScope TTS websocket message type: {msg.type}",
        status_code=-1,
        body=str(msg.data),
    )


def _load_text_message(msg: WSMessageNoDecodeText) -> dict[str, Any]:
    result: dict[str, Any] = json.loads(msg.data)
    return result


def _mime_type_from_format(audio_format: str) -> str:
    if audio_format == "pcm":
        return "audio/pcm"
    return f"audio/{audio_format}"


def _parse_timed_words(message: dict[str, Any]) -> list[TimedString] | None:
    output = message.get("payload", {}).get("output", {})
    sentence = output.get("sentence", {})
    words = sentence.get("words")
    if not isinstance(words, list) or not words:
        return None

    timed_words: list[TimedString] = []
    for word in words:
        if not isinstance(word, dict):
            continue
        text = str(word.get("text", ""))
        if not text:
            continue
        begin_ms = word.get("begin_time")
        end_ms = word.get("end_time")
        if begin_ms is None:
            continue
        if end_ms is None:
            end_ms = begin_ms
        timed_words.append(
            TimedString(
                text=text,
                start_time=begin_ms / 1000,
                end_time=end_ms / 1000,
            )
        )

    return timed_words or None
