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

"""Integration tests for livekit-plugins-dashscope STT plugin.

These tests require a real DashScope API key and make actual API calls.
Set DASHSCOPE_API_KEY to run them.

Usage:
    DASHSCOPE_API_KEY=sk-xxx uv run pytest packages/livekit-plugins-dashscope/tests/test_plugin_dashscope_integration.py -m plugin -v
"""

from __future__ import annotations

import asyncio
import os

import pytest

from livekit import rtc
from livekit.agents import stt, utils

pytestmark = [
    pytest.mark.plugin("dashscope"),
    pytest.mark.skipif(
        not os.environ.get("DASHSCOPE_API_KEY"),
        reason="Set DASHSCOPE_API_KEY to run DashScope integration tests",
    ),
]


@pytest.fixture
async def job_process():
    utils.http_context._new_session_ctx()
    yield
    await utils.http_context._close_http_ctx()


def _make_pcm_frame(
    duration_ms: int = 100,
    sample_rate: int = 16000,
    num_channels: int = 1,
) -> rtc.AudioFrame:
    """Create a silent PCM audio frame."""
    num_samples = int(sample_rate * duration_ms / 1000)
    data = bytes(num_samples * num_channels * 2)
    return rtc.AudioFrame(
        data=data,
        sample_rate=sample_rate,
        num_channels=num_channels,
        samples_per_channel=num_samples,
    )


@pytest.mark.asyncio
async def test_stream_silence_no_crash(job_process: None) -> None:
    """Push 5 seconds of silence — should not crash, may or may not produce FINAL."""
    from livekit.plugins.dashscope import STT

    instance = STT()
    stream = instance.stream()

    try:
        # Push 5 seconds of silence in 100ms chunks
        for _ in range(50):
            stream.push_frame(_make_pcm_frame(duration_ms=100))

        stream.end_input()

        events: list[stt.SpeechEvent] = []
        try:
            async with asyncio.timeout(10.0):
                async for ev in stream:
                    events.append(ev)
        except (TimeoutError, asyncio.TimeoutError):
            pass

        # Should not have any FINAL_TRANSCRIPT for silence
        # (but we don't assert it won't — some ASR systems may produce empty results)

    finally:
        await stream.aclose()
        await instance.aclose()


@pytest.mark.asyncio
async def test_tts_to_stt_roundtrip(job_process: None) -> None:
    """CosyVoice speaks a sentence, Fun-ASR transcribes it back, across a flush() reconnect.

    Checks recognition end to end on the real API, and that the second segment's timestamps
    continue the first one's instead of restarting at 0.
    """
    from livekit.plugins.dashscope import STT, TTS

    text = "今天我们聊一聊调式爵士。"
    tts_instance = TTS()
    stt_instance = STT()
    resampler = rtc.AudioResampler(input_rate=tts_instance.sample_rate, output_rate=16000)

    speech: list[rtc.AudioFrame] = []
    async with tts_instance.synthesize(text) as synth:
        async for audio in synth:
            speech.extend(resampler.push(audio.frame))
    speech.extend(resampler.flush())
    assert speech, "TTS produced no audio"

    silence = [_make_pcm_frame(duration_ms=100) for _ in range(10)]
    stream = stt_instance.stream()
    finals: list[stt.SpeechEvent] = []
    try:

        async def _collect() -> None:
            async for ev in stream:
                if ev.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    finals.append(ev)

        collector = asyncio.create_task(_collect())
        for segment in range(2):
            for frame in [*speech, *silence]:
                stream.push_frame(frame)
                await asyncio.sleep(frame.duration)  # real time, as from a microphone
            if segment == 0:
                stream.flush()
        stream.end_input()
        async with asyncio.timeout(20.0):
            await collector
    finally:
        await stream.aclose()
        await stt_instance.aclose()
        await tts_instance.aclose()

    assert len(finals) == 2, [f.alternatives[0].text for f in finals]
    for f in finals:
        assert "爵士" in f.alternatives[0].text
    first, second = (f.alternatives[0] for f in finals)
    assert second.start_time > first.end_time
