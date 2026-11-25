from __future__ import annotations

import pytest

from livekit.agents.tokenize.tokenizer import SentenceStream
from livekit.plugins.dashscope import SentenceTokenizer

pytestmark = pytest.mark.unit

REPLY = (
    "巴赫的平均律分两册。每册二十四组前奏曲与赋格，共四十八首。它覆盖了全部大小调，想听我接着讲吗？"
)


def _ready(stream: SentenceStream) -> list[str]:
    """Chunks already emitted, without ending the input."""
    out: list[str] = []
    while True:
        try:
            out.append(stream._event_ch.recv_nowait().token)  # type: ignore[attr-defined]
        except Exception:
            return out


async def _stream_all(text: str, *, step: int = 1) -> list[str]:
    stream = SentenceTokenizer().stream()
    for i in range(0, len(text), step):
        stream.push_text(text[i : i + step])
    stream.end_input()
    return [ev.token async for ev in stream]


async def test_short_first_sentence_is_sent_without_look_ahead() -> None:
    # the SDK's basic tokenizer held this until ~30 characters had streamed
    stream = SentenceTokenizer().stream()
    stream.push_text("巴赫的平均律分两册。")
    assert _ready(stream) == ["巴赫的平均律分两册。"]
    await stream.aclose()


async def test_first_chunk_ends_at_a_clause_mark_once_long_enough() -> None:
    stream = SentenceTokenizer().stream()
    stream.push_text("好的，")  # too short to be a chunk of its own
    assert _ready(stream) == []
    stream.push_text("巴赫的平均律，分两册。")
    assert _ready(stream) == ["好的，巴赫的平均律，", "分两册。"]
    await stream.aclose()


async def test_later_chunks_end_at_sentence_marks_only() -> None:
    assert await _stream_all(REPLY) == [
        "巴赫的平均律分两册。",
        "每册二十四组前奏曲与赋格，共四十八首。",
        "它覆盖了全部大小调，想听我接着讲吗？",
    ]


async def test_clause_split_applies_to_the_first_chunk() -> None:
    assert await _stream_all("这张专辑录于一九五九年，只用了两次录音。") == [
        "这张专辑录于一九五九年，",
        "只用了两次录音。",
    ]


@pytest.mark.parametrize("step", [1, 3, 7, 1000])
async def test_streaming_matches_whole_text(step: int) -> None:
    text = "Kind of Blue 录于 1959 年，全长约 46 分钟。Miles Davis 主导。 这是例子，对吧？好的"
    assert await _stream_all(text, step=step) == SentenceTokenizer().tokenize(text)


async def test_a_mark_arriving_after_the_cut_starts_the_next_chunk() -> None:
    # no look-ahead after CJK marks, by design: a late closing quote / second "…" moves on
    assert await _stream_all("他说“好。”然后呢……嗯", step=1) == ["他说“好。", "”然后呢…", "…嗯"]


def test_ascii_marks_inside_numbers_do_not_split() -> None:
    assert SentenceTokenizer().tokenize("大约 3.5 小时，录了 2.1.0 版。1,000 张。") == [
        "大约 3.5 小时，",
        "录了 2.1.0 版。",
        "1,000 张。",
    ]


def test_ascii_sentence_mark_splits_before_space_or_cjk() -> None:
    assert SentenceTokenizer().tokenize("I love it. 它很好!真的") == [
        "I love it.",
        "它很好!",
        "真的",
    ]


def test_trailing_marks_and_quotes_stay_with_their_sentence() -> None:
    assert SentenceTokenizer().tokenize("他说：“好。”然后呢……嗯？！走吧") == [
        "他说：“好。”",
        "然后呢……",
        "嗯？！",
        "走吧",
    ]


async def test_flush_emits_the_rest_and_starts_a_new_segment() -> None:
    stream = SentenceTokenizer().stream()
    stream.push_text("第一段没有标点")
    stream.flush()
    stream.push_text("好的，第二段的开头，")
    stream.end_input()
    events = [ev async for ev in stream]
    assert [ev.token for ev in events] == ["第一段没有标点", "好的，第二段的开头，"]
    assert events[0].segment_id != events[1].segment_id


async def test_punctuation_only_rest_is_dropped() -> None:
    stream = SentenceTokenizer().stream()
    stream.push_text("好。")
    stream.push_text("”")
    stream.end_input()
    assert [ev.token async for ev in stream] == ["好。"]


def test_tts_uses_it_by_default() -> None:
    import os

    os.environ.setdefault("DASHSCOPE_API_KEY", "test-key-for-unit-tests")
    from livekit.plugins.dashscope import TTS

    assert isinstance(TTS()._sentence_tokenizer, SentenceTokenizer)
