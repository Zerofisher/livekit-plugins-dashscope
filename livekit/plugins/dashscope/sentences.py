"""Sentence segmentation for streaming CosyVoice synthesis.

A CosyVoice duplex task starts synthesizing a chunk as soon as it ends in a punctuation mark
(a comma included) and holds an unterminated fragment until more text arrives. Measured
2026-10-12 on cosyvoice-v3-flash, time to first audio for the first chunk, median of 4:
"巴赫的平均律分两册。" 0.43s, "巴赫的平均律，" 0.46s, "巴赫的" 2.88s (= until the next push).

The SDK's basic tokenizer is built for English: it merges sentences until they reach 20
characters and waits for 10 characters of look-ahead, so a short Chinese first sentence is held
back until about 30 characters have streamed from the LLM. This tokenizer instead sends

- the first chunk of a segment at the first clause or sentence mark (a clause mark only once the
  chunk has `first_min_len` word characters, so "好的，" is not a chunk of its own),
- every later chunk at the next sentence mark,

with no look-ahead after CJK marks: a closing quote or a second "…" that arrives after the cut
starts the next chunk. ASCII marks also occur inside numbers
("3.5", "1,000"), so they only end a chunk when followed by whitespace or a non-ASCII character
(an abbreviation such as "e.g." before a space or a CJK character is still cut there).
"""

from __future__ import annotations

from livekit.agents import tokenize
from livekit.agents.tokenize.tokenizer import SentenceStream, TokenData
from livekit.agents.utils import shortuuid

_CJK_SENTENCE = frozenset("。！？；…\n")
_CJK_CLAUSE = frozenset("，、：")
_ASCII_SENTENCE = frozenset(".!?;")
_ASCII_CLAUSE = frozenset(",:")
# carried along with the mark they follow when already buffered ("好。”" / "……" / "?!")
_TRAILING = _CJK_SENTENCE | _ASCII_SENTENCE | frozenset("”’」』）)\"'")


def _find_cut(text: str, *, first: bool, first_min_len: int) -> int | None:
    """End index (exclusive) of the next chunk in `text`, or None while it is incomplete."""
    words = 0
    for i, ch in enumerate(text):
        if ch.isalnum():
            words += 1
            continue
        if not words:  # leading marks never make a chunk of their own
            continue
        sentence = ch in _CJK_SENTENCE or ch in _ASCII_SENTENCE
        clause = first and (ch in _CJK_CLAUSE or ch in _ASCII_CLAUSE)
        if not (sentence or (clause and words >= first_min_len)):
            continue
        if ch.isascii() and ch != "\n":
            if i + 1 == len(text):
                return None  # decided by the next character
            nxt = text[i + 1]
            if nxt.isascii() and not nxt.isspace():
                continue
        end = i + 1
        while end < len(text) and text[end] in _TRAILING:
            end += 1
        return end
    return None


class SentenceTokenizer(tokenize.SentenceTokenizer):
    def __init__(self, *, first_min_len: int = 4) -> None:
        self._first_min_len = first_min_len

    def tokenize(self, text: str, *, language: str | None = None) -> list[str]:
        chunks: list[str] = []
        first = True
        while (cut := _find_cut(text, first=first, first_min_len=self._first_min_len)) is not None:
            if chunk := text[:cut].strip():
                chunks.append(chunk)
            text, first = text[cut:], False
        if rest := text.strip():
            chunks.append(rest)
        return chunks

    def stream(self, *, language: str | None = None) -> SentenceStream:
        return _SentenceStream(first_min_len=self._first_min_len)


class _SentenceStream(SentenceStream):
    def __init__(self, *, first_min_len: int) -> None:
        super().__init__()
        self._first_min_len = first_min_len
        self._buf = ""
        self._first = True
        self._segment_id = shortuuid()

    def push_text(self, text: str) -> None:
        self._check_not_closed()
        self._buf += text
        while (
            cut := _find_cut(self._buf, first=self._first, first_min_len=self._first_min_len)
        ) is not None:
            self._send(self._buf[:cut])
            self._buf = self._buf[cut:]
            self._first = False

    def flush(self) -> None:
        self._check_not_closed()
        self._send(self._buf)
        self._buf = ""
        self._first = True
        self._segment_id = shortuuid()

    def end_input(self) -> None:
        self.flush()
        self._do_close()

    async def aclose(self) -> None:
        self._do_close()

    def _send(self, chunk: str) -> None:
        chunk = chunk.strip()
        if any(ch.isalnum() for ch in chunk):  # a stray closing quote is not worth a request
            self._event_ch.send_nowait(TokenData(segment_id=self._segment_id, token=chunk))
