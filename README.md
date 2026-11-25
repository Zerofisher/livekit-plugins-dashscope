# livekit-plugins-dashscope

LiveKit Agents plugin for Alibaba Cloud DashScope: Fun-ASR realtime speech-to-text (`STT`) and
CosyVoice text-to-speech (`TTS`), plus a Chinese-aware `SentenceTokenizer`.

Not published to PyPI; install it from a git tag.

## Install

```bash
uv add "livekit-plugins-dashscope @ git+https://github.com/Zerofisher/livekit-plugins-dashscope@v0.4.1"
```

or in `pyproject.toml`:

```toml
[project]
dependencies = ["livekit-plugins-dashscope"]

[tool.uv.sources]
livekit-plugins-dashscope = { git = "https://github.com/Zerofisher/livekit-plugins-dashscope", tag = "v0.4.1" }
```

Requires `livekit-agents>=1.8.6,<1.9`.

## Usage

```python
from livekit.agents import AgentSession
from livekit.plugins import dashscope

session = AgentSession(
    stt=dashscope.STT(),  # fun-asr-realtime, zh
    tts=dashscope.TTS(),
    # ...
)
```

Credentials and region come from the environment:

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL` (optional): defaults to the Beijing endpoint; use
  `wss://dashscope-intl.aliyuncs.com/api-ws/v1/inference/` for Singapore keys.

## Development

```bash
uv sync
uv run pytest                 # hermetic unit tests
uv run pytest -m plugin       # integration tests against the real API (DASHSCOPE_API_KEY)
uv run ruff format --check . && uv run ruff check . && uv run mypy
```

Releases: bump `livekit/plugins/dashscope/version.py`, commit, then `git tag vX.Y.Z && git push --tags`.
