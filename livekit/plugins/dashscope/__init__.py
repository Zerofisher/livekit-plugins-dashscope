from .sentences import SentenceTokenizer
from .stt import STT, SpeechStream
from .tts import TTS, SynthesizeStream
from .version import __version__

__all__ = ["STT", "SpeechStream", "SentenceTokenizer", "TTS", "SynthesizeStream", "__version__"]

from livekit.agents import Plugin

from .log import logger


class DashScopePlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(__name__, __version__, __package__, logger)


Plugin.register_plugin(DashScopePlugin())

_module = dir()
NOT_IN_ALL = [m for m in _module if m not in __all__]
__pdoc__ = dict.fromkeys(NOT_IN_ALL, False)
