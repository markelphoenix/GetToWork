"""The interface every local-LLM backend implements."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from ..types import LLMResult, ModelEntry
from ..ui import UI


class BackendError(RuntimeError):
    """A backend failed in a way the player should be told about."""


class EngineStopped(BackendError):
    """The engine started, then broke on its first real work and couldn't be
    recovered (not even in CPU mode): setup treats it like a start-up failure."""


# Shown while a thinking model that used its whole budget thinking is asked again without thinking.
RETRY_WITHOUT_THINKING_NOTICE = "Your model thought for so long it ran out of room - asking it to just answer…"


class LLMBackend(ABC):
    name: str = "base"
    #: Set by the game while a call runs, to show a short note in the spinner
    #: (e.g. that a thinking model is being asked again without thinking).
    on_notice: Optional[Callable[[str], None]] = None

    def _notice(self, text: str) -> None:
        """Tell the player what's happening mid-call, if anyone is listening. Never raises."""
        callback = getattr(self, "on_notice", None)
        if callback is None:
            return
        try:
            callback(text)
        except Exception:  # a status line is never worth failing a call over
            pass

    @abstractmethod
    def is_available(self) -> tuple[bool, str]:
        """(usable?, human explanation). Must be fast and never raise."""

    @abstractmethod
    def prepare(self, ui: UI, entry: Optional[ModelEntry] = None) -> None:
        """Make sure the model is downloaded/pulled and loadable.

        Shows progress through `ui`. Raises BackendError on failure.
        """

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 700,
        json_mode: bool = False,
        think: Optional[bool] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResult:
        """Run one chat completion. `messages` use OpenAI-style roles.

        * ``think``: ``False`` asks a "thinking" model to skip its visible
          reasoning (faster), ``True`` allows it, ``None`` = the model's default.
        * ``stop``: text that ends the answer early if the model writes it
          (e.g. ``"\nPlayer:"`` when a small model starts playing both sides).

        Implementations must split exposed reasoning (e.g. <think> blocks or a
        separate `thinking` field) out of the answer into `LLMResult.reasoning`,
        and set `LLMResult.truncated` when the answer hit `max_tokens`.
        Raises BackendError on failure.
        """

    @property
    @abstractmethod
    def model_label(self) -> str:
        """Human-readable model name for display."""

    def benchmark(self, ui: Optional[UI] = None) -> Optional[float]:
        """Measure generation speed in tokens/second, or None if unsupported."""
        return None

    def close(self) -> None:
        """Release resources (e.g. stop a server subprocess). Safe to call twice."""


def supported_chat_options(backend: Any) -> frozenset[str]:
    """Which optional ``chat()`` keywords (``think``, ``stop``) a backend accepts.

    Backends written against the original interface (or simple test fakes)
    don't take them; the game only passes the ones a backend understands.
    """
    try:
        params = inspect.signature(backend.chat).parameters
    except (TypeError, ValueError, AttributeError):
        return frozenset()
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return frozenset({"think", "stop"})
    return frozenset(name for name in ("think", "stop") if name in params)
