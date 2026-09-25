"""The interface every local-LLM backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..types import LLMResult, ModelEntry
from ..ui import UI


class BackendError(RuntimeError):
    """A backend failed in a way the player should be told about."""


class LLMBackend(ABC):
    name: str = "base"

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
    ) -> LLMResult:
        """Run one chat completion. `messages` use OpenAI-style roles.

        Implementations must split exposed reasoning (e.g. <think> blocks or a
        separate `thinking` field) out of the answer into `LLMResult.reasoning`.
        Raises BackendError on failure.
        """

    @property
    @abstractmethod
    def model_label(self) -> str:
        """Human-readable model name for display."""
