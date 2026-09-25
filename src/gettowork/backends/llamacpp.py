"""Run a GGUF model inside this Python process with ``llama-cpp-python``.

This backend is for tinkerers (``--backend llamacpp``). The optional
`llama-cpp-python <https://github.com/abetlen/llama-cpp-python>`_ package
(MIT) wraps the same llama.cpp engine the default backend downloads, but as a
Python library instead of a separate server program. Installing it may
compile C++ code, which is why it isn't the default.

The package is imported only when it's actually needed ("lazy import"), so
the rest of the game works fine without it. Tests pass ``llama_factory=`` to
stand in for ``llama_cpp.Llama``.
"""

from __future__ import annotations

import json
import time
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable, Optional

from rich.markup import escape

from ..reasoning import split_reasoning
from ..types import LLMResult, ModelEntry
from ..ui import UI
from .base import RETRY_WITHOUT_THINKING_NOTICE, BackendError, LLMBackend

__all__ = ["LlamaCppBackend", "INSTALL_HINT", "is_qwen3", "add_no_think"]

INSTALL_HINT = (
    "llama-cpp-python isn't installed. You can add it with: pip install llama-cpp-python "
    "(or just use the default engine, which needs no extra installs)."
)
BENCHMARK_PROMPT = "Count from 1 to 40, separated by commas. Reply with the numbers only."


def is_qwen3(entry: Optional[ModelEntry], model_path: Optional[Path] = None) -> bool:
    """Is this a Qwen3-family model? (They understand the ``/no_think`` switch.)"""
    names: list[str] = []
    if entry is not None:
        names += [entry.family, entry.key, entry.hf_repo, entry.architecture or "", entry.display_name]
    if model_path is not None:
        names.append(Path(model_path).name)
    return any("qwen3" in n.lower().replace(" ", "").replace("-", "") for n in names if n)


def add_no_think(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """Copy of ``messages`` with Qwen3's ``/no_think`` switch added to the last user turn."""
    copied = [dict(m) for m in messages]
    for message in reversed(copied):
        if message.get("role") == "user":
            message["content"] = f"{message.get('content', '')} /no_think"
            return copied
    copied.append({"role": "user", "content": "/no_think"})
    return copied


class LlamaCppBackend(LLMBackend):
    """Load a GGUF file with ``llama_cpp.Llama`` and chat with it directly."""

    name = "llamacpp"

    def __init__(
        self,
        model_path: Optional[Path] = None,
        entry: Optional[ModelEntry] = None,
        *,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        llama_factory: Optional[Callable[..., Any]] = None,
        quant: Optional[str] = None,
        downloader: Optional[Callable[..., Path]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        """
        Args:
            model_path: a GGUF file on disk. If omitted, :meth:`prepare` downloads ``entry``.
            entry: the catalog entry (used for the download, the label and Qwen3 detection).
            n_ctx: context window in tokens.
            n_gpu_layers: layers to put on the graphics card; ``-1`` = all of them, ``0`` = CPU only.
            llama_factory: stands in for ``llama_cpp.Llama`` (tests).
            quant: quantization to download (defaults to ``entry.quant``).
            downloader: stands in for ``gettowork.download.download_gguf`` (tests).
        """
        self.model_path = Path(model_path) if model_path is not None else None
        self.entry = entry
        self.n_ctx = int(n_ctx)
        self.n_gpu_layers = int(n_gpu_layers)
        self.quant = quant
        self._factory = llama_factory
        self._downloader = downloader
        self._clock = clock or time.perf_counter
        self._llm: Any = None

    # -- LLMBackend API --------------------------------------------------------

    @property
    def model_label(self) -> str:
        if self.entry is not None:
            quant = self.quant or self.entry.quant
            return f"{self.entry.display_name} ({quant})" if quant else self.entry.display_name
        if self.model_path is not None:
            return self.model_path.stem
        return "llama-cpp-python model"

    def is_available(self) -> tuple[bool, str]:
        """Is the ``llama_cpp`` package installed? (Checked without importing it.)"""
        if self._factory is not None:
            return True, "llama-cpp-python is ready."
        try:
            found = find_spec("llama_cpp") is not None
        except Exception:
            found = False
        return (True, "llama-cpp-python is installed.") if found else (False, INSTALL_HINT)

    def prepare(self, ui: UI, entry: Optional[ModelEntry] = None) -> None:
        """Download the model if needed, then load it into memory."""
        if entry is not None:
            self.entry = entry
        factory = self._get_factory()  # fail early (before a big download) if the package is missing
        self._ensure_model_file(ui)
        self.close()
        with ui.status("Loading the model into memory... big brains take a moment"):
            self._load(factory, ui)
        where = "on the CPU" if self.n_gpu_layers == 0 else "with graphics-card acceleration where available"
        ui.success(f"{escape(self.model_label)} is loaded and ready ({where}).")

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
        """One chat completion via ``Llama.create_chat_completion``.

        ``think=False`` adds Qwen3's "/no_think" switch (other models have no
        portable switch here); ``stop`` ends the answer early.
        """
        llm = self._ensure_loaded()
        qwen3 = is_qwen3(self.entry, self.model_path)
        request = add_no_think(messages) if think is False and qwen3 else messages
        start = self._clock()
        data = self._complete(llm, request, temperature, max_tokens, json_mode, stop)
        answer, reasoning = _extract_answer(data)
        if not answer:
            # Thinking models can use the whole budget thinking. For Qwen3 the
            # "/no_think" switch turns thinking off, so the same budget is
            # plenty; other models get more room instead.
            self._notice(RETRY_WITHOUT_THINKING_NOTICE)
            retry_messages = add_no_think(messages) if qwen3 else messages
            budget = max_tokens if qwen3 else max(max_tokens * 2, max_tokens + 512)
            data = self._complete(llm, retry_messages, temperature, budget, json_mode, stop)
            answer, second = _extract_answer(data)
            reasoning = _join(reasoning, second)
        return LLMResult(
            text=answer,
            reasoning=reasoning,
            model=self.model_label,
            backend=self.name,
            elapsed_s=self._clock() - start,
            messages=[dict(m) for m in messages],
            raw=_json_safe(data),
            truncated=_finish_reason(data) == "length",
        )

    def benchmark(self, ui: Optional[UI] = None) -> Optional[float]:
        """Tokens per second from a short test generation, or None if it fails."""
        if self._llm is None:
            return None
        messages = [{"role": "user", "content": BENCHMARK_PROMPT}]
        if is_qwen3(self.entry, self.model_path):
            messages = add_no_think(messages)
        spinner = ui.status("Timing a quick test sentence to see how fast your model talks...") if ui else nullcontext()
        try:
            with spinner:
                start = self._clock()
                data = self._complete(self._llm, messages, 0.0, 64, False)
                elapsed = self._clock() - start
        except BackendError:
            return None
        usage = data.get("usage") if isinstance(data, dict) else None
        tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if isinstance(tokens, (int, float)) and tokens > 0 and elapsed > 0:
            return float(tokens) / elapsed
        return None

    def close(self) -> None:
        """Free the model's memory. Safe to call more than once."""
        llm, self._llm = self._llm, None
        closer = getattr(llm, "close", None)
        if callable(closer):
            try:
                closer()
            except Exception:
                pass

    # -- internals ---------------------------------------------------------------

    def _get_factory(self) -> Callable[..., Any]:
        if self._factory is not None:
            return self._factory
        try:
            import llama_cpp  # optional dependency, imported only when needed
        except ImportError as exc:
            raise BackendError(INSTALL_HINT) from exc
        except Exception as exc:  # e.g. a broken native library inside the package
            raise BackendError(f"llama-cpp-python is installed but failed to start ({exc}).") from exc
        return llama_cpp.Llama

    def _ensure_model_file(self, ui: UI) -> Path:
        if self.model_path is not None:
            if self.model_path.is_file():
                return self.model_path
            if self.entry is None:
                raise BackendError(f"I can't find the model file {self.model_path}.")
        if self.entry is None:
            raise BackendError("No model has been chosen yet, so there's nothing to load.")
        downloader = self._downloader
        if downloader is None:
            from ..download import download_gguf  # imported lazily: only needed for downloads

            downloader = download_gguf
        try:
            path = downloader(self.entry, ui, quant=self.quant or self.entry.quant)
        except BackendError:
            raise
        except Exception as exc:  # DownloadError and friends already carry a friendly message
            raise BackendError(f"The model download didn't work: {exc}") from exc
        self.model_path = Path(path)
        return self.model_path

    def _load(self, factory: Callable[..., Any], ui: Optional[UI]) -> None:
        assert self.model_path is not None
        try:
            self._llm = self._construct(factory, self.n_gpu_layers)
            return
        except Exception as exc:
            if self.n_gpu_layers == 0:
                raise BackendError(_load_failure_message(self.model_path, exc)) from exc
            first_error = exc
        # The graphics card didn't cooperate (old driver, not enough video
        # memory...). Every layer on the CPU is slower but works almost anywhere.
        if ui is not None:
            ui.warn(f"The graphics card couldn't load the model ({escape(str(first_error))}). "
                    "Trying again on the CPU only...")
        try:
            self._llm = self._construct(factory, 0)
        except Exception as exc:
            raise BackendError(_load_failure_message(self.model_path, exc)) from exc
        self.n_gpu_layers = 0

    def _construct(self, factory: Callable[..., Any], n_gpu_layers: int) -> Any:
        return factory(model_path=str(self.model_path), n_ctx=self.n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def _ensure_loaded(self) -> Any:
        if self._llm is None:
            if self.model_path is None or not self.model_path.is_file():
                raise BackendError("The model isn't loaded yet - it needs to be prepared first.")
            self._load(self._get_factory(), None)
        return self._llm

    def _complete(self, llm: Any, messages: list[dict[str, str]], temperature: float,
                  max_tokens: int, json_mode: bool, stop: Optional[list[str]] = None) -> dict:
        kwargs: dict[str, Any] = {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if stop:
            kwargs["stop"] = [str(x) for x in stop]
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}  # grammar-constrained valid JSON
        try:
            data = llm.create_chat_completion(**kwargs)
        except Exception as exc:
            raise BackendError(f"The model hit a problem while writing its answer ({exc}).") from exc
        if not isinstance(data, dict):
            raise BackendError("llama-cpp-python returned an unexpected answer.")
        return data


def _extract_answer(data: dict) -> tuple[str, Optional[str]]:
    """Answer + reasoning from an OpenAI-style chat completion dict."""
    choices = data.get("choices") or []
    first = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first.get("message") if isinstance(first.get("message"), dict) else {}
    content = message.get("content")
    answer, tagged = split_reasoning(content if isinstance(content, str) else "")
    separate = message.get("reasoning_content")
    return answer, _join(separate if isinstance(separate, str) else None, tagged)


def _finish_reason(data: Any) -> Optional[str]:
    choices = data.get("choices") if isinstance(data, dict) else None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        reason = choices[0].get("finish_reason")
        return reason if isinstance(reason, str) else None
    return None


def _join(*parts: Optional[str]) -> Optional[str]:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned) if cleaned else None


def _json_safe(data: Any) -> Any:
    try:
        return json.loads(json.dumps(data, default=str))
    except (TypeError, ValueError):
        return None


def _load_failure_message(path: Path, exc: BaseException) -> str:
    text = str(exc).lower()
    if "memory" in text or "alloc" in text:
        return f"Your computer ran out of memory loading {path.name}. Try closing other apps or pick a smaller model."
    return (
        f"llama-cpp-python couldn't load {path.name} ({exc}). The file may be incomplete - deleting it and "
        "downloading again, or picking another model, usually fixes this."
    )
