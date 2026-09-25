"""Talk to a local Ollama server over its plain HTTP API.

`Ollama <https://ollama.com>`_ is a free app that runs open-weight models on
your computer and listens on ``http://127.0.0.1:11434``. This backend uses
four of its endpoints, with nothing but the standard library's
``urllib.request`` so you can see exactly what is sent:

* ``GET  /api/version`` - is Ollama running?
* ``GET  /api/tags``    - which models are already downloaded?
* ``POST /api/pull``    - download a model. For Hugging Face GGUF files the
  name looks like ``hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M``. Progress arrives as
  "NDJSON": one small JSON object per line, e.g.
  ``{"status": "pulling abc123", "digest": "abc123", "total": 2500000000, "completed": 1200000}``.
* ``POST /api/chat``    - one chat reply (``"stream": false``). Thinking
  models can return their chain-of-thought in ``message.thinking``.
* ``HEAD/POST /api/blobs/sha256:<digest>`` + ``POST /api/create`` - hand
  Ollama a GGUF file the game already downloaded, instead of downloading the
  same model again (used when the built-in engine couldn't start it).

The HTTP layer is injectable (``http=``), which is how the tests pretend to
be Ollama without a network.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import socket
import time
import urllib.error
import urllib.request
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional

from rich.markup import escape

from ..reasoning import split_reasoning
from ..types import LLMResult, ModelEntry
from ..ui import UI
from .base import RETRY_WITHOUT_THINKING_NOTICE, BackendError, LLMBackend

# What a flaky connection can raise: socket errors are OSErrors, but a reply
# cut off part-way (Ollama quit or restarted mid-answer, a remote OLLAMA_HOST
# behind a proxy...) raises http.client's IncompleteRead, which isn't one.
_NETWORK_ERRORS: tuple[type[BaseException], ...] = (OSError, http.client.HTTPException)

__all__ = [
    "OllamaBackend",
    "OllamaError",
    "HttpResponse",
    "UrllibHttp",
    "normalize_host",
    "same_model",
    "DEFAULT_HOST",
    "OLLAMA_DOWNLOAD_URL",
    "START_OLLAMA_HINT",
]

DEFAULT_PORT = 11434
DEFAULT_HOST = f"http://127.0.0.1:{DEFAULT_PORT}"
OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
START_OLLAMA_HINT = "Start the Ollama app, or run 'ollama serve' in a terminal."

PROBE_TIMEOUT_S = 1.5  # "is it running?" must feel instant
TAGS_TIMEOUT_S = 5.0
CHAT_TIMEOUT_S = 300.0  # the minimum; scaled up for slow models once we've measured them
CHAT_TIMEOUT_MAX_S = 1800.0
CHAT_PROMPT_ALLOWANCE_S = 60.0
PULL_READ_TIMEOUT_S = 600.0  # per read: Ollama may go quiet while verifying a big file
MIN_BAR_BYTES = 1_000_000  # don't draw progress bars for tiny layers (license, template...)
BENCHMARK_PROMPT = "Count from 1 to 40, separated by commas. Reply with the numbers only."

# What Ollama's pull status lines mean, in plain English.
_PULL_STATUS_MESSAGES = {
    "pulling manifest": "Asking for the model's list of files...",
    "verifying sha256 digest": "Checking the download isn't damaged (comparing its SHA-256 fingerprint)...",
    "writing manifest": "Filing the model away in Ollama's library...",
}


# ---------------------------------------------------------------------------
# OLLAMA_HOST handling
# ---------------------------------------------------------------------------


def normalize_host(value: Optional[str]) -> str:
    """Turn an ``OLLAMA_HOST``-style value into a base URL we can connect to.

    Follows the same rules as Ollama itself (``envconfig.Host`` +
    ``ConnectableHost`` in the Ollama source):

    * empty -> ``http://127.0.0.1:11434``
    * no scheme -> ``http://`` and port 11434 (``"localhost"`` -> ``http://localhost:11434``)
    * explicit scheme but no port -> that scheme's usual port (http 80, https 443)
    * ``0.0.0.0`` / ``::`` mean "every network card" to a *server*; a client
      can't connect to them (it fails on Windows), so they become loopback.

    >>> normalize_host("0.0.0.0:11434")
    'http://127.0.0.1:11434'
    """
    s = (value or "").strip()
    if not s:
        return DEFAULT_HOST
    scheme, sep, rest = s.partition("://")
    if sep:
        scheme = scheme.lower()
        default_port = {"http": 80, "https": 443}.get(scheme, DEFAULT_PORT)
    else:
        scheme, rest, default_port = "http", s, DEFAULT_PORT

    hostport, _, path = rest.partition("/")
    host, port_text = _split_host_port(hostport)
    port = int(port_text) if port_text.isdecimal() and 0 < int(port_text) <= 65535 else default_port

    host = host or "127.0.0.1"
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        ip = None  # a name like "localhost" or "my-server"
    if ip is not None and ip.is_unspecified:
        host = "127.0.0.1" if ip.version == 4 else "::1"
    if ":" in host:  # IPv6 literals need brackets in URLs
        host = f"[{host}]"
    url = f"{scheme}://{host}:{port}"
    path = path.strip("/")
    return f"{url}/{path}" if path else url


def _split_host_port(hostport: str) -> tuple[str, str]:
    """``"[::1]:80"`` -> ``("::1", "80")``; ``"box:1"`` -> ``("box", "1")``; ``"box"`` -> ``("box", "")``."""
    if hostport.startswith("["):
        inside, _, after = hostport[1:].partition("]")
        return inside, after[1:] if after.startswith(":") else ""
    if hostport.count(":") == 1:
        host, _, port = hostport.partition(":")
        return host, port
    return hostport, ""  # plain name, or a bare IPv6 address without a port


def _is_loopback_url(url: str) -> bool:
    host = url.split("://", 1)[-1].split("/", 1)[0]
    host, _ = _split_host_port(host)
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_local_host(url: str) -> bool:
    """Is this Ollama address this computer (localhost, 127.x, ::1)?"""
    return _is_loopback_url(url)


def display_host(url: str) -> str:
    """"http://gpu-box.lan:11434" -> "gpu-box.lan:11434" (for telling the player where it is)."""
    return url.split("://", 1)[-1].rstrip("/")


def same_model(wanted: str, listed: str) -> bool:
    """Do two Ollama model names refer to the same model?

    Case-insensitive; a missing tag means ``:latest``; ``huggingface.co/`` is
    the same as ``hf.co/``; ``registry.ollama.ai/library/`` prefixes are
    dropped (``ollama list`` shows short names).
    """
    return bool(wanted.strip()) and _canonical_name(wanted) == _canonical_name(listed)


def _canonical_name(name: str) -> str:
    n = name.strip().lower()
    for prefix, replacement in (("https://", ""), ("huggingface.co/", "hf.co/"),
                                ("registry.ollama.ai/", ""), ("library/", "")):
        if n.startswith(prefix):
            n = replacement + n[len(prefix):]
    if ":" not in n.rsplit("/", 1)[-1]:
        n += ":latest"
    return n


# ---------------------------------------------------------------------------
# A tiny HTTP layer (standard library only)
# ---------------------------------------------------------------------------


class HttpResponse:
    """A minimal HTTP response: ``status`` plus file-like ``read``/``readline``."""

    def __init__(self, status: int, stream: Any) -> None:
        self.status = int(status)
        self._stream = stream

    def read(self, n: int = -1) -> bytes:
        if self._stream is None:
            return b""
        return self._stream.read() if n is None or n < 0 else self._stream.read(n)

    def readline(self) -> bytes:
        if self._stream is None:
            return b""
        return self._stream.readline()

    def close(self) -> None:
        try:
            if self._stream is not None:
                self._stream.close()
        except Exception:
            pass

    def __enter__(self) -> "HttpResponse":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class UrllibHttp:
    """The real HTTP layer: ``request(method, url, *, headers, body, timeout)``.

    HTTP error statuses (404, 500...) come back as responses, not exceptions,
    so the caller can read Ollama's ``{"error": "..."}`` message. Network
    problems (refused, timed out) raise ``OSError``. ``use_proxy=False``
    stops a system-wide proxy setting from intercepting ``127.0.0.1`` traffic.
    """

    def __init__(self, *, use_proxy: bool = True) -> None:
        handlers = [] if use_proxy else [urllib.request.ProxyHandler({})]
        self._opener = urllib.request.build_opener(*handlers)

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Optional[Mapping[str, str]] = None,
        body: Any = None,  # bytes, or an open file (streamed; give a Content-Length header)
        timeout: float = 30.0,
    ) -> HttpResponse:
        req = urllib.request.Request(url, data=body, headers=dict(headers or {}), method=method)
        try:
            resp = self._opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as err:
            return HttpResponse(err.code, err.fp)
        return HttpResponse(resp.status, resp)


def _is_timeout(exc: BaseException) -> bool:
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return True
    reason = getattr(exc, "reason", None)
    return isinstance(reason, (socket.timeout, TimeoutError))


def _error_text(raw: bytes) -> str:
    """Ollama's error bodies look like ``{"error": "model 'x' not found"}``."""
    text = raw.decode("utf-8", "replace").strip()
    try:
        data = json.loads(text)
    except ValueError:
        return text[:300] or "no details"
    if isinstance(data, dict) and data.get("error"):
        return str(data["error"])
    return text[:300] or "no details"


class OllamaError(BackendError):
    """An error reported by Ollama itself. ``detail`` is Ollama's own wording."""

    def __init__(self, message: str, *, status: Optional[int] = None, detail: str = "") -> None:
        super().__init__(message)
        self.status = status
        self.detail = detail


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class OllamaBackend(LLMBackend):
    """Run the model through a local Ollama server."""

    name = "ollama"

    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        *,
        http: Any = None,
        think: Optional[bool] = None,
        n_ctx: Optional[int] = None,
        clock: Any = None,
        gguf_path: Optional[Path] = None,
    ) -> None:
        """
        Args:
            model: the Ollama model name, e.g. ``"hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"``
                (may be ``""`` if :meth:`prepare` will be given a catalog entry).
            host: server address; defaults to ``$OLLAMA_HOST`` or ``http://127.0.0.1:11434``.
            http: object with a ``request(...)`` method like :class:`UrllibHttp` (tests pass a fake).
            think: ask the model to think out loud? ``None`` = decide from the
                catalog entry's ``reasoning`` flag in :meth:`prepare` (or let Ollama decide).
            n_ctx: context window to request (``options.num_ctx``); default from the entry, else 4096.
            gguf_path: a model file already on this computer. If Ollama doesn't have
                the model yet, this file is handed to it (``ollama create``) instead
                of pulling it from the internet again.
        """
        self.model = (model or "").strip()
        self.host = normalize_host(host if host is not None else os.environ.get("OLLAMA_HOST"))
        self._http = http or UrllibHttp(use_proxy=not _is_loopback_url(self.host))
        self.think = think
        self.n_ctx = n_ctx
        self.entry: Optional[ModelEntry] = None
        self.version: Optional[str] = None
        self._clock = clock or time.perf_counter
        self._think_supported = True  # flips to False if Ollama says "does not support thinking"
        self.tokens_per_s: Optional[float] = None  # measured by benchmark(); scales chat timeouts
        self.gguf_path: Optional[Path] = Path(gguf_path) if gguf_path else None

    # -- LLMBackend API --------------------------------------------------------

    @property
    def model_label(self) -> str:
        if self.entry is not None:
            return f"{self.entry.display_name} (via Ollama)"
        return f"{self.model or 'no model chosen'} (via Ollama)"

    def is_available(self) -> tuple[bool, str]:
        """``GET /api/version`` with a short timeout. Never raises."""
        not_running = (
            f"Ollama isn't running at {self.host}. {START_OLLAMA_HINT} "
            f"(Don't have it? It's free: {OLLAMA_DOWNLOAD_URL})"
        )
        try:
            status, raw = self._request("GET", "/api/version", timeout=PROBE_TIMEOUT_S)
            if status != 200:
                return False, f"Something answered at {self.host}, but it doesn't look like Ollama (HTTP {status})."
            data = json.loads(raw.decode("utf-8"))
            version = str(data.get("version", "")).strip() if isinstance(data, dict) else ""
        except Exception:  # unreachable, garbled reply, anything at all: is_available must never raise
            return False, not_running
        self.version = version or None
        return True, f"Ollama {version or '(unknown version)'} is running at {self.host}."

    def list_models(self) -> list[str]:
        """Names of the models Ollama already has (``GET /api/tags``). Raises BackendError."""
        data = self._get_json("/api/tags", TAGS_TIMEOUT_S)
        names: list[str] = []
        for item in data.get("models") or []:
            if isinstance(item, dict):
                for key in ("name", "model"):
                    value = item.get(key)
                    if isinstance(value, str) and value and value not in names:
                        names.append(value)
        return names

    def has_model(self) -> bool:
        """Has Ollama already downloaded our model? ``False`` if unsure (never raises)."""
        if not self.model:
            return False
        try:
            return any(same_model(self.model, name) for name in self.list_models())
        except BackendError:
            return False

    def prepare(self, ui: UI, entry: Optional[ModelEntry] = None) -> None:
        """Make sure Ollama is running and has the model, downloading it if needed."""
        if entry is not None:
            self.entry = entry
            if not self.model:
                self.model = entry.ollama_ref
            if self.think is None:
                self.think = bool(entry.reasoning)
            if self.n_ctx is None:
                self.n_ctx = entry.context_tokens
        if not self.model:
            raise BackendError("No model was chosen for Ollama to run.")

        ok, why = self.is_available()
        if not ok:
            raise BackendError(why)
        if self.has_model():
            ui.success(f"Ollama already has {self.model} - no download needed.")
            return
        if self.gguf_path is not None and self.gguf_path.is_file():
            self._import_gguf(ui, self.gguf_path)
            ui.success(f"Ollama now has {self.model} - nothing new was downloaded.")
            return
        ui.info(f"Asking Ollama to download {self.model}. It's saved by Ollama, so this only happens once.")
        self._pull(ui)
        ui.success(f"Ollama finished downloading {self.model}.")

    # -- handing over a file we already have ------------------------------------------

    def _import_gguf(self, ui: UI, path: Path) -> None:
        """Give Ollama a local GGUF file: upload it as a "blob", then create a model from it.

        Ollama stores models in its own folder, so it keeps its own copy - but
        nothing is downloaded from the internet again. If Ollama already has
        that exact file (same SHA-256), the upload is skipped.
        """
        with ui.status(f"Fingerprinting {escape(path.name)} for Ollama (SHA-256)..."):
            digest = "sha256:" + _sha256_file(path)
        status, _raw = self._request("HEAD", f"/api/blobs/{digest}", timeout=TAGS_TIMEOUT_S)
        if status != 200:
            with ui.status(f"Handing {escape(path.name)} to Ollama (a local copy, no download)..."):
                self._upload_blob(path, digest)
        body = {"model": self.model, "files": {path.name: digest}, "stream": False}
        status, raw = self._request("POST", "/api/create", body=body, timeout=PULL_READ_TIMEOUT_S)
        if status != 200:
            detail = _error_text(raw)
            raise OllamaError(f"Ollama couldn't take the model file I downloaded ({detail}).", status=status,
                              detail=detail)

    def _upload_blob(self, path: Path, digest: str) -> None:
        size = path.stat().st_size
        headers = {"Content-Type": "application/octet-stream", "Content-Length": str(size)}
        try:
            with open(path, "rb") as fh:
                resp = self._http.request("POST", f"{self.host}/api/blobs/{digest}", headers=headers, body=fh,
                                          timeout=PULL_READ_TIMEOUT_S)
                with resp:
                    status, raw = resp.status, resp.read()
        except _NETWORK_ERRORS as exc:
            raise BackendError(self._network_message(exc, "hand it the model file")) from exc
        if status not in (200, 201):
            detail = _error_text(raw)
            raise OllamaError(f"Ollama couldn't take the model file I downloaded ({detail}).", status=status,
                              detail=detail)

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
        """One chat reply via ``POST /api/chat``, with thinking split out.

        ``think`` overrides the backend's default for this one call
        (``False`` = answer straight away); ``stop`` ends the answer early.
        """
        if not self.model:
            raise BackendError("No model was chosen for Ollama to run.")
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,  # Ollama's name for "max tokens to generate"
            "num_ctx": self.n_ctx or 4096,  # context window (Ollama's own default is smaller)
        }
        if stop:
            options["stop"] = [str(x) for x in stop]
        body: dict[str, Any] = {"model": self.model, "messages": messages, "stream": False, "options": options}
        if json_mode:
            body["format"] = "json"  # Ollama constrains the output to valid JSON
        wanted = self.think if think is None else bool(think)
        if wanted is not None and self._think_supported:
            body["think"] = bool(wanted)

        timeout = self._chat_timeout(max_tokens)
        start = self._clock()
        data = self._post_chat(body, timeout)
        answer, reasoning = _extract_answer(data)
        if not answer:
            # A thinking model can spend its whole budget thinking and never
            # answer. Ask again with thinking off: the answer alone is short,
            # so the same budget is plenty. ("think": false is always
            # accepted; only "think": true can be refused.)
            self._notice(RETRY_WITHOUT_THINKING_NOTICE)
            retry = {k: v for k, v in body.items() if k != "think"}
            if self._think_supported:
                retry["think"] = False
            data = self._post_chat(retry, timeout)
            answer, second = _extract_answer(data)
            reasoning = _join(reasoning, second)
        return LLMResult(
            text=answer,
            reasoning=reasoning,
            model=self.model_label,
            backend=self.name,
            elapsed_s=self._clock() - start,
            messages=[dict(m) for m in messages],
            raw=data,
            truncated=data.get("done_reason") == "length",
        )

    def _chat_timeout(self, max_tokens: int) -> float:
        """At least CHAT_TIMEOUT_S; more for a slow model (1.5 x tokens / measured speed + prompt time)."""
        tps = self.tokens_per_s
        if not tps or tps <= 0:
            return CHAT_TIMEOUT_S
        needed = CHAT_PROMPT_ALLOWANCE_S + 1.5 * max(1, int(max_tokens)) / tps
        return float(min(CHAT_TIMEOUT_MAX_S, max(CHAT_TIMEOUT_S, needed)))

    def benchmark(self, ui: Optional[UI] = None) -> Optional[float]:
        """Tokens per second from a short test reply, or None if it fails.

        Ollama reports ``eval_count`` (tokens generated) and ``eval_duration``
        (nanoseconds spent generating them), so no stopwatch is needed.
        """
        if not self.model:
            return None
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": BENCHMARK_PROMPT}],
            "stream": False,
            "options": {"temperature": 0.0, "num_predict": 64, "num_ctx": self.n_ctx or 4096},
        }
        if self.think is not None and self._think_supported:
            body["think"] = False  # we're timing the talking, not the pondering
        spinner = ui.status("Timing a quick test sentence to see how fast your model talks...") if ui else nullcontext()
        try:
            with spinner:
                start = self._clock()
                data = self._post_chat(body)
                elapsed = self._clock() - start
        except BackendError:
            return None
        count, duration_ns = data.get("eval_count"), data.get("eval_duration")
        speed: Optional[float] = None
        if isinstance(count, (int, float)) and isinstance(duration_ns, (int, float)) and count > 0 and duration_ns > 0:
            speed = float(count) / (float(duration_ns) / 1e9)
        elif isinstance(count, (int, float)) and count > 0 and elapsed > 0:
            speed = float(count) / elapsed
        if speed is not None:
            self.tokens_per_s = speed
        return speed

    # -- pulling (downloading) ---------------------------------------------------

    def _pull(self, ui: UI) -> None:
        body = json.dumps({"model": self.model, "stream": True}).encode("utf-8")
        try:
            resp = self._http.request(
                "POST", self.host + "/api/pull",
                headers={"Content-Type": "application/json", "Accept": "application/x-ndjson"},
                body=body, timeout=PULL_READ_TIMEOUT_S,
            )
        except _NETWORK_ERRORS as exc:
            raise BackendError(self._network_message(exc, "download the model")) from exc

        progress = _PullProgress(ui, self.model.rsplit("/", 1)[-1])
        succeeded = False
        announced: set[str] = set()
        try:
            with resp:
                if resp.status != 200:
                    detail = _error_text(resp.read())
                    raise OllamaError(_friendly_pull_error(self.model, detail), status=resp.status, detail=detail)
                for event in _iter_ndjson(resp):
                    if event.get("error"):
                        detail = str(event["error"])
                        raise OllamaError(_friendly_pull_error(self.model, detail), detail=detail)
                    status = str(event.get("status") or "")
                    total, completed = event.get("total"), event.get("completed")
                    if event.get("digest") and isinstance(total, int) and total > 0:
                        progress.update(str(event["digest"]), total, completed if isinstance(completed, int) else 0)
                    elif status == "success":
                        succeeded = True
                    elif status in _PULL_STATUS_MESSAGES and status not in announced:
                        progress.close()
                        announced.add(status)
                        ui.info(_PULL_STATUS_MESSAGES[status])
        except _NETWORK_ERRORS as exc:
            if _is_timeout(exc):
                raise BackendError(
                    f"The download of {self.model} went quiet for over {int(PULL_READ_TIMEOUT_S // 60)} minutes. "
                    "Check your internet connection and try again - Ollama resumes where it left off."
                ) from exc
            raise BackendError(self._network_message(exc, "download the model")) from exc
        finally:
            progress.close()
        if not succeeded:
            raise BackendError(
                f"Ollama stopped before it finished downloading {self.model}. "
                "Try again - Ollama resumes where it left off."
            )

    # -- HTTP helpers --------------------------------------------------------------

    def _request(self, method: str, path: str, *, body: Optional[dict] = None, timeout: float) -> tuple[int, bytes]:
        """Send one request; returns ``(status, body bytes)``. Network failures -> BackendError."""
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        try:
            resp = self._http.request(method, self.host + path, headers=headers, body=payload, timeout=timeout)
            with resp:
                return resp.status, resp.read()
        except _NETWORK_ERRORS as exc:
            raise BackendError(self._network_message(exc, "talk to it", timeout)) from exc

    def _get_json(self, path: str, timeout: float) -> dict:
        status, raw = self._request("GET", path, timeout=timeout)
        return self._decode(status, raw)

    def _post_chat(self, body: dict, timeout: float = CHAT_TIMEOUT_S) -> dict:
        """``POST /api/chat``; if Ollama rejects the ``think`` flag, retry once without it."""
        try:
            status, raw = self._request("POST", "/api/chat", body=body, timeout=timeout)
            return self._decode(status, raw)
        except OllamaError as exc:
            if "think" not in body or "think" not in exc.detail.lower():
                raise
            # e.g. '"hf.co/...:Q4_K_M" does not support thinking' - remember and carry on.
            self._think_supported = False
            body = {k: v for k, v in body.items() if k != "think"}
            status, raw = self._request("POST", "/api/chat", body=body, timeout=timeout)
            return self._decode(status, raw)

    def _decode(self, status: int, raw: bytes) -> dict:
        if status != 200:
            detail = _error_text(raw)
            raise OllamaError(self._friendly_status_error(status, detail), status=status, detail=detail)
        try:
            data = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise BackendError("Ollama sent back something that wasn't valid JSON.") from exc
        if not isinstance(data, dict):
            raise BackendError("Ollama sent back an unexpected answer.")
        if data.get("error"):
            detail = str(data["error"])
            raise OllamaError(self._friendly_status_error(status, detail), status=status, detail=detail)
        return data

    def _friendly_status_error(self, status: int, detail: str) -> str:
        low = detail.lower()
        if status == 404 or ("not found" in low and "model" in low):
            return (
                f"Ollama doesn't have the model '{self.model}' yet. Pick it again in the game and "
                f"I'll download it, or run: ollama pull {self.model}"
            )
        if "memory" in low:
            return f"Ollama ran out of memory loading {self.model}. Close some apps, or pick a smaller model. ({detail})"
        return f"Ollama reported a problem (HTTP {status}): {detail}"

    def _network_message(self, exc: BaseException, doing: str, timeout: Optional[float] = None) -> str:
        if _is_timeout(exc):
            if timeout is not None and timeout >= 60:
                return (
                    f"Ollama took longer than {int(timeout // 60)} minutes to answer. The model may be too big "
                    "for this computer - a smaller one would be much snappier."
                )
            return f"Ollama at {self.host} didn't answer in time. {START_OLLAMA_HINT}"
        if isinstance(exc, http.client.HTTPException):
            return (f"Ollama at {self.host} stopped part-way through its answer (was it closed or restarted?) "
                    f"while I tried to {doing}. Trying again usually works. {START_OLLAMA_HINT}")
        return f"I couldn't reach Ollama at {self.host} to {doing}. {START_OLLAMA_HINT}"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _iter_ndjson(resp: Any) -> Iterator[dict]:
    """Yield each JSON object from a newline-delimited JSON stream, as it arrives."""
    if hasattr(resp, "readline"):
        lines: Any = iter(resp.readline, b"")  # call readline() until it returns b""
    else:
        lines = iter(resp.read().splitlines())
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line.decode("utf-8", "replace"))
        except ValueError:
            continue  # ignore a garbled line rather than abort a big download
        if isinstance(event, dict):
            yield event


class _PullProgress:
    """Turns Ollama's per-layer progress events into download bars.

    A model is stored as a few "layers" (the big weights file plus tiny ones
    such as the license and chat template). Ollama downloads them one after
    another, so we show one bar at a time and skip the tiny ones.
    """

    def __init__(self, ui: UI, model_name: str) -> None:
        self._ui = ui
        self._model_name = model_name
        self._digest: Optional[str] = None
        self._bar: Any = None
        self._advance: Any = None
        self._shown = 0
        self._bars_opened = 0

    def update(self, digest: str, total: int, completed: int) -> None:
        if digest != self._digest:
            self.close()
            self._digest = digest
            self._shown = 0
            if total >= MIN_BAR_BYTES:
                label = f"Downloading {self._model_name}" if self._bars_opened == 0 else "Downloading extra files"
                self._bar = self._ui.download_progress(label, total)
                self._advance = self._bar.__enter__()
                self._bars_opened += 1
        if self._advance is not None and completed > self._shown:
            self._advance(completed - self._shown)
            self._shown = completed

    def close(self) -> None:
        bar, self._bar, self._advance = self._bar, None, None
        if bar is not None:
            bar.__exit__(None, None, None)


def _friendly_pull_error(model: str, detail: str) -> str:
    low = detail.lower()
    if "does not exist" in low or "not found" in low or "manifest unknown" in low or "invalid model name" in low:
        return (
            f"Ollama couldn't find a model called '{model}'. Double-check the name - Hugging Face models "
            "look like hf.co/<user>/<repo>:<quant>."
        )
    if "401" in low or "403" in low or "unauthorized" in low or "gated" in low or "access" in low:
        return (
            f"'{model}' needs you to sign in or accept its license on Hugging Face first, which Ollama "
            "can't do for you. Try a different model."
        )
    if "no space" in low or "disk" in low:
        return f"Your disk is too full to download {model}. Free up some space and try again."
    if "dial tcp" in low or "lookup" in low or "timeout" in low or "connection" in low or "network" in low:
        return (
            f"Ollama couldn't reach the internet to download {model} ({detail}). "
            "Check your connection and try again - it resumes where it left off."
        )
    return f"Ollama couldn't download {model}: {detail}"


def _extract_answer(data: dict) -> tuple[str, Optional[str]]:
    """Answer + reasoning from an ``/api/chat`` reply.

    Reasoning is ``message.thinking`` when Ollama separated it, plus anything
    left inside ``<think>`` tags in the content (older Ollama versions).
    """
    message = data.get("message")
    message = message if isinstance(message, dict) else {}
    content = message.get("content")
    thinking = message.get("thinking")
    answer, tagged = split_reasoning(content if isinstance(content, str) else "")
    return answer, _join(thinking if isinstance(thinking, str) else None, tagged)


def _join(*parts: Optional[str]) -> Optional[str]:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned) if cleaned else None
