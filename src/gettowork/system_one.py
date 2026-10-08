"""Local Clef as a System One referee.

Clef speaks the same ``POST /v1/systemone`` shape as Jev, but the weights run
on this computer through llama-server. The story model is a different program;
Clef only scores the referee's questions.

llama.cpp text support for architecture ``clef`` starts at release **b11371**
(3 October 2026; the release notes say "text-only"). Image input was added in
PR 29969 (5 October 2026) and is included in the pinned **b11485** engine.
The referee here stays text-only and does not download ``mmproj``. The menu
still shows Clef when the installed engine is older, and it offers to
download the pinned release when this copy of the game may download engines.
It does not download the weights until the engine can load them. A manual
"download anyway" cannot override a missing architecture.

Nothing here calls Cloudflare. The GGUF file, if the engine can load it, comes
from Hugging Face, and the server listens on 127.0.0.1.
"""

from __future__ import annotations

import atexit
import secrets
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Optional

from . import catalog, distribution, download, runtime_install
from .backends.llamaserver import build_server_args, find_free_port, server_env
from .config import runtime_dir
from .jev import JevClient
from .notices import LOCAL_RUN_NOTICES
from .types import FitResult, ModelEntry
from .ui import UI

# Official llama.cpp release whose notes say: "model: add support for clef
# decision model (text-only)". Image support after this build was not verified
# from a release note; the referee in this game is text-only and does not
# download the mmproj file.
CLEF_TEXT_MIN_BUILD = 11371

# "Learn" panel. Sizes are the published GGUF file sizes (decimal GB). Local
# video-memory need at the full 65,536-token window is not published.
TEACH_CLEF = """\
**Clef** is Cloudflare's open-weight decision model (**Apache-2.0**). It does not write the story.
It scores the referee's questions in one forward pass and returns probabilities.

Two GGUF repos are published by ggml-org (file sizes from Hugging Face):

- **Clef-flash** (9.08B parameters): Q4_K_M is 6.49 GB, Q8_0 is 9.66 GB, BF16 is 18.16 GB.
- **Clef** (27.02B parameters): Q4_K_M is 19.23 GB, Q8_0 is 28.73 GB, BF16 is 54.06 GB.

Cloudflare has not published how much video memory a local run needs, so the fit line is this
game's usual estimate (weights + a rule-of-thumb KV cache + overhead). The KV shape is
**unverified**. The published context is 65,536 tokens; this game asks for 4,096 for a short
referee call. The optional vision file (mmproj) is not downloaded.

llama.cpp **b11371** (3 October 2026) added **text-only** Clef support. Image input
was added in PR 29969 (5 October 2026). This game pins **b11485**, which includes both;
the referee stays text-only. If the installed engine is older, the menu can download
the pin when engine downloads are allowed, and it does not download the weights until
the engine can load them. Whether Ollama, vLLM or LM Studio can load architecture
`clef` was not verified.

Scoring stays on this computer (`127.0.0.1`). Nothing is sent to Cloudflare.
"""

Launcher = Callable[[ModelEntry, FitResult], Any]


class LocalClefUnavailable(RuntimeError):
    """The local Clef server was not started. The message is safe to show."""


class LocalClefReferee:
    """A Jev-shaped client aimed at a llama-server on this computer.

    ``referee_name`` is what the game prints instead of "Jev (...)".
    ``close`` stops the server process. ``system_one`` is the only call the
    game makes during a round.
    """

    def __init__(self, client: JevClient, stop: Callable[[], None], referee_name: str) -> None:
        self._client = client
        self._stop = stop
        self.referee_name = referee_name
        self.model = client.model
        self.base_url = client.base_url
        self._closed = False

    def system_one(self, state: Any, questions: dict) -> Any:
        return self._client.system_one(state, questions)

    def secret_values(self) -> frozenset:
        return self._client.secret_values()

    def scrub(self, value: Any) -> Any:
        return self._client.scrub(value)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop()


def clef_gguf_name(entry: ModelEntry, quant: str) -> str:
    """The ggml-org filename for this quant. Flash must not use the 27B name."""
    stem = "Clef-Flash" if entry.key == "clef-flash" else "Clef"
    return f"{stem}-{quant}.gguf"


def clef_engine_status(tag: Optional[str]) -> str:
    """``ok``, ``too_old`` or ``unknown``.

    Unknown (no tag, or no build number) refuses the download. A guess that
    the engine is new enough would fetch a multi-gigabyte file this process
    cannot load.
    """
    number = _tag_number(tag)
    if number < 0:
        return "unknown"
    if number < CLEF_TEXT_MIN_BUILD:
        return "too_old"
    return "ok"


def engine_block_message(tag: Optional[str], status: Optional[str] = None) -> str:
    """Why Clef will not be downloaded, in plain language."""
    status = status or clef_engine_status(tag)
    need = f"b{CLEF_TEXT_MIN_BUILD}"
    if status == "too_old":
        shown = tag or "an older build"
        return (
            f"This engine is llama.cpp {shown}, which cannot load Clef (architecture clef). "
            f"Text support starts at llama.cpp {need} "
            "(official release notes, 2026-10-03: text-only). "
            "The game will not download the model file until this engine can open it. "
            "Choosing Clef here cannot override that."
        )
    return (
        "I can't tell which llama.cpp build would run Clef, so I won't download the model yet. "
        f"Text support needs llama.cpp {need} or newer."
    )


def engine_upgrade_question(tag: Optional[str]) -> Optional[str]:
    """The yes/no question that offers the pinned engine, or None when we cannot download one."""
    if not runtime_install.downloads_allowed():
        return None
    pin = pinned_engine_tag()
    if clef_engine_status(pin) != "ok":
        return None
    shown = tag or "the one installed now"
    return (
        f"Download llama.cpp {pin} from GitHub (the release this game is tested with) "
        f"so Clef can run? The engine here ({shown}) is too old. "
        "The Clef file is a separate download, and it still asks before it starts."
    )


def bundled_engine_update_message(tag: Optional[str]) -> str:
    """What to say when this copy of the game cannot download a newer engine."""
    pin = pinned_engine_tag() or f"b{CLEF_TEXT_MIN_BUILD}"
    shown = tag or "this copy"
    return (
        f"This copy of the game does not download a replacement llama.cpp. "
        f"The engine it would use ({shown}) cannot load Clef. "
        f"Update the game to one that ships llama.cpp {pin} or newer. "
        "Nothing was downloaded."
    )


def install_pinned_clef_engine(ui: UI, specs: Any) -> Optional[str]:
    """Download the pinned llama.cpp release. Returns its tag, or None if that failed."""
    pin = pinned_engine_tag()
    if specs is None or clef_engine_status(pin) != "ok" or pin is None:
        return None
    try:
        exe, _variant = runtime_install.install_tagged_release(ui, specs, pin)
    except Exception as exc:
        ui.warn(escape_message(exc))
        return None
    info = runtime_install.install_info(exe) or {}
    tag = info.get("tag")
    if isinstance(tag, str) and tag.strip():
        return tag.strip()
    return pin


def escape_message(exc: BaseException) -> str:
    from .ui import escape

    return escape(f"The newer engine wasn't installed ({exc}).")


def pinned_engine_tag() -> Optional[str]:
    """The tag in ``packaging/llama_cpp_tag.txt``, or None if this install has no pin file.

    A wheel or a built game may not ship that file next to the package. Missing
    means unknown, and unknown refuses the download.
    """
    path = Path(__file__).resolve().parents[2] / "packaging" / "llama_cpp_tag.txt"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#") and " " not in line:
            return line
    return None


def resolve_engine_tag() -> Optional[str]:
    """Newest llama.cpp tag we can see: a built game, an installed runtime, then the pin.

    The newest build number wins, so an installed b11371 is used even when the
    repo pin is still older. A tag with no digits does not beat a real one.
    """
    tags: list[str] = []
    try:
        built = distribution.load().llama_cpp_tag
        if isinstance(built, str) and built.strip():
            tags.append(built.strip())
    except Exception:
        pass
    try:
        for _exe, tag, _variant in runtime_install.installed_runtimes():
            if tag and str(tag).strip():
                tags.append(str(tag).strip())
    except Exception:
        pass
    pinned = pinned_engine_tag()
    if pinned:
        tags.append(pinned)
    if not tags:
        return None
    return max(tags, key=lambda tag: (_tag_number(tag) >= 0, _tag_number(tag)))


def launch_local_clef(
    ui: UI,
    entry: ModelEntry,
    fit: FitResult,
    *,
    engine_tag: Optional[str],
    launcher: Optional[Launcher] = None,
) -> Any:
    """Start a local Clef referee, or raise :class:`LocalClefUnavailable`.

    ``launcher`` replaces the real process (tests). It is not called when the
    engine is too old or unknown, and nothing is downloaded in that case.
    """
    status = clef_engine_status(engine_tag)
    if status != "ok":
        raise LocalClefUnavailable(engine_block_message(engine_tag, status))
    if launcher is not None:
        return launcher(entry, fit)
    return _start_process(ui, entry, fit)


def _start_process(ui: UI, entry: ModelEntry, fit: FitResult) -> LocalClefReferee:
    exe = _newest_clef_server()
    if exe is None:
        raise LocalClefUnavailable(
            "llama.cpp b11371 or newer can load Clef text, but I couldn't find a llama-server that new "
            "on this computer. Nothing was downloaded."
        )
    quant = fit.quant or entry.quant
    named = _entry_for_quant(entry, quant, fit.download_gb)
    try:
        model_path = download.download_gguf(named, ui, quant=quant)
    except Exception as exc:
        raise LocalClefUnavailable(
            f"Clef wasn't downloaded ({exc}). Nothing is running."
        ) from exc
    port = find_free_port()
    n_ctx = int(fit.context_tokens or entry.context_tokens or 4096)
    api_key = "clef_local_" + secrets.token_hex(16)
    args = build_server_args(
        exe, model_path, port=port, n_ctx=n_ctx, cpu_only=fit.placement == "cpu",
    )
    log_dir = runtime_dir() / "logs"
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_handle = (log_dir / "clef-server.log").open("ab")
    except OSError as exc:
        raise LocalClefUnavailable(f"Couldn't open a log file for the Clef server ({exc}).") from exc
    try:
        proc = subprocess.Popen(
            args,
            env=server_env(exe, api_key=api_key),
            cwd=str(Path(exe).resolve().parent),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
        )
    except OSError as exc:
        log_handle.close()
        raise LocalClefUnavailable(f"The llama.cpp engine didn't start ({exc}).") from exc

    def stop() -> None:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_handle.close()

    atexit.register(stop)
    if not _wait_healthy(proc, port, stop):
        raise LocalClefUnavailable(
            "Clef's engine started but never became ready. The log is runtime/logs/clef-server.log. "
            "The server was stopped."
        )
    client = JevClient(api_key, base_url=f"http://127.0.0.1:{port}", model=entry.key, max_retries=0)
    name = entry.display_name
    return LocalClefReferee(client, stop, f"Clef ({name})")


def _entry_for_quant(entry: ModelEntry, quant: str, download_gb: Optional[float]) -> ModelEntry:
    from dataclasses import replace

    return replace(
        entry,
        quant=quant,
        file_size_gb=download_gb or entry.file_size_gb,
        gguf_files=(clef_gguf_name(entry, quant),),
    )


def clef_server_installed() -> bool:
    """True when a llama-server new enough for Clef text is already on disk."""
    return _newest_clef_server() is not None


def _newest_clef_server() -> Optional[Path]:
    try:
        found = runtime_install.installed_runtimes()
    except Exception:
        return None
    ok = [(exe, tag) for exe, tag, _variant in found if clef_engine_status(tag) == "ok"]
    if not ok:
        return None
    exe, _tag = max(ok, key=lambda item: _tag_number(item[1]))
    return exe


def _wait_healthy(proc: subprocess.Popen, port: int, stop: Callable[[], None], *, attempts: int = 60) -> bool:
    """Poll ``GET /health`` on 127.0.0.1. Give up if the process exits or never answers."""
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(attempts):
        if proc.poll() is not None:
            stop()
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.0) as resp:  # loopback only
                if getattr(resp, "status", 200) == 200:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    stop()
    return False


def _tag_number(tag: Optional[str]) -> int:
    return runtime_install._tag_number(tag or "")


def consent_lines(entry: ModelEntry) -> list[str]:
    """License, hardware, warranty and output lines shown before a Clef download."""
    page = entry.license_url or f"https://huggingface.co/{entry.hf_repo}"
    return [
        f"{entry.display_name} is {entry.license} licensed. Read it: {page}. "
        "The weights are not part of this game; confirming downloads them from Hugging Face.",
        *LOCAL_RUN_NOTICES,
        "Clef does not write the story. It only scores the referee's questions. "
        "Any tokens/s number is this game's chat-model estimate, not a measured Clef latency. "
        "The download is from Hugging Face. Scoring stays on 127.0.0.1. Nothing is sent to Cloudflare.",
    ]
