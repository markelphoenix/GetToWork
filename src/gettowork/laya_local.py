"""Install and run Laya on this computer, when the machine can spare the room.

Hosted Laya (Laya Studio) needs an API key. The same open weights can run here
instead: a private Python environment, the English checkpoint from Hugging Face,
and ``laya-serve`` on ``127.0.0.1``. The game only offers this when enough memory
and disk are free *after* the story model is already loaded, and a Python 3.10+
interpreter can build the environment.

Nothing in this module phones a referee service. The one-time install does
download from PyPI and Hugging Face.
"""

from __future__ import annotations

import atexit
import os
import platform
import secrets
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from .config import child_env, runtime_dir
from .jev import LAYA_LOCAL_OPTION, JevClient
from .types import SystemSpecs

# Pinned so a later laya release can't change the server out from under a saved install.
LAYA_PACKAGE = "laya[serve]==0.3.22"
# One English checkpoint (ModernBERT-large, ~0.8 GB of weights) plus the process.
# The install uses the normal PyTorch build, so the weights sit in system memory
# even when a graphics card is present (Apple silicon is the exception: that
# build can use the graphics chip, which shares this same memory).
MIN_FREE_RAM_GB = 3.0
# CPU PyTorch + Laya + the checkpoint.
MIN_FREE_DISK_GB = 4.0
HEALTH_TIMEOUT_S = 15 * 60  # the first start also downloads the checkpoint
HEALTH_INTERVAL_S = 0.5

Runner = Callable[[list[str], dict[str, str]], tuple[int, str]]
HealthCheck = Callable[[str], bool]


class LocalLayaError(Exception):
    """The local install or server didn't come up. ``message`` is safe to show."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class LocalLayaFit:
    """Whether this machine can run Laya beside the story model."""

    ok: bool
    device: str = ""  # "cpu" or "mps"
    threads: int = 1
    reason: str = ""  # why not, in plain language, when ``ok`` is False


def assess_specs(specs: SystemSpecs) -> LocalLayaFit:
    """Memory, disk and a usable device. Does not look for Python."""
    threads = _threads(specs)
    if 0 <= specs.disk_free_gb < MIN_FREE_DISK_GB:
        return LocalLayaFit(
            False, reason=(
                f"only about {specs.disk_free_gb:.0f} GB of disk is free, and Laya on this computer "
                f"needs about {MIN_FREE_DISK_GB:.0f} GB the first time (the program plus its English checkpoint)"
            ),
        )
    ram = specs.ram_available_gb
    if ram < MIN_FREE_RAM_GB:
        return LocalLayaFit(
            False, reason=(
                f"only about {ram:.0f} GB of memory is free, and Laya on this computer needs about "
                f"{MIN_FREE_RAM_GB:.0f} GB on top of the story model"
            ),
        )
    if specs.unified_memory and specs.os_name == "Darwin":
        return LocalLayaFit(True, device="mps", threads=threads)
    return LocalLayaFit(True, device="cpu", threads=threads)


def _threads(specs: SystemSpecs) -> int:
    cores = specs.cpu_cores_physical or specs.cpu_cores_logical or 4
    return max(1, min(int(cores), 8))


def venv_python(root: Path) -> Path:
    """The interpreter inside a virtual environment at ``root``."""
    if platform.system() == "Windows":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def laya_home() -> Path:
    """Where the private environment, the log and the Hugging Face cache live."""
    return runtime_dir() / "laya"


class LocalLaya:
    """One local ``laya-serve`` for this game. ``close()`` stops it."""

    def __init__(
        self,
        *,
        python_finder: Optional[Callable[[], Optional[str]]] = None,
        runner: Optional[Runner] = None,
        popen: Optional[Callable[..., object]] = None,
        health: Optional[HealthCheck] = None,
        sleep: Optional[Callable[[float], None]] = None,
        clock: Optional[Callable[[], float]] = None,
        pick_port: Optional[Callable[[], int]] = None,
    ) -> None:
        self._python_finder = python_finder or find_python
        self._runner = runner or _subprocess_runner
        self._popen = popen or subprocess.Popen
        self._health = health or _http_health
        self._sleep = sleep or time.sleep
        self._clock = clock or time.monotonic
        self._pick_port = pick_port or _free_port
        self._proc: Optional[object] = None
        self._log = None
        self._atexit = False

    def fit(self, specs: Optional[SystemSpecs]) -> LocalLayaFit:
        """Specs plus a Python 3.10+ the game can build an environment with."""
        if specs is None:
            return LocalLayaFit(False, reason="")
        base = assess_specs(specs)
        if not base.ok:
            return base
        if self.python_executable() is None:
            return LocalLayaFit(
                False,
                reason="Laya on this computer needs Python 3.10 or newer, and I couldn't find one",
            )
        return base

    def python_executable(self) -> Optional[str]:
        return self._python_finder()

    def installed(self) -> bool:
        """True when a previous run already finished ``pip install``."""
        return (laya_home() / "installed.json").is_file() and venv_python(laya_home() / "venv").is_file()

    def start(self, ui: object, *, device: str, threads: int) -> JevClient:
        """Install if needed, start the server, and return a client pointed at it.

        Raises :class:`LocalLayaError` with a message safe to show. The API key
        exists only so the local server isn't open to other programs; it is not
        a Laya Studio key.
        """
        home = laya_home()
        home.mkdir(parents=True, exist_ok=True)
        py = self._ensure_env(ui, home)
        port = self._pick_port()
        key = "laya_local_" + secrets.token_hex(16)
        env = child_env()
        env.update({
            "LAYA_HOST": "127.0.0.1",
            "LAYA_PORT": str(port),
            "LAYA_DEVICE": device or "cpu",
            "LAYA_MODELS": "english",
            "LAYA_PRELOAD": "1",
            "LAYA_MAX_LOADED": "1",
            "LAYA_THREADS": str(max(1, threads)),
            "LAYA_API_KEY": key,
            "LAYA_LOG_LEVEL": "warning",
            "HF_HOME": str(home / "hf"),
        })
        log_path = home / "laya-serve.log"
        log = open(log_path, "ab")  # noqa: SIM115 - closed in close() / on failure
        self._log = log
        try:
            proc = self._popen(
                [str(py), "-m", "laya.serve"],
                env=env,
                cwd=str(home),
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=(os.name != "nt"),
            )
        except OSError as exc:
            log.close()
            self._log = None
            raise LocalLayaError(f"Couldn't start Laya ({exc}).") from exc
        self._proc = proc
        if not self._atexit:
            atexit.register(self.close)
            self._atexit = True
        url = f"http://127.0.0.1:{port}"
        self._wait_until_ready(ui, url, proc, log_path)
        client = JevClient(key, option=LAYA_LOCAL_OPTION, base_url=url, model="english")
        client.stop = self.close  # type: ignore[attr-defined]
        return client

    def close(self) -> None:
        """Stop the server. Safe to call more than once."""
        proc = self._proc
        self._proc = None
        if proc is not None and getattr(proc, "poll", lambda: 0)() is None:
            _shutdown(proc)
        log = self._log
        self._log = None
        if log is not None:
            try:
                log.close()
            except OSError:
                pass

    # -- install --------------------------------------------------------------------------------

    def _ensure_env(self, ui: object, home: Path) -> Path:
        py = venv_python(home / "venv")
        if not py.is_file():
            base = self.python_executable()
            if not base:
                raise LocalLayaError(
                    "Laya on this computer needs Python 3.10 or newer, and I couldn't find one."
                )
            self._run(ui, "Setting up a private Python environment for Laya...", [base, "-m", "venv", str(home / "venv")])
            if not py.is_file():
                raise LocalLayaError(
                    "Python couldn't create an environment for Laya. On Linux that usually means the "
                    "python3-venv package isn't installed."
                )
        if not self.installed():
            self._run(
                ui,
                "Installing Laya (the first time this downloads a few GB)...",
                [str(py), "-m", "pip", "install", "--disable-pip-version-check", LAYA_PACKAGE],
            )
            (home / "installed.json").write_text('{"laya": "0.3.22"}\n', encoding="utf-8")
        return py

    def _run(self, ui: object, label: str, args: list[str]) -> None:
        with ui.status(label):  # type: ignore[attr-defined]
            code, output = self._runner(args, child_env())
        if code != 0:
            tail = (output or "").strip().splitlines()[-8:]
            detail = " ".join(tail) if tail else f"exit code {code}"
            raise LocalLayaError(f"Installing Laya failed. {detail}")

    def _wait_until_ready(self, ui: object, url: str, proc: object, log_path: Path) -> None:
        health = url + "/health"
        deadline = self._clock() + HEALTH_TIMEOUT_S
        with ui.status("Starting Laya on this computer (the first time also downloads its checkpoint)...") as _update:  # type: ignore[attr-defined]
            while self._clock() < deadline:
                if getattr(proc, "poll", lambda: None)() is not None:
                    raise LocalLayaError("Laya stopped while starting. " + _log_tail(log_path))
                try:
                    if self._health(health):
                        return
                except Exception:
                    pass
                self._sleep(HEALTH_INTERVAL_S)
        raise LocalLayaError(
            "Laya didn't finish starting. The first run downloads its checkpoint; check your connection. "
            + _log_tail(log_path)
        )


def find_python() -> Optional[str]:
    """A Python 3.10+ executable, or None.

    A copy run from source uses its own interpreter. A built game looks on
    ``PATH``, because the frozen program can't ``pip install`` into itself.
    """
    candidates: list[str] = []
    if not getattr(sys, "frozen", False):
        candidates.append(sys.executable)
    for name in ("python3", "python"):
        found = _which(name)
        if found and found not in candidates:
            candidates.append(found)
    for path in candidates:
        if _python_ok(path):
            return path
    return None


def _which(name: str) -> Optional[str]:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for folder in paths:
        candidate = Path(folder) / (name + (".exe" if os.name == "nt" else ""))
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _python_ok(path: str) -> bool:
    try:
        out = subprocess.run(
            [path, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True, text=True, timeout=15, env=child_env(),
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    if out.returncode != 0:
        return False
    try:
        major, minor = (int(part) for part in (out.stdout or "").strip().split(".", 1))
    except ValueError:
        return False
    return (major, minor) >= (3, 10)


def _subprocess_runner(args: list[str], env: dict[str, str]) -> tuple[int, str]:
    try:
        done = subprocess.run(args, capture_output=True, text=True, env=env)
    except OSError as exc:
        return 1, str(exc)
    text = ((done.stdout or "") + "\n" + (done.stderr or "")).strip()
    return done.returncode, text


def _http_health(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            body = response.read(200)
            return response.status == 200 and b'"ok"' in body
    except (OSError, urllib.error.URLError, ValueError):
        return False


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _shutdown(proc: object) -> None:
    custom = getattr(proc, "stop", None)
    if callable(custom):
        custom()
        return
    if os.name == "nt":
        proc.terminate()  # type: ignore[attr-defined]
    else:
        try:
            os.killpg(proc.pid, signal.SIGTERM)  # type: ignore[attr-defined]
        except (OSError, ProcessLookupError, AttributeError):
            proc.terminate()  # type: ignore[attr-defined]
    try:
        proc.wait(timeout=5)  # type: ignore[attr-defined]
    except Exception:
        try:
            proc.kill()  # type: ignore[attr-defined]
        except Exception:
            pass


def _log_tail(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "Last notes: " + " | ".join(lines[-4:])
