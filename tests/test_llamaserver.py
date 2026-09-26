"""Tests for gettowork.backends.llamaserver.

No real processes, network or models: a fake `popen` hands back scripted
process objects (and writes a fake log), and a fake HTTP layer plays the part
of llama-server's /health and /v1/chat/completions endpoints.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest
from rich.console import Console

from gettowork import distribution, runtime_install
from gettowork.backends import llamaserver as ls
from gettowork.backends.base import RETRY_WITHOUT_THINKING_NOTICE, BackendError
from gettowork.backends.llamaserver import (
    CUSTOM_VARIANT,
    LlamaServerBackend,
    build_server_args,
    classify_server_log,
    find_free_port,
    server_env,
)
from gettowork.runtime_install import CPU, CUDA12, VULKAN, RuntimeInstallError, plan_variants
from gettowork.types import GPUInfo, ModelEntry, SystemSpecs
from gettowork.ui import UI, UserQuit

ENTRY = ModelEntry(
    key="qwen3-4b",
    display_name="Qwen3 4B",
    family="Qwen3",
    params_b=4.0,
    active_params_b=None,
    license="Apache-2.0",
    license_url="https://huggingface.co/Qwen/Qwen3-4B",
    hf_repo="unsloth/Qwen3-4B-GGUF",
    quant="Q4_K_M",
    file_size_gb=2.5,
    ollama_ref="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M",
    reasoning=True,
    blurb="A tiny storyteller.",
)
NVIDIA = GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0)

# The real host helpers, before the autouse fixture stands fakes in for them.
REAL_POSIX_LOCKS_AVAILABLE = ls._posix_locks_available
REAL_LOCK_EXCLUSIVELY = ls._lock_exclusively
try:
    import fcntl as _host_fcntl
except ImportError:  # Windows
    _host_fcntl = None
needs_real_flock = pytest.mark.skipif(
    _host_fcntl is None,
    reason="exercises the real flock() call, which only exists on POSIX (Linux, macOS); Windows has no fcntl "
           "module, and there each game uses its own log (see test_windows_games_each_use_their_own_log)",
)

CUDA_CRASH_LOG = """\
build: 7000 (abcdef0) with cc (Ubuntu 13.3.0) for x86_64-linux-gnu
ggml_cuda_init: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 3060, compute capability 8.6, VMM: yes
CUDA error: the provided PTX was compiled with an unsupported toolchain.
  current device: 0, in function ggml_cuda_compute_forward at ggml-cuda.cu:2400
"""


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def make_specs(os_name="Linux", arch="x86_64", gpus=(), flags=()) -> SystemSpecs:
    return SystemSpecs(
        os_name=os_name, os_version="test", arch=arch, cpu_name="Test CPU",
        cpu_cores_physical=4, cpu_cores_logical=8, ram_total_gb=16.0, ram_available_gb=12.0,
        disk_free_gb=100.0, gpus=list(gpus), cpu_flags=list(flags),
    )


def make_ui() -> UI:
    return UI(console=Console(file=io.StringIO(), width=200), input_fn=lambda prompt: "")


def output(ui: UI) -> str:
    return ui.console.file.getvalue()


class FakeProcess:
    """Stands in for subprocess.Popen's return value."""

    def __init__(self, *, dies_with=None, dies_after_polls=0, ignores_terminate=False):
        self.pid = 4242
        self.returncode = None
        self._dies_with = dies_with
        self._dies_after = dies_after_polls
        self._polls = 0
        self._ignores_terminate = ignores_terminate
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_timeouts = []

    def poll(self):
        if self.returncode is None and self._dies_with is not None and self._polls >= self._dies_after:
            self.returncode = self._dies_with
        self._polls += 1
        return self.returncode

    def crash(self, code=1):
        self.returncode = code

    def terminate(self):
        self.terminate_calls += 1
        if not self._ignores_terminate:
            self.returncode = -15

    def wait(self, timeout=None):
        self.wait_timeouts.append(timeout)
        if self.returncode is None:
            raise subprocess.TimeoutExpired("llama-server", timeout)
        return self.returncode

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9


class FakePopen:
    """Each call pops a scripted (log_text, process) pair and writes the log."""

    def __init__(self, *scripts):
        self.scripts = list(scripts)
        self.calls: list[tuple[list[str], dict]] = []
        self.processes: list[FakeProcess] = []

    def __call__(self, args, **kwargs):
        assert isinstance(args, list) and "shell" not in kwargs  # list args, never a shell
        self.calls.append((list(args), kwargs))
        log, proc = self.scripts.pop(0) if self.scripts else ("", FakeProcess())
        fh = kwargs["stdout"]
        fh.write(log.encode("utf-8"))
        fh.flush()
        self.processes.append(proc)
        return proc

    @property
    def args(self):
        return [c[0] for c in self.calls]


class FakeResponse:
    def __init__(self, status, body: bytes):
        self.status = status
        self.headers = {}
        self._body = body

    def read(self, n=-1):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass


def chat_reply(content="Hello there!", reasoning=None, **extra):
    msg = {"role": "assistant", "content": content}
    if reasoning is not None:
        msg["reasoning_content"] = reasoning
    data = {"choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
            "usage": {"completion_tokens": 20, "prompt_tokens": 10, "total_tokens": 30}}
    data.update(extra)
    return data


class FakeServerHttp:
    """Plays llama-server: /health answers from a list (last one repeats);
    chat answers are consumed in order (dict = 200 JSON, (status, bytes), or an exception)."""

    def __init__(self, health=(200,), chat=(), on_chat=None):
        self.health = list(health)
        self.chat = list(chat)
        self.on_chat = on_chat
        self.calls: list[dict] = []
        self.chat_bodies: list[dict] = []

    def request(self, method, url, *, headers=None, body=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "timeout": timeout, "headers": dict(headers or {})})
        if url.endswith("/health"):
            item = self.health.pop(0) if len(self.health) > 1 else self.health[0]
            if isinstance(item, BaseException):
                raise item
            if item == 200:
                return FakeResponse(200, b'{"status": "ok"}')
            return FakeResponse(item, b'{"error": {"code": 503, "message": "Loading model", "type": "unavailable_error"}}')
        if url.endswith("/v1/chat/completions"):
            self.chat_bodies.append(json.loads(body.decode("utf-8")))
            if self.on_chat:
                self.on_chat()
            item = self.chat.pop(0)
            if isinstance(item, BaseException):
                raise item
            if isinstance(item, tuple):
                return FakeResponse(*item)
            return FakeResponse(200, json.dumps(item).encode("utf-8"))
        raise AssertionError(f"unexpected URL {url}")


class FakeClock:
    def __init__(self):
        self.t = 1000.0
        self.sleeps: list[float] = []

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.t += seconds


class FakeInstaller:
    """Like ensure_llama_server: returns a fake exe per variant; can be told to fail."""

    def __init__(self, root: Path, fail=(), marker=False, update_to=None):
        self.root = root
        self.fail = set(fail)
        self.calls: list[str] = []
        self.marker = marker  # write an install.json, like a real install
        self.update_to = update_to  # tag a newer build gets when update=True (None = "already newest")

    def __call__(self, ui, specs, *, variant=None, update=False):
        v = variant or plan_variants(specs)[0]
        self.calls.append(v.name + (" (update)" if update else ""))
        if v.name in self.fail or (update and "update" in self.fail):
            raise RuntimeInstallError(f"No {v.name} build today.")
        tag = self.update_to if update and self.update_to else "b7000"
        folder = f"{tag}-{v.name}" if update and self.update_to else v.name
        exe = self.root / "engines" / folder / "llama-server"
        exe.parent.mkdir(parents=True, exist_ok=True)
        exe.write_bytes(b"fake")
        if self.marker:
            (exe.parent / "install.json").write_text(json.dumps({"tag": tag, "variant": v.name, "exe": "llama-server"}))
        return exe, v


class FakeDownloader:
    def __init__(self, root: Path, error: BaseException | None = None):
        self.root = root
        self.error = error
        self.calls: list[tuple] = []

    def __call__(self, entry, ui, dest_dir=None, *, quant=None, hf_api=None, hf_download=None):
        self.calls.append((entry.key, quant))
        if self.error:
            raise self.error
        path = self.root / "models" / f"{entry.key}-{quant or entry.quant}.gguf"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"GGUF")
        return path


class FakeFileLocks:
    """flock() as a Linux/macOS host does it - in-process, so it behaves the same on every OS.

    A file can be locked through one open handle at a time, until that handle
    is closed; a second ``open()`` of the same file can't take the lock (flock
    locks belong to the open file, not the process).
    """

    def __init__(self):
        self._holders: dict[str, object] = {}

    @staticmethod
    def _key(path) -> str:
        return os.path.normcase(os.path.realpath(path))

    def lock(self, fh) -> bool:
        key = self._key(fh.name)
        holder = self._holders.get(key)
        if holder is not None and holder is not fh and not holder.closed:
            return False
        self._holders[key] = fh
        return True

    def hold(self, path):
        """Another copy of the game holding `path` locked; close the returned file to let go."""
        fh = open(path, "ab")
        assert self.lock(fh)
        return fh


def fake_file_sizes(monkeypatch, sizes: dict) -> None:
    """Make the backend see these file sizes (e.g. a 40 GB model) without writing them.

    Never create big files in a test, not even with truncate(): that is only
    "sparse" (free) on some file systems - NTFS really allocates it and fills
    the disk. Files not listed keep their real size.
    """
    real = ls._file_size
    wanted = {os.path.normcase(os.path.realpath(p)): int(n) for p, n in sizes.items()}

    def size(path):
        key = os.path.normcase(os.path.realpath(path))
        return wanted[key] if key in wanted else real(path)

    monkeypatch.setattr(ls, "_file_size", size)


@pytest.fixture(autouse=True)
def posix_file_locks(monkeypatch):
    """The log-file locking a Linux host has, whatever OS runs the tests.

    The backend decides between one shared ``llama-server.log`` (guarded by
    flock) and a log per game by asking whether ``fcntl`` exists - a host fact
    that faking ``platform.system()`` doesn't change - so set it explicitly.
    """
    locks = FakeFileLocks()
    monkeypatch.setattr(ls, "_posix_locks_available", lambda: True)
    monkeypatch.setattr(ls, "_lock_exclusively", locks.lock)
    return locks


@pytest.fixture(autouse=True)
def linux_host(monkeypatch, tmp_path, posix_file_locks):
    """Pretend to be Linux, keep data dirs in tmp, and never touch real atexit."""
    monkeypatch.setattr(ls.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runtime_install, "_system_has_vulkan_loader", lambda: False)
    monkeypatch.setattr(runtime_install, "_glibc_version", lambda: (2, 39))
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path / "home"))
    registered: list = []
    monkeypatch.setattr(ls.atexit, "register", lambda fn: registered.append(fn))
    monkeypatch.setattr(ls.atexit, "unregister", lambda fn: registered.remove(fn) if fn in registered else None)
    monkeypatch.setattr(ls, "find_free_port", lambda: 50123)
    return registered


@pytest.fixture(autouse=True)
def developer_copy(monkeypatch):
    """Every test starts as a developer copy (downloads on, no built-in engine), whatever the shell has set."""
    for var in ("GETTOWORK_DISTRIBUTION", "GETTOWORK_ENGINE_DIR", "GETTOWORK_ALLOW_ENGINE_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(distribution, "_cache", distribution.Distribution())


def make_backend(tmp_path, *, popen=None, http=None, specs=None, installer=None, downloader=None,
                 clock=None, entry=ENTRY, **kwargs):
    clock = clock or FakeClock()
    backend = LlamaServerBackend(
        entry,
        specs=specs or make_specs(),
        http=http or FakeServerHttp(),
        popen=popen or FakePopen(),
        installer=installer or FakeInstaller(tmp_path),
        downloader=downloader or FakeDownloader(tmp_path),
        sleep=clock.sleep,
        clock=clock,
        log_dir=tmp_path / "logs",
        **kwargs,
    )
    return backend


def started(tmp_path, chat=(), **kwargs):
    http = FakeServerHttp(health=(200,), chat=chat, on_chat=kwargs.pop("on_chat", None))
    popen = FakePopen()
    backend = make_backend(tmp_path, popen=popen, http=http, **kwargs)
    backend.prepare(make_ui())
    return backend, http, popen


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_build_server_args():
    full = build_server_args("/e/llama-server", "/m/model.gguf", port=8081, n_ctx=4096)
    assert full == ["/e/llama-server", "-m", "/m/model.gguf", "--host", "127.0.0.1", "--port", "8081",
                    "-c", "4096", "--reasoning-format", "deepseek", "--no-webui", "-np", "1"]
    assert build_server_args("s", "m", port=1, n_ctx=2048, minimal=True) == [
        "s", "-m", "m", "--host", "127.0.0.1", "--port", "1", "-c", "2048"]
    # CPU mode: "--device none" keeps the GPU out entirely (-ngl 0 alone still borrows it for prompts).
    assert build_server_args("s", "m", port=1, n_ctx=2048, cpu_only=True)[-4:] == ["--device", "none", "-ngl", "0"]
    # Minimal mode (for old builds) keeps only the basics.
    minimal_cpu = build_server_args("s", "m", port=1, n_ctx=2048, cpu_only=True, minimal=True)
    assert minimal_cpu[-2:] == ["-ngl", "0"] and "--device" not in minimal_cpu


def test_find_free_port_is_a_real_bindable_port():
    import socket

    port = find_free_port()  # the real function (the autouse fixture only patches the module attribute)
    assert 0 < port < 65536
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_server_env_per_platform(monkeypatch, tmp_path):
    exe = tmp_path / "rt" / "llama-server"
    env = server_env(exe, {"LD_LIBRARY_PATH": "/usr/lib/x", "PATH": "/bin"})
    assert env["LD_LIBRARY_PATH"].split(ls.os.pathsep) == [str(exe.parent), "/usr/lib/x"]
    assert server_env(exe, {})["LD_LIBRARY_PATH"] == str(exe.parent)

    monkeypatch.setattr(ls.platform, "system", lambda: "Windows")
    env = server_env(exe, {"PATH": "C:\\Windows"})
    assert env["PATH"].startswith(str(exe.parent) + ls.os.pathsep)
    assert "LD_LIBRARY_PATH" not in env

    monkeypatch.setattr(ls.platform, "system", lambda: "Darwin")
    assert server_env(exe, {"PATH": "/bin"}) == {"PATH": "/bin"}


class _FakeKernel32:
    """ctypes.windll.kernel32 as far as windows_system_dll_search uses it."""

    def __init__(self, dll_directory: str = "", fail: bool = False) -> None:
        self.dll_directory, self.fail, self.calls = dll_directory, fail, []

    def GetDllDirectoryW(self, size, buffer):  # noqa: N802 - the Windows name
        if self.fail:
            raise OSError("no such function")
        buffer.value = self.dll_directory
        return len(self.dll_directory)

    def SetDllDirectoryW(self, value):  # noqa: N802
        self.calls.append(value)
        return 1


def _pretend_built_game_on_windows(monkeypatch, kernel32) -> None:
    import ctypes

    monkeypatch.setattr(ctypes, "windll", types.SimpleNamespace(kernel32=kernel32), raising=False)
    monkeypatch.setattr(ls.platform, "system", lambda: "Windows")
    monkeypatch.setattr(sys, "frozen", True, raising=False)


def test_a_built_game_on_windows_starts_the_engine_with_the_normal_dll_search(monkeypatch):
    # PyInstaller's SetDllDirectory(_internal) would be inherited by llama-server.exe.
    kernel32 = _FakeKernel32(r"C:\Games\GetToWork\_internal")
    _pretend_built_game_on_windows(monkeypatch, kernel32)
    with ls.windows_system_dll_search():
        kernel32.calls.append("the engine starts")
    assert kernel32.calls == [None, "the engine starts", r"C:\Games\GetToWork\_internal"]

    kernel32.calls.clear()
    with pytest.raises(OSError):
        with ls.windows_system_dll_search():
            raise OSError("the engine couldn't start")
    assert kernel32.calls == [None, r"C:\Games\GetToWork\_internal"]  # put back even then


def test_the_dll_search_is_left_alone_when_there_is_nothing_to_undo(monkeypatch):
    kernel32 = _FakeKernel32("")  # no DLL directory set
    _pretend_built_game_on_windows(monkeypatch, kernel32)
    with ls.windows_system_dll_search():
        pass
    assert kernel32.calls == []

    broken = _FakeKernel32(fail=True)
    _pretend_built_game_on_windows(monkeypatch, broken)
    with ls.windows_system_dll_search():  # never raises
        pass
    assert broken.calls == []


@pytest.mark.parametrize("system, frozen", [("Windows", False), ("Linux", True), ("Darwin", True)])
def test_the_dll_search_is_only_touched_in_a_built_game_on_windows(monkeypatch, system, frozen):
    kernel32 = _FakeKernel32(r"C:\Games\GetToWork\_internal")
    _pretend_built_game_on_windows(monkeypatch, kernel32)
    monkeypatch.setattr(ls.platform, "system", lambda: system)
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    with ls.windows_system_dll_search():
        pass
    assert kernel32.calls == []


def test_the_engine_is_started_inside_the_normal_dll_search(monkeypatch, tmp_path):
    kernel32 = _FakeKernel32(r"C:\Games\GetToWork\_internal")
    events = []
    kernel32.SetDllDirectoryW = lambda value: events.append(("dll dir", value)) or 1  # type: ignore[method-assign]
    backend = make_backend(tmp_path, popen=lambda args, **kw: events.append(("start", args[0])) or FakeProcess())
    _pretend_built_game_on_windows(monkeypatch, kernel32)
    backend._launch(tmp_path / "llama-server", tmp_path / "m.gguf", cpu_only=False, minimal=False)
    assert [e[0] for e in events] == ["dll dir", "start", "dll dir"]
    assert events[0][1] is None and events[2][1] == r"C:\Games\GetToWork\_internal"
    backend._close_log()


@pytest.mark.parametrize(
    "log, code, expected",
    [
        ("error: invalid argument: --no-webui", 1, "bad_args"),
        ("error: unknown argument: --reasoning-format", 1, "bad_args"),
        ("couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 8080", 1, "port"),
        ("./llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found", 1, "glibc"),
        (CUDA_CRASH_LOG, 1, "gpu"),
        ("ggml_cuda_init: failed to initialize CUDA: CUDA driver version is insufficient", 1, "gpu"),
        ("./llama-server: error while loading shared libraries: libcudart.so.12: cannot open shared object file", 127, "gpu"),
        ("ggml_vulkan: Device memory allocation of size 4096 failed.", 1, "gpu"),
        ("vk::Device::allocateMemory: ErrorOutOfDeviceMemory", 1, "gpu"),
        ("ggml_metal_init: error: failed to create command queue", 1, "gpu"),
        ("hipErrorNoBinaryForGpu: Unable to find code object for all current devices!", 1, "gpu"),
        ("./llama-server: error while loading shared libraries: libgomp.so.1: cannot open shared object file", 127, "missing_library"),
        ("", 0xC0000135, "missing_library"),
        ("", -1073741515, "missing_library"),
        ("terminate called after throwing an instance of 'std::bad_alloc'", 134, "memory"),
        ("", -9, "memory"),
        ("llama_model_load: error loading model: unknown model architecture: 'qwen9'", 1, "model_unsupported"),
        ("gguf_init_from_file: invalid magic characters 'abcd'", 1, "model"),
        # strerror(EINVAL) and CUDA's "invalid argument" are not rejected command-line flags.
        ("llama_model_load: error loading model: mmap failed: Invalid argument\nfailed to load model", 1, "model"),
        ("CUDA error: invalid argument\n  current device: 0, in function ggml_backend_cuda_buffer_set_tensor", 1, "gpu"),
        ("error: unrecognized arguments: --device", 2, "bad_args"),
        ("Illegal instruction (core dumped)", 132, "cpu_unsupported"),
        ("", -4, "cpu_unsupported"),
        ("something odd happened", 1, "unknown"),
        ("", None, "unknown"),
    ],
)
def test_classify_server_log(log, code, expected):
    assert classify_server_log(log, code) == expected


def test_healthy_cuda_lines_alone_are_not_a_gpu_failure():
    log = "ggml_cuda_init: found 1 CUDA devices:\n  Device 0: NVIDIA GeForce RTX 3060, VMM: yes\nmain: model loaded\n"
    assert classify_server_log(log, 1) == "unknown"


# ---------------------------------------------------------------------------
# prepare(): install, download, start, wait
# ---------------------------------------------------------------------------


def test_prepare_happy_path(tmp_path):
    clock = FakeClock()
    http = FakeServerHttp(health=(ConnectionRefusedError(), 503, 503, 200))
    popen = FakePopen()
    installer, downloader = FakeInstaller(tmp_path), FakeDownloader(tmp_path)
    backend = make_backend(tmp_path, popen=popen, http=http, installer=installer, downloader=downloader,
                           clock=clock, quant="Q5_K_M")
    ui = make_ui()
    backend.prepare(ui)

    assert installer.calls == ["cpu"]  # no GPU in these specs
    assert downloader.calls == [("qwen3-4b", "Q5_K_M")]
    assert len(popen.calls) == 1
    args, kwargs = popen.calls[0]
    exe = (tmp_path / "engines" / "cpu" / "llama-server").resolve()
    model = (tmp_path / "models" / "qwen3-4b-Q5_K_M.gguf").resolve()
    assert args == build_server_args(exe, model, port=50123, n_ctx=4096, cpu_only=True)
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["stderr"] is subprocess.STDOUT
    assert Path(kwargs["stdout"].name) == tmp_path / "logs" / "llama-server.log"
    assert kwargs["cwd"] == str(exe.parent)
    assert kwargs["env"]["LD_LIBRARY_PATH"].startswith(str(exe.parent))
    assert "creationflags" not in kwargs
    assert clock.sleeps == [ls.HEALTH_POLL_S] * 3  # refused, 503, 503, then ok
    assert all(c["url"] == "http://127.0.0.1:50123/health" for c in http.calls)

    assert backend.port == 50123 and backend.variant is CPU and backend.cpu_only
    assert backend.server_exe == exe and backend.model_path == tmp_path / "models" / "qwen3-4b-Q5_K_M.gguf"
    assert backend.model_label == "Qwen3 4B (Q5_K_M)"
    assert "awake and ready" in output(ui)
    backend.close()


def test_windows_launch_hides_console_and_extends_path(tmp_path, monkeypatch):
    monkeypatch.setattr(ls.platform, "system", lambda: "Windows")
    backend, _http, popen = started(tmp_path)
    _args, kwargs = popen.calls[0]
    assert kwargs["creationflags"] == 0x08000000  # CREATE_NO_WINDOW
    assert kwargs["env"]["PATH"].startswith(str(Path(popen.args[0][0]).parent))
    backend.close()


def test_default_log_location_is_runtime_dir(tmp_path):
    backend = LlamaServerBackend(ENTRY, specs=make_specs(), http=FakeServerHttp(), popen=FakePopen(),
                                 installer=FakeInstaller(tmp_path), downloader=FakeDownloader(tmp_path),
                                 sleep=lambda s: None, clock=FakeClock())
    backend.prepare(make_ui())
    assert backend.log_path == tmp_path / "home" / "runtime" / "logs" / "llama-server.log"
    backend.close()


def test_gpu_crash_falls_back_to_cpu_build(tmp_path):
    specs = make_specs(gpus=[NVIDIA])  # Linux + NVIDIA, no Vulkan -> plan: cuda-12, cpu
    popen = FakePopen((CUDA_CRASH_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer)
    ui = make_ui()
    backend.prepare(ui)

    assert installer.calls == ["cuda-12", "cpu"]
    first, second = popen.args
    assert "cuda-12" in first[0] and "-ngl" not in first
    assert "cpu" in second[0] and second[-2:] == ["-ngl", "0"]
    assert backend.variant is CPU and backend.cpu_only
    assert "engines/cpu" in backend.server_exe.as_posix()
    text = output(ui)
    assert "graphics-driver hiccup" in text
    assert "CPU mode" in text
    assert popen.processes[0].returncode == 1  # the crashed one is gone
    backend.close()


def test_gpu_crashes_walk_the_whole_plan(tmp_path):
    specs = make_specs(gpus=[NVIDIA], flags=["vulkan"])  # plan: cuda-12, vulkan, cpu
    popen = FakePopen(
        (CUDA_CRASH_LOG, FakeProcess(dies_with=1)),
        ("ggml_vulkan: Device memory allocation of size 1 failed.", FakeProcess(dies_with=1)),
        ("", FakeProcess()),
    )
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer)
    backend.prepare(make_ui())
    assert installer.calls == ["cuda-12", "vulkan", "cpu"]
    assert backend.variant is CPU
    assert popen.args[1][-2:] != ["-ngl", "0"]  # Vulkan still tries the GPU
    assert popen.args[2][-2:] == ["-ngl", "0"]


def test_windows_missing_dll_on_gpu_build_falls_back(tmp_path):
    specs = make_specs(gpus=[NVIDIA])
    popen = FakePopen(("", FakeProcess(dies_with=0xC0000135)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=specs)
    backend.prepare(make_ui())
    assert backend.variant is CPU


def test_offline_fallback_runs_same_build_in_cpu_mode(tmp_path):
    specs = make_specs(gpus=[NVIDIA])
    popen = FakePopen((CUDA_CRASH_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path, fail={"cpu"})
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cuda-12", "cpu"]
    first, second = popen.args
    assert first[0] == second[0]  # same CUDA executable...
    assert second[-2:] == ["-ngl", "0"]  # ...with every layer kept off the GPU
    assert backend.cpu_only and backend.variant is CUDA12
    assert "No cpu build today." in output(ui)


def test_unknown_argument_retries_with_minimal_args(tmp_path):
    popen = FakePopen(("error: invalid argument: --no-webui\nusage: ...", FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen)
    ui = make_ui()
    backend.prepare(ui)
    first, second = popen.args
    assert "--no-webui" in first
    assert second[:9] == first[:9]
    assert "--reasoning-format" not in second and "--no-webui" not in second and "-np" not in second
    assert second[-2:] == ["-ngl", "0"]
    assert backend._minimal_args is True
    assert "just the basics" in output(ui)


def test_unknown_argument_even_with_minimal_args_is_an_error(tmp_path):
    bad = "error: invalid argument: -c"
    popen = FakePopen((bad, FakeProcess(dies_with=1)), (bad, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen)
    with pytest.raises(BackendError, match="start-up settings"):
        backend.prepare(make_ui())
    assert len(popen.calls) == 2


def test_port_conflict_picks_another_port(tmp_path, monkeypatch):
    ports = iter([4001, 4002])
    monkeypatch.setattr(ls, "find_free_port", lambda: next(ports))
    popen = FakePopen(("couldn't bind HTTP server socket, hostname: 127.0.0.1, port: 4001", FakeProcess(dies_with=1)),
                      ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen)
    backend.prepare(make_ui())
    assert popen.args[0][popen.args[0].index("--port") + 1] == "4001"
    assert popen.args[1][popen.args[1].index("--port") + 1] == "4002"
    assert backend.port == 4002


def test_cpu_out_of_memory_explains_and_shows_log(tmp_path):
    log = "llama_model_load: loading model\nggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer of size 9000000000\n"
    proc = FakeProcess(dies_with=1)
    popen = FakePopen((log, proc))
    backend = make_backend(tmp_path, popen=popen)
    ui = make_ui()
    with pytest.raises(BackendError) as err:
        backend.prepare(ui)
    assert "ran out of memory" in str(err.value)
    assert "llama-server.log" in str(err.value)
    assert "failed to allocate buffer" in output(ui)  # the log tail is shown
    assert backend._proc is None


def test_model_error_on_gpu_does_not_switch_builds(tmp_path):
    log = "gguf_init_from_file: invalid magic characters 'abcd'\nllama_model_load: error loading model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer)
    with pytest.raises(BackendError, match="couldn't read the model file"):
        backend.prepare(make_ui())
    assert installer.calls == ["cuda-12"]


def test_health_timeout_stops_the_server(tmp_path):
    proc = FakeProcess()
    http = FakeServerHttp(health=(503,))
    backend = make_backend(tmp_path, popen=FakePopen(("", proc)), http=http, health_timeout_s=10)
    with pytest.raises(BackendError, match="taking too long"):
        backend.prepare(make_ui())
    assert proc.terminate_calls == 1
    assert backend._proc is None


def test_ctrl_c_while_waiting_stops_the_server(tmp_path):
    proc = FakeProcess()
    http = FakeServerHttp(health=(503,))

    def interrupt(_seconds):
        raise KeyboardInterrupt

    backend = make_backend(tmp_path, popen=FakePopen(("", proc)), http=http)
    backend._sleep = interrupt
    with pytest.raises(KeyboardInterrupt):
        backend.prepare(make_ui())
    assert proc.terminate_calls == 1


def test_popen_permission_error_is_friendly(tmp_path):
    def denied(args, **kwargs):
        raise PermissionError(13, "Permission denied")

    backend = make_backend(tmp_path, popen=denied)
    with pytest.raises(BackendError, match="GETTOWORK_HOME"):
        backend.prepare(make_ui())


def test_existing_exe_and_model_skip_install_and_download(tmp_path):
    folder = tmp_path / "rt" / "b7000-vulkan"
    folder.mkdir(parents=True)
    exe = folder / "llama-server"
    exe.write_bytes(b"x")
    (folder / "install.json").write_text(json.dumps({"tag": "b7000", "variant": "vulkan", "exe": "llama-server"}))
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    installer, downloader = FakeInstaller(tmp_path), FakeDownloader(tmp_path)
    popen = FakePopen()
    backend = make_backend(tmp_path, popen=popen, installer=installer, downloader=downloader,
                           server_exe=exe, model_path=model)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == [] and downloader.calls == []
    assert backend.variant is VULKAN and not backend.cpu_only
    assert "-ngl" not in popen.args[0]
    assert "Using your downloaded model: m.gguf" in output(ui)


def test_custom_exe_without_marker_is_treated_as_unknown_gpu_build(tmp_path):
    exe = tmp_path / "mine" / "llama-server"
    exe.parent.mkdir()
    exe.write_bytes(b"x")
    backend = make_backend(tmp_path, server_exe=exe)
    backend.prepare(make_ui())
    assert backend.variant is CUSTOM_VARIANT


def test_missing_saved_exe_is_reinstalled(tmp_path):
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, installer=installer, server_exe=tmp_path / "gone" / "llama-server")
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cpu"]
    assert "gone missing" in output(ui)


def test_install_failure_becomes_backend_error(tmp_path):
    backend = make_backend(tmp_path, installer=FakeInstaller(tmp_path, fail={"cpu"}))
    with pytest.raises(BackendError, match="No cpu build today"):
        backend.prepare(make_ui())


def test_download_failure_becomes_backend_error_but_user_quit_propagates(tmp_path):
    backend = make_backend(tmp_path, downloader=FakeDownloader(tmp_path, error=RuntimeError("repo not found")))
    with pytest.raises(BackendError, match="repo not found"):
        backend.prepare(make_ui())
    backend = make_backend(tmp_path, downloader=FakeDownloader(tmp_path, error=UserQuit()))
    with pytest.raises(UserQuit):
        backend.prepare(make_ui())


def test_no_model_at_all_is_an_error(tmp_path):
    backend = make_backend(tmp_path, entry=None)
    with pytest.raises(BackendError, match="No model"):
        backend.prepare(make_ui())


def test_download_module_is_imported_lazily(tmp_path, monkeypatch):
    calls = []

    def download_gguf(entry, ui, dest_dir=None, *, quant=None, hf_api=None, hf_download=None):
        calls.append((entry.key, quant))
        p = tmp_path / "lazy.gguf"
        p.write_bytes(b"GGUF")
        return p

    fake = types.ModuleType("gettowork.download")
    fake.download_gguf = download_gguf
    monkeypatch.setitem(sys.modules, "gettowork.download", fake)
    backend = LlamaServerBackend(ENTRY, specs=make_specs(), http=FakeServerHttp(), popen=FakePopen(),
                                 installer=FakeInstaller(tmp_path), sleep=lambda s: None, clock=FakeClock(),
                                 log_dir=tmp_path / "logs")
    backend.prepare(make_ui())
    assert calls == [("qwen3-4b", None)]
    assert backend.model_path == tmp_path / "lazy.gguf"


# ---------------------------------------------------------------------------
# chat()
# ---------------------------------------------------------------------------

MESSAGES = [{"role": "system", "content": "TASK: intro"}, {"role": "user", "content": "Begin!"}]


def test_chat_basic_request_and_result(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply("You wake up late!")])
    result = backend.chat(MESSAGES, temperature=0.7, max_tokens=300)
    assert result.text == "You wake up late!"
    assert result.reasoning is None
    assert result.backend == "llamacpp-server"
    assert result.model == "Qwen3 4B (Q4_K_M)"
    assert result.messages == MESSAGES
    assert result.raw["usage"]["completion_tokens"] == 20
    body = http.chat_bodies[0]
    assert body == {"messages": MESSAGES, "temperature": 0.7, "max_tokens": 300, "stream": False}
    call = [c for c in http.calls if c["url"].endswith("/v1/chat/completions")][0]
    assert call["url"] == "http://127.0.0.1:50123/v1/chat/completions"
    assert call["timeout"] == 300.0


def test_chat_json_mode(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply('{"made_progress": true, "explanation": "ok"}')])
    result = backend.chat(MESSAGES, json_mode=True)
    assert http.chat_bodies[0]["response_format"] == {"type": "json_object"}
    assert json.loads(result.text)["made_progress"] is True


def test_chat_reasoning_content_is_separated(tmp_path):
    backend, _, _ = started(tmp_path, chat=[chat_reply("A goose blocks the door.", reasoning="Let me think of something silly.")])
    result = backend.chat(MESSAGES)
    assert result.text == "A goose blocks the door."
    assert result.reasoning == "Let me think of something silly."


def test_chat_inline_think_tags_are_split(tmp_path):
    backend, _, _ = started(tmp_path, chat=[chat_reply("<think>Plan the joke.</think>The bus is a dragon.")])
    result = backend.chat(MESSAGES)
    assert result.text == "The bus is a dragon."
    assert result.reasoning == "Plan the joke."


def test_empty_answer_retries_without_thinking(tmp_path):
    backend, http, _ = started(
        tmp_path,
        chat=[chat_reply("", reasoning="Thinking... thinking... out of tokens"), chat_reply("CHALLENGE: A llama took your car.")],
    )
    notices = []
    backend.on_notice = notices.append  # the game puts this in its spinner
    result = backend.chat(MESSAGES, max_tokens=700)
    assert notices == [RETRY_WITHOUT_THINKING_NOTICE]
    assert result.text == "CHALLENGE: A llama took your car."
    assert result.reasoning == "Thinking... thinking... out of tokens"
    first, retry = http.chat_bodies
    assert "chat_template_kwargs" not in first
    assert retry["chat_template_kwargs"]["enable_thinking"] is False
    assert retry["chat_template_kwargs"]["reasoning_effort"] == "low"  # gpt-oss's switch
    assert retry["reasoning_budget_tokens"] == 0  # the engine's own switch, for always-thinking templates
    # Same budget: the answer alone is short, and a model that ignores the
    # switch can't double the wait.
    assert retry["max_tokens"] == 700
    assert retry["messages"] == MESSAGES


def test_empty_answer_twice_returns_empty_text(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply(""), chat_reply("   ")])
    result = backend.chat(MESSAGES)
    assert result.text == "" and len(http.chat_bodies) == 2


@pytest.mark.parametrize(
    "reply, match",
    [
        ((500, b'{"error": {"code": 500, "message": "the request exceeds the available context size"}}'), "context size"),
        ((400, b"plain text failure"), "plain text failure"),
        ((200, b"not json"), "valid JSON"),
        ((200, b'{"choices": []}'), "without any text"),
        ((200, b"[1, 2]"), "unexpected answer"),
        ((200, b'{"choices": [{"message": "just a string"}]}'), None),
        (ConnectionResetError("reset"), "lost contact"),
        (TimeoutError("timed out"), "longer than 300 seconds"),
        (ls.urllib.error.URLError(TimeoutError("timed out")), "longer than 300 seconds"),
    ],
)
def test_chat_errors_are_friendly_backend_errors(tmp_path, reply, match):
    backend, _, _ = started(tmp_path, chat=[reply, reply])
    if match is None:  # malformed but survivable: treated as an empty answer
        assert backend.chat(MESSAGES).text == ""
        return
    with pytest.raises(BackendError, match=match):
        backend.chat(MESSAGES)


def test_chat_before_prepare_is_an_error(tmp_path):
    backend = make_backend(tmp_path)
    with pytest.raises(BackendError, match="isn't running"):
        backend.chat(MESSAGES)


def test_chat_quietly_restarts_a_crashed_server(tmp_path):
    backend, http, popen = started(tmp_path, chat=[chat_reply("Back in action!")])
    popen.processes[0].crash(139)
    result = backend.chat(MESSAGES)
    assert result.text == "Back in action!"
    assert len(popen.calls) == 2
    assert popen.args[1] == popen.args[0]  # same settings as the last good start


def test_chat_gives_up_if_restart_fails(tmp_path):
    backend, _http, popen = started(tmp_path)
    popen.scripts.append(("boom", FakeProcess(dies_with=1)))
    popen.processes[0].crash(139)
    with pytest.raises(BackendError, match="wouldn't restart"):
        backend.chat(MESSAGES)


# ---------------------------------------------------------------------------
# benchmark()
# ---------------------------------------------------------------------------


def test_benchmark_prefers_server_timings(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply("1, 2, 3", timings={"predicted_per_second": 42.5, "predicted_n": 30})])
    assert backend.benchmark(make_ui()) == 42.5
    body = http.chat_bodies[0]
    assert body["chat_template_kwargs"]["enable_thinking"] is False and body["reasoning_budget_tokens"] == 0
    assert body["max_tokens"] == 64 and body["temperature"] == 0.0


def test_benchmark_falls_back_to_tokens_over_time(tmp_path):
    clock = FakeClock()
    backend, _, _ = started(tmp_path, chat=[chat_reply("1, 2, 3")], clock=clock, on_chat=lambda: clock.sleep(2.0))
    assert backend.benchmark() == pytest.approx(10.0)  # 20 completion tokens in 2 s


def test_benchmark_failures_return_none(tmp_path):
    assert make_backend(tmp_path).benchmark() is None  # not started
    backend, _, _ = started(tmp_path, chat=[(500, b"nope")])
    assert backend.benchmark() is None
    backend2, _, _ = started(tmp_path / "b", chat=[{"choices": [], "usage": {}}])
    assert backend2.benchmark() is None


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------


def test_close_terminates_and_is_idempotent(tmp_path, linux_host):
    backend, _, popen = started(tmp_path)
    proc = popen.processes[0]
    assert backend.close in linux_host  # registered with atexit while running
    backend.close()
    assert proc.terminate_calls == 1 and proc.kill_calls == 0
    assert proc.wait_timeouts == [5.0]
    assert backend.close not in linux_host
    backend.close()
    backend.close()
    assert proc.terminate_calls == 1


def test_close_kills_a_stubborn_process(tmp_path):
    proc = FakeProcess(ignores_terminate=True)
    backend = make_backend(tmp_path, popen=FakePopen(("", proc)))
    backend.prepare(make_ui())
    backend.close()
    assert proc.terminate_calls == 1 and proc.kill_calls == 1
    assert proc.returncode == -9


def test_close_before_prepare_is_harmless(tmp_path):
    make_backend(tmp_path).close()


# ---------------------------------------------------------------------------
# is_available() / model_label
# ---------------------------------------------------------------------------


def test_is_available(tmp_path, monkeypatch):
    ok, why = make_backend(tmp_path).is_available()
    assert ok and "downloaded automatically" in why

    exe = tmp_path / "home" / "runtime" / "llama.cpp" / "b7000-cpu" / "llama-server"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"x")
    (exe.parent / "install.json").write_text(json.dumps({"tag": "b7000", "variant": "cpu", "exe": "llama-server"}))
    ok, why = make_backend(tmp_path).is_available()
    assert ok and "b7000" in why

    ok, why = make_backend(tmp_path, server_exe=exe).is_available()
    assert ok and str(exe) in why

    monkeypatch.setattr(ls.platform, "machine", lambda: "s390x")
    ok, why = make_backend(tmp_path).is_available()
    assert not ok and "Ollama" in why


def test_model_label_variants(tmp_path):
    assert make_backend(tmp_path).model_label == "Qwen3 4B (Q4_K_M)"
    assert make_backend(tmp_path, entry=None, model_path=tmp_path / "tiny-model.gguf").model_label == "tiny-model"
    assert make_backend(tmp_path, entry=None).model_label == "llama.cpp model"


def test_locked_log_file_falls_back_to_a_per_port_name(tmp_path):
    (tmp_path / "logs" / "llama-server.log").mkdir(parents=True)  # can't be opened as a file
    backend, _, _ = started(tmp_path)
    assert backend.log_path == tmp_path / "logs" / "llama-server-50123.log"
    backend.close()


# ---------------------------------------------------------------------------
# Review fixes: fallbacks, engine updates, lifecycle, privacy, timeouts
# ---------------------------------------------------------------------------

GLIBC_LOG = "./llama-server: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found (required by ./libllama.so)\n"


def test_glibc_error_on_cuda_build_falls_back_to_vulkan_then_cpu(tmp_path):
    # Ubuntu 22.04 + NVIDIA: the CUDA build (made on Ubuntu 24.04) can't start,
    # but the Vulkan / CPU builds (made on 22.04) run fine.
    specs = make_specs(gpus=[NVIDIA], flags=["vulkan"])
    popen = FakePopen((GLIBC_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cuda-12", "vulkan"]
    assert backend.variant is VULKAN
    assert "needs a newer Linux" in output(ui)


def test_glibc_error_on_the_cpu_build_explains_the_old_linux(tmp_path):
    popen = FakePopen((GLIBC_LOG, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen)
    with pytest.raises(BackendError, match="glibc"):
        backend.prepare(make_ui())


def test_permanent_failures_are_remembered_on_the_install(tmp_path):
    specs = make_specs(gpus=[NVIDIA])
    popen = FakePopen((GLIBC_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path, marker=True)
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer)
    backend.prepare(make_ui())
    marker = json.loads((tmp_path / "engines" / "cuda-12" / "install.json").read_text())
    assert marker["unusable"]["reason"] == "glibc"
    cpu_marker = json.loads((tmp_path / "engines" / "cpu" / "install.json").read_text())
    assert "unusable" not in cpu_marker


def test_unknown_architecture_fetches_a_newer_engine_and_retries(tmp_path):
    log = "llama_model_load: error loading model architecture: unknown model architecture: 'qwen35'\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path, marker=True, update_to="b9000")
    downloader = FakeDownloader(tmp_path)
    backend = make_backend(tmp_path, popen=popen, installer=installer, downloader=downloader)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cpu", "cpu (update)"]
    assert "b9000-cpu" in backend.server_exe.as_posix()
    assert len(downloader.calls) == 1  # the model is NOT downloaded again
    text = output(ui)
    assert "No need to download the model again" in text and "qwen35" in text


def test_unknown_architecture_with_the_newest_engine_says_redownloading_wont_help(tmp_path):
    log = "unknown model architecture: 'qwen99'\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    installer = FakeInstaller(tmp_path, marker=True)  # update returns the same build
    backend = make_backend(tmp_path, popen=popen, installer=installer)
    with pytest.raises(BackendError) as err:
        backend.prepare(make_ui())
    message = str(err.value)
    assert "qwen99" in message and "won't help" in message and "damaged" not in message
    assert installer.calls == ["cpu", "cpu (update)"]


def test_unknown_architecture_offline_says_try_again_online(tmp_path):
    log = "unknown model architecture: 'qwen99'\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    installer = FakeInstaller(tmp_path, marker=True, fail={"update"})
    backend = make_backend(tmp_path, popen=popen, installer=installer)
    with pytest.raises(BackendError, match="when you're online"):
        backend.prepare(make_ui())


def test_own_llama_server_is_never_replaced(tmp_path):
    exe = tmp_path / "mine" / "llama-server"
    exe.parent.mkdir()
    exe.write_bytes(b"x")
    installer = FakeInstaller(tmp_path)
    popen = FakePopen(("unknown model architecture: 'x'\n", FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen, installer=installer, server_exe=exe)
    with pytest.raises(BackendError, match="your own llama-server"):
        backend.prepare(make_ui())
    assert installer.calls == []


def test_model_error_without_corruption_signs_doesnt_blame_the_download(tmp_path):
    popen = FakePopen(("llama_model_load: error loading model: vocab mismatch\nfailed to load model\n", FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen)
    with pytest.raises(BackendError) as err:
        backend.prepare(make_ui())
    assert "couldn't load this model file" in str(err.value) and "damaged" not in str(err.value)


def test_mmap_invalid_argument_is_not_treated_as_a_bad_flag(tmp_path):
    log = "llama_model_load: error loading model: mmap failed: Invalid argument\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen)
    with pytest.raises(BackendError, match="couldn't load this model file"):
        backend.prepare(make_ui())
    assert len(popen.calls) == 1  # no pointless "minimal args" restart


def test_cuda_invalid_argument_still_falls_back_to_the_next_build(tmp_path):
    log = "CUDA error: invalid argument\n  current device: 0, in function ggml_cuda_op\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)), ("", FakeProcess()))
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer)
    backend.prepare(make_ui())
    assert installer.calls == ["cuda-12", "cpu"] and backend.variant is CPU


def test_gpu_build_that_quietly_runs_on_the_cpu_is_reported(tmp_path):
    log = "load_tensors: loading model tensors, this can take a while... (mmap = true)\nmain: server is listening\n"
    popen = FakePopen((log, FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]))
    ui = make_ui()
    backend.prepare(ui)
    assert backend.cpu_only is True and backend.gpu_note == "driver"
    text = " ".join(output(ui).split())
    assert "Updating your graphics driver" in text and "CPU mode" in text


def test_gpu_build_that_offloads_layers_is_left_alone(tmp_path):
    log = "load_tensors: loading model tensors\nload_tensors: offloaded 37/37 layers to GPU\n"
    backend = make_backend(tmp_path, popen=FakePopen((log, FakeProcess())), specs=make_specs(gpus=[NVIDIA]))
    backend.prepare(make_ui())
    assert backend.cpu_only is False and backend.gpu_note is None


def test_gpu_offload_from_log():
    assert ls.gpu_offload_from_log("load_tensors: offloaded 0/29 layers to GPU") is False
    assert ls.gpu_offload_from_log("load_tensors: offloaded 12/29 layers to GPU") is True
    assert ls.gpu_offload_from_log("load_tensors:   CPU_Mapped model buffer size") is False
    assert ls.gpu_offload_from_log("some other build's log") is None


def test_child_environment_drops_secrets_and_llama_settings(tmp_path):
    base = {"PATH": "/bin", "LLAMA_API_KEY": "mine", "LLAMA_ARG_N_GPU_LAYERS": "0", "GITHUB_TOKEN": "ghp_x",
            "HF_TOKEN": "hf_x", "TYPESAFE_API_KEY": "tsk_x", "HOME": "/home/me"}
    env = server_env(tmp_path / "llama-server", base, api_key="session-key")
    assert env["LLAMA_API_KEY"] == "session-key"
    for gone in ("LLAMA_ARG_N_GPU_LAYERS", "GITHUB_TOKEN", "HF_TOKEN", "TYPESAFE_API_KEY"):
        assert gone not in env
    assert env["HOME"] == "/home/me"


def test_server_requires_a_per_launch_key_and_chat_sends_it(tmp_path):
    backend, http, popen = started(tmp_path, chat=[chat_reply("Hi!")])
    key = popen.calls[0][1]["env"]["LLAMA_API_KEY"]
    assert len(key) >= 20 and key not in " ".join(popen.args[0])  # never on the command line
    backend.chat(MESSAGES)
    chat_call = [c for c in http.calls if c["url"].endswith("/v1/chat/completions")][0]
    assert chat_call["headers"]["Authorization"] == f"Bearer {key}"
    health_call = [c for c in http.calls if c["url"].endswith("/health")][0]
    assert "Authorization" not in health_call["headers"]  # /health stays public


def test_posix_launch_uses_its_own_session_and_a_parent_death_signal(tmp_path, monkeypatch):
    # The real hook needs Linux's prctl(), which a macOS/Windows test runner doesn't
    # have even while the autouse fixture pretends to be Linux - so stand it in.
    def hook() -> None:
        pass

    monkeypatch.setattr(ls, "_parent_death_signal_hook", lambda: hook)
    backend, _http, popen = started(tmp_path)
    kwargs = popen.calls[0][1]
    assert kwargs["start_new_session"] is True  # terminal Ctrl+C doesn't reach the engine
    assert kwargs.get("preexec_fn") is hook  # Linux: the kernel stops it if the game dies
    backend.close()


def test_posix_launch_without_a_death_signal_hook_still_starts(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "_parent_death_signal_hook", lambda: None)  # e.g. macOS
    backend, _http, popen = started(tmp_path)
    kwargs = popen.calls[0][1]
    assert kwargs["start_new_session"] is True
    assert "preexec_fn" not in kwargs
    backend.close()


def test_parent_death_signal_hook_is_linux_only(monkeypatch):
    monkeypatch.setattr(ls.platform, "system", lambda: "Darwin")
    assert ls._parent_death_signal_hook() is None


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="prctl() only exists on Linux")
def test_parent_death_signal_hook_on_real_linux():
    assert callable(ls._parent_death_signal_hook())


def test_windows_launch_joins_the_kill_on_close_job(tmp_path, monkeypatch):
    monkeypatch.setattr(ls.platform, "system", lambda: "Windows")
    joined = []
    monkeypatch.setattr(ls, "_assign_to_kill_on_close_job", lambda proc: joined.append(proc) or True)
    backend, _http, popen = started(tmp_path)
    assert joined == [popen.processes[0]]
    assert "start_new_session" not in popen.calls[0][1]
    backend.close()


def test_kill_on_close_job_is_harmless_where_it_cant_work():
    assert ls._assign_to_kill_on_close_job(object()) is False  # no process handle (or not Windows)


def _sleeper_named_llama_server(tmp_path):
    exe = tmp_path / "bin" / "llama-server"
    exe.parent.mkdir(parents=True, exist_ok=True)
    exe.write_bytes(Path("/bin/sleep").read_bytes())
    exe.chmod(0o755)
    return exe


@pytest.mark.skipif(sys.platform != "linux", reason="uses a real Linux process")
def test_real_engine_process_is_in_its_own_process_group(tmp_path):
    exe = _sleeper_named_llama_server(tmp_path)
    backend = make_backend(tmp_path, popen=lambda args, **kw: subprocess.Popen([str(exe), "30"], **kw))
    backend._launch(exe, tmp_path / "m.gguf", cpu_only=True, minimal=False)
    try:
        assert os.getpgid(backend._proc.pid) != os.getpgid(0)
    finally:
        backend.close()


@pytest.mark.skipif(sys.platform != "linux", reason="uses a real Linux process")
def test_orphaned_engine_from_a_force_quit_game_is_stopped(tmp_path):
    exe = _sleeper_named_llama_server(tmp_path)
    orphan = subprocess.Popen([str(exe), "60"])
    try:
        logs = tmp_path / "logs"
        logs.mkdir()
        record = {"pid": orphan.pid, "started": ls._process_started(orphan.pid), "exe": str(exe),
                  "game_pid": 999999, "game_started": 1.0}  # that game is long gone
        (logs / f"llama-server-999999-{orphan.pid}{ls.OWNER_SUFFIX}").write_text(json.dumps(record))
        # A record from a game that's still running (us) must be left alone.
        ours = {"pid": os.getpid(), "started": ls._process_started(os.getpid()), "exe": "x",
                "game_pid": os.getpid(), "game_started": ls._process_started(os.getpid())}
        (logs / f"llama-server-{os.getpid()}-1{ls.OWNER_SUFFIX}").write_text(json.dumps(ours))

        assert ls.reap_orphaned_servers(logs) == 1
        assert orphan.wait(timeout=10) is not None
        assert [p.name for p in logs.iterdir()] == [f"llama-server-{os.getpid()}-1{ls.OWNER_SUFFIX}"]
    finally:
        if orphan.poll() is None:
            orphan.kill()


def test_owner_record_is_written_while_running_and_removed_on_close(tmp_path):
    backend, _http, _popen = started(tmp_path)
    records = list((tmp_path / "logs").glob(f"*{ls.OWNER_SUFFIX}"))
    assert len(records) == 1
    backend.close()
    assert not list((tmp_path / "logs").glob(f"*{ls.OWNER_SUFFIX}"))


def test_log_in_use_by_another_game_gets_its_own_file(tmp_path, posix_file_locks):
    logs = tmp_path / "logs"
    logs.mkdir()
    other = posix_file_locks.hold(logs / "llama-server.log")  # another copy of the game, still running
    other.write(b"the other game's engine is busy\n")
    other.flush()
    try:
        backend, _http, _ = started(tmp_path)
        assert backend.log_path == logs / "llama-server-50123.log"
        assert (logs / "llama-server.log").read_bytes() == b"the other game's engine is busy\n"  # untouched
        backend.close()
    finally:
        other.close()
    again, _http, _ = started(tmp_path)  # that game has quit: the shared log is free again
    assert again.log_path == logs / "llama-server.log"
    again.close()


@needs_real_flock
def test_real_flock_keeps_a_second_game_off_the_log(tmp_path, monkeypatch):
    monkeypatch.setattr(ls, "_posix_locks_available", REAL_POSIX_LOCKS_AVAILABLE)
    monkeypatch.setattr(ls, "_lock_exclusively", REAL_LOCK_EXCLUSIVELY)
    assert ls._posix_locks_available() is True
    logs = tmp_path / "logs"
    logs.mkdir()
    with open(logs / "llama-server.log", "ab") as other:  # another copy of the game, still running
        _host_fcntl.flock(other.fileno(), _host_fcntl.LOCK_EX | _host_fcntl.LOCK_NB)
        other.write(b"the other game's engine is busy\n")
        other.flush()
        backend, _http, _ = started(tmp_path)
        assert backend.log_path == logs / "llama-server-50123.log"
        assert (logs / "llama-server.log").read_bytes() == b"the other game's engine is busy\n"  # untouched
        backend.close()
    with open(logs / "llama-server.log", "ab") as first, open(logs / "llama-server.log", "ab") as second:
        assert ls._lock_exclusively(first) is True
        assert ls._lock_exclusively(second) is False  # a lock belongs to the open file, not the process


def test_health_timeout_grows_with_the_model_size():
    assert ls.health_timeout_for(2.5) == ls.HEALTH_TIMEOUT_S
    assert ls.health_timeout_for(63.4) == pytest.approx(63.4 * 15)
    assert ls.health_timeout_for(500) == ls.HEALTH_MAX_TIMEOUT_S


def test_big_model_file_gets_a_longer_wake_up_time(tmp_path, monkeypatch):
    model = tmp_path / "big.gguf"
    model.write_bytes(b"GGUF")
    fake_file_sizes(monkeypatch, {model: 40 * 10**9})  # 40 GB on paper, nothing on disk
    backend = LlamaServerBackend(ENTRY, specs=make_specs(), http=FakeServerHttp(), popen=FakePopen(),
                                 installer=FakeInstaller(tmp_path), downloader=FakeDownloader(tmp_path),
                                 sleep=lambda s: None, clock=FakeClock(), log_dir=tmp_path / "logs", model_path=model)
    backend.prepare(make_ui())
    assert backend._health_timeout_s == pytest.approx(600.0)
    backend.close()


def test_a_growing_log_keeps_the_wait_going_past_the_timeout(tmp_path):
    clock = FakeClock()
    proc = FakeProcess()
    http = FakeServerHttp(health=[503] * 30 + [200])
    popen = FakePopen(("loading\n", proc))
    backend = make_backend(tmp_path, popen=popen, http=http, clock=clock, health_timeout_s=5)

    def sleep_and_log(seconds):
        clock.sleep(seconds)
        with open(backend.log_path, "ab") as fh:
            fh.write(b".")  # the engine is still busy loading

    backend._sleep = sleep_and_log
    backend.prepare(make_ui())  # 15 s of polling, well past the 5 s timeout: still fine
    assert clock.t - 1000.0 > 5
    backend.close()


def test_chat_timeout_scales_with_the_measured_speed(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply("1, 2", timings={"predicted_per_second": 3.0}),
                                               chat_reply("A long story.")])
    assert backend.benchmark() == 3.0
    backend.chat(MESSAGES, max_tokens=1200)
    chat_calls = [c for c in http.calls if c["url"].endswith("/v1/chat/completions")]
    assert chat_calls[-1]["timeout"] == pytest.approx(60 + 1.5 * 1200 / 3.0)  # 660 s, not 300


def test_chat_can_switch_thinking_off_and_stop_early(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply("Story.", finish_reason="stop")])
    backend.chat(MESSAGES, think=False, stop=["\nPlayer:", "\nYou:"])
    body = http.chat_bodies[0]
    # Every switch we know: Qwen3/SmolLM3, gpt-oss, Seed-OSS's thinking budget...
    assert body["chat_template_kwargs"] == {"enable_thinking": False, "reasoning_effort": "low", "thinking_budget": 0}
    # ...and llama.cpp's own budget, which also stops models whose template forces thinking (QwQ, R1 distills).
    assert body["reasoning_budget_tokens"] == 0
    assert body["stop"] == ["\nPlayer:", "\nYou:"]


def test_chat_marks_answers_cut_off_by_the_token_limit(tmp_path):
    reply = chat_reply("You race for the door, where")
    reply["choices"][0]["finish_reason"] = "length"
    backend, _, _ = started(tmp_path, chat=[reply])
    assert backend.chat(MESSAGES).truncated is True
    backend2, _, _ = started(tmp_path / "b", chat=[chat_reply("Done.")])
    assert backend2.chat(MESSAGES).truncated is False


def test_log_tail_is_stripped_of_terminal_control_codes(tmp_path):
    log = "error loading model: unknown model architecture: '\x1b]0;PWNED\x1b\\evil'\n\x1b[2Aoverwritten\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen, installer=FakeInstaller(tmp_path, marker=True))
    ui = make_ui()
    with pytest.raises(BackendError):
        backend.prepare(ui)
    assert "\x1b" not in output(ui)


# ---------------------------------------------------------------------------
# Round 3: engine check before the model download; crashes after start-up;
# a GPU build stuck starting; one log per game on Windows; tidying up
# ---------------------------------------------------------------------------


class FakeRunner:
    """Like subprocess.run for `llama-server --version`: results per exe folder name."""

    def __init__(self, results=None):
        self.results = dict(results or {})
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        self.calls.append(list(args))
        folder = Path(args[0]).parent.name
        item = self.results.get(folder, (0, "version: 7000 (abcdef0)"))
        if isinstance(item, BaseException):
            raise item
        code, out = item
        return subprocess.CompletedProcess(args, code, stdout=out.encode("utf-8"))


GLIBC_FAIL = (1, "llama-server: /lib/aarch64-linux-gnu/libc.so.6: version `GLIBC_2.38' not found")


def test_an_engine_that_cant_run_here_is_found_before_the_model_download(tmp_path):
    installer = FakeInstaller(tmp_path, marker=True)
    downloader = FakeDownloader(tmp_path)
    runner = FakeRunner({"cpu": GLIBC_FAIL})
    backend = make_backend(tmp_path, installer=installer, downloader=downloader, runner=runner)
    with pytest.raises(BackendError, match="glibc"):
        backend.prepare(make_ui())
    assert downloader.calls == []  # the multi-GB model was never downloaded
    info = json.loads((tmp_path / "engines" / "cpu" / "install.json").read_text())
    assert info["unusable"]["reason"] == "glibc"  # remembered: never downloaded again


def test_a_gpu_build_that_cant_run_moves_to_the_next_build_before_downloading(tmp_path):
    installer = FakeInstaller(tmp_path, marker=True)
    downloader = FakeDownloader(tmp_path)
    runner = FakeRunner({"cuda-12": GLIBC_FAIL})
    popen = FakePopen()
    backend = make_backend(tmp_path, installer=installer, downloader=downloader, runner=runner, popen=popen,
                           specs=make_specs(gpus=[NVIDIA]))
    backend.prepare(make_ui())
    assert installer.calls == ["cuda-12", "cpu"]
    assert backend.variant.name != "cuda-12"
    assert len(downloader.calls) == 1 and len(popen.calls) == 1


def test_an_exe_the_system_cant_execute_is_reported_clearly(tmp_path):
    runner = FakeRunner({"cpu": FileNotFoundError(2, "No such file or directory")})
    backend = make_backend(tmp_path, installer=FakeInstaller(tmp_path, marker=True), runner=runner)
    with pytest.raises(BackendError, match="can't run the prebuilt llama.cpp engine"):
        backend.prepare(make_ui())


@pytest.mark.skipif(os.name == "nt", reason="a shell-script stand-in for llama-server (Linux/macOS)")
def test_the_engine_checks_run_a_relative_engine_path_by_its_full_path(tmp_path, monkeypatch):
    """GETTOWORK_ENGINE_DIR=game/engine: the checks run with the engine's folder as the working directory,
    where Linux and macOS would look for the relative path - a false "can't run here", remembered for good."""
    folder = tmp_path / "game" / "engine" / "b7000-cpu"
    folder.mkdir(parents=True)
    exe = folder / "llama-server"
    exe.write_text("#!/bin/sh\necho 'Available devices:'\necho 'version: 7000 (fake)'\n", encoding="utf-8")
    exe.chmod(0o755)
    monkeypatch.chdir(tmp_path)
    backend = make_backend(tmp_path, runner=subprocess.run)
    relative = Path("game") / "engine" / "b7000-cpu" / "llama-server"
    assert backend._probe_engine(relative) is None  # it ran fine
    assert backend._probe_devices(relative) is not None  # and listed its devices


def test_an_unclear_engine_check_result_doesnt_block_the_start(tmp_path):
    runner = FakeRunner({"cpu": subprocess.TimeoutExpired("llama-server", 15)})
    backend = make_backend(tmp_path, runner=runner)
    backend.prepare(make_ui())
    assert backend._proc is not None


def test_is_available_says_no_up_front_when_no_build_can_run(tmp_path, monkeypatch):
    monkeypatch.setattr(ls.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(runtime_install, "_glibc_version", lambda: (2, 36))
    ok, why = make_backend(tmp_path).is_available()
    assert not ok and "glibc" in why


def gpu_backend(tmp_path, *, chat, popen=None, **kwargs):
    http = FakeServerHttp(health=(200,), chat=chat)
    popen = popen or FakePopen()
    backend = make_backend(tmp_path, popen=popen, http=http, specs=make_specs(gpus=[NVIDIA]),
                           installer=FakeInstaller(tmp_path), **kwargs)
    backend.prepare(make_ui())
    return backend, http, popen


def test_a_gpu_crash_during_the_warm_up_falls_back_instead_of_saying_up_and_running(tmp_path):
    crash = ConnectionResetError("connection reset by peer")
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()))
    backend, http, popen = gpu_backend(tmp_path, popen=popen, chat=[crash, chat_reply("1, 2, 3", timings={"predicted_per_second": 9.0})])
    first = popen.processes[0]
    http.on_chat = lambda: first.crash(-6) if len(http.chat_bodies) == 1 else None
    ui = make_ui()
    speed = backend.benchmark(ui)
    assert speed == 9.0
    assert len(popen.calls) == 2  # restarted on a safer setup
    assert backend.cpu_only or not backend.variant.gpu
    assert "couldn't get going" in output(ui) or "CPU mode" in output(ui)


def test_the_warm_up_uses_a_prompt_the_size_of_a_real_turn(tmp_path):
    backend, http, _ = started(tmp_path, chat=[chat_reply("1, 2", timings={"predicted_per_second": 30.0})])
    backend.benchmark()
    text = " ".join(m["content"] for m in http.chat_bodies[-1]["messages"])
    assert len(text.split()) > 500


def test_an_engine_that_dies_on_real_work_and_cant_recover_is_reported(tmp_path):
    from gettowork.backends.base import EngineStopped

    crash = ConnectionResetError("reset")
    backend, http, popen = started(tmp_path, chat=[crash])  # a CPU build: nothing safer to switch to
    http.on_chat = lambda: popen.processes[-1].crash(-9)
    with pytest.raises(EngineStopped):
        backend.benchmark(make_ui())


def test_a_gpu_crash_mid_game_switches_to_cpu_mode_and_answers(tmp_path):
    notices: list[str] = []
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()))
    backend, http, popen = gpu_backend(tmp_path, popen=popen,
                                       chat=[ConnectionResetError("reset"), chat_reply("The goose bows.")])
    backend.on_notice = notices.append
    http.on_chat = lambda: popen.processes[0].crash(-6) if len(http.chat_bodies) == 1 else None
    result = backend.chat([{"role": "user", "content": "hi"}])
    assert result.text == "The goose bows."
    assert backend.cpu_only and popen.args[-1][-4:] == ["--device", "none", "-ngl", "0"]
    assert any("CPU mode" in n for n in notices)


def test_closing_mid_answer_never_starts_a_new_engine(tmp_path, linux_host):
    """The window closed while the model was writing: the game quits, atexit runs
    close() while the game's thread is still waiting for the answer. Its request
    then fails - that must not look like a GPU crash that starts a CPU-mode engine
    nobody would ever stop (macOS has no parent-death signal)."""
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()))
    backend, http, popen = gpu_backend(tmp_path, popen=popen, chat=[ConnectionResetError("reset"), chat_reply("x")])
    assert backend.variant.gpu and not backend.cpu_only
    http.on_chat = backend.close  # the quit arrives while the answer is being written
    with pytest.raises(BackendError):
        backend.chat([{"role": "user", "content": "hi"}])
    assert len(popen.calls) == 1  # no second engine
    assert popen.processes[0].terminate_calls == 1
    assert backend._proc is None and backend.close not in linux_host


def test_a_closed_backend_refuses_to_start_or_restart_the_engine(tmp_path):
    backend, http, popen = gpu_backend(tmp_path, chat=[chat_reply("a")])
    backend.close()
    with pytest.raises(BackendError, match="closing"):
        backend.chat([{"role": "user", "content": "hi"}])
    assert not backend._switch_after_crash(None, "gpu")
    with pytest.raises(BackendError, match="closing"):
        backend._launch(backend.server_exe, backend.model_path, cpu_only=True, minimal=False)
    assert backend.benchmark() is None
    assert len(popen.calls) == 1
    backend.prepare(make_ui())  # prepare() starts afresh
    assert len(popen.calls) == 2 and backend.chat([{"role": "user", "content": "hi"}]).text == "a"
    backend.close()


def test_close_racing_a_launch_on_another_thread_leaves_no_engine(tmp_path):
    """close() from the main thread while the game thread is inside Popen: close()
    waits for the launch to finish, then stops the engine it started (a launch
    after close() is refused - see the test above)."""
    import threading as _threading

    backend, http, popen = gpu_backend(tmp_path, chat=[])
    entered, release = _threading.Event(), _threading.Event()
    real_call = popen.__call__

    def slow_popen(args, **kwargs):
        entered.set()
        release.wait(5)
        return real_call(args, **kwargs)

    backend._stop_process()  # the game thread is about to relaunch (as after a crash)
    backend._popen = slow_popen
    errors: list = []

    def relaunch():
        try:
            backend._launch(backend.server_exe, backend.model_path, cpu_only=True, minimal=False)
        except BackendError as exc:
            errors.append(exc)

    worker = _threading.Thread(target=relaunch)
    worker.start()
    assert entered.wait(5)
    closer = _threading.Thread(target=backend.close)
    closer.start()
    release.set()
    worker.join(5)
    closer.join(5)
    assert backend._proc is None and backend._owner_file is None
    assert len(popen.processes) == 2  # prepare's engine, then the racing relaunch...
    assert all(p.returncode is not None for p in popen.processes)  # ...and every one was stopped
    assert errors == []  # the launch won the race here; close() then stopped what it started


def test_a_repeated_idle_stop_switches_setup_instead_of_reloading_forever(tmp_path):
    notices: list[str] = []
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()))
    backend, http, popen = gpu_backend(tmp_path, popen=popen, chat=[chat_reply("a"), chat_reply("b")])
    backend.on_notice = notices.append
    popen.processes[-1].crash(1)
    backend.chat([{"role": "user", "content": "hi"}])  # first stop: a quiet restart of the same setup
    assert any("restarting" in n for n in notices) and not backend.cpu_only
    popen.processes[-1].crash(1)
    backend._restarts = 1  # it stopped again before answering
    backend.chat([{"role": "user", "content": "hi"}])
    assert backend.cpu_only


def test_a_gpu_build_stuck_setting_up_the_device_falls_back(tmp_path):
    clock = FakeClock()
    stuck = FakeProcess()
    popen = FakePopen(("ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: 0 = Intel(R) Iris(R) Xe Graphics\n", stuck))
    http = FakeServerHttp(health=(503, 503, 503) * 1000 + (200,))
    specs = make_specs(gpus=[GPUInfo(name="Intel Iris Xe", vendor="intel", vram_gb=0.0)], flags=["vulkan"])
    installer = FakeInstaller(tmp_path)
    backend = make_backend(tmp_path, popen=popen, http=http, specs=specs, installer=installer, clock=clock,
                           health_timeout_s=30.0)

    def health_after_switch(*a, **k):
        return "ok" if len(popen.calls) > 1 else "loading"

    backend._probe_health = health_after_switch
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["vulkan", "cpu"]
    assert "never finished getting ready" in output(ui)


def test_a_slow_but_loading_gpu_start_still_times_out_with_the_disk_message(tmp_path):
    clock = FakeClock()
    popen = FakePopen(("load_tensors: loading model tensors\n.........\n", FakeProcess()))
    specs = make_specs(gpus=[NVIDIA])
    backend = make_backend(tmp_path, popen=popen, specs=specs, clock=clock, health_timeout_s=30.0)
    backend._probe_health = lambda *a, **k: "loading"
    with pytest.raises(BackendError, match="taking too long"):
        backend.prepare(make_ui())


def test_windows_games_each_use_their_own_log(tmp_path, monkeypatch):
    # A Windows host: no fcntl (so no flock to guard a shared log file).
    monkeypatch.setattr(ls.platform, "system", lambda: "Windows")
    monkeypatch.setattr(ls, "_assign_to_kill_on_close_job", lambda proc: True)
    monkeypatch.setattr(ls, "_posix_locks_available", lambda: False)

    def no_flock(fh):
        raise AssertionError("there is no flock() on Windows to rely on")

    monkeypatch.setattr(ls, "_lock_exclusively", no_flock)
    first = make_backend(tmp_path)
    first.prepare(make_ui())
    first_log = first.log_path
    first_log.write_bytes(b"first game's engine output\n")
    second = make_backend(tmp_path)  # a second copy of the game, while the first still runs
    second.prepare(make_ui())
    try:
        assert second.log_path != first_log
        assert first_log.read_bytes() == b"first game's engine output\n"  # not truncated
        for log in (first_log, second.log_path):
            assert log.parent == tmp_path / "logs" and log.is_file()
            assert log.name.startswith(f"llama-server-{os.getpid()}-") and log.suffix == ".log"
            assert log.name not in ("llama-server.log", "llama-server-50123.log")  # never a shared name
    finally:
        first.close()
        second.close()


def test_lock_helper_says_no_without_flock(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_fcntl(name, *a, **k):
        if name == "fcntl":
            raise ImportError("no fcntl on Windows")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    assert REAL_LOCK_EXCLUSIVELY(io.BytesIO()) is False
    assert REAL_POSIX_LOCKS_AVAILABLE() is False


def test_a_successful_start_tidies_older_engine_copies(tmp_path, monkeypatch):
    seen = {}

    def fake_prune(exe, *, runtime_root=None, in_use=None):
        seen["exe"] = exe
        return 1, 900_000_000

    monkeypatch.setattr(runtime_install, "prune_old_installs", fake_prune)
    ui = make_ui()
    backend = make_backend(tmp_path, installer=FakeInstaller(tmp_path, marker=True))
    backend.prepare(ui)
    assert seen["exe"] == backend.server_exe
    assert "900 MB freed" in output(ui)


# ---------------------------------------------------------------------------
# Round 4: real engine logs; seeing the graphics card; slow loads; updates
# ---------------------------------------------------------------------------

# What the official b11100 build writes at its normal log level: nothing at
# all about tensors, buffers or offloading.
REAL_B11100_LOG = """\
0.00.000.675 I srv  llama_server: initializing ...
0.00.025.912 I cmn  common_param: common_params_print_info: verbosity = 3 (adjust with the `-lv N` CLI arg)
0.00.026.083 I srv          init: The UI is disabled
0.00.026.104 I srv          init: Use --ui/--no-ui (or deprecated --webui/--no-webui) to enable/disable
0.00.027.604 I srv    load_model: loading model '/models/qwen3-4b-Q4_K_M.gguf'
"""
REAL_B11100_READY = """\
0.00.177.891 I cmn          init: llama threadpool init, n_threads = 4
0.00.181.901 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 4096, kv_unified = 'false'
0.00.187.107 I srv  llama_server: model loaded
"""
# The same build at -lv 4 on a computer with no graphics card: the RPC backend
# makes it say "offloaded 3/3 layers to GPU" while the weights sit in RAM.
REAL_B11100_LV4_NO_GPU = """\
0.00.136.085 I load_tensors: loading model tensors, this can take a while... (load_mode = mmap)
0.00.136.525 I load_tensors: offloading output layer to GPU
0.00.136.555 I load_tensors: offloading 1 repeating layers to GPU
0.00.136.556 I load_tensors: offloaded 3/3 layers to GPU
0.00.136.558 I load_tensors:   CPU_Mapped model buffer size =     8.13 MiB
0.00.137.465 I llama_kv_cache:        CPU KV buffer size =     2.00 MiB
"""


def test_gpu_use_is_read_from_where_the_weights_went_not_the_offload_count():
    assert ls.gpu_offload_from_log(REAL_B11100_LOG + REAL_B11100_READY) is None
    assert ls.gpu_offload_from_log(REAL_B11100_LV4_NO_GPU) is False  # RPC's "offloaded 3/3" is not a GPU
    cuda = ("load_tensors: offloaded 37/37 layers to GPU\n"
            "load_tensors:        CUDA0 model buffer size =  2249.11 MiB\n"
            "load_tensors:   CPU_Mapped model buffer size =   303.75 MiB\n")
    assert ls.gpu_offload_from_log(cuda) is True
    assert ls.gpu_offload_from_log("llm_load_tensors:    Vulkan0 buffer size =  4403.49 MiB\n") is True
    assert ls.gpu_offload_from_log("load_tensors:    CUDA_Host model buffer size = 300 MiB\n") is False


def test_device_listing_is_parsed():
    assert ls.gpu_devices_from_listing("0.00.000.665 I srv  llama_server: initializing ...\n"
                                       "Available devices:\n  (none)\n") == []
    listing = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 3060 (12288 MiB, 11000 MiB free)\n"
    assert ls.gpu_devices_from_listing(listing) == ["CUDA0"]
    assert ls.gpu_devices_from_listing("Available devices:\n  BLAS: Accelerate\n") == []
    assert ls.gpu_devices_from_listing("error: invalid argument: --list-devices") is None


class DeviceRunner(FakeRunner):
    """`--version` works everywhere; `--list-devices` answers per build folder."""

    def __init__(self, devices):
        super().__init__()
        self.devices = devices

    def __call__(self, args, **kwargs):
        if args[1:] == ["--list-devices"]:
            self.calls.append(list(args))
            out = self.devices.get(Path(args[0]).parent.name, "Available devices:\n  (none)\n")
            return subprocess.CompletedProcess(args, 0, stdout=out.encode("utf-8"))
        return super().__call__(args, **kwargs)


def test_a_gpu_build_that_sees_no_graphics_card_is_swapped_before_the_model_loads(tmp_path):
    installer = FakeInstaller(tmp_path)
    popen = FakePopen((REAL_B11100_LOG + REAL_B11100_READY, FakeProcess()))
    specs = make_specs(gpus=[NVIDIA], flags=["vulkan"])
    runner = DeviceRunner({"vulkan": "Available devices:\n  Vulkan0: NVIDIA GeForce RTX 3060 (12288 MiB)\n"})
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=installer, runner=runner)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cuda-12", "vulkan"]
    assert len(popen.calls) == 1 and "vulkan" in popen.args[0][0]  # the model was only loaded once
    assert "couldn't find your graphics card" in " ".join(output(ui).split())
    assert backend.cpu_only is False


def test_with_no_other_build_a_blind_gpu_build_runs_in_cpu_mode_with_a_driver_hint(tmp_path):
    installer = FakeInstaller(tmp_path, fail={"cpu"})
    popen = FakePopen((REAL_B11100_LOG + REAL_B11100_READY, FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer,
                           runner=DeviceRunner({}))
    ui = make_ui()
    backend.prepare(ui)
    assert popen.args[0][-4:] == ["--device", "none", "-ngl", "0"]
    assert backend.cpu_only is True and backend.gpu_note == "driver"
    assert "graphics driver" in " ".join(output(ui).split())


def test_a_slow_load_that_keeps_reading_the_disk_is_not_called_a_gpu_hang(tmp_path):
    """A 20 GB model from a slow disk on a CUDA build: the log is silent (as with
    current engines), but the engine keeps reading the file - so keep waiting."""
    clock = FakeClock()
    installer = FakeInstaller(tmp_path)
    popen = FakePopen((REAL_B11100_LOG, FakeProcess()))
    reads = {"n": 0}

    def io_probe(proc):
        reads["n"] += 1
        return reads["n"] * 30_000_000  # ~60 MB/s at one check every 0.5 s

    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer,
                           clock=clock, health_timeout_s=30.0, io_probe=io_probe)
    start = clock.t
    backend._probe_health = lambda *a, **k: "ok" if clock.t - start > 330 else "loading"
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cuda-12"] and len(popen.calls) == 1
    assert "never finished getting ready" not in output(ui)


def test_a_quiet_gpu_start_asks_before_switching_builds(tmp_path):
    clock = FakeClock()
    installer = FakeInstaller(tmp_path)
    popen = FakePopen((REAL_B11100_LOG, FakeProcess()))
    answers = iter(["wait", "keep"])
    ui = UI(console=Console(file=io.StringIO(), width=200), input_fn=lambda prompt: next(answers, ""))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer,
                           clock=clock, health_timeout_s=30.0)
    start = clock.t
    backend._probe_health = lambda *a, **k: "ok" if clock.t - start > 75 else "loading"
    backend.prepare(ui)
    assert installer.calls == ["cuda-12"] and len(popen.calls) == 1  # waited on the same engine
    assert "Keep waiting" in output(ui) and "hasn't started loading the model" in " ".join(output(ui).split())


def test_a_split_model_gets_time_to_load_every_part(tmp_path, monkeypatch):
    folder = tmp_path / "split"
    folder.mkdir()
    parts = [folder / f"big-Q4_K_M-0000{part}-of-00003.gguf" for part in (1, 2, 3)]
    for path, size in zip(parts, (3, 5, 7)):
        path.write_bytes(b"x" * size)
    first = parts[0]
    assert ls._model_total_bytes(first) == 15  # real (tiny) files: every part is counted
    fake_file_sizes(monkeypatch, dict(zip(parts, (15 * 10**9, 15 * 10**9, 5 * 10**9))))  # nothing big on disk
    assert ls._model_total_bytes(first) == 35 * 10**9
    backend = make_backend(tmp_path, model_path=first)
    backend.prepare(make_ui())
    assert backend._health_timeout_s == pytest.approx(ls.health_timeout_for(35.0))


def test_a_cuda_build_without_code_for_this_card_is_remembered_and_skipped(tmp_path):
    log = ("ggml_cuda_init: found 1 CUDA devices:\n  Device 0: NVIDIA GeForce GTX 1080, compute capability 6.1\n"
           "CUDA error: no kernel image is available for execution on the device\n")
    assert classify_server_log(log, 1) == "gpu_arch"
    installer = FakeInstaller(tmp_path, marker=True)
    popen = FakePopen((log, FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]), installer=installer)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == ["cuda-12", "cpu"]
    info = json.loads((tmp_path / "engines" / "cuda-12" / "install.json").read_text())
    assert info["unusable"]["reason"] == "gpu_arch"
    assert "graphics card's generation" in output(ui)


def test_a_missing_linux_library_gets_a_linux_fix_not_windows_advice(tmp_path):
    text = ("./llama-server: error while loading shared libraries: libgomp.so.1: cannot open shared object file: "
            "No such file or directory")
    assert ls.missing_library_name(text) == "libgomp.so.1"
    hint = ls.missing_library_hint(text, "Linux")
    assert "libgomp1" in hint and "apt install" in hint and "Visual C++" not in hint
    assert "Visual C++" in ls.missing_library_hint("MSVCP140.dll was not found", "Windows")
    runner = FakeRunner({"cpu": (127, text)})
    backend = make_backend(tmp_path, installer=FakeInstaller(tmp_path, marker=True), runner=runner)
    with pytest.raises(BackendError) as err:
        backend.prepare(make_ui())
    assert "libgomp1" in str(err.value) and "Visual C++" not in str(err.value)


@pytest.mark.parametrize("winerror, words", [(4551, "Smart App Control"), (1260, "Smart App Control"),
                                              (225, "antivirus"), (226, "antivirus")])
def test_windows_refusing_to_start_the_engine_is_explained(tmp_path, winerror, words):
    """Smart App Control blocks unsigned programs outright (WinError 4551, no "Run anyway"), even ones Steam
    installed; an antivirus quarantine is WinError 225. Say which, and what still works."""
    exc = OSError(22, "An Application Control policy has blocked this file")
    exc.winerror = winerror
    assert words in ls.windows_block_message(exc)
    assert ls.windows_block_message(OSError(2, "No such file")) is None

    def popen(*args, **kwargs):
        raise exc

    backend = make_backend(tmp_path, popen=popen)
    with pytest.raises(BackendError) as err:
        backend.prepare(make_ui())
    assert words in str(err.value) and "Ollama" in str(err.value)


def test_steam_players_are_never_told_to_apt_install_a_library(monkeypatch):
    """SteamOS is read-only, and Steam runs the game in its own Linux runtime, which never sees the computer's
    libraries: a missing library there means Steam's file check, not a package."""
    text = "./llama-server: error while loading shared libraries: libssl.so.3: cannot open shared object file"
    steam = ls.missing_library_hint(text, "Linux", env={"SteamAppId": "480"})
    assert "apt" not in steam and "Verify integrity of game files" in steam and "libssl.so.3" in steam
    # The built-in engine ships its own OpenSSL: missing, it means an incomplete download of the game.
    bundled = ls.missing_library_hint(text, "Linux", env={}, bundled=True)
    assert "apt" not in bundled and "comes with the game" in bundled
    # A developer copy's downloaded engine on a desktop Linux: the package is the fix.
    assert "sudo apt install libssl3" in ls.missing_library_hint(text, "Linux", env={})
    gomp = "error while loading shared libraries: libgomp.so.1: cannot open shared object file"
    assert "libgomp1" in ls.missing_library_hint(gomp, "Linux", env={}, bundled=True)


def test_a_newer_engine_that_cant_run_here_is_explained_and_the_old_one_kept(tmp_path):
    """A model needs a newer engine; the newest build needs a newer Linux. Only that
    install is marked, and the player hears the real reason."""
    unsupported = "llama_model_load: error loading model: unknown model architecture: 'qwen9'\n"
    installer = FakeInstaller(tmp_path, marker=True, update_to="b9000")
    runner = FakeRunner({"b9000-cpu": GLIBC_FAIL})
    popen = FakePopen((unsupported, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen, installer=installer, runner=runner)
    with pytest.raises(BackendError) as err:
        backend.prepare(make_ui())
    message = str(err.value)
    assert "needs a newer llama.cpp engine" in message and "still works for other models" in message
    assert len(popen.calls) == 1  # the broken new engine was never started with the model
    new_marker = json.loads((tmp_path / "engines" / "b9000-cpu" / "install.json").read_text())
    old_marker = json.loads((tmp_path / "engines" / "cpu" / "install.json").read_text())
    assert new_marker["unusable"]["reason"] == "glibc" and "unusable" not in old_marker


def test_disk_reads_of_a_real_process_can_be_measured():
    import os as _os

    class Me:
        pid = _os.getpid()

    value = ls._process_read_bytes(Me())
    assert value is None or value >= 0
    assert ls._process_read_bytes(object()) is None


# ---------------------------------------------------------------------------
# The built game: the engine ships inside the game and is never downloaded
# ---------------------------------------------------------------------------

AMD = GPUInfo(name="AMD Radeon RX 7800 XT", vendor="amd", vram_gb=16.0)
APPLE = GPUInfo(name="Apple M2", vendor="apple", vram_gb=11.2)
VULKAN_CRASH_LOG = "ggml_vulkan: Found 1 Vulkan devices:\nggml_vulkan: vk::Device::createComputePipeline: ErrorDeviceLost\n"
METAL_CRASH_LOG = "ggml_metal_init: error: failed to create command queue\n"
BUNDLE_ASSETS = {"vulkan": "llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", "cpu": "llama-{tag}-bin-ubuntu-x64.tar.gz",
                 "metal": "llama-{tag}-bin-macos-arm64.tar.gz"}


def built_game(monkeypatch, tmp_path, *variants: str, tag: str = "b7000") -> dict:
    """A built game (downloads off) whose engine folder holds fake builds; returns {variant: exe}."""
    engine = tmp_path / "game" / "engine"
    engine.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("GETTOWORK_ENGINE_DIR", str(engine))
    monkeypatch.setenv("GETTOWORK_ALLOW_ENGINE_DOWNLOAD", "0")
    distribution.load(refresh=True)
    exes = {}
    for variant in variants:
        folder = engine / f"{tag}-{variant}"
        folder.mkdir()
        (folder / "llama-server").write_bytes(b"#!engine")
        asset = BUNDLE_ASSETS[variant].format(tag=tag)
        (folder / "install.json").write_text(json.dumps({"tag": tag, "variant": variant, "exe": "llama-server",
                                                         "assets": [asset], "bundled": True}))
        exes[variant] = folder / "llama-server"
    return exes


@pytest.fixture
def no_engine_downloads(monkeypatch):
    """Fail the test if the engine installer tries to reach GitHub."""
    def refuse(*args, **kwargs):
        raise AssertionError("a built game must never download the engine")

    monkeypatch.setattr(runtime_install, "UrllibHttp", refuse)
    monkeypatch.setattr(runtime_install, "fetch_releases", refuse)
    monkeypatch.setattr(runtime_install, "fetch_release", refuse)


def test_built_game_vulkan_crash_falls_back_to_the_bundled_cpu_build(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    before = (exes["vulkan"].parent / "install.json").read_bytes()
    specs = make_specs(gpus=[AMD], flags=["vulkan"])  # plan: vulkan, cpu
    popen = FakePopen((VULKAN_CRASH_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=specs, installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    backend.prepare(ui)

    first, second = popen.args
    assert Path(first[0]).resolve() == exes["vulkan"].resolve() and "-ngl" not in first
    assert Path(second[0]).resolve() == exes["cpu"].resolve() and second[-4:] == ["--device", "none", "-ngl", "0"]
    assert backend.variant is CPU and backend.cpu_only
    text = " ".join(output(ui).split())
    assert "built-in llama.cpp engine (b7000, Vulkan build)" in text
    assert "graphics-driver hiccup" in text
    # A driver hiccup isn't a permanent verdict, and the game's own files are never changed.
    assert (exes["vulkan"].parent / "install.json").read_bytes() == before
    assert not (tmp_path / "home" / "runtime" / runtime_install.BUNDLED_UNUSABLE_FILE).exists()
    backend.close()


def test_built_game_metal_crash_uses_cpu_mode_on_the_same_build(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "metal")
    mac = make_specs("Darwin", "arm64", gpus=[APPLE])  # plan: metal, cpu
    popen = FakePopen((METAL_CRASH_LOG, FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=mac, installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    backend.prepare(ui)
    first, second = popen.args
    assert Path(first[0]).resolve() == Path(second[0]).resolve() == exes["metal"].resolve()
    assert "--device" not in first
    assert second[-4:] == ["--device", "none", "-ngl", "0"]  # the Metal build, on the processor
    assert backend.variant is CPU and backend.cpu_only
    assert "Apple Metal build, in CPU mode" in " ".join(output(ui).split())
    backend.close()


def test_built_game_gpu_crash_on_the_warm_up_moves_to_the_bundled_cpu_build(tmp_path, monkeypatch,
                                                                           no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()), ("", FakeProcess()))
    http = FakeServerHttp(health=(200,), chat=[ConnectionResetError("reset"),
                                               chat_reply("1, 2, 3", timings={"predicted_per_second": 9.0})])
    backend = make_backend(tmp_path, popen=popen, http=http, specs=make_specs(gpus=[AMD], flags=["vulkan"]),
                           installer=runtime_install.ensure_llama_server)
    backend.prepare(make_ui())
    assert backend.variant is VULKAN
    http.on_chat = lambda: popen.processes[0].crash(-6) if len(http.chat_bodies) == 1 else None
    assert backend.benchmark(make_ui()) == 9.0
    assert Path(popen.args[1][0]).resolve() == exes["cpu"].resolve()
    assert backend.variant is CPU and backend.cpu_only
    backend.close()


def test_built_game_gpu_crash_mid_game_switches_to_cpu_mode_without_downloading(tmp_path, monkeypatch,
                                                                              no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    notices: list[str] = []
    popen = FakePopen(("offloaded 37/37 layers to GPU\n", FakeProcess()))
    http = FakeServerHttp(health=(200,), chat=[ConnectionResetError("reset"), chat_reply("The goose bows.")])
    backend = make_backend(tmp_path, popen=popen, http=http, specs=make_specs(gpus=[AMD], flags=["vulkan"]),
                           installer=runtime_install.ensure_llama_server)
    backend.prepare(make_ui())
    backend.on_notice = notices.append
    http.on_chat = lambda: popen.processes[0].crash(-6) if len(http.chat_bodies) == 1 else None
    assert backend.chat([{"role": "user", "content": "hi"}]).text == "The goose bows."
    assert backend.cpu_only and Path(popen.args[-1][0]).resolve() == exes["vulkan"].resolve()
    assert popen.args[-1][-4:] == ["--device", "none", "-ngl", "0"]
    assert any("CPU mode" in n for n in notices)
    backend.close()


def test_built_game_with_its_engine_missing_says_how_to_repair_it(tmp_path, monkeypatch, no_engine_downloads):
    built_game(monkeypatch, tmp_path)  # an empty engine folder
    ok, why = make_backend(tmp_path).is_available()
    assert not ok and why == runtime_install.ENGINE_MISSING_MESSAGE
    downloader = FakeDownloader(tmp_path)
    backend = make_backend(tmp_path, installer=runtime_install.ensure_llama_server, downloader=downloader)
    with pytest.raises(BackendError, match="Verify integrity"):
        backend.prepare(make_ui())
    assert downloader.calls == []  # the model isn't downloaded for an engine that isn't there


def test_built_game_is_available_names_the_built_in_engine(tmp_path, monkeypatch):
    built_game(monkeypatch, tmp_path, "cpu")
    ok, why = make_backend(tmp_path).is_available()
    assert ok and "built-in llama.cpp engine is ready (b7000, cpu)" in why


def test_built_game_permanent_failure_is_noted_outside_the_game(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    before = (exes["vulkan"].parent / "install.json").read_bytes()
    runner = FakeRunner({"b7000-vulkan": GLIBC_FAIL})
    popen = FakePopen()
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[AMD], flags=["vulkan"]),
                           installer=runtime_install.ensure_llama_server, runner=runner)
    backend.prepare(make_ui())
    assert Path(popen.args[0][0]).resolve() == exes["cpu"].resolve()
    assert (exes["vulkan"].parent / "install.json").read_bytes() == before
    notes = json.loads((tmp_path / "home" / "runtime" / runtime_install.BUNDLED_UNUSABLE_FILE).read_text())
    assert notes["builds"]["b7000-vulkan"]["reason"] == "glibc"
    # Next launch: the Vulkan build is skipped straight away.
    assert [e for e, _t, _v in runtime_install.installed_runtimes()] == [exes["cpu"]]
    backend.close()


def test_a_saved_engine_from_a_moved_game_is_found_again(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    old = tmp_path / "OldLibrary" / "steamapps" / "common" / "GetToWork" / "engine" / "b7000-cpu" / "llama-server"
    model = tmp_path / "m.gguf"
    model.write_bytes(b"GGUF")
    installer = FakeInstaller(tmp_path, fail={"cpu", "vulkan", "cuda-12"})  # must not be needed
    popen = FakePopen()
    backend = make_backend(tmp_path, popen=popen, installer=installer, server_exe=old, model_path=model)
    ui = make_ui()
    backend.prepare(ui)
    assert installer.calls == []
    assert backend.server_exe == exes["cpu"] and backend.variant is CPU
    text = " ".join(output(ui).split())
    assert "The game has moved since last time" in text and "gone missing" not in text
    backend.close()


def test_a_lost_saved_engine_in_a_built_game_uses_the_built_in_one(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "cpu")
    backend = make_backend(tmp_path, installer=runtime_install.ensure_llama_server,
                           server_exe=tmp_path / "gone" / "llama-server")
    ui = make_ui()
    backend.prepare(ui)
    assert backend.server_exe == exes["cpu"]
    text = " ".join(output(ui).split())
    assert "let me find the game's built-in one" in text and "fetch a fresh copy" not in text
    backend.close()


def test_a_model_too_new_for_the_built_in_engine_never_triggers_an_update(tmp_path, monkeypatch, no_engine_downloads):
    built_game(monkeypatch, tmp_path, "cpu")
    log = "unknown model architecture: 'qwen99'\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen, installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    with pytest.raises(BackendError) as err:
        backend.prepare(ui)
    message = str(err.value)
    assert "game's built-in llama.cpp engine doesn't know yet" in message and "qwen99" in message
    assert "game updates bring newer engines" in message
    assert "fetch the newest engine" not in output(ui)


VULKAN_LISTING = "Available devices:\n  Vulkan0: AMD Radeon RX 7800 XT (16384 MiB, 15000 MiB free)\n"


def _session(tmp_path, *, runner, server_exe=None, specs=None, popen=None):
    """One launch of a built game: a fresh backend (as setup_flow makes it), prepared."""
    backend = make_backend(tmp_path, popen=popen or FakePopen(("", FakeProcess())), runner=runner,
                           specs=specs or make_specs(gpus=[AMD], flags=["vulkan"]), server_exe=server_exe,
                           installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    backend.prepare(ui)
    return backend, " ".join(output(ui).split())


def test_after_a_driver_fix_a_later_launch_goes_back_to_the_graphics_build(tmp_path, monkeypatch,
                                                                          no_engine_downloads):
    """Session 1: the Vulkan build sees no graphics card, so the game switches to its CPU build (and saves it).
    The player updates the driver. Session 2 (welcome back, same saved engine): the Vulkan build is asked
    again - it sees the card now - and is used, instead of the CPU build forever."""
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    first, said = _session(tmp_path, runner=DeviceRunner({}))
    assert first.server_exe == exes["cpu"] and "updating the graphics driver usually fixes this" in said
    first.close()
    note = json.loads((tmp_path / "home" / "runtime" / ls.GPU_SWITCH_FILE).read_text())
    assert note["reason"] == "no_device" and Path(note["from"]) == exes["vulkan"].resolve()

    # Still no card: stays on the CPU build - and says why, every launch.
    again, said = _session(tmp_path, runner=DeviceRunner({}), server_exe=exes["cpu"])
    assert again.server_exe == exes["cpu"] and "still can't find your graphics card" in said
    again.close()

    fixed, said = _session(tmp_path, runner=DeviceRunner({"b7000-vulkan": VULKAN_LISTING}), server_exe=exes["cpu"])
    assert fixed.server_exe == exes["vulkan"] and fixed.variant is VULKAN and not fixed.cpu_only
    assert "can see your graphics card again" in said
    assert not (tmp_path / "home" / "runtime" / ls.GPU_SWITCH_FILE).exists()
    fixed.close()


def _note_switch(tmp_path, backend_exes, reason, *, age_s=60.0, fingerprint=None):
    from gettowork import __version__

    note = {"from": str(backend_exes["vulkan"].resolve()), "to": str(backend_exes["cpu"].resolve()),
            "reason": reason, "at": time.time() - age_s,
            "fingerprint": fingerprint or {"game": __version__, "gpus": [f"{AMD.name}|"]}}
    path = tmp_path / "home" / "runtime" / ls.GPU_SWITCH_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(note))


def test_a_warm_up_crash_switch_is_retried_after_a_driver_change_or_a_while(tmp_path, monkeypatch,
                                                                           no_engine_downloads):
    """A one-off VRAM shortage or driver hiccup made the game switch to the CPU build. It isn't retried on
    every launch (a permanent problem would crash every warm-up), but after a driver or game update - or a
    week - the graphics build gets another go."""
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    runner = DeviceRunner({"b7000-vulkan": VULKAN_LISTING})
    _note_switch(tmp_path, exes, "memory")
    stay, said = _session(tmp_path, runner=runner, server_exe=exes["cpu"])
    assert stay.server_exe == exes["cpu"] and "ran out of memory" in said and "tries the graphics card again" in said
    stay.close()

    _note_switch(tmp_path, exes, "crash", age_s=8 * 24 * 3600)
    retry, said = _session(tmp_path, runner=runner, server_exe=exes["cpu"])
    assert retry.server_exe == exes["vulkan"] and "It's been a while - trying the graphics card again" in said
    retry.close()

    driver = dataclasses.replace(AMD, driver_version="24.3.1")
    _note_switch(tmp_path, exes, "gpu_hang")  # the player chose to leave a hung start: only a change retries
    stay, _said = _session(tmp_path, runner=runner, server_exe=exes["cpu"],
                           specs=make_specs(gpus=[AMD], flags=["vulkan"]))
    assert stay.server_exe == exes["cpu"]
    stay.close()
    _note_switch(tmp_path, exes, "gpu_hang", age_s=30 * 24 * 3600)
    stay, _said = _session(tmp_path, runner=runner, server_exe=exes["cpu"])
    assert stay.server_exe == exes["cpu"]
    stay.close()
    retry, said = _session(tmp_path, runner=runner, server_exe=exes["cpu"],
                           specs=make_specs(gpus=[driver], flags=["vulkan"]))
    assert retry.server_exe == exes["vulkan"] and "Your graphics driver or the game has been updated since" in said
    retry.close()


def test_a_crash_during_the_warm_up_notes_the_switch_to_the_cpu_build(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    backend = make_backend(tmp_path, popen=FakePopen(("", FakeProcess()), ("", FakeProcess())),
                           runner=DeviceRunner({"b7000-vulkan": VULKAN_LISTING}),
                           specs=make_specs(gpus=[AMD], flags=["vulkan"]), installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    backend.prepare(ui)
    assert backend.server_exe == exes["vulkan"]
    assert backend._switch_after_crash(ui, "memory") is True
    assert backend.server_exe == exes["cpu"]
    note = json.loads((tmp_path / "home" / "runtime" / ls.GPU_SWITCH_FILE).read_text())
    assert note["reason"] == "memory" and Path(note["to"]) == exes["cpu"].resolve()
    backend.close()


def test_a_saved_cpu_build_the_player_never_switched_to_stays(tmp_path, monkeypatch, no_engine_downloads):
    """No note (e.g. a computer without a usable graphics card from the start): nothing is retried."""
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    backend, said = _session(tmp_path, runner=DeviceRunner({"b7000-vulkan": VULKAN_LISTING}),
                             server_exe=exes["cpu"])
    assert backend.server_exe == exes["cpu"] and "graphics card again" not in said
    backend.close()


def _other_copys_engine(tmp_path, tag="b7000", variant="cpu"):
    """An engine inside another (older) copy of the game, still on disk."""
    folder = tmp_path / "gameOld" / "engine" / f"{tag}-{variant}"
    folder.mkdir(parents=True)
    (folder / "llama-server").write_bytes(b"#!old engine")
    asset = BUNDLE_ASSETS[variant].format(tag=tag)
    (folder / "install.json").write_text(json.dumps({"tag": tag, "variant": variant, "exe": "llama-server",
                                                     "assets": [asset], "bundled": True}))
    return folder / "llama-server"


def test_a_new_build_uses_its_own_engine_not_the_one_saved_by_an_older_copy(tmp_path, monkeypatch,
                                                                            no_engine_downloads):
    """Welcome back in a newly downloaded build: settings still name the older copy's engine,
    which is still on disk. The new build must run the engine it ships."""
    exes = built_game(monkeypatch, tmp_path, "cpu", tag="b7100")
    old = _other_copys_engine(tmp_path)
    popen = FakePopen(("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, installer=runtime_install.ensure_llama_server, server_exe=old)
    ui = make_ui()
    backend.prepare(ui)
    assert backend.server_exe == exes["cpu"]
    assert Path(popen.args[0][0]).resolve() == exes["cpu"].resolve()
    assert "this copy of the game's own built-in llama.cpp engine" in " ".join(output(ui).split())
    backend.close()


def test_a_model_too_new_for_an_old_saved_engine_tries_the_games_newer_one_offline(tmp_path, monkeypatch,
                                                                                  no_engine_downloads):
    """The engine that failed is older than the one this game ships (another copy's, saved last time):
    the newest build already on disk is tried - no download - before saying "game updates bring newer engines"."""
    exes = built_game(monkeypatch, tmp_path, "cpu", tag="b7100")
    old = _other_copys_engine(tmp_path)
    log = "unknown model architecture: 'qwen9'\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, installer=runtime_install.ensure_llama_server)
    backend.server_exe = old  # (as if it had been kept from last time)
    backend._ensure_engine = lambda ui: None  # skip the swap above: exercise the fallback on its own
    backend.variant = CPU
    ui = make_ui()
    backend.prepare(ui)
    assert Path(popen.args[0][0]).resolve() == old.resolve()
    assert Path(popen.args[1][0]).resolve() == exes["cpu"].resolve()
    assert backend.server_exe == exes["cpu"]
    assert "trying the game's built-in engine (b7100) instead" in " ".join(output(ui).split())
    backend.close()


def test_a_newer_engine_that_isnt_the_games_is_never_called_the_games(tmp_path, monkeypatch, no_engine_downloads):
    """A saved CUDA build from a developer copy (this game ships none) can't load the model; a newer CUDA build
    the developer copy also downloaded is tried - and named for what it is, not "the newest engine the game has"."""
    built_game(monkeypatch, tmp_path, "vulkan", "cpu", tag="b7100")
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    exes = {}
    for tag in ("b7000", "b7200"):
        folder = home / "runtime" / "llama.cpp" / f"{tag}-cuda-12"
        folder.mkdir(parents=True)
        (folder / "llama-server").write_bytes(b"#!dev engine")
        (folder / "install.json").write_text(json.dumps({"tag": tag, "variant": "cuda-12", "exe": "llama-server",
                                                         "assets": [f"llama-{tag}-bin-ubuntu-cuda-12-x64.tar.gz"]}))
        exes[tag] = folder / "llama-server"
    log = "unknown model architecture: 'qwen9'\nfailed to load model\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)), ("", FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA]),
                           installer=runtime_install.ensure_llama_server)
    backend.server_exe, backend.variant = exes["b7000"], runtime_install.CUDA12
    backend._ensure_engine = lambda ui: None
    ui = make_ui()
    backend.prepare(ui)
    assert Path(popen.args[1][0]).resolve() == exes["b7200"].resolve()
    said = " ".join(output(ui).split())
    assert "trying another llama.cpp engine already on this computer (b7200) instead" in said
    assert "the game has" not in said
    backend.close()


def test_when_the_games_newest_engine_is_the_one_that_failed_it_says_game_updates_bring_newer_ones(
        tmp_path, monkeypatch, no_engine_downloads):
    built_game(monkeypatch, tmp_path, "cpu", tag="b7100")
    log = "unknown model architecture: 'qwen9'\n"
    popen = FakePopen((log, FakeProcess(dies_with=1)))
    backend = make_backend(tmp_path, popen=popen, installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    with pytest.raises(BackendError) as err:
        backend.prepare(ui)
    message = str(err.value)
    assert "game updates bring newer engines" in message
    # No line before the error suggesting a newer engine was tried: none was (the same one failed).
    shown = " ".join(output(ui).split())
    assert "newest llama.cpp engine in this copy" not in shown and "trying the newest engine" not in shown
    assert "even the newest" not in message and "when you're online" not in message
    assert len(popen.calls) == 1


def test_a_mac_whose_metal_build_cant_run_isnt_told_to_verify_its_files(tmp_path, monkeypatch):
    exes = built_game(monkeypatch, tmp_path, "metal")
    monkeypatch.setattr(ls.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(ls.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(runtime_install, "_macos_version", lambda: (14, 0))
    runtime_install.mark_unusable(exes["metal"], "cpu_unsupported")
    ok, why = LlamaServerBackend(ENTRY, log_dir=tmp_path / "logs").is_available()
    assert not ok and "Verify integrity" not in why and "can't run on this computer" in why
    ok, why = make_backend(tmp_path, specs=make_specs("Darwin", "arm64", gpus=[APPLE])).is_available()
    assert not ok and "Verify integrity" not in why


def test_the_fallback_list_in_a_built_game_skips_builds_it_doesnt_have(tmp_path, monkeypatch):
    specs = make_specs(gpus=[NVIDIA], flags=["vulkan"])  # plan: cuda-12, vulkan, cpu
    backend = make_backend(tmp_path, specs=specs)
    assert backend._fallback_variants(CUDA12) == [VULKAN, CPU]  # a developer copy can download either
    built_game(monkeypatch, tmp_path, "cpu")
    assert backend._fallback_variants(CUDA12) == [CPU]  # no Vulkan build inside the game
    assert backend._fallback_variants(VULKAN) == [CPU]


def test_built_game_picks_vulkan_when_the_plan_starts_with_cuda(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "vulkan", "cpu")
    popen = FakePopen((REAL_B11100_LOG + REAL_B11100_READY, FakeProcess()))
    backend = make_backend(tmp_path, popen=popen, specs=make_specs(gpus=[NVIDIA], flags=["vulkan"]),
                           installer=runtime_install.ensure_llama_server)
    backend.prepare(make_ui())
    assert Path(popen.args[0][0]).resolve() == exes["vulkan"].resolve()
    assert backend.variant is VULKAN
    backend.close()


def test_a_bundled_engine_in_use_is_never_tidied_away(tmp_path, monkeypatch, no_engine_downloads):
    exes = built_game(monkeypatch, tmp_path, "cpu")
    older = tmp_path / "home" / "runtime" / "llama.cpp" / "b6000-cpu"
    older.mkdir(parents=True)
    (older / "llama-server").write_bytes(b"x" * 2_000_000)
    (older / "install.json").write_text(json.dumps({"tag": "b6000", "variant": "cpu", "exe": "llama-server"}))
    backend = make_backend(tmp_path, installer=runtime_install.ensure_llama_server)
    ui = make_ui()
    backend.prepare(ui)
    assert backend.server_exe == exes["cpu"] and exes["cpu"].exists()
    assert not older.exists()  # the player's old download is tidied; the game's own copy stays
    assert "Tidied away 1 engine copy" in output(ui)
    backend.close()
