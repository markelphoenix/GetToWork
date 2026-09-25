"""Tests for gettowork.backends.llamaserver.

No real processes, network or models: a fake `popen` hands back scripted
process objects (and writes a fake log), and a fake HTTP layer plays the part
of llama-server's /health and /v1/chat/completions endpoints.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest
from rich.console import Console

from gettowork import runtime_install
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


@pytest.fixture(autouse=True)
def linux_host(monkeypatch, tmp_path):
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


def test_posix_launch_uses_its_own_session_and_a_parent_death_signal(tmp_path):
    backend, _http, popen = started(tmp_path)
    kwargs = popen.calls[0][1]
    assert kwargs["start_new_session"] is True  # terminal Ctrl+C doesn't reach the engine
    assert callable(kwargs.get("preexec_fn"))  # Linux: the kernel stops it if the game dies
    backend.close()


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
    import os

    exe = _sleeper_named_llama_server(tmp_path)
    backend = make_backend(tmp_path, popen=lambda args, **kw: subprocess.Popen([str(exe), "30"], **kw))
    backend._launch(exe, tmp_path / "m.gguf", cpu_only=True, minimal=False)
    try:
        assert os.getpgid(backend._proc.pid) != os.getpgid(0)
    finally:
        backend.close()


@pytest.mark.skipif(sys.platform != "linux", reason="uses a real Linux process")
def test_orphaned_engine_from_a_force_quit_game_is_stopped(tmp_path):
    import os

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


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")
def test_log_in_use_by_another_game_gets_its_own_file(tmp_path):
    import fcntl

    logs = tmp_path / "logs"
    logs.mkdir()
    other = open(logs / "llama-server.log", "ab")  # another copy of the game, still running
    fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    other.write(b"the other game's engine is busy\n")
    other.flush()
    try:
        backend, _http, _ = started(tmp_path)
        assert backend.log_path == logs / "llama-server-50123.log"
        assert (logs / "llama-server.log").read_bytes() == b"the other game's engine is busy\n"  # untouched
        backend.close()
    finally:
        other.close()


def test_health_timeout_grows_with_the_model_size():
    assert ls.health_timeout_for(2.5) == ls.HEALTH_TIMEOUT_S
    assert ls.health_timeout_for(63.4) == pytest.approx(63.4 * 15)
    assert ls.health_timeout_for(500) == ls.HEALTH_MAX_TIMEOUT_S


def test_big_model_file_gets_a_longer_wake_up_time(tmp_path):
    model = tmp_path / "big.gguf"
    with open(model, "wb") as fh:
        fh.truncate(40 * 10**9)  # sparse: 40 GB on paper, nothing on disk
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
    monkeypatch.setattr(ls, "_posix_locks_available", lambda: False)
    first = make_backend(tmp_path)
    first.prepare(make_ui())
    first_log = first.log_path
    first_log.write_bytes(b"first game's engine output\n")
    second = make_backend(tmp_path)
    second.prepare(make_ui())
    assert second.log_path != first_log
    assert first_log.read_bytes() == b"first game's engine output\n"  # not truncated
    assert first_log.name.startswith("llama-server-") and second.log_path.name.startswith("llama-server-")


def test_lock_helper_says_no_without_flock(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_fcntl(name, *a, **k):
        if name == "fcntl":
            raise ImportError("no fcntl on Windows")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", no_fcntl)
    assert ls._lock_exclusively(io.BytesIO()) is False
    assert ls._posix_locks_available() is False


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


def test_a_split_model_gets_time_to_load_every_part(tmp_path):
    folder = tmp_path / "split"
    folder.mkdir()
    for part in (1, 2, 3):
        with open(folder / f"big-Q4_K_M-0000{part}-of-00003.gguf", "wb") as fh:
            fh.truncate(15 * 10**9 if part < 3 else 5 * 10**9)  # sparse: no real disk space used
    first = folder / "big-Q4_K_M-00001-of-00003.gguf"
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
