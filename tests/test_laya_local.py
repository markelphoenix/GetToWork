"""Local Laya: fit rules, and an install/start that never touches the network."""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from gettowork.laya_local import (
    HEALTH_TIMEOUT_S,
    LAYA_PACKAGE,
    MIN_FREE_DISK_GB,
    MIN_FREE_RAM_GB,
    LocalLaya,
    LocalLayaError,
    assess_specs,
    venv_python,
)
from gettowork.types import GPUInfo, SystemSpecs
from gettowork.ui import UI


def specs(**kwargs) -> SystemSpecs:
    base = dict(
        os_name="Linux", os_version="1", arch="x86_64", cpu_name="Test CPU",
        cpu_cores_physical=4, cpu_cores_logical=8, ram_total_gb=16, ram_available_gb=8,
        disk_free_gb=40, gpus=[],
    )
    base.update(kwargs)
    return SystemSpecs(**base)


def test_a_comfortable_computer_can_run_laya_on_the_cpu():
    fit = assess_specs(specs())
    assert fit.ok and fit.device == "cpu" and fit.threads == 4


def test_a_graphics_card_does_not_lower_the_memory_bar():
    """The install uses the normal PyTorch build, so the checkpoint still needs host memory."""
    tight = assess_specs(specs(
        ram_available_gb=2, gpus=[GPUInfo(name="RTX", vendor="nvidia", vram_gb=8)],
    ))
    assert not tight.ok
    fit = assess_specs(specs(
        ram_available_gb=4, gpus=[GPUInfo(name="RTX", vendor="nvidia", vram_gb=8)],
    ))
    assert fit.ok and fit.device == "cpu"


def test_apple_silicon_uses_its_graphics_chip():
    fit = assess_specs(specs(os_name="Darwin", unified_memory=True, ram_available_gb=6))
    assert fit.ok and fit.device == "mps"


def test_too_little_free_memory_is_not_offered():
    fit = assess_specs(specs(ram_available_gb=1.2))
    assert not fit.ok and "1 GB" in fit.reason and f"{MIN_FREE_RAM_GB:.0f} GB" in fit.reason


def test_too_little_disk_is_not_offered():
    fit = assess_specs(specs(disk_free_gb=1))
    assert not fit.ok and f"{MIN_FREE_DISK_GB:.0f} GB" in fit.reason


def test_unknown_disk_space_does_not_block_a_machine_with_memory():
    assert assess_specs(specs(disk_free_gb=-1)).ok


def test_no_python_hides_the_option_even_when_the_hardware_fits():
    fit = LocalLaya(python_finder=lambda: None).fit(specs())
    assert not fit.ok and "Python 3.10" in fit.reason


class Proc:
    def __init__(self, *, dead: bool = False) -> None:
        self.returncode = 1 if dead else None
        self.pid = 424242
        self.stopped = False

    def poll(self):
        return self.returncode

    def stop(self):
        self.stopped = True
        self.returncode = -15

    def wait(self, timeout=None):
        self.returncode = -15
        return self.returncode

    def terminate(self):
        self.stop()

    def kill(self):
        self.returncode = -9


def _ui() -> UI:
    return UI(console=Console(file=io.StringIO(), width=120, color_system=None, force_terminal=False))


def test_start_creates_an_environment_then_serves_on_localhost(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    commands: list[list[str]] = []

    def runner(args, env):
        commands.append(list(args))
        if args[1:3] == ["-m", "venv"]:
            py = venv_python(__import__("pathlib").Path(args[3]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        return 0, ""

    launched: list[tuple] = []
    procs: list[Proc] = []

    def popen(args, **kwargs):
        launched.append((args, kwargs))
        proc = Proc()
        procs.append(proc)
        return proc

    local = LocalLaya(
        python_finder=lambda: "/usr/bin/python3", runner=runner, popen=popen,
        health=lambda url: url == "http://127.0.0.1:8765/health", pick_port=lambda: 8765,
        sleep=lambda _s: None, clock=lambda: 0.0,
    )
    client = local.start(_ui(), device="cpu", threads=4)
    assert client.base_url == "http://127.0.0.1:8765" and client.model == "english"
    assert any(LAYA_PACKAGE in arg for command in commands for arg in command)
    env = launched[0][1]["env"]
    assert env["LAYA_HOST"] == "127.0.0.1" and env["LAYA_PORT"] == "8765"
    assert env["LAYA_MODELS"] == "english" and env["LAYA_DEVICE"] == "cpu"
    assert env["LAYA_API_KEY"].startswith("laya_local_")
    assert env["LAYA_API_KEY"] in client.secret_values()
    assert "TYPESAFE_API_KEY" not in env
    assert local.installed()
    local.close()
    assert procs[0].stopped

    # A second start reuses the install.
    commands.clear()
    local.start(_ui(), device="cpu", threads=4)
    assert not any(LAYA_PACKAGE in arg for command in commands for arg in command)
    local.close()
    assert procs[1].stopped


def test_a_server_that_exits_is_explained(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))

    def runner(args, env):
        if args[1:3] == ["-m", "venv"]:
            py = venv_python(__import__("pathlib").Path(args[3]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        return 0, ""

    local = LocalLaya(
        python_finder=lambda: "/usr/bin/python3", runner=runner, popen=lambda *a, **k: Proc(dead=True),
        health=lambda url: False, pick_port=lambda: 9, sleep=lambda _s: None, clock=lambda: 0.0,
    )
    (tmp_path / "runtime" / "laya").mkdir(parents=True)  # start() creates it too
    with pytest.raises(LocalLayaError) as info:
        local.start(_ui(), device="cpu", threads=1)
    assert "stopped while starting" in info.value.message


def test_giving_up_on_a_slow_start_names_the_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    ticks = {"n": 0}

    def clock():
        ticks["n"] += 1
        return 0.0 if ticks["n"] < 4 else HEALTH_TIMEOUT_S + 5

    def runner(args, env):
        if args[1:3] == ["-m", "venv"]:
            py = venv_python(__import__("pathlib").Path(args[3]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
        return 0, ""

    local = LocalLaya(
        python_finder=lambda: "/usr/bin/python3", runner=runner, popen=lambda *a, **k: Proc(),
        health=lambda url: False, pick_port=lambda: 9, sleep=lambda _s: None, clock=clock,
    )
    with pytest.raises(LocalLayaError) as info:
        local.start(_ui(), device="cpu", threads=1)
    assert "didn't finish starting" in info.value.message
    local.close()


def test_a_failed_install_keeps_the_pip_error(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))

    def runner(args, env):
        if args[1:3] == ["-m", "venv"]:
            py = venv_python(__import__("pathlib").Path(args[3]))
            py.parent.mkdir(parents=True, exist_ok=True)
            py.write_text("")
            return 0, ""
        return 1, "No matching distribution\nNo space left on device"

    local = LocalLaya(python_finder=lambda: "/usr/bin/python3", runner=runner)
    with pytest.raises(LocalLayaError) as info:
        local.start(_ui(), device="cpu", threads=1)
    assert "No space left on device" in info.value.message
