"""Tests for gettowork.specs (hardware detection).

Nothing here touches the real machine's tools: `subprocess.run`, `platform`,
`psutil`, /proc and /sys are all replaced with fakes describing imaginary
computers.
"""

from __future__ import annotations

import os
import subprocess
from collections import namedtuple
from pathlib import Path

import pytest

from gettowork import perf, specs
from gettowork.types import GPUInfo, SystemSpecs

GIB = 1024**3
VirtualMemory = namedtuple("VirtualMemory", "total available")
DiskUsage = namedtuple("DiskUsage", "total used free")

NVIDIA_3060 = "NVIDIA GeForce RTX 3060, 12288, 550.54.14\n"
LSPCI_NVIDIA_LAPTOP = (
    "00:00.0 Host bridge: Intel Corporation Device 4641 (rev 02)\n"
    "00:02.0 VGA compatible controller: Intel Corporation Alder Lake-P GT2 [Iris Xe Graphics] (rev 0c)\n"
    "01:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate] (rev a1)\n"
)
CPUINFO_X86 = (
    "processor\t: 0\nvendor_id\t: AuthenticAMD\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\n"
    "flags\t\t: fpu vme sse sse2 avx f16c fma avx2 bmi2 sha_ni\n\n"
    "processor\t: 1\nmodel name\t: AMD Ryzen 7 5800X 8-Core Processor\nflags\t\t: fpu avx avx2\n"
)


class FakeRun:
    """Stands in for subprocess.run. `outputs` maps a program name to:
    a string (stdout, exit code 0), an (exit_code, stdout) tuple, an exception
    to raise, or a callable(args) returning one of those. Unknown programs
    raise FileNotFoundError, exactly like a missing executable."""

    def __init__(self, outputs: dict):
        self.outputs = outputs
        self.calls: list[list[str]] = []

    def __call__(self, args, **kwargs):
        assert isinstance(args, list), "commands must be passed as a list (no shell)"
        assert not kwargs.get("shell"), "never use shell=True"
        assert 0 < kwargs.get("timeout", 999) <= 5, "every command needs a timeout of at most 5 s"
        self.calls.append(list(args))
        value = self.outputs.get(args[0])
        if callable(value) and not isinstance(value, BaseException):
            value = value(args)
        if value is None:
            raise FileNotFoundError(args[0])
        if isinstance(value, BaseException):
            raise value
        code, out = value if isinstance(value, tuple) else (0, value)
        return subprocess.CompletedProcess(args, code, stdout=out, stderr="")


@pytest.fixture
def machine(monkeypatch, tmp_path):
    """Build an imaginary computer. Returns a function that applies the settings."""

    def build(
        *,
        system="Linux",
        machine_name="x86_64",
        processor="x86_64",
        ram_gb=16.0,
        available_gb=10.0,
        disk_free_gb=200.0,
        physical=8,
        logical=16,
        commands=None,
        cpuinfo=CPUINFO_X86,
        vulkan=False,
        bandwidth=42.0,
    ):
        monkeypatch.setattr(specs.platform, "system", lambda: system)
        monkeypatch.setattr(specs.platform, "machine", lambda: machine_name)
        monkeypatch.setattr(specs.platform, "processor", lambda: processor)
        monkeypatch.setattr(specs.platform, "release", lambda: "6.8.0" if system == "Linux" else "10")
        monkeypatch.setattr(specs.platform, "version", lambda: "10.0.22631")
        monkeypatch.setattr(specs.platform, "mac_ver", lambda: ("14.5", ("", "", ""), "arm64"))
        monkeypatch.setattr(specs.platform, "freedesktop_os_release", lambda: {"PRETTY_NAME": "Ubuntu 24.04 LTS"})
        monkeypatch.setattr(
            specs.psutil, "virtual_memory", lambda: VirtualMemory(int(ram_gb * GIB), int(available_gb * GIB))
        )
        monkeypatch.setattr(specs.psutil, "cpu_count", lambda logical=True: 16 if logical else 8)
        if physical is None:
            monkeypatch.setattr(specs.psutil, "cpu_count", lambda logical=True: logical if logical else None)
        monkeypatch.setattr(
            specs.psutil, "disk_usage", lambda path: DiskUsage(1000 * GIB, 0, int(disk_free_gb * GIB))
        )
        fake = FakeRun(commands or {})
        monkeypatch.setattr(specs.subprocess, "run", fake)
        cpuinfo_path = tmp_path / "cpuinfo"
        if cpuinfo is not None:
            cpuinfo_path.write_text(cpuinfo)
        monkeypatch.setattr(specs, "_PROC_CPUINFO", cpuinfo_path)
        drm = tmp_path / "drm"
        drm.mkdir(exist_ok=True)
        monkeypatch.setattr(specs, "_SYSFS_DRM", drm)
        monkeypatch.setattr(specs, "_vulkan_loader_present", lambda os_name: vulkan)
        monkeypatch.setattr(specs, "_windows_registry_cpu_name", lambda: None)
        monkeypatch.setattr(specs, "_windows_registry_vram", lambda: {})
        monkeypatch.setattr(specs, "_windows_cpu_flags", lambda arch: ["avx", "avx2"])
        monkeypatch.setattr(specs.perf, "measure_ram_bandwidth", lambda budget_s=0.3: bandwidth)
        # The real benchmark leaves a note behind (e.g. on a low-memory CI runner); the fake leaves none.
        monkeypatch.setattr(specs.perf, "last_benchmark_note", None)
        return fake

    return build


def add_drm_card(tmp_path: Path, index: int, slot: str, vendor: str, vram_bytes: int | None) -> None:
    """Create /sys/class/drm/cardN/device -> ../devices/<slot> like the kernel does."""
    devices = tmp_path / "devices"
    device = devices / slot
    device.mkdir(parents=True)
    (device / "vendor").write_text(vendor + "\n")
    if vram_bytes is not None:
        (device / "mem_info_vram_total").write_text(f"{vram_bytes}\n")
    card = tmp_path / "drm" / f"card{index}"
    card.mkdir(parents=True)
    try:
        os.symlink(device, card / "device")
    except (OSError, NotImplementedError):  # pragma: no cover - Windows without symlink rights
        pytest.skip("symlinks not available")
    (tmp_path / "drm" / f"card{index}-DP-1").mkdir()  # connectors must be ignored


# ---------------------------------------------------------------------------
# Linux + NVIDIA
# ---------------------------------------------------------------------------


def test_linux_with_nvidia_gpu(machine, tmp_path):
    fake = machine(commands={"nvidia-smi": NVIDIA_3060, "lspci": LSPCI_NVIDIA_LAPTOP}, vulkan=True)
    s = specs.detect_specs(tmp_path / "models")

    assert s.os_name == "Linux" and s.arch == "x86_64"
    assert "Ubuntu 24.04" in s.os_version
    assert s.cpu_name == "AMD Ryzen 7 5800X 8-Core Processor"
    assert (s.cpu_cores_physical, s.cpu_cores_logical) == (8, 16)
    assert s.ram_total_gb == 16.0 and s.ram_available_gb == 10.0
    assert s.disk_free_gb == 200.0
    assert s.ram_bandwidth_gbs == 42.0
    assert {"avx", "avx2", "fma", "f16c"} <= set(s.cpu_flags)
    assert "vulkan" in s.cpu_flags and specs.has_vulkan(s)
    assert not s.unified_memory

    rtx = s.gpus[0]
    assert rtx.vendor == "nvidia" and rtx.name == "NVIDIA GeForce RTX 3060"
    assert rtx.vram_gb == 12.0
    assert rtx.driver_version == "550.54.14"
    assert rtx.bandwidth_gbs == 360
    # The Intel iGPU is listed but has no dedicated memory; NVIDIA isn't listed twice.
    assert [g.vendor for g in s.gpus] == ["nvidia", "intel"]
    assert s.gpus[1].vram_gb == 0.0
    assert s.best_vram_gb == 12.0
    assert s.notes == []
    assert specs._NVIDIA_QUERY in fake.calls

    summary = specs.friendly_summary(s)
    assert "16 GB of RAM" in summary and "an NVIDIA GeForce RTX 3060" in summary and "12 GB of video memory" in summary


def test_multiple_nvidia_gpus_and_na_memory():
    out = "NVIDIA GeForce RTX 4090, 24564, 580.65\nNVIDIA GH200 480GB, [N/A], 580.65\ngarbage line\n"
    gpus = specs._parse_nvidia_smi(out)
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4090", "NVIDIA GH200 480GB"]
    assert gpus[0].vram_gb == 24.0 and gpus[1].vram_gb == 0.0
    assert gpus[0].driver_version == "580.65"


def test_nvidia_smi_missing_but_card_present(machine, tmp_path):
    machine(commands={"lspci": LSPCI_NVIDIA_LAPTOP})
    s = specs.detect_specs(tmp_path)
    nv = [g for g in s.gpus if g.vendor == "nvidia"]
    assert len(nv) == 1 and nv[0].vram_gb == 0.0
    assert "RTX 3060" in nv[0].name
    assert any("driver" in note.lower() for note in s.notes)
    summary = specs.friendly_summary(s)
    assert "an NVIDIA card we couldn't talk to" in summary
    assert "driver" in summary


def test_nvidia_smi_missing_and_no_nvidia_card_is_silent(machine, tmp_path):
    machine(commands={"lspci": "00:02.0 VGA compatible controller: Intel Corporation UHD Graphics 620\n"})
    s = specs.detect_specs(tmp_path)
    assert all("nvidia" not in n.lower() for n in s.notes)
    assert [g.vendor for g in s.gpus] == ["intel"]
    assert s.gpus[0].vram_gb == 0.0
    assert any("integrated" in n.lower() for n in s.notes)


def test_nvidia_smi_failing(machine, tmp_path):
    machine(commands={"nvidia-smi": (9, "NVIDIA-SMI has failed because it couldn't communicate with the driver")})
    s = specs.detect_specs(tmp_path)
    assert s.gpus == []
    assert any("nvidia-smi" in note for note in s.notes)


def test_nvidia_smi_timeout(machine, tmp_path):
    machine(commands={"nvidia-smi": subprocess.TimeoutExpired(["nvidia-smi"], 5)})
    s = specs.detect_specs(tmp_path)
    assert s.gpus == []
    assert any("nvidia-smi" in note for note in s.notes)


def test_garbage_everywhere_never_crashes(machine, tmp_path):
    machine(
        commands={
            "nvidia-smi": "\x00\x01 total nonsense without commas\n",
            "lspci": "!!!\n\x00",
            "rocm-smi": "{not json",
            "lscpu": "???",
        },
        cpuinfo="\x00garbage\nflags : \n",
        processor="",
    )
    s = specs.detect_specs(tmp_path)
    assert isinstance(s, SystemSpecs)
    assert s.cpu_name == "Unknown CPU"
    assert s.gpus == []
    assert any("couldn't read" in note for note in s.notes)
    assert specs.friendly_summary(s)
    assert specs.describe_specs(s)


def test_platform_blowing_up_still_returns_specs(monkeypatch, tmp_path):
    def boom():
        raise RuntimeError("platform is broken")

    monkeypatch.setattr(specs.platform, "machine", boom)
    monkeypatch.setattr(specs, "_detect", lambda *a: (_ for _ in ()).throw(RuntimeError("kaboom")))
    s = specs.detect_specs(tmp_path, benchmark=False)
    assert isinstance(s, SystemSpecs)
    assert s.cpu_name == "Unknown CPU"
    assert any("snag" in n for n in s.notes)


def test_psutil_failures_fall_back_gracefully(machine, monkeypatch, tmp_path):
    machine()

    def broken(*args, **kwargs):
        raise OSError("nope")

    monkeypatch.setattr(specs.psutil, "virtual_memory", broken)
    monkeypatch.setattr(specs.psutil, "disk_usage", broken)
    s = specs.detect_specs(tmp_path)
    assert s.ram_total_gb == 8.0
    assert s.disk_free_gb == -1.0
    assert len([n for n in s.notes if "RAM" in n or "disk" in n]) == 2
    rows = dict(specs.describe_specs(s))
    assert rows["Free disk space"] == "unknown"


def test_disk_space_uses_nearest_existing_parent(machine, monkeypatch, tmp_path):
    machine()
    seen = []
    monkeypatch.setattr(specs.psutil, "disk_usage", lambda path: seen.append(path) or DiskUsage(1, 0, 50 * GIB))
    s = specs.detect_specs(tmp_path / "does" / "not" / "exist" / "yet")
    assert s.disk_free_gb == 50.0
    assert Path(seen[0]) == tmp_path


def test_benchmark_can_be_skipped(machine, monkeypatch, tmp_path):
    machine()
    monkeypatch.setattr(specs.perf, "measure_ram_bandwidth", lambda budget_s=0.3: pytest.fail("should not run"))
    assert specs.detect_specs(tmp_path, benchmark=False).ram_bandwidth_gbs is None


def test_failed_benchmark_adds_a_note(machine, tmp_path):
    machine(bandwidth=None)
    s = specs.detect_specs(tmp_path)
    assert s.ram_bandwidth_gbs is None
    assert any("memory-speed test" in n for n in s.notes)


def test_cpu_name_falls_back_to_lscpu(machine, tmp_path):
    machine(cpuinfo="processor : 0\n", commands={"lscpu": "Architecture: x86_64\nModel name:   Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz\n"})
    assert specs.detect_specs(tmp_path).cpu_name == "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz"


def test_linux_arm_features(machine, tmp_path):
    machine(
        machine_name="aarch64",
        processor="",
        cpuinfo="processor : 0\nFeatures : fp asimd evtstrm crc32 asimddp\nModel : Raspberry Pi 5 Model B Rev 1.0\n",
    )
    s = specs.detect_specs(tmp_path)
    assert s.arch == "arm64"
    assert s.cpu_name == "Raspberry Pi 5 Model B Rev 1.0"
    assert {"neon", "dotprod"} <= set(s.cpu_flags)


# ---------------------------------------------------------------------------
# AMD on Linux
# ---------------------------------------------------------------------------


def test_amd_gpu_vram_from_sysfs(machine, tmp_path):
    add_drm_card(tmp_path, 0, "0000:03:00.0", "0x1002", 24 * GIB)
    machine(
        commands={
            "lspci": "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 31 "
            "[Radeon RX 7900 XTX] (rev c8)\n"
        }
    )
    s = specs.detect_specs(tmp_path)
    assert len(s.gpus) == 1
    gpu = s.gpus[0]
    assert gpu.vendor == "amd" and gpu.name == "AMD Radeon RX 7900 XTX"
    assert gpu.vram_gb == 24.0
    assert gpu.bandwidth_gbs == 960
    assert "an AMD Radeon RX 7900 XTX with 24 GB" in specs.friendly_summary(s)


def test_amd_gpu_vram_from_rocm_smi(machine, tmp_path):
    add_drm_card(tmp_path, 0, "0000:03:00.0", "0x1002", None)
    machine(
        commands={
            "lspci": "03:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Navi 21 [Radeon RX 6800]\n",
            "rocm-smi": '{"card0": {"VRAM Total Memory (B)": "17163091968", "VRAM Total Used Memory (B)": "12"}}',
        }
    )
    s = specs.detect_specs(tmp_path)
    assert s.gpus[0].vram_gb == 16.0


def test_amd_apu_carve_out_is_not_vram(machine, tmp_path):
    add_drm_card(tmp_path, 0, "0000:c4:00.0", "0x1002", 4 * GIB)
    machine(commands={"lspci": "c4:00.0 VGA compatible controller: Advanced Micro Devices, Inc. [AMD/ATI] Phoenix1 (rev c4)\n"})
    s = specs.detect_specs(tmp_path)
    assert s.gpus[0].vram_gb == 0.0
    assert s.gpus[0].name == "AMD Phoenix1"
    assert perf.primary_gpu(s) is None


def test_sysfs_only_when_lspci_missing(machine, tmp_path):
    add_drm_card(tmp_path, 1, "0000:0a:00.0", "0x1002", 8 * GIB)
    machine(commands={})
    s = specs.detect_specs(tmp_path)
    assert [(g.name, g.vram_gb) for g in s.gpus] == [("AMD GPU", 8.0)]


# ---------------------------------------------------------------------------
# macOS
# ---------------------------------------------------------------------------


def apple_commands(brand="Apple M2", translated=None):
    def sysctl(args):
        if args[1:] == ["-n", "machdep.cpu.brand_string"]:
            return brand + "\n"
        if args[1:] == ["hw.optional"]:
            return "hw.optional.arm.FEAT_DotProd: 1\nhw.optional.neon: 1\nhw.optional.floatingpoint: 1\n"
        if args[1:] == ["-n", "sysctl.proc_translated"]:
            return translated
        return None

    return {"sysctl": sysctl}


def test_apple_silicon_unified_memory(machine, tmp_path):
    fake = machine(system="Darwin", machine_name="arm64", processor="arm", commands=apple_commands(), cpuinfo=None)
    s = specs.detect_specs(tmp_path)
    assert s.unified_memory is True
    assert s.cpu_name == "Apple M2"
    assert s.os_version == "macOS 14.5"
    assert {"neon", "dotprod"} <= set(s.cpu_flags)
    assert len(s.gpus) == 1
    gpu = s.gpus[0]
    assert gpu.vendor == "apple" and gpu.name == "Apple M2 GPU"
    assert gpu.vram_gb == pytest.approx(11.2)  # 70 % of 16 GB
    assert gpu.bandwidth_gbs == 100
    assert not any(call[0] == "nvidia-smi" for call in fake.calls)  # no point on a Mac
    summary = specs.friendly_summary(s)
    assert "Mac" in summary and "Apple M2" in summary and "unified memory" in summary


def test_big_macs_can_lend_more_memory_to_the_gpu(machine, tmp_path):
    machine(system="Darwin", machine_name="arm64", processor="arm", ram_gb=64, commands=apple_commands("Apple M1 Max"))
    s = specs.detect_specs(tmp_path)
    assert s.gpus[0].vram_gb == pytest.approx(48.0)  # 75 % of 64 GB
    assert s.gpus[0].bandwidth_gbs == 400


def test_rosetta_is_seen_through(machine, tmp_path):
    machine(system="Darwin", machine_name="x86_64", processor="i386", commands=apple_commands(translated="1\n"))
    s = specs.detect_specs(tmp_path)
    assert s.arch == "arm64" and s.unified_memory
    assert any("Rosetta" in n for n in s.notes)


def test_intel_mac_runs_on_cpu(machine, tmp_path):
    machine(system="Darwin", machine_name="x86_64", processor="i386",
            commands=apple_commands("Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz", translated="0\n"))
    s = specs.detect_specs(tmp_path)
    assert s.gpus == [] and not s.unified_memory
    assert s.cpu_name.startswith("Intel(R) Core(TM) i7-9750H")
    assert "Intel Core i7-9750H" in specs.friendly_summary(s)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------

WIN_ADAPTERS = (
    '[{"Name":"AMD Radeon RX 6700 XT","AdapterRAM":4293918720},'
    '{"Name":"Microsoft Basic Display Adapter","AdapterRAM":0}]'
)


def test_windows_with_amd_card_and_registry_vram(machine, monkeypatch, tmp_path):
    machine(
        system="Windows",
        machine_name="AMD64",
        processor="AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD",
        commands={"powershell": lambda args: "AMD Ryzen 5 5600X 6-Core Processor\n" if "Win32_Processor" in args[-1] else WIN_ADAPTERS},
        cpuinfo=None,
        vulkan=True,
    )
    monkeypatch.setattr(specs, "_windows_registry_vram", lambda: {"AMD Radeon RX 6700 XT": 12.0})
    s = specs.detect_specs(tmp_path)
    assert s.os_name == "Windows" and s.arch == "x86_64"
    assert s.os_version.startswith("Windows 11")
    assert s.cpu_name == "AMD Ryzen 5 5600X 6-Core Processor"
    assert [(g.name, g.vendor, g.vram_gb) for g in s.gpus] == [("AMD Radeon RX 6700 XT", "amd", 12.0)]
    assert s.gpus[0].bandwidth_gbs == 384
    assert set(s.cpu_flags) == {"avx", "avx2", "vulkan"}
    rows = dict(specs.describe_specs(s))
    assert "Vulkan" in rows["GPU acceleration"]


def test_windows_adapter_ram_is_capped_at_4gb(machine, tmp_path):
    machine(system="Windows", machine_name="AMD64", processor="Intel64 Family 6 Model 154 Stepping 3, GenuineIntel",
            commands={"powershell": WIN_ADAPTERS}, cpuinfo=None)
    s = specs.detect_specs(tmp_path)
    assert s.gpus[0].vram_gb == 4.0
    assert any("at least 4 GB" in n for n in s.notes)


def test_windows_nvidia_via_nvidia_smi_and_wmic_fallback(machine, monkeypatch, tmp_path):
    fake = machine(
        system="Windows",
        machine_name="AMD64",
        processor="Intel64 Family 6 Model 183 Stepping 1, GenuineIntel",
        commands={
            "nvidia-smi": "NVIDIA GeForce RTX 4070, 12282, 581.15\n",
            "powershell": (1, ""),
            "wmic": "Node,AdapterRAM,Name\nPC,1073741824,Intel(R) UHD Graphics 770\nPC,4293918720,NVIDIA GeForce RTX 4070\n",
        },
        cpuinfo=None,
    )
    monkeypatch.setattr(specs, "_windows_registry_cpu_name", lambda: "13th Gen Intel(R) Core(TM) i7-13700K")
    s = specs.detect_specs(tmp_path)
    assert s.cpu_name == "13th Gen Intel(R) Core(TM) i7-13700K"
    assert [(g.vendor, g.vram_gb) for g in s.gpus] == [("nvidia", 12.0), ("intel", 0.0)]
    assert s.gpus[0].driver_version == "581.15"
    assert any(call[0] == "wmic" for call in fake.calls)
    summary = specs.friendly_summary(s)
    assert "an NVIDIA GeForce RTX 4070 with 12 GB" in summary and "plenty of muscle" in summary


# ---------------------------------------------------------------------------
# describe_specs / friendly_summary
# ---------------------------------------------------------------------------


def plain_specs(**overrides) -> SystemSpecs:
    values = dict(
        os_name="Linux", os_version="Ubuntu 24.04 LTS, kernel 6.8.0", arch="x86_64",
        cpu_name="Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz", cpu_cores_physical=4, cpu_cores_logical=8,
        ram_total_gb=7.7, ram_available_gb=3.2, disk_free_gb=120.0, ram_bandwidth_gbs=18.4,
    )
    values.update(overrides)
    return SystemSpecs(**values)


def test_describe_specs_rows():
    s = plain_specs(cpu_flags=["avx2", "fma"])
    rows = dict(specs.describe_specs(s))
    assert rows["Operating system"] == "Linux · Ubuntu 24.04 LTS, kernel 6.8.0 · x86_64"
    assert rows["Processor"] == "Intel(R) Core(TM) i5-8250U CPU @ 1.60GHz · 4 cores / 8 threads"
    assert rows["CPU features"] == "AVX2, FMA"
    assert rows["Memory (RAM)"].startswith("8 GB total")
    assert rows["RAM speed"] == "~18 GB/s (quick read test)"
    assert "no dedicated graphics card" in rows["Graphics"]
    assert rows["GPU acceleration"].startswith("none")
    assert rows["Free disk space"].startswith("120 GB")
    assert "Notes" not in rows
    assert all(isinstance(k, str) and isinstance(v, str) for k, v in specs.describe_specs(s))


def test_describe_specs_with_gpus_and_notes():
    gpus = [
        GPUInfo("NVIDIA GeForce RTX 3060", "nvidia", 12.0, 360.0, "550.54"),
        GPUInfo("Intel Iris Xe Graphics", "intel", 0.0),
    ]
    s = plain_specs(gpus=gpus, notes=["A note."], ram_bandwidth_gbs=None, cpu_flags=["vulkan"])
    rows = dict(specs.describe_specs(s))
    assert "12 GB video memory" in rows["Graphics 1"] and "driver 550.54" in rows["Graphics 1"]
    assert "rough guess" in rows["Graphics 1"]
    assert "shares system RAM" in rows["Graphics 2"]
    assert rows["GPU acceleration"] == "CUDA (NVIDIA), Vulkan"
    assert rows["RAM speed"] == "not measured"
    assert rows["CPU features"].startswith("not detected")
    assert rows["Notes"] == "A note."


def test_describe_specs_for_a_mac():
    s = plain_specs(os_name="Darwin", os_version="macOS 14.5", arch="arm64", cpu_name="Apple M3 Pro",
                    unified_memory=True, gpus=[GPUInfo("Apple M3 Pro GPU", "apple", 25.2, 150.0)])
    rows = dict(specs.describe_specs(s))
    assert rows["Operating system"] == "macOS 14.5 · arm64"
    assert "unified memory" in rows["Graphics"]
    assert rows["GPU acceleration"] == "Metal (Apple)"


@pytest.mark.parametrize(
    "ram, bandwidth, expected",
    [
        (3.0, 20.0, "memory is very tight"),  # no promise when nothing would fit
        (15.6, 3.0, "run, but slowly"),  # plenty of RAM, but very slow memory
        (15.6, 40.0, "should keep a comfortable pace"),
    ],
)
def test_friendly_summary_without_gpu(ram, bandwidth, expected):
    text = specs.friendly_summary(plain_specs(ram_total_gb=ram, ram_bandwidth_gbs=bandwidth))
    assert expected in text
    assert "Intel Core i5-8250U" in text  # prettified: no (R)/(TM)/clock speed
    assert "no dedicated graphics card" in text
    assert "fly" not in text  # no fixed, RAM-only promises


def test_friendly_summary_agrees_with_the_fit_engine():
    from gettowork import catalog

    s = plain_specs(ram_total_gb=15.6, ram_bandwidth_gbs=12.8)
    text = specs.friendly_summary(s)
    ranked = catalog.rank_models(s)
    comfy = [f for f in ranked if f.verdict != "no" and (f.est_tokens_per_s or 0) >= 8 and not f.model.active_params_b]
    biggest = max((f.model.params_b for f in comfy), default=0.0)
    import re

    m = re.search(r"up to about ([\d.]+)B", text)
    if biggest == 0.0:
        assert m is None
    else:
        assert m is not None and float(m.group(1)) == pytest.approx(biggest, abs=0.5)


def test_friendly_summary_with_integrated_graphics_and_low_disk():
    s = plain_specs(gpus=[GPUInfo("Intel UHD Graphics 620", "intel", 0.0)], disk_free_gb=6.2)
    text = specs.friendly_summary(s)
    assert "built-in graphics" in text
    assert "only about 6 GB of disk space" in text


@pytest.mark.parametrize(
    "vram, phrase",
    [(24, "serious horsepower"), (12, "plenty of muscle"), (8, "solid setup"), (4, "small models"), (2, "happiest")],
)
def test_friendly_summary_gpu_tiers(vram, phrase):
    s = plain_specs(ram_total_gb=32, gpus=[GPUInfo("NVIDIA GeForce RTX Thing", "nvidia", float(vram))])
    text = specs.friendly_summary(s)
    assert phrase in text
    assert "You've got 32 GB of RAM and an NVIDIA GeForce RTX Thing" in text


def test_articles():
    assert specs._article("NVIDIA GeForce RTX 4090") == "an"
    assert specs._article("AMD Radeon RX 6800") == "an"
    assert specs._article("Intel Arc A770") == "an"
    assert specs._article("Radeon RX 580") == "a"
    assert specs._article("GeForce GTX 1060") == "a"


def test_run_helper_reports_status(monkeypatch):
    monkeypatch.setattr(specs.subprocess, "run", FakeRun({"ok": "hi", "bad": (2, "x"), "slow": subprocess.TimeoutExpired("slow", 5)}))
    assert specs._run(["ok"]) == ("hi", "ok")
    assert specs._run(["bad"]) == (None, "failed")
    assert specs._run(["slow"]) == (None, "failed")
    assert specs._run(["missing-program"]) == (None, "missing")


@pytest.mark.parametrize(
    "raw, arch",
    [("AMD64", "x86_64"), ("x86_64", "x86_64"), ("aarch64", "arm64"), ("ARM64", "arm64"), ("i686", "x86"), ("", "unknown")],
)
def test_arch_normalisation(raw, arch):
    assert specs._normalise_arch(raw) == arch


def test_real_detection_smoke(monkeypatch, tmp_path):
    """Real platform/psutil/proc data (but no external programs): must not raise."""
    monkeypatch.setattr(specs.subprocess, "run", FakeRun({}))
    s = specs.detect_specs(tmp_path, benchmark=False)
    assert s.ram_total_gb > 0
    assert specs.friendly_summary(s)
    assert specs.describe_specs(s)


# ---------------------------------------------------------------------------
# Round 3: Windows tools by their real path; no secrets for child programs
# ---------------------------------------------------------------------------


def test_windows_system_tools_run_from_their_real_location(monkeypatch, tmp_path):
    root = tmp_path / "Windows"
    (root / "System32" / "WindowsPowerShell" / "v1.0").mkdir(parents=True)
    real = root / "System32" / "WindowsPowerShell" / "v1.0" / "powershell.exe"
    real.write_text("")
    monkeypatch.setenv("SystemRoot", str(root))
    monkeypatch.setattr(specs, "_on_windows", lambda: True)
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["args"], seen["env"] = list(args), dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(specs.subprocess, "run", fake_run)
    monkeypatch.setenv("TYPESAFE_API_KEY", "tsk_secret_value")
    assert specs._run(["powershell", "-Command", "x"]) == ("ok", "ok")
    assert seen["args"][0] == str(real)
    assert "TYPESAFE_API_KEY" not in seen["env"] and seen["env"]["NoDefaultCurrentDirectoryInExePath"] == "1"
    # A tool that isn't in its real place counts as missing - never looked up in the current folder.
    assert specs._run(["wmic", "cpu", "get", "name"]) == (None, "missing")


def test_child_programs_never_see_api_keys(monkeypatch):
    seen: dict = {}

    def fake_run(args, **kwargs):
        seen["env"] = dict(kwargs.get("env") or {})
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr(specs.subprocess, "run", fake_run)
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_secret")
    specs._run(["lscpu"])
    assert "HF_TOKEN" not in seen["env"] and "GITHUB_TOKEN" not in seen["env"] and "PATH" in seen["env"]


def test_nvidia_compute_capability_is_read_and_old_drivers_still_work(machine, tmp_path):
    machine(commands={"nvidia-smi": "NVIDIA GeForce GTX 1080, 8192, 580.95.05, 6.1\n"})
    gtx = specs.detect_specs(tmp_path / "models").gpus[0]
    assert gtx.compute_capability == 6.1 and gtx.driver_version == "580.95.05"

    def old_driver(args):  # drivers before ~510 reject the whole query when it names compute_cap
        if any("compute_cap" in a for a in args):
            return (2, 'Field "compute_cap" is not a valid field to query.')
        return "NVIDIA GeForce GTX 970, 4096, 470.82.01\n"

    fake = machine(commands={"nvidia-smi": old_driver})
    old = specs.detect_specs(tmp_path / "models").gpus[0]
    assert old.vram_gb == 4.0 and old.compute_capability is None
    assert specs._NVIDIA_QUERY_OLD in fake.calls
