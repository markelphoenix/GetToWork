"""Find out what computer we're running on: CPU, RAM, graphics card(s), disk.

Running a language model locally is mostly a question of *memory*: how much
you have (does the model fit?) and how fast it is (how quickly will it talk?).
So this module gathers:

* the operating system and CPU (with the SIMD features llama.cpp can use),
* total / available RAM and a quick RAM-speed test (see `perf.py`),
* graphics cards and their video memory (VRAM):
  - NVIDIA via ``nvidia-smi`` (ships with the NVIDIA driver),
  - AMD via Linux sysfs / ``rocm-smi``, other cards by name via ``lspci`` or
    Windows' ``Win32_VideoController``,
  - Apple Silicon, where the GPU shares ("unified") system memory,
* free disk space where models will be downloaded,
* whether a Vulkan loader is installed (lets AMD/Intel GPUs accelerate).

Detection is best-effort and must never crash the game: every external
command has a short timeout, every failure is caught, and anything we could
not figure out becomes a friendly note in ``SystemSpecs.notes``.

Units: RAM and VRAM are reported in binary gigabytes (GiB, what your OS and
the box your GPU came in call "GB"), rounded to 0.1.
"""

from __future__ import annotations

import ctypes.util
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional

from . import config, perf
from .types import GPUInfo, SystemSpecs

try:  # psutil is a declared dependency, but stay alive without it.
    import psutil
except ImportError:  # pragma: no cover - exercised only on broken installs
    psutil = None  # type: ignore[assignment]

__all__ = ["detect_specs", "describe_specs", "friendly_summary", "has_vulkan"]

_GIB = 1024**3
_COMMAND_TIMEOUT_S = 5.0

# Paths are module constants so tests can point them at fake files.
_PROC_CPUINFO = Path("/proc/cpuinfo")
_SYSFS_DRM = Path("/sys/class/drm")

# CPU features that matter to llama.cpp's CPU kernels. "sse2"/"sse4_2" are
# there so a successful check of a CPU *without* AVX (Celeron, Atom...) is
# never mistaken for "couldn't tell" (an empty list).
_X86_FLAGS = ("sse2", "sse4_2", "avx", "avx2", "avx512f", "avx_vnni", "f16c", "fma")
_ARM_FEATURES = {"asimd": "neon", "asimddp": "dotprod", "sve": "sve", "i8mm": "i8mm"}

# Display adapters that are not useful for AI (virtual machines, remote
# desktop, server management chips) - we don't list them.
_IGNORED_GPU_RE = re.compile(
    r"basic (display|render)|microsoft remote|remote display|parsec|virtual|vmware|virtualbox"
    r"|qxl|cirrus|bochs|llvmpipe|aspeed|matrox|hyper-v|citrix|spacedesk|displaylink",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Small, safe helpers
# ---------------------------------------------------------------------------


# Windows system tools, by their real location. Started by bare name, Windows
# would look in the *current folder* first - so a stray "powershell.exe" in
# your Downloads folder could run instead of the real one.
_WINDOWS_TOOLS = {
    "powershell": (r"System32\WindowsPowerShell\v1.0\powershell.exe",),
    "wmic": (r"System32\wbem\WMIC.exe",),
    "nvidia-smi": (r"System32\nvidia-smi.exe",),
}


def _on_windows() -> bool:
    return os.name == "nt"


def _windows_tool(name: str) -> Optional[str]:
    """The full path of a Windows system tool we run, or None if it isn't installed."""
    root = os.environ.get("SystemRoot") or os.environ.get("windir") or r"C:\Windows"
    candidates = [os.path.join(root, *rel.split("\\")) for rel in _WINDOWS_TOOLS.get(name, ())]
    if name == "nvidia-smi":
        program_files = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles") or r"C:\Program Files"
        candidates.append(os.path.join(program_files, "NVIDIA Corporation", "NVSMI", "nvidia-smi.exe"))
    return next((path for path in candidates if os.path.isfile(path)), None)


def _run(args: list[str], timeout: float = _COMMAND_TIMEOUT_S) -> tuple[Optional[str], str]:
    """Run a command without a shell and return ``(stdout, status)``.

    status is "ok", "missing" (the program isn't installed - usually perfectly
    normal), or "failed" (it exists but errored, timed out, or crashed).
    On Windows, system tools are started from their real location (never the
    current folder). The program never sees the player's API keys. Never raises.
    """
    kwargs: dict = {"env": config.child_env()}
    if _on_windows():
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)  # don't flash a console window
        kwargs["env"]["NoDefaultCurrentDirectoryInExePath"] = "1"
        if args and args[0] in _WINDOWS_TOOLS:
            full = _windows_tool(args[0])
            if full is None:
                return None, "missing"
            args = [full, *args[1:]]
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=min(timeout, _COMMAND_TIMEOUT_S),
            **kwargs,
        )
    except FileNotFoundError:
        return None, "missing"
    except Exception:  # timeout, permission denied, anything else
        return None, "failed"
    if getattr(proc, "returncode", 1) != 0:
        return None, "failed"
    out = proc.stdout
    return (out if isinstance(out, str) else None), "ok"


def _read_text(path: Path) -> Optional[str]:
    try:
        return Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def _gib(n_bytes: float) -> float:
    return round(float(n_bytes) / _GIB, 1)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ---------------------------------------------------------------------------
# OS / CPU
# ---------------------------------------------------------------------------


def _normalise_arch(machine: str) -> str:
    m = (machine or "").strip().lower()
    if m in ("x86_64", "amd64", "x64"):
        return "x86_64"
    if m in ("arm64", "aarch64", "armv8", "armv8l"):
        return "arm64"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        return "x86"
    return m or "unknown"


def macos_release() -> str:
    """This Mac's real macOS version ("14.5"), or "" if unknown.

    Python builds made with an old macOS SDK (e.g. Anaconda's Intel Python,
    also under Rosetta) are told "10.16" on Big Sur and later - Apple's
    compatibility shim. Then ask the system directly (``sysctl -n
    kern.osproductversion``), or a fresh Python with the shim switched off
    (``SYSTEM_VERSION_COMPAT=0``, as the ``packaging`` library does); if both
    fail, "" (unknown) is better than a wrong 10.16.
    """
    try:
        release = platform.mac_ver()[0] or ""
    except Exception:
        return ""
    if not re.match(r"^10\.16(?:\.|$)", release):
        return release
    out, status = _run(["sysctl", "-n", "kern.osproductversion"])
    answer = (out or "").strip()
    if status == "ok" and re.match(r"^\d+(?:\.\d+)*$", answer) and not answer.startswith("10.16"):
        return answer
    try:
        env = dict(config.child_env(), SYSTEM_VERSION_COMPAT="0")
        proc = subprocess.run([sys.executable, "-sS", "-c", "import platform; print(platform.mac_ver()[0])"],
                              capture_output=True, text=True, timeout=_COMMAND_TIMEOUT_S, env=env)
        answer = (proc.stdout or "").strip()
    except Exception:
        answer = ""
    if re.match(r"^\d+(?:\.\d+)*$", answer) and not answer.startswith("10.16"):
        return answer
    return ""


def _os_version(os_name: str) -> str:
    try:
        if os_name == "Darwin":
            return f"macOS {macos_release()}".strip()
        if os_name == "Windows":
            release, version = platform.release(), platform.version()
            build = version.rsplit(".", 1)[-1]
            if release == "10" and build.isdecimal() and int(build) >= 22000:
                release = "11"  # Windows 11 still calls itself 10 internally
            return f"Windows {release} ({version})"
        if os_name == "Linux":
            pretty = ""
            try:
                pretty = platform.freedesktop_os_release().get("PRETTY_NAME", "")
            except Exception:
                pass
            kernel = platform.release()
            return f"{pretty}, kernel {kernel}" if pretty else f"kernel {kernel}"
        return platform.release() or ""
    except Exception:
        return ""


def _is_descriptive_cpu_name(name: str) -> bool:
    """platform.processor() is often just "x86_64", "arm", "i386", or on Windows
    "Intel64 Family 6 Model 158 Stepping 10, GenuineIntel" - not a real name."""
    n = (name or "").strip()
    if not n or n.lower() in ("x86_64", "amd64", "arm", "arm64", "aarch64", "i386", "i686", "x86", "unknown"):
        return False
    if re.match(r"^(Intel64|AMD64|EM64T|x86|ARMv?\d*)\b.*Family", n, re.IGNORECASE):
        return False
    return True


def _cpu_name(os_name: str) -> str:
    """The marketing name of the CPU, e.g. "AMD Ryzen 7 5800X 8-Core Processor"."""
    try:
        name = platform.processor()
    except Exception:
        name = ""
    if _is_descriptive_cpu_name(name):
        return _clean(name)

    if os_name == "Linux":
        text = _read_text(_PROC_CPUINFO) or ""
        for key in ("model name", "Model", "Hardware", "cpu model"):
            match = re.search(rf"^{key}\s*:\s*(.+)$", text, re.MULTILINE)
            if match and _is_descriptive_cpu_name(match.group(1)):
                return _clean(match.group(1))
        out, _ = _run(["lscpu"])
        match = re.search(r"^Model name:\s*(.+)$", out or "", re.MULTILINE)
        if match and _is_descriptive_cpu_name(match.group(1)):
            return _clean(match.group(1))
    elif os_name == "Darwin":
        out, _ = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if out and out.strip():
            return _clean(out)
    elif os_name == "Windows":
        name = _windows_registry_cpu_name()
        if name:
            return name
        out, _ = _run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", "(Get-CimInstance Win32_Processor).Name"]
        )
        if out and out.strip():
            return _clean(out.splitlines()[0])
        out, _ = _run(["wmic", "cpu", "get", "name"])
        lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip() and ln.strip().lower() != "name"]
        if lines:
            return _clean(lines[0])
    return "Unknown CPU"


def _windows_registry_cpu_name() -> Optional[str]:
    try:
        import winreg  # type: ignore[import-not-found]

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
        winreg.CloseKey(key)
        return _clean(str(value)) or None
    except Exception:
        return None


def _cpu_flags(os_name: str, arch: str) -> list[str]:
    """SIMD features llama.cpp cares about (best effort; empty list = unknown)."""
    flags: list[str] = []
    try:
        if os_name == "Linux":
            text = _read_text(_PROC_CPUINFO) or ""
            match = re.search(r"^(?:flags|Features)\s*:\s*(.+)$", text, re.MULTILINE)
            tokens = set(match.group(1).split()) if match else set()
            flags += [f for f in _X86_FLAGS if f in tokens]
            flags += [name for token, name in _ARM_FEATURES.items() if token in tokens]
        elif os_name == "Darwin":
            out, _ = _run(["sysctl", "hw.optional"])
            mapping = {
                "sse2": "sse2", "sse4_2": "sse4_2",
                "avx1_0": "avx", "avx2_0": "avx2", "avx512f": "avx512f", "fma": "fma", "f16c": "f16c",
                "neon": "neon", "AdvSIMD": "neon", "arm.FEAT_DotProd": "dotprod", "arm.FEAT_I8MM": "i8mm",
            }
            for line in (out or "").splitlines():
                key, _, value = line.partition(":")
                name = mapping.get(key.strip().removeprefix("hw.optional."))
                if name and value.strip() == "1" and name not in flags:
                    flags.append(name)
            if arch == "arm64" and "neon" not in flags:
                flags.append("neon")  # every Apple Silicon chip has NEON
        elif os_name == "Windows":
            flags += _windows_cpu_flags(arch)
    except Exception:
        pass
    return flags


def _windows_cpu_flags(arch: str) -> list[str]:
    """Ask Windows' IsProcessorFeaturePresent (older Windows may not know AVX)."""
    try:
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
    except Exception:
        return []
    # PF_XMMI64 (SSE2) = 10, PF_SSE4_2 = 38, PF_AVX = 39, PF_AVX2 = 40, PF_AVX512F = 41.
    features = {"sse2": 10, "sse4_2": 38, "avx": 39, "avx2": 40, "avx512f": 41} if arch != "arm64" else {"neon": 19}
    found = []
    for name, code in features.items():
        try:
            if kernel32.IsProcessorFeaturePresent(code):
                found.append(name)
        except Exception:
            pass
    return found


# ---------------------------------------------------------------------------
# GPUs
# ---------------------------------------------------------------------------


def _vendor_from_name(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "tesla" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n or "ati " in n or "advanced micro devices" in n:
        return "amd"
    if "intel" in n or re.search(r"\barc\b", n):
        return "intel"
    if "apple" in n:
        return "apple"
    return "unknown"


def _parse_nvidia_smi(out: str) -> list[GPUInfo]:
    """Parse ``name, memory.total [MiB], driver_version[, compute_cap]`` CSV lines."""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2 or not parts[0]:
            continue
        try:
            vram = _gib(float(parts[1]) * 1024 * 1024)
        except ValueError:
            vram = 0.0  # "[N/A]" on some systems
        driver = parts[2] if len(parts) > 2 and re.match(r"^\d+(\.\d+)*$", parts[2]) else None
        compute = float(parts[3]) if len(parts) > 3 and re.match(r"^\d+\.\d+$", parts[3]) else None
        gpus.append(GPUInfo(name=_clean(parts[0]), vendor="nvidia", vram_gb=vram, driver_version=driver,
                            compute_capability=compute))
    return gpus


_NVIDIA_QUERY = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap",
                 "--format=csv,noheader,nounits"]
# Drivers older than ~510 don't know "compute_cap" and reject the whole query.
_NVIDIA_QUERY_OLD = ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"]


def _nvidia_gpus(notes: list[str]) -> tuple[list[GPUInfo], str]:
    out, status = _run(_NVIDIA_QUERY)
    if status == "failed":
        out, status = _run(_NVIDIA_QUERY_OLD)
    if status != "ok":
        if status == "failed":
            notes.append("nvidia-smi is installed but didn't answer, so NVIDIA video memory is unknown.")
        return [], status
    gpus = _parse_nvidia_smi(out or "")
    if not gpus:
        notes.append("nvidia-smi gave an answer we couldn't read, so NVIDIA video memory is unknown.")
        return [], "failed"
    if any(g.vram_gb <= 0 for g in gpus):
        notes.append("An NVIDIA GPU didn't report its video memory; we'll plan as if it had none.")
    return gpus, "ok"


def _lspci_display_devices() -> list[tuple[str, str]]:
    """[(pci_slot, device description)] for display controllers, via lspci."""
    out, _ = _run(["lspci"])
    devices = []
    for line in (out or "").splitlines():
        match = re.match(r"^(\S+)\s+(?:VGA compatible controller|3D controller|Display controller)[^:]*:\s*(.+)$", line)
        if match:
            desc = re.sub(r"\s*\(rev [0-9a-f]+\)\s*$", "", match.group(2), flags=re.IGNORECASE)
            devices.append((match.group(1), _clean(desc)))
    return devices


def _pretty_lspci_name(desc: str) -> str:
    """"Advanced Micro Devices, Inc. [AMD/ATI] Navi 22 [Radeon RX 6700 XT]" ->
    "AMD Radeon RX 6700 XT"."""
    vendor = _vendor_from_name(desc)
    brackets = [b for b in re.findall(r"\[([^\]]+)\]", desc) if b not in ("AMD/ATI", "AMD")]
    model = brackets[-1] if brackets else ""
    if not model:
        model = re.sub(r"^.*?(Corporation|Inc\.)\s*", "", desc)
        model = _clean(re.sub(r"\[(AMD/ATI|AMD)\]", "", model)) or desc
    prefix = {"amd": "AMD", "intel": "Intel", "nvidia": "NVIDIA"}.get(vendor, "")
    return f"{prefix} {model}".strip() if prefix and prefix.lower() not in model.lower() else model


def _linux_drm_cards() -> list[tuple[str, str, float]]:
    """[(pci_slot, vendor, vram_gb)] from /sys/class/drm (amdgpu reports VRAM there)."""
    cards = []
    vendor_ids = {"0x1002": "amd", "0x10de": "nvidia", "0x8086": "intel"}
    try:
        entries = sorted(p for p in _SYSFS_DRM.iterdir() if re.fullmatch(r"card\d+", p.name))
    except Exception:
        return cards
    for card in entries:
        device = card / "device"
        vendor = vendor_ids.get((_read_text(device / "vendor") or "").strip().lower(), "unknown")
        vram = 0.0
        raw = (_read_text(device / "mem_info_vram_total") or "").strip()
        if raw.isdecimal():
            vram = _gib(int(raw))
        try:
            slot = os.path.basename(os.path.realpath(device))
        except Exception:
            slot = ""
        cards.append((slot, vendor, vram))
    return cards


def _rocm_vram() -> list[float]:
    """VRAM per AMD card from ``rocm-smi --showmeminfo vram --json`` (best effort)."""
    out, status = _run(["rocm-smi", "--showmeminfo", "vram", "--json"])
    if status != "ok" or not out:
        return []
    try:
        data = json.loads(out)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    sizes = []
    for card, info in sorted(data.items()):  # {"card0": {"VRAM Total Memory (B)": "17163091968", ...}}
        if not (isinstance(info, dict) and str(card).lower().startswith("card")):
            continue
        for key, value in info.items():
            if "total memory" in key.lower() and "used" not in key.lower():
                try:
                    sizes.append(_gib(int(str(value).strip())))
                except ValueError:
                    pass
                break
    return sizes


def _linux_gpus(nvidia: list[GPUInfo], nvidia_status: str, notes: list[str]) -> list[GPUInfo]:
    """Non-NVIDIA GPUs (and NVIDIA ones nvidia-smi missed) on Linux."""
    drm = _linux_drm_cards()
    drm_by_slot = {slot: (vendor, vram) for slot, vendor, vram in drm if slot}
    found: list[GPUInfo] = []

    devices = _lspci_display_devices()
    if devices:
        for slot, desc in devices:
            if _IGNORED_GPU_RE.search(desc):
                continue
            vendor = _vendor_from_name(desc)
            if vendor == "nvidia" and nvidia:
                continue  # nvidia-smi already told us everything
            # sysfs slots carry a PCI domain prefix ("0000:03:00.0"); lspci's usually don't.
            vram = next((v for s, (_, v) in drm_by_slot.items() if s == slot or s.endswith(":" + slot)), 0.0)
            found.append(GPUInfo(name=_pretty_lspci_name(desc), vendor=vendor, vram_gb=vram))  # type: ignore[arg-type]
    else:  # no lspci (minimal installs): fall back to sysfs vendor ids
        for _, vendor, vram in drm:
            if vendor == "nvidia" and nvidia:
                continue
            if vendor in ("amd", "intel", "nvidia"):
                label = {"amd": "AMD GPU", "intel": "Intel GPU", "nvidia": "NVIDIA GPU"}[vendor]
                found.append(GPUInfo(name=label, vendor=vendor, vram_gb=vram))  # type: ignore[arg-type]

    # rocm-smi can fill in AMD VRAM that sysfs didn't give us.
    amd_unknown = [g for g in found if g.vendor == "amd" and g.vram_gb <= 0]
    if amd_unknown:
        sizes = [s for s in _rocm_vram() if s > 0]
        if len(sizes) == len(amd_unknown):
            for gpu, size in zip(amd_unknown, sizes):
                gpu.vram_gb = size

    for gpu in found:
        if gpu.vendor == "nvidia" and gpu.vram_gb <= 0:
            notes.append(
                f"Found {gpu.name}, but couldn't read its video memory (nvidia-smi "
                f"{'is missing' if nvidia_status == 'missing' else 'failed'}). Is the NVIDIA driver installed?"
            )
    return found


def _windows_registry_vram() -> dict[str, float]:
    """{adapter name: VRAM GiB} from the display-driver registry keys.

    Win32_VideoController.AdapterRAM is a 32-bit number and tops out at 4 GB,
    but drivers also store the real size as a 64-bit value in the registry.
    """
    result: dict[str, float] = {}
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return result
    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    for index in range(16):
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{base}\\{index:04d}")
        except OSError:
            continue
        try:
            name = str(winreg.QueryValueEx(key, "DriverDesc")[0])
            size = None
            for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                try:
                    raw = winreg.QueryValueEx(key, value_name)[0]
                except OSError:
                    continue
                size = int.from_bytes(raw, "little") if isinstance(raw, (bytes, bytearray)) else int(raw)
                break
            if size:
                result[_clean(name)] = _gib(size)
        except Exception:
            pass
        finally:
            winreg.CloseKey(key)
    return result


def _windows_gpus(nvidia: list[GPUInfo], notes: list[str]) -> list[GPUInfo]:
    """Non-NVIDIA GPUs (and NVIDIA ones nvidia-smi missed) on Windows."""
    adapters: list[tuple[str, float]] = []
    out, _ = _run(
        [
            "powershell", "-NoProfile", "-NonInteractive", "-Command",
            "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM | ConvertTo-Json -Compress",
        ]
    )
    try:
        data = json.loads(out) if out and out.strip() else []
        for item in data if isinstance(data, list) else [data]:
            if isinstance(item, dict) and item.get("Name"):
                adapters.append((_clean(str(item["Name"])), float(item.get("AdapterRAM") or 0)))
    except (ValueError, TypeError):
        adapters = []
    if not adapters:  # older systems: wmic
        out, _ = _run(["wmic", "path", "win32_VideoController", "get", "Name,AdapterRAM", "/format:csv"])
        for line in (out or "").splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3 and parts[2] and parts[2].lower() != "name":  # Node,AdapterRAM,Name
                ram = float(parts[1]) if parts[1].isdecimal() else 0.0
                adapters.append((_clean(parts[2]), ram))

    registry = _windows_registry_vram()
    found = []
    for name, adapter_ram in adapters:
        if _IGNORED_GPU_RE.search(name):
            continue
        vendor = _vendor_from_name(name)
        if vendor == "nvidia" and nvidia:
            continue
        vram = registry.get(name, 0.0)
        if not vram and adapter_ram > 0:
            vram = _gib(adapter_ram)
            if vram >= 3.9:
                notes.append(f"Windows only reports 'at least 4 GB' for {name}; we'll assume 4 GB.")
                vram = 4.0
        if vram < 1.0:
            vram = 0.0  # integrated graphics: a tiny slice of shared memory, not real VRAM
        found.append(GPUInfo(name=name, vendor=vendor, vram_gb=vram))  # type: ignore[arg-type]
        if vendor == "nvidia" and not nvidia:
            notes.append(f"Found {name}, but nvidia-smi didn't answer. Is the NVIDIA driver up to date?")
    return found


# AMD APU code names as lspci shows them (their "VRAM" is a slice of system RAM).
_AMD_APU_RE = re.compile(
    r"\b\d{3}m\b|radeon graphics|vega \d+\b|phoenix|rembrandt|renoir|cezanne|raphael|barcelo|lucienne"
    r"|picasso|raven|mendocino|hawk point|granite ridge|strix point|krackan|van gogh|dragon range"
)


def _is_integrated(gpu: GPUInfo) -> bool:
    """Built-in graphics that borrow system RAM rather than having their own VRAM."""
    n = re.sub(r"\((tm|r)\)|[®™]", "", gpu.name.lower())
    if gpu.vendor == "intel":
        return not re.search(r"\barc\b.*\b[ab]\d{3}", n) and "arc pro" not in n
    if gpu.vendor == "amd":
        return bool(_AMD_APU_RE.search(n))  # (Strix Halo "8060S" with a big carve-out counts as a GPU)
    return False


def _apple_gpu(cpu_name: str, ram_total_gb: float) -> GPUInfo:
    """Apple Silicon: the GPU can use most (not all) of the unified memory.

    macOS lets the GPU "wire" roughly 70-75% of RAM by default, so that is
    the budget we plan with.
    """
    share = 0.75 if _nice_gb(ram_total_gb) >= 64 else 0.70
    chip = cpu_name if cpu_name.lower().startswith("apple") else "Apple Silicon"
    return GPUInfo(name=f"{chip} GPU", vendor="apple", vram_gb=round(ram_total_gb * share, 1))


def _vulkan_loader_present(os_name: str) -> bool:
    """Is the Vulkan loader installed? (Lets llama.cpp's Vulkan build use AMD/Intel GPUs.)"""
    try:
        if os_name == "Windows":
            system_root = os.environ.get("SystemRoot", r"C:\Windows")
            return (Path(system_root) / "System32" / "vulkan-1.dll").exists()
        if os_name == "Linux":
            return bool(ctypes.util.find_library("vulkan"))
    except Exception:
        pass
    return False


def has_vulkan(specs: SystemSpecs) -> bool:
    """True if detection found a Vulkan loader (recorded as "vulkan" in cpu_flags)."""
    return "vulkan" in specs.cpu_flags


def uses_built_in_graphics(specs: SystemSpecs) -> bool:
    """Will the engine's Vulkan build run on built-in graphics (no dedicated video memory)?

    True for an Intel Iris Xe / AMD APU PC where the built-in engine picks its
    Vulkan build (always on Windows; on Linux when the Vulkan loader is there).
    """
    if any(g.vendor != "apple" and g.vram_gb > 0 for g in specs.gpus):
        return False
    built_in = [g for g in specs.gpus if g.vendor in ("intel", "amd") and g.vram_gb <= 0]
    if not built_in or specs.gpu_offload is False:
        return False
    return specs.os_name == "Windows" or (specs.os_name == "Linux" and has_vulkan(specs))


# ---------------------------------------------------------------------------
# RAM / disk
# ---------------------------------------------------------------------------


def _ram(notes: list[str]) -> tuple[float, float]:
    try:
        vm = psutil.virtual_memory()  # type: ignore[union-attr]
        return _gib(vm.total), _gib(vm.available)
    except Exception:
        notes.append("Couldn't read how much memory (RAM) this computer has; assuming 8 GB.")
        return 8.0, 4.0


def _cpu_counts() -> tuple[Optional[int], Optional[int]]:
    physical = logical = None
    try:
        physical = psutil.cpu_count(logical=False)  # type: ignore[union-attr]
        logical = psutil.cpu_count(logical=True)  # type: ignore[union-attr]
    except Exception:
        pass
    if not logical:
        logical = os.cpu_count()
    return (physical or None), (logical or None)


def _disk_free(models_path: Optional[Path], notes: list[str]) -> float:
    """Free space (GiB) where models go; -1.0 means "unknown" (don't block on it)."""
    try:
        path = Path(models_path) if models_path else config.models_dir()
        path = path.expanduser().absolute()
        while not path.exists() and path.parent != path:
            path = path.parent  # the models folder may not exist yet
        usage = psutil.disk_usage(str(path)) if psutil else __import__("shutil").disk_usage(str(path))
        return _gib(usage.free)
    except Exception:
        notes.append("Couldn't check free disk space; make sure you have room for a few GB of downloads.")
        return -1.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_specs(models_path: Optional[Path] = None, benchmark: bool = True) -> SystemSpecs:
    """Inspect this computer and return a `SystemSpecs` snapshot. Never raises.

    `models_path` is where models will be downloaded (default:
    ``config.models_dir()``); we report the free space there. Set
    `benchmark=False` to skip the ~0.3 s RAM-speed test (e.g. in tests or for
    ``--specs`` on a slow machine).
    """
    notes: list[str] = []
    try:
        return _detect(models_path, benchmark, notes)
    except Exception as exc:  # a last line of defence: never crash the game over this
        notes.append(f"Hardware detection hit a snag ({type(exc).__name__}); using safe defaults.")
        return SystemSpecs(
            os_name=_safe(platform.system, "Unknown"), os_version="", arch=_normalise_arch(_safe(platform.machine, "")),
            cpu_name="Unknown CPU", cpu_cores_physical=None, cpu_cores_logical=_safe(os.cpu_count, None),
            ram_total_gb=8.0, ram_available_gb=4.0, disk_free_gb=-1.0, notes=notes,
        )


def _safe(fn, default):
    """fn(), or `default` if it raises or returns nothing."""
    try:
        return fn() or default
    except Exception:
        return default


def _detect(models_path: Optional[Path], benchmark: bool, notes: list[str]) -> SystemSpecs:
    os_name = platform.system() or "Unknown"
    arch = _normalise_arch(platform.machine())
    if os_name == "Darwin" and arch == "x86_64":
        # Python running under Rosetta on an Apple Silicon Mac reports x86_64.
        out, _ = _run(["sysctl", "-n", "sysctl.proc_translated"])
        if (out or "").strip() == "1":
            arch = "arm64"
            notes.append("Python is running under Rosetta; this is really an Apple Silicon Mac.")

    cpu_name = _cpu_name(os_name)
    physical, logical = _cpu_counts()
    ram_total, ram_available = _ram(notes)
    flags = _cpu_flags(os_name, arch)

    gpus: list[GPUInfo] = []
    unified = False
    if os_name == "Darwin":
        if arch == "arm64":
            gpus.append(_apple_gpu(cpu_name, ram_total))
            unified = True
        else:
            notes.append("On Intel Macs the game runs models on the CPU.")
    else:
        nvidia, nvidia_status = _nvidia_gpus(notes)
        gpus.extend(nvidia)
        if os_name == "Linux":
            gpus.extend(_linux_gpus(nvidia, nvidia_status, notes))
        elif os_name == "Windows":
            gpus.extend(_windows_gpus(nvidia, notes))
        for gpu in gpus:
            if gpu.vendor != "nvidia" and gpu.vram_gb > 0 and (gpu.vram_gb < 1.0 or _is_integrated(gpu)):
                gpu.vram_gb = 0.0  # a carve-out of system RAM, not dedicated video memory
        has_dedicated = any(g.vram_gb > 0 for g in gpus)
        if not has_dedicated and any(g.vendor in ("amd", "intel", "unknown") for g in gpus):
            notes.append("Built-in (integrated) graphics share system RAM, so we plan with the CPU and RAM.")

    if _vulkan_loader_present(os_name):
        flags.append("vulkan")

    specs = SystemSpecs(
        os_name=os_name,
        os_version=_os_version(os_name),
        arch=arch,
        cpu_name=cpu_name,
        cpu_cores_physical=physical,
        cpu_cores_logical=logical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_available,
        disk_free_gb=_disk_free(models_path, notes),
        gpus=gpus,
        unified_memory=unified,
        notes=notes,
        cpu_flags=flags,
    )
    for gpu in specs.gpus:
        gpu.bandwidth_gbs = perf.estimate_gpu_bandwidth(gpu)
    if benchmark:
        specs.ram_bandwidth_gbs = perf.measure_ram_bandwidth()
        if specs.ram_bandwidth_gbs is None:
            notes.append("The quick memory-speed test didn't work here; we'll use a typical value instead.")
        elif getattr(perf, "last_benchmark_note", None):
            notes.append(perf.last_benchmark_note)
    return specs


# Memory sizes printed on the box. Operating systems report a little less
# (15.6 GB for a 16 GB machine), so we snap to these when we're close.
_COMMON_SIZES_GB = (2, 4, 6, 8, 12, 16, 18, 24, 32, 36, 48, 64, 96, 128, 192, 256, 512)


def _nice_gb(value: float) -> float:
    for size in _COMMON_SIZES_GB:
        if abs(value - size) <= 0.08 * size:
            return float(size)
    return value


def _fmt_gb(value: float) -> str:
    """Friendly sizes: 15.6 -> "16 GB", 31.3 -> "32 GB", 11.2 -> "11 GB", 1.5 -> "1.5 GB"."""
    value = _nice_gb(value)
    if value >= 1.75:
        return f"{round(value):.0f} GB"
    return f"{value:.1f}".rstrip("0").rstrip(".") + " GB"


def _cpu_row(specs: SystemSpecs) -> str:
    parts = [specs.cpu_name]
    if specs.cpu_cores_physical and specs.cpu_cores_logical and specs.cpu_cores_logical != specs.cpu_cores_physical:
        parts.append(f"{specs.cpu_cores_physical} cores / {specs.cpu_cores_logical} threads")
    elif specs.cpu_cores_logical or specs.cpu_cores_physical:
        parts.append(f"{specs.cpu_cores_physical or specs.cpu_cores_logical} cores")
    return " · ".join(parts)


def describe_specs(specs: SystemSpecs) -> list[tuple[str, str]]:
    """Rows for a two-column "what we found" table."""
    rows: list[tuple[str, str]] = []
    if specs.os_name == "Linux" or not specs.os_version:
        os_text = " · ".join(x for x in (specs.os_name, specs.os_version) if x)
    else:
        os_text = specs.os_version  # "macOS 14.5" / "Windows 11 (...)" already name the OS
    rows.append(("Operating system", f"{os_text} · {specs.arch}"))
    rows.append(("Processor", _cpu_row(specs)))
    simd = [f.upper() for f in specs.cpu_flags if f not in ("vulkan", "sse2")]  # SSE2 is on every x86-64 CPU
    rows.append(("CPU features", ", ".join(simd) if simd else "not detected (that's OK)"))
    rows.append(("Memory (RAM)", f"{_fmt_gb(specs.ram_total_gb)} total · {specs.ram_available_gb:.1f} GB free right now"))
    if specs.ram_bandwidth_gbs:
        rows.append(("RAM speed", f"~{specs.ram_bandwidth_gbs:.0f} GB/s (quick read test)"))
    else:
        rows.append(("RAM speed", "not measured"))

    if not specs.gpus:
        rows.append(("Graphics", "no dedicated graphics card found — models will run on the CPU"))
    for i, gpu in enumerate(specs.gpus, 1):
        label = "Graphics" if len(specs.gpus) == 1 else f"Graphics {i}"
        bits = [gpu.name]
        if gpu.vendor == "apple":
            bits.append(f"can use ~{gpu.vram_gb:.0f} GB of unified memory")
        elif gpu.vram_gb > 0:
            bits.append(f"{_fmt_gb(gpu.vram_gb)} video memory")
        else:
            bits.append("shares system RAM (no dedicated video memory detected)")
        if gpu.bandwidth_gbs:
            bits.append(f"~{gpu.bandwidth_gbs:.0f} GB/s (rough guess)")
        if gpu.driver_version:
            bits.append(f"driver {gpu.driver_version}")
        rows.append((label, " · ".join(bits)))

    accel = []
    if any(g.vendor == "nvidia" and g.vram_gb > 0 for g in specs.gpus):
        accel.append("CUDA (NVIDIA)")
    if specs.unified_memory:
        accel.append("Metal (Apple)")
    if has_vulkan(specs) and any(g.vendor != "apple" and g.vram_gb > 0 for g in specs.gpus):
        accel.append("Vulkan")
    elif not accel and uses_built_in_graphics(specs):
        # The engine's Vulkan build will use built-in graphics (they read the same
        # RAM, so the speed estimate for "the processor" still holds).
        accel.append("Vulkan on the built-in graphics (they share your RAM, so I plan as if the model runs "
                     "from RAM)")
    rows.append(("GPU acceleration", ", ".join(accel) if accel else "none — CPU only (still works!)"))

    if specs.disk_free_gb >= 0:
        rows.append(("Free disk space", f"{specs.disk_free_gb:.0f} GB where models are saved"))
    else:
        rows.append(("Free disk space", "unknown"))
    if specs.notes:
        rows.append(("Notes", "\n".join(specs.notes)))
    return rows


def _pretty_cpu_name(name: str) -> str:
    """"Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz" -> "Intel Core i7-9750H" ("" if unknown)."""
    if not name or name == "Unknown CPU":
        return ""
    text = re.sub(r"\((R|TM|C)\)|[®™]", "", name, flags=re.IGNORECASE)
    text = re.sub(r"\s*@.*$", "", text)  # clock speed
    text = re.sub(r"\s+(w/|with)\s+Radeon.*$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\b(CPU|Processor|\d+-Core Processor)$", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+\d+-Core$", "", text.strip(), flags=re.IGNORECASE)
    return _clean(text) or _clean(name)


def _article(name: str) -> str:
    """"an AMD ...", "an NVIDIA ...", "an Intel ...", but "a GeForce ...", "a Radeon ..."."""
    return "an" if name[:1].lower() in "aeiou" or name.upper().startswith("NVIDIA") else "a"


_COMFORTABLE_TOKENS_PER_S = 8.0  # the same "comfortable" line the model menu uses


def _cpu_verdict(specs: SystemSpecs) -> str:
    """What to expect on the processor, worked out by the fit engine itself.

    Rates the built-in model list for *this* computer (RAM size, measured
    memory speed, cores - see ``catalog.py`` and ``perf.py``), so it agrees
    with the model menu shown a moment later, and never promises a model
    when none would fit.
    """
    from . import catalog  # imported here: catalog is only needed for this sentence

    runnable = [f for f in catalog.rank_models(specs) if f.verdict != "no"]
    if not runnable:
        return "memory is very tight, so I'll look for the tiniest model that still fits (there may not be one)."
    comfortable = [f for f in runnable if (f.est_tokens_per_s or 0.0) >= _COMFORTABLE_TOKENS_PER_S]
    if not comfortable:
        return "models that fit here will run, but slowly, so expect a little wait between turns."
    dense = [f.model.params_b for f in comfortable if not f.model.active_params_b]
    moe = [f for f in comfortable if f.model.active_params_b]
    if not dense:
        return "that's fine: small Mixture-of-Experts models should keep a comfortable pace."
    best = max(dense)
    size = f"{best:.0f}" if best >= 3 else f"{best:.1f}".rstrip("0").rstrip(".")
    text = f"that's fine: models up to about {size}B parameters should keep a comfortable pace"
    if any(f.model.params_b > best for f in moe):
        text += " (plus some bigger Mixture-of-Experts ones, which only use part of themselves per word)"
    if any(f.model.params_b > best and not f.model.active_params_b for f in runnable):
        text += ", and bigger ones will run, just slowly"
    return text + "."


def friendly_summary(specs: SystemSpecs) -> str:
    """One or two warm, plain-English sentences about this computer."""
    ram = _fmt_gb(specs.ram_total_gb)
    gpu = perf.primary_gpu(specs)

    if specs.unified_memory and gpu is not None and gpu.vendor == "apple":
        chip = gpu.name.removesuffix(" GPU")
        usable = gpu.vram_gb
        if usable >= 40:
            verdict = "serious muscle — even big models are on the table!"
        elif usable >= 20:
            verdict = "room for some genuinely clever mid-sized models!"
        elif usable >= 10:
            verdict = "great for small-to-mid-sized models!"
        else:
            verdict = "perfect for small, speedy models!"
        text = (
            f"You're on a Mac with an {chip} chip and {ram} of unified memory — the graphics side can "
            f"borrow about {usable:.0f} GB of it, which is {verdict}"
        )
    elif gpu is not None:
        vram = gpu.vram_gb
        if vram >= 20:
            verdict = "serious horsepower — even big 14B-32B models are on the table!"
        elif vram >= 10:
            verdict = "plenty of muscle for mid-sized models!"
        elif vram >= 6:
            verdict = "a solid setup for local AI!"
        elif vram >= 3.5:
            verdict = "enough to run small models nice and quickly."
        else:
            verdict = "small models will be happiest here."
        text = (
            f"You've got {ram} of RAM and {_article(gpu.name)} {gpu.name} with {_fmt_gb(vram)} of "
            f"video memory — {verdict}"
        )
    else:
        pretty = _pretty_cpu_name(specs.cpu_name)
        cpu = f"your {pretty} processor" if pretty else "your processor"
        if specs.gpu_offload is False and any(g.vendor != "apple" and g.vram_gb > 0 for g in specs.gpus):
            why = "a graphics card the built-in engine can't use here (see the notes below)"
        elif any(g.vendor == "nvidia" for g in specs.gpus):
            why = "an NVIDIA card we couldn't talk to (is its driver installed?)"
        elif specs.gpus:
            why = "built-in graphics that share your RAM"
        else:
            why = "no dedicated graphics card"
        text = f"You've got {ram} of RAM and {why}, so models will run on {cpu} — {_cpu_verdict(specs)}"

    if 0 <= specs.disk_free_gb < 10:
        text += (
            f" Heads up: only about {specs.disk_free_gb:.0f} GB of disk space is free, "
            "so we'll favour smaller downloads."
        )
    return text
