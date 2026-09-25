"""Tests for gettowork.runtime_install (no network: a fake HTTP layer serves
release JSON and archives that are built in memory)."""

from __future__ import annotations

import hashlib
import io
import json
import time
import os
import stat
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from gettowork import runtime_install as ri
from gettowork.runtime_install import (
    CPU,
    CUDA12,
    CUDA13,
    METAL,
    ROCM,
    VULKAN,
    RuntimeInstallError,
    ensure_llama_server,
    fetch_releases,
    install_info,
    installed_runtimes,
    is_platform_supported,
    pick_release,
    plan_variants,
    safe_extract,
    select_assets,
    variant_by_name,
)
from gettowork.types import GPUInfo, SystemSpecs
from gettowork.ui import UI

_REAL_GLIBC_VERSION = ri._glibc_version  # before the autouse fixture pins it

API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
DL = "https://github.com/ggml-org/llama.cpp/releases/download"
POSIX = os.name != "nt"


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


def release_yml_names(tag: str = "b7000") -> list[str]:
    """Every asset name the official release workflow (.github/workflows/release.yml
    in ggml-org/llama.cpp) publishes for one release."""
    t = tag
    return [
        f"llama-{t}-bin-macos-arm64.tar.gz",
        f"llama-{t}-bin-macos-x64.tar.gz",
        f"llama-{t}-xcframework.zip",
        f"llama-{t}-bin-ubuntu-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-arm64.tar.gz",
        f"llama-{t}-bin-ubuntu-s390x.tar.gz",
        f"llama-{t}-bin-ubuntu-vulkan-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-vulkan-arm64.tar.gz",
        f"llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        f"cudart-llama-{t}-bin-ubuntu-cuda-12.8-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"cudart-llama-{t}-bin-ubuntu-cuda-13.4-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        f"cudart-llama-{t}-bin-ubuntu-cuda-13.4-arm64.tar.gz",
        f"llama-{t}-bin-ubuntu-rocm-10.0-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-openvino-2025.3-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-sycl-fp32-x64.tar.gz",
        f"llama-{t}-bin-ubuntu-sycl-fp16-x64.tar.gz",
        f"llama-{t}-bin-linux-arm64-snapdragon.tar.gz",
        f"llama-{t}-bin-android-arm64.tar.gz",
        f"llama-{t}-bin-android-arm64-snapdragon.tar.gz",
        f"llama-{t}-bin-win-cpu-x64.zip",
        f"llama-{t}-bin-win-cpu-arm64.zip",
        f"llama-{t}-bin-win-opencl-adreno-arm64.zip",
        f"llama-{t}-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
        f"llama-{t}-bin-win-cuda-13.4-x64.zip",
        "cudart-llama-bin-win-cuda-13.4-x64.zip",
        f"llama-{t}-bin-win-cuda-13.4-arm64.zip",
        "cudart-llama-bin-win-cuda-13.4-arm64.zip",
        f"llama-{t}-bin-win-vulkan-x64.zip",
        f"llama-{t}-bin-win-openvino-2025.3-x64.zip",
        f"llama-{t}-bin-win-sycl-x64.zip",
        f"llama-{t}-bin-win-rocm-10.0-x64.zip",
        f"llama-{t}-ui.tar.gz",
    ]


def assets_from(names: list[str], tag: str = "b7000") -> list[dict]:
    return [{"name": n, "size": 1000, "browser_download_url": f"{DL}/{tag}/{n}", "state": "uploaded"} for n in names]


ALL_ASSETS = assets_from(release_yml_names())


def make_specs(os_name="Linux", arch="x86_64", gpus=(), notes=(), flags=()) -> SystemSpecs:
    return SystemSpecs(
        os_name=os_name,
        os_version="test",
        arch=arch,
        cpu_name="Test CPU",
        cpu_cores_physical=4,
        cpu_cores_logical=8,
        ram_total_gb=16.0,
        ram_available_gb=12.0,
        disk_free_gb=100.0,
        gpus=list(gpus),
        notes=list(notes),
        cpu_flags=list(flags),
    )


NVIDIA = GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0)
AMD = GPUInfo(name="AMD Radeon RX 7800 XT", vendor="amd", vram_gb=16.0)
INTEL = GPUInfo(name="Intel Arc A770", vendor="intel", vram_gb=16.0)
APPLE = GPUInfo(name="Apple M2", vendor="apple", vram_gb=11.2)
PASCAL = GPUInfo(name="NVIDIA GeForce GTX 1080", vendor="nvidia", vram_gb=8.0, compute_capability=6.1)
TURING = GPUInfo(name="NVIDIA GeForce RTX 2060", vendor="nvidia", vram_gb=6.0, compute_capability=7.5)


def make_ui() -> UI:
    return UI(console=Console(file=io.StringIO(), width=200), input_fn=lambda prompt: "")


def output(ui: UI) -> str:
    return ui.console.file.getvalue()


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None, fail_with=None):
        self.status = status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._buf = io.BytesIO(body)
        self._fail_with = fail_with
        self._reads = 0

    def read(self, n=-1):
        if self._fail_with is not None and self._reads >= 1:
            raise self._fail_with
        self._reads += 1
        return self._buf.read() if n is None or n < 0 else self._buf.read(n)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class BrokenResponse(FakeResponse):
    def read(self, n=-1):
        raise ConnectionResetError("connection reset by peer")


class FakeHttp:
    """Routes URL prefixes to handlers; a handler returns a FakeResponse or raises."""

    def __init__(self, routes=None):
        self.routes = dict(routes or {})
        self.calls: list[dict] = []

    def request(self, method, url, *, headers=None, body=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {}), "timeout": timeout})
        for prefix, handler in self.routes.items():
            if url.startswith(prefix):
                result = handler() if callable(handler) else handler
                if isinstance(result, BaseException):
                    raise result
                return result
        raise AssertionError(f"unexpected request to {url}")

    def urls(self):
        return [c["url"] for c in self.calls]


class ExplodingHttp:
    def request(self, *a, **k):
        raise AssertionError("no HTTP expected")


def make_tar_gz(files: dict[str, bytes], *, top: str | None = "llama-b7000", links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if top:
            d = tarfile.TarInfo(top)
            d.type = tarfile.DIRTYPE
            d.mode = 0o755
            tf.addfile(d)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size = len(data)
            info.mode = 0o644  # deliberately not executable: the installer must chmod +x
            tf.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)
    return buf.getvalue()


def make_zip(files: dict[str, bytes], links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return buf.getvalue()


def raw_tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for info, data in members:
            if data is not None:
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            else:
                tf.addfile(info)
    return buf.getvalue()


def tinfo(name, type_=tarfile.REGTYPE, linkname="", mode=0o644):
    info = tarfile.TarInfo(name)
    info.type = type_
    info.linkname = linkname
    info.mode = mode
    return info


def published(name: str, data: bytes, tag: str = "b7000", *, digest: bool = True, size: int | None = None) -> dict:
    a = {
        "name": name,
        "size": len(data) if size is None else size,
        "browser_download_url": f"{DL}/{tag}/{name}",
        "state": "uploaded",
    }
    if digest:
        a["digest"] = "sha256:" + hashlib.sha256(data).hexdigest()
    return a


def make_release(tag: str, assets: list[dict], when: str = "2026-09-20T10:00:00Z", **extra) -> dict:
    r = {"tag_name": tag, "published_at": when, "prerelease": True, "draft": False, "assets": assets,
         "html_url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}"}
    r.update(extra)
    return r


def serve(releases: list[dict], files: dict[str, bytes], tag: str = "b7000", **overrides) -> FakeHttp:
    """A FakeHttp answering the release list and each archive download."""
    routes = {API: lambda: FakeResponse(200, json.dumps(releases).encode())}
    for name, data in files.items():
        routes[f"{DL}/{tag}/{name}"] = (lambda d=data: FakeResponse(200, d))
    routes.update(overrides)
    return FakeHttp(routes)


@pytest.fixture(autouse=True)
def no_live_vulkan_probe(monkeypatch):
    """Keep plan_variants deterministic: never probe this machine's Vulkan loader."""
    monkeypatch.setattr(ri, "_system_has_vulkan_loader", lambda: False)
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 39))  # a modern Linux, whatever runs the tests
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def plenty_of_disk_space(monkeypatch):
    """The installer checks the real free space first; a test must not depend on the machine's disk.

    (A nearly full CI disk once turned every install test into "not enough free disk space".)
    Tests about the check itself set their own figure.
    """
    monkeypatch.setattr(ri.shutil, "disk_usage", lambda path: SimpleNamespace(total=10**13, used=10**12, free=9 * 10**12))


def names(assets: list[dict]) -> list[str]:
    return [a["name"] for a in assets]


# ---------------------------------------------------------------------------
# select_assets: every OS / GPU combination from release.yml
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "os_name, arch, variant, expected",
    [
        ("Windows", "AMD64", CUDA12, ["llama-b7000-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"]),
        ("Windows", "AMD64", CUDA13, ["llama-b7000-bin-win-cuda-13.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"]),
        ("Windows", "AMD64", VULKAN, ["llama-b7000-bin-win-vulkan-x64.zip"]),
        ("Windows", "AMD64", ROCM, ["llama-b7000-bin-win-rocm-10.0-x64.zip"]),
        ("Windows", "AMD64", CPU, ["llama-b7000-bin-win-cpu-x64.zip"]),
        ("Windows", "x86_64", CPU, ["llama-b7000-bin-win-cpu-x64.zip"]),
        ("Windows", "ARM64", CPU, ["llama-b7000-bin-win-cpu-arm64.zip"]),
        ("Windows", "ARM64", CUDA13, ["llama-b7000-bin-win-cuda-13.4-arm64.zip", "cudart-llama-bin-win-cuda-13.4-arm64.zip"]),
        ("Linux", "x86_64", CUDA12, ["llama-b7000-bin-ubuntu-cuda-12.8-x64.tar.gz", "cudart-llama-b7000-bin-ubuntu-cuda-12.8-x64.tar.gz"]),
        ("Linux", "x86_64", CUDA13, ["llama-b7000-bin-ubuntu-cuda-13.4-x64.tar.gz", "cudart-llama-b7000-bin-ubuntu-cuda-13.4-x64.tar.gz"]),
        ("Linux", "x86_64", VULKAN, ["llama-b7000-bin-ubuntu-vulkan-x64.tar.gz"]),
        ("Linux", "x86_64", ROCM, ["llama-b7000-bin-ubuntu-rocm-10.0-x64.tar.gz"]),
        ("Linux", "x86_64", CPU, ["llama-b7000-bin-ubuntu-x64.tar.gz"]),
        ("Linux", "aarch64", CPU, ["llama-b7000-bin-ubuntu-arm64.tar.gz"]),
        ("Linux", "aarch64", VULKAN, ["llama-b7000-bin-ubuntu-vulkan-arm64.tar.gz"]),
        ("Linux", "aarch64", CUDA13, ["llama-b7000-bin-ubuntu-cuda-13.4-arm64.tar.gz", "cudart-llama-b7000-bin-ubuntu-cuda-13.4-arm64.tar.gz"]),
        ("Darwin", "arm64", METAL, ["llama-b7000-bin-macos-arm64.tar.gz"]),
        ("Darwin", "arm64", CPU, ["llama-b7000-bin-macos-arm64.tar.gz"]),
        ("Darwin", "x86_64", CPU, ["llama-b7000-bin-macos-x64.tar.gz"]),
    ],
)
def test_select_assets_for_every_platform(os_name, arch, variant, expected):
    assert names(select_assets(ALL_ASSETS, variant, os_name, arch)) == expected


@pytest.mark.parametrize(
    "os_name, arch, variant",
    [
        ("Linux", "x86_64", METAL),  # Metal is Apple-only
        ("Windows", "AMD64", METAL),
        ("Windows", "ARM64", CUDA12),
        ("Windows", "ARM64", VULKAN),
        ("Linux", "aarch64", CUDA12),
        ("Linux", "aarch64", ROCM),
        ("Darwin", "arm64", CUDA12),
        ("Darwin", "arm64", VULKAN),
        ("Linux", "s390x", CPU),  # published, but not an architecture the game supports
        ("FreeBSD", "amd64", CPU),
    ],
)
def test_select_assets_returns_empty_when_no_build(os_name, arch, variant):
    assert select_assets(ALL_ASSETS, variant, os_name, arch) == []


def test_cpu_never_picks_other_accelerators_or_extras():
    chosen = set()
    for os_name, arch in [("Linux", "x86_64"), ("Linux", "aarch64"), ("Windows", "AMD64"), ("Windows", "ARM64"),
                          ("Darwin", "arm64"), ("Darwin", "x86_64")]:
        chosen.update(names(select_assets(ALL_ASSETS, CPU, os_name, arch)))
    for bad in ("snapdragon", "android", "openvino", "sycl", "opencl", "vulkan", "cuda", "rocm", "xcframework", "-ui."):
        assert not any(bad in n for n in chosen), bad


@pytest.mark.parametrize(
    "variant, os_name, arch, asset_names, expected",
    [
        # extra suffixes after the architecture
        (CPU, "Linux", "x86_64", ["llama-b7001-bin-ubuntu-x64-extra.tar.gz"], ["llama-b7001-bin-ubuntu-x64-extra.tar.gz"]),
        (CPU, "Linux", "x86_64", ["llama-b7001-bin-ubuntu-22.04-x64.tar.gz"], ["llama-b7001-bin-ubuntu-22.04-x64.tar.gz"]),
        (CPU, "Linux", "x86_64", ["llama-b3000-bin-ubuntu-x64.zip"], ["llama-b3000-bin-ubuntu-x64.zip"]),
        (CPU, "Windows", "AMD64", ["llama-bin-win-cpu-x64.zip"], ["llama-bin-win-cpu-x64.zip"]),
        (CPU, "Windows", "AMD64", ["llama-b2000-bin-win-avx2-x64.zip"], ["llama-b2000-bin-win-avx2-x64.zip"]),
        (VULKAN, "Windows", "AMD64", ["LLAMA-B7001-BIN-WIN-VULKAN-X64.ZIP"], ["LLAMA-B7001-BIN-WIN-VULKAN-X64.ZIP"]),
        (VULKAN, "Linux", "x86_64", ["llama-b7001-bin-linux-vulkan-x86_64.tar.gz"], ["llama-b7001-bin-linux-vulkan-x86_64.tar.gz"]),
        (
            CUDA12, "Windows", "AMD64",
            ["llama-b7001-bin-win-cuda-12.6-x64-v2.zip", "cudart-llama-bin-win-cuda-12.6-x64-v2.zip"],
            ["llama-b7001-bin-win-cuda-12.6-x64-v2.zip", "cudart-llama-bin-win-cuda-12.6-x64-v2.zip"],
        ),
        (  # the older "cu12.x" spelling
            CUDA12, "Windows", "AMD64",
            ["llama-b4000-bin-win-cuda-cu12.2.0-x64.zip", "cudart-llama-bin-win-cu12.2.0-x64.zip"],
            ["llama-b4000-bin-win-cuda-cu12.2.0-x64.zip", "cudart-llama-bin-win-cu12.2.0-x64.zip"],
        ),
        (METAL, "Darwin", "arm64", ["llama-b7001-bin-macos-14-arm64.tar.gz"], ["llama-b7001-bin-macos-14-arm64.tar.gz"]),
    ],
)
def test_select_assets_tolerates_name_drift(variant, os_name, arch, asset_names, expected):
    assert names(select_assets(assets_from(asset_names), variant, os_name, arch)) == expected


def test_prefers_plain_name_and_newest_cuda_version():
    plain = assets_from(["llama-b1-bin-ubuntu-x64-debug.tar.gz", "llama-b1-bin-ubuntu-x64.tar.gz"])
    assert names(select_assets(plain, CPU, "Linux", "x86_64")) == ["llama-b1-bin-ubuntu-x64.tar.gz"]

    multi = assets_from([
        "llama-b1-bin-win-cuda-12.4-x64.zip", "llama-b1-bin-win-cuda-12.8-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip",
    ])
    assert names(select_assets(multi, CUDA12, "Windows", "AMD64")) == [
        "llama-b1-bin-win-cuda-12.8-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip"]


def test_cudart_same_major_used_when_exact_version_missing():
    assets = assets_from(["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip"])
    assert names(select_assets(assets, CUDA12, "Windows", "AMD64")) == [
        "llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.8-x64.zip"]


def test_cuda_without_its_runtime_archive_is_not_usable():
    assets = assets_from(["llama-b1-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-13.4-x64.zip"])
    assert select_assets(assets, CUDA12, "Windows", "AMD64") == []


def test_accelerator_suffix_is_not_mistaken_for_cpu_and_uploading_assets_skipped():
    assert select_assets(assets_from(["llama-b1-bin-ubuntu-x64-vulkan.tar.gz"]), CPU, "Linux", "x86_64") == []
    uploading = [{"name": "llama-b1-bin-ubuntu-x64.tar.gz", "size": 5, "state": "starter"}]
    assert select_assets(uploading, CPU, "Linux", "x86_64") == []
    assert select_assets([{"size": 3}, "junk", None], CPU, "Linux", "x86_64") == []  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# plan_variants
# ---------------------------------------------------------------------------


def plan_names(specs):
    return [v.name for v in plan_variants(specs)]


@pytest.mark.parametrize(
    "specs, expected",
    [
        (make_specs("Windows", "AMD64", [NVIDIA]), ["cuda-12", "vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", [NVIDIA], notes=["NVIDIA driver 581.29"]), ["cuda-13", "cuda-12", "vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", [NVIDIA], notes=["NVIDIA driver 552.44"]), ["cuda-12", "vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", [NVIDIA], notes=["NVIDIA driver version: 470.82"]), ["vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", [AMD]), ["vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", [INTEL]), ["vulkan", "cpu"]),
        (make_specs("Windows", "AMD64", []), ["cpu"]),
        (make_specs("Windows", "ARM64", [NVIDIA]), ["cuda-13", "cpu"]),
        (make_specs("Windows", "ARM64", [NVIDIA], notes=["NVIDIA driver 572.16"]), ["cpu"]),
        (make_specs("Windows", "ARM64", [AMD]), ["cpu"]),
        (make_specs("Windows", "AMD64", [PASCAL], notes=["NVIDIA driver 580.95"]), ["cuda-12", "vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", [PASCAL], notes=["NVIDIA driver 580.95.05"]), ["cuda-12", "cpu"]),
        (make_specs("Linux", "x86_64", [TURING], notes=["NVIDIA driver 580.95.05"]), ["cuda-13", "cuda-12", "cpu"]),
        (make_specs("Linux", "x86_64", [NVIDIA], flags=["avx2", "vulkan"]), ["cuda-12", "vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", [NVIDIA], flags=["avx2"]), ["cuda-12", "cpu"]),
        (make_specs("Linux", "x86_64", [NVIDIA], notes=["NVIDIA driver 580.65.06", "Vulkan loader found"]), ["cuda-13", "cuda-12", "vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", [AMD], flags=["vulkan"]), ["vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", [INTEL], notes=["vulkan: libvulkan.so.1 present"]), ["vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", [AMD]), ["cpu"]),
        (make_specs("Linux", "aarch64", [NVIDIA], flags=["vulkan"]), ["cuda-13", "vulkan", "cpu"]),
        (make_specs("Linux", "x86_64", []), ["cpu"]),
        (make_specs("Darwin", "arm64", [APPLE]), ["metal", "cpu"]),
        (make_specs("Darwin", "x86_64", []), ["cpu"]),
        (make_specs("FreeBSD", "amd64", []), ["cpu"]),
    ],
)
def test_plan_variants(specs, expected):
    assert plan_names(specs) == expected


def test_plan_reads_gpu_driver_version_field():
    new_driver = GPUInfo(name="NVIDIA GeForce RTX 5090", vendor="nvidia", vram_gb=32.0, driver_version="581.15")
    old_driver = GPUInfo(name="NVIDIA GeForce GTX 780", vendor="nvidia", vram_gb=3.0, driver_version="470.256.02")
    assert plan_names(make_specs("Windows", "AMD64", [new_driver])) == ["cuda-13", "cuda-12", "vulkan", "cpu"]
    assert plan_names(make_specs("Windows", "AMD64", [old_driver])) == ["vulkan", "cpu"]
    assert plan_names(make_specs("Linux", "aarch64", [old_driver], flags=["vulkan"])) == ["vulkan", "cpu"]


def test_plan_negative_vulkan_note_beats_live_probe(monkeypatch):
    monkeypatch.setattr(ri, "_system_has_vulkan_loader", lambda: True)
    assert plan_names(make_specs("Linux", "x86_64", [AMD], notes=["Vulkan loader not found"])) == ["cpu"]
    # With no hint from specs at all, the live probe decides.
    assert plan_names(make_specs("Linux", "x86_64", [AMD])) == ["vulkan", "cpu"]


def test_plan_always_ends_with_cpu():
    for os_name, arch in [("Windows", "AMD64"), ("Linux", "x86_64"), ("Darwin", "arm64"), ("Plan9", "mips")]:
        for gpus in ([], [NVIDIA], [AMD], [APPLE]):
            plan = plan_variants(make_specs(os_name, arch, gpus, flags=["vulkan"]))
            assert plan[-1] is CPU
            assert len({v.name for v in plan}) == len(plan)


def test_variant_lookup_and_platform_support():
    assert variant_by_name("CUDA-12") is CUDA12
    assert variant_by_name(None) is None
    assert is_platform_supported("Windows", "AMD64")
    assert is_platform_supported("Darwin", "arm64")
    assert not is_platform_supported("Linux", "s390x")
    assert not is_platform_supported("SunOS", "x86_64")


# ---------------------------------------------------------------------------
# pick_release
# ---------------------------------------------------------------------------


def test_pick_release_uses_newest_release_that_has_the_variant():
    newest = make_release("b7002", assets_from(["llama-b7002-bin-ubuntu-x64.tar.gz"], "b7002"), "2026-09-22T00:00:00Z")
    older = make_release("b7001", assets_from(release_yml_names("b7001"), "b7001"), "2026-09-21T00:00:00Z")
    draft = make_release("b7003", assets_from(release_yml_names("b7003"), "b7003"), "2026-09-23T00:00:00Z", draft=True)
    releases = [older, draft, newest]  # deliberately unsorted
    rel, assets = pick_release(releases, CPU, "Linux", "x86_64")
    assert rel["tag_name"] == "b7002"
    rel, assets = pick_release(releases, CUDA12, "Linux", "x86_64")
    assert rel["tag_name"] == "b7001"
    assert names(assets)[1].startswith("cudart-")
    assert pick_release(releases, METAL, "Linux", "x86_64") is None
    assert pick_release([], CPU, "Linux", "x86_64") is None


# ---------------------------------------------------------------------------
# fetch_releases
# ---------------------------------------------------------------------------


def test_fetch_releases_request_shape():
    http = FakeHttp({API: lambda: FakeResponse(200, json.dumps([{"tag_name": "b1"}, "junk"]).encode())})
    assert fetch_releases(http=http, limit=5) == [{"tag_name": "b1"}]
    call = http.calls[0]
    assert call["method"] == "GET"
    assert call["url"] == f"{API}?per_page=5"
    assert call["headers"]["Accept"] == "application/vnd.github+json"
    assert call["headers"]["User-Agent"].startswith("GetToWork/")
    assert "Authorization" not in call["headers"]


def test_fetch_releases_uses_github_token_without_leaking_it(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_supersecret123")
    ok = FakeHttp({API: lambda: FakeResponse(200, b"[]")})
    fetch_releases(http=ok)
    assert ok.calls[0]["headers"]["Authorization"] == "Bearer ghp_supersecret123"

    limited = FakeHttp({API: lambda: FakeResponse(403, b'{"message": "API rate limit exceeded"}', {"X-RateLimit-Remaining": "0"})})
    with pytest.raises(RuntimeInstallError) as err:
        fetch_releases(http=limited)
    assert "rate limit" in str(err.value)
    assert "ghp_supersecret123" not in str(err.value)


@pytest.mark.parametrize(
    "handler, words",
    [
        (lambda: FakeResponse(429, b"slow down"), "rate limit"),
        (lambda: FakeResponse(500, b"oops"), "HTTP 500"),
        (lambda: FakeResponse(200, b"not json"), "couldn't understand"),
        (lambda: FakeResponse(200, b'{"a": 1}'), "unusual"),
        (lambda: OSError("Network is unreachable"), "Ollama"),
        (lambda: BrokenResponse(200), "couldn't reach GitHub"),  # connection drops mid-read
    ],
)
def test_fetch_releases_friendly_errors(handler, words):
    with pytest.raises(RuntimeInstallError) as err:
        fetch_releases(http=FakeHttp({API: handler}))
    assert words in str(err.value)


# ---------------------------------------------------------------------------
# safe_extract
# ---------------------------------------------------------------------------


def write(tmp_path: Path, name: str, data: bytes) -> Path:
    p = tmp_path / name
    p.write_bytes(data)
    return p


@pytest.mark.parametrize(
    "evil_name",
    ["../evil.txt", "/abs/evil.txt", "C:/evil.txt", "C:\\evil.txt", "..\\..\\evil.txt", "ok/../../evil.txt", "\\\\server\\share\\x"],
)
def test_zip_slip_is_rejected(tmp_path, evil_name):
    archive = write(tmp_path, "evil.zip", make_zip({"fine.txt": b"ok", evil_name: b"pwned"}))
    dest = tmp_path / "out" / "dest"
    with pytest.raises(RuntimeInstallError):
        safe_extract(archive, dest)
    assert not (tmp_path / "out" / "evil.txt").exists()
    assert not (tmp_path / "evil.txt").exists()


def test_zip_symlink_escaping_is_rejected(tmp_path):
    archive = write(tmp_path, "link.zip", make_zip({"a.txt": b"a"}, links={"passwd": "../../../etc/passwd"}))
    with pytest.raises(RuntimeInstallError):
        safe_extract(archive, tmp_path / "dest")


def test_zip_in_tree_symlink_becomes_a_copy(tmp_path):
    archive = write(tmp_path, "ok.zip", make_zip({"lib/real.dll": b"REAL"}, links={"lib/alias.dll": "real.dll"}))
    dest = tmp_path / "dest"
    safe_extract(archive, dest)
    assert (dest / "lib" / "alias.dll").read_bytes() == b"REAL"


def test_zip_extracts_normal_files(tmp_path):
    archive = write(tmp_path, "ok.zip", make_zip({"llama-server.exe": b"MZ", "sub/ggml.dll": b"dll"}))
    dest = tmp_path / "dest"
    safe_extract(archive, dest)
    assert (dest / "llama-server.exe").read_bytes() == b"MZ"
    assert (dest / "sub" / "ggml.dll").read_bytes() == b"dll"


def evil_tars():
    return {
        "dotdot": [(tinfo("../evil"), b"x")],
        "absolute": [(tinfo("/tmp/evil-abs"), b"x")],
        "symlink-absolute": [(tinfo("top/passwd", tarfile.SYMTYPE, "/etc/passwd"), None)],
        "symlink-escape": [(tinfo("top/up", tarfile.SYMTYPE, "../../outside"), None)],
        "hardlink-escape": [(tinfo("top/hl", tarfile.LNKTYPE, "../outside"), None)],
        "device": [(tinfo("top/null", tarfile.CHRTYPE), None)],
        "fifo": [(tinfo("top/pipe", tarfile.FIFOTYPE), None)],
    }


@pytest.mark.parametrize("data_filter", [True, False], ids=["data-filter", "manual"])
@pytest.mark.parametrize("case", list(evil_tars()))
def test_tar_slip_is_rejected(tmp_path, monkeypatch, case, data_filter):
    monkeypatch.setattr(ri, "_HAS_TAR_DATA_FILTER", data_filter and hasattr(tarfile, "data_filter"))
    archive = write(tmp_path, "evil.tar.gz", raw_tar(evil_tars()[case]))
    dest = tmp_path / "a" / "b" / "dest"
    with pytest.raises(RuntimeInstallError):
        safe_extract(archive, dest)
    assert not (tmp_path / "a" / "b" / "evil").exists()
    assert not (tmp_path / "a" / "outside").exists()
    assert not list(dest.rglob("*"))  # nothing was written before the check failed


@pytest.mark.skipif(not POSIX, reason="symlinks and modes are POSIX features")
@pytest.mark.parametrize("data_filter", [True, False], ids=["data-filter", "manual"])
def test_tar_in_tree_symlink_and_modes(tmp_path, monkeypatch, data_filter):
    monkeypatch.setattr(ri, "_HAS_TAR_DATA_FILTER", data_filter and hasattr(tarfile, "data_filter"))
    archive = write(tmp_path, "ok.tar.gz", raw_tar([
        (tinfo("llama-b1", tarfile.DIRTYPE, mode=0o755), None),
        (tinfo("llama-b1/libllama.so.0.0.1", mode=0o644), b"ELF"),
        (tinfo("llama-b1/libllama.so", tarfile.SYMTYPE, "libllama.so.0.0.1"), None),
        (tinfo("llama-b1/llama-server", mode=0o4755), b"server"),  # setuid bit must be dropped
    ]))
    dest = tmp_path / "dest"
    safe_extract(archive, dest)
    link = dest / "llama-b1" / "libllama.so"
    assert link.is_symlink() and link.read_bytes() == b"ELF"
    mode = (dest / "llama-b1" / "llama-server").stat().st_mode
    assert mode & stat.S_IXUSR
    assert not mode & stat.S_ISUID


def test_damaged_or_unknown_archives(tmp_path):
    with pytest.raises(RuntimeInstallError, match="damaged"):
        safe_extract(write(tmp_path, "bad.zip", b"definitely not a zip"), tmp_path / "d1")
    with pytest.raises(RuntimeInstallError, match="damaged"):
        safe_extract(write(tmp_path, "bad.tar.gz", b"\x1f\x8bnope"), tmp_path / "d2")
    with pytest.raises(RuntimeInstallError, match="don't know"):
        safe_extract(write(tmp_path, "x.rar", b"rar"), tmp_path / "d3")


# ---------------------------------------------------------------------------
# ensure_llama_server: full install flows through the fake HTTP layer
# ---------------------------------------------------------------------------

LINUX_CPU_TAR = make_tar_gz({"llama-server": b"#!server", "libllama.so": b"lib", "LICENSE": b"MIT"})


def linux_cpu_setup(tag="b7000"):
    name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    rel = make_release(tag, [published(name, LINUX_CPU_TAR, tag)])
    return serve([rel], {name: LINUX_CPU_TAR}, tag), name


def no_staging_left(root: Path) -> bool:
    llama = root / "llama.cpp"
    return not llama.exists() or not any(p.name.startswith(".staging-") for p in llama.iterdir())


def test_install_linux_cpu_end_to_end(tmp_path):
    http, name = linux_cpu_setup()
    ui = make_ui()
    exe, variant = ensure_llama_server(ui, make_specs("Linux", "x86_64"), http=http, runtime_root=tmp_path)

    assert variant is CPU
    assert exe == tmp_path / "llama.cpp" / "b7000-cpu" / "llama-server"
    assert exe.read_bytes() == b"#!server"
    assert (exe.parent / "LICENSE").exists()  # flattened out of llama-b7000/
    assert not (exe.parent / "llama-b7000").exists()
    if POSIX:
        assert os.access(exe, os.X_OK)
    marker = json.loads((exe.parent / "install.json").read_text())
    assert marker["tag"] == "b7000" and marker["variant"] == "cpu"
    assert marker["assets"] == [name] and marker["exe"] == "llama-server"
    assert marker["license"] == "MIT"
    assert not list(exe.parent.glob("*.tar.gz"))  # archives are cleaned up
    assert no_staging_left(tmp_path)
    assert installed_runtimes(tmp_path) == [(exe, "b7000", "cpu")]
    assert install_info(exe)["variant"] == "cpu"
    text = output(ui)
    assert "Installing the llama.cpp engine (one-time, ~" in text
    assert "MIT" in text

    # Second launch: reused, no network at all.
    ui2 = make_ui()
    exe2, variant2 = ensure_llama_server(ui2, make_specs("Linux", "x86_64"), http=ExplodingHttp(), runtime_root=tmp_path)
    assert (exe2, variant2) == (exe, CPU)
    assert "already installed" in output(ui2)


def test_install_linux_cuda_merges_runtime_libraries(tmp_path):
    tag = "b7000"
    main = make_tar_gz({"llama-server": b"srv", "libggml-cuda.so": b"cuda"}, top=f"llama-{tag}")
    cudart_top = f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64"
    cudart = make_tar_gz({"libcudart.so.12": b"rt", "libcublas.so.12": b"blas"}, top=cudart_top)
    main_name, cudart_name = f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"{cudart_top}.tar.gz"
    cpu_name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    rel = make_release(tag, [published(main_name, main), published(cudart_name, cudart), published(cpu_name, LINUX_CPU_TAR)])
    http = serve([rel], {main_name: main, cudart_name: cudart, cpu_name: LINUX_CPU_TAR})

    specs = make_specs("Linux", "x86_64", [NVIDIA], flags=["vulkan"])
    exe, variant = ensure_llama_server(make_ui(), specs, http=http, runtime_root=tmp_path)
    assert variant is CUDA12
    folder = exe.parent
    assert folder.name == "b7000-cuda-12"
    assert {"llama-server", "libggml-cuda.so", "libcudart.so.12", "libcublas.so.12"} <= {p.name for p in folder.iterdir()}
    assert json.loads((folder / "install.json").read_text())["assets"] == [main_name, cudart_name]
    assert cpu_name not in " ".join(http.urls())  # only what was needed was downloaded


def test_install_windows_cuda_extracts_both_zips_into_one_folder(tmp_path):
    tag = "b7000"
    main = make_zip({"llama-server.exe": b"MZ", "ggml-cuda.dll": b"cuda", "ggml-cpu.dll": b"cpu"})
    cudart = make_zip({"cudart64_12.dll": b"rt", "cublas64_12.dll": b"blas", "cublasLt64_12.dll": b"lt"})
    main_name, cudart_name = f"llama-{tag}-bin-win-cuda-12.4-x64.zip", "cudart-llama-bin-win-cuda-12.4-x64.zip"
    rel = make_release(tag, [published(main_name, main), published(cudart_name, cudart)])
    http = serve([rel], {main_name: main, cudart_name: cudart})

    ui = make_ui()
    exe, variant = ensure_llama_server(ui, make_specs("Windows", "AMD64", [NVIDIA]), http=http, runtime_root=tmp_path)
    assert variant is CUDA12
    assert exe.name == "llama-server.exe"
    assert (exe.parent / "cudart64_12.dll").read_bytes() == b"rt"
    assert (exe.parent / "ggml-cuda.dll").exists()
    assert "includes NVIDIA's CUDA runtime" in output(ui)


def test_missing_variant_falls_through_to_next(tmp_path):
    tag = "b7000"
    vulkan = make_zip({"llama-server.exe": b"MZ", "ggml-vulkan.dll": b"vk"})
    name = f"llama-{tag}-bin-win-vulkan-x64.zip"
    rel = make_release(tag, [published(name, vulkan), *assets_from([f"llama-{tag}-bin-win-cpu-x64.zip"])])
    http = serve([rel], {name: vulkan})
    ui = make_ui()
    exe, variant = ensure_llama_server(ui, make_specs("Windows", "AMD64", [NVIDIA]), http=http, runtime_root=tmp_path)
    assert variant is VULKAN
    assert exe.parent.name == "b7000-vulkan"
    assert "no ready-made NVIDIA CUDA 12 build" in output(ui)


def test_explicit_variant_that_is_missing_raises(tmp_path):
    http, _ = linux_cpu_setup()
    with pytest.raises(RuntimeInstallError, match="couldn't find an official prebuilt"):
        ensure_llama_server(make_ui(), make_specs("Linux", "x86_64"), variant=VULKAN, http=http, runtime_root=tmp_path)


def test_size_mismatch_is_rejected_and_cleaned_up(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    rel = make_release("b7000", [published(name, LINUX_CPU_TAR, size=len(LINUX_CPU_TAR) + 10, digest=False)])
    http = serve([rel], {name: LINUX_CPU_TAR})
    with pytest.raises(RuntimeInstallError, match="incomplete"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert installed_runtimes(tmp_path) == []
    assert no_staging_left(tmp_path)
    assert not (tmp_path / "llama.cpp" / "b7000-cpu").exists()


def test_sha256_digest_is_verified(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    bad = published(name, LINUX_CPU_TAR)
    bad["digest"] = "sha256:" + "0" * 64
    http = serve([make_release("b7000", [bad])], {name: LINUX_CPU_TAR})
    with pytest.raises(RuntimeInstallError, match="checksum"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert no_staging_left(tmp_path)


def test_ctrl_c_during_download_cleans_up(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    rel = make_release("b7000", [published(name, LINUX_CPU_TAR)])
    http = serve([rel], {}, **{f"{DL}/b7000/{name}": lambda: FakeResponse(200, LINUX_CPU_TAR, fail_with=KeyboardInterrupt())})
    with pytest.raises(KeyboardInterrupt):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert no_staging_left(tmp_path)
    assert not list((tmp_path / "llama.cpp").rglob("*.part"))
    assert installed_runtimes(tmp_path) == []


def test_network_drop_mid_download_is_friendly(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    rel = make_release("b7000", [published(name, LINUX_CPU_TAR)])
    http = serve([rel], {}, **{f"{DL}/b7000/{name}": lambda: FakeResponse(200, LINUX_CPU_TAR, fail_with=ConnectionResetError())})
    with pytest.raises(RuntimeInstallError, match="interrupted"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert no_staging_left(tmp_path)


def test_http_error_on_download(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    rel = make_release("b7000", [published(name, LINUX_CPU_TAR)])
    http = serve([rel], {}, **{f"{DL}/b7000/{name}": lambda: FakeResponse(404, b"gone")})
    with pytest.raises(RuntimeInstallError, match="HTTP 404"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)


def test_github_token_is_not_sent_to_download_host(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_topsecret")
    http, name = linux_cpu_setup()
    ui = make_ui()
    ensure_llama_server(ui, make_specs(), http=http, runtime_root=tmp_path)
    api_calls = [c for c in http.calls if c["url"].startswith(API)]
    downloads = [c for c in http.calls if c["url"].startswith(DL)]
    assert api_calls and all("Authorization" in c["headers"] for c in api_calls)
    assert downloads and all("Authorization" not in c["headers"] for c in downloads)
    assert "ghp_topsecret" not in output(ui)


def test_offline_reuses_an_older_lower_ranked_install(tmp_path):
    http, _ = linux_cpu_setup()
    cpu_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)

    offline = FakeHttp({API: lambda: OSError("no network")})
    ui = make_ui()
    specs = make_specs("Linux", "x86_64", [NVIDIA], flags=["vulkan"])
    exe, variant = ensure_llama_server(ui, specs, http=offline, runtime_root=tmp_path)
    assert (exe, variant) == (cpu_exe, CPU)
    assert "installed earlier" in output(ui)


def test_offline_with_nothing_installed_raises(tmp_path):
    with pytest.raises(RuntimeInstallError, match="couldn't reach GitHub"):
        ensure_llama_server(make_ui(), make_specs(), http=FakeHttp({API: lambda: OSError("down")}), runtime_root=tmp_path)


def test_archive_without_llama_server_is_an_error(tmp_path):
    empty = make_tar_gz({"README.md": b"hi"})
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    http = serve([make_release("b7000", [published(name, empty)])], {name: empty})
    with pytest.raises(RuntimeInstallError, match="didn't contain llama-server"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert no_staging_left(tmp_path)


def test_no_build_for_this_computer(tmp_path):
    rel = make_release("b7000", assets_from(["llama-b7000-bin-win-cpu-x64.zip"]))
    http = serve([rel], {})
    with pytest.raises(RuntimeInstallError, match="Ollama"):
        ensure_llama_server(make_ui(), make_specs("Linux", "x86_64"), http=http, runtime_root=tmp_path)


def test_macos_metal_and_cpu_share_one_download(tmp_path):
    tar = make_tar_gz({"llama-server": b"mac", "libggml-metal.dylib": b"metal"})
    name = "llama-b7000-bin-macos-arm64.tar.gz"
    rel = make_release("b7000", [published(name, tar)])
    http = serve([rel], {name: tar})
    specs = make_specs("Darwin", "arm64", [APPLE])
    exe, variant = ensure_llama_server(make_ui(), specs, http=http, runtime_root=tmp_path)
    assert variant is METAL

    http_again = serve([rel], {})  # answers the release list but has no downloads
    exe2, variant2 = ensure_llama_server(make_ui(), specs, variant=CPU, http=http_again, runtime_root=tmp_path)
    assert variant2 is CPU and exe2 == exe
    assert not [u for u in http_again.urls() if u.startswith(DL)]


def test_explicit_installed_variant_needs_no_network(tmp_path):
    http, _ = linux_cpu_setup()
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    exe2, v = ensure_llama_server(make_ui(), make_specs("Linux", "x86_64", [NVIDIA]), variant=CPU,
                                  http=ExplodingHttp(), runtime_root=tmp_path)
    assert exe2 == exe and v is CPU


def test_not_enough_disk_space(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()

    class Usage:
        free = 1024

    monkeypatch.setattr(ri.shutil, "disk_usage", lambda path: Usage())
    with pytest.raises(RuntimeInstallError, match="disk space"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)


def test_installed_runtimes_ignores_broken_installs(tmp_path):
    llama = tmp_path / "llama.cpp"
    (llama / "b1-cpu").mkdir(parents=True)  # no marker
    (llama / "b2-cpu").mkdir()
    (llama / "b2-cpu" / "install.json").write_text(json.dumps({"tag": "b2", "variant": "cpu", "exe": "llama-server"}))
    # marker present but executable missing -> ignored
    good = llama / "b10-vulkan"
    good.mkdir()
    (good / "llama-server").write_bytes(b"x")
    (good / "install.json").write_text(json.dumps({"tag": "b10", "variant": "vulkan", "exe": "llama-server"}))
    (llama / "b3-cpu").mkdir()
    (llama / "b3-cpu" / "install.json").write_text("{not json")
    assert installed_runtimes(tmp_path) == [(good / "llama-server", "b10", "vulkan")]
    assert installed_runtimes(tmp_path / "nowhere") == []


def test_runtime_explainer_mentions_the_key_ideas():
    text = ri.RUNTIME_EXPLAINER
    for word in ("llama-server", "CUDA", "Vulkan", "Metal", "CPU", "MIT", "127.0.0.1"):
        assert word in text


def test_final_rename_retries_through_a_brief_windows_lock(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        if Path(src).name == "install":  # the final "staging -> install dir" rename
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise PermissionError(13, "file in use by antivirus")
        return real_replace(src, dst)

    monkeypatch.setattr(ri.os, "replace", flaky_replace)
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert exe.is_file() and attempts["n"] == 2


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_linux_cuda_builds_are_skipped_on_glibc_older_than_2_38(monkeypatch):
    # The CUDA builds are made on Ubuntu 24.04; Ubuntu 22.04 / Debian 12 can't run them.
    specs = make_specs("Linux", "x86_64", [NVIDIA], flags=["vulkan"])
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 35))
    assert plan_names(specs) == ["vulkan", "cpu"]
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 38))
    assert plan_names(specs) == ["cuda-12", "vulkan", "cpu"]
    monkeypatch.setattr(ri, "_glibc_version", lambda: None)  # musl / unknown: let the start-up fallback decide
    assert plan_names(specs) == ["cuda-12", "vulkan", "cpu"]


def test_glibc_version_reads_this_computer(monkeypatch):
    import platform

    monkeypatch.setattr(platform, "system", lambda: "Linux")
    monkeypatch.setattr(platform, "libc_ver", lambda: ("glibc", "2.35"))
    assert _REAL_GLIBC_VERSION() == (2, 35)
    monkeypatch.setattr(platform, "libc_ver", lambda: ("", ""))  # musl (Alpine) and friends
    assert _REAL_GLIBC_VERSION() is None
    monkeypatch.setattr(platform, "system", lambda: "Windows")
    assert _REAL_GLIBC_VERSION() is None


def test_nvidia_card_without_a_working_driver_gets_no_cuda_build():
    nouveau = GPUInfo(name="NVIDIA GeForce GTX 1060", vendor="nvidia", vram_gb=0.0)
    assert plan_names(make_specs("Linux", "x86_64", [nouveau], flags=["vulkan"])) == ["vulkan", "cpu"]
    # With a driver version known, CUDA is still offered.
    known = GPUInfo(name="NVIDIA GeForce GTX 1060", vendor="nvidia", vram_gb=0.0, driver_version="550.1")
    assert plan_names(make_specs("Linux", "x86_64", [known])) == ["cuda-12", "cpu"]


def test_engine_can_use_gpu():
    assert ri.engine_can_use_gpu(make_specs("Linux", "x86_64", [AMD], flags=["vulkan"])) is True
    assert ri.engine_can_use_gpu(make_specs("Linux", "x86_64", [AMD])) is False  # no Vulkan loader
    assert ri.engine_can_use_gpu(make_specs("Darwin", "x86_64", [])) is False


def test_cuda_license_text_names_nvidia_terms():
    assert ri.license_text(CPU) == "MIT license"
    text = ri.license_text(CUDA12)
    assert "llama.cpp: MIT" in text and "NVIDIA" in text and "NOTICE.md" in text


def test_cuda_install_records_each_archive_license(tmp_path):
    tag = "b7000"
    main = make_tar_gz({"llama-server": b"srv"}, top=f"llama-{tag}")
    cudart_top = f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64"
    cudart = make_tar_gz({"libcudart.so.12": b"rt"}, top=cudart_top)
    main_name, cudart_name = f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"{cudart_top}.tar.gz"
    rel = make_release(tag, [published(main_name, main), published(cudart_name, cudart)])
    http = serve([rel], {main_name: main, cudart_name: cudart})
    ui = make_ui()
    exe, _ = ensure_llama_server(ui, make_specs("Linux", "x86_64", [NVIDIA]), http=http, runtime_root=tmp_path)
    marker = json.loads((exe.parent / "install.json").read_text())
    assert marker["licenses"] == {main_name: "MIT", cudart_name: "NVIDIA CUDA EULA"}
    text = " ".join(output(ui).split())
    assert "NVIDIA's own license terms" in text and "(MIT license)" not in text


def test_stale_github_token_is_ignored_with_a_warning(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_expired")
    answers = iter([FakeResponse(401, b'{"message": "Bad credentials"}'), FakeResponse(200, b'[{"tag_name": "b1"}]')])
    http = FakeHttp({API: lambda: next(answers)})
    ui = make_ui()
    assert fetch_releases(http=http, ui=ui) == [{"tag_name": "b1"}]
    assert http.calls[0]["headers"]["Authorization"] == "Bearer ghp_expired"
    assert "Authorization" not in http.calls[1]["headers"]
    assert "GITHUB_TOKEN" in output(ui) and "ghp_expired" not in output(ui)


def test_401_without_a_token_is_not_retried():
    http = FakeHttp({API: lambda: FakeResponse(401, b"nope")})
    with pytest.raises(RuntimeInstallError, match="HTTP 401"):
        fetch_releases(http=http)
    assert len(http.calls) == 1


def test_update_installs_the_newest_release_when_older_is_installed(tmp_path):
    http, _ = linux_cpu_setup("b7000")
    old_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    http2, _ = linux_cpu_setup("b9000")
    ui = make_ui()
    new_exe, variant = ensure_llama_server(ui, make_specs(), http=http2, runtime_root=tmp_path, update=True)
    assert variant is CPU and new_exe.parent.name == "b9000-cpu" and new_exe != old_exe
    # Already the newest: nothing downloaded.
    http3, name = linux_cpu_setup("b9000")
    ui3 = make_ui()
    again, _ = ensure_llama_server(ui3, make_specs(), http=http3, runtime_root=tmp_path, update=True)
    assert again == new_exe and "already have the newest" in output(ui3)
    assert not any(name in url for url in http3.urls())


def test_unusable_builds_are_skipped_by_automatic_setup(tmp_path):
    tag = "b7000"
    main = make_tar_gz({"llama-server": b"srv"}, top=f"llama-{tag}")
    cudart_top = f"cudart-llama-{tag}-bin-ubuntu-cuda-12.8-x64"
    cudart = make_tar_gz({"libcudart.so.12": b"rt"}, top=cudart_top)
    main_name, cudart_name = f"llama-{tag}-bin-ubuntu-cuda-12.8-x64.tar.gz", f"{cudart_top}.tar.gz"
    cpu_name = f"llama-{tag}-bin-ubuntu-x64.tar.gz"
    rel = make_release(tag, [published(main_name, main), published(cudart_name, cudart), published(cpu_name, LINUX_CPU_TAR)])
    specs = make_specs("Linux", "x86_64", [NVIDIA])
    cuda_exe, variant = ensure_llama_server(make_ui(), specs, http=serve([rel], {main_name: main, cudart_name: cudart}),
                                            runtime_root=tmp_path)
    assert variant is CUDA12
    assert ri.mark_unusable(cuda_exe, "glibc") is True
    assert ri.unusable_variants(tmp_path) == {"cuda-12"}
    assert installed_runtimes(tmp_path) == []  # the broken build no longer counts
    cpu_exe, variant = ensure_llama_server(make_ui(), specs, http=serve([rel], {cpu_name: LINUX_CPU_TAR}),
                                           runtime_root=tmp_path)
    assert variant is CPU and cpu_exe.parent.name == "b7000-cpu"


def test_mark_unusable_is_harmless_without_an_install(tmp_path):
    assert ri.mark_unusable(tmp_path / "nowhere" / "llama-server", "glibc") is False


def test_another_copy_finishing_the_same_install_mid_rename_is_reused(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()
    real_replace = os.replace

    def racing_replace(src, dst):
        if Path(src).name == "install":  # just before our rename, another game finishes first
            import shutil

            shutil.copytree(src, dst)
        return real_replace(src, dst)

    monkeypatch.setattr(ri.os, "replace", racing_replace)
    exe, variant = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert variant is CPU and exe.is_file() and exe.parent.name == "b7000-cpu"
    assert no_staging_left(tmp_path)


def test_final_rename_failure_is_a_friendly_error(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()
    real_replace = os.replace

    def broken_replace(src, dst):
        if Path(src).name == "install":
            raise OSError(5, "Input/output error")
        return real_replace(src, dst)

    monkeypatch.setattr(ri.os, "replace", broken_replace)
    with pytest.raises(RuntimeInstallError, match="couldn't move the finished engine"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)


# ---------------------------------------------------------------------------
# Round 3: certificates, redirects, old Linux, antivirus locks, tidying up
# ---------------------------------------------------------------------------

import http.server  # noqa: E402
import socketserver  # noqa: E402
import ssl  # noqa: E402
import threading  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402

from gettowork import tls  # noqa: E402


def _cert_error():
    return urllib.error.URLError(ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))


def test_a_certificate_failure_is_explained_not_blamed_on_the_network():
    http = FakeHttp({API: _cert_error()})
    with pytest.raises(RuntimeInstallError) as err:
        fetch_releases(http=http)
    text = str(err.value)
    assert "Install Certificates.command" in text and "isn't a network problem" in text
    assert "are you offline" not in text


def test_a_plain_network_failure_still_says_offline():
    http = FakeHttp({API: urllib.error.URLError(ConnectionRefusedError(111, "refused"))})
    with pytest.raises(RuntimeInstallError, match="are you offline"):
        fetch_releases(http=http)


def test_certificate_failure_during_the_download_is_explained(tmp_path):
    name = "llama-b7000-bin-ubuntu-x64.tar.gz"
    rel = make_release("b7000", [published(name, LINUX_CPU_TAR)])
    http = serve([rel], {}, **{f"{DL}/b7000/{name}": _cert_error()})
    with pytest.raises(RuntimeInstallError, match="Install Certificates"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)


def test_https_context_always_verifies_and_prefers_the_os_trust_store():
    ctx = tls.https_context(fresh=True)
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname


def test_https_context_falls_back_to_certifi_without_truststore(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "truststore", None)  # "not installed"
    ctx = tls.https_context(fresh=True)
    assert type(ctx) is ssl.SSLContext and ctx.verify_mode == ssl.CERT_REQUIRED
    try:
        import certifi  # noqa: F401
        assert ctx.cert_store_stats()["x509_ca"] > 0  # certifi's authorities are loaded
    except ImportError:  # pragma: no cover
        pass
    tls.https_context(fresh=True)  # restore the normal one for other tests


def test_urllib_http_uses_the_game_trust_store():
    http = ri.UrllibHttp()
    handlers = [h for h in http._opener.handlers if isinstance(h, urllib.request.HTTPSHandler)]
    assert handlers and handlers[0]._context is tls.https_context()


def test_is_certificate_error_sees_through_wrappers():
    assert tls.is_certificate_error(_cert_error())
    assert not tls.is_certificate_error(urllib.error.URLError(TimeoutError()))
    wrapped = RuntimeInstallError("x")
    wrapped.__cause__ = _cert_error()
    assert tls.is_certificate_error(wrapped)


class _Server:
    """A tiny local HTTP server that records the headers it receives."""

    def __init__(self, handler_fn):
        seen = self.seen = []

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):  # noqa: N802
                seen.append(dict(self.headers.items()))
                handler_fn(self)

            def log_message(self, *a):
                pass

        self.httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def test_a_redirect_to_another_host_never_carries_the_github_token():
    def ok(h):
        h.send_response(200)
        h.end_headers()
        h.wfile.write(b"[]")

    target = _Server(ok)

    def redirect(h):
        h.send_response(302)
        h.send_header("Location", f"http://localhost:{target.port}/elsewhere")
        h.end_headers()

    origin = _Server(redirect)
    try:
        resp = ri.UrllibHttp(use_proxy=False).request(
            "GET", f"http://127.0.0.1:{origin.port}/releases", headers={"Authorization": "Bearer ghp_secret_token"})
        assert resp.status == 200
        assert any(k.lower() == "authorization" for k in origin.seen[0])
        assert not any(k.lower() == "authorization" for k in target.seen[0])
    finally:
        origin.close()
        target.close()


def test_a_same_host_redirect_keeps_the_token():
    def handler(h):
        if h.path == "/renamed":
            h.send_response(200)
            h.end_headers()
            h.wfile.write(b"[]")
        else:
            h.send_response(301)
            h.send_header("Location", "/renamed")
            h.end_headers()

    server = _Server(handler)
    try:
        ri.UrllibHttp(use_proxy=False).request(
            "GET", f"http://127.0.0.1:{server.port}/old", headers={"Authorization": "Bearer ghp_secret_token"})
        assert all(any(k.lower() == "authorization" for k in seen) for seen in server.seen)
    finally:
        server.close()


def test_an_https_to_http_redirect_is_refused():
    handler = ri._SafeRedirects()
    req = urllib.request.Request("https://api.github.com/x", headers={"Authorization": "Bearer t"})
    with pytest.raises(urllib.error.HTTPError, match="insecure"):
        handler.redirect_request(req, None, 302, "Found", {}, "http://evil.example/")


def test_user_agent_names_this_project_not_llama_cpp():
    assert ri.USER_AGENT.startswith("GetToWork/") and "markelphoenix/GetToWork" in ri.USER_AGENT
    assert "ggml-org" not in ri.USER_AGENT


@pytest.mark.parametrize(
    "arch, glibc, ok",
    [("x86_64", (2, 35), True), ("x86_64", (2, 31), False), ("x86_64", (2, 28), False),
     ("aarch64", (2, 36), False), ("aarch64", (2, 38), True), ("aarch64", (2, 35), False)],
)
def test_platform_problem_knows_each_builds_glibc_minimum(arch, glibc, ok):
    problem = ri.platform_problem("Linux", arch, glibc=glibc)
    assert (problem is None) == ok
    if not ok:
        assert "glibc" in problem and "Ollama" in problem


def test_platform_problem_checks_the_macos_version():
    assert ri.platform_problem("Darwin", "arm64", macos=(14, 5)) is None
    assert "13.3" in ri.platform_problem("Darwin", "arm64", macos=(12, 7))


def test_the_macos_10_16_compatibility_answer_never_blocks_the_engine(monkeypatch):
    """Pythons built with an old macOS SDK (Anaconda's Intel Python...) are told "10.16"."""
    import platform as _platform

    from gettowork import specs as _specs

    assert ri.platform_problem("Darwin", "x86_64", macos=(10, 16)) is None
    monkeypatch.setattr(_platform, "system", lambda: "Darwin")
    monkeypatch.setattr(_platform, "mac_ver", lambda: ("10.16", ("", "", ""), "x86_64"))
    monkeypatch.setattr(_specs, "_run", lambda args, timeout=5.0: ("14.5\n", "ok"))
    assert _specs.macos_release() == "14.5"
    assert ri._macos_version() == (14, 5)
    monkeypatch.setattr(_specs, "_run", lambda args, timeout=5.0: (None, "missing"))
    monkeypatch.setattr(_specs.subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(OSError("no")))
    assert _specs.macos_release() == ""
    assert ri._macos_version() is None
    assert ri.platform_problem("Darwin", "x86_64") is None


def test_plan_variants_leaves_out_vulkan_on_arm64_with_an_older_glibc(monkeypatch):
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 36))  # Raspberry Pi OS Bookworm
    plan = plan_variants(make_specs("Linux", "aarch64", gpus=[AMD], flags=["vulkan"]))
    assert [v.name for v in plan] == ["cpu"]


def test_nothing_is_downloaded_when_no_build_can_run_here(tmp_path, monkeypatch):
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 36))
    with pytest.raises(RuntimeInstallError, match="newer Linux"):
        ensure_llama_server(make_ui(), make_specs("Linux", "aarch64"), http=ExplodingHttp(), runtime_root=tmp_path)


def test_a_build_marked_unusable_is_never_downloaded_again(tmp_path):
    http, _ = linux_cpu_setup()
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert ri.mark_unusable(exe, "glibc")
    for _ in range(2):  # the failure menu's "retry", or the next launch
        with pytest.raises(RuntimeInstallError, match="won't download it again"):
            ensure_llama_server(make_ui(), make_specs(), http=ExplodingHttp(), runtime_root=tmp_path)
    with pytest.raises(RuntimeInstallError, match="won't download it again"):
        ensure_llama_server(make_ui(), make_specs(), variant=CPU, http=ExplodingHttp(), runtime_root=tmp_path)
    assert ri.engine_problem(make_specs(), runtime_root=tmp_path) is not None
    assert ri.usable_plan(make_specs(), runtime_root=tmp_path) == []


def test_an_antivirus_lock_on_the_fresh_download_is_waited_out(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()
    real_replace = os.replace
    attempts = {"n": 0}

    def flaky_replace(src, dst):
        if str(src).endswith(".part"):
            attempts["n"] += 1
            if attempts["n"] <= 3:
                raise PermissionError(13, "The process cannot access the file because it is being used by another process")
        return real_replace(src, dst)

    monkeypatch.setattr(ri.os, "replace", flaky_replace)
    monkeypatch.setattr(ri.time, "sleep", lambda s: None)
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert exe.is_file() and attempts["n"] == 4


def test_a_locked_archive_that_cant_be_deleted_doesnt_matter(tmp_path, monkeypatch):
    http, name = linux_cpu_setup()
    real_unlink = Path.unlink

    def locked_unlink(self, *a, **k):
        if self.name == name:
            raise PermissionError(13, "in use by antivirus")
        return real_unlink(self, *a, **k)

    monkeypatch.setattr(Path, "unlink", locked_unlink)
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert exe.is_file()


def test_other_disk_errors_while_installing_are_friendly(tmp_path, monkeypatch):
    http, _ = linux_cpu_setup()

    def full_disk(*a, **k):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(ri, "_merge_tree", full_disk)
    with pytest.raises(RuntimeInstallError, match="disk filled up"):
        ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)


def test_an_engine_update_tidies_away_the_older_copy(tmp_path):
    http1, _ = linux_cpu_setup("b7000")
    old_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http1, runtime_root=tmp_path)
    http2, _ = linux_cpu_setup("b8000")
    new_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http2, runtime_root=tmp_path, update=True)
    assert old_exe.is_file() and new_exe.is_file() and old_exe != new_exe
    tidied, freed = ri.prune_old_installs(new_exe, runtime_root=tmp_path)
    assert tidied == 1 and freed > 0
    assert not old_exe.exists() and new_exe.is_file()


def test_tidying_keeps_an_engine_another_game_is_running(tmp_path):
    http1, _ = linux_cpu_setup("b7000")
    old_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http1, runtime_root=tmp_path)
    http2, _ = linux_cpu_setup("b8000")
    new_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http2, runtime_root=tmp_path, update=True)
    assert ri.prune_old_installs(new_exe, runtime_root=tmp_path, in_use={old_exe}) == (0, 0)
    assert old_exe.is_file()


def test_tidying_an_unusable_build_keeps_its_note(tmp_path):
    http, _ = linux_cpu_setup("b7000")
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    folder = exe.parent
    (folder / "big.bin").write_bytes(b"x" * 200_000)
    ri.mark_unusable(exe, "cpu_unsupported")
    keep = tmp_path / "llama.cpp" / "b9000-vulkan"
    keep.mkdir(parents=True)
    (keep / "llama-server").write_text("")
    (keep / ri.INSTALL_MARKER).write_text(json.dumps({"tag": "b9000", "variant": "vulkan", "exe": "llama-server"}))
    tidied, _ = ri.prune_old_installs(keep / "llama-server", runtime_root=tmp_path)
    assert tidied == 1
    assert not (folder / "big.bin").exists() and not exe.exists()
    assert ri.unusable_reasons(tmp_path) == {"cpu": "cpu_unsupported"}  # still remembered


# ---------------------------------------------------------------------------
# Round 4: a failed engine update marks only that install
# ---------------------------------------------------------------------------


def test_a_broken_engine_update_never_blocks_the_older_install_that_works(tmp_path):
    http, _ = linux_cpu_setup("b100")
    old_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    http2, _ = linux_cpu_setup("b200")
    new_exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http2, runtime_root=tmp_path, update=True)
    assert new_exe.parent.name == "b200-cpu"
    assert ri.mark_unusable(new_exe, "glibc")  # e.g. upstream moved its build machines to a newer Ubuntu
    assert ri.unusable_reasons(tmp_path) == {}
    assert ri.engine_problem(make_specs(), runtime_root=tmp_path) is None
    assert [v.name for v in ri.usable_plan(make_specs(), runtime_root=tmp_path)] == ["cpu"]
    exe, variant = ensure_llama_server(make_ui(), make_specs(), http=ExplodingHttp(), runtime_root=tmp_path)
    assert exe == old_exe and variant is CPU
    # Asking for the newest engine again says what's really wrong - without downloading it again.
    http3, name = linux_cpu_setup("b200")
    with pytest.raises(RuntimeInstallError, match="needs a newer llama.cpp engine") as err:
        ensure_llama_server(make_ui(), make_specs(), http=http3, runtime_root=tmp_path, update=True)
    assert "still works for other models" in str(err.value)
    assert not any(name in url for url in http3.urls())


def test_an_old_unusable_note_stops_blocking_new_downloads(tmp_path):
    http, _ = linux_cpu_setup("b100")
    exe, _ = ensure_llama_server(make_ui(), make_specs(), http=http, runtime_root=tmp_path)
    assert ri.mark_unusable(exe, "cpu_unsupported")
    assert ri.unusable_reasons(tmp_path) == {"cpu": "cpu_unsupported"}
    later = time.time() + (ri.UNUSABLE_RETRY_DAYS + 1) * 86400
    assert ri.unusable_reasons(tmp_path, now=later) == {}  # a newer release may have fixed it
    # ...but that exact release is still never downloaded again.
    http2, name = linux_cpu_setup("b100")
    with pytest.raises(RuntimeInstallError, match="won't download it again"):
        ri._install(make_ui(), http2, make_release("b100", []), [], CPU, tmp_path / "llama.cpp")
    assert not any(name in url for url in http2.urls())


def test_engine_can_use_gpu_ignores_gpu_builds_that_failed_here_for_good(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    folder = ri._llama_root(None) / "b9999-cuda-12"
    folder.mkdir(parents=True)
    (folder / "llama-server").write_bytes(b"x")
    (folder / ri.INSTALL_MARKER).write_text(json.dumps({"tag": "b9999", "variant": "cuda-12",
                                                        "exe": "llama-server"}), encoding="utf-8")
    rtx = make_specs("Linux", "x86_64", [NVIDIA])
    assert ri.engine_can_use_gpu(rtx)
    ri.mark_unusable(folder / "llama-server", "cpu_unsupported")
    assert [v.name for v in ri.usable_plan(rtx)] == ["cpu"]
    assert not ri.engine_can_use_gpu(rtx)
