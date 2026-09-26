"""Tests for packaging/fetch_engine.py, the CI script that bundles llama.cpp into the built game.

No network: a fake HTTP layer serves the GitHub release list and archives
built in memory. `--verify` runs a fake engine (a tiny shell script on
Linux/macOS, or a fake `runner` everywhere).
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import re
import struct
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from gettowork import distribution
from gettowork import runtime_install as ri

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("fetch_engine", ROOT / "packaging" / "fetch_engine.py")
fe = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fe)

REAL_DEFAULT_OPENSSL_DIR = fe.default_openssl_dir  # (the hermetic fixture swaps it out)
API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
DL = "https://github.com/ggml-org/llama.cpp/releases/download"
RAW_LICENSE = "https://raw.githubusercontent.com/ggml-org/llama.cpp/"
POSIX = os.name != "nt"
MIT = b"MIT License\n\nCopyright (c) 2023-2026 The ggml authors\n"
HTTPLIB = b"The MIT License (MIT)\n\nCopyright (c) 2017 yhirose\n"
JSONHPP = b"MIT License\n\nCopyright (c) 2013-2025 Niels Lohmann\n"
BORINGSSL = b"                                 Apache License\n                           Version 2.0, January 2004\n"
LLVM_OMP = b"The LLVM Project is under the Apache License v2.0 with LLVM Exceptions:\n"
EMBEDDED = {"llama.cpp": MIT, "cpp-httplib": HTTPLIB, "jsonhpp": JSONHPP, "BoringSSL": BORINGSSL,
            "LLVM OpenMP": LLVM_OMP}
LINUX_PARTS = ("llama.cpp", "cpp-httplib", "jsonhpp")  # what the official Ubuntu builds embed
WINDOWS_PARTS = ("LLVM OpenMP", "llama.cpp", "jsonhpp", "BoringSSL", "cpp-httplib")  # ...and the Windows ones
BORINGSSL_VERSION = "0.20260903.0"
OPENSSL_LICENSE = b"                                 Apache License\n  (OpenSSL 3's LICENSE.txt)\n"
# A slice of llama.cpp's src/llama-arch.cpp, shaped like the real file (the LLM_ARCH_NAMES table).
ARCHS = ("clip", "llama", "llama4", "qwen2", "qwen3", "qwen3moe", "phi3", "gemma", "gemma2", "gemma3", "gemma3n",
         "mistral3", "gpt-oss", "glm4moe", "granite", "granitemoe", "olmo2", "smollm3", "lfm2", "exaone4",
         "deepseek2", "command-r", "falcon-h1", "nemotron_h")
ARCH_SOURCE = ("#include \"llama-arch.h\"\n\n#include <map>\n\n"
               "static const std::map<llm_arch, const char *> LLM_ARCH_NAMES = {\n"
               + "".join(f'    {{ LLM_ARCH_{n.upper().replace("-", "_")}, "{n}" }},\n' for n in ARCHS)
               + '    { LLM_ARCH_UNKNOWN,          "(unknown)"        },\n};\n\n'
               "static const std::map<llm_kv, const char *> LLM_KV_NAMES = {\n"
               '    { LLM_KV_GENERAL_TYPE, "general.type" },\n};\n').encode()


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status=200, body=b"", headers=None):
        self.status = status
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._buf = io.BytesIO(body)

    def read(self, n=-1):
        return self._buf.read() if n is None or n < 0 else self._buf.read(n)

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class FakeHttp:
    """Answers exact URLs (or URL prefixes ending in '*'); records every request."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.calls: list[dict] = []

    def request(self, method, url, *, headers=None, body=None, timeout=30.0):
        self.calls.append({"method": method, "url": url, "headers": dict(headers or {})})
        handler = self.routes.get(url)
        if handler is None:
            for key, value in self.routes.items():
                if key.endswith("*") and url.startswith(key[:-1]):
                    handler = value
                    break
        if handler is None:
            raise AssertionError(f"unexpected request to {url}")
        result = handler() if callable(handler) else handler
        if isinstance(result, BaseException):
            raise result
        return result

    def urls(self):
        return [c["url"] for c in self.calls]


def tar_gz(files: dict[str, bytes], top: str | None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        if top:
            d = tarfile.TarInfo(top)
            d.type, d.mode = tarfile.DIRTYPE, 0o755
            tf.addfile(d)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{top}/{name}" if top else name)
            info.size, info.mode = len(data), 0o644  # not executable: the script must chmod +x
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()


def program_with_licenses(parts, code: bytes = b"") -> bytes:
    """A stand-in program with license texts embedded the way llama.cpp's cmake/license.cmake does it."""
    blob = b"\x7fELF fake program\x00" + code + b"\x00"
    for name in parts:
        title = f"License for {name}".encode()
        blob += title + b"\n" + b"=" * len(title) + b"\n\n" + EMBEDDED[name] + b"\x00\x00"
    return blob


def linux_archive(tag: str, *, extra: dict[str, bytes] | None = None, license_files: bool = True,
                  server: bytes = b"#!engine", embedded=LINUX_PARTS) -> bytes:
    files = {"llama-server": server, "libggml-base.so": b"lib", "libllama.so": b"lib"}
    if embedded:
        files["llama"] = program_with_licenses(embedded)
    if license_files:
        files.update({"LICENSE": MIT, "LICENSE-curl": b"curl license", "licenses/LICENSE-httplib": b"httplib"})
    files.update(extra or {})
    return tar_gz(files, top=f"llama-{tag}")


def tiny_pe(imports=(), delay_imports=()) -> bytes:
    """The smallest Windows program (PE32+) that imports these DLLs - enough for fetch_engine.pe_imports."""
    rva, raw = 0x1000, 0x200
    imports, delay_imports = list(imports), list(delay_imports)
    import_size, delay_size = 20 * (len(imports) + 1), 32 * (len(delay_imports) + 1)
    names, name_rvas = b"", []
    for name in imports + delay_imports:
        name_rvas.append(rva + import_size + delay_size + len(names))
        names += name.encode() + b"\0"
    tables = b"".join(struct.pack("<IIIII", 0, 0, 0, name_rvas[i], 0) for i in range(len(imports))) + bytes(20)
    tables += b"".join(struct.pack("<8I", 1, name_rvas[len(imports) + i], 0, 0, 0, 0, 0, 0)
                       for i in range(len(delay_imports))) + bytes(32)
    section = tables + names
    optional = bytearray(240)
    struct.pack_into("<H", optional, 0, 0x20B)  # PE32+
    struct.pack_into("<I", optional, 108, 16)  # 16 data directories
    struct.pack_into("<II", optional, 112 + 8 * 1, rva, import_size)
    if delay_imports:
        struct.pack_into("<II", optional, 112 + 8 * 13, rva + import_size, delay_size)
    header = (b"MZ" + bytes(0x3A) + struct.pack("<I", 0x40) + b"PE\0\0"
              + struct.pack("<HHIIIHH", 0x8664, 1, 0, 0, 0, len(optional), 0x22) + bytes(optional)
              + b".idata\0\0" + struct.pack("<IIII", len(section), rva, len(section), raw) + bytes(16))
    return header + bytes(raw - len(header)) + section


def tiny_elf(needed=(), *, machine: int = 62, bits: int = 64, big: bool = False, extra: bytes = b"") -> bytes:
    """The smallest Linux program/library (ELF) that needs these libraries - enough for fetch_engine.elf_info."""
    end = ">" if big else "<"
    base, ehsize = 0x400000, 64 if bits == 64 else 52
    phentsize, phnum = (56 if bits == 64 else 32), 2
    dyn_format = end + ("qQ" if bits == 64 else "iI")
    strtab, offsets = b"\0", []
    for name in needed:
        offsets.append(len(strtab))
        strtab += name.encode() + b"\0"
    dyn_off = ehsize + phentsize * phnum
    dyn_size = struct.calcsize(dyn_format) * (len(offsets) + 3)
    str_off = dyn_off + dyn_size
    total = str_off + len(strtab) + len(extra)
    dyn = b"".join(struct.pack(dyn_format, 1, o) for o in offsets)
    dyn += struct.pack(dyn_format, 5, base + str_off) + struct.pack(dyn_format, 10, len(strtab))
    dyn += struct.pack(dyn_format, 0, 0)
    if bits == 64:
        header = struct.pack(end + "HHIQQQIHHHHHH", 3, machine, 1, 0, ehsize, 0, 0, ehsize, phentsize, phnum, 0, 0, 0)
        load = struct.pack(end + "IIQQQQQQ", 1, 5, 0, base, base, total, total, 0x1000)
        dynamic = struct.pack(end + "IIQQQQQQ", 2, 6, dyn_off, base + dyn_off, base + dyn_off, dyn_size, dyn_size, 8)
    else:
        header = struct.pack(end + "HHIIIIIHHHHHH", 3, machine, 1, 0, ehsize, 0, 0, ehsize, phentsize, phnum, 0, 0, 0)
        load = struct.pack(end + "IIIIIIII", 1, 0, base, base, total, total, 5, 0x1000)
        dynamic = struct.pack(end + "IIIIIIII", 2, dyn_off, base + dyn_off, base + dyn_off, dyn_size, dyn_size, 6, 4)
    ident = b"\x7fELF" + bytes([2 if bits == 64 else 1, 2 if big else 1, 1, 0]) + bytes(8)
    blob = ident + header + load + dynamic + dyn + strtab + extra
    assert len(blob) == total
    return blob


def asset(name: str, data: bytes, tag: str, *, digest: bool = True, bad_digest: bool = False) -> dict:
    a = {"name": name, "size": len(data), "browser_download_url": f"{DL}/{tag}/{name}", "state": "uploaded"}
    if digest:
        a["digest"] = "sha256:" + ("0" * 64 if bad_digest else hashlib.sha256(data).hexdigest())
    return a


def release(tag: str, assets: list[dict], when: str, **extra) -> dict:
    r = {"tag_name": tag, "published_at": when, "prerelease": True, "draft": False, "assets": assets,
         "html_url": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}"}
    r.update(extra)
    return r


class Publisher:
    """Builds releases + archives and serves them through one FakeHttp."""

    def __init__(self):
        self.releases: list[dict] = []
        self.files: dict[str, bytes] = {}
        self.extra_routes: dict = {}

    def add(self, tag: str, builds: dict[str, bytes], when: str, **asset_kwargs) -> dict:
        assets = []
        for name, data in builds.items():
            assets.append(asset(name, data, tag, **asset_kwargs))
            self.files[f"{DL}/{tag}/{name}"] = data
        rel = release(tag, assets, when)
        self.releases.append(rel)
        return rel

    def http(self) -> FakeHttp:
        routes = {f"{API}?per_page={fe.SEARCH_RELEASES}": lambda: FakeResponse(200, json.dumps(self.releases).encode())}
        for rel in self.releases:
            routes[f"{API}/tags/{rel['tag_name']}"] = (lambda r=rel: FakeResponse(200, json.dumps(r).encode()))
        routes[f"{API}/tags/*"] = lambda: FakeResponse(404, b'{"message": "Not Found"}')
        for url, data in self.files.items():
            routes[url] = (lambda d=data: FakeResponse(200, d))
        for rel in self.releases:  # every release's model-architecture table (src/llama-arch.cpp)
            routes[f"{RAW_LICENSE}{rel['tag_name']}/src/llama-arch.cpp"] = lambda: FakeResponse(200, ARCH_SOURCE)
        routes.update(self.extra_routes)
        return FakeHttp(routes)

    def serve_source_licenses(self, tag: str, *, boringssl_version: str | None = BORINGSSL_VERSION) -> None:
        """Serve the license files from llama.cpp's source (and BoringSSL's) that fetch_engine falls back to."""
        source = f"{RAW_LICENSE}{tag}/"
        cmake = f'set(BORINGSSL_VERSION "{boringssl_version}" CACHE STRING "BoringSSL version")\n'.encode()
        self.extra_routes.update({
            f"{source}LICENSE": lambda: FakeResponse(200, MIT),
            f"{source}vendor/cpp-httplib/LICENSE": lambda: FakeResponse(200, HTTPLIB),
            f"{source}licenses/LICENSE-jsonhpp": lambda: FakeResponse(200, JSONHPP),
            f"{source}vendor/cpp-httplib/CMakeLists.txt": lambda: FakeResponse(200, cmake if boringssl_version else b"# none"),
            f"https://raw.githubusercontent.com/google/boringssl/{boringssl_version}/LICENSE":
                lambda: FakeResponse(200, BORINGSSL),
        })


def linux_names(tag: str) -> tuple[str, str]:
    return f"llama-{tag}-bin-ubuntu-vulkan-x64.tar.gz", f"llama-{tag}-bin-ubuntu-x64.tar.gz"


def standard_publisher() -> Publisher:
    """b7002 (newest) has only the Linux Vulkan build; b7001 has Vulkan + CPU; b7003 is a draft."""
    pub = Publisher()
    vk, cpu = linux_names("b7002")
    pub.add("b7002", {vk: linux_archive("b7002")}, "2026-09-22T00:00:00Z")
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001", extra={"libggml-vulkan.so": b"vk"}), cpu: linux_archive("b7001")},
            "2026-09-21T00:00:00Z")
    vk, cpu = linux_names("b7003")
    draft = pub.add("b7003", {vk: linux_archive("b7003"), cpu: linux_archive("b7003")}, "2026-09-23T00:00:00Z")
    draft["draft"] = True
    return pub


class FakeRunner:
    """Like subprocess.run for `llama-server --version`; exit codes per build folder name."""

    def __init__(self, codes=None, error=None):
        self.codes = dict(codes or {})
        self.error = error
        self.calls: list[tuple[list[str], dict]] = []

    def __call__(self, args, **kwargs):
        self.calls.append((list(args), kwargs))
        if self.error is not None:
            raise self.error
        code = self.codes.get(Path(args[0]).parent.name, 0)
        return subprocess.CompletedProcess(args, code, stdout=b"version: 7001 (abcdef0)\nbuilt with cc\n")


@pytest.fixture(autouse=True)
def hermetic(monkeypatch):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(fe, "default_vc_runtime_dir", lambda: None)  # never this computer's System32
    monkeypatch.setattr(fe, "default_openssl_dir", lambda arch: None)  # ...or its /usr/lib
    for var in ("GETTOWORK_DISTRIBUTION", "GETTOWORK_ENGINE_DIR", "GETTOWORK_ALLOW_ENGINE_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(distribution, "_cache", distribution.Distribution())


def run(argv, *, http, runner=None, host=("linux", "x64")) -> tuple[int, str]:
    out = io.StringIO()
    code = fe.main(argv, http=http, runner=runner or FakeRunner(), host=host, out=out)
    return code, out.getvalue()


def linux_args(dest: Path, *extra: str) -> list[str]:
    return ["--os", "linux", "--arch", "x64", "--variants", "vulkan,cpu", "--dest", str(dest), *extra]


# ---------------------------------------------------------------------------
# Choosing the release
# ---------------------------------------------------------------------------


def test_it_picks_the_newest_release_that_has_every_requested_build(tmp_path):
    pub = standard_publisher()
    http = pub.http()
    code, out = run(linux_args(tmp_path / "engine"), http=http)
    assert code == 0, out
    assert out.strip().splitlines()[-1] == "LLAMA_CPP_TAG=b7001"
    assert sorted(p.name for p in (tmp_path / "engine").iterdir()) == ["b7001-cpu", "b7001-vulkan"]
    assert http.calls[0]["url"] == f"{API}?per_page={fe.SEARCH_RELEASES}"
    downloads = [u for u in http.urls() if u.startswith(DL)]
    assert downloads and all("/b7001/" in u for u in downloads)  # b7002 lacks CPU, b7003 is a draft


def test_choose_release_directly():
    pub = standard_publisher()
    rel, picked = fe.choose_release(pub.releases, [ri.VULKAN, ri.CPU], "Linux", "x64")
    assert rel["tag_name"] == "b7001" and set(picked) == {"vulkan", "cpu"}
    rel, picked = fe.choose_release(pub.releases, [ri.VULKAN], "Linux", "x64")
    assert rel["tag_name"] == "b7002"
    assert fe.choose_release(pub.releases, [ri.METAL], "Linux", "x64") is None
    assert fe.choose_release([], [ri.CPU], "Linux", "x64") is None


def test_an_exact_tag_is_fetched_by_name(tmp_path):
    pub = standard_publisher()
    http = pub.http()
    code, out = run(["--os", "linux", "--arch", "x64", "--variants", "vulkan", "--dest", str(tmp_path / "e"),
                     "--tag", "b7002"], http=http)
    assert code == 0, out
    assert http.calls[0]["url"] == f"{API}/tags/b7002"
    assert out.strip().splitlines()[-1] == "LLAMA_CPP_TAG=b7002"
    assert (tmp_path / "e" / "b7002-vulkan" / "llama-server").is_file()


def test_the_pinned_release_is_committed_and_readable():
    tag = fe.pinned_tag()
    assert tag.startswith("b") and tag[1:].isdigit()


def pin_file(tmp_path, monkeypatch, pub: Publisher, tag: str = "b7001", *, skip=(), wrong=()) -> Path:
    """A llama_cpp_tag.txt pinning `tag` and the SHA-256 of its archives (as `sha256sum` prints them)."""
    lines = ["# the release the game ships", tag]
    for url, data in pub.files.items():
        name = url.rsplit("/", 1)[1]
        if f"/{tag}/" in url and name not in skip:
            digest = hashlib.sha256(b"not it" if name in wrong else data).hexdigest()
            lines.append(f"{digest}  {name}")
    pinned = tmp_path / "llama_cpp_tag.txt"
    pinned.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(fe, "PINNED_TAG_FILE", pinned)
    return pinned


def test_tag_pinned_fetches_the_committed_release(tmp_path, monkeypatch):
    """The game builds all fetch the release named in packaging/llama_cpp_tag.txt, never "the newest"."""
    pub = standard_publisher()  # b7002 and b7003 are newer - and must be ignored
    pin_file(tmp_path, monkeypatch, pub)
    http = pub.http()
    code, out = run(linux_args(tmp_path / "e", "--tag", "pinned"), http=http)
    assert code == 0, out
    assert http.calls[0]["url"] == f"{API}/tags/b7001"
    assert out.strip().splitlines()[-1] == "LLAMA_CPP_TAG=b7001"
    assert fe.pinned_release().digests == {name: hashlib.sha256(pub.files[f"{DL}/b7001/{name}"]).hexdigest()
                                           for name in linux_names("b7001")}


def test_tag_pinned_refuses_a_release_file_replaced_after_the_pin(tmp_path, monkeypatch, capsys):
    """The pin is of the reviewed bytes, not just the name: a re-uploaded archive (GitHub now reports another
    SHA-256 for it) fails the build, and so do bytes that don't match the pin even when GitHub's own
    fingerprint agrees with them."""
    pub = standard_publisher()
    vk, cpu = linux_names("b7001")
    pin_file(tmp_path, monkeypatch, pub, wrong=(cpu,))
    code, out = run(linux_args(tmp_path / "a", "--tag", "pinned"), http=pub.http())
    assert code == 1
    err = capsys.readouterr().err
    assert "GitHub now reports a different SHA-256 for " + cpu in err and "changed after the pin" in err
    assert "LLAMA_CPP_TAG" not in out and not (tmp_path / "a").exists()  # stopped before downloading anything

    # GitHub publishes no fingerprint at all, and the bytes that arrive aren't the pinned ones.
    pub = Publisher()
    pub.add("b7001", {vk: linux_archive("b7001"), cpu: linux_archive("b7001")}, "2026-09-21T00:00:00Z", digest=False)
    pin_file(tmp_path, monkeypatch, pub, wrong=(vk,))
    code, out = run(linux_args(tmp_path / "b", "--tag", "pinned"), http=pub.http())
    assert code == 1 and "checksum" in capsys.readouterr().err
    # ...while the pinned bytes are accepted without GitHub's fingerprint (the pin is the fingerprint).
    pin_file(tmp_path, monkeypatch, pub)
    code, out = run(linux_args(tmp_path / "c", "--tag", "pinned"), http=pub.http())
    assert code == 0, out


def test_tag_pinned_needs_a_fingerprint_for_every_archive_it_uses(tmp_path, monkeypatch, capsys):
    pub = standard_publisher()
    vk, _cpu = linux_names("b7001")
    pin_file(tmp_path, monkeypatch, pub, skip=(vk,))
    code, _out = run(linux_args(tmp_path / "e", "--tag", "pinned"), http=pub.http())
    assert code == 1
    assert f"pins no SHA-256 for {vk}" in capsys.readouterr().err


def test_the_committed_pin_has_a_fingerprint_for_every_archive_the_game_builds_use():
    """build.yml's matrix (Windows vulkan+cpu, Linux vulkan+cpu, macOS arm64) and live-check.yml (Linux cpu)
    all fetch --tag pinned: each archive they use needs its SHA-256 in llama_cpp_tag.txt."""
    pin = fe.pinned_release()
    t = pin.tag
    assert set(pin.digests) == {f"llama-{t}-bin-win-vulkan-x64.zip", f"llama-{t}-bin-win-cpu-x64.zip",
                                f"llama-{t}-bin-ubuntu-vulkan-x64.tar.gz", f"llama-{t}-bin-ubuntu-x64.tar.gz",
                                f"llama-{t}-bin-macos-arm64.tar.gz"}
    for os_name, arch, variants in (("Windows", "x64", (ri.VULKAN, ri.CPU)), ("Linux", "x64", (ri.VULKAN, ri.CPU)),
                                    ("Darwin", "arm64", (ri.METAL,))):
        for variant in variants:  # the names runtime_install really picks for each build
            picked = ri.select_assets([{"name": n} for n in pin.digests], variant, os_name, arch)
            assert picked, (os_name, variant.name)


@pytest.mark.parametrize("text, message", [
    ("b7001\nnot-a-digest  llama-b7001-bin-ubuntu-x64.tar.gz\n", "isn't a '<sha256>  <archive name>' line"),
    ("b7001\n" + "a" * 64 + "  llama-b7000-bin-ubuntu-x64.tar.gz\n", "isn't an archive of the pinned release"),
    ("b7001\n" + ("a" * 64 + "  llama-b7001-x.zip\n") * 2, "twice"),
    ("aaaa\nb7001\n", "exactly one llama.cpp release"),
])
def test_a_broken_pinned_fingerprint_line_fails_clearly(tmp_path, text, message):
    pinned = tmp_path / "llama_cpp_tag.txt"
    pinned.write_text(text, encoding="utf-8")
    with pytest.raises(ri.RuntimeInstallError, match=re.escape(message)):
        fe.pinned_release(pinned)
    ok = tmp_path / "ok.txt"
    ok.write_text("b7001\nsha256:" + "A" * 64 + " *llama-b7001-bin-ubuntu-x64.tar.gz\n", encoding="utf-8")
    assert fe.pinned_release(ok).digests == {"llama-b7001-bin-ubuntu-x64.tar.gz": "a" * 64}


@pytest.mark.parametrize("text", ["", "# only comments\n", "b7001\nb7002\n", "latest\n"])
def test_a_broken_pin_file_fails_clearly(tmp_path, text):
    pinned = tmp_path / "llama_cpp_tag.txt"
    pinned.write_text(text, encoding="utf-8")
    with pytest.raises(ri.RuntimeInstallError, match="exactly one llama.cpp release"):
        fe.pinned_tag(pinned)
    with pytest.raises(ri.RuntimeInstallError, match="Couldn't read"):
        fe.pinned_tag(tmp_path / "missing.txt")


def test_an_exact_tag_missing_a_build_fails(tmp_path, capsys):
    pub = standard_publisher()
    code, _out = run(linux_args(tmp_path / "e", "--tag", "b7002"), http=pub.http())
    assert code == 1
    assert "None of llama.cpp release b7002 has all of these builds for linux/x64: vulkan, cpu" in capsys.readouterr().err


def test_an_unknown_tag_fails_with_a_clear_message(tmp_path, capsys):
    code, _out = run(linux_args(tmp_path / "e", "--tag", "b1"), http=standard_publisher().http())
    assert code == 1 and "no llama.cpp release called 'b1'" in capsys.readouterr().err


def test_no_release_with_every_build_fails(tmp_path, capsys):
    code, _out = run(["--os", "macos", "--arch", "arm64", "--variants", "metal", "--dest", str(tmp_path / "e")],
                     http=standard_publisher().http())
    assert code == 1
    assert "None of the 3 newest llama.cpp releases has all of these builds for macos/arm64: metal" in capsys.readouterr().err
    assert not (tmp_path / "e").exists() or not list((tmp_path / "e").iterdir())


# ---------------------------------------------------------------------------
# Downloading, unpacking and the install.json note
# ---------------------------------------------------------------------------


def test_the_extracted_layout_and_install_markers(tmp_path):
    pub = standard_publisher()
    dest = tmp_path / "build" / "engine"
    code, out = run(linux_args(dest), http=pub.http())
    assert code == 0, out
    vk_name, cpu_name = linux_names("b7001")
    for variant, name in (("vulkan", vk_name), ("cpu", cpu_name)):
        folder = dest / f"b7001-{variant}"
        exe = folder / "llama-server"
        assert exe.read_bytes() == b"#!engine"
        assert not (folder / "llama-b7001").exists()  # flattened out of the tarball's top folder
        if POSIX:
            assert os.access(exe, os.X_OK)
        marker = json.loads((folder / "install.json").read_text())
        assert marker["tag"] == "b7001" and marker["variant"] == variant and marker["exe"] == "llama-server"
        assert marker["assets"] == [name] and marker["licenses"] == {name: "MIT"}
        assert marker["bundled"] is True and marker["license"] == "MIT"
        assert marker["source"] == "https://github.com/ggml-org/llama.cpp/releases/tag/b7001"
        # Every license file is gathered in licenses/ (and the originals stay put), plus the texts
        # embedded in the programs that the archive doesn't carry already (llama.cpp's own is the same text).
        assert sorted(marker["license_files"]) == ["licenses/LICENSE", "licenses/LICENSE-cpp-httplib",
                                                   "licenses/LICENSE-curl", "licenses/LICENSE-httplib",
                                                   "licenses/LICENSE-jsonhpp"]
        assert marker["license_components"] == {"llama.cpp": "licenses/LICENSE",
                                                "cpp-httplib": "licenses/LICENSE-httplib",
                                                "jsonhpp": "licenses/LICENSE-jsonhpp"}
        assert (folder / "licenses" / "LICENSE").read_bytes() == MIT
        assert (folder / "licenses" / "LICENSE-jsonhpp").read_bytes() == JSONHPP
        assert "vc_runtime" not in marker
        assert (folder / "LICENSE").read_bytes() == MIT
        assert not list(folder.glob("*.tar.gz"))
    assert (dest / "b7001-vulkan" / "libggml-vulkan.so").is_file()
    assert [p.name for p in dest.iterdir() if p.name.startswith(".")] == []  # no staging left behind


def test_license_files_from_sub_folders_never_overwrite_each_other(tmp_path):
    payload = tmp_path / "build"
    (payload / "licenses").mkdir(parents=True)
    (payload / "licenses" / "LICENSE").write_bytes(b"the archive's own licenses/LICENSE")
    (payload / "LICENSE").write_bytes(MIT)
    (payload / "vendor").mkdir()
    (payload / "vendor" / "COPYING").write_bytes(b"GPL-free, promise")
    (payload / "NOTICE.md").write_bytes(MIT)
    (payload / "not-a-license.txt").write_bytes(b"x")
    found = fe.copy_licenses(payload, "b1", FakeHttp({}), fe.PlainProgress(io.StringIO()))
    assert sorted(found) == ["licenses/LICENSE", "licenses/LICENSE-2", "licenses/NOTICE.md", "licenses/vendor-COPYING"]
    assert (payload / "licenses" / "LICENSE").read_bytes() == b"the archive's own licenses/LICENSE"
    assert (payload / "licenses" / "LICENSE-2").read_bytes() == MIT


def windows_zip(*, parts=WINDOWS_PARTS, files: dict[str, bytes] | None = None, server: bytes = b"MZ") -> bytes:
    """Shaped like the official b11100 Windows zips: no top folder, and LICENSE-LLVM-OpenMP as the only
    license file (for libomp.dll) - llama.cpp's own MIT text is only embedded in its programs."""
    content = {"llama-server.exe": server, "ggml-cpu-haswell.dll": b"cpu", "libomp.dll": b"MZ omp",
               "LICENSE-LLVM-OpenMP": LLVM_OMP}
    if parts:
        content["llama.exe"] = program_with_licenses(parts, code=b"build/_deps/boringssl-src/ssl/ssl_lib.cc")
    content.update(files or {})
    return zip_bytes(content)


def windows_publisher(tag: str = "b7001", **zip_kwargs) -> Publisher:
    pub = Publisher()
    pub.add(tag, {f"llama-{tag}-bin-win-vulkan-x64.zip": windows_zip(files={"ggml-vulkan.dll": b"vk"}, **zip_kwargs),
                  f"llama-{tag}-bin-win-cpu-x64.zip": windows_zip(**zip_kwargs)}, "2026-09-21T00:00:00Z")
    return pub


def windows_args(dest: Path, *extra: str) -> list[str]:
    return ["--os", "windows", "--arch", "x64", "--variants", "vulkan,cpu", "--dest", str(dest), *extra]


def test_windows_zips_without_a_top_folder(tmp_path):
    http = windows_publisher().http()
    code, out = run(windows_args(tmp_path / "e"), http=http)
    assert code == 0, out
    marker = json.loads((tmp_path / "e" / "b7001-vulkan" / "install.json").read_text())
    assert marker["exe"] == "llama-server.exe"
    assert (tmp_path / "e" / "b7001-cpu" / "llama-server.exe").read_bytes() == b"MZ"
    assert (tmp_path / "e" / "b7001-vulkan" / "ggml-vulkan.dll").is_file()
    # Every license text was in the programs: the only source file read is the architecture table.
    assert [u for u in http.urls() if u.startswith("https://raw.")] == [f"{RAW_LICENSE}b7001/src/llama-arch.cpp"]


def test_windows_builds_carry_llama_cpps_own_license_not_just_llvm_openmps(tmp_path):
    """The b11100 zips' only license file is LICENSE-LLVM-OpenMP: that must not stand in for llama.cpp's."""
    code, out = run(windows_args(tmp_path / "e"), http=windows_publisher().http())
    assert code == 0, out
    for variant in ("vulkan", "cpu"):
        folder = tmp_path / "e" / f"b7001-{variant}"
        marker = json.loads((folder / "install.json").read_text())
        assert marker["license_components"] == {
            "llama.cpp": "licenses/LICENSE-llama.cpp", "cpp-httplib": "licenses/LICENSE-cpp-httplib",
            "jsonhpp": "licenses/LICENSE-jsonhpp", "BoringSSL": "licenses/LICENSE-BoringSSL",
            "LLVM OpenMP": "licenses/LICENSE-LLVM-OpenMP"}
        assert (folder / "licenses" / "LICENSE-llama.cpp").read_bytes() == MIT
        assert (folder / "licenses" / "LICENSE-BoringSSL").read_bytes() == BORINGSSL
        # The embedded LLVM OpenMP text is the zip's own file: kept once, not twice.
        assert sorted(p.name for p in (folder / "licenses").iterdir()) == [
            "LICENSE-BoringSSL", "LICENSE-LLVM-OpenMP", "LICENSE-cpp-httplib", "LICENSE-jsonhpp", "LICENSE-llama.cpp"]
        assert sorted(marker["license_files"]) == sorted(marker["license_components"].values())


def test_windows_builds_without_embedded_texts_fetch_every_required_license(tmp_path):
    pub = windows_publisher(parts=())
    pub.serve_source_licenses("b7001")
    http = pub.http()
    code, out = run(windows_args(tmp_path / "e"), http=http)
    assert code == 0, out
    folder = tmp_path / "e" / "b7001-cpu"
    marker = json.loads((folder / "install.json").read_text())
    assert list(marker["license_components"]) == ["llama.cpp", "cpp-httplib", "jsonhpp", "BoringSSL", "LLVM OpenMP"]
    assert (folder / "licenses" / "LICENSE-llama.cpp").read_bytes() == MIT
    assert (folder / "licenses" / "LICENSE-cpp-httplib").read_bytes() == HTTPLIB
    assert (folder / "licenses" / "LICENSE-jsonhpp").read_bytes() == JSONHPP
    assert (folder / "licenses" / "LICENSE-BoringSSL").read_bytes() == BORINGSSL
    raw = [u for u in http.urls() if u.startswith("https://raw.")]
    assert f"{RAW_LICENSE}b7001/LICENSE" in raw
    assert f"https://raw.githubusercontent.com/google/boringssl/{BORINGSSL_VERSION}/LICENSE" in raw
    assert "has no copy of llama.cpp's license" in out


def test_a_windows_build_whose_boringssl_license_cant_be_found_fails(tmp_path, capsys):
    pub = windows_publisher(parts=("llama.cpp", "cpp-httplib", "jsonhpp"))
    pub.serve_source_licenses("b7001", boringssl_version=None)  # CMakeLists.txt names no BoringSSL version
    code, out = run(windows_args(tmp_path / "e"), http=pub.http())
    assert code == 1
    err = capsys.readouterr().err
    assert "BoringSSL" in err and "never bundled without its license" in err
    assert "LLAMA_CPP_TAG" not in out
    assert list((tmp_path / "e").iterdir()) == []


def test_boringssl_found_in_a_builds_code_makes_its_license_required_anywhere(tmp_path):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    code_with_tls = {"libllama-common.so": b"\x7fELF build/_deps/boringssl-src/crypto/mem.cc"}
    pub.add("b7001", {vk: linux_archive("b7001", extra=code_with_tls), cpu: linux_archive("b7001")},
            "2026-09-21T00:00:00Z")
    pub.serve_source_licenses("b7001")
    code, out = run(linux_args(tmp_path / "e"), http=pub.http())
    assert code == 0, out
    vulkan = json.loads((tmp_path / "e" / "b7001-vulkan" / "install.json").read_text())
    cpu_marker = json.loads((tmp_path / "e" / "b7001-cpu" / "install.json").read_text())
    assert vulkan["license_components"]["BoringSSL"] == "licenses/LICENSE-BoringSSL"
    assert "BoringSSL" not in cpu_marker["license_components"]


def test_embedded_license_texts_are_read_exactly(tmp_path):
    payload = tmp_path / "build"
    payload.mkdir()
    (payload / "llama").write_bytes(program_with_licenses(("llama.cpp", "BoringSSL")))
    (payload / "libomp.dll").write_bytes(b"MZ")
    embedded, found = fe.scan_programs(payload)
    assert embedded == {"llama.cpp": MIT, "BoringSSL": BORINGSSL}  # exactly, indentation and all
    assert found == {"LLVM OpenMP"}  # (a license text naming BoringSSL isn't BoringSSL's code)
    (payload / "libllama-common.dylib").write_bytes(b"\xcf\xfa\xed\xfe build/_deps/boringssl-src/ssl/d1_both.cc")
    assert fe.scan_programs(payload)[1] == {"LLVM OpenMP", "BoringSSL"}
    assert fe.identify_license("LICENSE", MIT) == "llama.cpp"
    assert fe.identify_license("LICENSE-LLVM-OpenMP", LLVM_OMP) == "LLVM OpenMP"
    assert fe.identify_license("LICENSE-curl", b"curl") is None
    assert fe.component_named("LLVM OpenMP").name == "LLVM OpenMP" and fe.component_named("stb") is None


# ---------------------------------------------------------------------------
# Windows: the Visual C++ runtime and the DLLs each program needs
# ---------------------------------------------------------------------------


def vc_folder(root: Path, names=fe.VC_RUNTIME_DLLS) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        (root / name).write_bytes(b"MZ " + name.encode())
    return root


def test_windows_builds_get_the_visual_cpp_runtime_next_to_llama_server(tmp_path):
    server = tiny_pe(["KERNEL32.dll", "VCRUNTIME140.dll", "api-ms-win-crt-runtime-l1-1-0.dll"])
    dll = tiny_pe(["MSVCP140.dll", "VCRUNTIME140_1.dll", "ggml-base.dll", "vulkan-1.dll"], delay_imports=["WS2_32.dll"])
    pub = windows_publisher(server=server)
    pub.files[f"{DL}/b7001/llama-b7001-bin-win-vulkan-x64.zip"] = windows_zip(
        server=server, files={"ggml-vulkan.dll": dll, "ggml-base.dll": tiny_pe(["KERNEL32.dll"])})
    pub.releases[0]["assets"] = [asset(n, pub.files[f"{DL}/b7001/{n}"], "b7001")
                                 for n in ("llama-b7001-bin-win-vulkan-x64.zip", "llama-b7001-bin-win-cpu-x64.zip")]
    vc = vc_folder(tmp_path / "System32")
    code, out = run(windows_args(tmp_path / "e", "--vc-runtime", str(vc)), http=pub.http())
    assert code == 0, out
    for variant in ("vulkan", "cpu"):
        folder = tmp_path / "e" / f"b7001-{variant}"
        for name in fe.VC_RUNTIME_DLLS:
            assert (folder / name).read_bytes() == b"MZ " + name.encode()
        marker = json.loads((folder / "install.json").read_text())
        assert marker["vc_runtime"] == list(fe.VC_RUNTIME_DLLS)
    assert "Added the Visual C++ runtime" in out


def test_a_missing_visual_cpp_runtime_file_fails(tmp_path, capsys):
    vc = vc_folder(tmp_path / "System32", names=("vcruntime140.dll",))
    code, _out = run(windows_args(tmp_path / "e", "--vc-runtime", str(vc)), http=windows_publisher().http())
    assert code == 1
    assert "msvcp140.dll" in capsys.readouterr().err
    assert list((tmp_path / "e").iterdir()) == []
    assert run(windows_args(tmp_path / "e", "--vc-runtime", str(tmp_path / "nope")),
               http=windows_publisher().http())[0] == 1


def test_by_default_the_runtime_comes_from_this_windows_computer(tmp_path, monkeypatch):
    vc = vc_folder(tmp_path / "Windows" / "System32")
    monkeypatch.setattr(fe, "default_vc_runtime_dir", lambda: vc)
    code, out = run(windows_args(tmp_path / "e"), http=windows_publisher().http())
    assert code == 0, out
    assert (tmp_path / "e" / "b7001-cpu" / "vcruntime140_1.dll").is_file()
    # Linux and macOS builds never get it.
    assert run(linux_args(tmp_path / "l"), http=standard_publisher().http())[0] == 0
    assert not (tmp_path / "l" / "b7001-cpu" / "vcruntime140.dll").exists()


def test_without_a_windows_computer_the_runtime_is_skipped_with_a_warning(tmp_path):
    server = tiny_pe(["KERNEL32.dll", "VCRUNTIME140.dll"])
    code, out = run(windows_args(tmp_path / "e", "--vc-runtime", "none"), http=windows_publisher(server=server).http())
    assert code == 0, out
    assert "warning: Not adding the Visual C++ runtime" in out
    assert not (tmp_path / "e" / "b7001-cpu" / "vcruntime140.dll").exists()


def test_a_program_needing_a_dll_nobody_ships_fails_the_build(tmp_path, capsys):
    server = tiny_pe(["KERNEL32.dll", "VCRUNTIME140.dll"], delay_imports=["cudart64_12.dll"])
    vc = vc_folder(tmp_path / "System32")
    code, _out = run(windows_args(tmp_path / "e", "--vc-runtime", str(vc)), http=windows_publisher(server=server).http())
    assert code == 1
    err = capsys.readouterr().err
    assert "llama-server.exe" in err and "cudart64_12.dll" in err
    # Without the Visual C++ runtime files, MSVCP140 is the missing one - caught just the same.
    payload = tmp_path / "p"
    payload.mkdir()
    (payload / "llama-server.exe").write_bytes(tiny_pe(["MSVCP140.dll", "KERNEL32.dll"]))
    assert fe.unresolved_windows_imports(payload) == {"llama-server.exe": ["MSVCP140.dll"]}
    assert fe.unresolved_windows_imports(payload, allow=fe.VC_RUNTIME_DLLS) == {}
    vc_folder(payload)
    assert fe.unresolved_windows_imports(payload) == {}


def test_pe_imports_reads_normal_and_delay_loaded_imports_and_ignores_other_files(tmp_path):
    path = tmp_path / "x.dll"
    path.write_bytes(tiny_pe(["KERNEL32.dll", "MSVCP140.dll"], delay_imports=["vulkan-1.dll"]))
    assert fe.pe_imports(path) == ["KERNEL32.dll", "MSVCP140.dll", "vulkan-1.dll"]
    for junk in (b"", b"MZ", b"MZ" + bytes(100), b"\x7fELF" + bytes(500), tiny_pe(["A.dll"])[:300]):
        path.write_bytes(junk)
        assert fe.pe_imports(path) in (None, [])


def test_the_mac_build_is_downloaded_once_even_for_metal_and_cpu(tmp_path):
    pub = Publisher()
    name = "llama-b7001-bin-macos-arm64.tar.gz"
    pub.add("b7001", {name: linux_archive("b7001", extra={"libggml-metal.dylib": b"metal"},
                                          embedded=LINUX_PARTS + ("BoringSSL",))}, "2026-09-21T00:00:00Z")
    http = pub.http()
    code, out = run(["--os", "macos", "--arch", "arm64", "--variants", "metal,cpu", "--dest", str(tmp_path / "e")],
                    http=http)
    assert code == 0, out
    assert [u for u in http.urls() if u.startswith(DL)] == [f"{DL}/b7001/{name}"]
    assert (tmp_path / "e" / "b7001-metal" / "libggml-metal.dylib").is_file()
    assert (tmp_path / "e" / "b7001-cpu" / "llama-server").is_file()


def test_a_checksum_mismatch_fails_and_leaves_nothing_behind(tmp_path, capsys):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001"), cpu: linux_archive("b7001")}, "2026-09-21T00:00:00Z",
            bad_digest=True)
    dest = tmp_path / "e"
    code, out = run(linux_args(dest), http=pub.http())
    assert code == 1
    assert "checksum" in capsys.readouterr().err
    assert "LLAMA_CPP_TAG" not in out
    assert list(dest.iterdir()) == []  # no half-made builds, no staging


def test_a_failure_part_way_keeps_the_previous_builds(tmp_path):
    dest = tmp_path / "e"
    assert run(linux_args(dest), http=standard_publisher().http())[0] == 0
    before = (dest / "b7001-cpu" / "install.json").read_bytes()
    broken = standard_publisher()
    cpu_url = f"{DL}/b7001/{linux_names('b7001')[1]}"
    broken.extra_routes[cpu_url] = lambda: FakeResponse(404, b"gone")
    assert run(linux_args(dest), http=broken.http())[0] == 1
    assert (dest / "b7001-cpu" / "install.json").read_bytes() == before  # untouched: nothing was replaced


def test_a_missing_checksum_is_refused_unless_allowed(tmp_path, capsys):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001"), cpu: linux_archive("b7001")}, "2026-09-21T00:00:00Z", digest=False)
    code, _out = run(linux_args(tmp_path / "a"), http=pub.http())
    assert code == 1 and "no SHA-256 fingerprint" in capsys.readouterr().err
    code, out = run(linux_args(tmp_path / "b", "--allow-missing-digest"), http=pub.http())
    assert code == 0, out


def test_an_archive_without_llama_server_fails(tmp_path, capsys):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    empty = tar_gz({"README.md": b"hi", "LICENSE": MIT}, top="llama-b7001")
    pub.add("b7001", {vk: empty, cpu: empty}, "2026-09-21T00:00:00Z")
    code, _out = run(linux_args(tmp_path / "e"), http=pub.http())
    assert code == 1 and "didn't contain llama-server" in capsys.readouterr().err


def test_an_archive_without_a_license_gets_llama_cpps_license_from_the_same_release(tmp_path):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001", license_files=False, embedded=()), cpu: linux_archive("b7001")},
            "2026-09-21T00:00:00Z")
    pub.serve_source_licenses("b7001")
    code, out = run(linux_args(tmp_path / "e"), http=pub.http())
    assert code == 0, out
    marker = json.loads((tmp_path / "e" / "b7001-vulkan" / "install.json").read_text())
    assert marker["license_files"] == ["licenses/LICENSE-llama.cpp", "licenses/LICENSE-cpp-httplib",
                                       "licenses/LICENSE-jsonhpp"]
    assert (tmp_path / "e" / "b7001-vulkan" / "licenses" / "LICENSE-llama.cpp").read_bytes() == MIT
    assert "has no copy of llama.cpp's license" in out


def test_a_license_file_that_isnt_llama_cpps_doesnt_count_as_it(tmp_path):
    """Only a text with llama.cpp's copyright line counts: any other license file makes no difference."""
    payload = tmp_path / "build"
    payload.mkdir()
    (payload / "LICENSE-LLVM-OpenMP").write_bytes(LLVM_OMP)
    pub = Publisher()
    pub.serve_source_licenses("b1")
    components: dict = {}
    found = fe.copy_licenses(payload, "b1", pub.http(), fe.PlainProgress(io.StringIO()), components=components)
    assert found == ["licenses/LICENSE-LLVM-OpenMP", "licenses/LICENSE-llama.cpp"]
    assert components == {"llama.cpp": "licenses/LICENSE-llama.cpp", "LLVM OpenMP": "licenses/LICENSE-LLVM-OpenMP"}


def test_a_build_is_never_bundled_without_its_license(tmp_path, capsys):
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001", license_files=False, embedded=()), cpu: linux_archive("b7001")},
            "2026-09-21T00:00:00Z")
    pub.extra_routes[f"{RAW_LICENSE}b7001/LICENSE"] = lambda: FakeResponse(404, b"")
    code, _out = run(linux_args(tmp_path / "e"), http=pub.http())
    assert code == 1 and "never bundled without its license" in capsys.readouterr().err
    assert list((tmp_path / "e").iterdir()) == []


def test_running_again_replaces_the_builds_and_removes_older_bundled_ones(tmp_path):
    dest = tmp_path / "e"
    old = Publisher()
    vk, cpu = linux_names("b6000")
    old.add("b6000", {vk: linux_archive("b6000"), cpu: linux_archive("b6000")}, "2026-01-01T00:00:00Z")
    assert run(linux_args(dest), http=old.http())[0] == 0
    (dest / "notes.txt").write_text("keep me")
    (dest / "custom").mkdir()
    assert run(linux_args(dest), http=standard_publisher().http())[0] == 0
    assert run(linux_args(dest), http=standard_publisher().http())[0] == 0  # twice: replaces cleanly
    assert sorted(p.name for p in dest.iterdir()) == ["b7001-cpu", "b7001-vulkan", "custom", "notes.txt"]


def test_the_github_token_is_used_but_never_printed_or_sent_to_downloads(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_verysecret")
    http = standard_publisher().http()
    code, out = run(linux_args(tmp_path / "e"), http=http)
    assert code == 0
    api = [c for c in http.calls if c["url"].startswith(API)]
    downloads = [c for c in http.calls if c["url"].startswith(DL)]
    assert api and all(c["headers"].get("Authorization") == "Bearer ghp_verysecret" for c in api)
    assert downloads and all("Authorization" not in c["headers"] for c in downloads)
    assert "ghp_verysecret" not in out + capsys.readouterr().err
    assert "ghp_verysecret" not in (tmp_path / "e" / "b7001-cpu" / "install.json").read_text()


def test_github_trouble_is_a_clean_failure(tmp_path, capsys):
    limited = FakeHttp({f"{API}*": lambda: FakeResponse(403, b"API rate limit exceeded", {"X-RateLimit-Remaining": "0"})})
    assert run(linux_args(tmp_path / "e"), http=limited)[0] == 1
    assert "rate limit" in capsys.readouterr().err
    offline = FakeHttp({f"{API}*": lambda: OSError("Network is unreachable")})
    assert run(linux_args(tmp_path / "e"), http=offline)[0] == 1


def test_plain_progress_lines_for_ci_logs(tmp_path):
    code, out = run(linux_args(tmp_path / "e"), http=standard_publisher().http())
    assert code == 0
    assert "Looking for the newest llama.cpp release with vulkan, cpu builds for linux/x64..." in out
    assert "Using llama.cpp b7001" in out
    assert "Downloading llama-b7001-bin-ubuntu-x64.tar.gz" in out and "100%" in out
    assert "\x1b" not in out and "\r" not in out  # no colours or redrawn bars


def test_plain_progress_survives_a_legacy_code_page():
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="cp1252", errors="strict")
    fe.PlainProgress(out).say("llama-server \u2192 ready \u2713")
    out.flush()
    assert raw.getvalue().decode("cp1252").startswith("llama-server ? ready ?")


def test_plain_progress_without_a_known_size():
    out = io.StringIO()
    progress = fe.PlainProgress(out)
    with progress.download_progress("thing", None) as advance:
        for _ in range(25):
            advance(1_000_000)
    text = out.getvalue()
    assert "10 MB" in text and "20 MB" in text and "done (25.0 MB)" in text


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def test_usage_errors_exit_1_and_help_exits_0(tmp_path, capsys):
    http = standard_publisher().http()
    assert run(["--os", "linux"], http=http)[0] == 1
    assert run(["--os", "beos", "--arch", "x64", "--variants", "cpu", "--dest", str(tmp_path)], http=http)[0] == 1
    assert run(["--help"], http=http)[0] == 0
    assert "--variants" in capsys.readouterr().out
    assert http.calls == []


def test_unknown_or_empty_variant_lists_fail(tmp_path, capsys):
    http = standard_publisher().http()
    assert run(["--os", "linux", "--arch", "x64", "--variants", "vulkan,warp-drive", "--dest", str(tmp_path)],
               http=http)[0] == 1
    assert "Unknown engine build 'warp-drive'" in capsys.readouterr().err
    assert run(["--os", "linux", "--arch", "x64", "--variants", " , ", "--dest", str(tmp_path)], http=http)[0] == 1
    assert http.calls == []


def test_the_script_runs_as_a_program(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "packaging" / "fetch_engine.py"), "--help"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0 and "llama.cpp" in result.stdout


# ---------------------------------------------------------------------------
# --verify
# ---------------------------------------------------------------------------


def test_verify_runs_each_build_that_can_run_here(tmp_path):
    runner = FakeRunner()
    code, out = run(linux_args(tmp_path / "e", "--verify"), http=standard_publisher().http(), runner=runner)
    assert code == 0, out
    ran = sorted(Path(args[0]).parent.name for args, _kw in runner.calls)
    assert ran == ["b7001-cpu", "b7001-vulkan"]
    args, kwargs = runner.calls[0]
    assert args[1:] == ["--version"] and kwargs["timeout"] == fe.VERIFY_TIMEOUT_S
    assert kwargs["cwd"] == str(Path(args[0]).parent)
    assert "GITHUB_TOKEN" not in kwargs["env"]
    assert "version: 7001" in out
    assert out.strip().splitlines()[-1] == "LLAMA_CPP_TAG=b7001"


def test_verify_fails_when_a_build_doesnt_start(tmp_path, capsys):
    runner = FakeRunner({"b7001-vulkan": 127})
    code, out = run(linux_args(tmp_path / "e", "--verify"), http=standard_publisher().http(), runner=runner)
    assert code == 1
    assert "vulkan: exit code 127" in capsys.readouterr().err
    assert "LLAMA_CPP_TAG" not in out


def test_verify_fails_when_a_build_cant_be_run_at_all(tmp_path, capsys):
    runner = FakeRunner(error=OSError(8, "Exec format error"))
    assert run(linux_args(tmp_path / "e", "--verify"), http=standard_publisher().http(), runner=runner)[0] == 1
    assert "couldn't run it" in capsys.readouterr().err


def test_verify_skips_builds_for_another_computer(tmp_path):
    runner = FakeRunner({"b7001-cpu": 1})
    code, out = run(linux_args(tmp_path / "e", "--verify"), http=standard_publisher().http(), runner=runner,
                    host=("windows", "x64"))
    assert code == 0 and runner.calls == []
    assert "Skipping the --version check" in out


@pytest.mark.skipif(not POSIX, reason="runs a real shell-script stand-in for llama-server (Linux/macOS)")
@pytest.mark.parametrize("exit_code, expected", [(0, 0), (3, 1)])
@pytest.mark.parametrize("relative_dest", [False, True], ids=["absolute-dest", "relative-dest"])
def test_verify_with_a_real_fake_executable(tmp_path, monkeypatch, exit_code, expected, relative_dest):
    """With the real subprocess.run - and a relative --dest, exactly as the build workflow passes it."""
    script = f"#!/bin/sh\necho \"version: 7001 (fake)\"\necho \"lib path: $LD_LIBRARY_PATH\"\nexit {exit_code}\n".encode()
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: linux_archive("b7001", server=script), cpu: linux_archive("b7001", server=script)},
            "2026-09-21T00:00:00Z")
    monkeypatch.chdir(tmp_path)
    dest = Path("build") / "engine" if relative_dest else tmp_path / "e"
    out = io.StringIO()
    code = fe.main(linux_args(dest, "--verify"), http=pub.http(), host=("linux", "x64"), out=out)
    assert code == expected, out.getvalue()
    assert out.getvalue().count("version: 7001 (fake)") == 2  # both builds really ran
    assert "couldn't run it" not in out.getvalue()


def test_verify_runs_programs_by_their_full_path(tmp_path, monkeypatch):
    """Linux and macOS look a relative program up inside `cwd`: verify_builds must never pass one."""
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "install.json").write_text(json.dumps({"exe": "bin/llama-server"}))
    monkeypatch.chdir(tmp_path)
    runner = FakeRunner()
    fe.verify_builds([(ri.CPU, Path("x"))], runner=runner, host=("linux", "x64"), target=("linux", "x64"),
                     progress=fe.PlainProgress(io.StringIO()))
    args, kwargs = runner.calls[0]
    assert Path(args[0]) == tmp_path / "x" / "bin" / "llama-server"
    assert Path(kwargs["cwd"]) == tmp_path / "x" / "bin"


def test_host_platform_names_this_computer():
    os_key, arch = fe.host_platform()
    assert os_key in ("windows", "linux", "macos", None)
    assert arch in ("x64", "arm64", None)


# ---------------------------------------------------------------------------
# Linux: OpenSSL 3 ships with the engine, and every needed library is accounted for
# ---------------------------------------------------------------------------

OPENSSL_LICENSE_URL = "https://raw.githubusercontent.com/openssl/openssl/openssl-3.0.2/LICENSE.txt"


def elf_linux_archive(tag: str, *, server_needs=("libllama-server-impl.so", "libgomp.so.1", "libc.so.6"),
                      impl_needs=("libssl.so.3", "libcrypto.so.3", "libllama.so.0", "libstdc++.so.6", "libc.so.6"),
                      extra: dict[str, bytes] | None = None) -> bytes:
    """Shaped like the official b11100 Ubuntu builds: real ELF files, and the server's library links OpenSSL 3
    from the system (libssl.so.3 / libcrypto.so.3 aren't in the archive)."""
    files = {"llama-server": tiny_elf(server_needs), "libllama-server-impl.so": tiny_elf(impl_needs),
             "libllama.so.0": tiny_elf(["libggml.so.0", "libm.so.6", "libc.so.6"]),
             "libggml.so.0": tiny_elf(["libgcc_s.so.1", "libc.so.6"]),
             "llama": program_with_licenses(LINUX_PARTS), "LICENSE": MIT}
    files.update(extra or {})
    return tar_gz(files, top=f"llama-{tag}")


def openssl_folder(root: Path, *, machine: int = 62, version: bytes = b"OpenSSL 3.0.2 15 Mar 2022") -> Path:
    """A build machine's system library folder with OpenSSL 3 in it."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "libssl.so.3").write_bytes(tiny_elf(["libcrypto.so.3", "libc.so.6"], machine=machine))
    (root / "libcrypto.so.3").write_bytes(tiny_elf(["libc.so.6"], machine=machine, extra=b"\0" + version + b"\0"))
    return root


def elf_publisher(**archive_kwargs) -> Publisher:
    pub = Publisher()
    vk, cpu = linux_names("b7001")
    pub.add("b7001", {vk: elf_linux_archive("b7001", **archive_kwargs), cpu: elf_linux_archive("b7001", **archive_kwargs)},
            "2026-09-21T00:00:00Z")
    pub.extra_routes[OPENSSL_LICENSE_URL] = lambda: FakeResponse(200, OPENSSL_LICENSE)
    return pub


def test_elf_info_reads_the_needed_libraries_of_every_kind_of_elf(tmp_path):
    path = tmp_path / "lib.so"
    for bits in (64, 32):
        for big in (False, True):
            path.write_bytes(tiny_elf(["libssl.so.3", "libc.so.6"], bits=bits, big=big, machine=183))
            assert fe.elf_info(path) == fe.ElfInfo(183, ("libssl.so.3", "libc.so.6")), (bits, big)
    path.write_bytes(tiny_elf([]))
    assert fe.elf_info(path) == fe.ElfInfo(62, ())
    for junk in (b"", b"\x7fELF", b"\x7fELF fake program\x00" + bytes(100), b"MZ" + bytes(200),
                 tiny_elf(["libc.so.6"])[:80], program_with_licenses(LINUX_PARTS)):
        path.write_bytes(junk)
        assert fe.elf_info(path) is None or fe.elf_info(path).needed == ()
    assert fe.elf_info(tmp_path / "missing") is None


@pytest.mark.skipif(not sys.platform.startswith("linux") or not Path("/bin/ls").is_file(), reason="needs a Linux program")
def test_elf_info_matches_readelf_on_a_real_program():
    import shutil

    info = fe.elf_info(Path("/bin/ls").resolve())
    assert info is not None and "libc.so.6" in info.needed
    if shutil.which("readelf"):
        out = subprocess.run(["readelf", "-d", str(Path("/bin/ls").resolve())], capture_output=True, text=True).stdout
        assert list(info.needed) == re.findall(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]", out)


def test_linux_builds_carry_openssl_3_next_to_llama_server(tmp_path):
    """The official Ubuntu builds link the system's OpenSSL 3, which Steam's Linux runtime doesn't have:
    libssl.so.3 and libcrypto.so.3 are copied into each build, recorded, and their license ships too."""
    pub = elf_publisher()
    http = pub.http()
    ssl = openssl_folder(tmp_path / "usr-lib")
    code, out = run(linux_args(tmp_path / "e", "--linux-openssl", str(ssl)), http=http)
    assert code == 0, out
    for variant in ("vulkan", "cpu"):
        folder = tmp_path / "e" / f"b7001-{variant}"
        for name in fe.LINUX_OPENSSL_LIBS:
            assert (folder / name).read_bytes() == (ssl / name).read_bytes()
            assert not (folder / name).is_symlink()
        marker = json.loads((folder / "install.json").read_text())
        assert marker["openssl"] == list(fe.LINUX_OPENSSL_LIBS) and marker["openssl_version"] == "3.0.2"
        assert marker["license_components"]["OpenSSL"] == "licenses/LICENSE-OpenSSL"
        assert (folder / "licenses" / "LICENSE-OpenSSL").read_bytes() == OPENSSL_LICENSE
        assert fe.unresolved_linux_libraries(folder) == {}
    assert OPENSSL_LICENSE_URL in http.urls()
    assert "Added OpenSSL 3.0.2 from" in out


def test_a_linux_build_that_doesnt_need_openssl_gets_no_copy(tmp_path):
    pub = elf_publisher(impl_needs=("libllama.so.0", "libstdc++.so.6", "libc.so.6"))
    code, out = run(linux_args(tmp_path / "e", "--linux-openssl", str(openssl_folder(tmp_path / "ssl"))),
                    http=pub.http())
    assert code == 0, out
    folder = tmp_path / "e" / "b7001-cpu"
    assert not (folder / "libssl.so.3").exists()
    marker = json.loads((folder / "install.json").read_text())
    assert "openssl" not in marker and "OpenSSL" not in marker["license_components"]


def test_without_a_linux_build_machine_openssl_is_skipped_with_a_warning(tmp_path):
    code, out = run(linux_args(tmp_path / "e", "--linux-openssl", "none"), http=elf_publisher().http())
    assert code == 0, out
    assert "warning: Not adding OpenSSL 3" in out and "Steam's Linux runtime" in out
    assert not (tmp_path / "e" / "b7001-cpu" / "libssl.so.3").exists()


def test_missing_or_wrong_openssl_on_the_build_machine_fails(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(linux_args(tmp_path / "a", "--linux-openssl", str(empty)), http=elf_publisher().http())[0] == 1
    assert "libssl.so.3 isn't in" in capsys.readouterr().err
    arm = openssl_folder(tmp_path / "arm", machine=183)
    assert run(linux_args(tmp_path / "b", "--linux-openssl", str(arm)), http=elf_publisher().http())[0] == 1
    assert "isn't made for x64" in capsys.readouterr().err
    assert run(linux_args(tmp_path / "c", "--linux-openssl", str(tmp_path / "nope")), http=elf_publisher().http())[0] == 1


def test_a_linux_program_needing_a_library_nobody_ships_fails_the_build(tmp_path, capsys):
    pub = elf_publisher(server_needs=("libllama-server-impl.so", "libcurl.so.4", "libc.so.6"))
    code, _out = run(linux_args(tmp_path / "e", "--linux-openssl", str(openssl_folder(tmp_path / "ssl"))),
                     http=pub.http())
    assert code == 1
    err = capsys.readouterr().err
    assert "llama-server" in err and "libcurl.so.4" in err and "LINUX_RUNTIME_LIBS" in err
    assert list((tmp_path / "e").iterdir()) == []


def test_unresolved_linux_libraries_counts_links_and_the_runtime_allowlist(tmp_path):
    folder = tmp_path / "build"
    folder.mkdir()
    (folder / "llama-server").write_bytes(tiny_elf(["libllama.so.0", "libssl.so.3", "libvulkan.so.1", "libc.so.6"]))
    (folder / "libllama.so.0.4.1").write_bytes(tiny_elf(["libgomp.so.1", "libstdc++.so.6"]))
    assert fe.unresolved_linux_libraries(folder) == {"llama-server": ["libllama.so.0", "libssl.so.3"]}
    if POSIX:
        os.symlink("libllama.so.0.4.1", folder / "libllama.so.0")  # how the official builds ship their sonames
        assert fe.unresolved_linux_libraries(folder) == {"llama-server": ["libssl.so.3"]}
        assert fe.unresolved_linux_libraries(folder, allow=fe.LINUX_OPENSSL_LIBS) == {}


def test_the_default_openssl_comes_from_this_linux_computers_library_folder(tmp_path, monkeypatch):
    good = openssl_folder(tmp_path / "x86_64-linux-gnu")
    arm = openssl_folder(tmp_path / "aarch64-linux-gnu", machine=183)
    monkeypatch.setattr(fe, "LINUX_LIBRARY_DIRS", {"x64": (str(tmp_path / "none"), str(arm), str(good)),
                                                   "arm64": (str(good), str(arm))})
    monkeypatch.setattr(fe.sys, "platform", "linux")
    assert REAL_DEFAULT_OPENSSL_DIR("x64") == good  # the arm64 copy is skipped for an x64 build
    assert REAL_DEFAULT_OPENSSL_DIR("arm64") == arm
    monkeypatch.setattr(fe.sys, "platform", "win32")
    assert REAL_DEFAULT_OPENSSL_DIR("x64") is None


@pytest.mark.skipif(not (sys.platform.startswith("linux") and fe.host_platform() == ("linux", "x64")
                         and Path("/usr/lib/x86_64-linux-gnu/libcrypto.so.3").is_file()),
                    reason="needs an x86-64 Debian/Ubuntu machine with OpenSSL 3, like the build's ubuntu-22.04")
def test_the_real_system_openssl_is_found_and_its_version_read(tmp_path):
    found = REAL_DEFAULT_OPENSSL_DIR("x64")
    assert found is not None
    for name in fe.LINUX_OPENSSL_LIBS:
        assert fe.elf_info(found / name).machine == 62
    (tmp_path / "libcrypto.so.3").write_bytes((found / "libcrypto.so.3").read_bytes())
    assert re.fullmatch(r"3\.\d+\.\d+[a-z]?", fe.openssl_version(tmp_path) or "")


# ---------------------------------------------------------------------------
# The model architectures the release knows
# ---------------------------------------------------------------------------


def test_the_release_architectures_are_recorded_in_every_build(tmp_path):
    http = standard_publisher().http()
    code, out = run(linux_args(tmp_path / "e"), http=http)
    assert code == 0, out
    for variant in ("vulkan", "cpu"):
        marker = json.loads((tmp_path / "e" / f"b7001-{variant}" / "install.json").read_text())
        assert marker["architectures"] == sorted(ARCHS)
    # Read once per run, from the chosen release's own source.
    assert [u for u in http.urls() if u.endswith("llama-arch.cpp")] == [f"{RAW_LICENSE}b7001/src/llama-arch.cpp"]


def test_parse_architectures_reads_only_the_names_table():
    names = fe.parse_architectures(ARCH_SOURCE.decode())
    assert names == sorted(ARCHS) and "(unknown)" not in names and "general.type" not in names
    assert fe.parse_architectures("nothing here") == []


def test_a_build_whose_architectures_cant_be_read_fails(tmp_path, capsys):
    pub = standard_publisher()
    pub.extra_routes[f"{RAW_LICENSE}b7001/src/llama-arch.cpp"] = lambda: FakeResponse(404, b"Not Found")
    assert run(linux_args(tmp_path / "a"), http=pub.http())[0] == 1
    assert "model architectures llama.cpp b7001 knows" in capsys.readouterr().err
    pub.extra_routes[f"{RAW_LICENSE}b7001/src/llama-arch.cpp"] = lambda: FakeResponse(200, b"// moved elsewhere")
    assert run(linux_args(tmp_path / "b"), http=pub.http())[0] == 1
    assert "LLM_ARCH_NAMES" in capsys.readouterr().err
    assert not (tmp_path / "a").exists() or list((tmp_path / "a").iterdir()) == []


# ---------------------------------------------------------------------------
# The result is exactly what the game looks for
# ---------------------------------------------------------------------------


def test_the_game_finds_and_uses_the_fetched_builds(tmp_path, monkeypatch):
    dest = tmp_path / "GetToWork" / "engine"
    assert run(linux_args(dest), http=standard_publisher().http())[0] == 0
    (tmp_path / "GetToWork" / "distribution.json").write_text(json.dumps(
        {"schema": 1, "channel": "release", "engine_downloads": False, "engine_dir": "engine",
         "llama_cpp_tag": "b7001", "app_version": "0.2.0"}))
    monkeypatch.setenv("GETTOWORK_DISTRIBUTION", str(tmp_path / "GetToWork" / "distribution.json"))
    monkeypatch.setattr(ri, "_glibc_version", lambda: (2, 39))
    dist = distribution.load(refresh=True)
    assert dist.bundled and not dist.engine_downloads and dist.llama_cpp_tag == "b7001"
    runtimes = ri.installed_runtimes(tmp_path / "runtime")
    assert [(t, v) for _e, t, v in runtimes] == [("b7001", "vulkan"), ("b7001", "cpu")]
    assert all(ri.is_bundled(e) for e, _t, _v in runtimes)
    from gettowork.types import GPUInfo, SystemSpecs
    from gettowork.ui import UI
    from rich.console import Console

    amd = SystemSpecs(os_name="Linux", os_version="t", arch="x86_64", cpu_name="CPU", cpu_cores_physical=4,
                      cpu_cores_logical=8, ram_total_gb=16, ram_available_gb=12, disk_free_gb=100,
                      gpus=[GPUInfo(name="AMD Radeon", vendor="amd", vram_gb=8)], cpu_flags=["vulkan"])
    ui = UI(console=Console(file=io.StringIO(), width=200), input_fn=lambda p: "")
    exe, variant = ri.ensure_llama_server(ui, amd, http=FakeHttp({}), runtime_root=tmp_path / "runtime")
    assert exe == dest / "b7001-vulkan" / "llama-server" and variant is ri.VULKAN
