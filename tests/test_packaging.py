"""Tests for the double-click / Steam packaging (packaging/ and the build workflows).

Nothing here runs PyInstaller or touches the network:

* the build recipe (gettowork.spec) is executed with stand-ins for
  PyInstaller's classes, once per operating system;
* assemble.py packs fake build folders (a few small files shaped like
  PyInstaller's output) with a fake llama.cpp engine;
* collect_licenses.py reads the real installed dependencies of the game;
* the workflow files and Steam templates are parsed and cross-checked
  against the files and constants they refer to.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import types
import zipfile
from pathlib import Path
from typing import Any

import pytest

import gettowork
from gettowork import catalog, distribution, notices
from gettowork.types import GPUInfo, SystemSpecs

ROOT = Path(__file__).resolve().parents[1]
PACKAGING = ROOT / "packaging"
SPEC = PACKAGING / "gettowork.spec"
WORKFLOWS = ROOT / ".github" / "workflows"
STEAM = PACKAGING / "steam"
ICON = ROOT / "src" / "gettowork" / "assets" / "icon.png"
POSIX = os.name != "nt"
MIT = "MIT License\n\nCopyright (c) 2023-2026 The ggml authors\n"


def load_script(name: str) -> types.ModuleType:
    """Import one of the packaging/*.py scripts as a module (they aren't a package)."""
    spec = importlib.util.spec_from_file_location(f"packaging_{name}", PACKAGING / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module  # dataclasses look their module up while the class is made
    spec.loader.exec_module(module)
    return module


assemble = load_script("assemble")
collect_licenses = load_script("collect_licenses")
make_icon = load_script("make_icon")


def fake_engine(root: Path, *, tag: str = "b9999", variants: tuple[str, ...] = ("cpu", "vulkan"),
                exe: str = "llama-server", license_text: str = MIT) -> Path:
    """An engine folder shaped like fetch_engine.py's output: <tag>-<variant>/ with install.json."""
    root.mkdir(parents=True, exist_ok=True)
    for variant in variants:
        build = root / f"{tag}-{variant}"
        (build / "licenses").mkdir(parents=True)
        server = build / exe
        server.write_text("#!/bin/sh\necho 'version: 9999 (fake)'\n", encoding="utf-8")
        server.chmod(0o755)
        (build / "licenses" / "LICENSE").write_text(license_text, encoding="utf-8")
        marker = {"tag": tag, "variant": variant, "label": variant, "assets": [f"llama-{tag}-bin-{variant}.zip"],
                  "exe": exe, "source": f"https://github.com/ggml-org/llama.cpp/releases/tag/{tag}",
                  "license": "MIT", "licenses": {}, "installed_at": 0,
                  "license_files": ["licenses/LICENSE"], "bundled": True}
        (build / "install.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
    return root


def fake_dist(root: Path, os_key: str) -> Path:
    """A folder shaped like PyInstaller's output for `os_key` (tiny stand-in files)."""
    dist = root / "dist"
    if os_key == "macos":
        contents = dist / "Get To Work.app" / "Contents"
        for sub in ("MacOS", "Resources", "Frameworks"):
            (contents / sub).mkdir(parents=True)
        for name in ("GetToWork", "gettowork-cli"):
            (contents / "MacOS" / name).write_bytes(b"\xcf\xfa\xed\xfe fake program")
            (contents / "MacOS" / name).chmod(0o755)
        (contents / "Frameworks" / "libpython3.12.dylib").write_bytes(b"fake library")
        (contents / "Info.plist").write_text("<plist/>", encoding="utf-8")
        if POSIX:  # PyInstaller cross-links files between Frameworks and Resources
            os.symlink("../Frameworks/libpython3.12.dylib", contents / "Resources" / "libpython3.12.dylib")
        (dist / "GetToWork").mkdir()  # PyInstaller's plain folder build, also made on macOS
    else:
        folder = dist / "GetToWork"
        (folder / "_internal").mkdir(parents=True)
        ext = ".exe" if os_key == "windows" else ""
        for name in ("GetToWork", "gettowork-cli"):
            (folder / f"{name}{ext}").write_bytes(b"fake program")
            (folder / f"{name}{ext}").chmod(0o755)
        (folder / "_internal" / "base_library.zip").write_bytes(b"PK fake")
    return dist


@pytest.fixture
def licenses_file(tmp_path: Path) -> Path:
    path = tmp_path / "THIRD_PARTY_LICENSES.txt"
    path.write_text("Third-party software in Get To Work\n", encoding="utf-8")
    return path


def zip_names(path: Path) -> set[str]:
    with zipfile.ZipFile(path) as archive:
        return set(archive.namelist())


# ---------------------------------------------------------------------------
# The build recipe (gettowork.spec)
# ---------------------------------------------------------------------------


class _Built:
    """Stand-in for PyInstaller's EXE / PYZ / COLLECT / BUNDLE: records its arguments."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args, self.kwargs = args, kwargs


class _FakeImage:
    saved: list[tuple[str, dict]] = []

    def convert(self, _mode: str) -> "_FakeImage":
        return self

    def resize(self, size: tuple[int, int], _resample: Any = None) -> "_FakeImage":
        self.size = size
        return self

    def save(self, out: Any, **kwargs: Any) -> None:
        Path(out).write_bytes(b"icon")
        _FakeImage.saved.append((str(out), kwargs))


def run_spec(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, platform: str, *, pillow: bool = True) -> dict:
    """Execute gettowork.spec as PyInstaller would, with stand-ins; returns what it built."""
    built: dict[str, Any] = {"exe": []}

    class Analysis:
        def __init__(self, scripts: list[str], **kwargs: Any) -> None:
            self.scripts_in, self.kwargs = scripts, kwargs
            self.pure, self.binaries, self.datas = ["pure"], ["binaries"], ["datas"]
            # PyInstaller's own start-up scripts come first, then the entry scripts.
            self.scripts = [("pyiboot01_bootstrap", "/pyi/boot.py", "PYSOURCE"),
                            ("gui_entry", scripts[0], "PYSOURCE"), ("cli_entry", scripts[1], "PYSOURCE")]
            built["analysis"] = self

    class EXE(_Built):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built["exe"].append(self)

    class COLLECT(_Built):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built["collect"] = self

    class BUNDLE(_Built):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built["bundle"] = self

    hooks = types.ModuleType("PyInstaller.utils.hooks")
    hooks.collect_submodules = lambda name, **_kw: [f"{name}.some_module"]
    hooks.collect_data_files = lambda name, **_kw: [(f"/site/{name}/cacert.pem", name)]
    monkeypatch.setitem(sys.modules, "PyInstaller", types.ModuleType("PyInstaller"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils", types.ModuleType("PyInstaller.utils"))
    monkeypatch.setitem(sys.modules, "PyInstaller.utils.hooks", hooks)
    if pillow:
        pil = types.ModuleType("PIL")
        pil.Image = types.SimpleNamespace(open=lambda _path: _FakeImage(),
                                          Resampling=types.SimpleNamespace(NEAREST="nearest"))
        monkeypatch.setitem(sys.modules, "PIL", pil)
    else:
        monkeypatch.setitem(sys.modules, "PIL", None)  # "import PIL" then raises ImportError
    _FakeImage.saved = []
    monkeypatch.setattr(sys, "platform", platform)
    namespace = {"SPECPATH": str(PACKAGING), "workpath": str(tmp_path / "work"), "Analysis": Analysis,
                 "PYZ": _Built, "EXE": EXE, "COLLECT": COLLECT, "BUNDLE": BUNDLE}
    exec(compile(SPEC.read_text(encoding="utf-8"), str(SPEC), "exec"), namespace)
    built["namespace"] = namespace
    return built


def test_spec_is_valid_python() -> None:
    ast.parse(SPEC.read_text(encoding="utf-8"))


@pytest.mark.parametrize("platform", ["win32", "darwin", "linux"])
def test_spec_builds_one_analysis_and_two_programs_in_one_folder(monkeypatch, tmp_path, platform) -> None:
    built = run_spec(monkeypatch, tmp_path, platform)
    analysis = built["analysis"]
    assert [Path(s).name for s in analysis.scripts_in] == ["gui_entry.py", "cli_entry.py"]
    hidden = analysis.kwargs["hiddenimports"]
    for package in ("rich", "truststore", "huggingface_hub", "gettowork"):
        assert f"{package}.some_module" in hidden
    assert "tkinter" in hidden
    assert {"tests", "pytest"} <= set(analysis.kwargs["excludes"])
    # GNU readline is GPL-3.0; linking it into the Linux build would break the license story.
    assert "readline" in analysis.kwargs["excludes"]
    assert (str(ICON), "gettowork/assets") in analysis.kwargs["datas"]

    gui, cli = built["exe"]
    assert gui.kwargs["name"] == "GetToWork" and gui.kwargs["console"] is False
    assert cli.kwargs["name"] == "gettowork-cli" and cli.kwargs["console"] is True
    for exe in (gui, cli):
        assert exe.kwargs["exclude_binaries"] is True  # one-folder build
        assert exe.kwargs["upx"] is False
    # Each program runs only its own entry script (after PyInstaller's start-up scripts).
    assert [name for name, *_ in gui.args[1]] == ["pyiboot01_bootstrap", "gui_entry"]
    assert [name for name, *_ in cli.args[1]] == ["pyiboot01_bootstrap", "cli_entry"]

    collect = built["collect"]
    assert collect.kwargs["name"] == "GetToWork"
    assert collect.args[0] is gui and collect.args[1] is cli  # the GUI first: the Mac app's main program
    assert ("bundle" in built) == (platform == "darwin")


def test_spec_makes_a_mac_app_with_the_right_details(monkeypatch, tmp_path) -> None:
    bundle = run_spec(monkeypatch, tmp_path, "darwin")["bundle"]
    assert bundle.kwargs["name"] == "Get To Work.app"
    assert bundle.kwargs["bundle_identifier"] == "com.markelphoenix.gettowork"
    assert bundle.kwargs["version"] == gettowork.__version__
    plist = bundle.kwargs["info_plist"]
    assert plist["NSHighResolutionCapable"] is True
    # PyInstaller writes LSBackgroundOnly=True when the COLLECT's last program is a console one
    # (gettowork-cli is): a background-only app gets no Dock icon, no menu and no keyboard focus.
    assert plist["LSBackgroundOnly"] is False
    assert plist["CFBundleShortVersionString"] == gettowork.__version__
    # The bundled llama.cpp engine is built for macOS 13.3 and later (see the spec's comment).
    assert plist["LSMinimumSystemVersion"] == "13.3"
    assert bundle.kwargs["icon"].endswith(".icns")


@pytest.mark.parametrize("platform, suffix", [("win32", ".ico"), ("darwin", ".icns")])
def test_spec_converts_the_icon_for_windows_and_macos(monkeypatch, tmp_path, platform, suffix) -> None:
    built = run_spec(monkeypatch, tmp_path, platform)
    gui, cli = built["exe"]
    assert gui.kwargs["icon"] == cli.kwargs["icon"]
    assert gui.kwargs["icon"].endswith(suffix) and Path(gui.kwargs["icon"]).is_file()
    ((_out, options),) = _FakeImage.saved
    if platform == "win32":
        assert (256, 256) in options["sizes"] and (16, 16) in options["sizes"]
    else:
        assert options["format"] == "ICNS"


def test_spec_builds_without_pillow_and_on_linux_without_an_icon(monkeypatch, tmp_path) -> None:
    assert run_spec(monkeypatch, tmp_path, "win32", pillow=False)["exe"][0].kwargs["icon"] is None
    assert run_spec(monkeypatch, tmp_path, "linux")["exe"][0].kwargs["icon"] is None


def test_spec_reads_the_version_from_the_package(monkeypatch, tmp_path) -> None:
    assert run_spec(monkeypatch, tmp_path, "linux")["namespace"]["VERSION"] == gettowork.__version__


def test_spec_and_assemble_agree_on_names(monkeypatch, tmp_path) -> None:
    ns = run_spec(monkeypatch, tmp_path, "linux")["namespace"]
    assert ns["GUI_NAME"] == assemble.GUI_NAME == assemble.APP_FOLDER
    assert ns["CLI_NAME"] == assemble.CLI_NAME
    assert f"{ns['APP_NAME']}.app" == assemble.MAC_APP
    # Windows and macOS ignore upper/lower case: the two programs need names that differ otherwise too.
    assert ns["GUI_NAME"].lower() != ns["CLI_NAME"].lower()


# ---------------------------------------------------------------------------
# Entry scripts
# ---------------------------------------------------------------------------


def test_the_old_single_entry_script_is_gone() -> None:
    assert not (PACKAGING / "gettowork_entry.py").exists()


def test_gui_entry_imports_quietly_and_opens_the_window(monkeypatch) -> None:
    from gettowork import launcher

    module = load_script("gui_entry")  # importing must not start the game
    monkeypatch.setattr(launcher, "gui_main", lambda: 7)
    assert module.main() == 7


def test_gui_entry_gives_a_windowless_process_somewhere_to_print(monkeypatch) -> None:
    module = load_script("gui_entry")
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    module._give_windowless_process_somewhere_to_print()
    try:
        assert sys.stdout is not None and sys.stderr is not None
        print("this goes nowhere, and doesn't crash")
    finally:
        sys.stdout.close()
        sys.stderr.close()


def test_cli_entry_imports_quietly_and_runs_the_terminal_game(monkeypatch) -> None:
    from gettowork import cli

    module = load_script("cli_entry")
    monkeypatch.setattr(cli, "main", lambda: 3)
    assert module.main() == 3


def test_both_programs_give_what_they_start_the_systems_own_libraries(monkeypatch) -> None:
    # PyInstaller points LD_LIBRARY_PATH at the game's bundled libraries; the engine, the
    # browser and hardware checks must not inherit that (see launcher.restore_system_library_path).
    from gettowork import cli, launcher

    calls = []
    monkeypatch.setattr(launcher, "restore_system_library_path", lambda: calls.append("restore") or False)
    monkeypatch.setattr(launcher, "gui_main", lambda: calls.append("window") or 0)
    monkeypatch.setattr(cli, "main", lambda: calls.append("terminal") or 0)
    assert load_script("gui_entry").main() == 0
    assert load_script("cli_entry").main() == 0
    assert calls == ["restore", "window", "restore", "terminal"]


# ---------------------------------------------------------------------------
# The icon
# ---------------------------------------------------------------------------


def test_icon_is_committed_and_matches_the_drawing() -> None:
    width, height, pixels = make_icon.read_png_rgba(ICON.read_bytes())
    assert (width, height) == (256, 256)
    assert (width, height, pixels) == make_icon.render(), "run: python packaging/make_icon.py"


def test_icon_has_transparent_corners_and_a_solid_middle() -> None:
    width, _height, pixels = make_icon.render()

    def alpha(x: int, y: int) -> int:
        return pixels[(y * width + x) * 4 + 3]

    assert alpha(0, 0) == 0 and alpha(255, 255) == 0
    assert alpha(128, 128) == 255


def test_png_writer_round_trips_and_checks_its_input(tmp_path) -> None:
    rgba = bytes([255, 0, 0, 255, 0, 255, 0, 128, 0, 0, 255, 0, 9, 9, 9, 9])
    data = make_icon.png_bytes(2, 2, rgba)
    assert data.startswith(b"\x89PNG\r\n\x1a\n")
    assert make_icon.read_png_rgba(data) == (2, 2, rgba)
    with pytest.raises(ValueError):
        make_icon.png_bytes(2, 2, rgba[:-1])
    with pytest.raises(ValueError):
        make_icon.read_png_rgba(b"not a png")


def test_make_icon_writes_where_asked(tmp_path, capsys) -> None:
    out = tmp_path / "nested" / "icon.png"
    assert make_icon.main(["--out", str(out)]) == 0
    assert make_icon.read_png_rgba(out.read_bytes()) == make_icon.render()
    assert "256 x 256" in capsys.readouterr().out


def test_icon_ships_in_the_wheel_and_the_window_finds_it() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(r'gettowork\s*=\s*\[[^\]]*"assets/\*"', pyproject), "package-data must include assets/*"
    from gettowork.gui import app

    assert app._icon_path() == ICON


# ---------------------------------------------------------------------------
# collect_licenses.py
# ---------------------------------------------------------------------------


def test_collect_licenses_covers_the_runtime_dependencies() -> None:
    text, problems = collect_licenses.collect()
    assert not problems
    assert len(text) > 5000
    for name in ("rich", "psutil"):
        assert re.search(rf"^{name} \S+$", text, re.MULTILINE | re.IGNORECASE), name
    assert re.search(r"^huggingface[-_]hub \S+$", text, re.MULTILINE | re.IGNORECASE)
    assert "MIT" in text  # rich's license
    for heading in ("Python ", "Tcl/Tk", "PyInstaller"):
        assert heading in text
    assert "bootloader exception" in text


def test_collect_licenses_leaves_out_the_game_and_its_extras() -> None:
    names = {collect_licenses.canonical(d.metadata["Name"]) for d in collect_licenses.runtime_distributions()}
    assert {"rich", "psutil", "huggingface-hub"} <= names
    assert "gettowork" not in names
    assert "pytest" not in names and "llama-cpp-python" not in names  # extras aren't bundled


def test_collect_licenses_adds_llama_cpp_once_per_release(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine")
    text, problems = collect_licenses.collect([engine])
    assert not problems
    assert text.count("llama.cpp b9999 (the AI engine) - builds: cpu, vulkan") == 2  # summary + heading
    assert text.count("Copyright (c) 2023-2026 The ggml authors") == 1  # same text in both builds: shown once
    assert "https://github.com/ggml-org/llama.cpp/tree/b9999/vendor" in text


def test_collect_licenses_uses_license_files_found_in_the_build_when_not_listed(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine", variants=("cpu",))
    marker_path = engine / "b9999-cpu" / "install.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    del marker["license_files"]
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    (section,) = collect_licenses.llama_cpp_sections(engine)
    assert section.texts and "ggml authors" in section.texts[0][1]


def test_collect_licenses_fails_without_the_engine_license(tmp_path, capsys) -> None:
    engine = fake_engine(tmp_path / "engine", variants=("cpu",))
    shutil.rmtree(engine / "b9999-cpu" / "licenses")
    _text, problems = collect_licenses.collect([engine])
    assert any("no license file" in p for p in problems)
    assert collect_licenses.main(["--engine-dir", str(engine), "--out", str(tmp_path / "out.txt")]) == 1
    assert not (tmp_path / "out.txt").exists()
    assert "no license file" in capsys.readouterr().err


def test_collect_licenses_fails_when_the_only_engine_license_isnt_llama_cpps(tmp_path, capsys) -> None:
    """Like the b11100 Windows zips: LICENSE-LLVM-OpenMP alone must not pass for llama.cpp's MIT license."""
    engine = fake_engine(tmp_path / "engine", variants=("cpu",), exe="llama-server.exe",
                         license_text="The LLVM Project is under the Apache License v2.0 with LLVM Exceptions\n")
    _text, problems = collect_licenses.collect([engine])
    assert any("has no copy of llama.cpp's own MIT license" in p for p in problems)
    assert collect_licenses.main(["--engine-dir", str(engine), "--out", str(tmp_path / "out.txt")]) == 1
    assert "The ggml authors" in capsys.readouterr().err


def _add_components(build: Path, texts: dict[str, str], *, vc_runtime: tuple[str, ...] = ()) -> None:
    """Give a fake engine build more license files, listed in install.json the way fetch_engine does."""
    marker = json.loads((build / "install.json").read_text(encoding="utf-8"))
    marker["license_components"] = {"llama.cpp": "licenses/LICENSE"}
    for name, text in texts.items():
        rel = f"licenses/LICENSE-{name.replace(' ', '-')}"
        (build / rel).write_text(text, encoding="utf-8")
        marker["license_files"].append(rel)
        marker["license_components"][name] = rel
    if vc_runtime:
        marker["vc_runtime"] = list(vc_runtime)
    (build / "install.json").write_text(json.dumps(marker), encoding="utf-8")


def test_collect_licenses_ships_the_full_text_of_every_part_of_the_engine(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine", exe="llama-server.exe")
    parts = {"LLVM OpenMP": "LLVM Exceptions text", "cpp-httplib": "Copyright (c) 2017 yhirose",
             "jsonhpp": "Copyright (c) 2013-2025 Niels Lohmann", "BoringSSL": "Apache License Version 2.0 (BoringSSL)"}
    for variant in ("cpu", "vulkan"):
        _add_components(engine / f"b9999-{variant}", parts,
                        vc_runtime=("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"))
    text, problems = collect_licenses.collect([engine])
    assert not problems
    section = text[text.index("\nllama.cpp b9999 (the AI engine)"):]
    assert "License: MIT (llama.cpp), plus the licenses of LLVM OpenMP, cpp-httplib, jsonhpp, BoringSSL" in section
    labels = re.findall(r"^--- (.+) ---$", section, re.MULTILINE)[:5]
    assert labels[0] == "llama.cpp (LICENSE)"  # llama.cpp's own text first
    assert "BoringSSL (LICENSE-BoringSSL)" in labels and "cpp-httplib (LICENSE-cpp-httplib)" in labels
    for body in parts.values():
        assert section.count(body) == 1  # the full texts, once - not just a link
    assert "msvcp140.dll, vcruntime140.dll, vcruntime140_1.dll" in section


def test_collect_licenses_names_the_openssl_the_linux_engine_carries(tmp_path) -> None:
    """fetch_engine copies OpenSSL 3 into the Linux builds (Steam's runtime has none): its license text and a
    note saying what was copied, and why, end up in THIRD_PARTY_LICENSES.txt."""
    engine = fake_engine(tmp_path / "engine")
    for variant in ("cpu", "vulkan"):
        build = engine / f"b9999-{variant}"
        _add_components(build, {"OpenSSL": "Apache License Version 2.0 (OpenSSL 3's LICENSE.txt)"})
        marker = json.loads((build / "install.json").read_text(encoding="utf-8"))
        marker.update(openssl=["libssl.so.3", "libcrypto.so.3"], openssl_version="3.0.2")
        (build / "install.json").write_text(json.dumps(marker), encoding="utf-8")
    text, problems = collect_licenses.collect([engine])
    assert not problems
    section = text[text.index("\nllama.cpp b9999 (the AI engine)"):]
    assert "plus the licenses of OpenSSL" in section
    assert "OpenSSL 3.0.2 (libssl.so.3, libcrypto.so.3)" in section and "Steam's Linux runtime" in section
    assert section.count("OpenSSL 3's LICENSE.txt") == 1


def test_collect_licenses_fails_when_a_listed_part_of_the_engine_has_no_text(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine", variants=("metal",))
    _add_components(engine / "b9999-metal", {"BoringSSL": "Apache License Version 2.0"})
    (engine / "b9999-metal" / "licenses" / "LICENSE-BoringSSL").unlink()
    _text, problems = collect_licenses.collect([engine])
    assert problems == ["the engine build b9999-metal is missing the license text of BoringSSL "
                        "(licenses/LICENSE-BoringSSL)"]


def test_collect_licenses_fails_for_an_empty_engine_folder(tmp_path) -> None:
    (tmp_path / "engine").mkdir()
    _text, problems = collect_licenses.collect([tmp_path / "engine"])
    assert any("no engine builds" in p for p in problems)


def test_collect_licenses_main_writes_the_file(tmp_path, capsys) -> None:
    out = tmp_path / "build" / "THIRD_PARTY_LICENSES.txt"
    engine = fake_engine(tmp_path / "engine")
    assert collect_licenses.main(["--engine-dir", str(engine), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("Third-party software in Get To Work")
    assert "llama.cpp b9999" in text
    assert "Wrote" in capsys.readouterr().out


def _fake_app(root: Path, files: list[str], libs: str = "_internal") -> Path:
    """A PyInstaller output folder holding (empty) files with these names."""
    for name in files:
        path = root / libs / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\x7fELF")
    return root


LINUX_LIBS = ["libpython3.12.so.1.0", "libtcl8.6.so", "libtk8.6.so", "libssl.so.3", "libcrypto.so.3", "libffi.so.8",
              "libz.so.1", "libbz2.so.1.0", "liblzma.so.5", "libexpat.so.1", "libsqlite3.so.0", "libuuid.so.1",
              "libX11.so.6", "libXau.so.6", "libXdmcp.so.6", "libXext.so.6", "libXft.so.2", "libXrender.so.1",
              "libXss.so.1", "libfontconfig.so.1", "libfreetype.so.6", "libpng16.so.16", "libbrotlicommon.so.1",
              "libbrotlidec.so.1", "libbsd.so.0", "libmd.so.0", "libgcc_s.so.1", "libtinfo.so.6",
              # Python extension modules and package folders belong to Python distributions:
              "_tkinter.cpython-312-x86_64-linux-gnu.so", "hf_xet/hf_xet.abi3.so",
              "python3.12/lib-dynload/readline.cpython-312-x86_64-linux-gnu.so"]
WINDOWS_LIBS = ["python312.dll", "python3.dll", "VCRUNTIME140.dll", "VCRUNTIME140_1.dll", "MSVCP140.dll",
                "libcrypto-3.dll", "libssl-3.dll", "libffi-8.dll", "sqlite3.dll", "tcl86t.dll", "tk86t.dll",
                "zlib1.dll", "ucrtbase.dll", "api-ms-win-crt-runtime-l1-1-0.dll", "_tkinter.pyd", "select.pyd"]
MAC_LIBS = ["Python", "libssl.3.dylib", "libcrypto.3.dylib", "libtcl8.6.dylib", "libtk8.6.dylib",
            "liblzma.5.dylib", "libmpdec.4.dylib", "libffi.8.dylib", "libsqlite3.0.dylib"]


def test_collect_licenses_covers_every_native_library_a_linux_build_bundles(tmp_path) -> None:
    app = _fake_app(tmp_path / "GetToWork", LINUX_LIBS)
    looked_up: list[str] = []

    def lookup(name):
        looked_up.append(name)
        return ("libx11-6", "Copyright 1985-2024 X Consortium\nPermission is hereby granted...") \
            if name == "libX11.so.6" else None

    text, problems = collect_licenses.collect(app_dir=app, lookup=lookup)
    assert problems == []
    for title in ("X Window System libraries", "Fontconfig", "FreeType", "libpng", "ncurses (libtinfo)",
                  "libuuid (util-linux)", "Brotli", "libbsd", "libmd", "OpenSSL", "libffi", "zlib", "Expat",
                  "SQLite", "GCC runtime libraries"):
        assert title in text, title
    assert "bundled as libX11.so.6, libXau.so.6" in text
    assert "Copyright 1985-2024 X Consortium" in text  # the package's own copyright file, when there is one
    assert "The FreeType Project (www.freetype.org)" in text  # the FTL credit, from the maintained notice
    assert "libpython3.12.so.1.0" not in looked_up and "hf_xet.abi3.so" not in looked_up  # covered elsewhere


@pytest.mark.parametrize("files, libs", [(WINDOWS_LIBS, "_internal"), (MAC_LIBS, "_internal"),
                                         (MAC_LIBS, "Get To Work.app/Contents/Frameworks")])
def test_collect_licenses_knows_the_windows_and_mac_libraries(tmp_path, files, libs) -> None:
    app = _fake_app(tmp_path / "GetToWork", files, libs)
    _text, problems = collect_licenses.collect(app_dir=app, lookup=lambda name: None)
    assert problems == []


def test_collect_licenses_refuses_gpl3_readline_and_unknown_libraries(tmp_path, capsys) -> None:
    app = _fake_app(tmp_path / "GetToWork", ["libreadline.so.8", "libmystery.so.2", "libgdbm.so.6"])
    _text, problems = collect_licenses.collect(app_dir=app, lookup=lambda name: None)
    assert any("libreadline.so.8" in p and "GPL-3.0" in p for p in problems)
    assert any("libgdbm.so.6" in p and "GPL-3.0" in p for p in problems)
    assert any("libmystery.so.2" in p and "no license entry" in p for p in problems)
    out = tmp_path / "out.txt"
    assert collect_licenses.main(["--app-dir", str(app), "--out", str(out)]) == 1 and not out.exists()
    assert "libreadline" in capsys.readouterr().err


def test_collect_licenses_needs_the_built_game_folder_it_was_given(tmp_path) -> None:
    _text, problems = collect_licenses.collect(app_dir=tmp_path / "missing")
    assert any("isn't a folder" in p for p in problems)


def _fake_distribution(site: Path, name: str, *, license_file: bool) -> Any:
    import importlib.metadata as md

    info = site / f"{name}-1.0.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\nLicense: MIT\n"
        f"Project-URL: Homepage, https://example.org/{name}\n", encoding="utf-8")
    record = [f"{name}-1.0.dist-info/METADATA,,"]
    if license_file:
        (info / "licenses").mkdir()
        (info / "licenses" / "LICENSE").write_text(f"The {name} license text\n", encoding="utf-8")
        record.append(f"{name}-1.0.dist-info/licenses/LICENSE,,")
    (info / "RECORD").write_text("\n".join(record) + "\n", encoding="utf-8")
    return md.PathDistribution(info)


def test_distribution_sections_include_license_files_or_say_where_to_find_them(tmp_path) -> None:
    with_file = collect_licenses.distribution_section(_fake_distribution(tmp_path, "withlicense", license_file=True))
    assert with_file.title == "withlicense 1.0" and with_file.license == "MIT"
    assert with_file.source == "https://example.org/withlicense"
    assert with_file.texts == [("licenses/LICENSE", "The withlicense license text")]

    without = collect_licenses.distribution_section(_fake_distribution(tmp_path, "nolicense", license_file=False))
    assert not without.texts
    assert "MIT" in without.note and "https://example.org/nolicense" in without.note


def test_requirement_markers() -> None:
    assert collect_licenses.split_requirement('rich>=13 ; python_version >= "3"') == ("rich", 'python_version >= "3"')
    assert collect_licenses.split_requirement("huggingface_hub>=1.1") == ("huggingface_hub", "")
    assert collect_licenses.marker_applies("")
    assert not collect_licenses.marker_applies('extra == "dev"')
    assert collect_licenses.marker_applies('python_version >= "3"')


def test_python_license_is_found_or_linked(tmp_path) -> None:
    (tmp_path / "LICENSE.txt").write_text("PSF LICENSE AGREEMENT\n", encoding="utf-8")
    assert collect_licenses.python_section(tmp_path).texts == [("LICENSE.txt", "PSF LICENSE AGREEMENT")]
    empty = tmp_path / "empty"
    empty.mkdir()
    fallback = collect_licenses.python_section(empty)
    assert not fallback.texts and "https://docs.python.org/3/license.html" in fallback.note


def test_tcl_tk_license_is_found_or_summarised(tmp_path) -> None:
    library = tmp_path / "lib" / "tcl8.6"
    (tmp_path / "lib" / "tk8.6" / "demos").mkdir(parents=True)
    library.mkdir(parents=True)
    (library / "license.terms").write_text("This software is copyrighted by the Regents\n", encoding="utf-8")
    (tmp_path / "lib" / "tk8.6" / "demos" / "license.terms").write_text("Tk terms\n", encoding="utf-8")
    section = collect_licenses.tcl_tk_section(library, tmp_path)
    assert [label for label, _text in section.texts] == ["tcl8.6/license.terms", "demos/license.terms"]

    nothing = tmp_path / "nothing"
    nothing.mkdir()
    fallback = collect_licenses.tcl_tk_section(nothing / "tcl8.6", nothing)
    assert not fallback.texts
    assert "https://www.tcl-lang.org/software/tcltk/license.html" in fallback.note


# ---------------------------------------------------------------------------
# assemble.py
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("os_key", ["windows", "linux"])
def test_assemble_windows_and_linux_layout(tmp_path, licenses_file, os_key) -> None:
    dist = fake_dist(tmp_path, os_key)
    engine = fake_engine(tmp_path / "engine", exe="llama-server.exe" if os_key == "windows" else "llama-server")
    layout, archive = assemble.assemble(dist, engine, os_key, "x64", tmp_path / "out", licenses=licenses_file,
                                        built_from="abc123", host="linux", say=lambda _text: None)
    folder = dist / "GetToWork"
    assert layout.app_dir == folder and layout.resources == folder and layout.readme_dir == folder
    ext = ".exe" if os_key == "windows" else ""
    assert layout.gui == folder / f"GetToWork{ext}" and layout.cli == folder / f"gettowork-cli{ext}"
    for build in ("b9999-cpu", "b9999-vulkan"):
        assert (folder / "engine" / build / "install.json").is_file()
        assert (folder / "engine" / build / "licenses" / "LICENSE").is_file()
    assert (folder / "THIRD_PARTY_LICENSES.txt").read_text(encoding="utf-8") == licenses_file.read_text(encoding="utf-8")
    assert "GetToWork" in (folder / "README.txt").read_text(encoding="utf-8")

    info = json.loads((folder / "distribution.json").read_text(encoding="utf-8"))
    assert info == {"schema": 1, "channel": "release", "engine_downloads": False, "engine_dir": "engine",
                    "llama_cpp_tag": "b9999", "app_version": gettowork.__version__, "built_from": "abc123"}

    assert archive.name == f"GetToWork-{gettowork.__version__}-{os_key}-x64{'.zip' if os_key == 'windows' else '.tar.gz'}"
    if os_key == "windows":
        names = zip_names(archive)
    else:
        with tarfile.open(archive) as tar:
            members = {m.name: m for m in tar.getmembers()}
        names = set(members)
        if POSIX:  # the programs stay executable
            assert members["GetToWork/GetToWork"].mode & 0o111
            assert members["GetToWork/engine/b9999-cpu/llama-server"].mode & 0o111
            assert members["GetToWork/GetToWork"].uname == ""  # no builder's user name in the archive
    for inside in (f"GetToWork/GetToWork{ext}", f"GetToWork/gettowork-cli{ext}", "GetToWork/distribution.json",
                   "GetToWork/README.txt", "GetToWork/THIRD_PARTY_LICENSES.txt",
                   "GetToWork/engine/b9999-vulkan/install.json", "GetToWork/_internal/base_library.zip"):
        assert inside in names, inside


def test_assemble_macos_layout(tmp_path, licenses_file) -> None:
    dist = fake_dist(tmp_path, "macos")
    engine = fake_engine(tmp_path / "engine", variants=("metal",))
    layout, archive = assemble.assemble(dist, engine, "macos", "arm64", tmp_path / "out", licenses=licenses_file,
                                        host="linux", say=lambda _text: None)
    stage = dist / "macos-package" / "GetToWork"
    app = stage / "Get To Work.app"
    assert not (dist / "Get To Work.app").exists()  # moved next to its README
    assert layout.app_dir == app and layout.readme_dir == stage
    assert layout.gui == app / "Contents" / "MacOS" / "GetToWork"
    assert layout.cli == app / "Contents" / "MacOS" / "gettowork-cli"
    resources = app / "Contents" / "Resources"
    assert (resources / "engine" / "b9999-metal" / "install.json").is_file()
    assert json.loads((resources / "distribution.json").read_text(encoding="utf-8"))["engine_downloads"] is False
    assert (resources / "THIRD_PARTY_LICENSES.txt").is_file()
    assert (stage / "README.txt").is_file() and not (resources / "README.txt").exists()

    assert archive.name == f"GetToWork-{gettowork.__version__}-macos-arm64.zip"
    names = zip_names(archive)
    for inside in ("GetToWork/README.txt", "GetToWork/Get To Work.app/Contents/MacOS/GetToWork",
                   "GetToWork/Get To Work.app/Contents/Resources/distribution.json",
                   "GetToWork/Get To Work.app/Contents/Resources/engine/b9999-metal/install.json"):
        assert inside in names, inside
    if POSIX:  # the app's symbolic links survive as links
        with zipfile.ZipFile(archive) as z:
            info = z.getinfo("GetToWork/Get To Work.app/Contents/Resources/libpython3.12.dylib")
            assert stat.S_ISLNK(info.external_attr >> 16)
            assert z.read(info) == b"../Frameworks/libpython3.12.dylib"
            assert (z.getinfo("GetToWork/Get To Work.app/Contents/MacOS/GetToWork").external_attr >> 16) & 0o111


def test_assemble_on_a_mac_re_signs_the_app_and_zips_with_ditto(tmp_path, licenses_file, monkeypatch) -> None:
    dist = fake_dist(tmp_path, "macos")
    engine = fake_engine(tmp_path / "engine", variants=("metal",))
    calls: list[list[str]] = []

    def runner(cmd: list[str], **kwargs: Any) -> None:
        assert kwargs.get("check") is True
        calls.append(cmd)
        if cmd[0] == "ditto":
            Path(cmd[-1]).write_bytes(b"PK zip made by ditto")

    monkeypatch.setattr(assemble.shutil, "which", lambda name: f"/usr/bin/{name}")
    layout, archive = assemble.assemble(dist, engine, "macos", "arm64", tmp_path / "out", licenses=licenses_file,
                                        runner=runner, host="darwin", say=lambda _text: None)
    assert calls == [
        ["codesign", "--force", "--deep", "--sign", "-", str(layout.app_dir)],
        ["codesign", "--verify", "--deep", "--strict", str(layout.app_dir)],
        ["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent", str(layout.archive_root), str(archive)],
    ]
    assert archive.read_bytes() == b"PK zip made by ditto"


def test_a_mac_engines_programs_are_signed_before_the_app(tmp_path, licenses_file, monkeypatch) -> None:
    dist = fake_dist(tmp_path, "macos")
    engine = fake_engine(tmp_path / "engine", variants=("metal",))
    (engine / "b9999-metal" / "llama-server").write_bytes(bytes.fromhex("cffaedfe") + b"arm64 program")
    (engine / "b9999-metal" / "libggml.dylib").write_bytes(bytes.fromhex("cafebabe") + b"fat library")
    calls: list[list[str]] = []
    monkeypatch.setattr(assemble.shutil, "which", lambda name: f"/usr/bin/{name}")
    layout, _archive = assemble.assemble(
        dist, engine, "macos", "arm64", tmp_path / "out", licenses=licenses_file, host="darwin",
        runner=lambda cmd, **kw: calls.append(cmd) or (Path(cmd[-1]).write_bytes(b"PK") if cmd[0] == "ditto" else None),
        say=lambda _text: None)
    build = layout.resources / "engine" / "b9999-metal"
    assert calls[:3] == [
        ["codesign", "--force", "--sign", "-", str(build / "libggml.dylib")],
        ["codesign", "--force", "--sign", "-", str(build / "llama-server")],
        ["codesign", "--force", "--deep", "--sign", "-", str(layout.app_dir)],
    ]  # (the license and install.json aren't programs: left alone)


@pytest.mark.parametrize("os_key", ["windows", "linux", "macos"])
def test_the_game_finds_what_assemble_wrote(tmp_path, licenses_file, monkeypatch, os_key) -> None:
    """distribution.py reads distribution.json from where assemble.py put it, next to the real program."""
    dist = fake_dist(tmp_path, os_key)
    engine = fake_engine(tmp_path / "engine", variants=("metal",) if os_key == "macos" else ("cpu",))
    layout, _archive = assemble.assemble(dist, engine, os_key, assemble.DEFAULT_ARCH[os_key], tmp_path / "out",
                                         licenses=licenses_file, host="linux", say=lambda _text: None)
    monkeypatch.delenv("GETTOWORK_DISTRIBUTION", raising=False)
    monkeypatch.delenv("GETTOWORK_ENGINE_DIR", raising=False)
    monkeypatch.delenv("GETTOWORK_ALLOW_ENGINE_DOWNLOAD", raising=False)
    monkeypatch.setattr(sys, "executable", str(layout.gui))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    found = distribution.find_distribution_file()
    assert found == layout.resources / "distribution.json"
    try:
        dist_info = distribution.load(refresh=True)
        assert dist_info.channel == "release" and dist_info.engine_downloads is False
        assert dist_info.engine_dirs == (layout.resources / "engine",) and dist_info.bundled
        assert dist_info.llama_cpp_tag == "b9999"
    finally:
        monkeypatch.undo()
        distribution.load(refresh=True)  # forget the fake build for the other tests


def test_assemble_twice_replaces_the_previous_result(tmp_path, licenses_file) -> None:
    for os_key in ("linux", "macos"):
        base = tmp_path / os_key
        dist = fake_dist(base, os_key)
        engine = fake_engine(base / "engine", variants=("cpu",))
        quiet = {"licenses": licenses_file, "host": "linux", "say": lambda _text: None}
        layout, first = assemble.assemble(dist, engine, os_key, "x64", base / "out", **quiet)
        (layout.resources / "engine" / "stale-build").mkdir()
        _layout, second = assemble.assemble(dist, engine, os_key, "x64", base / "out", **quiet)
        assert first == second and second.is_file()
        assert not (layout.resources / "engine" / "stale-build").exists()


def test_assemble_refuses_incomplete_inputs(tmp_path, licenses_file) -> None:
    quiet = {"licenses": licenses_file, "host": "linux", "say": lambda _text: None}
    engine = fake_engine(tmp_path / "engine")
    with pytest.raises(assemble.AssembleError, match="pyinstaller"):
        assemble.assemble(tmp_path / "no-dist", engine, "linux", "x64", tmp_path / "out", **quiet)

    dist = fake_dist(tmp_path, "linux")
    with pytest.raises(assemble.AssembleError, match="fetch_engine"):
        assemble.assemble(dist, tmp_path / "no-engine", "linux", "x64", tmp_path / "out", **quiet)
    (tmp_path / "empty-engine").mkdir()
    with pytest.raises(assemble.AssembleError, match="no engine builds"):
        assemble.assemble(dist, tmp_path / "empty-engine", "linux", "x64", tmp_path / "out", **quiet)
    with pytest.raises(assemble.AssembleError, match="collect_licenses"):
        assemble.assemble(dist, engine, "linux", "x64", tmp_path / "out", licenses=tmp_path / "missing.txt",
                          host="linux", say=lambda _text: None)

    broken = fake_engine(tmp_path / "broken", variants=("cpu",))
    (broken / "b9999-cpu" / "llama-server").unlink()
    with pytest.raises(assemble.AssembleError, match="isn't there"):
        assemble.assemble(dist, broken, "linux", "x64", tmp_path / "out", **quiet)

    mixed = fake_engine(tmp_path / "mixed", tag="b1", variants=("cpu",))
    fake_engine(mixed, tag="b2", variants=("vulkan",))
    with pytest.raises(assemble.AssembleError, match="one llama.cpp release"):
        assemble.assemble(dist, mixed, "linux", "x64", tmp_path / "out", **quiet)


@pytest.mark.parametrize("os_key", ["windows", "macos", "linux"])
def test_readme_is_three_friendly_lines(os_key) -> None:
    text = assemble.readme_text(os_key, "1.2.3")
    lines = text.splitlines()
    assert len(lines) == 3 and all(lines)
    assert "Get To Work 1.2.3" in lines[0] and "double-click" in lines[0]
    expected = {"windows": ["Extract All", "extract the whole zip first", "More info", "Run anyway", "SmartScreen",
                            "Smart App Control", "0x11C7", "from Steam too"],
                "macos": ["right-click", "Open", "xattr -dr com.apple.quarantine", "On Steam, just press Play"],
                "linux": ["tar", "no chmod +x is needed"]}[os_key]
    for words in expected:
        assert words in text
    if os_key == "macos":
        # The same README lands at the top of the Steam install folder (depot_build_macos.vdf): dragging the app
        # to Applications there would move it out of Steam's library - so that advice is only for GitHub copies.
        drag = text.index("drag it into Applications")
        assert "downloaded from GitHub" in text[max(0, drag - 40):drag]


def test_zip_folder_keeps_permissions_and_links(tmp_path) -> None:
    folder = tmp_path / "Thing"
    (folder / "sub").mkdir(parents=True)
    program = folder / "run"
    program.write_text("#!/bin/sh\n", encoding="utf-8")
    program.chmod(0o755)
    (folder / "sub" / "data.txt").write_text("hello", encoding="utf-8")
    if POSIX:
        os.symlink("sub/data.txt", folder / "link")
    out = tmp_path / "thing.zip"
    assemble.zip_folder(folder, out)
    with zipfile.ZipFile(out) as z:
        assert {"Thing/", "Thing/sub/", "Thing/run", "Thing/sub/data.txt"} <= set(z.namelist())
        assert z.read("Thing/sub/data.txt") == b"hello"
        if POSIX:
            assert (z.getinfo("Thing/run").external_attr >> 16) & 0o111
            link = z.getinfo("Thing/link")
            assert stat.S_ISLNK(link.external_attr >> 16) and z.read(link) == b"sub/data.txt"
    if POSIX and shutil.which("unzip"):  # the standard tool restores both
        target = tmp_path / "unzipped"
        subprocess.run(["unzip", "-q", str(out), "-d", str(target)], check=True)
        assert os.access(target / "Thing" / "run", os.X_OK)
        assert (target / "Thing" / "link").is_symlink()


def test_main_prints_and_records_the_outputs(tmp_path, licenses_file, monkeypatch, capsys) -> None:
    dist = fake_dist(tmp_path, "linux")
    engine = fake_engine(tmp_path / "engine")
    outputs = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(outputs))
    monkeypatch.setenv("GITHUB_SHA", "deadbeef")
    code = assemble.main(["--dist", str(dist), "--engine-dir", str(engine), "--os", "linux", "--out",
                          str(tmp_path / "out"), "--licenses", str(licenses_file)])
    assert code == 0
    printed = capsys.readouterr().out
    recorded = dict(line.split("=", 1) for line in outputs.read_text(encoding="utf-8").splitlines())
    assert set(recorded) == {"archive", "app_dir", "gui", "cli", "engine"}
    assert recorded["cli"] == (dist / "GetToWork" / "gettowork-cli").as_posix()
    assert recorded["engine"] == (dist / "GetToWork" / "engine").as_posix()
    assert f"ARCHIVE={recorded['archive']}" in printed and Path(recorded["archive"]).is_file()
    assert json.loads((dist / "GetToWork" / "distribution.json").read_text(encoding="utf-8"))["built_from"] == "deadbeef"


def test_main_reports_problems_without_a_traceback(tmp_path, licenses_file, capsys) -> None:
    code = assemble.main(["--dist", str(tmp_path / "nothing"), "--engine-dir", str(tmp_path), "--os", "windows",
                          "--out", str(tmp_path / "out"), "--licenses", str(licenses_file)])
    assert code == 1
    assert capsys.readouterr().err.startswith("error: ")


# ---------------------------------------------------------------------------
# smoke_test.sh
# ---------------------------------------------------------------------------


def test_smoke_test_script_is_valid_bash() -> None:
    bash = shutil.which("bash")
    if not bash or not POSIX:
        pytest.skip("needs bash")
    subprocess.run([bash, "-n", str(PACKAGING / "smoke_test.sh")], check=True)


def _smoke_env(tmp_path: Path, *, distribution_file: bool) -> tuple[Path, dict]:
    """A fake built game (engine + distribution.json, like assemble.py writes) and the smoke test's environment."""
    game = tmp_path / "GetToWork"
    engine = fake_engine(game / "engine")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "GETTOWORK_HOME": str(tmp_path / "home")}
    for var in ("GETTOWORK_ENGINE_DIR", "GETTOWORK_ALLOW_ENGINE_DOWNLOAD", "GETTOWORK_DISTRIBUTION"):
        env.pop(var, None)
    if distribution_file:
        (game / "distribution.json").write_text(json.dumps(
            {"schema": 1, "channel": "release", "engine_downloads": False, "engine_dir": "engine",
             "llama_cpp_tag": "b9999"}), encoding="utf-8")
        env["GETTOWORK_DISTRIBUTION"] = str(game / "distribution.json")  # (a real build finds it next to itself)
    return engine, env


@pytest.mark.skipif(not POSIX or not shutil.which("bash"), reason="needs bash and runnable shell scripts")
def test_smoke_test_passes_for_the_game_and_its_bundled_engine(tmp_path) -> None:
    engine, env = _smoke_env(tmp_path, distribution_file=True)
    result = subprocess.run(
        ["bash", str(PACKAGING / "smoke_test.sh"), "--engine-dir", str(engine), sys.executable, "-m", "gettowork"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "version: 9999 (fake)" in result.stdout
    assert "built into the game: llama.cpp b9999 (CPU, Vulkan); engine downloads off" in result.stdout
    assert "Smoke test passed" in result.stdout


@pytest.mark.skipif(not POSIX or not shutil.which("bash"), reason="needs bash and runnable shell scripts")
def test_smoke_test_fails_when_the_game_doesnt_use_its_built_in_engine(tmp_path) -> None:
    """A build that lost its distribution.json would download llama.cpp while playing: the build check fails."""
    engine, env = _smoke_env(tmp_path, distribution_file=False)
    result = subprocess.run(
        ["bash", str(PACKAGING / "smoke_test.sh"), "--engine-dir", str(engine), sys.executable, "-m", "gettowork"],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert result.returncode == 1
    assert "doesn't say it uses its built-in llama.cpp b9999" in result.stdout


@pytest.mark.skipif(not POSIX or not shutil.which("bash"), reason="needs bash")
def test_smoke_test_fails_when_a_bundled_engine_is_missing(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine", variants=("cpu",))
    (engine / "b9999-cpu" / "llama-server").unlink()
    result = subprocess.run(["bash", str(PACKAGING / "smoke_test.sh"), "--engine-dir", str(engine), "true"],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 1 and "isn't there" in result.stdout


# ---------------------------------------------------------------------------
# engine_isolation_check.sh (the Linux engine starts without the system's OpenSSL 3)
# ---------------------------------------------------------------------------


def test_engine_isolation_check_script_is_valid_bash() -> None:
    bash = shutil.which("bash")
    if not bash or not POSIX:
        pytest.skip("needs bash")
    subprocess.run([bash, "-n", str(PACKAGING / "engine_isolation_check.sh")], check=True)
    assert os.access(PACKAGING / "engine_isolation_check.sh", os.X_OK)


def _isolation_check(engine: Path, lib_dirs: list[Path]) -> subprocess.CompletedProcess:
    env = {**os.environ, "SYSTEM_LIB_DIRS": " ".join(str(d) for d in lib_dirs)}
    return subprocess.run(["bash", str(PACKAGING / "engine_isolation_check.sh"), "--inside", str(engine)],
                          capture_output=True, text=True, env=env, timeout=60)


@pytest.mark.skipif(not POSIX or not shutil.which("bash"), reason="needs bash and runnable shell scripts")
def test_engine_isolation_check_runs_every_build_on_a_system_without_openssl_3(tmp_path) -> None:
    engine = fake_engine(tmp_path / "engine")
    system = tmp_path / "usr-lib"
    system.mkdir()
    result = _isolation_check(engine, [system])
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("version: 9999 (fake)") == 2
    assert "Every bundled engine build starts without the system's OpenSSL 3." in result.stdout
    # A build that doesn't start fails the check...
    (engine / "b9999-cpu" / "llama-server").write_text("#!/bin/sh\necho 'libssl.so.3: cannot open'\nexit 127\n")
    result = _isolation_check(engine, [system])
    assert result.returncode == 1 and "doesn't start without the system's OpenSSL 3" in result.stdout
    # ...and on a system that has OpenSSL 3 the check refuses to prove nothing.
    (system / "libssl.so.3").write_bytes(b"")
    result = _isolation_check(engine, [tmp_path / "elsewhere", system])
    assert result.returncode == 1 and "this system has OpenSSL 3" in result.stdout
    empty = tmp_path / "no-engine"
    empty.mkdir()
    assert _isolation_check(empty, [tmp_path / "elsewhere"]).returncode == 1


def test_the_linux_build_checks_its_engine_without_openssl_3() -> None:
    data = load_workflow("build.yml")
    step = step_named(data, "without the system's OpenSSL 3")
    assert step["if"] == "runner.os == 'Linux'"
    assert "packaging/engine_isolation_check.sh" in step["run"] and step["env"]["ENGINE"]
    names = [s.get("name", "") for s in data["jobs"]["standalone"]["steps"]]
    assert names.index(step["name"]) > next(i for i, n in enumerate(names) if n.startswith("Add the engine"))


# ---------------------------------------------------------------------------
# Workflows
# ---------------------------------------------------------------------------


def load_workflow(name: str) -> dict:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    data["on"] = data.get("on", data.get(True))  # PyYAML reads the bare key `on` as True
    return data


def all_steps(data: dict) -> list[dict]:
    return [step for job in data["jobs"].values() for step in job["steps"]]


def step_named(data: dict, words: str) -> dict:
    matches = [s for s in all_steps(data) if words.lower() in s.get("name", "").lower()]
    assert matches, f"no step named like {words!r}"
    return matches[0]


@pytest.mark.parametrize("name", ["build.yml", "ci.yml", "live-check.yml"])
def test_workflow_loads_and_uses_the_agreed_actions(name) -> None:
    data = load_workflow(name)
    assert data["permissions"] == {"contents": "read"}
    uses = {step["uses"] for step in all_steps(data) if "uses" in step}
    assert uses <= {"actions/checkout@v4", "actions/setup-python@v5", "actions/upload-artifact@v7"}
    assert "actions/checkout@v4" in uses and "actions/setup-python@v5" in uses


@pytest.mark.parametrize("name", ["build.yml", "ci.yml", "live-check.yml"])
def test_workflow_refers_only_to_files_that_exist(name) -> None:
    data = load_workflow(name)
    scripts = "\n".join(str(step.get("run", "")) for step in all_steps(data))
    referenced = set(re.findall(r"(packaging/[\w./-]+\.(?:py|sh|spec))", scripts))
    for path in referenced:
        assert (ROOT / path).is_file(), f"{name} runs {path}, which doesn't exist"
    if name == "build.yml":
        assert {"packaging/fetch_engine.py", "packaging/collect_licenses.py", "packaging/gettowork.spec",
                "packaging/assemble.py", "packaging/smoke_test.sh"} <= referenced


def test_build_workflow_triggers_and_budget() -> None:
    data = load_workflow("build.yml")
    on = data["on"]
    assert on["push"]["branches"] == ["main"] and "workflow_dispatch" in on
    # Docs- or tests-only pushes make no new (big) game artifacts...
    ignored = on["push"]["paths-ignore"]
    assert {"**/*.md", "docs/**", "tests/**"} <= set(ignored)
    # ...but nothing the builds are made from is ignored.
    for path in ("src/gettowork/cli.py", "packaging/gettowork.spec", "packaging/llama_cpp_tag.txt",
                 ".github/workflows/build.yml", "pyproject.toml"):
        assert not any(Path(path).match(pattern.replace("**/", "")) or path.startswith(pattern.rstrip("*"))
                       for pattern in ignored), path
    paths = on["pull_request"]["paths"]
    assert set(paths) == {"packaging/**", "src/gettowork/gui/**", "src/gettowork/launcher.py",
                          "src/gettowork/distribution.py", ".github/workflows/build.yml"}
    for pattern in paths:
        assert (ROOT / pattern.replace("/**", "")).exists(), pattern
    assert data["concurrency"]["cancel-in-progress"] is True


def test_build_workflow_matrix_and_steps() -> None:
    data = load_workflow("build.yml")
    job = data["jobs"]["standalone"]
    targets = {entry["target"]: entry for entry in job["strategy"]["matrix"]["include"]}
    assert set(targets) == {"windows-x64", "linux-x64", "macos-arm64"}
    assert (targets["windows-x64"]["os"], targets["windows-x64"]["engines"]) == ("windows-latest", "vulkan,cpu")
    assert (targets["linux-x64"]["os"], targets["linux-x64"]["engines"]) == ("ubuntu-22.04", "vulkan,cpu")
    assert (targets["macos-arm64"]["os"], targets["macos-arm64"]["engines"]) == ("macos-latest", "metal")
    for entry in targets.values():
        assert entry["platform"] in assemble.README_TEXT and entry["arch"] in ("x64", "arm64")

    names = [step.get("name", "") for step in job["steps"]]
    order = ["Fetch the llama.cpp engine", "Build the game with PyInstaller", "Gather the third-party license",
             "Add the engine", "Smoke test", "Self-test the game window", "Upload the game"]
    positions = [next(i for i, n in enumerate(names) if n.startswith(words)) for words in order]
    assert positions == sorted(positions)

    fetch = step_named(data, "Fetch the llama.cpp engine")
    assert fetch["env"]["GITHUB_TOKEN"] == "${{ secrets.GITHUB_TOKEN }}"
    assert "--verify" in fetch["run"]
    # Every OS fetches the pinned release (packaging/llama_cpp_tag.txt), never "whatever is newest".
    assert "--tag pinned" in fetch["run"]
    assert (PACKAGING / "llama_cpp_tag.txt").is_file()
    licenses = step_named(data, "Gather the third-party license")
    assert "--app-dir dist/GetToWork" in licenses["run"]  # the native libraries PyInstaller bundled, too
    install = step_named(data, "Install the game, PyInstaller")
    assert "pip install . pyinstaller pillow" in install["run"]
    selftest = step_named(data, "Self-test the game window")
    assert "xvfb-run -a" in selftest["run"] and "--gui-selftest" in selftest["run"]
    # A home folder that outlives the self-test, so its crash report can be shown when the window fails.
    assert selftest["env"]["GETTOWORK_HOME"] == "${{ runner.temp }}/selftest-home"
    show = step_named(data, "Show the window self-test transcript")
    assert show["if"] == "always()" and "SELFTEST_HOME" in show["env"] and "crash" in show["run"]
    wheel_selftest = step_named(data, "Self-test the installed game's window")
    assert wheel_selftest["env"]["GETTOWORK_HOME"]
    assert "logs" in step_named(data, "crash reports (if any)")["run"]
    assert "--engine-dir" in step_named(data, "Smoke test the terminal version")["run"]
    # The Hugging Face check handles an outage itself (a warning): a crash of the built game must fail the build.
    search = step_named(data, "search Hugging Face")
    assert "continue-on-error" not in search and "exit 1" in search["run"] and "::warning::" in search["run"]
    upload = step_named(data, "Upload the game")
    assert upload["uses"] == "actions/upload-artifact@v7"
    # Uploaded as the archive itself (one layer to unpack), and kept briefly: storage is small.
    assert upload["with"]["archive"] is False and upload["with"]["retention-days"] == 3
    assert upload["with"]["if-no-files-found"] == "error"
    assert upload["with"]["path"] == "${{ steps.assemble.outputs.archive }}"
    package_upload = step_named(data, "Upload the package")["with"]
    assert package_upload["retention-days"] == 7 and package_upload["if-no-files-found"] == "error"


def _matrix_for(matrix: dict, event: str) -> set[tuple[str, str]]:
    """Which (os, python) jobs GitHub runs for `event`, evaluating the exclude expressions."""
    expression = re.compile(r"\$\{\{\s*github\.event_name\s*(==|!=)\s*'(\w+)'\s*&&\s*'([^']*)'\s*\|\|\s*'([^']*)'\s*\}\}")

    def resolve(value: Any) -> str:
        text = str(value)
        match = expression.fullmatch(text)
        if match:
            op, name, if_true, if_false = match.groups()
            return if_true if (event == name) == (op == "==") else if_false
        assert "${{" not in text, f"an expression this test can't evaluate: {text}"
        return text

    jobs = {(os_name, py) for os_name in matrix["os"] for py in matrix["python-version"]}
    for rule in matrix.get("exclude", []):
        wanted = {key: resolve(value) for key, value in rule.items()}
        jobs = {job for job in jobs
                if not all({"os": job[0], "python-version": job[1]}[key] == value for key, value in wanted.items())}
    return jobs


def test_ci_runs_a_small_matrix_on_pull_requests_and_the_full_one_on_main() -> None:
    data = load_workflow("ci.yml")
    assert data["on"]["push"] == {"branches": ["main"]}
    assert "pull_request" in data["on"] and "workflow_dispatch" in data["on"]
    (job,) = data["jobs"].values()
    matrix = job["strategy"]["matrix"]
    pull_request = {("ubuntu-latest", "3.10"), ("ubuntu-latest", "3.12"), ("windows-latest", "3.12")}
    assert _matrix_for(matrix, "pull_request") == pull_request
    main = pull_request | {("windows-latest", "3.10")}
    assert _matrix_for(matrix, "push") == main  # macOS is covered on main by build.yml's Mac app checks
    assert _matrix_for(matrix, "workflow_dispatch") == main | {("macos-latest", "3.12")}
    assert job["env"]["HF_HUB_OFFLINE"] == "1"


def test_ci_runs_the_window_tests_under_a_virtual_screen_on_linux() -> None:
    data = load_workflow("ci.yml")
    linux = step_named(data, "Run the tests (with a virtual screen)")
    assert linux["if"] == "runner.os == 'Linux'"
    assert linux["run"].startswith("xvfb-run -a python -m pytest")


def test_live_check_workflow() -> None:
    data = load_workflow("live-check.yml")
    on = data["on"]
    assert "workflow_dispatch" in on and on["schedule"][0]["cron"]
    job = data["jobs"]["live"]
    assert job["runs-on"].startswith("ubuntu")
    assert re.search(r"SmolLM2-135M-Instruct-GGUF", job["env"]["MODEL"])
    assert "--list-models" in step_named(data, "Real Hugging Face model search")["run"]
    fetch = step_named(data, "Fetch the official llama.cpp CPU engine")
    assert "--variants cpu" in fetch["run"] and "--verify" in fetch["run"]
    # The release the game builds ship, and the newest one (is moving the pin safe?).
    assert "--tag pinned" in fetch["run"] and "--tag auto" in fetch["run"]
    newest = step_named(data, "Play the same game with the newest llama.cpp release")
    assert newest["env"]["GETTOWORK_ENGINE_DIR"].endswith("engine-newest")
    helper = step_named(data, "Write the scripted-game helper")["run"]
    for check in ("traceback", "130", "pgrep", "Your model is awake and ready"):
        assert check in helper
    play = step_named(data, "Play a scripted game with a tiny real model")
    assert play["env"]["GETTOWORK_ENGINE_DIR"] and play["env"]["GETTOWORK_ALLOW_ENGINE_DOWNLOAD"] == "0"
    assert "--no-jev" in play["run"]
    jev = step_named(data, "real Jev referee")
    assert "workflow_dispatch" in jev["if"] and "HAS_JEV_KEY" in jev["if"]
    assert job["env"]["HAS_JEV_KEY"] == "${{ secrets.TYPESAFE_API_KEY != '' }}"


def test_live_check_model_is_one_the_game_can_parse() -> None:
    model = load_workflow("live-check.yml")["jobs"]["live"]["env"]["MODEL"]
    default = re.search(r"'([^']+)'", model).group(1)
    assert re.fullmatch(r"[\w.-]+/[\w.-]+-GGUF", default)


# ---------------------------------------------------------------------------
# Steam
# ---------------------------------------------------------------------------


def parse_vdf(text: str) -> dict:
    """A tiny reader for Valve's KeyValues (.vdf) files: quoted strings, braces and // comments."""
    text = "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))
    tokens = re.findall(r'"((?:[^"\\]|\\.)*)"|([{}])', text)
    pos = 0

    def block() -> dict:
        nonlocal pos
        out: dict = {}
        while pos < len(tokens):
            key, brace = tokens[pos]
            pos += 1
            if brace == "}":
                return out
            value, value_brace = tokens[pos]
            pos += 1
            out[key] = block() if value_brace == "{" else value
        return out

    return block()


STEAM_OSES = {"windows": "<DEPOT_ID_WIN>", "macos": "<DEPOT_ID_MAC>", "linux": "<DEPOT_ID_LINUX>"}


def test_steam_folder_has_every_template() -> None:
    for name in ["README.md", "STORE_PAGE.md", "app_build_all.vdf"] + \
            [f"{kind}_build_{os_key}.vdf" for kind in ("app", "depot") for os_key in STEAM_OSES]:
        assert (STEAM / name).is_file(), name


@pytest.mark.parametrize("os_key", list(STEAM_OSES))
def test_steam_app_and_depot_templates_fit_together(os_key) -> None:
    app = parse_vdf((STEAM / f"app_build_{os_key}.vdf").read_text(encoding="utf-8"))["AppBuild"]
    depot_id = STEAM_OSES[os_key]
    assert app["AppID"] == "<APP_ID>" and app["ContentRoot"] == "../content/"
    assert app["Depots"] == {depot_id: f"depot_build_{os_key}.vdf"}
    depot = parse_vdf((STEAM / f"depot_build_{os_key}.vdf").read_text(encoding="utf-8"))["DepotBuild"]
    assert depot["DepotID"] == depot_id
    assert depot["FileMapping"]["DepotPath"] == "." and depot["FileMapping"]["Recursive"] == "1"
    assert depot["FileMapping"]["LocalPath"].startswith(f"{os_key}/")


def test_steam_all_in_one_build_lists_every_depot() -> None:
    app = parse_vdf((STEAM / "app_build_all.vdf").read_text(encoding="utf-8"))["AppBuild"]
    assert app["Depots"] == {depot: f"depot_build_{os_key}.vdf" for os_key, depot in STEAM_OSES.items()}


def test_steam_launch_options_match_the_archive_layout() -> None:
    """The unpacked archives + depot mappings put the programs exactly where the launch options point."""
    readme = (STEAM / "README.md").read_text(encoding="utf-8")
    windows = f"{assemble.APP_FOLDER}\\{assemble.GUI_NAME}.exe"
    linux = f"{assemble.APP_FOLDER}/{assemble.GUI_NAME}"
    for launch in (windows, assemble.MAC_APP, linux):
        assert f"`{launch}`" in readme, launch
    # macOS: only the inside of the archive's GetToWork folder is uploaded, so the app sits at the top.
    mac_depot = parse_vdf((STEAM / "depot_build_macos.vdf").read_text(encoding="utf-8"))["DepotBuild"]
    assert mac_depot["FileMapping"]["LocalPath"] == f"macos/{assemble.APP_FOLDER}/*"


def test_steam_readme_covers_steam_deck_and_uploading() -> None:
    readme = (STEAM / "README.md").read_text(encoding="utf-8")
    for words in ("Proton", "STEAM + X", "Linux + SteamOS", "steamcmd", "run_app_build", "macOS or Linux",
                  "Visual C++ Redist", "ditto -x -k", "tar xzf", "Gridfall", "STEAM_APP_ID", "Report a problem",
                  "--models-dir", "llama_cpp_tag.txt", "kept for 3 days"):
        assert words in _flat(readme), words
    assert "7 days" not in readme and "gettowork-windows-x64" not in readme  # the old artifact names and retention


def test_steam_readme_linux_runtime_advice_matches_the_engine() -> None:
    """The Linux engine carries its own OpenSSL 3 (Steam's 1.0/3.0 runtimes have only 1.1); the README says
    so, points at the newest runtime, and never asks players to install packages on a read-only SteamOS."""
    readme = _flat((STEAM / "README.md").read_text(encoding="utf-8"))
    for words in ("Steam Linux Runtime 4.0", "OpenSSL 3", "fetch_engine.py", "verify the game's files"):
        assert words in readme, words
    assert "apt install" not in readme and "Leave the Linux runtime at Steam's default" not in readme


def test_steam_upload_commands_use_the_sdks_own_steamcmd() -> None:
    """The Steamworks SDK doesn't put steamcmd on PATH: its copy is ContentBuilder/builder_osx/steamcmd.sh
    (builder_linux/ on Linux). A bare "steamcmd ..." line fails with "command not found"."""
    for path in [STEAM / "README.md", *sorted(STEAM.glob("app_build_*.vdf"))]:
        text = path.read_text(encoding="utf-8")
        commands = [line for line in text.splitlines() if "+run_app_build" in line]
        assert commands, path.name
        for line in commands:
            assert re.search(r"\./builder_(osx|linux)/steamcmd\.sh \+login", line), (path.name, line)
    readme = _flat((STEAM / "README.md").read_text(encoding="utf-8"))
    assert "builder_osx/steamcmd.sh" in readme and "builder_linux/steamcmd.sh" in readme


def _flat(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def test_store_page_carries_the_exact_ai_disclosure() -> None:
    page = (STEAM / "STORE_PAGE.md").read_text(encoding="utf-8")
    assert _flat(notices.STEAM_AI_DISCLOSURE) in _flat(page), (
        "packaging/steam/STORE_PAGE.md must quote notices.STEAM_AI_DISCLOSURE exactly - paste the new text in")


def test_store_page_has_the_jev_and_privacy_notices() -> None:
    page = (STEAM / "STORE_PAGE.md").read_text(encoding="utf-8")
    for words in ("TypeSafe AI", "paid third-party service", "isn't affiliated", "api.typesafe.ai",
                  "Privacy summary", "No account, no tracking"):
        assert words in page, words


def _requirement(page: str, os_heading: str, row: str) -> tuple[str, str]:
    section = page.split(f"### {os_heading}", 1)[1].split("###", 1)[0]
    match = re.search(rf"^\| {re.escape(row)} \| ([^|]+) \| ([^|]+) \|$", section, re.MULTILINE)
    assert match, f"no '{row}' row under {os_heading}"
    return match.group(1).strip(), match.group(2).strip()


def _gb(cell: str) -> float:
    return float(re.search(r"(\d+(?:\.\d+)?) GB", cell).group(1))


def _specs(os_name: str, ram_gb: float, vram_gb: float = 0.0) -> SystemSpecs:
    apple = os_name == "Darwin"
    gpus = [GPUInfo("Apple M1", "apple", round(ram_gb * 0.66, 1))] if apple else (
        [GPUInfo("A graphics card", "nvidia", vram_gb)] if vram_gb else [])
    return SystemSpecs(os_name, "", "arm64" if apple else "x86_64", "processor", 4, 8, ram_gb, ram_gb * 0.6, 100.0,
                       gpus, unified_memory=apple, ram_bandwidth_gbs=20.0,
                       cpu_flags=["neon"] if apple else ["avx2", "fma", "f16c"])


@pytest.mark.parametrize("heading, os_name", [("Windows", "Windows"), ("macOS", "Darwin"),
                                               ("SteamOS + Linux", "Linux")])
def test_store_page_system_requirements_match_the_fit_engine(heading, os_name) -> None:
    page = (STEAM / "STORE_PAGE.md").read_text(encoding="utf-8")
    minimum, recommended = _requirement(page, heading, "Memory")
    # Minimum: the fit engine finds a model that runs comfortably, with no graphics card needed.
    pick = catalog.recommend(_specs(os_name, _gb(minimum)))
    assert pick is not None and pick.verdict in ("great", "ok"), (heading, minimum)
    # Recommended: a 4B-class model or better.
    vram = 0.0
    if os_name != "Darwin":
        vram = _gb(re.search(r"(\d+) GB\+ video memory", _requirement(page, heading, "Graphics")[1]).group(0))
    pick = catalog.recommend(_specs(os_name, _gb(recommended), vram))
    assert pick is not None and pick.verdict in ("great", "ok") and pick.model.params_b >= 4, (heading, recommended)


def test_store_page_mac_minimum_matches_the_app() -> None:
    page = (STEAM / "STORE_PAGE.md").read_text(encoding="utf-8")
    minimum, _recommended = _requirement(page, "macOS", "OS")
    spec_text = SPEC.read_text(encoding="utf-8")
    version = re.search(r'^MACOS_MINIMUM = "([\d.]+)"', spec_text, re.MULTILINE).group(1)
    assert f"macOS {version}" in minimum
