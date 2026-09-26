"""Tests for gettowork.distribution: which copy of the game is this?

Every test builds its own fake game folder in tmp_path and points
``sys.executable`` (and friends) at it, so nothing depends on how pytest runs.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest

from gettowork import __version__, distribution
from gettowork.distribution import Distribution, find_distribution_file, load

RELEASE = {"schema": 1, "channel": "release", "engine_downloads": False, "engine_dir": "engine",
           "llama_cpp_tag": "b7000", "app_version": "0.2.0", "built_from": "abc1234"}


@pytest.fixture(autouse=True)
def clean_slate(monkeypatch, tmp_path):
    """No distribution.json anywhere, no overrides, and a fresh cache (restored afterwards)."""
    for var in ("GETTOWORK_DISTRIBUTION", "GETTOWORK_ENGINE_DIR", "GETTOWORK_ALLOW_ENGINE_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    python = tmp_path / "python" / "bin" / "python3"
    python.parent.mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.setattr(distribution, "_cache", None)
    yield


def write(path: Path, data) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def game_folder(tmp_path: Path, monkeypatch, data=RELEASE, name: str = "GetToWork") -> Path:
    """A Windows/Linux-style built game: GetToWork(.exe) + distribution.json side by side."""
    folder = tmp_path / name
    folder.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    write(folder / "distribution.json", data)
    return folder


# ---------------------------------------------------------------------------
# No distribution.json: a developer copy
# ---------------------------------------------------------------------------


def test_no_file_means_a_developer_copy_that_downloads_the_engine():
    dist = load()
    assert dist == Distribution(channel="dev", engine_downloads=True, engine_dirs=(), llama_cpp_tag=None,
                                app_version=__version__, root=None)
    assert not dist.bundled
    assert find_distribution_file() is None


def test_a_distribution_json_in_the_current_folder_is_never_picked_up(tmp_path, monkeypatch):
    # sys._MEIPASS only exists in a built game; without it there's no "." candidate.
    work = tmp_path / "work"
    write(work / "distribution.json", RELEASE)
    monkeypatch.chdir(work)
    assert find_distribution_file() is None
    assert load().channel == "dev"


def test_an_empty_sys_executable_never_means_the_current_folder(tmp_path, monkeypatch):
    work = tmp_path / "work"
    write(work / "distribution.json", RELEASE)
    monkeypatch.chdir(work)
    monkeypatch.setattr(sys, "executable", "")
    assert find_distribution_file() is None


# ---------------------------------------------------------------------------
# Finding distribution.json in a built game
# ---------------------------------------------------------------------------


def test_a_built_game_folder(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, monkeypatch)
    dist = load()
    assert find_distribution_file() == folder / "distribution.json"
    assert dist.channel == "release"
    assert dist.engine_downloads is False
    assert dist.engine_dirs == (folder / "engine",)
    assert dist.llama_cpp_tag == "b7000" and dist.app_version == "0.2.0" and dist.built_from == "abc1234"
    assert dist.root == folder
    assert dist.notes == ()
    assert not dist.bundled  # the engine folder doesn't exist yet...
    (folder / "engine").mkdir()
    assert dist.bundled  # ...and now it does (checked live)


def test_a_mac_app_keeps_it_in_contents_resources(tmp_path, monkeypatch):
    app = tmp_path / "Get To Work.app" / "Contents"
    (app / "MacOS").mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(app / "MacOS" / "GetToWork"))
    write(app / "Resources" / "distribution.json", RELEASE)
    (app / "Resources" / "engine").mkdir()
    dist = load()
    assert find_distribution_file() == app / "Resources" / "distribution.json"
    assert dist.engine_dirs == (app / "Resources" / "engine",) and dist.bundled


def test_pyinstallers_bundle_folder_is_the_last_candidate(tmp_path, monkeypatch):
    internal = tmp_path / "GetToWork" / "_internal"
    write(internal / "distribution.json", dict(RELEASE, llama_cpp_tag="b7100"))
    monkeypatch.setattr(sys, "_MEIPASS", str(internal), raising=False)
    assert find_distribution_file() == internal / "distribution.json"
    assert load().llama_cpp_tag == "b7100"

    # ...but a file next to the executable wins.
    game_folder(tmp_path, monkeypatch)
    assert load(refresh=True).llama_cpp_tag == "b7000"


def test_the_environment_variable_wins_over_everything(tmp_path, monkeypatch):
    game_folder(tmp_path, monkeypatch)
    other = write(tmp_path / "elsewhere" / "custom.json", dict(RELEASE, channel="beta", engine_dir="builds"))
    monkeypatch.setenv("GETTOWORK_DISTRIBUTION", str(other))
    dist = load()
    assert dist.channel == "beta"
    assert dist.engine_dirs == (tmp_path / "elsewhere" / "builds",)  # relative to the file's own folder
    assert dist.root == tmp_path / "elsewhere"


def test_a_missing_file_named_by_the_environment_is_skipped(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, monkeypatch)
    monkeypatch.setenv("GETTOWORK_DISTRIBUTION", str(tmp_path / "nope.json"))
    assert find_distribution_file() == folder / "distribution.json"


def test_a_folder_called_distribution_json_is_not_a_file(tmp_path, monkeypatch):
    folder = tmp_path / "GetToWork"
    (folder / "distribution.json").mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    assert find_distribution_file() is None
    assert load().channel == "dev"


# ---------------------------------------------------------------------------
# Reading the file: every field, tolerant of mistakes
# ---------------------------------------------------------------------------


def test_minimal_file_uses_sensible_release_defaults(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, monkeypatch, data={"schema": 1})
    dist = load()
    assert dist.channel == "release"
    assert dist.engine_downloads is False  # a built game ships its engine
    assert dist.engine_dirs == (folder / "engine",)
    assert dist.llama_cpp_tag is None and dist.app_version == __version__ and dist.built_from is None


def test_a_file_may_allow_engine_downloads(tmp_path, monkeypatch):
    game_folder(tmp_path, monkeypatch, data=dict(RELEASE, engine_downloads=True))
    assert load().engine_downloads is True


def test_several_and_absolute_engine_dirs(tmp_path, monkeypatch):
    absolute = tmp_path / "shared-engines"
    folder = game_folder(tmp_path, monkeypatch, data=dict(RELEASE, engine_dir=["engine", str(absolute)]))
    assert load().engine_dirs == (folder / "engine", absolute)


def test_wrong_types_fall_back_to_defaults_with_a_note(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, monkeypatch, data={
        "channel": 7, "engine_downloads": "no", "engine_dir": 5, "llama_cpp_tag": 7000, "app_version": ["x"],
        "built_from": None})
    dist = load()
    assert dist.channel == "release"
    assert dist.engine_downloads is False
    assert dist.engine_dirs == (folder / "engine",)
    assert dist.llama_cpp_tag is None and dist.app_version == __version__
    assert len(dist.notes) == 2  # engine_downloads and engine_dir


@pytest.mark.parametrize("text", ["{not json", "[1, 2, 3]", "", "\"release\"", "null"])
def test_a_damaged_file_never_crashes_the_game(tmp_path, monkeypatch, caplog, text):
    folder = game_folder(tmp_path, monkeypatch, data=text)
    with caplog.at_level(logging.WARNING, logger="gettowork.distribution"):
        dist = load()
    assert dist.channel == "dev" and dist.engine_downloads is True
    assert dist.llama_cpp_tag is None
    assert dist.notes and "distribution.json" in dist.notes[0]
    assert "distribution.json" in caplog.text
    # The engine folder next to the file is still looked in, so the game keeps its own engine.
    assert dist.engine_dirs == (folder / "engine",)


@pytest.mark.parametrize("text", ["{not json", ""])
def test_a_damaged_file_in_a_built_game_never_turns_on_engine_downloads(tmp_path, monkeypatch, text):
    """A built game never downloads programs, even if its distribution.json is damaged."""
    folder = game_folder(tmp_path, monkeypatch, data=text)
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    dist = load()
    assert dist.engine_downloads is False
    assert dist.engine_dirs == (folder / "engine",)
    assert dist.notes


def test_a_built_game_without_its_file_keeps_its_engine_and_never_downloads(tmp_path, monkeypatch, caplog):
    folder = tmp_path / "GetToWork"
    (folder / "engine").mkdir(parents=True)
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    with caplog.at_level(logging.WARNING, logger="gettowork.distribution"):
        dist = load()
    assert dist.engine_downloads is False
    assert dist.engine_dirs[0] == folder / "engine" and dist.bundled
    assert dist.engine_dirs[1] == tmp_path / "Resources" / "engine"  # (the Mac app layout)
    assert dist.llama_cpp_tag is None  # unknown: --specs says so, and the build check fails on it
    assert dist.notes and "distribution.json" in caplog.text


def test_undecodable_bytes_are_a_damaged_file_too(tmp_path, monkeypatch):
    folder = tmp_path / "GetToWork"
    folder.mkdir()
    (folder / "distribution.json").write_bytes(b"\xff\xfe\x00garbage")
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    dist = load()
    assert dist.channel == "dev" and dist.notes


def test_load_never_raises_even_if_something_unexpected_breaks(monkeypatch, caplog):
    def boom():
        raise RuntimeError("surprise")

    monkeypatch.setattr(distribution, "_discover", boom)
    with caplog.at_level(logging.WARNING, logger="gettowork.distribution"):
        dist = load(refresh=True)
    assert dist.channel == "dev" and dist.engine_downloads is True and dist.engine_dirs == ()
    assert "surprise" in dist.notes[0]


def test_a_surprise_in_a_built_game_still_keeps_its_engine_and_never_downloads(tmp_path, monkeypatch):
    def boom():
        raise RecursionError("maximum recursion depth exceeded")  # e.g. a pathologically nested file

    folder = tmp_path / "GetToWork"
    folder.mkdir()
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(distribution, "_discover", boom)
    dist = load(refresh=True)
    assert dist.engine_downloads is False
    assert folder / "engine" in dist.engine_dirs
    assert "recursion" in dist.notes[0]


@pytest.mark.parametrize("variable, value", [("GETTOWORK_ENGINE_DIR", "~nosuchuser_zz/engine"),
                                             ("GETTOWORK_DISTRIBUTION", "~nosuchuser_zz/distribution.json")])
def test_a_path_with_an_unknown_home_folder_is_skipped_not_fatal(tmp_path, monkeypatch, variable, value):
    folder = tmp_path / "GetToWork"
    folder.mkdir()
    monkeypatch.setattr(sys, "executable", str(folder / "GetToWork"))
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setenv(variable, value)
    dist = load(refresh=True)
    assert dist.engine_downloads is False and folder / "engine" in dist.engine_dirs


def test_relative_paths_from_the_environment_are_made_absolute(tmp_path, monkeypatch):
    """The engine runs with its own folder as the working directory: a relative engine path would
    point somewhere else there."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GETTOWORK_ENGINE_DIR", os.path.join("game", "engine"))
    dist = load(refresh=True)
    assert dist.engine_dirs == (Path.cwd() / "game" / "engine",) and dist.engine_dirs[0].is_absolute()
    built = tmp_path / "built"
    built.mkdir()
    (built / "distribution.json").write_text(json.dumps({"schema": 1, "channel": "release"}), encoding="utf-8")
    monkeypatch.delenv("GETTOWORK_ENGINE_DIR")
    monkeypatch.setenv("GETTOWORK_DISTRIBUTION", os.path.join("built", "distribution.json"))
    dist = load(refresh=True)
    assert dist.root == Path.cwd() / "built" and dist.engine_dirs == (Path.cwd() / "built" / "engine",)


# ---------------------------------------------------------------------------
# Environment overrides
# ---------------------------------------------------------------------------


def test_an_extra_engine_dir_is_checked_first(tmp_path, monkeypatch):
    folder = game_folder(tmp_path, monkeypatch)
    extra = tmp_path / "ci-engine"
    extra.mkdir()
    monkeypatch.setenv("GETTOWORK_ENGINE_DIR", str(extra))
    dist = load()
    assert dist.engine_dirs == (extra, folder / "engine")
    assert dist.bundled and dist.engine_downloads is False and dist.channel == "release"


def test_an_extra_engine_dir_works_in_a_developer_copy_too(tmp_path, monkeypatch):
    first, second = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("GETTOWORK_ENGINE_DIR", f"{first}{os.pathsep}{second}{os.pathsep}")
    dist = load()
    assert dist.engine_dirs == (first, second)
    assert dist.channel == "dev" and dist.engine_downloads is True  # downloads stay on
    assert not dist.bundled  # neither folder exists


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("yes", True), ("0", False),
                                             ("off", False), ("maybe", None), ("", None)])
def test_the_download_switch(tmp_path, monkeypatch, value, expected):
    game_folder(tmp_path, monkeypatch)  # a built game: downloads off by default
    monkeypatch.setenv("GETTOWORK_ALLOW_ENGINE_DOWNLOAD", value)
    assert load(refresh=True).engine_downloads is (False if expected is None else expected)

    (tmp_path / "GetToWork" / "distribution.json").unlink()  # a developer copy: downloads on by default
    assert load(refresh=True).engine_downloads is (True if expected is None else expected)


def test_overrides_keep_the_rest_of_the_file(tmp_path, monkeypatch):
    game_folder(tmp_path, monkeypatch)
    monkeypatch.setenv("GETTOWORK_ALLOW_ENGINE_DOWNLOAD", "1")
    dist = load()
    assert (dist.channel, dist.llama_cpp_tag, dist.app_version, dist.built_from) == ("release", "b7000", "0.2.0", "abc1234")


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


def test_the_answer_is_remembered_until_refreshed(tmp_path, monkeypatch):
    first = load()
    assert load() is first
    game_folder(tmp_path, monkeypatch)
    assert load() is first  # still the remembered answer
    fresh = load(refresh=True)
    assert fresh.channel == "release" and load() is fresh


def test_distribution_is_frozen():
    with pytest.raises(Exception):
        load().channel = "hacked"  # type: ignore[misc]
