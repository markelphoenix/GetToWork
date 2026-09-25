"""Tests for gettowork.config: where things are stored, and a settings file that never breaks the game."""

from __future__ import annotations

import json
import os
import stat
import sys

import pytest

from gettowork import config
from gettowork.catalog import MODEL_CATALOG
from gettowork.config import Settings
from gettowork.setup_flow import entry_from_dict, entry_to_dict

KEY = "tsk_live_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456wxyz"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("content", ["[]", "null", "5", '"x"', "{not json", ""])
def test_a_damaged_settings_file_gives_defaults(home, content):
    (home / "settings.json").write_text(content, encoding="utf-8")
    assert Settings.load() == Settings()


def test_settings_with_the_wrong_types_are_ignored_one_by_one(home):
    (home / "settings.json").write_text(json.dumps({
        "backend": 5, "model_key": "qwen3-4b", "jev_enabled": "yes", "jev_api_key": 1234,
        "extra": [], "model_path": None, "unknown": "ignored",
    }), encoding="utf-8")
    loaded = Settings.load()
    assert loaded.model_key == "qwen3-4b"
    assert loaded.backend is None and loaded.jev_enabled is None and loaded.jev_api_key is None
    assert loaded.extra == {} and loaded.model_path is None


def test_settings_round_trip(home):
    Settings(backend="managed", model_key="qwen3-4b", jev_enabled=True, jev_api_key=KEY, extra={"a": 1}).save()
    loaded = Settings.load()
    assert loaded.backend == "managed" and loaded.jev_api_key == KEY and loaded.extra == {"a": 1}


def test_repr_never_shows_the_api_key():
    assert KEY not in repr(Settings(jev_api_key=KEY)) and KEY not in str(Settings(jev_api_key=KEY))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_the_key_file_is_owner_only_from_the_moment_it_is_created(home, monkeypatch):
    modes = []
    real_replace = os.replace

    def spy(src, dst):
        modes.append(stat.S_IMODE(os.stat(src).st_mode))  # the temp file, just before it becomes settings.json
        return real_replace(src, dst)

    monkeypatch.setattr(config.os, "replace", spy)
    old_umask = os.umask(0o022)
    try:
        path = Settings(jev_api_key=KEY).save()
    finally:
        os.umask(old_umask)
    assert modes == [0o600]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600


def test_a_failed_save_leaves_no_copy_of_the_key_behind(home, monkeypatch):
    def broken(src, dst):
        raise PermissionError("the file is locked")

    monkeypatch.setattr(config.os, "replace", broken)
    with pytest.raises(OSError):
        Settings(jev_api_key=KEY).save()
    assert list(home.iterdir()) == []


def test_reset_also_removes_a_leftover_temp_file(home):
    (home / "settings.json").write_text("{}", encoding="utf-8")
    (home / "settings.tmp").write_text(json.dumps({"jev_api_key": KEY}), encoding="utf-8")
    Settings.reset()
    assert list(home.iterdir()) == []
    Settings.reset()  # nothing left to delete: still fine


def test_windows_keeps_big_files_out_of_the_roaming_profile(tmp_path, monkeypatch):
    monkeypatch.delenv("GETTOWORK_HOME", raising=False)
    monkeypatch.setattr(config.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config.models_dir() == tmp_path / "Local" / "GetToWork" / "models"
    assert config.runtime_dir() == tmp_path / "Local" / "GetToWork" / "runtime"
    monkeypatch.delenv("LOCALAPPDATA")
    assert config.config_dir().parts[-3:] == ("AppData", "Local", "GetToWork")


def test_saved_model_entries_are_validated():
    entry = MODEL_CATALOG[0]
    data = entry_to_dict(entry)
    assert entry_from_dict(data) == entry
    for broken in ({"quant": None}, {"key": ""}, {"hf_repo": 5}, {"file_size_gb": "big"}, {"params_b": -1}):
        assert entry_from_dict({**data, **broken}) is None, broken
    assert entry_from_dict([]) is None and entry_from_dict(None) is None


def test_on_windows_a_saved_key_gets_an_owner_only_access_list(tmp_path, monkeypatch):
    """Permission bits don't protect a file on Windows: it inherits its folder's
    access list (on a second drive, usually readable by every account)."""
    from gettowork import config as cfg

    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    monkeypatch.setattr(cfg, "_on_windows", lambda: True)
    calls = []

    def fake_icacls(path, **kwargs):
        calls.append(str(path))
        return True

    monkeypatch.setattr(cfg, "restrict_to_owner_windows", fake_icacls)
    s = cfg.Settings(jev_api_key="tsk_live_" + "a" * 32)
    s.save()
    assert calls and calls[0].endswith("settings.tmp") and s.key_file_protected is True
    no_key = cfg.Settings()
    no_key.save()
    assert len(calls) == 1 and no_key.key_file_protected is None  # nothing secret: nothing to lock down
    assert "key_file_protected" not in (tmp_path / "settings.json").read_text()


def test_the_icacls_command_removes_inherited_rights_and_grants_only_this_user(monkeypatch, tmp_path):
    from gettowork import config as cfg

    monkeypatch.setenv("USERNAME", "ada")
    monkeypatch.setenv("USERDOMAIN", "PC")
    monkeypatch.setenv("SystemRoot", r"C:\Windows")
    seen = {}

    class Result:
        returncode = 0

    def run(args, **kwargs):
        seen["args"] = args
        return Result()

    assert cfg.restrict_to_owner_windows(tmp_path / "settings.tmp", runner=run)
    args = seen["args"]
    assert args[0].endswith("icacls.exe") and "/inheritance:r" in args and args[-1] == "PC\\ada:F"

    def fails(args, **kwargs):
        raise OSError("no icacls")

    assert cfg.restrict_to_owner_windows(tmp_path / "x", runner=fails) is False
