"""Where the game keeps its settings and downloaded models.

Nothing here is required: the game works fine with no saved config. Settings
are a small JSON file; model files live in a separate data directory.
"""

from __future__ import annotations

import json
import os
import platform
import stat
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import ClassVar, Optional

# Credentials for other services that no child program (the llama.cpp engine,
# a hardware-detection tool) ever needs: left out of their environment.
SECRET_ENV_VARS = frozenset({
    "TYPESAFE_API_KEY", "GITHUB_TOKEN", "GH_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN",
    "HUGGINGFACE_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
})


def child_env(base: Optional[dict] = None) -> dict[str, str]:
    """A copy of the environment for a child program, without SECRET_ENV_VARS."""
    source = os.environ if base is None else base
    return {k: v for k, v in source.items() if k.upper() not in SECRET_ENV_VARS}

APP_DIR_NAME = "GetToWork"


def config_dir() -> Path:
    """The game's one folder: settings, models, the engine and caches (delete it to remove them all).

    On Windows this is ``%LOCALAPPDATA%\\GetToWork``, not the *roaming*
    ``%APPDATA%``: multi-GB models and the engine are specific to this
    computer, and on managed PCs a roaming profile would copy them to a
    server at every log-off. (A folder an older version made in the roaming
    ``%APPDATA%`` keeps being used, so nothing is downloaded twice.)
    ``GETTOWORK_HOME`` overrides it everywhere.
    """
    override = os.environ.get("GETTOWORK_HOME")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Windows":
        local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / APP_DIR_NAME
        roaming_base = os.environ.get("APPDATA")
        roaming = Path(roaming_base) / APP_DIR_NAME if roaming_base else None
        if roaming is not None and roaming.is_dir() and not local.exists():
            return roaming  # an install from an older version: keep its models and settings
        return local
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIR_NAME.lower()


_chosen_models_dir: Optional[Path] = None  # set by use_models_dir() (--models-dir, or the saved choice)


def models_dir() -> Path:
    """Where GGUF files downloaded for the llama.cpp backend are stored.

    ``GETTOWORK_MODELS_DIR`` wins; then the folder chosen with ``--models-dir``
    (remembered in the settings, see :func:`use_models_dir`); else
    ``<config dir>/models``.
    """
    override = os.environ.get("GETTOWORK_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    if _chosen_models_dir is not None:
        return _chosen_models_dir
    return config_dir() / "models"


def use_models_dir(folder: Optional[str | Path]) -> None:
    """Keep models in `folder` from now on in this run (None = the default place).

    ``cli`` calls it at start-up with the ``--models-dir`` option or the
    folder remembered in the settings - so a player whose system drive is
    small can keep multi-GB models on another drive (on Steam for Windows,
    where launch options can't set environment variables, too).
    """
    global _chosen_models_dir
    _chosen_models_dir = Path(folder).expanduser() if folder else None


def command_name() -> str:
    """What to type to run the game in a terminal.

    ``gettowork`` when run from source (pip), ``gettowork-cli`` in a built game
    (Steam, the double-click builds), whose terminal program has that name.
    """
    return "gettowork-cli" if getattr(sys, "frozen", False) else "gettowork"


def runtime_dir() -> Path:
    """Where the automatically installed llama.cpp engine lives."""
    return config_dir() / "runtime"


def cache_dir() -> Path:
    """Where cached Hugging Face search results are kept."""
    return config_dir() / "cache"


@dataclass
class Settings:
    backend: Optional[str] = None  # "managed" | "ollama" | "llamacpp"
    model_key: Optional[str] = None  # catalog key or Hugging Face repo id
    model_quant: Optional[str] = None  # quantization chosen for this machine, e.g. "Q4_K_M"
    model_path: Optional[str] = None  # local GGUF path (managed / llamacpp backends)
    server_exe: Optional[str] = None  # installed llama-server executable (managed backend)
    ollama_model: Optional[str] = None  # model tag used with ollama
    jev_enabled: Optional[bool] = None  # remembered answer to "enable Jev?"
    # Only stored if the player explicitly opted in. Never shown by repr(), so a
    # stray debug print or traceback can't reveal it.
    jev_api_key: Optional[str] = field(default=None, repr=False)
    models_dir: Optional[str] = None  # where models are kept, when chosen with --models-dir
    extra: dict = field(default_factory=dict)
    # Set by save(): could the saved key's file be made readable by this user only?
    # (None = no key saved.) Not a setting, so it's never written to the file.
    key_file_protected: ClassVar[Optional[bool]] = None

    @property
    def path(self) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        """The saved settings, or defaults.

        A missing, unreadable or hand-edited file never stops the game: anything
        that isn't the expected shape (a list instead of an object, a number
        where a name belongs...) is ignored and that setting starts fresh.
        """
        p = config_dir() / "settings.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = {k: v for k, v in data.items() if k in _FIELD_TYPES and isinstance(v, _FIELD_TYPES[k])}
        return cls(**known)

    def save(self) -> Path:
        """Write the settings atomically. The file may hold an API key, so it is
        created owner-only from the start (there's never a moment when other
        users could read it), and a failed save leaves no copy behind.

        On Windows, permission bits don't protect a file - it inherits its
        folder's access list, and on a second drive (``GETTOWORK_HOME=D:\\...``)
        that usually lets every account on the PC read it. So while a key is
        saved, the file gets an owner-only access list of its own (``icacls``).
        ``key_file_protected`` then says whether that worked (None = no key).
        """
        p = self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        data = json.dumps(asdict(self), indent=2).encode("utf-8")
        try:
            tmp.unlink()  # a leftover from an interrupted save may have looser permissions
        except FileNotFoundError:
            pass
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
                         stat.S_IRUSR | stat.S_IWUSR)
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            try:
                os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)  # in case a umask/ACL widened it (no-op on Windows)
            except OSError:
                pass
            self.key_file_protected = None
            if self.jev_api_key:
                self.key_file_protected = restrict_to_owner_windows(tmp) if _on_windows() else True
            os.replace(tmp, p)  # (the file keeps its own access list when it's moved into place)
        except BaseException:
            try:
                tmp.unlink()
            except OSError:
                pass
            raise
        return p

    @staticmethod
    def reset() -> None:
        """Forget everything saved (including a key), leaving downloaded models alone."""
        for name in ("settings.json", "settings.tmp"):
            try:
                (config_dir() / name).unlink()
            except FileNotFoundError:
                pass


def _on_windows() -> bool:
    return platform.system() == "Windows"


def restrict_to_owner_windows(path: Path, *, runner: Optional[object] = None) -> bool:
    """Give a file an access list of its own that only the current Windows user can use.

    ``icacls FILE /inheritance:r /grant:r DOMAIN\\user:F`` removes the rights it
    would inherit from its folder (e.g. "Users: read" on a second drive) and
    grants full control to this user alone. Returns True if that worked.
    Never raises.
    """
    import subprocess

    user = os.environ.get("USERNAME") or ""
    if not user:
        return False
    domain = os.environ.get("USERDOMAIN") or ""
    account = f"{domain}\\{user}" if domain else user
    icacls = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "icacls.exe")
    run = runner or subprocess.run
    try:
        result = run(  # type: ignore[operator]
            [icacls, str(path), "/inheritance:r", "/grant:r", f"{account}:F"],
            capture_output=True, timeout=10, env=child_env(),
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except Exception:
        return False
    return getattr(result, "returncode", 1) == 0


# The type each saved setting must have; anything else in the file is ignored.
_FIELD_TYPES: dict[str, type | tuple[type, ...]] = {
    "backend": str,
    "model_key": str,
    "model_quant": str,
    "model_path": str,
    "server_exe": str,
    "ollama_model": str,
    "jev_enabled": bool,
    "jev_api_key": str,
    "models_dir": str,
    "extra": dict,
}
