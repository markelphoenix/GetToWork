"""Where the game keeps its settings and downloaded models.

Nothing here is required: the game works fine with no saved config. Settings
are a small JSON file; model files live in a separate data directory.
"""

from __future__ import annotations

import json
import os
import platform
import stat
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

APP_DIR_NAME = "GetToWork"


def config_dir() -> Path:
    override = os.environ.get("GETTOWORK_HOME")
    if override:
        return Path(override).expanduser()
    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return base / APP_DIR_NAME
    if system == "Darwin":
        return Path.home() / "Library" / "Application Support" / APP_DIR_NAME
    base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / APP_DIR_NAME.lower()


def models_dir() -> Path:
    """Where GGUF files downloaded for the llama.cpp backend are stored."""
    override = os.environ.get("GETTOWORK_MODELS_DIR")
    if override:
        return Path(override).expanduser()
    return config_dir() / "models"


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
    jev_api_key: Optional[str] = None  # only stored if the player explicitly opted in
    extra: dict = field(default_factory=dict)

    @property
    def path(self) -> Path:
        return config_dir() / "settings.json"

    @classmethod
    def load(cls) -> "Settings":
        p = config_dir() / "settings.json"
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def save(self) -> Path:
        p = self.path
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        try:
            # The file may contain an API key: owner read/write only (no-op on Windows).
            os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
        except OSError:
            pass
        os.replace(tmp, p)
        return p

    @staticmethod
    def reset() -> None:
        try:
            (config_dir() / "settings.json").unlink()
        except FileNotFoundError:
            pass
