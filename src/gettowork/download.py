"""Download GGUF model files from Hugging Face, exactly and politely.

A GGUF repo usually holds many versions of one model (Q4_K_M, Q8_0, ...), and
big models are split into several "shards" (`-00001-of-00003.gguf` ...) that
must all sit in the same folder. This module:

1. works out the exact file(s) for the chosen quantization (`resolve_files`),
2. skips files that are already complete on disk (so it also works offline),
3. checks there's enough free disk space,
4. tells the player what it's about to fetch, from where, under which license,
5. downloads with `huggingface_hub.hf_hub_download` and a friendly progress bar,
6. turns every failure into one clear sentence (`DownloadError`).

The weights themselves are never part of this game: they come straight from
Hugging Face, shared by each model's authors under their own license.
"""

from __future__ import annotations

import dataclasses
import errno
import functools
import inspect
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Optional

from rich.markup import escape

from . import catalog, config
from .hf_discovery import (
    FALLBACK_QUANT_ORDER,
    entry_from_hub,
    group_quant_files,
    license_of,
    make_hub_api,
    parse_quant,
    prettify_repo_name,
    rejection_reason,
    repo_gguf_files,
    restricted_family,
    shard_info,
    with_deadline,
)
from .types import ModelEntry
from .ui import UI

__all__ = [
    "DownloadError",
    "pick_gguf_file",
    "resolve_files",
    "download_gguf",
    "download_custom_gguf",
    "custom_entry",
    "model_folder",
    "normalize_repo_id",
    "find_local_copy",
    "downloaded_quants",
]

DISK_HEADROOM_BYTES = 500 * 1000**2  # keep ~0.5 GB free after the download
HF_LOGIN_HINT = "hf auth login"  # the huggingface_hub command that stores a login token
LISTING_DEADLINE_S = 60.0  # the longest we wait for a repo's file list (or its metadata)


class DownloadError(RuntimeError):
    """A download problem, explained in one plain-English sentence.

    `kind` lets callers react: "not_found", "gated", "no_gguf", "missing_file",
    "network", "offline_mode", "server", "disk", "incomplete", "bad_repo_id" or "other".
    """

    def __init__(self, message: str, kind: str = "other") -> None:
        super().__init__(message)
        self.kind = kind


# ---------------------------------------------------------------------------
# Choosing files
# ---------------------------------------------------------------------------


def _basename(path: str) -> str:
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def _by_location(files: list[str]) -> list[str]:
    """Repo root first, then shorter paths (sub-folders usually hold alternatives)."""
    return sorted(files, key=lambda f: (f.count("/"), len(f), f))


def pick_gguf_file(filenames: list[str], quant: str) -> Optional[str]:
    """Pick the GGUF file for `quant` from a repo's file list (the first shard if split).

    Case-insensitive. Only `.gguf` files; vision projectors (`mmproj`) are
    skipped. Exact quant first ("UD-Q4_K_XL" also matches a plain "Q4_K_XL"
    and vice versa), then the fallback order Q4_K_M, Q4_K_S, Q5_K_M, Q4_0,
    IQ4_XS, Q6_K, Q8_0, MXFP4, then anything (nearest ~4.8 bits). Files at the
    repo root win over sub-folders. None if the repo has no usable GGUF.
    """
    candidates = [
        f for f in filenames
        if f.lower().endswith(".gguf") and "mmproj" not in _basename(f).lower() and shard_info(f)[1] == 1
    ]
    if not candidates:
        return None
    tags = {f: (parse_quant(f) or "").upper() for f in candidates}
    wanted = (quant or "").strip().upper()

    def matching(test: Callable[[str], bool]) -> Optional[str]:
        found = [f for f in candidates if tags[f] and test(tags[f])]
        return _by_location(found)[0] if found else None

    if wanted:
        plain = wanted.removeprefix("UD-")
        choice = matching(lambda t: t == wanted) or matching(lambda t: t.removeprefix("UD-") == plain)
        if choice:
            return choice
    for tag in FALLBACK_QUANT_ORDER:
        choice = matching(lambda t, tag=tag: t == tag)
        if choice:
            return choice
    # Anything else: the known quant nearest Q4_K_M's ~4.8 bits, else the first file.
    def closeness(f: str) -> tuple:
        bits = catalog.quant_bits(tags[f])
        return (bits is None, abs((bits or 0.0) - 4.8), f.count("/"), len(f), f)

    return min(candidates, key=closeness)


def _shard_set(first: str, filenames: list[str], repo_id: str) -> tuple[str, ...]:
    """All parts of a split model, in order (just `first` for a single file)."""
    group, _, total = shard_info(first)
    if total == 1:
        return (first,)
    parts = {}
    for name in filenames:
        g, part, of = shard_info(name)
        if g == group and of == total:
            parts[part] = name
    missing = [i for i in range(1, total + 1) if i not in parts]
    if missing:
        raise DownloadError(
            f"Part {missing[0]} of {total} of {_basename(first)} is missing from {repo_id}, so the model "
            "can't be loaded. Please pick another model or version.",
            "missing_file",
        )
    return tuple(parts[i] for i in range(1, total + 1))


def _hint_matches(entry: ModelEntry, wanted: str) -> bool:
    """Do the entry's pre-recorded `gguf_files` belong to the quant we want?"""
    if not entry.gguf_files:
        return False
    if not wanted or wanted.upper() == (entry.quant or "").upper():
        return True
    return (parse_quant(entry.gguf_files[0]) or "").upper() == wanted.upper()


def resolve_files(entry: ModelEntry, quant: Optional[str] = None, *, hf_api: Any = None) -> tuple[str, ...]:
    """The exact file name(s) to download for `entry` at `quant` (default: `entry.quant`).

    Uses the entry's recorded `gguf_files` when they match the quant (no
    network); otherwise lists the repo once and picks with `pick_gguf_file`,
    including every shard of a split model. Raises DownloadError.
    """
    wanted = (quant or entry.quant or "").strip()
    if _hint_matches(entry, wanted):
        return tuple(entry.gguf_files)
    listing = _list_repo(hf_api or _default_api(), entry.hf_repo)
    return _choose_files([name for name, _ in listing], wanted, entry.hf_repo)


def _choose_files(names: list[str], wanted: str, repo_id: str) -> tuple[str, ...]:
    first = pick_gguf_file(names, wanted)
    if first is None:
        raise DownloadError(
            f"{repo_id} doesn't contain any GGUF model files, which is the format the game's engine needs. "
            "Try a repo whose name ends in -GGUF.",
            "no_gguf",
        )
    return _shard_set(first, names, repo_id)


# ---------------------------------------------------------------------------
# Hub access and friendly errors
# ---------------------------------------------------------------------------


def _default_api() -> Any:
    return make_hub_api()  # HfApi() with finite request timeouts


def _default_downloader() -> Callable[..., Any]:
    from huggingface_hub import hf_hub_download

    return hf_hub_download


@functools.lru_cache(maxsize=1)
def _hub_errors() -> SimpleNamespace:
    """huggingface_hub's exception classes, looked up by name.

    Their home moved between versions: some releases have a
    ``huggingface_hub.errors`` module that holds only *some* of them, with the
    rest still in ``huggingface_hub.utils``. So each name is looked up in
    both places (a stand-in that never matches if neither has it).
    """
    import importlib

    modules = []
    for name in ("huggingface_hub.errors", "huggingface_hub.utils"):
        try:
            modules.append(importlib.import_module(name))
        except ImportError:
            continue

    class _Never(Exception):
        """Stands in for a class this huggingface_hub version doesn't have."""

    def find(name: str) -> type:
        for module in modules:
            found = getattr(module, name, None)
            if isinstance(found, type) and issubclass(found, BaseException):
                return found
        return _Never

    names = ("GatedRepoError", "RepositoryNotFoundError", "EntryNotFoundError", "LocalEntryNotFoundError",
             "OfflineModeIsEnabled", "RevisionNotFoundError")
    return SimpleNamespace(**{name: find(name) for name in names})


def _friendly_error(exc: BaseException, repo_id: str, filename: Optional[str] = None) -> DownloadError:
    """Translate whatever went wrong into one clear sentence."""
    if isinstance(exc, DownloadError):
        return exc
    errors = _hub_errors()
    page = f"https://huggingface.co/{repo_id}"
    what = _basename(filename) if filename else repo_id
    if isinstance(exc, errors.GatedRepoError):
        return DownloadError(
            f"{repo_id} is a gated model: its authors ask you to accept their terms first. Open {page} while "
            f"logged in to Hugging Face, accept the terms, run `{HF_LOGIN_HINT}` in a terminal, then try "
            "again - or simply pick a different model.",
            "gated",
        )
    if isinstance(exc, errors.RepositoryNotFoundError):
        return DownloadError(
            f"I couldn't find {repo_id} on Hugging Face. Check the spelling (it looks like owner/name) - it may "
            f"also be private or have been removed. If it's private, run `{HF_LOGIN_HINT}` first.",
            "not_found",
        )
    if isinstance(exc, errors.OfflineModeIsEnabled):
        return DownloadError(
            "Downloads are switched off because the HF_HUB_OFFLINE setting is on. Turn it off, or pick a "
            "model you've already downloaded.",
            "offline_mode",
        )
    if isinstance(exc, errors.LocalEntryNotFoundError):  # hf_hub_download's "couldn't connect" error
        return DownloadError(
            f"I couldn't reach Hugging Face to download {what}. Check your internet connection and try again.",
            "network",
        )
    if isinstance(exc, (errors.EntryNotFoundError, errors.RevisionNotFoundError)):
        return DownloadError(
            f"{what} isn't in {repo_id} any more (the authors may have renamed it). Try again and I'll look "
            "for the right file, or pick another model.",
            "missing_file",
        )
    if isinstance(exc, TimeoutError):
        return DownloadError(
            f"Hugging Face didn't answer in time while I was looking up {what}. Your connection may have "
            "stalled - check it and try again.",
            "network",
        )
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        return DownloadError(
            f"Your disk filled up while downloading {what}. Free up some space and try again - finished files are kept.",
            "disk",
        )
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if status == 429:
        return DownloadError("Hugging Face asked us to slow down (too many requests). Wait a minute and try again.",
                             "server")
    if isinstance(status, int) and status >= 500:
        return DownloadError(f"Hugging Face is having trouble right now (HTTP {status}). Please try again in a few minutes.",
                             "server")
    detail = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else type(exc).__name__
    return DownloadError(
        f"I couldn't download {what} from Hugging Face ({escape(detail)}). Check your internet connection and "
        "try again - anything already downloaded is kept.",
        "network",
    )


def _list_repo(api: Any, repo_id: str) -> list[tuple[str, int]]:
    """(file, bytes) for every GGUF in the repo, or a friendly DownloadError.

    Never waits longer than LISTING_DEADLINE_S, even on a stalled connection.
    """
    try:
        return with_deadline(lambda: repo_gguf_files(api, repo_id), LISTING_DEADLINE_S)
    except Exception as exc:
        raise _friendly_error(exc, repo_id) from exc


# ---------------------------------------------------------------------------
# Local files, disk space, progress
# ---------------------------------------------------------------------------


def model_folder(repo_id: str) -> Path:
    """Where a repo's files are saved by default: `models_dir()/<owner>--<name>`."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", repo_id.replace("/", "--"))
    return config.models_dir() / safe


def _local(dest: Path, name: str) -> Path:
    """Where hf_hub_download(local_dir=dest) puts a repo file (sub-folders included)."""
    return dest.joinpath(*[p for p in name.replace("\\", "/").split("/") if p])


def _is_complete(path: Path, size: Optional[int]) -> bool:
    """hf_hub_download only moves a file into place once it's fully downloaded,
    so an existing file is complete - if we know the size, it must match too."""
    try:
        return path.is_file() and path.stat().st_size > 0 and (not size or path.stat().st_size == size)
    except OSError:
        return False


def _fmt_bytes(n: float) -> str:
    if n >= 1e9:
        return f"{n / 1e9:.1f} GB"
    if n >= 1e6:
        return f"{n / 1e6:.0f} MB"
    if n >= 1e3:
        return f"{n / 1e3:.0f} KB"
    return f"{max(int(n), 0)} bytes"


def _check_disk(dest: Path, need_bytes: int) -> None:
    """Raise DownloadError if `dest`'s drive can't hold `need_bytes` plus some headroom."""
    probe = dest
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent  # the folder may not exist yet: measure its nearest existing parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # can't tell: let the download try
    if need_bytes + DISK_HEADROOM_BYTES > free:
        raise DownloadError(
            f"There isn't enough free disk space: this download needs about {_fmt_bytes(need_bytes)} (plus a "
            f"little spare) but only {_fmt_bytes(free)} is free at {probe}. Free up some space or pick a smaller model.",
            "disk",
        )


def _progress_bridge(advance: Callable[[int], None]) -> type:
    """A tqdm stand-in that feeds huggingface_hub's byte counts into our rich progress bar.

    hf_hub_download(tqdm_class=...) creates the bar and calls `update(n_bytes)`
    as data arrives; everything else it might call is accepted and ignored.
    """

    class RichProgressBridge:
        def __init__(self, *args: Any, total: Optional[int] = None, initial: int = 0, **kwargs: Any) -> None:
            self.total = total
            self.n = initial or 0
            self.disable = False
            if initial:
                advance(initial)  # resuming a partial download

        def update(self, n: int = 1) -> None:
            if n:
                self.n += n
                advance(n)

        def update_transfer(self, n: int = 1) -> None:
            """Xet downloads also report network bytes; we count bytes written (`update`)."""

        def close(self) -> None:
            pass

        def __enter__(self) -> "RichProgressBridge":
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def __getattr__(self, name: str) -> Callable[..., None]:
            return lambda *args, **kwargs: None  # set_postfix_str, refresh, reset, ...: harmless no-ops

    return RichProgressBridge


def _accepts(func: Callable[..., Any], name: str) -> bool:
    """Can `func` take the keyword argument `name`?"""
    try:
        params = inspect.signature(func).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is inspect.Parameter.VAR_KEYWORD for p in params)


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------


def _show_plan(ui: UI, entry: ModelEntry, quant: str, files: tuple[str, ...], total: int, dest: Path) -> None:
    """The friendly "here's what I'm about to download" summary."""
    first = _basename(files[0])
    shown = first if len(files) == 1 else f"{first} + {len(files) - 1} more part{'s' if len(files) > 2 else ''}"
    page = f"https://huggingface.co/{entry.hf_repo}"
    license_text = entry.license or "unknown"
    if entry.license_url and entry.license_url.rstrip("/") != page:
        license_text += f" (details: {entry.license_url})"
    ui.info(f"Getting [bold]{escape(entry.display_name)}[/bold] ({escape(quant)}) from Hugging Face:")
    rows = [("File" if len(files) == 1 else "Files", shown), ("Size", _fmt_bytes(total) if total else "unknown"),
            ("License", license_text), ("Model page", page), ("Saving to", str(dest))]
    for label, value in rows:
        ui.say(f"    [bold]{label + ':':<12}[/bold] {escape(value)}")
    ui.say("    The model's weights come straight from Hugging Face, shared by their authors under the license "
           "above - they aren't part of this game.", style="dim")


def _download_one(ui: UI, downloader: Callable[..., Any], repo_id: str, name: str, dest: Path,
                  size: Optional[int], label: str) -> Path:
    """Fetch one file with a progress bar; verify it landed complete."""
    with ui.download_progress(label, size or None) as advance:
        kwargs: dict[str, Any] = {"local_dir": str(dest)}
        if _accepts(downloader, "tqdm_class"):
            kwargs["tqdm_class"] = _progress_bridge(advance)
        try:
            result = downloader(repo_id=repo_id, filename=name, **kwargs)
        except Exception as exc:
            raise _friendly_error(exc, repo_id, name) from exc
    path = Path(result) if isinstance(result, (str, Path)) and Path(result).is_file() else _local(dest, name)
    if not path.is_file():
        raise DownloadError(f"The download of {_basename(name)} finished, but I can't find the file. Please try again.",
                            "incomplete")
    if size and path.stat().st_size != size:
        raise DownloadError(
            f"The download of {_basename(name)} looks incomplete ({_fmt_bytes(path.stat().st_size)} of "
            f"{_fmt_bytes(size)}). Please try again.",
            "incomplete",
        )
    return path


def downloaded_quants(repo_id: str, dest_dir: Optional[Path] = None) -> set[str]:
    """Which quantizations of `repo_id` are completely downloaded already ({"Q4_K_M", ...}).

    Looks only at the model's own folder (no network); split models count
    only when every part is there. The fit engine uses this so a model you
    already have never needs disk space - or a new download - again.
    """
    dest = Path(dest_dir) if dest_dir is not None else model_folder(repo_id)
    try:
        names = [p.relative_to(dest).as_posix() for p in dest.rglob("*.gguf") if ".cache" not in p.parts]
    except OSError:
        return set()
    found: set[str] = set()
    for quant, (files, _total) in group_quant_files([(name, 0) for name in names]).items():
        if all(_is_complete(_local(dest, f), None) for f in files):
            found.add(quant.upper())
    return found


def find_local_copy(repo_id: str, quant: str, dest_dir: Optional[Path] = None) -> Optional[Path]:
    """A finished download of `repo_id` at `quant` (its first shard), or None. Never uses the network."""
    return _local_fallback(Path(dest_dir) if dest_dir is not None else model_folder(repo_id), quant)


def _local_fallback(dest: Path, wanted: str) -> Optional[Path]:
    """Offline? Look for an already-downloaded copy of this quant in `dest`."""
    try:
        names = [p.relative_to(dest).as_posix() for p in dest.rglob("*.gguf") if ".cache" not in p.parts]
    except OSError:
        return None
    first = pick_gguf_file(names, wanted) if names else None
    if first is None or (wanted and (parse_quant(first) or "").upper() != wanted.upper()):
        return None
    try:
        files = _shard_set(first, names, str(dest))
    except DownloadError:
        return None
    return _local(dest, files[0]) if all(_is_complete(_local(dest, f), None) for f in files) else None


def _download(entry: ModelEntry, ui: UI, dest: Path, wanted: str, api: Any,
              downloader: Optional[Callable[..., Any]], listing: Optional[list[tuple[str, int]]] = None) -> Path:
    repo = entry.hf_repo

    # 1. Already downloaded? Then we don't even need the internet.
    hint = tuple(entry.gguf_files) if _hint_matches(entry, wanted) else ()
    if hint and all(_is_complete(_local(dest, f), None) for f in hint):
        ui.success(f"{escape(entry.display_name)} is already downloaded - no need to fetch it again.")
        return _local(dest, hint[0])
    if not hint and wanted:
        # No exact file names on record (e.g. the fit engine picked another quant):
        # a finished copy of that quant in the model's folder is just as good.
        # (hf_hub_download only moves a file into place once it's complete.)
        found = _local_fallback(dest, wanted)
        if found is not None:
            ui.success(f"{escape(entry.display_name)} ({escape(wanted)}) is already downloaded - no need to fetch it again.")
            return found

    # 2. Ask the Hub for exact file names and sizes (one request).
    if listing is None:
        try:
            with ui.status(f"Asking Hugging Face for the file list of {escape(repo)}..."):
                listing = _list_repo(api, repo)
        except DownloadError as err:
            found = _local_fallback(dest, wanted) if err.kind in ("network", "offline_mode", "server") else None
            if found is None:
                raise
            ui.warn(f"I couldn't reach Hugging Face, but {escape(found.name)} is already downloaded - using it.")
            return found
    sizes = {name: size for name, size in listing}
    names = [name for name, _ in listing]
    files = hint if hint and all(f in sizes for f in hint) else _choose_files(names, wanted, repo)
    got = parse_quant(files[0]) or wanted or "GGUF"
    if wanted and got.upper() != wanted.upper():
        ui.info(f"This repo doesn't offer {escape(wanted)}, so I'll use {escape(got)} instead.")

    # 3. Skip whatever is already complete.
    total = sum(sizes.get(f) or 0 for f in files)
    todo = [f for f in files if not _is_complete(_local(dest, f), sizes.get(f))]
    if not todo:
        ui.success(f"{escape(entry.display_name)} ({escape(got)}) is already downloaded - no need to fetch it again.")
        return _local(dest, files[0])

    # 4. Say what's about to happen, and make sure it fits on the disk.
    _show_plan(ui, entry, got, files, total, dest)
    estimate_each = int(max(entry.file_size_gb, 0.0) * 1e9 / len(files))  # for files of unknown size
    _check_disk(dest, sum(sizes.get(f) or estimate_each for f in todo))

    # 5. Download, one file (shard) at a time.
    downloader = downloader or _default_downloader()
    try:
        dest.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise DownloadError(f"I couldn't create the folder {dest} ({exc.strerror or exc}).", "disk") from exc
    first_path = _local(dest, files[0])
    for i, name in enumerate(todo, 1):
        # The part number goes first so it survives when a narrow terminal shortens the name.
        label = (f"Downloading part {i}/{len(todo)}: " if len(todo) > 1 else "Downloading ") + _basename(name)
        path = _download_one(ui, downloader, repo, name, dest, sizes.get(name) or None, escape(label))
        if name == files[0]:
            first_path = path
    ui.success(f"Downloaded {escape(entry.display_name)} ({_fmt_bytes(total) if total else escape(got)}).")
    return first_path


def download_gguf(entry: ModelEntry, ui: UI, dest_dir: Optional[Path] = None, *, quant: Optional[str] = None,
                  hf_api: Any = None, hf_download: Any = None) -> Path:
    """Download `entry`'s GGUF file(s) for `quant` (default `entry.quant`); return the path
    of the file to load (the first shard for split models).

    Files go to `dest_dir` (default `model_folder(entry.hf_repo)`); all shards
    land in the same folder. Complete files are skipped. `hf_api` /
    `hf_download` default to `HfApi()` / `hf_hub_download` (tests inject fakes).
    Raises DownloadError with a friendly message.
    """
    dest = Path(dest_dir) if dest_dir else model_folder(entry.hf_repo)
    wanted = (quant or entry.quant or "").strip()
    return _download(entry, ui, dest, wanted, hf_api or _default_api(), hf_download)


def normalize_repo_id(text: str) -> tuple[str, Optional[str]]:
    """Accept whatever a player pastes; return (repo_id, quant or None), or ("", None).

    "unsloth/Qwen3-4B-GGUF", "https://huggingface.co/unsloth/Qwen3-4B-GGUF",
    "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M" (Ollama style) and links to a single
    .gguf file all work.
    """
    value = (text or "").strip().strip("\"'")
    value = re.sub(r"^(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/", "", value, flags=re.IGNORECASE)
    value, _, quant = value.partition(":")
    parts = [p for p in value.split("?")[0].split("/") if p]
    if len(parts) < 2:
        return "", None
    repo = f"{parts[0]}/{parts[1]}"
    if not re.fullmatch(r"[A-Za-z0-9][\w.-]*/[\w.-]+", repo):
        return "", None
    if not quant and parts[-1].lower().endswith(".gguf"):
        quant = parse_quant(parts[-1]) or ""
    return repo, (quant.strip() or None)


def _inspect_custom(repo_id: str, quant: Optional[str], api: Any) -> tuple[ModelEntry, Any, list[tuple[str, int]], str]:
    """Look a player-named repo up: (entry, metadata or None, file listing, wanted quant)."""
    repo, quant_in_ref = normalize_repo_id(repo_id)
    if not repo:
        raise DownloadError(
            f"'{repo_id}' doesn't look like a Hugging Face model id. It should look like owner/name, "
            "for example unsloth/Qwen3-4B-GGUF.",
            "bad_repo_id",
        )
    wanted = (quant or quant_in_ref or "").strip()
    listing = _list_repo(api, repo)  # also tells us early if the repo doesn't exist
    try:
        info: Any = with_deadline(lambda: api.model_info(repo), LISTING_DEADLINE_S / 2)
    except Exception:
        info = None  # metadata is nice to have, not essential
    names = [name for name, _ in listing]
    if pick_gguf_file(names, wanted) is None:
        _choose_files(names, wanted, repo)  # raises the friendly "no GGUF files" error right away
    entry = entry_from_hub(info if info is not None else SimpleNamespace(id=repo), listing)
    if entry is None:  # sizes/params unknown: a bare-bones entry is still enough to download
        license_id = (license_of(info) if info is not None else None) or "unknown"
        family = restricted_family(info if info is not None else SimpleNamespace(id=repo))
        if family and catalog.is_permissive(license_id):
            # A Llama/Gemma re-upload tagged "apache-2.0" is still under its family's own terms.
            license_id = f"{family} license (tagged {license_id})"
        entry = ModelEntry(
            key=repo, display_name=prettify_repo_name(repo), family="Custom", params_b=0.0, active_params_b=None,
            license=license_id, license_url=f"https://huggingface.co/{repo}", hf_repo=repo,
            quant=wanted or "GGUF", file_size_gb=0.0, ollama_ref=f"hf.co/{repo}", reasoning=False,
            blurb="A model you picked yourself from Hugging Face.", source="huggingface",
        )
    if wanted and not _hint_matches(entry, wanted):
        # The player asked for a specific version: describe exactly that one.
        files = _choose_files(names, wanted, repo)
        sizes = dict(listing)
        tag = parse_quant(files[0]) or wanted
        entry = dataclasses.replace(entry, quant=tag, gguf_files=files, ollama_ref=f"hf.co/{repo}:{tag}",
                                    file_size_gb=round(sum(sizes.get(f) or 0 for f in files) / 1e9, 2))
    return entry, info, listing, wanted


def custom_entry(repo_id: str, quant: Optional[str] = None, *, hf_api: Any = None) -> ModelEntry:
    """Describe any Hugging Face GGUF repo as a ModelEntry (real sizes, license, params...),
    e.g. so the fit engine can check a "custom" pick before anything is downloaded.

    `repo_id` may be "owner/name", a huggingface.co link, or "hf.co/owner/name:QUANT".
    Raises DownloadError (bad id, not found, gated, no GGUF, network).
    """
    entry, _, _, _ = _inspect_custom(repo_id, quant, hf_api or _default_api())
    return entry


def download_custom_gguf(repo_id: str, quant: str, ui: UI, dest_dir: Optional[Path] = None, *,
                         hf_api: Any = None, hf_download: Any = None) -> Path:
    """Download a GGUF from any Hugging Face repo the player names (for "custom" picks).

    `repo_id` may be "owner/name", a huggingface.co link, or "hf.co/owner/name:QUANT".
    Reads the repo's metadata to show its license (warning if it isn't
    Apache-2.0/MIT or looks unsuitable), then behaves like `download_gguf`.
    """
    api = hf_api or _default_api()
    entry, info, listing, wanted = _inspect_custom(repo_id, quant, api)
    page = f"https://huggingface.co/{entry.hf_repo}"
    if not catalog.is_permissive(entry.license):
        ui.warn(f"Heads-up: this model's license is {escape(entry.license)}, not Apache-2.0 or MIT. "
                f"Please read it on {page} before using the model.")
    reason = rejection_reason(info, allow_all_licenses=True) if info is not None else None
    if reason:
        ui.warn(f"Heads-up: I'd normally leave this one out ({escape(reason)}), so it may not suit the game.")
    elif restricted_family(info if info is not None else SimpleNamespace(id=entry.hf_repo)) and \
            "license (tagged" not in entry.license:
        ui.warn("Heads-up: this model is built on a family with its own license terms (Llama or Gemma), "
                f"whatever its tag says - please read {page} before using it.")
    dest = Path(dest_dir) if dest_dir else model_folder(entry.hf_repo)
    return _download(entry, ui, dest, wanted, api, hf_download, listing=listing)
