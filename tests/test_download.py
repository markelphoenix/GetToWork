"""Tests for gettowork.download: exact GGUF files, shards, disk checks, friendly errors.

No network: a fake Hub API lists repo files and a fake `hf_hub_download`
writes small placeholder files into the destination folder.
"""

from __future__ import annotations

import errno
import io
import inspect
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from rich.console import Console

from gettowork import catalog, download
from gettowork.download import (
    DownloadError,
    custom_entry,
    download_custom_gguf,
    download_gguf,
    model_folder,
    normalize_repo_id,
    pick_gguf_file,
    resolve_files,
)
from gettowork.types import ModelEntry
from gettowork.ui import UI

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def hub_error(name: str, message: str = "boom") -> Exception:
    """Build a real huggingface_hub exception without an HTTP response object."""
    from huggingface_hub import errors

    cls = getattr(errors, name)
    exc = cls.__new__(cls)
    Exception.__init__(exc, message)
    return exc


class RecordingUI(UI):
    """A UI that captures output and remembers every progress bar."""

    def __init__(self) -> None:
        self.buffer = io.StringIO()
        super().__init__(console=Console(file=self.buffer, width=200), input_fn=lambda prompt: "")
        self.bars: list[dict] = []

    @contextmanager
    def download_progress(self, description, total_bytes):
        bar = {"description": description, "total": total_bytes, "advanced": 0}
        self.bars.append(bar)

        def advance(n: int) -> None:
            bar["advanced"] += n

        yield advance

    @property
    def text(self) -> str:
        return self.buffer.getvalue()


class FakeApi:
    """Lists repo files like HfApi.list_repo_tree; optional model_info metadata."""

    def __init__(self, trees: dict[str, list[tuple[str, int]]], *, error: Exception | None = None, info=None):
        self.trees = trees
        self.error = error
        self.info = info
        self.tree_calls: list[str] = []

    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, expand=False, **kwargs):
        self.tree_calls.append(repo_id)
        if self.error:
            raise self.error
        if repo_id not in self.trees:
            raise hub_error("RepositoryNotFoundError", f"404 {repo_id}")
        items = [SimpleNamespace(path=p, size=s, lfs=None) for p, s in self.trees[repo_id]]
        items.append(SimpleNamespace(path="README.md", size=100, lfs=None))
        return iter(items)

    def model_info(self, repo_id, **kwargs):
        if self.info is None:
            raise ConnectionError("no metadata today")
        return self.info


class ExplodingApi:
    def __getattr__(self, name):
        raise AssertionError(f"the Hub should not be contacted (tried {name})")


class FakeDownloader:
    """Stands in for hf_hub_download(repo_id, filename, local_dir=...): writes `size` bytes."""

    def __init__(self, sizes: dict[str, int] | None = None, *, error: Exception | None = None,
                 short_by: int = 0, use_progress: bool = True):
        self.sizes = sizes or {}
        self.error = error
        self.short_by = short_by
        self.use_progress = use_progress
        self.calls: list[tuple[str, str, str]] = []

    def __call__(self, repo_id, filename, *, local_dir=None, tqdm_class=None, **kwargs):
        self.calls.append((repo_id, filename, local_dir))
        if self.error:
            raise self.error
        size = self.sizes.get(filename, 10)
        if tqdm_class is not None and self.use_progress:
            # What huggingface_hub does: create a bar, update it as chunks arrive, call extras.
            with tqdm_class(total=size, initial=0, desc=filename, unit="B", unit_scale=True, position=1) as bar:
                bar.update(size // 2)
                bar.update(size - size // 2)
                bar.set_postfix_str("12 MB/s", refresh=False)
                bar.update_transfer(size)
        path = Path(local_dir, *filename.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * max(size - self.short_by, 0))
        return str(path)


def make_entry(**overrides) -> ModelEntry:
    base = dict(
        key="unsloth/Qwen3-4B-GGUF", display_name="Qwen3 4B", family="Qwen3", params_b=4.0, active_params_b=None,
        license="Apache-2.0", license_url="https://huggingface.co/Qwen/Qwen3-4B", hf_repo="unsloth/Qwen3-4B-GGUF",
        quant="Q4_K_M", file_size_gb=0.0, ollama_ref="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", reasoning=True,
        blurb="A test model.", source="huggingface", gguf_files=("Qwen3-4B-Q4_K_M.gguf",),
    )
    base.update(overrides)
    return ModelEntry(**base)


QWEN_TREE = [
    ("Qwen3-4B-Q4_K_M.gguf", 40),
    ("Qwen3-4B-Q8_0.gguf", 80),
    ("Qwen3-4B-UD-Q4_K_XL.gguf", 44),
    ("mmproj-F16.gguf", 7),
    ("Q8_0/Qwen3-4B-Q8_0-00001-of-00002.gguf", 50),  # an alternative copy in a sub-folder
    ("Q8_0/Qwen3-4B-Q8_0-00002-of-00002.gguf", 30),
]
BIG_REPO = "unsloth/Qwen3-32B-GGUF"
BIG_TREE = [
    ("Qwen3-32B-Q4_K_M.gguf", 60),
    ("BF16/Qwen3-32B-BF16-00001-of-00003.gguf", 100),
    ("BF16/Qwen3-32B-BF16-00002-of-00003.gguf", 100),
    ("BF16/Qwen3-32B-BF16-00003-of-00003.gguf", 20),
]


@pytest.fixture
def plenty_of_disk(monkeypatch):
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(total=10**13, used=0, free=10**12))


# ---------------------------------------------------------------------------
# pick_gguf_file
# ---------------------------------------------------------------------------

FILES = [
    "README.md",
    "mmproj-model-Q4_K_M.gguf",
    "Model-Q8_0.gguf",
    "model-q4_k_m.gguf",
    "sub/Model-Q4_K_M.gguf",
    "Model-UD-Q4_K_XL.gguf",
    "Model-IQ4_XS.gguf",
    "Q6_K/Model-Q6_K-00001-of-00002.gguf",
    "Q6_K/Model-Q6_K-00002-of-00002.gguf",
]


def test_pick_exact_quant_case_insensitive_and_root_first():
    assert pick_gguf_file(FILES, "Q4_K_M") == "model-q4_k_m.gguf"
    assert pick_gguf_file(FILES, "q8_0") == "Model-Q8_0.gguf"


def test_pick_skips_mmproj_and_non_gguf():
    assert pick_gguf_file(["README.md", "mmproj-F16.gguf", "config.json"], "F16") is None
    assert pick_gguf_file([], "Q4_K_M") is None


def test_pick_split_model_returns_first_shard():
    assert pick_gguf_file(FILES, "Q6_K") == "Q6_K/Model-Q6_K-00001-of-00002.gguf"


def test_pick_unsloth_dynamic_quants():
    assert pick_gguf_file(FILES, "UD-Q4_K_XL") == "Model-UD-Q4_K_XL.gguf"
    assert pick_gguf_file(FILES, "Q4_K_XL") == "Model-UD-Q4_K_XL.gguf"  # the UD- prefix is optional
    assert pick_gguf_file(["m-Q4_K_XL.gguf"], "UD-Q4_K_XL") == "m-Q4_K_XL.gguf"


@pytest.mark.parametrize(
    "files, expected",
    [
        (["m-Q8_0.gguf", "m-Q4_K_S.gguf", "m-Q5_K_M.gguf"], "m-Q4_K_S.gguf"),
        (["m-Q8_0.gguf", "m-Q5_K_M.gguf"], "m-Q5_K_M.gguf"),
        (["m-Q8_0.gguf", "m-Q6_K.gguf", "m-IQ4_XS.gguf"], "m-IQ4_XS.gguf"),
        (["gpt-oss-20b-mxfp4.gguf"], "gpt-oss-20b-mxfp4.gguf"),
        (["m-Q2_K.gguf", "m-Q3_K_M.gguf", "m-BF16.gguf"], "m-Q3_K_M.gguf"),  # anything: nearest ~4.8 bits
        (["weird-name.gguf"], "weird-name.gguf"),
    ],
)
def test_pick_fallback_order(files, expected):
    assert pick_gguf_file(files, "Q5_K_S") == expected  # Q5_K_S isn't offered in any of these


# ---------------------------------------------------------------------------
# resolve_files
# ---------------------------------------------------------------------------


def test_resolve_uses_recorded_files_without_network():
    assert resolve_files(make_entry(), hf_api=ExplodingApi()) == ("Qwen3-4B-Q4_K_M.gguf",)
    assert resolve_files(make_entry(), "Q4_K_M", hf_api=ExplodingApi()) == ("Qwen3-4B-Q4_K_M.gguf",)


def test_resolve_lists_the_repo_for_another_quant():
    api = FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE})
    assert resolve_files(make_entry(), "Q8_0", hf_api=api) == ("Qwen3-4B-Q8_0.gguf",)
    assert api.tree_calls == ["unsloth/Qwen3-4B-GGUF"]


def test_resolve_includes_every_shard():
    entry = make_entry(hf_repo=BIG_REPO, key=BIG_REPO, gguf_files=())
    assert resolve_files(entry, "BF16", hf_api=FakeApi({BIG_REPO: BIG_TREE})) == (
        "BF16/Qwen3-32B-BF16-00001-of-00003.gguf",
        "BF16/Qwen3-32B-BF16-00002-of-00003.gguf",
        "BF16/Qwen3-32B-BF16-00003-of-00003.gguf",
    )


def test_resolve_missing_shard_and_no_gguf():
    entry = make_entry(hf_repo=BIG_REPO, key=BIG_REPO, gguf_files=())
    broken = FakeApi({BIG_REPO: [f for f in BIG_TREE if "00002" not in f[0]]})
    with pytest.raises(DownloadError) as info:
        resolve_files(entry, "BF16", hf_api=broken)
    assert info.value.kind == "missing_file" and "Part 2 of 3" in str(info.value)

    empty = FakeApi({BIG_REPO: [("mmproj-F16.gguf", 5)]})
    with pytest.raises(DownloadError) as info:
        resolve_files(entry, "Q4_K_M", hf_api=empty)
    assert info.value.kind == "no_gguf" and "GGUF" in str(info.value)


# ---------------------------------------------------------------------------
# download_gguf
# ---------------------------------------------------------------------------


def test_downloads_the_exact_file_with_a_friendly_summary(tmp_path, plenty_of_disk):
    ui = RecordingUI()
    api = FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE})
    fetch = FakeDownloader({"Qwen3-4B-Q4_K_M.gguf": 40})
    path = download_gguf(make_entry(), ui, tmp_path, hf_api=api, hf_download=fetch)

    assert path == tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    assert path.stat().st_size == 40
    assert fetch.calls == [("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf", str(tmp_path))]
    text = ui.text
    for expected in ("Qwen3 4B", "Q4_K_M", "Qwen3-4B-Q4_K_M.gguf", "Apache-2.0",
                     "https://huggingface.co/unsloth/Qwen3-4B-GGUF", str(tmp_path), "straight from Hugging Face"):
        assert expected in text, expected
    # Progress flowed from huggingface_hub's tqdm calls into our bar.
    assert ui.bars == [{"description": "Downloading Qwen3-4B-Q4_K_M.gguf", "total": 40, "advanced": 40}]


def test_default_destination_is_the_models_folder(tmp_path, monkeypatch, plenty_of_disk):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    monkeypatch.delenv("GETTOWORK_MODELS_DIR", raising=False)
    assert model_folder("unsloth/Qwen3-4B-GGUF") == tmp_path / "models" / "unsloth--Qwen3-4B-GGUF"
    path = download_gguf(make_entry(), RecordingUI(), hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}),
                         hf_download=FakeDownloader({"Qwen3-4B-Q4_K_M.gguf": 40}))
    assert path == tmp_path / "models" / "unsloth--Qwen3-4B-GGUF" / "Qwen3-4B-Q4_K_M.gguf"


def test_downloads_all_shards_into_one_folder(tmp_path, plenty_of_disk):
    ui = RecordingUI()
    entry = make_entry(hf_repo=BIG_REPO, key=BIG_REPO, gguf_files=(), display_name="Qwen3 32B")
    fetch = FakeDownloader(dict(BIG_TREE))
    path = download_gguf(entry, ui, tmp_path, quant="BF16", hf_api=FakeApi({BIG_REPO: BIG_TREE}), hf_download=fetch)

    assert path == tmp_path / "BF16" / "Qwen3-32B-BF16-00001-of-00003.gguf"
    assert [c[1] for c in fetch.calls] == [f for f, _ in BIG_TREE if "BF16" in f]
    assert sorted(p.name for p in (tmp_path / "BF16").iterdir()) == [
        "Qwen3-32B-BF16-00001-of-00003.gguf", "Qwen3-32B-BF16-00002-of-00003.gguf", "Qwen3-32B-BF16-00003-of-00003.gguf"
    ]
    assert [b["description"] for b in ui.bars][0].startswith("Downloading part 1/3: ")
    assert "+ 2 more parts" in ui.text and "220 bytes" in ui.text  # total of all three shards


def test_skips_complete_files_and_redownloads_partial_ones(tmp_path, plenty_of_disk):
    folder = tmp_path / "BF16"
    folder.mkdir()
    (folder / "Qwen3-32B-BF16-00001-of-00003.gguf").write_bytes(b"\0" * 100)  # complete
    (folder / "Qwen3-32B-BF16-00002-of-00003.gguf").write_bytes(b"\0" * 7)  # wrong size
    entry = make_entry(hf_repo=BIG_REPO, key=BIG_REPO, gguf_files=())
    fetch = FakeDownloader(dict(BIG_TREE))
    download_gguf(entry, RecordingUI(), tmp_path, quant="BF16", hf_api=FakeApi({BIG_REPO: BIG_TREE}), hf_download=fetch)
    assert [c[1] for c in fetch.calls] == ["BF16/Qwen3-32B-BF16-00002-of-00003.gguf",
                                          "BF16/Qwen3-32B-BF16-00003-of-00003.gguf"]


def test_everything_already_downloaded(tmp_path):
    (tmp_path / "Qwen3-4B-Q8_0.gguf").write_bytes(b"\0" * 80)
    ui = RecordingUI()
    fetch = FakeDownloader()
    path = download_gguf(make_entry(), ui, tmp_path, quant="Q8_0",
                         hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}), hf_download=fetch)
    assert path == tmp_path / "Qwen3-4B-Q8_0.gguf"
    assert fetch.calls == []
    assert "already downloaded" in ui.text


def test_recorded_files_on_disk_need_no_internet(tmp_path):
    (tmp_path / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"\0" * 40)
    ui = RecordingUI()
    path = download_gguf(make_entry(), ui, tmp_path, hf_api=ExplodingApi(), hf_download=FakeDownloader())
    assert path == tmp_path / "Qwen3-4B-Q4_K_M.gguf"
    assert "already downloaded" in ui.text


def test_offline_falls_back_to_a_previous_download(tmp_path):
    (tmp_path / "Q8_0").mkdir()
    (tmp_path / "Q8_0" / "Qwen3-4B-Q8_0-00001-of-00002.gguf").write_bytes(b"1")
    (tmp_path / "Q8_0" / "Qwen3-4B-Q8_0-00002-of-00002.gguf").write_bytes(b"2")
    ui = RecordingUI()
    offline = FakeApi({}, error=ConnectionError("Failed to establish a new connection"))
    path = download_gguf(make_entry(gguf_files=()), ui, tmp_path, quant="Q8_0", hf_api=offline,
                         hf_download=FakeDownloader())
    assert path == tmp_path / "Q8_0" / "Qwen3-4B-Q8_0-00001-of-00002.gguf"
    assert "already downloaded" in ui.text


def test_wrongly_guessed_recorded_file_is_replaced_by_the_real_one(tmp_path, plenty_of_disk):
    entry = make_entry(gguf_files=("Qwen3-4B-q4_k_m-GUESS.gguf",))
    fetch = FakeDownloader({"Qwen3-4B-Q4_K_M.gguf": 40})
    path = download_gguf(entry, RecordingUI(), tmp_path, hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}),
                         hf_download=fetch)
    assert path.name == "Qwen3-4B-Q4_K_M.gguf"


def test_missing_quant_falls_back_and_says_so(tmp_path, plenty_of_disk):
    ui = RecordingUI()
    path = download_gguf(make_entry(gguf_files=()), ui, tmp_path, quant="Q5_K_M",
                         hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}), hf_download=FakeDownloader(dict(QWEN_TREE)))
    assert path.name == "Qwen3-4B-Q4_K_M.gguf"
    assert "doesn't offer Q5_K_M" in ui.text and "use Q4_K_M instead" in ui.text


def test_not_enough_disk_space(tmp_path, monkeypatch):
    monkeypatch.setattr(download.shutil, "disk_usage", lambda path: SimpleNamespace(total=10**9, used=0, free=10**6))
    fetch = FakeDownloader()
    tree = {"unsloth/Qwen3-4B-GGUF": [("Qwen3-4B-Q4_K_M.gguf", 2_500_000_000)]}
    with pytest.raises(DownloadError) as info:
        download_gguf(make_entry(), RecordingUI(), tmp_path / "new" / "folder", hf_api=FakeApi(tree), hf_download=fetch)
    assert info.value.kind == "disk"
    assert "2.5 GB" in str(info.value) and "free" in str(info.value)
    assert fetch.calls == []
    assert not (tmp_path / "new").exists()


def test_disk_check_counts_only_missing_files(tmp_path, monkeypatch):
    (tmp_path / "BF16").mkdir()
    for name in ("Qwen3-32B-BF16-00001-of-00003.gguf", "Qwen3-32B-BF16-00002-of-00003.gguf"):
        (tmp_path / "BF16" / name).write_bytes(b"\0" * 100)
    seen = {}

    def usage(path):
        seen["path"] = path
        return SimpleNamespace(total=10**12, used=0, free=download.DISK_HEADROOM_BYTES + 25)

    monkeypatch.setattr(download.shutil, "disk_usage", usage)
    entry = make_entry(hf_repo=BIG_REPO, key=BIG_REPO, gguf_files=())
    download_gguf(entry, RecordingUI(), tmp_path, quant="BF16", hf_api=FakeApi({BIG_REPO: BIG_TREE}),
                  hf_download=FakeDownloader(dict(BIG_TREE)))  # only the 20-byte part is missing: fits
    assert seen["path"] == tmp_path


@pytest.mark.parametrize(
    "error, kind, fragments",
    [
        (hub_error("GatedRepoError", "403 restricted"), "gated", ("gated", "hf auth login", "accept")),
        (hub_error("RepositoryNotFoundError", "404"), "not_found", ("couldn't find", "unsloth/Qwen3-4B-GGUF")),
        (hub_error("RemoteEntryNotFoundError", "404 entry"), "missing_file", ("isn't in",)),
        (hub_error("LocalEntryNotFoundError", "cannot find"), "network", ("internet connection",)),
        (hub_error("OfflineModeIsEnabled", "offline"), "offline_mode", ("HF_HUB_OFFLINE",)),
        (OSError(errno.ENOSPC, "No space left on device"), "disk", ("disk filled up",)),
        (type("HTTPError", (Exception,), {"response": SimpleNamespace(status_code=503)})("x"), "server", ("HTTP 503",)),
        (type("HTTPError", (Exception,), {"response": SimpleNamespace(status_code=429)})("x"), "server", ("slow down",)),
        (ConnectionError("[Errno 111] Connection refused"), "network", ("internet connection", "Connection refused")),
    ],
)
def test_download_errors_are_friendly(tmp_path, plenty_of_disk, error, kind, fragments):
    api = FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE})
    with pytest.raises(DownloadError) as info:
        download_gguf(make_entry(), RecordingUI(), tmp_path, hf_api=api, hf_download=FakeDownloader(error=error))
    assert info.value.kind == kind
    for fragment in fragments:
        assert fragment in str(info.value)


@pytest.mark.parametrize(
    "error, kind",
    [
        (hub_error("GatedRepoError", "403"), "gated"),
        (hub_error("RepositoryNotFoundError", "404"), "not_found"),
        (ConnectionError("Failed to establish a new connection"), "network"),
    ],
)
def test_listing_errors_are_friendly(tmp_path, error, kind):
    with pytest.raises(DownloadError) as info:
        download_gguf(make_entry(gguf_files=()), RecordingUI(), tmp_path, hf_api=FakeApi({}, error=error),
                      hf_download=FakeDownloader())
    assert info.value.kind == kind


def test_gated_message_mentions_the_model_page(tmp_path):
    with pytest.raises(DownloadError) as info:
        download_gguf(make_entry(gguf_files=()), RecordingUI(), tmp_path,
                      hf_api=FakeApi({}, error=hub_error("GatedRepoError")), hf_download=FakeDownloader())
    assert "https://huggingface.co/unsloth/Qwen3-4B-GGUF" in str(info.value)


def test_incomplete_download_is_detected(tmp_path, plenty_of_disk):
    with pytest.raises(DownloadError) as info:
        download_gguf(make_entry(), RecordingUI(), tmp_path, hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}),
                      hf_download=FakeDownloader({"Qwen3-4B-Q4_K_M.gguf": 40}, short_by=5))
    assert info.value.kind == "incomplete"


def test_downloader_without_tqdm_support_still_works(tmp_path, plenty_of_disk):
    calls = []

    def simple_download(repo_id, filename, local_dir=None):
        calls.append(filename)
        Path(local_dir, filename).write_bytes(b"\0" * 40)
        return str(Path(local_dir, filename))

    path = download_gguf(make_entry(), RecordingUI(), tmp_path, hf_api=FakeApi({"unsloth/Qwen3-4B-GGUF": QWEN_TREE}),
                         hf_download=simple_download)
    assert calls == ["Qwen3-4B-Q4_K_M.gguf"] and path.is_file()


def test_real_hf_hub_download_accepts_our_arguments():
    from huggingface_hub import hf_hub_download

    params = inspect.signature(hf_hub_download).parameters
    assert {"repo_id", "filename", "local_dir"} <= set(params)
    assert download._accepts(hf_hub_download, "local_dir")


def test_progress_bridge_is_forgiving():
    seen = []
    bridge = download._progress_bridge(seen.append)(total=100, initial=10, desc="x", leave=True, bar_format="{l_bar}")
    bridge.update(30)
    bridge.update(-10)  # huggingface_hub rewinds on a retry
    bridge.refresh()
    bridge.reset(total=5)
    bridge.close()
    assert seen == [10, 30, -10] and bridge.n == 30


# ---------------------------------------------------------------------------
# Custom repos
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, expected",
    [
        ("unsloth/Qwen3-4B-GGUF", ("unsloth/Qwen3-4B-GGUF", None)),
        ("  https://huggingface.co/unsloth/Qwen3-4B-GGUF  ", ("unsloth/Qwen3-4B-GGUF", None)),
        ("hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", ("unsloth/Qwen3-4B-GGUF", "Q4_K_M")),
        ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/blob/main/Qwen3-4B-Q8_0.gguf", ("unsloth/Qwen3-4B-GGUF", "Q8_0")),
        ("https://huggingface.co/unsloth/Qwen3-4B-GGUF/tree/main", ("unsloth/Qwen3-4B-GGUF", None)),
        ('"bartowski/Qwen_Qwen3-4B-GGUF"', ("bartowski/Qwen_Qwen3-4B-GGUF", None)),
        ("qwen3", ("", None)),
        ("", ("", None)),
        ("bad name/with spaces", ("", None)),
    ],
)
def test_normalize_repo_id(text, expected):
    assert normalize_repo_id(text) == expected


def test_custom_download_reads_metadata_and_warns_about_licenses(tmp_path, plenty_of_disk):
    repo = "bartowski/Llama-3.2-3B-Instruct-GGUF"
    info = SimpleNamespace(id=repo, tags=["gguf", "conversational", "license:llama3.2"], downloads=5000,
                           gguf={"total": 3_212_749_888, "architecture": "llama"}, card_data=None, gated=False,
                           pipeline_tag="text-generation")
    api = FakeApi({repo: [("Llama-3.2-3B-Instruct-Q4_K_M.gguf", 20), ("Llama-3.2-3B-Instruct-Q8_0.gguf", 34)]}, info=info)
    ui = RecordingUI()
    path = download_custom_gguf(f"https://huggingface.co/{repo}", "", ui, tmp_path, hf_api=api,
                                hf_download=FakeDownloader({"Llama-3.2-3B-Instruct-Q4_K_M.gguf": 20}))
    assert path == tmp_path / "Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    assert "llama3.2" in ui.text and "Heads-up" in ui.text
    assert api.tree_calls == [repo]  # listed once, reused for the download


def test_custom_download_quant_from_reference_and_missing_metadata(tmp_path, plenty_of_disk):
    repo = "someone/My-Model-7B-Instruct-GGUF"
    api = FakeApi({repo: [("m-Q4_K_M.gguf", 20), ("m-Q8_0.gguf", 34)]}, info=None)
    ui = RecordingUI()
    path = download_custom_gguf(f"hf.co/{repo}:Q8_0", None, ui, tmp_path, hf_api=api,
                                hf_download=FakeDownloader({"m-Q8_0.gguf": 34}))
    assert path.name == "m-Q8_0.gguf"
    assert "unknown" in ui.text  # license couldn't be read: shown as unknown, with a heads-up


def test_custom_download_warns_about_unsuitable_models(tmp_path, plenty_of_disk):
    repo = "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"
    info = SimpleNamespace(id=repo, tags=["gguf", "conversational", "license:apache-2.0"], downloads=5000,
                           gguf={"total": 7_615_616_512}, card_data=None, gated=False, pipeline_tag="text-generation")
    ui = RecordingUI()
    download_custom_gguf(repo, "Q4_K_M", ui, tmp_path, hf_api=FakeApi({repo: [("c-Q4_K_M.gguf", 10)]}, info=info),
                         hf_download=FakeDownloader())
    assert "not a storyteller" in ui.text


def test_custom_download_bad_ids_and_missing_repos(tmp_path):
    with pytest.raises(DownloadError) as info:
        download_custom_gguf("just-a-name", "Q4_K_M", RecordingUI(), tmp_path, hf_api=ExplodingApi())
    assert info.value.kind == "bad_repo_id" and "owner/name" in str(info.value)
    with pytest.raises(DownloadError) as info:
        download_custom_gguf("nobody/Nothing-GGUF", "Q4_K_M", RecordingUI(), tmp_path, hf_api=FakeApi({}))
    assert info.value.kind == "not_found"


def test_curated_seed_downloads_its_guessed_file_via_listing(tmp_path, plenty_of_disk):
    seed = catalog.get_model("qwen3-4b")
    tree = {seed.hf_repo: [("Qwen3-4B-Q4_K_M.gguf", 25), ("Qwen3-4B-Q8_0.gguf", 43)]}
    fetch = FakeDownloader({"Qwen3-4B-Q4_K_M.gguf": 25, "Qwen3-4B-Q8_0.gguf": 43})
    assert download_gguf(seed, RecordingUI(), tmp_path, hf_api=FakeApi(tree), hf_download=fetch).name == "Qwen3-4B-Q4_K_M.gguf"
    assert download_gguf(seed, RecordingUI(), tmp_path, quant="Q8_0", hf_api=FakeApi(tree), hf_download=fetch).name == "Qwen3-4B-Q8_0.gguf"


def test_custom_entry_describes_a_repo_before_downloading():
    repo = "unsloth/Qwen3-8B-GGUF"
    info = SimpleNamespace(id=repo, tags=["gguf", "conversational", "license:apache-2.0"], downloads=5000,
                           gguf={"total": 8_190_735_360, "architecture": "qwen3", "context_length": 40960},
                           card_data=None, gated=False, pipeline_tag="text-generation")
    tree = {repo: [("Qwen3-8B-Q4_K_M.gguf", 5_027_783_488), ("Qwen3-8B-Q8_0.gguf", 8_709_519_040)]}
    entry = custom_entry(f"https://huggingface.co/{repo}", hf_api=FakeApi(tree, info=info))
    assert entry.hf_repo == repo and entry.quant == "Q4_K_M" and entry.params_b == pytest.approx(8.19)
    assert dict(entry.quant_options)["Q8_0"] == pytest.approx(8.71)
    # Asking for a version describes exactly that one (files, size, Ollama tag).
    q8 = custom_entry(f"hf.co/{repo}:q8_0", hf_api=FakeApi(tree, info=info))
    assert q8.quant == "Q8_0" and q8.gguf_files == ("Qwen3-8B-Q8_0.gguf",)
    assert q8.file_size_gb == pytest.approx(8.71) and q8.ollama_ref.endswith(":Q8_0")
    # ...and it fits into the fit engine like any discovered model.
    assert catalog.evaluate_fit(_big_machine(), entry).verdict != "no"


def test_custom_entry_without_metadata_or_sizes():
    repo = "someone/Odd-GGUF"
    entry = custom_entry(repo, hf_api=FakeApi({repo: [("odd-Q4_K_M.gguf", 0)]}, info=None))
    assert entry.family == "Custom" and entry.license == "unknown"
    with pytest.raises(DownloadError) as info:
        custom_entry("nobody/None-GGUF", hf_api=FakeApi({}))
    assert info.value.kind == "not_found"


def _big_machine():
    from gettowork.types import SystemSpecs

    return SystemSpecs(os_name="Linux", os_version="", arch="x86_64", cpu_name="Test", cpu_cores_physical=8,
                       cpu_cores_logical=16, ram_total_gb=64, ram_available_gb=32, disk_free_gb=500,
                       ram_bandwidth_gbs=60, cpu_flags=["avx2"])


def test_a_finished_copy_of_the_chosen_quant_is_reused_without_the_hub(tmp_path):
    """The fit engine may pick a quant with no recorded file names: a finished local copy still counts."""
    (tmp_path / "Qwen3-4B-Q8_0.gguf").write_bytes(b"GGUF" + b"\0" * 10)
    ui = RecordingUI()
    entry = make_entry(quant="Q8_0", gguf_files=())
    path = download_gguf(entry, ui, tmp_path, hf_api=ExplodingApi(), hf_download=FakeDownloader())
    assert path == tmp_path / "Qwen3-4B-Q8_0.gguf"
    assert "already downloaded" in ui.text and ui.bars == []
    assert download.find_local_copy("unsloth/Qwen3-4B-GGUF", "Q8_0", tmp_path) == path
    assert download.find_local_copy("unsloth/Qwen3-4B-GGUF", "Q4_K_M", tmp_path) is None


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_a_stalled_file_listing_has_a_deadline(monkeypatch):
    import threading

    release = threading.Event()

    class StalledApi:
        def list_repo_tree(self, repo_id, **kwargs):
            release.wait(10)
            return iter([])

    monkeypatch.setattr(download, "LISTING_DEADLINE_S", 0.2)
    try:
        with pytest.raises(DownloadError) as err:
            custom_entry("unsloth/Qwen3-4B-GGUF", hf_api=StalledApi())
    finally:
        release.set()
    assert err.value.kind == "network" and "didn't answer in time" in str(err.value)


def test_hub_errors_are_found_wherever_this_huggingface_hub_keeps_them(monkeypatch):
    # huggingface_hub 0.23/0.24: an "errors" module exists but holds only a few
    # classes; GatedRepoError & co. still live in "utils".
    import sys
    import types

    class GatedRepoError(Exception):
        pass

    fake_errors = types.ModuleType("huggingface_hub.errors")
    fake_errors.OfflineModeIsEnabled = type("OfflineModeIsEnabled", (Exception,), {})
    fake_utils = types.ModuleType("huggingface_hub.utils")
    fake_utils.GatedRepoError = GatedRepoError
    monkeypatch.setitem(sys.modules, "huggingface_hub.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "huggingface_hub.utils", fake_utils)
    download._hub_errors.cache_clear()
    try:
        err = download._friendly_error(GatedRepoError("401 gated"), "meta-llama/Llama-3.2-1B")
        assert err.kind == "gated"
    finally:
        download._hub_errors.cache_clear()


def test_a_repo_without_gguf_files_is_reported_at_once():
    api = FakeApi({"meta-llama/Llama-3.2-1B-Instruct": [("model.safetensors", 2_000_000_000)]})
    with pytest.raises(DownloadError) as err:
        custom_entry("meta-llama/Llama-3.2-1B-Instruct", hf_api=api)
    assert err.value.kind == "no_gguf"


def test_downloaded_quants_lists_only_complete_downloads(tmp_path):
    folder = tmp_path / "models" / "repo"
    folder.mkdir(parents=True)
    (folder / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"GGUF")
    (folder / "Q8_0").mkdir()
    (folder / "Q8_0" / "Qwen3-4B-Q8_0-00001-of-00002.gguf").write_bytes(b"GGUF")  # part 2 missing
    (folder / "Qwen3-4B-IQ4_XS.gguf").write_bytes(b"")  # empty: not a finished download
    assert download.downloaded_quants("unsloth/Qwen3-4B-GGUF", folder) == {"Q4_K_M"}
    assert download.downloaded_quants("nobody/nothing", tmp_path / "missing") == set()


def test_the_dependency_floor_has_the_progress_hook_the_download_bar_needs():
    """huggingface_hub < 1.1 has no tqdm_class, so the bar would sit at 0% for a multi-GB download."""
    import re

    import huggingface_hub

    pyproject = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    floor = re.search(r'"huggingface_hub>=([0-9.]+)"', pyproject)
    assert floor is not None
    assert tuple(int(x) for x in floor.group(1).split(".")[:2]) >= (1, 1)
    assert download._accepts(huggingface_hub.hf_hub_download, "tqdm_class")


def test_custom_entry_without_sizes_still_sees_through_a_relabelled_llama():
    repo = "someone/Llama-3.2-3B-Instruct-GGUF"
    info = SimpleNamespace(id=repo, tags=["gguf", "license:apache-2.0", "base_model:meta-llama/Llama-3.2-3B-Instruct"],
                           downloads=5000, gguf=None, card_data=None, gated=False, pipeline_tag="text-generation")
    entry = custom_entry(repo, hf_api=FakeApi({repo: [("odd-Q4_K_M.gguf", 0)]}, info=info))
    assert entry.license == "Llama license (tagged Apache-2.0)"
    assert not catalog.is_permissive(entry.license)
