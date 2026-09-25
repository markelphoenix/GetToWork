"""Tests for gettowork.hf_discovery: live Hugging Face search, screening, cache.

No network: a FakeHubApi answers `list_models` / `list_repo_tree` with
realistic ModelInfo-like / RepoFile-like objects built from SimpleNamespace.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import threading
import time
import typing
from types import SimpleNamespace

import pytest

from gettowork import catalog, hf_discovery
from gettowork.hf_discovery import (
    CACHE_SCHEMA_VERSION,
    DiscoveryResult,
    discover_models,
    entry_from_hub,
    group_quant_files,
    license_of,
    params_from_name,
    parse_quant,
    prettify_repo_name,
    rejection_reason,
    repo_gguf_files,
    shard_info,
)
from gettowork.types import GPUInfo, ModelEntry, SystemSpecs

GB = 1_000_000_000

# ---------------------------------------------------------------------------
# Fakes that look like huggingface_hub's ModelInfo / RepoFile / RepoFolder
# ---------------------------------------------------------------------------


def hub_model(
    repo: str,
    *,
    downloads: int = 50_000,
    license: str | None = "apache-2.0",
    total: int | None = None,
    arch: str | None = None,
    context: int | None = None,
    base: str | None = None,
    tags: tuple[str, ...] = (),
    conversational: bool = True,
    gated: object = False,
    pipeline: str | None = "text-generation",
    card: object = None,
) -> SimpleNamespace:
    """A ModelInfo-like search result (what list_models(expand=[...]) returns)."""
    tag_list = ["gguf", "text-generation", "endpoints_compatible", "region:us"]
    if conversational:
        tag_list.append("conversational")
    if license:
        tag_list.append(f"license:{license}")
    if base:
        tag_list += [f"base_model:{base}", f"base_model:quantized:{base}"]
    tag_list += list(tags)
    gguf = None
    if total or arch or context:
        gguf = {"total": total, "architecture": arch, "context_length": context, "chat_template": "{% for m in messages %}"}
    return SimpleNamespace(
        id=repo, author=repo.split("/")[0], downloads=downloads, likes=downloads // 100, tags=tag_list, gguf=gguf,
        gated=gated, pipeline_tag=pipeline, card_data=card, private=False, disabled=False, last_modified=None,
    )


def rfile(path: str, size: int) -> SimpleNamespace:
    """A RepoFile-like tree entry (LFS files also carry lfs.size)."""
    return SimpleNamespace(path=path, size=size, lfs=SimpleNamespace(size=size, sha256="0" * 64))


def rfolder(path: str) -> SimpleNamespace:
    """A RepoFolder-like tree entry: no size."""
    return SimpleNamespace(path=path, tree_id="abc")


MODELS = [
    hub_model("unsloth/Qwen3-4B-GGUF", downloads=900_000, total=4_022_468_096, arch="qwen3", context=40960,
              base="Qwen/Qwen3-4B", tags=("qwen3", "unsloth", "imatrix")),
    hub_model("bartowski/Qwen_Qwen3-4B-GGUF", downloads=400_000, total=4_022_468_096, arch="qwen3",
              base="Qwen/Qwen3-4B"),
    hub_model("unsloth/Qwen3-30B-A3B-GGUF", downloads=700_000, total=30_532_122_624, arch="qwen3moe",
              context=40960, base="Qwen/Qwen3-30B-A3B"),
    hub_model("ggml-org/gpt-oss-20b-GGUF", downloads=300_000, total=20_914_757_184, arch="gpt-oss",
              context=131072, base="openai/gpt-oss-20b"),
    hub_model("unsloth/gpt-oss-20b-GGUF", downloads=800_000, total=20_914_757_184, arch="gpt-oss",
              base="openai/gpt-oss-20b"),
    hub_model("unsloth/Phi-4-mini-instruct-GGUF", downloads=200_000, license="mit", base="microsoft/Phi-4-mini-instruct"),
    hub_model("bartowski/Llama-3.2-3B-Instruct-GGUF", downloads=600_000, license="llama3.2", total=3_212_749_888,
              arch="llama", base="meta-llama/Llama-3.2-3B-Instruct"),
    hub_model("huihui-ai/Qwen3-8B-abliterated-GGUF", downloads=90_000, total=8_190_735_360, arch="qwen3"),
    hub_model("mradermacher/Spicy-Tales-12B-GGUF", downloads=80_000, tags=("not-for-all-audiences",)),
    hub_model("unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", downloads=500_000, total=7_615_616_512, arch="qwen2"),
    hub_model("Qwen/Qwen3-Embedding-0.6B-GGUF", downloads=150_000, total=595_776_512, arch="qwen3",
              pipeline="feature-extraction", conversational=False),
    hub_model("unsloth/Qwen3-4B-Base-GGUF", downloads=40_000, total=4_022_468_096, conversational=False),
    hub_model("someone/Mystery-7B-Instruct-GGUF", downloads=60_000, license=None, total=7_241_732_096),
    hub_model("randomuser/Tiny-Chat-3B-Instruct-GGUF", downloads=50, total=3_000_000_000),
    hub_model("ibm-granite/granite-3.3-8b-instruct-GGUF", downloads=30_000, total=8_170_864_640, arch="granite",
              gated="manual"),
]

TREES = {
    "unsloth/Qwen3-4B-GGUF": [
        rfile("Qwen3-4B-BF16.gguf", 8_051_285_312),
        rfile("Qwen3-4B-Q8_0.gguf", 4_280_405_248),
        rfile("Qwen3-4B-Q4_K_M.gguf", 2_497_280_256),
        rfile("Qwen3-4B-UD-Q4_K_XL.gguf", 2_553_745_664),
        rfile("Qwen3-4B-IQ4_XS.gguf", 2_270_750_976),
        rfile("README.md", 5_000),
        rfolder("BF16"),
    ],
    "bartowski/Qwen_Qwen3-4B-GGUF": [rfile("Qwen_Qwen3-4B-Q4_K_M.gguf", 2_497_280_256)],
    "unsloth/Qwen3-30B-A3B-GGUF": [
        rfile("Qwen3-30B-A3B-Q4_K_M.gguf", 18_556_689_568),
        rfolder("Q8_0"),
        rfile("Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00002.gguf", 16_000_000_000),
        rfile("Q8_0/Qwen3-30B-A3B-Q8_0-00002-of-00002.gguf", 16_451_000_000),
        rfile("mmproj-F16.gguf", 600_000_000),
    ],
    "ggml-org/gpt-oss-20b-GGUF": [rfile("gpt-oss-20b-mxfp4.gguf", 12_109_566_560)],
    "unsloth/gpt-oss-20b-GGUF": [rfile("gpt-oss-20b-Q4_K_M.gguf", 11_600_000_000)],
    "unsloth/Phi-4-mini-instruct-GGUF": [rfile("Phi-4-mini-instruct-Q4_K_M.gguf", 2_491_874_688),
                                         rfile("Phi-4-mini-instruct-Q8_0.gguf", 4_081_004_224)],
    "bartowski/Llama-3.2-3B-Instruct-GGUF": [rfile("Llama-3.2-3B-Instruct-Q4_K_M.gguf", 2_019_377_696)],
    "someone/Mystery-7B-Instruct-GGUF": [rfile("mystery-7b-instruct.Q4_K_M.gguf", 4_368_439_584)],
    "ibm-granite/granite-3.3-8b-instruct-GGUF": [rfile("granite-3.3-8b-instruct-Q4_K_M.gguf", 4_942_859_456)],
}


class FakeHubApi:
    """Answers like HfApi, records every call, and can be told to fail or be slow."""

    def __init__(self, models=MODELS, trees=TREES, *, list_error=None, list_errors_for=(), tree_errors_for=(),
                 on_list=None, on_tree=None):
        self.models = list(models)
        self.trees = dict(trees)
        self.list_error = list_error
        self.list_errors_for = set(list_errors_for)
        self.tree_errors_for = set(tree_errors_for)
        self.on_list = on_list
        self.on_tree = on_tree
        self.list_calls: list[dict] = []
        self.tree_calls: list[str] = []
        self._lock = threading.Lock()

    def list_models(self, **kwargs):
        with self._lock:
            self.list_calls.append(kwargs)
        if self.on_list:
            self.on_list(kwargs)
        if self.list_error:
            raise self.list_error
        author = kwargs.get("author")
        if author in self.list_errors_for:
            raise ConnectionError(f"search for {author} failed")
        # The real API refuses these combinations - make sure we never send them.
        assert not (kwargs.get("expand") and (kwargs.get("full") or kwargs.get("cardData")))
        found = [m for m in self.models if author is None or m.id.split("/")[0] == author]
        found.sort(key=lambda m: -m.downloads)
        return iter(found[: kwargs.get("limit") or None])

    def list_repo_tree(self, repo_id, path_in_repo=None, *, recursive=False, expand=False, **kwargs):
        with self._lock:
            self.tree_calls.append(repo_id)
        if self.on_tree:
            self.on_tree(repo_id)
        if repo_id in self.tree_errors_for:
            raise ConnectionError("tree lookup failed")
        assert recursive is True
        return iter(self.trees.get(repo_id, []))


class ExplodingApi:
    """Any use of this API fails the test: proves a code path stays offline."""

    def __getattr__(self, name):
        raise AssertionError(f"the Hub API should not be used here (tried {name})")


class FakeClock:
    def __init__(self, start: float = 1_750_000_000.0):
        self.now = start
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


def by_repo(result: DiscoveryResult) -> dict[str, ModelEntry]:
    return {m.hf_repo: m for m in result.models}


def live_repos(result: DiscoveryResult) -> set[str]:
    return {m.hf_repo for m in result.models if m.source == "huggingface"}


# ---------------------------------------------------------------------------
# parse_quant / group_quant_files / shard_info
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("Qwen3-4B-Q4_K_M.gguf", "Q4_K_M"),
        ("Qwen3-4B-UD-Q4_K_XL.gguf", "UD-Q4_K_XL"),
        ("gpt-oss-20b-mxfp4.gguf", "MXFP4"),
        ("Qwen3-30B-A3B-MXFP4_MOE.gguf", "MXFP4"),
        ("model-Q8_0-00001-of-00002.gguf", "Q8_0"),
        ("Q4_K_M/xxx-00001-of-00003.gguf", "Q4_K_M"),
        ("UD-Q2_K_XL/Qwen3-235B-A22B-UD-Q2_K_XL-00002-of-00003.gguf", "UD-Q2_K_XL"),
        ("Qwen3-4B-IQ4_XS.gguf", "IQ4_XS"),
        ("x-IQ2_XXS.gguf", "IQ2_XXS"),
        ("x-IQ4_NL.gguf", "IQ4_NL"),
        ("Qwen3-4B-Q4_0.gguf", "Q4_0"),
        ("Qwen3-4B-Q5_K_S.gguf", "Q5_K_S"),
        ("Qwen3-4B-BF16.gguf", "BF16"),
        ("Qwen3-4B-F16.gguf", "F16"),
        ("Phi-3-mini-4k-instruct-fp16.gguf", "F16"),
        ("qwen2.5-7b-instruct-q4_k_m.gguf", "Q4_K_M"),
        ("model.i1-Q4_K_M.gguf", "Q4_K_M"),
        ("BF16/x-BF16-00001-of-00002.gguf", "BF16"),
        ("sub\\dir\\Model-Q6_K.gguf", "Q6_K"),
        ("mmproj-F16.gguf", None),
        ("mmproj-Qwen3-4B-Q8_0.gguf", None),
        ("old-arm-repack-Q4_0_4_4.gguf", None),
        ("README.md", None),
        ("Qwen3-4B-Q4_K_M.safetensors", None),
        ("model.gguf", None),
    ],
)
def test_parse_quant_real_world_names(filename, expected):
    assert parse_quant(filename) == expected


def test_shard_info():
    assert shard_info("Q8_0/m-Q8_0-00002-of-00003.gguf") == ("Q8_0/m-Q8_0", 2, 3)
    assert shard_info("m-Q4_K_M.gguf") == ("m-Q4_K_M", 1, 1)


def test_group_quant_files_shards_totals_and_order():
    groups = group_quant_files([
        ("m-Q4_K_M.gguf", 100),
        ("Q8_0/m-Q8_0-00002-of-00002.gguf", 60),
        ("Q8_0/m-Q8_0-00001-of-00002.gguf", 90),
        ("m-UD-Q4_K_XL.gguf", 110),
        ("mmproj-F16.gguf", 5),
        ("README.md", 1),
    ])
    assert groups["Q8_0"] == (("Q8_0/m-Q8_0-00001-of-00002.gguf", "Q8_0/m-Q8_0-00002-of-00002.gguf"), 150)
    assert groups["Q4_K_M"] == (("m-Q4_K_M.gguf",), 100)
    assert groups["UD-Q4_K_XL"] == (("m-UD-Q4_K_XL.gguf",), 110)  # its own tag, not merged with Q4_K_M
    assert list(groups) == ["Q8_0", "UD-Q4_K_XL", "Q4_K_M"]  # biggest first
    assert all("mmproj" not in f for files, _ in groups.values() for f in files)


def test_group_quant_files_drops_incomplete_shard_sets():
    groups = group_quant_files([
        ("m-Q8_0-00001-of-00003.gguf", 10),
        ("m-Q8_0-00003-of-00003.gguf", 10),  # part 2 missing
        ("m-Q4_K_M.gguf", 5),
    ])
    assert "Q8_0" not in groups
    assert "Q4_K_M" in groups


def test_group_quant_files_prefers_root_copy():
    groups = group_quant_files([("old/m-Q4_K_M.gguf", 7), ("m-Q4_K_M.gguf", 5)])
    assert groups["Q4_K_M"] == (("m-Q4_K_M.gguf",), 5)


# ---------------------------------------------------------------------------
# Names, params, licenses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [
        ("Qwen3-4B", (4.0, None)),
        ("Qwen3-0.6B", (0.6, None)),
        ("Qwen3-30B-A3B", (30.0, 3.0)),
        ("Qwen3-235B-A22B-Instruct-2507", (235.0, 22.0)),
        ("SmolLM2-360M-Instruct", (0.36, None)),
        ("gpt-oss-20b", (20.9, 3.6)),  # known Mixture-of-Experts models come from a small table
        ("gpt-oss-120b", (116.8, 5.1)),
        ("Mistral-Small-3.2-24B-Instruct-2506", (24.0, None)),
        ("Qwen2.5-7B-Instruct-1M", (7.0, None)),  # "1M" is the context length, not the size
        ("Llama-3.2-3B-Instruct", (3.0, None)),
        ("Phi-4-mini-instruct", (None, None)),
    ],
)
def test_params_from_name(name, expected):
    assert params_from_name(name) == expected


@pytest.mark.parametrize(
    "repo, expected",
    [
        ("unsloth/Qwen3-4B-GGUF", "Qwen3 4B"),
        ("unsloth/Qwen3-30B-A3B-GGUF", "Qwen3 30B-A3B"),
        ("ggml-org/gpt-oss-20b-GGUF", "gpt-oss 20B"),
        ("bartowski/Qwen_Qwen3-4B-GGUF", "Qwen3 4B"),
        ("bartowski/microsoft_Phi-4-mini-instruct-GGUF", "Phi-4 mini Instruct"),
        ("bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF", "Mistral Small 3.2 24B Instruct 2506"),
        ("bartowski/Llama-3.2-3B-Instruct-GGUF", "Llama-3.2 3B Instruct"),
    ],
)
def test_prettify_repo_name(repo, expected):
    assert prettify_repo_name(repo) == expected


def test_license_from_tag_card_and_other():
    assert license_of(hub_model("a/b", license="apache-2.0")) == "Apache-2.0"
    assert license_of(hub_model("a/b", license="mit")) == "MIT"
    assert license_of(hub_model("a/b", license="llama3.2")) == "llama3.2"
    assert license_of(hub_model("a/b", license=None)) is None
    # No tag: fall back to the model card (dict or object), incl. "other" + license_name.
    assert license_of(hub_model("a/b", license=None, card={"license": "MIT"})) == "MIT"
    card = SimpleNamespace(license="other", license_name="qwen-research")
    assert license_of(hub_model("a/b", license=None, card=card)) == "other (qwen-research)"
    assert license_of(hub_model("a/b", license=None, card={"license": ["apache-2.0"]})) == "Apache-2.0"


# ---------------------------------------------------------------------------
# entry_from_hub
# ---------------------------------------------------------------------------


def files_of(repo: str) -> list[tuple[str, int]]:
    return [(f.path, f.size) for f in TREES[repo] if hasattr(f, "size")]


def model(repo: str) -> SimpleNamespace:
    return next(m for m in MODELS if m.id == repo)


def test_entry_from_hub_qwen3_4b():
    entry = entry_from_hub(model("unsloth/Qwen3-4B-GGUF"), files_of("unsloth/Qwen3-4B-GGUF"))
    assert entry is not None
    assert entry.key == entry.hf_repo == "unsloth/Qwen3-4B-GGUF"
    assert entry.display_name == "Qwen3 4B"
    assert entry.family == "Qwen3"
    assert entry.params_b == pytest.approx(4.02)  # from the GGUF header, not the name
    assert entry.active_params_b is None
    assert entry.architecture == "qwen3"
    assert entry.native_context == 40960
    assert entry.license == "Apache-2.0"
    assert catalog.is_permissive(entry.license)
    # The original model's card is where its license lives.
    assert entry.license_url == "https://huggingface.co/Qwen/Qwen3-4B"
    assert entry.quant == "Q4_K_M"
    assert entry.gguf_files == ("Qwen3-4B-Q4_K_M.gguf",)
    assert entry.file_size_gb == pytest.approx(2.5, abs=0.01)
    assert entry.ollama_ref == "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"
    assert entry.reasoning is True
    assert entry.source == "huggingface"
    assert entry.downloads == 900_000
    assert entry.base_model == "Qwen/Qwen3-4B"
    assert entry.gated is False
    options = dict(entry.quant_options)
    assert set(options) == {"BF16", "Q8_0", "Q4_K_M", "UD-Q4_K_XL", "IQ4_XS"}
    assert options["Q8_0"] == pytest.approx(4.28, abs=0.01)
    # Same original model as a curated seed: reuse its hand-written blurb.
    assert entry.blurb == catalog.get_model("qwen3-4b").blurb


def test_entry_from_hub_moe_with_shards():
    entry = entry_from_hub(model("unsloth/Qwen3-30B-A3B-GGUF"), files_of("unsloth/Qwen3-30B-A3B-GGUF"))
    assert entry.params_b == pytest.approx(30.53)  # GGUF total
    assert entry.active_params_b == 3.0  # "A3B" in the name
    options = dict(entry.quant_options)
    assert options["Q8_0"] == pytest.approx(32.45, abs=0.01)  # both shards added up
    assert "F16" not in options  # the mmproj file is not a quant of the model


def test_entry_from_hub_gpt_oss_mxfp4():
    entry = entry_from_hub(model("ggml-org/gpt-oss-20b-GGUF"), files_of("ggml-org/gpt-oss-20b-GGUF"))
    assert entry.quant == "MXFP4"
    assert entry.gguf_files == ("gpt-oss-20b-mxfp4.gguf",)
    assert entry.active_params_b == 3.6
    assert entry.params_b == pytest.approx(20.91)
    assert entry.reasoning is True
    assert entry.family == "gpt-oss"


def test_entry_from_hub_params_fall_back_to_name_then_file_size():
    by_name = entry_from_hub(hub_model("unsloth/Qwen3-8B-GGUF"), [("Qwen3-8B-Q4_K_M.gguf", 5_027_783_488)])
    assert by_name.params_b == 8.0
    by_size = entry_from_hub(model("unsloth/Phi-4-mini-instruct-GGUF"), files_of("unsloth/Phi-4-mini-instruct-GGUF"))
    # No header and no size in the name: estimated from the Q4_K_M file (~4.8 bits per weight).
    assert by_size.params_b == pytest.approx(3.96, abs=0.05)
    assert by_size.license == "MIT"
    assert by_size.family == "Phi-4"
    assert by_size.reasoning is False


def test_entry_from_hub_unusable_repos():
    assert entry_from_hub(hub_model("a/Vision-Proj-GGUF"), [("mmproj-model-f16.gguf", 600_000_000)]) is None
    assert entry_from_hub(hub_model("a/Empty-GGUF"), []) is None
    assert entry_from_hub(hub_model("a/Unsized-GGUF"), [("model-Q4_K_M.gguf", 0)]) is None


def test_entry_from_hub_non_default_quants_and_gated():
    info = hub_model("unsloth/Some-Model-7B-Instruct-GGUF", gated="auto")
    entry = entry_from_hub(info, [("Some-Model-7B-Instruct-UD-Q4_K_XL.gguf", 4_500_000_000),
                                  ("Some-Model-7B-Instruct-UD-Q2_K_XL.gguf", 2_900_000_000)])
    assert entry.quant == "UD-Q4_K_XL"  # nearest Q4_K_M's size when the usual tags are missing
    assert entry.gated is True
    assert "Mixture" not in entry.blurb and "7B" in entry.blurb


def test_qwen3_2507_instruct_is_not_a_thinking_model():
    instruct = entry_from_hub(hub_model("unsloth/Qwen3-4B-Instruct-2507-GGUF"), [("Qwen3-4B-Instruct-2507-Q4_K_M.gguf", 2_500_000_000)])
    thinking = entry_from_hub(hub_model("unsloth/Qwen3-4B-Thinking-2507-GGUF"), [("Qwen3-4B-Thinking-2507-Q4_K_M.gguf", 2_500_000_000)])
    assert instruct.reasoning is False
    assert thinking.reasoning is True


# ---------------------------------------------------------------------------
# rejection_reason
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info, fragment",
    [
        (hub_model("huihui-ai/Qwen3-8B-abliterated-GGUF"), "family-friendly"),
        (hub_model("someone/Llama-3-8B-Uncensored-Instruct-GGUF"), "family-friendly"),
        (hub_model("someone/Nice-Name-7B-Instruct-GGUF", tags=("not-for-all-audiences",)), "family-friendly"),
        (hub_model("someone/Nice-Name-7B-Instruct-GGUF", tags=("nsfw",)), "family-friendly"),
        (hub_model("someone/Story-12B-Roleplay-GGUF"), "family-friendly"),
        (hub_model("unsloth/Qwen3-8B-GGUF", tags=("base_model:finetune:huihui-ai/Qwen3-8B-abliterated",)), "family-friendly"),
        (hub_model("unsloth/Qwen2.5-Coder-7B-Instruct-GGUF"), "specialist"),
        (hub_model("Qwen/Qwen3-Embedding-0.6B-GGUF", pipeline="feature-extraction"), "specialist"),
        (hub_model("gpustack/bge-reranker-v2-m3-GGUF"), "specialist"),
        (hub_model("unsloth/Qwen2.5-VL-7B-Instruct-GGUF"), "specialist"),
        (hub_model("unsloth/Qwen3-4B-Base-GGUF", conversational=False), "base model"),
        (hub_model("someone/Plain-7B-GGUF", conversational=False), "base model"),
        (hub_model("bartowski/Llama-3.2-3B-Instruct-GGUF", license="llama3.2"), "license (llama3.2)"),
        (hub_model("someone/Mystery-7B-Instruct-GGUF", license=None), "license (unknown)"),
        (hub_model("unsloth/SmolLM2-135M-Instruct-GGUF"), "too small"),
        (hub_model("unsloth/DeepSeek-V3-0324-GGUF", total=671_000_000_000), "too big"),
        (hub_model("randomuser/Tiny-Chat-3B-Instruct-GGUF", downloads=50), "publisher we don't know"),
    ],
)
def test_rejection_reasons(info, fragment):
    reason = rejection_reason(info)
    assert reason is not None and fragment in reason


@pytest.mark.parametrize(
    "info",
    [
        model("unsloth/Qwen3-4B-GGUF"),
        model("ggml-org/gpt-oss-20b-GGUF"),
        model("unsloth/Phi-4-mini-instruct-GGUF"),
        hub_model("bartowski/allenai_OLMo-2-0425-1B-it-GGUF", conversational=False, license="apache-2.0"),  # "-it" = chat
        hub_model("unsloth/Qwen3-8B-GGUF", conversational=False),  # a chat-only family
    ],
)
def test_good_models_pass(info):
    assert rejection_reason(info) is None


def test_allow_all_licenses_relaxes_only_the_license_check():
    llama = hub_model("bartowski/Llama-3.2-3B-Instruct-GGUF", license="llama3.2")
    assert rejection_reason(llama, allow_all_licenses=True) is None
    abliterated = hub_model("huihui-ai/Qwen3-8B-abliterated-GGUF", license="llama3.2")
    assert "family-friendly" in rejection_reason(abliterated, allow_all_licenses=True)


# ---------------------------------------------------------------------------
# discover_models: live
# ---------------------------------------------------------------------------


def test_live_discovery_filters_dedupes_and_caches(tmp_path):
    api = FakeHubApi()
    clock = FakeClock()
    cache = tmp_path / "hf_models.json"
    result = discover_models(api=api, cache_path=cache, clock=clock)

    assert result.source == "live"
    assert result.fetched_at == clock.now
    assert result.stale is False
    assert result.notes and result.notes[0].startswith("Found 4 models")
    models = by_repo(result)
    assert live_repos(result) == {
        "unsloth/Qwen3-4B-GGUF",
        "unsloth/Qwen3-30B-A3B-GGUF",
        "ggml-org/gpt-oss-20b-GGUF",  # curated repo wins over unsloth's more-downloaded copy
        "unsloth/Phi-4-mini-instruct-GGUF",
    }
    # Excluded: duplicates, non-permissive, unsafe, coder, embedding, base, unknown license, untrusted, gated.
    for repo in ("bartowski/Qwen_Qwen3-4B-GGUF", "unsloth/gpt-oss-20b-GGUF", "bartowski/Llama-3.2-3B-Instruct-GGUF",
                 "huihui-ai/Qwen3-8B-abliterated-GGUF", "mradermacher/Spicy-Tales-12B-GGUF",
                 "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen/Qwen3-Embedding-0.6B-GGUF",
                 "unsloth/Qwen3-4B-Base-GGUF", "someone/Mystery-7B-Instruct-GGUF",
                 "randomuser/Tiny-Chat-3B-Instruct-GGUF", "ibm-granite/granite-3.3-8b-instruct-GGUF"):
        assert repo not in models, repo
    assert any("license isn't Apache-2.0 or MIT" in n for n in result.notes)

    # Curated seeds fill in original models the search didn't find (and don't duplicate found ones).
    curated = {m.hf_repo for m in result.models if m.source == "curated"}
    assert "bartowski/SmolLM2-1.7B-Instruct-GGUF" in curated
    assert "unsloth/Qwen3-4B-GGUF" not in curated
    assert len([m for m in result.models if m.hf_repo.lower() == "unsloth/qwen3-4b-gguf"]) == 1

    # Efficient: one search per trusted publisher + one global, then lookups only for candidates.
    assert len(api.list_calls) == len(catalog.TRUSTED_PUBLISHERS) + 1
    assert {c.get("author") for c in api.list_calls} == set(catalog.TRUSTED_PUBLISHERS) | {None}
    for call in api.list_calls:
        assert call["filter"] == "gguf" and call["pipeline_tag"] == "text-generation" and call["sort"] == "downloads"
        assert {"gguf", "cardData", "tags", "downloads", "likes", "lastModified", "gated"} <= set(call["expand"])
        assert "full" not in call and "cardData" not in call
    assert sorted(api.tree_calls) == sorted(live_repos(result))

    # Cache on disk, with a schema version and the fetch time.
    data = json.loads(cache.read_text(encoding="utf-8"))
    assert data["schema_version"] == CACHE_SCHEMA_VERSION
    assert data["fetched_at"] == clock.now
    assert {m["hf_repo"] for m in data["models"]} == live_repos(result)  # seeds aren't cached


def test_allow_all_licenses_includes_other_licenses(tmp_path):
    result = discover_models(api=FakeHubApi(), cache_path=tmp_path / "c.json", clock=FakeClock(),
                             allow_all_licenses=True)
    models = by_repo(result)
    assert models["bartowski/Llama-3.2-3B-Instruct-GGUF"].license == "llama3.2"
    assert models["someone/Mystery-7B-Instruct-GGUF"].license == "unknown"
    assert "huihui-ai/Qwen3-8B-abliterated-GGUF" not in models  # still family-friendly
    assert not any("license isn't" in n for n in result.notes)


def test_max_candidates_limits_lookups(tmp_path):
    api = FakeHubApi()
    result = discover_models(api=api, cache_path=tmp_path / "c.json", clock=FakeClock(), max_candidates=2)
    assert len(api.tree_calls) == 2
    assert len(live_repos(result)) == 2
    # The curated / most popular original models are looked at first.
    assert set(api.tree_calls) <= {"unsloth/Qwen3-4B-GGUF", "unsloth/Qwen3-30B-A3B-GGUF", "ggml-org/gpt-oss-20b-GGUF",
                                   "unsloth/Phi-4-mini-instruct-GGUF"}


def test_gated_models_only_when_nothing_else(tmp_path):
    gated = [hub_model("unsloth/Qwen3-4B-GGUF", gated="manual", total=4_022_468_096, base="Qwen/Qwen3-4B")]
    result = discover_models(api=FakeHubApi(models=gated), cache_path=tmp_path / "c.json", clock=FakeClock())
    assert result.source == "live"
    assert by_repo(result)["unsloth/Qwen3-4B-GGUF"].gated is True
    assert any("login" in n for n in result.notes)


def test_one_failed_search_or_lookup_does_not_spoil_the_rest(tmp_path):
    api = FakeHubApi(list_errors_for={"ggml-org"}, tree_errors_for={"unsloth/Phi-4-mini-instruct-GGUF"})
    result = discover_models(api=api, cache_path=tmp_path / "c.json", clock=FakeClock())
    assert result.source == "live"
    live = live_repos(result)
    assert "unsloth/Qwen3-4B-GGUF" in live
    assert "unsloth/Phi-4-mini-instruct-GGUF" not in live
    assert any("Couldn't read the file list of 1 model" in n for n in result.notes)
    # gpt-oss wasn't found via ggml-org's search, but the global search still has it.
    assert "ggml-org/gpt-oss-20b-GGUF" in live


def test_discovered_models_work_with_the_fit_engine(tmp_path):
    result = discover_models(api=FakeHubApi(), cache_path=tmp_path / "c.json", clock=FakeClock())
    specs = SystemSpecs(
        os_name="Linux", os_version="", arch="x86_64", cpu_name="Test", cpu_cores_physical=8, cpu_cores_logical=16,
        ram_total_gb=32, ram_available_gb=16, disk_free_gb=500,
        gpus=[GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0, bandwidth_gbs=360.0)],
        ram_bandwidth_gbs=40, cpu_flags=["avx2"],
    )
    ranked = catalog.rank_models(specs, result.models)
    assert ranked and ranked[0].verdict != "no"
    shortlist = catalog.pick_shortlist(ranked)
    assert any("recommended" in f.badges for f in shortlist)
    qwen = next(f for f in ranked if f.model.hf_repo == "unsloth/Qwen3-4B-GGUF")
    assert qwen.quant in dict(qwen.model.quant_options)


# ---------------------------------------------------------------------------
# discover_models: cache, offline, failures
# ---------------------------------------------------------------------------


def test_fresh_cache_is_used_without_touching_the_network(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    first = discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    clock.advance(3600)  # an hour later
    second = discover_models(api=ExplodingApi(), cache_path=cache, clock=clock)
    assert second.source == "cache"
    assert second.stale is False
    assert second.fetched_at == first.fetched_at
    assert live_repos(second) == live_repos(first)
    assert by_repo(second)["unsloth/Qwen3-4B-GGUF"] == by_repo(first)["unsloth/Qwen3-4B-GGUF"]
    assert "saved 60 minutes ago" in second.notes[0]


def test_stale_cache_triggers_a_refresh(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    clock.advance(73 * 3600)
    api = FakeHubApi()
    result = discover_models(api=api, cache_path=cache, clock=clock)
    assert result.source == "live" and api.list_calls
    assert json.loads(cache.read_text())["fetched_at"] == clock.now


def test_refresh_ignores_a_fresh_cache(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    api = FakeHubApi()
    result = discover_models(api=api, cache_path=cache, clock=clock, refresh=True)
    assert result.source == "live" and api.list_calls


def test_stale_cache_is_the_fallback_when_the_hub_fails(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    clock.advance(10 * 86400)
    result = discover_models(api=FakeHubApi(list_error=ConnectionError("no route to host")), cache_path=cache,
                             clock=clock)
    assert result.source == "cache"
    assert result.stale is True
    assert "unsloth/Qwen3-4B-GGUF" in live_repos(result)
    assert "couldn't reach Hugging Face" in result.notes[0] and "10 days ago" in result.notes[0]


def test_offline_uses_cache_even_when_old_and_never_calls_the_hub(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    clock.advance(30 * 86400)
    result = discover_models(api=ExplodingApi(), cache_path=cache, clock=clock, offline=True, refresh=True)
    assert result.source == "cache" and result.stale is True
    assert "offline" in result.notes[0]


def test_offline_without_cache_uses_curated_seeds(tmp_path):
    result = discover_models(api=ExplodingApi(), cache_path=tmp_path / "missing.json", offline=True)
    assert result.source == "curated"
    assert result.models == list(catalog.MODEL_CATALOG)
    assert result.fetched_at is None
    assert "built-in picks" in result.notes[0]


@pytest.mark.parametrize(
    "error, fragment",
    [
        (ConnectionError("Failed to establish a new connection"), "no internet connection"),
        (TimeoutError("read timed out"), "took too long"),
        (type("OfflineModeIsEnabled", (ConnectionError,), {})("offline"), "HF_HUB_OFFLINE"),
        (type("HfHubHTTPError", (OSError,), {"response": SimpleNamespace(status_code=503)})("oops"), "HTTP 503"),
    ],
)
def test_network_error_without_cache_falls_back_to_curated_with_a_note(tmp_path, error, fragment):
    cache = tmp_path / "c.json"
    result = discover_models(api=FakeHubApi(list_error=error), cache_path=cache, clock=FakeClock())
    assert result.source == "curated"
    assert result.models == list(catalog.MODEL_CATALOG)
    assert fragment in result.notes[0]
    assert not cache.exists()


def test_no_suitable_models_counts_as_a_failure(tmp_path):
    only_bad = [m for m in MODELS if "abliterated" in m.id or "Coder" in m.id]
    result = discover_models(api=FakeHubApi(models=only_bad), cache_path=tmp_path / "c.json", clock=FakeClock())
    assert result.source == "curated"
    assert "no suitable models" in result.notes[0]


@pytest.mark.parametrize(
    "content",
    [
        "{not json",
        "[]",
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION + 1, "fetched_at": 1, "models": []}),
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "fetched_at": "yesterday", "models": []}),
        json.dumps({"schema_version": CACHE_SCHEMA_VERSION, "fetched_at": 1_750_000_000, "models": [{"key": 1}]}),
    ],
)
def test_corrupt_or_foreign_cache_is_ignored(tmp_path, content):
    cache = tmp_path / "c.json"
    cache.write_text(content, encoding="utf-8")
    offline = discover_models(api=ExplodingApi(), cache_path=cache, offline=True, clock=FakeClock())
    assert offline.source == "curated"  # the bad cache was ignored, not trusted
    live = discover_models(api=FakeHubApi(), cache_path=cache, clock=FakeClock())
    assert live.source == "live"
    assert json.loads(cache.read_text())["schema_version"] == CACHE_SCHEMA_VERSION  # replaced with a good one


def test_cache_skips_individual_bad_entries(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    data = json.loads(cache.read_text())
    data["models"].append({"key": "broken", "hf_repo": "no-slash"})
    data["models"].append("nonsense")
    cache.write_text(json.dumps(data))
    result = discover_models(api=ExplodingApi(), cache_path=cache, clock=clock)
    assert result.source == "cache"
    assert "unsloth/Qwen3-4B-GGUF" in live_repos(result)


def test_cache_from_the_future_counts_as_stale(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    clock.advance(-86400)  # the computer's clock went backwards
    api = FakeHubApi()
    assert discover_models(api=api, cache_path=cache, clock=clock).source == "live"


def test_cache_and_license_modes(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    # A cache built with all licenses spent its candidate budget on a different
    # pool of models, so the default view searches again rather than reuse it...
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock, allow_all_licenses=True)
    fresh_api = FakeHubApi()
    fresh_view = discover_models(api=fresh_api, cache_path=cache, clock=clock)
    assert fresh_view.source == "live" and fresh_api.list_calls
    # ...but it's still a fine (filtered) fallback when the Hub can't be reached.
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock, allow_all_licenses=True)
    default_view = discover_models(api=ExplodingApi(), cache_path=cache, clock=clock, offline=True)
    assert default_view.source == "cache"
    assert all(catalog.is_permissive(m.license) for m in default_view.models)
    # A permissive-only cache can't answer an all-licenses request: search again.
    discover_models(api=FakeHubApi(), cache_path=cache, clock=clock, refresh=True)
    api = FakeHubApi()
    wider = discover_models(api=api, cache_path=cache, clock=clock, allow_all_licenses=True)
    assert wider.source == "live" and api.list_calls


def test_unwritable_cache_is_only_a_note(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    result = discover_models(api=FakeHubApi(), cache_path=blocker / "sub" / "c.json", clock=FakeClock())
    assert result.source == "live"
    assert any("Couldn't save" in n for n in result.notes)


def test_default_cache_path_honours_gettowork_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    assert hf_discovery.default_cache_path() == tmp_path / "cache" / "hf_models.json"
    discover_models(api=FakeHubApi(), clock=FakeClock())
    assert (tmp_path / "cache" / "hf_models.json").is_file()


def test_entry_json_round_trip():
    entry = entry_from_hub(model("unsloth/Qwen3-30B-A3B-GGUF"), files_of("unsloth/Qwen3-30B-A3B-GGUF"))
    restored = hf_discovery._entry_from_json(json.loads(json.dumps(hf_discovery._entry_to_json(entry))))
    assert restored == entry


# ---------------------------------------------------------------------------
# Time budget
# ---------------------------------------------------------------------------


def test_slow_searches_use_up_the_budget_and_lookups_are_skipped(tmp_path):
    clock = FakeClock()
    api = FakeHubApi(on_list=lambda kwargs: clock.advance(5))  # 11 searches x 5 s > 20 s budget
    result = discover_models(api=api, cache_path=tmp_path / "c.json", clock=clock, timeout_s=20)
    assert api.tree_calls == []
    assert result.source == "curated"
    assert "took too long" in result.notes[0]


def test_budget_stops_further_lookups_but_keeps_finished_ones(tmp_path, monkeypatch):
    monkeypatch.setattr(hf_discovery, "MAX_WORKERS", 1)  # one at a time: deterministic order
    clock = FakeClock()
    api = FakeHubApi(on_tree=lambda repo: clock.advance(8))
    result = discover_models(api=api, cache_path=tmp_path / "c.json", clock=clock, timeout_s=20)
    assert result.source == "live"
    assert len(api.tree_calls) == 3  # started at t=0, 8 and 16 s; the 4th would start after the deadline
    assert len(live_repos(result)) == 3
    assert any("skipped checking 1 model" in n for n in result.notes)


def test_a_hanging_request_does_not_hang_discovery(tmp_path):
    release = threading.Event()

    def maybe_hang(repo):
        if repo == "unsloth/Phi-4-mini-instruct-GGUF":
            release.wait(10)

    try:
        started = time.monotonic()
        result = discover_models(api=FakeHubApi(on_tree=maybe_hang), cache_path=tmp_path / "c.json", timeout_s=0.5)
        elapsed = time.monotonic() - started
    finally:
        release.set()
    assert elapsed < 5
    assert result.source == "live"
    assert "unsloth/Phi-4-mini-instruct-GGUF" not in live_repos(result)
    assert any("skipped checking 1 model" in n for n in result.notes)


# ---------------------------------------------------------------------------
# Compatibility with the real huggingface_hub API (signatures only, no network)
# ---------------------------------------------------------------------------


def test_calls_match_the_real_huggingface_hub_signatures():
    from huggingface_hub import HfApi

    list_params = inspect.signature(HfApi.list_models).parameters
    for name in ("filter", "author", "pipeline_tag", "sort", "limit", "expand"):
        assert name in list_params
    from huggingface_hub import hf_api

    expand_literal = getattr(hf_api, "ExpandModelProperty_T", None)  # Literal["author", "gguf", ...]
    if expand_literal is not None:
        assert set(hf_discovery.EXPAND_FIELDS) <= set(typing.get_args(expand_literal))
    tree_params = inspect.signature(HfApi.list_repo_tree).parameters
    assert {"recursive", "expand"} <= set(tree_params)


def test_repo_gguf_files_with_tree_and_with_model_info():
    files = repo_gguf_files(FakeHubApi(), "unsloth/Qwen3-30B-A3B-GGUF")
    assert ("Q8_0/Qwen3-30B-A3B-Q8_0-00001-of-00002.gguf", 16_000_000_000) in files
    assert all(name.endswith(".gguf") for name, _ in files)  # README / folders dropped

    class InfoOnlyApi:  # no list_repo_tree: use model_info(files_metadata=True).siblings
        def model_info(self, repo_id, files_metadata=False):
            assert files_metadata
            return SimpleNamespace(siblings=[
                SimpleNamespace(rfilename="a-Q4_K_M.gguf", size=None, lfs=SimpleNamespace(size=123)),
                SimpleNamespace(rfilename="config.json", size=10, lfs=None),
            ])

    assert repo_gguf_files(InfoOnlyApi(), "x/y") == [("a-Q4_K_M.gguf", 123)]


def test_explainer_mentions_the_key_ideas():
    text = hf_discovery.DISCOVERY_EXPLAINER
    for word in ("GGUF", "Apache-2.0", "MIT", "no warranty", "offline"):
        assert word in text


def test_discovery_result_defaults():
    result = DiscoveryResult(models=[], source="live")
    assert result.notes == [] and result.fetched_at is None and result.stale is False
    assert dataclasses.is_dataclass(result)


def test_end_to_end_with_the_real_hfapi_class(tmp_path, monkeypatch):
    """Drive the genuine HfApi / ModelInfo / RepoFile code with Hub-shaped JSON.

    Only `huggingface_hub.hf_api.paginate` (the HTTP layer) is replaced, so this
    checks our calls and attribute access against the real library, offline.
    """
    from huggingface_hub import HfApi, hf_api

    search_json = [
        {
            "_id": "1", "id": "unsloth/Qwen3-4B-GGUF", "downloads": 900_000, "likes": 300,
            "tags": ["gguf", "qwen3", "text-generation", "license:apache-2.0", "conversational",
                     "base_model:quantized:Qwen/Qwen3-4B"],
            "gguf": {"total": 4_022_468_096, "architecture": "qwen3", "context_length": 40960,
                     "chat_template": "{% for message in messages %}..."},
            "cardData": {"license": "apache-2.0", "base_model": ["Qwen/Qwen3-4B"], "tags": ["unsloth"]},
            "lastModified": "2025-05-01T12:00:00.000Z", "gated": False, "pipeline_tag": "text-generation",
        },
        {
            "_id": "2", "id": "unsloth/Qwen3-8B-abliterated-GGUF", "downloads": 5_000_000, "likes": 1,
            "tags": ["gguf", "license:apache-2.0", "conversational"], "gated": False,
            "pipeline_tag": "text-generation",
        },
    ]
    tree_json = [
        {"type": "file", "oid": "a1", "size": 2_497_280_256, "path": "Qwen3-4B-Q4_K_M.gguf",
         "lfs": {"oid": "f" * 64, "size": 2_497_280_256, "pointerSize": 135}},
        {"type": "directory", "oid": "d1", "path": "BF16"},
        {"type": "file", "oid": "a2", "size": 4_100_000_000, "path": "BF16/Qwen3-4B-BF16-00001-of-00002.gguf",
         "lfs": {"oid": "e" * 64, "size": 4_100_000_000, "pointerSize": 135}},
        {"type": "file", "oid": "a3", "size": 3_951_285_312, "path": "BF16/Qwen3-4B-BF16-00002-of-00002.gguf",
         "lfs": {"oid": "d" * 64, "size": 3_951_285_312, "pointerSize": 135}},
        {"type": "file", "oid": "a4", "size": 1_234, "path": "README.md"},
    ]
    requests: list[tuple[str, dict]] = []

    def fake_paginate(path, params=None, headers=None):
        requests.append((path, dict(params or {})))
        if path.endswith("/api/models"):
            assert "full" not in params and "cardData" not in params  # the Hub rejects these with expand
            if params.get("author") in (None, "unsloth"):
                return iter([dict(item) for item in search_json])
            return iter([])
        if "/tree/" in path:
            assert "unsloth/Qwen3-4B-GGUF" in path and params["recursive"] is True
            return iter([dict(item) for item in tree_json])
        raise AssertionError(f"unexpected request {path}")

    monkeypatch.setattr(hf_api, "paginate", fake_paginate)
    result = discover_models(api=HfApi(), cache_path=tmp_path / "c.json", clock=FakeClock())

    assert result.source == "live"
    entry = by_repo(result)["unsloth/Qwen3-4B-GGUF"]
    assert entry.params_b == pytest.approx(4.02) and entry.architecture == "qwen3" and entry.native_context == 40960
    assert entry.base_model == "Qwen/Qwen3-4B" and entry.license == "Apache-2.0"
    assert dict(entry.quant_options)["BF16"] == pytest.approx(8.05, abs=0.01)
    assert "unsloth/Qwen3-8B-abliterated-GGUF" not in by_repo(result)
    searches = [params for path, params in requests if path.endswith("/api/models")]
    assert len(searches) == len(catalog.TRUSTED_PUBLISHERS) + 1
    assert all(p["filter"] == ["gguf"] and p["pipeline_tag"] == "text-generation" for p in searches)
    assert sum("/tree/" in path for path, _ in requests) == 1


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_stalled_hub_requests_never_block_the_game_from_exiting(tmp_path):
    # A request that hangs forever (flaky Wi-Fi, captive portal) must not keep
    # Python alive after discovery has given up on it.
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent(f"""
        import threading, time
        from pathlib import Path
        from gettowork import hf_discovery
        class Stalled:
            def list_models(self, **kw):
                threading.Event().wait()  # never answers
        started = time.monotonic()
        result = hf_discovery.discover_models(api=Stalled(), cache_path=Path({str(tmp_path / "c.json")!r}), timeout_s=0.5)
        print("returned", result.source, round(time.monotonic() - started, 1))
    """)
    started = time.monotonic()
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0, proc.stderr
    assert "returned curated" in proc.stdout
    assert time.monotonic() - started < 15  # exited promptly instead of waiting for the stuck thread


def test_hub_requests_get_a_finite_timeout():
    from huggingface_hub.utils import get_session

    assert hf_discovery.configure_hub_timeouts() is True
    timeout = get_session().timeout
    assert timeout.connect == hf_discovery.HUB_CONNECT_TIMEOUT_S
    assert timeout.read == hf_discovery.HUB_READ_TIMEOUT_S


def test_with_deadline():
    assert hf_discovery.with_deadline(lambda: 42, 5) == 42
    with pytest.raises(ValueError):
        hf_discovery.with_deadline(lambda: (_ for _ in ()).throw(ValueError("boom")), 5)
    release = threading.Event()
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        hf_discovery.with_deadline(lambda: release.wait(10), 0.2)
    release.set()
    assert time.monotonic() - started < 3


def test_a_partial_search_is_merged_and_only_trusted_briefly(tmp_path):
    cache = tmp_path / "c.json"
    clock = FakeClock()
    full = discover_models(api=FakeHubApi(), cache_path=cache, clock=clock)
    assert "unsloth/Phi-4-mini-instruct-GGUF" in live_repos(full)
    clock.advance(4 * 86400)  # the full list is now stale
    flaky = FakeHubApi(tree_errors_for={"unsloth/Phi-4-mini-instruct-GGUF", "unsloth/Qwen3-30B-A3B-GGUF"})
    partial = discover_models(api=flaky, cache_path=cache, clock=clock)
    # What we knew before is kept rather than thrown away...
    assert "unsloth/Phi-4-mini-instruct-GGUF" in live_repos(partial)
    saved = json.loads(cache.read_text())
    assert saved["complete"] is False
    assert "unsloth/Qwen3-30B-A3B-GGUF" in {m["hf_repo"] for m in saved["models"]}
    # ...and an hour later a healthy Hub is asked again instead of serving it as "fresh".
    clock.advance(2 * 3600)
    healthy = FakeHubApi()
    again = discover_models(api=healthy, cache_path=cache, clock=clock)
    assert again.source == "live" and healthy.list_calls
    assert json.loads(cache.read_text())["complete"] is True


def test_known_mixture_of_experts_names():
    assert params_from_name("Phi-3.5-MoE-instruct") == (41.9, 6.6)
    assert params_from_name("Mixtral-8x7B-Instruct-v0.1") == (46.7, 12.9)
    assert params_from_name("granite-4.0-h-small") == (32.2, 9.0)
    assert params_from_name("granite-3.1-3b-a800m-instruct") == (3.3, 0.8)
    assert params_from_name("OLMoE-1B-7B-0125-Instruct") == (6.9, 1.3)
    assert params_from_name("GLM-4.5-Air") == (106.0, 12.0)
    assert params_from_name("Ling-lite-1.5") == (16.8, 2.75)
    assert params_from_name("Hunyuan-80B-A13B-Instruct")[1] == 13.0


def test_moe_architecture_without_a_known_active_size_is_not_ranked_as_dense():
    info = hub_model("someorg/NewMoE-60B-Instruct-GGUF", total=60_000_000_000, arch="glm4moe", downloads=5000)
    entry = entry_from_hub(info, [("NewMoE-60B-Instruct-Q4_K_M.gguf", 36 * GB)])
    assert entry is not None and entry.active_params_b == pytest.approx(15.0)  # the 25% fallback


def _gguf_bytes(arch: str, values: dict, *, vocab: int = 0) -> bytes:
    import struct

    def string(text: str) -> bytes:
        raw = text.encode()
        return struct.pack("<Q", len(raw)) + raw

    kvs = [string("general.architecture") + struct.pack("<I", 8) + string(arch)]
    kvs.append(string("general.name") + struct.pack("<I", 8) + string("Some Model"))
    for key, value in values.items():
        kvs.append(string(f"{arch}.{key}") + struct.pack("<I", 4) + struct.pack("<I", value))
    kvs.append(string(f"{arch}.rope.freq_base") + struct.pack("<I", 6) + struct.pack("<f", 1e6))
    if vocab:  # a token array, like the real vocabulary (skipped without reading it)
        tokens = b"".join(string(f"t{i}") for i in range(vocab))
        kvs.append(string("tokenizer.ggml.tokens") + struct.pack("<I", 9) + struct.pack("<IQ", 8, vocab) + tokens)
    return b"GGUF" + struct.pack("<IQQ", 3, 100, len(kvs)) + b"".join(kvs)


QWEN3_30B_A3B_HEADER = {"expert_count": 128, "expert_used_count": 8, "block_count": 48, "embedding_length": 2048,
                        "expert_feed_forward_length": 768, "context_length": 40960}


def test_read_gguf_header_and_active_parameters():
    header = hf_discovery.read_gguf_header(_gguf_bytes("qwen3moe", QWEN3_30B_A3B_HEADER, vocab=50))
    assert header["architecture"] == "qwen3moe" and header["expert_count"] == 128
    assert hf_discovery.active_params_from_header(header, 30.53) == pytest.approx(3.35, abs=0.1)
    # Cut short mid-way: we keep whatever came before.
    data = _gguf_bytes("qwen3moe", QWEN3_30B_A3B_HEADER)
    partial = hf_discovery.read_gguf_header(data[:80])
    assert partial.get("architecture") == "qwen3moe" and "expert_count" not in partial
    assert hf_discovery.read_gguf_header(b"not a gguf file at all....") == {}
    dense = {"block_count": 36, "embedding_length": 2560}
    assert hf_discovery.active_params_from_header(dense, 4.0) is None


def test_discovery_reads_the_header_of_an_unlabelled_moe_model(tmp_path):
    info = hub_model("unsloth/Mystery-MoE-30B-Instruct-GGUF", total=30_532_122_624, arch="qwen3moe",
                     downloads=500_000)
    trees = {"unsloth/Mystery-MoE-30B-Instruct-GGUF": [rfile("Mystery-MoE-30B-Instruct-Q4_K_M.gguf", 18 * GB)]}
    asked: list = []

    class Api(FakeHubApi):
        def gguf_header(self, repo, filename):
            asked.append((repo, filename))
            return hf_discovery.read_gguf_header(_gguf_bytes("qwen3moe", QWEN3_30B_A3B_HEADER))

    result = discover_models(api=Api(models=[info], trees=trees), cache_path=tmp_path / "c.json", clock=FakeClock())
    entry = by_repo(result)["unsloth/Mystery-MoE-30B-Instruct-GGUF"]
    assert asked == [("unsloth/Mystery-MoE-30B-Instruct-GGUF", "Mystery-MoE-30B-Instruct-Q4_K_M.gguf")]
    assert entry.active_params_b == pytest.approx(3.35, abs=0.1)


@pytest.mark.parametrize(
    "info",
    [
        hub_model("someuser/Llama-3.1-8B-Instruct-GGUF", downloads=5000, base="meta-llama/Llama-3.1-8B-Instruct"),
        hub_model("NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF", base="NousResearch/Hermes-2-Pro-Llama-3-8B"),
        hub_model("someuser/gemma-3-4b-it-GGUF", license="mit", downloads=5000, base="google/gemma-3-4b-it"),
    ],
)
def test_relabelled_llama_and_gemma_copies_are_not_taken_at_face_value(info):
    reason = rejection_reason(info)
    assert reason is not None and "own license terms still apply" in reason
    assert rejection_reason(info, allow_all_licenses=True) is None


def test_relabelled_copy_shows_its_real_license_family():
    info = hub_model("someuser/Llama-3.1-8B-Instruct-GGUF", downloads=5000, base="meta-llama/Llama-3.1-8B-Instruct")
    entry = entry_from_hub(info, [("Llama-3.1-8B-Instruct-Q4_K_M.gguf", 5 * GB)])
    assert entry.license == "Llama license (tagged Apache-2.0)"
    assert not catalog.is_permissive(entry.license)
    assert entry.license_url == "https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct"


def test_tinyllama_is_not_mistaken_for_meta_llama():
    info = hub_model("TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", downloads=500_000)
    assert rejection_reason(info) is None


@pytest.mark.parametrize(
    "info, why",
    [
        (hub_model("bartowski/cognitivecomputations_Dolphin3.0-Mistral-24B-GGUF"), "family-friendly"),
        (hub_model("QuantFactory/Qwen2.5-7B-GGUF", downloads=50_000), "base model"),
        (hub_model("bartowski/WizardCoder-Python-7B-V1.0-GGUF"), "specialist"),
        (hub_model("bartowski/OpenCoder-8B-Instruct-GGUF"), "specialist"),
        (hub_model("bartowski/mathstral-7B-v0.1-GGUF"), "specialist"),
        (hub_model("unsloth/gemma-3-1b-pt-GGUF", license="apache-2.0"), "base model"),
    ],
)
def test_screening_catches_more_unsuitable_models(info, why):
    reason = rejection_reason(info, allow_all_licenses=True)
    assert reason is not None and why in reason


def test_ernie_pt_means_pytorch_not_pretrained():
    assert rejection_reason(hub_model("unsloth/ERNIE-4.5-21B-A3B-PT-GGUF")) is None
    base = rejection_reason(hub_model("unsloth/ERNIE-4.5-21B-A3B-Base-PT-GGUF"))
    assert base is not None and "base model" in base


@pytest.mark.parametrize(
    "repo, base, expected",
    [
        ("unsloth/Qwen3-Next-80B-A3B-Instruct-GGUF", "Qwen/Qwen3-Next-80B-A3B-Instruct", False),
        ("unsloth/Qwen3-Next-80B-A3B-Thinking-GGUF", "Qwen/Qwen3-Next-80B-A3B-Thinking", True),
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen/Qwen3-4B-Instruct-2507", False),
        ("unsloth/Qwen3-4B-GGUF", "Qwen/Qwen3-4B", True),
        ("unsloth/Seed-OSS-36B-Instruct-GGUF", "ByteDance-Seed/Seed-OSS-36B-Instruct", True),
    ],
)
def test_reasoning_flag(repo, base, expected):
    info = hub_model(repo, base=base, total=4 * GB)
    entry = entry_from_hub(info, [(f"{repo.split('/')[1]}-Q4_K_M.gguf", 3 * GB)])
    assert entry.reasoning is expected


def test_bare_q4_file_names_are_understood():
    assert parse_quant("Phi-3-mini-4k-instruct-q4.gguf") == "Q4"
    assert parse_quant("Phi-3-mini-4k-instruct-fp16.gguf") == "F16"
    assert parse_quant("model-Q4_K.gguf") == "Q4_K"
    assert parse_quant("model.Q2_K_S.gguf") == "Q2_K_S"
    info = hub_model("microsoft/Phi-3-mini-4k-instruct-gguf", license="mit", base="microsoft/Phi-3-mini-4k-instruct")
    entry = entry_from_hub(info, [("Phi-3-mini-4k-instruct-q4.gguf", 2_390_000_000),
                                  ("Phi-3-mini-4k-instruct-fp16.gguf", 7_640_000_000)])
    assert entry.quant == "Q4" and entry.file_size_gb == pytest.approx(2.39)
    assert dict(entry.quant_options) == {"Q4": 2.39, "F16": 7.64}


# ---------------------------------------------------------------------------
# Round 3: thinking modes from the chat template; cache rules version
# ---------------------------------------------------------------------------

from gettowork import catalog as _catalog  # noqa: E402
from gettowork import hf_discovery as _hd  # noqa: E402

QWEN3_HYBRID_TEMPLATE = (
    "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n"
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n    {%- endif %}\n{%- endif %}"
)
FORCED_THINK_TEMPLATE = "{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n<think>\\n' }}\n{%- endif %}"
R1_TEMPLATE = "{% if add_generation_prompt %}{{'<｜Assistant｜><think>\\n'}}{% endif %}"
SEED_TEMPLATE = "{%- if thinking_budget is defined %}{{ thinking_budget }}{% endif %}"
PLAIN_TEMPLATE = "{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}{%- endif %}"


def _with_template(info, template):
    info.gguf = dict(info.gguf or {}, chat_template=template)
    return info


@pytest.mark.parametrize(
    "repo, template, expected",
    [
        ("unsloth/Qwen3-4B-GGUF", QWEN3_HYBRID_TEMPLATE, "switchable"),
        ("unsloth/Qwen3-4B-Thinking-2507-GGUF", FORCED_THINK_TEMPLATE, "always"),
        ("unsloth/QwQ-32B-GGUF", FORCED_THINK_TEMPLATE, "always"),
        ("unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", R1_TEMPLATE, "always"),
        ("unsloth/Qwen3-4B-Instruct-2507-GGUF", PLAIN_TEMPLATE, "none"),
        ("unsloth/Seed-OSS-36B-Instruct-GGUF", SEED_TEMPLATE, "switchable"),
        ("unsloth/gpt-oss-20b-GGUF", "{{ reasoning_effort }}", "switchable"),
        # No template: name rules, extended for newer always-thinkers.
        ("unsloth/Phi-4-reasoning-plus-GGUF", None, "always"),
        ("unsloth/Phi-4-mini-reasoning-GGUF", None, "always"),
        ("bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF", None, "always"),
        ("unsloth/Ministral-3-8B-Reasoning-2512-GGUF", None, "always"),
        ("bartowski/ServiceNow-AI_Apriel-1.5-15b-Thinker-GGUF", None, "always"),
        ("bartowski/open-thoughts_OpenThinker3-7B-GGUF", None, "always"),
        ("bartowski/SmolLM3-3B-GGUF", None, "switchable"),
        ("bartowski/Mistral-7B-Instruct-v0.3-GGUF", None, "none"),
    ],
)
def test_thinking_mode_comes_from_the_template_then_the_name(repo, template, expected):
    info = hub_model(repo, total=4_000_000_000)
    if template is not None:
        _with_template(info, template)
    else:
        info.gguf = {"total": 4_000_000_000}
    name = repo.split("/")[1].removesuffix("-GGUF")
    entry = entry_from_hub(info, [(f"{name}-Q4_K_M.gguf", 2_500_000_000)])
    assert entry.thinking == expected
    assert entry.reasoning is (expected != "none")


def test_a_guarded_think_tag_is_not_forced():
    guarded = ("{%- if add_generation_prompt %}{{- '<|im_start|>assistant\\n' }}"
               "{%- if not x %}{{ '<think>' }}{% endif %}{%- endif %}")
    assert _hd.thinking_mode_for("Some-Model-7B-Instruct", guarded) == "none"


def test_always_thinkers_are_listed_but_warned_about():
    info = _with_template(hub_model("unsloth/QwQ-32B-GGUF", total=32_760_000_000), FORCED_THINK_TEMPLATE)
    entry = entry_from_hub(info, [("QwQ-32B-Q4_K_M.gguf", 19_850_000_000)])
    assert rejection_reason(info) is None  # still in the full list...
    assert "always thinks" in entry.blurb  # ...with a warning


def _stale_rules_cache(tmp_path, entry, *, age_s=3600.0, now=1_000_000.0):
    cache = tmp_path / "c.json"
    payload = {"schema_version": _hd.CACHE_SCHEMA_VERSION, "fetched_at": now - age_s, "allow_all_licenses": False,
               "complete": True, "rules_version": "older-rules", "models": [_hd._entry_to_json(entry)]}
    cache.write_text(json.dumps(payload))
    return cache


def _hub_entry(repo, **kw):
    info = hub_model(repo, total=3_212_749_888, **kw)
    name = repo.split("/")[1].removesuffix("-GGUF")
    return entry_from_hub(info, [(f"{name}-Q4_K_M.gguf", 2_000_000_000)])


def test_a_list_saved_under_older_rules_is_not_fresh(tmp_path):
    entry = _hub_entry("someone/Nice-3B-Instruct-GGUF")
    cache = _stale_rules_cache(tmp_path, entry)
    api = FakeHubApi()
    result = discover_models(api=api, cache_path=cache, clock=lambda: 1_000_000.0)
    assert result.source == "live" and api.list_calls
    assert json.loads(cache.read_text())["rules_version"] == _hd.RULES_VERSION


def test_offline_a_list_saved_under_older_rules_is_rescreened(tmp_path):
    # Saved before the Llama/Gemma check existed: labelled plain "Apache-2.0".
    relabelled = dataclasses.replace(_hub_entry("someone/Llama-3.2-3B-Instruct-GGUF", base="meta-llama/Llama-3.2-3B-Instruct"),
                                     license="Apache-2.0")
    thinker = dataclasses.replace(_hub_entry("unsloth/Phi-4-mini-reasoning-GGUF"), thinking="switchable")
    fine = _hub_entry("someone/Nice-3B-Instruct-GGUF")
    cache = tmp_path / "c.json"
    payload = {"schema_version": _hd.CACHE_SCHEMA_VERSION, "fetched_at": 0.0, "allow_all_licenses": False,
               "complete": True, "models": [_hd._entry_to_json(e) for e in (relabelled, thinker, fine)]}
    cache.write_text(json.dumps(payload))  # no rules_version at all: saved by an older game
    result = discover_models(api=ExplodingApi(), cache_path=cache, offline=True, clock=lambda: 30 * 86400.0)
    repos = by_repo(result)
    assert "someone/Llama-3.2-3B-Instruct-GGUF" not in repos  # its own license terms apply
    assert repos["unsloth/Phi-4-mini-reasoning-GGUF"].thinking == "always"
    assert "someone/Nice-3B-Instruct-GGUF" in repos


def test_rescreen_with_all_licenses_relabels_instead_of_dropping():
    entry = dataclasses.replace(_hub_entry("someone/Llama-3.2-3B-Instruct-GGUF", base="meta-llama/Llama-3.2-3B-Instruct"),
                                license="Apache-2.0")
    kept = _hd.rescreen_entry(entry, allow_all_licenses=True)
    assert kept is not None and kept.license == "Llama license (tagged Apache-2.0)"


def test_the_rules_version_changes_when_a_rule_changes(monkeypatch):
    before = _hd._rules_fingerprint()
    monkeypatch.setattr(_hd, "MIN_DOWNLOADS_UNTRUSTED", 5000)
    assert _hd._rules_fingerprint() != before


def test_entry_json_round_trip_keeps_the_thinking_mode():
    info = _with_template(hub_model("unsloth/QwQ-32B-GGUF", total=32_760_000_000), FORCED_THINK_TEMPLATE)
    entry = entry_from_hub(info, [("QwQ-32B-Q4_K_M.gguf", 19_850_000_000)])
    assert _hd._entry_from_json(json.loads(json.dumps(_hd._entry_to_json(entry)))).thinking == "always"
    assert _catalog.thinking_mode(entry) == "always"


# ---------------------------------------------------------------------------
# Round 4: a damaged cache entry never crashes the ranking
# ---------------------------------------------------------------------------


def test_every_model_entry_field_has_a_json_type_rule():
    from gettowork import types as t

    listed = set(t._ENTRY_TEXT + t._ENTRY_OPTIONAL_TEXT + t._ENTRY_NUMBER + t._ENTRY_OPTIONAL_NUMBER
                 + t._ENTRY_COUNT + t._ENTRY_OPTIONAL_COUNT + t._ENTRY_FLAG + t._ENTRY_LISTS)
    assert listed == {f.name for f in dataclasses.fields(ModelEntry)}


@pytest.mark.parametrize("field,value", [
    ("downloads", "123"), ("active_params_b", "3"), ("base_model", 3), ("family", None), ("family", 5),
    ("architecture", 5), ("native_context", "40960"), ("context_tokens", "4096"),
])
def test_wrong_typed_cache_fields_are_converted_and_rank_fine(field, value):
    entry = entry_from_hub(model("unsloth/Qwen3-30B-A3B-GGUF"), files_of("unsloth/Qwen3-30B-A3B-GGUF"))
    data = json.loads(json.dumps(hf_discovery._entry_to_json(entry)))
    data[field] = value
    restored = hf_discovery._entry_from_json(data)
    assert restored is not None
    specs = SystemSpecs(os_name="Linux", os_version="t", arch="x86_64", cpu_name="t", cpu_cores_physical=8,
                        cpu_cores_logical=16, ram_total_gb=32.0, ram_available_gb=24.0, disk_free_gb=500.0,
                        ram_bandwidth_gbs=50.0)
    ranked = catalog.rank_models(specs, [restored])
    catalog.pick_shortlist(ranked, 6)
    from gettowork.setup_flow import _model_context, why_line

    assert why_line(ranked[0])
    assert _model_context(restored) > 0


@pytest.mark.parametrize("field,value", [
    ("downloads", "lots"), ("params_b", "big"), ("native_context", [1]), ("reasoning", "maybe"),
    ("quant_options", "Q4_K_M"), ("gguf_files", "one.gguf"),
])
def test_unreadable_cache_fields_drop_the_entry(field, value):
    entry = entry_from_hub(model("unsloth/Qwen3-30B-A3B-GGUF"), files_of("unsloth/Qwen3-30B-A3B-GGUF"))
    data = json.loads(json.dumps(hf_discovery._entry_to_json(entry)))
    data[field] = value
    assert hf_discovery._entry_from_json(data) is None


# ---------------------------------------------------------------------------
# Round 4: dense hybrids aren't Mixture-of-Experts; specialists; families; KV shapes
# ---------------------------------------------------------------------------

GRANITE_MICRO_FILES = [("granite-4.0-h-micro-Q4_K_M.gguf", 1_940_000_000), ("granite-4.0-h-micro-Q8_0.gguf", 3_400_000_000)]


def test_a_dense_granite_hybrid_is_not_treated_as_mixture_of_experts():
    info = hub_model("unsloth/granite-4.0-h-micro-GGUF", total=3_190_000_000, arch="granitehybrid",
                     base="ibm-granite/granite-4.0-h-micro")
    dense_header = {"architecture": "granitehybrid", "expert_count": 0, "block_count": 40}
    assert entry_from_hub(info, GRANITE_MICRO_FILES, dense_header).active_params_b is None
    assert entry_from_hub(info, GRANITE_MICRO_FILES, None).active_params_b is None  # no header: no guess
    assert hf_discovery._needs_header(info)  # worth reading the header to find out
    moe_header = {"architecture": "granitehybrid", "expert_count": 64, "expert_used_count": 6, "block_count": 40,
                  "embedding_length": 1536, "expert_feed_forward_length": 512}
    tiny = hub_model("unsloth/granite-4.0-h-tinyish-GGUF", total=6_900_000_000, arch="granitehybrid")
    moe = entry_from_hub(tiny, [("granite-Q4_K_M.gguf", 4_200_000_000)], moe_header)
    assert moe.active_params_b is not None and moe.active_params_b < moe.params_b
    assert hf_discovery._is_moe("qwen3moe", None) and not hf_discovery._is_moe("qwen3moe", {"expert_count": 1})


def test_rescreening_drops_an_old_moe_guess_for_a_dense_hybrid():
    info = hub_model("unsloth/granite-4.0-h-micro-GGUF", total=3_190_000_000, arch="granitehybrid",
                     base="ibm-granite/granite-4.0-h-micro")
    entry = entry_from_hub(info, GRANITE_MICRO_FILES, None)
    old = dataclasses.replace(entry, active_params_b=round(entry.params_b * hf_discovery.MOE_UNKNOWN_ACTIVE_SHARE, 2),
                              blurb="A 3.19B Mixture-of-Experts Granite model that only wakes ~798M parameters")
    fixed = hf_discovery.rescreen_entry(old)
    assert fixed is not None and fixed.active_params_b is None and "Mixture-of-Experts" not in fixed.blurb


def test_the_moe_tables_are_part_of_the_rules_fingerprint(monkeypatch):
    before = hf_discovery._rules_fingerprint()
    monkeypatch.setattr(hf_discovery, "MOE_UNKNOWN_ACTIVE_SHARE", 0.3)
    assert hf_discovery._rules_fingerprint() != before


@pytest.mark.parametrize("repo, base", [
    ("bartowski/Qwen_Qwen3Guard-Gen-8B-GGUF", "Qwen/Qwen3Guard-Gen-8B"),
    ("bartowski/Skywork_Skywork-SWE-32B-GGUF", "Skywork/Skywork-SWE-32B"),
    ("bartowski/moonshotai_Kimi-Dev-72B-GGUF", "moonshotai/Kimi-Dev-72B"),
    ("bartowski/all-hands_openhands-lm-32b-v0.1-GGUF", "all-hands/openhands-lm-32b-v0.1"),
    ("bartowski/Alibaba-NLP_Tongyi-DeepResearch-30B-A3B-GGUF", "Alibaba-NLP/Tongyi-DeepResearch-30B-A3B"),
    ("mradermacher/II-Search-4B-GGUF", "Intelligent-Internet/II-Search-4B"),
    ("bartowski/Intelligent-Internet_II-Medical-8B-GGUF", "Intelligent-Internet/II-Medical-8B"),
    ("mradermacher/Pleias-RAG-1B-GGUF", "PleIAs/Pleias-RAG-1B"),
    ("someone/Friendly-8B-Instruct-GGUF", "someone/Qwen3-8B-Medical-Agent"),  # a specialist's GGUF, renamed
])
def test_agents_guards_and_domain_fine_tunes_are_not_storytellers(repo, base):
    info = hub_model(repo, downloads=500_000, total=8_000_000_000, arch="qwen3", base=base)
    assert "specialist" in (rejection_reason(info) or ""), repo


def test_ordinary_chat_models_still_pass_the_specialist_screen():
    for repo in ("unsloth/Qwen3-4B-Instruct-2507-GGUF", "unsloth/Phi-3-medium-4k-instruct-GGUF",
                 "unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "NousResearch/Hermes-3-Llama-3.2-3B-GGUF"):
        info = hub_model(repo, downloads=500_000, total=4_000_000_000, arch="qwen3")
        reason = rejection_reason(info) or ""
        assert "specialist" not in reason, (repo, reason)


def test_a_fine_tunes_family_comes_from_its_architecture_not_its_uploader():
    info = hub_model("bartowski/Menlo_Jan-nano-GGUF", total=4_022_468_096, arch="qwen3", base="Menlo/Jan-nano")
    entry = entry_from_hub(info, [("Menlo_Jan-nano-Q4_K_M.gguf", 2_500_000_000)])
    assert entry.family == "Qwen3"
    odd = hub_model("someone/Zephyrish-7B-chat-GGUF", total=7_000_000_000, arch="stablelm")
    assert entry_from_hub(odd, [("z-Q4_K_M.gguf", 4_000_000_000)]).family == "Stablelm"


def test_the_kv_shape_is_read_from_the_gguf_header():
    assert hf_discovery.kv_shape_from_header(
        {"block_count": 40, "attention.head_count": 40, "embedding_length": 5120}) == (40, 40, 128)
    assert hf_discovery.kv_shape_from_header(
        {"block_count": 36, "attention.head_count": 32, "attention.head_count_kv": 8,
         "attention.key_length": 128, "embedding_length": 4096}) == (36, 8, 128)
    assert hf_discovery.kv_shape_from_header({"block_count": 0}) == ()
    assert hf_discovery.kv_shape_from_header(None) == ()
    info = hub_model("someone/Newfangled-13B-Instruct-GGUF", total=13_700_000_000, arch="newarch")
    assert hf_discovery._needs_header(info)  # unknown family: read the header for its attention shape
    entry = entry_from_hub(info, [("n-Q4_K_M.gguf", 8_000_000_000)],
                           {"block_count": 40, "attention.head_count": 40, "embedding_length": 5120})
    assert entry.kv_shape == (40, 40, 128)
    restored = hf_discovery._entry_from_json(json.loads(json.dumps(hf_discovery._entry_to_json(entry))))
    assert restored.kv_shape == (40, 40, 128)
    assert not hf_discovery._needs_header(hub_model("unsloth/Qwen3-4B-GGUF", arch="qwen3", base="Qwen/Qwen3-4B"))
