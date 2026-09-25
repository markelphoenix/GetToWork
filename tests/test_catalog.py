"""Tests for gettowork.catalog: the curated seeds and the fit / ranking engine.

The "machine matrix" tests check that recommendations make sense on a range
of imaginary computers, from a 4 GB netbook to a 64 GB + RTX 4090 tower.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from gettowork import catalog, perf
from gettowork.types import FitResult, GPUInfo, ModelEntry, SystemSpecs

# ---------------------------------------------------------------------------
# Imaginary machines
# ---------------------------------------------------------------------------


def gpu(name: str, vendor: str, vram: float) -> GPUInfo:
    g = GPUInfo(name=name, vendor=vendor, vram_gb=vram)
    g.bandwidth_gbs = perf.estimate_gpu_bandwidth(g)
    return g


def machine(
    ram: float,
    ram_bw: float,
    gpus: tuple[GPUInfo, ...] = (),
    *,
    unified: bool = False,
    disk: float = 500.0,
    cores: int = 8,
) -> SystemSpecs:
    return SystemSpecs(
        os_name="Darwin" if unified else "Linux",
        os_version="",
        arch="arm64" if unified else "x86_64",
        cpu_name="Test CPU",
        cpu_cores_physical=cores,
        cpu_cores_logical=cores * 2,
        ram_total_gb=ram,
        ram_available_gb=ram / 2,
        disk_free_gb=disk,
        gpus=list(gpus),
        unified_memory=unified,
        ram_bandwidth_gbs=ram_bw,
        cpu_flags=["neon"] if unified else ["avx2", "fma"],
    )


TINY = machine(4, 12, cores=2)  # old netbook, no GPU
LAPTOP_8GB = machine(8, 20, cores=4)  # thin-and-light, integrated graphics only
DESKTOP_16GB = machine(16, 40, cores=6)  # DDR4 desktop, no GPU
RTX_3060 = machine(16, 40, (gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0),))
MAC_M2 = machine(16, 60, (gpu("Apple M2 GPU", "apple", 11.2),), unified=True)
RTX_4090 = machine(64, 50, (gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0),), cores=16)
LOW_DISK = machine(16, 40, (gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0),), disk=3.0)
BIG_CPU_BOX = machine(64, 40, cores=8)  # lots of RAM, no GPU: memory is plentiful, bandwidth isn't

MACHINES = {
    "tiny": TINY,
    "laptop_8gb": LAPTOP_8GB,
    "desktop_16gb": DESKTOP_16GB,
    "rtx_3060": RTX_3060,
    "mac_m2": MAC_M2,
    "rtx_4090": RTX_4090,
    "low_disk": LOW_DISK,
    "big_cpu_box": BIG_CPU_BOX,
}


def fit_for(specs: SystemSpecs, key: str) -> FitResult:
    model = catalog.get_model(key)
    assert model is not None
    return catalog.evaluate_fit(specs, model)


# ---------------------------------------------------------------------------
# The curated seed list
# ---------------------------------------------------------------------------

EXPECTED_REPOS = {
    "unsloth/Qwen3-0.6B-GGUF",
    "unsloth/Qwen3-1.7B-GGUF",
    "bartowski/SmolLM2-1.7B-Instruct-GGUF",
    "unsloth/Phi-4-mini-instruct-GGUF",
    "unsloth/Qwen3-4B-GGUF",
    "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
    "unsloth/Qwen3-8B-GGUF",
    "unsloth/Qwen3-14B-GGUF",
    "ggml-org/gpt-oss-20b-GGUF",
    "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF",
    "unsloth/Qwen3-30B-A3B-GGUF",
    "unsloth/Qwen3-32B-GGUF",
}


def test_seed_list_has_the_expected_repos():
    assert {m.hf_repo for m in catalog.MODEL_CATALOG} == EXPECTED_REPOS


def test_every_seed_is_permissively_licensed():
    for m in catalog.MODEL_CATALOG:
        assert catalog.is_permissive(m.license), m.key
        assert m.license in ("Apache-2.0", "MIT")
        assert not re.search(r"llama-?[234]|gemma|command-r|falcon", m.hf_repo, re.IGNORECASE), m.key


def test_seed_keys_and_repos_are_unique():
    keys = [m.key for m in catalog.MODEL_CATALOG]
    repos = [m.hf_repo.lower() for m in catalog.MODEL_CATALOG]
    assert len(keys) == len(set(keys))
    assert len(repos) == len(set(repos))


def test_seeds_are_ordered_small_to_large():
    sizes = [m.params_b for m in catalog.MODEL_CATALOG]
    assert all(later >= earlier - 0.05 for earlier, later in zip(sizes, sizes[1:]))  # two ~1.7B models tie


def test_seed_license_urls_point_at_the_original_model():
    for m in catalog.MODEL_CATALOG:
        assert m.license_url.startswith("https://huggingface.co/"), m.key
        assert m.license_url != f"https://huggingface.co/{m.hf_repo}", m.key  # not the GGUF mirror
        assert "gguf" not in m.license_url.lower()
        assert m.base_model and m.license_url.endswith(m.base_model)


def test_seed_publishers_are_trusted_and_metadata_is_complete():
    trusted = {p.lower() for p in catalog.TRUSTED_PUBLISHERS}
    for m in catalog.MODEL_CATALOG:
        assert m.hf_repo.split("/")[0].lower() in trusted
        assert m.source == "curated"
        assert m.blurb and m.display_name and m.family
        assert m.ollama_ref
        assert m.gguf_files and all(f.endswith(".gguf") for f in m.gguf_files)
        assert m.quant.lower() in m.gguf_files[0].lower()
        assert m.native_context and m.native_context >= 4096


def test_seed_quant_sizes_are_realistic():
    for m in catalog.MODEL_CATALOG:
        options = dict(m.quant_options)
        assert m.quant in options and options[m.quant] == m.file_size_gb
        for quant, size in m.quant_options:
            estimate = m.params_b * catalog.QUANT_BITS[quant] / 8
            assert 0.75 * estimate <= size <= 1.3 * estimate, (m.key, quant, size, estimate)
        # Bigger quants are bigger files.
        by_bits = sorted(m.quant_options, key=lambda qs: catalog.QUANT_BITS[qs[0]])
        assert [s for _, s in by_bits] == sorted(s for _, s in by_bits), m.key


def test_gpt_oss_seed():
    m = catalog.get_model("gpt-oss-20b")
    assert m.quant == "MXFP4" and m.quant_options == (("MXFP4", 12.1),)
    assert m.ollama_ref == "gpt-oss:20b"
    assert m.active_params_b and m.active_params_b < m.params_b
    assert m.gguf_files == ("gpt-oss-20b-mxfp4.gguf",)


def test_moe_seed_has_active_params():
    m = catalog.get_model("qwen3-30b-a3b")
    assert m.params_b == pytest.approx(30.5) and m.active_params_b == pytest.approx(3.3)


def test_get_model_by_key_or_repo():
    assert catalog.get_model("qwen3-4b").hf_repo == "unsloth/Qwen3-4B-GGUF"
    assert catalog.get_model("UNSLOTH/qwen3-4b-gguf").key == "qwen3-4b"
    assert catalog.get_model("  qwen3-8b ").key == "qwen3-8b"
    assert catalog.get_model("no-such-model") is None
    assert catalog.get_model("") is None


def test_trusted_publishers_and_licenses_constants():
    for publisher in ("unsloth", "bartowski", "ggml-org", "lmstudio-community", "Qwen", "microsoft"):
        assert publisher in catalog.TRUSTED_PUBLISHERS
    assert catalog.PERMISSIVE_LICENSES == frozenset({"apache-2.0", "mit"})


@pytest.mark.parametrize(
    "license_id, ok",
    [("Apache-2.0", True), ("apache-2.0", True), ("Apache 2.0", True), ("MIT", True), ("mit", True),
     ("llama3.1", False), ("gemma", False), ("other", False), ("", False), (None, False), ("cc-by-nc-4.0", False)],
)
def test_is_permissive(license_id, ok):
    assert catalog.is_permissive(license_id) is ok


# ---------------------------------------------------------------------------
# Quant helpers
# ---------------------------------------------------------------------------


def test_quant_tables_match_the_contract():
    expected = {"Q8_0": 8.5, "Q6_K": 6.6, "Q5_K_M": 5.7, "Q4_K_M": 4.8, "IQ4_XS": 4.3, "Q3_K_M": 3.9,
                "IQ3_M": 3.7, "Q2_K": 3.0, "MXFP4": 4.25, "F16": 16, "BF16": 16}
    for quant, bits in expected.items():
        assert catalog.QUANT_BITS[quant] == pytest.approx(bits)
    assert catalog.QUANT_PREFERENCE == ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q4_K_S", "Q3_K_M", "IQ3_M")
    assert all(catalog.QUANT_BITS[q] >= 3.7 for q in catalog.QUANT_PREFERENCE)


def test_quant_quality_follows_the_preference_ladder():
    qualities = [catalog.quant_quality(q) for q in catalog.QUANT_PREFERENCE]
    assert qualities == sorted(qualities, reverse=True)
    assert catalog.quant_quality("Q2_K") < catalog.quant_quality("IQ3_M")
    assert catalog.quant_quality("F16") == 1.0
    assert 0 < catalog.quant_quality("MYSTERY") < 1


def test_quant_helpers_handle_unsloth_dynamic_tags():
    assert catalog.quant_bits("UD-Q4_K_XL") == catalog.QUANT_BITS["Q4_K_XL"]
    assert catalog.quant_bits("q4_k_m") == 4.8
    assert catalog.quant_bits("nonsense") is None


def test_estimate_quant_size():
    assert catalog.estimate_quant_size_gb(8.19, "Q4_K_M") == pytest.approx(8.19 * 4.8 / 8 * 1.05, abs=0.01)
    assert catalog.estimate_quant_size_gb(8.0, "WHAT") is None
    assert catalog.estimate_quant_size_gb(0, "Q4_K_M") is None


# ---------------------------------------------------------------------------
# Memory model
# ---------------------------------------------------------------------------


def test_kv_cache_is_exact_for_known_architectures():
    qwen8b = catalog.get_model("qwen3-8b")
    # 2 (K and V) × 36 layers × 8 KV heads × 128 dims × 2 bytes × 4096 tokens
    assert catalog.kv_cache_gb(qwen8b) == pytest.approx(2 * 36 * 8 * 128 * 2 * 4096 / 1e9)
    assert catalog.kv_cache_gb(qwen8b, 8192) == pytest.approx(2 * catalog.kv_cache_gb(qwen8b))
    smol = catalog.get_model("smollm2-1.7b")  # no grouped-query attention: a big KV cache for its size
    assert catalog.kv_cache_gb(smol) > catalog.kv_cache_gb(catalog.get_model("qwen3-1.7b"))


def test_kv_cache_respects_native_context():
    smol = catalog.get_model("smollm2-1.7b")  # trained for 8,192 tokens
    assert catalog.kv_cache_gb(smol, 32768) == pytest.approx(catalog.kv_cache_gb(smol, 8192))


def make_entry(**overrides) -> ModelEntry:
    values = dict(
        key="someone/Mystery-8B-Instruct-GGUF", display_name="Mystery 8B", family="Mystery", params_b=8.0,
        active_params_b=None, license="Apache-2.0", license_url="https://huggingface.co/someone/Mystery-8B-Instruct",
        hf_repo="someone/Mystery-8B-Instruct-GGUF", quant="Q4_K_M", file_size_gb=0.0, ollama_ref="",
        reasoning=False, blurb="", source="huggingface",
        quant_options=(("Q8_0", 0.0), ("Q6_K", 0.0), ("Q4_K_M", 0.0), ("Q3_K_M", 0.0)),
    )
    values.update(overrides)
    return ModelEntry(**values)


def test_kv_cache_fallback_is_close_to_the_exact_maths():
    for m in catalog.MODEL_CATALOG:
        exact = catalog.kv_cache_gb(m)
        fallback = catalog.kv_cache_gb(dataclasses.replace(
            m, key="x", hf_repo="someone/unknown", base_model=None, display_name="Unknown"))
        assert 0.4 * exact <= fallback <= 2.5 * exact, m.key


def test_estimate_memory_adds_weights_kv_and_overhead():
    m = catalog.get_model("qwen3-8b")
    expected = 5.03 + catalog.kv_cache_gb(m) + catalog.OVERHEAD_GB
    assert catalog.estimate_memory_gb(m) == pytest.approx(expected)
    assert catalog.estimate_memory_gb(m, weights_gb=8.71) == pytest.approx(expected - 5.03 + 8.71)
    # Unknown size: estimated from parameters and bits.
    mystery = make_entry()
    assert catalog.estimate_memory_gb(mystery) == pytest.approx(
        catalog.estimate_quant_size_gb(8.0, "Q4_K_M") + catalog.kv_cache_gb(mystery) + catalog.OVERHEAD_GB
    )


# ---------------------------------------------------------------------------
# choose_quant
# ---------------------------------------------------------------------------


def test_big_gpu_gets_higher_quality_quants():
    assert catalog.choose_quant(RTX_4090, catalog.get_model("qwen3-8b"))[0] == "Q8_0"
    assert catalog.choose_quant(RTX_4090, catalog.get_model("qwen3-14b"))[0] == "Q8_0"
    assert catalog.choose_quant(RTX_3060, catalog.get_model("qwen3-8b"))[0] == "Q6_K"


def test_mac_and_slow_machines_stay_at_the_4bit_sweet_spot():
    assert catalog.choose_quant(MAC_M2, catalog.get_model("qwen3-8b"))[0] == "Q4_K_M"
    # Plenty of RAM but slow memory: Q8_0 would fit, but would crawl. Stay small and quick.
    quant, _ = catalog.choose_quant(BIG_CPU_BOX, catalog.get_model("qwen3-8b"))
    assert catalog.QUANT_BITS[quant] <= catalog.QUANT_BITS["Q4_K_M"]


def test_tiny_models_get_upgraded_even_on_a_cpu():
    quant, _ = catalog.choose_quant(DESKTOP_16GB, catalog.get_model("qwen3-0.6b"))
    assert quant == "Q8_0"


def test_higher_quants_on_bigger_machines_across_the_board():
    for key in ("qwen3-4b", "qwen3-8b", "mistral-7b"):
        model = catalog.get_model(key)
        small_bits = catalog.QUANT_BITS[catalog.choose_quant(LAPTOP_8GB, model)[0]]
        big_bits = catalog.QUANT_BITS[catalog.choose_quant(RTX_4090, model)[0]]
        assert big_bits > small_bits, key


def test_falls_back_down_the_ladder_when_memory_is_short():
    quant, size = catalog.choose_quant(LAPTOP_8GB, catalog.get_model("mistral-7b"))
    assert quant in ("IQ4_XS", "Q3_K_M") and size < 4.37


def test_nothing_fits_returns_none():
    assert catalog.choose_quant(TINY, catalog.get_model("qwen3-32b")) is None


def test_disk_space_limits_the_quant():
    specs = dataclasses.replace(RTX_4090, disk_free_gb=7.0)
    quant, size = catalog.choose_quant(specs, catalog.get_model("qwen3-8b"))
    assert size + catalog.DISK_SPARE_GB <= 7.0
    assert quant == "Q5_K_M"


def test_unknown_sizes_are_estimated():
    quant, size = catalog.choose_quant(RTX_3060, make_entry())
    assert quant in ("Q8_0", "Q6_K", "Q4_K_M")
    assert size == pytest.approx(catalog.estimate_quant_size_gb(8.0, quant))


def test_sub_3_bit_quants_are_a_last_resort_and_marked_tight():
    entry = make_entry(quant_options=(("Q4_K_M", 5.0), ("Q2_K", 3.1)), params_b=8.0)
    specs = machine(6.8, 20)  # 4.3 GB of RAM to spare
    assert catalog.choose_quant(specs, entry) == ("Q2_K", 3.1)
    fit = catalog.evaluate_fit(specs, entry)
    assert fit.verdict == "tight"
    assert "heavily compressed" in fit.reason


def test_model_with_only_one_quant():
    fit = fit_for(RTX_4090, "gpt-oss-20b")
    assert fit.quant == "MXFP4" and fit.verdict in ("great", "ok")


# ---------------------------------------------------------------------------
# evaluate_fit
# ---------------------------------------------------------------------------


def test_placements():
    assert fit_for(RTX_3060, "qwen3-8b").placement == "gpu"
    assert fit_for(MAC_M2, "qwen3-8b").placement == "unified"
    assert fit_for(RTX_3060, "qwen3-30b-a3b").placement == "partial"
    assert fit_for(DESKTOP_16GB, "qwen3-4b").placement == "cpu"
    none = fit_for(TINY, "qwen3-32b")
    assert none.placement == "none" and none.verdict == "no"
    assert none.est_tokens_per_s is None and none.est_speed == "n/a"
    assert "Too big" in none.reason


def test_fit_result_fields_are_filled():
    fit = fit_for(RTX_3060, "qwen3-8b")
    assert fit.quant == "Q6_K" and fit.download_gb == pytest.approx(6.73)
    # Hub sizes are decimal GB; memory budgets are the binary GB the computer reports.
    expected = (6.73 + catalog.kv_cache_gb(fit.model)) * catalog.GIB_PER_GB + 0.9
    assert fit.est_memory_gb == pytest.approx(expected, abs=0.05)
    assert 25 <= fit.est_tokens_per_s <= 60 and fit.est_speed == "fast"
    assert fit.verdict in ("great", "ok")
    assert fit.reason.endswith(".") and "RTX 3060" in fit.reason and "tokens/s" in fit.reason
    assert fit.badges == ()


def test_verdict_thresholds():
    # need / budget: <= 0.6 great, <= 0.85 ok, <= 1.0 tight, else no
    assert catalog._verdict_for(0.5) == "great"
    assert catalog._verdict_for(0.6) == "great"
    assert catalog._verdict_for(0.7) == "ok"
    assert catalog._verdict_for(0.95) == "tight"
    assert catalog._verdict_for(1.01) == "no"


def test_low_disk_says_so():
    fit = fit_for(LOW_DISK, "qwen3-8b")
    assert fit.verdict == "no"
    assert "disk" in fit.reason


def test_unknown_disk_space_does_not_block():
    specs = dataclasses.replace(RTX_3060, disk_free_gb=-1.0)
    assert fit_for(specs, "qwen3-8b").verdict != "no"


def test_moe_is_much_faster_than_dense_on_cpu():
    moe = fit_for(BIG_CPU_BOX, "qwen3-30b-a3b")
    dense = fit_for(BIG_CPU_BOX, "qwen3-32b")
    assert moe.est_tokens_per_s > 5 * dense.est_tokens_per_s
    assert moe.est_speed in ("usable", "fast") and dense.est_speed == "very slow"


def test_multiple_gpus_pool_their_memory():
    two = machine(64, 40, (gpu("NVIDIA GeForce RTX 3090", "nvidia", 24.0), gpu("NVIDIA GeForce RTX 3090", "nvidia", 24.0)))
    one = machine(64, 40, (gpu("NVIDIA GeForce RTX 3090", "nvidia", 24.0),))
    assert fit_for(two, "qwen3-32b").placement == "gpu"
    assert fit_for(one, "qwen3-32b").placement in ("gpu", "partial")
    assert fit_for(two, "qwen3-32b").verdict in ("great", "ok")


def test_gpu_without_known_vram_is_ignored():
    specs = machine(16, 40, (GPUInfo("NVIDIA GeForce RTX 3060", "nvidia", 0.0),))
    assert fit_for(specs, "qwen3-4b").placement == "cpu"


# ---------------------------------------------------------------------------
# Scoring and ranking
# ---------------------------------------------------------------------------


def test_score_components():
    m = catalog.get_model("qwen3-8b")
    base = dict(quant="Q4_K_M", verdict="great", tokens_per_s=30.0, ratio=0.5, download_gb=5.0)
    good = catalog.score_fit(m, **base)
    assert catalog.score_fit(m, **{**base, "verdict": "no"}) < -50
    assert catalog.score_fit(m, **{**base, "verdict": "tight", "ratio": 0.95}) < good
    assert catalog.score_fit(m, **{**base, "tokens_per_s": 5.0}) < good - 10
    assert catalog.score_fit(m, **{**base, "tokens_per_s": 2.0}) < good - 40
    assert catalog.score_fit(m, **{**base, "placement": "partial"}) < good
    assert catalog.score_fit(dataclasses.replace(m, gated=True), **base) < good
    assert catalog.score_fit(dataclasses.replace(m, license="llama3.1"), **base) < good
    live = dataclasses.replace(m, source="huggingface", hf_repo="someone/Qwen3-8B-GGUF")
    assert catalog.score_fit(live, **base) < good  # not curated, not a trusted publisher
    popular = dataclasses.replace(live, downloads=1_000_000)
    assert catalog.score_fit(popular, **base) > catalog.score_fit(live, **base)
    bigger = catalog.get_model("qwen3-14b")
    assert catalog.score_fit(bigger, **base) > good  # same speed and fit, more parameters


def test_rank_models_puts_runnable_models_first_by_score():
    ranked = catalog.rank_models(DESKTOP_16GB)
    assert len(ranked) == len(catalog.MODEL_CATALOG)
    runnable = [f for f in ranked if f.verdict != "no"]
    assert ranked[: len(runnable)] == runnable
    assert [f.score for f in runnable] == sorted((f.score for f in runnable), reverse=True)
    assert all(f.verdict == "no" for f in ranked[len(runnable):])


def test_rank_models_accepts_a_custom_catalog():
    ranked = catalog.rank_models(RTX_3060, [make_entry(), catalog.get_model("qwen3-4b")])
    assert {f.model.display_name for f in ranked} == {"Mystery 8B", "Qwen3 4B"}


@pytest.mark.parametrize("name", list(MACHINES))
def test_recommendation_is_never_very_slow(name):
    rec = catalog.recommend(MACHINES[name])
    assert rec is not None
    assert rec.est_speed != "very slow" and rec.est_tokens_per_s >= 3
    assert rec.verdict != "no"
    assert "recommended" in rec.badges


def test_tiny_box_gets_a_tiny_model():
    rec = catalog.recommend(TINY)
    assert rec.model.params_b <= 2.1


def test_8gb_laptop_gets_a_small_model():
    rec = catalog.recommend(LAPTOP_8GB)
    assert rec.model.params_b <= 4.1
    assert rec.placement == "cpu" and rec.verdict in ("great", "ok")


def test_16gb_desktop_gets_a_small_to_mid_model():
    rec = catalog.recommend(DESKTOP_16GB)
    assert 1.5 <= rec.model.params_b <= 8.5
    assert rec.est_tokens_per_s >= 8


def test_rtx_3060_gets_a_capable_gpu_model():
    rec = catalog.recommend(RTX_3060)
    assert 4 <= rec.model.params_b <= 21
    assert rec.placement in ("gpu", "partial") and rec.est_tokens_per_s >= 20


def test_m2_gets_a_model_in_unified_memory():
    rec = catalog.recommend(MAC_M2)
    assert 4 <= rec.model.params_b <= 15
    assert rec.placement == "unified" and rec.est_tokens_per_s >= 8


def test_rtx_4090_gets_a_big_model():
    rec = catalog.recommend(RTX_4090)
    assert rec.model.params_b >= 14 or rec.model.key == "qwen3-30b-a3b"
    assert rec.placement == "gpu"


def test_low_disk_gets_something_that_fits_on_disk():
    rec = catalog.recommend(LOW_DISK)
    # Free disk space is reported in binary GB (like the OS shows it), downloads in decimal GB.
    assert rec.download_gb * catalog.GIB_PER_GB + catalog.DISK_SPARE_GB <= LOW_DISK.disk_free_gb


def test_nothing_playable_means_no_recommendation():
    hopeless = machine(2.0, 5, cores=1)
    assert catalog.recommend(hopeless) is None
    assert catalog.pick_shortlist(catalog.rank_models(hopeless)) == []


# ---------------------------------------------------------------------------
# pick_shortlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(MACHINES))
def test_shortlist_shape(name):
    ranked = catalog.rank_models(MACHINES[name])
    picks = catalog.pick_shortlist(ranked)
    assert 1 <= len(picks) <= 6
    assert "recommended" in picks[0].badges
    assert all(f.verdict != "no" for f in picks)
    bases = [catalog._base_identity(f.model) for f in picks]
    assert len(bases) == len(set(bases))
    assert all(f.badges == () for f in ranked)  # the input list was not modified
    for f in picks:
        if "smartest" in f.badges:
            assert f.est_tokens_per_s >= 5
        if "fastest" in f.badges:
            assert f.verdict in ("great", "ok")
        if not f.badges:
            assert f.est_tokens_per_s >= 3  # fillers are always playable


def test_shortlist_badges_on_a_big_machine():
    picks = catalog.pick_shortlist(catalog.rank_models(RTX_4090))
    assert len(picks) == 6
    by_badge = {badge: f for f in picks for badge in f.badges}
    assert {"recommended", "smartest"} <= set(by_badge)
    # The recommended pick is already quick on a 4090, so "fastest" (if shown at all)
    # must be much quicker *and* nearly as capable - never a toy 0.6B model.
    rec = by_badge["recommended"]
    fastest = by_badge.get("fastest")
    if fastest is not None and fastest is not rec:
        assert fastest.est_tokens_per_s >= 1.4 * rec.est_tokens_per_s
        assert catalog._quality_points(fastest.model, fastest.quant) >= (
            catalog.FASTEST_QUALITY_SHARE * catalog._quality_points(rec.model, rec.quant))
        assert catalog._effective_params_b(fastest.model) >= catalog.FASTEST_MIN_EFFECTIVE_B
    assert all(f.model.params_b >= 1 for f in picks if "fastest" in f.badges)
    assert by_badge["smartest"].model.params_b >= by_badge["recommended"].model.params_b


def test_shortlist_respects_n():
    assert len(catalog.pick_shortlist(catalog.rank_models(RTX_4090), n=3)) == 3


def test_shortlist_never_shows_two_conversions_of_one_model():
    curated = catalog.get_model("qwen3-8b")
    mirror = dataclasses.replace(
        curated, key="Qwen/Qwen3-8B-GGUF", hf_repo="Qwen/Qwen3-8B-GGUF", source="huggingface",
        downloads=900_000, quant_options=(("Q8_0", 8.71), ("Q4_K_M", 5.03)),
    )
    picks = catalog.pick_shortlist(catalog.rank_models(RTX_3060, [curated, mirror, catalog.get_model("qwen3-4b")]))
    assert sum(1 for f in picks if catalog._base_identity(f.model) == "qwen3-8b") == 1


def test_base_identity_strips_publisher_and_suffixes():
    assert catalog._base_identity(catalog.get_model("mistral-small-3.2-24b")) == "mistral-small-3.2-24b-2506"
    assert catalog._base_identity(make_entry(base_model=None)) == "mystery-8b"
    assert catalog._base_identity(make_entry(base_model="org/Summit-7B")) == "summit-7b"  # "it" inside a word stays


def test_one_model_can_earn_two_badges():
    picks = catalog.pick_shortlist(catalog.rank_models(TINY))
    assert len(picks) == 1
    assert set(picks[0].badges) >= {"recommended"}


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def test_explain_fit_shows_the_working():
    for specs, key in ((RTX_3060, "qwen3-8b"), (RTX_3060, "qwen3-30b-a3b"), (DESKTOP_16GB, "qwen3-4b"),
                       (MAC_M2, "qwen3-8b"), (TINY, "qwen3-32b")):
        fit = fit_for(specs, key)
        text = catalog.explain_fit(specs, fit)
        assert fit.quant in text
        assert "Memory:" in text and "Budget:" in text and "Why" in text
        assert "not a guarantee" in text
        if fit.est_tokens_per_s:
            assert "Speed:" in text and "tokens/s" in text


def test_memory_explainer_is_short_friendly_and_honest():
    words = len(catalog.MEMORY_FORMULA_EXPLAINER.split())
    assert 60 <= words <= 180
    text = catalog.MEMORY_FORMULA_EXPLAINER.lower()
    assert "kv cache" in text and "quantization" in text
    assert "no warranty" in text


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_turn_speed_counts_prompt_reading():
    # On a processor, reading ~1,200 prompt tokens makes each turn much longer
    # than the writing speed alone suggests; on a graphics card it barely matters.
    assert catalog.turn_tokens_per_s(10.0, "cpu") == pytest.approx(10.0 * 250 / (250 + 1200 / 8))
    assert catalog.turn_tokens_per_s(10.0, "gpu") == pytest.approx(10.0 * 250 / (250 + 1200 / 60))
    assert catalog.turn_tokens_per_s(None, "cpu") == 0.0
    assert catalog.turn_tokens_per_s(10.0, "cpu") < catalog.turn_tokens_per_s(10.0, "gpu") < 10.0


def test_thinking_bonus_only_when_the_game_lets_it_think():
    qwen = catalog.get_model("qwen3-4b")
    slow = catalog.score_fit(qwen, quant="Q4_K_M", verdict="great", tokens_per_s=12.0, ratio=0.3, download_gb=2.5)
    plain = dataclasses.replace(qwen, reasoning=False, thinking="none")
    slow_plain = catalog.score_fit(plain, quant="Q4_K_M", verdict="great", tokens_per_s=12.0, ratio=0.3, download_gb=2.5)
    assert slow == slow_plain  # too slow for the game to let it think: no bonus
    fast = catalog.score_fit(qwen, quant="Q4_K_M", verdict="great", tokens_per_s=40.0, ratio=0.3, download_gb=2.5)
    fast_plain = catalog.score_fit(plain, quant="Q4_K_M", verdict="great", tokens_per_s=40.0, ratio=0.3, download_gb=2.5)
    assert fast - fast_plain == pytest.approx(catalog.BONUS_REASONING)


def test_windows_keeps_more_ram_for_itself():
    linux = machine(16, 40)
    windows = dataclasses.replace(linux, os_name="Windows")
    assert catalog._ram_budget_gb(linux) - catalog._ram_budget_gb(windows) == pytest.approx(1.0)
    assert "minus 3.5 GB for your system" in catalog.explain_fit(windows, catalog.evaluate_fit(windows, catalog.get_model("qwen3-4b")))


def test_memory_need_uses_the_computers_own_units():
    fit = catalog.evaluate_fit(DESKTOP_16GB, catalog.get_model("qwen3-4b"))
    kv = catalog.kv_cache_gb(fit.model)
    assert fit.est_memory_gb == pytest.approx((fit.download_gb + kv) * catalog.GIB_PER_GB + catalog.OVERHEAD_GB, abs=0.05)
    assert "binary gigabytes" in catalog.explain_fit(DESKTOP_16GB, fit)


def test_fastest_is_never_a_toy_on_a_fast_machine_but_helps_a_slow_one():
    big = catalog.pick_shortlist(catalog.rank_models(RTX_4090))
    assert not any("fastest" in f.badges and f.model.params_b < 1 for f in big)
    slow = machine(8, 20, cores=4)
    picks = catalog.pick_shortlist(catalog.rank_models(slow))
    rec = next(f for f in picks if "recommended" in f.badges)
    fastest = next((f for f in picks if "fastest" in f.badges), None)
    assert fastest is not None and (fastest is rec or fastest.est_tokens_per_s > rec.est_tokens_per_s)


def test_smartest_is_never_a_snug_squeeze_on_the_processor():
    for spec in (machine(16, 40), machine(8, 20, cores=4), dataclasses.replace(machine(16, 40), os_name="Windows")):
        for f in catalog.pick_shortlist(catalog.rank_models(spec)):
            if "smartest" in f.badges:
                assert not (f.verdict == "tight" and f.placement == "cpu")


def test_dated_refreshes_dont_take_two_menu_slots():
    base = catalog.get_model("qwen3-30b-a3b")
    refresh = dataclasses.replace(base, key="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                                  hf_repo="unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF",
                                  base_model="Qwen/Qwen3-30B-A3B-Instruct-2507", display_name="Qwen3 30B-A3B Instruct 2507",
                                  source="huggingface", reasoning=False)
    assert catalog._variant_family(base) == catalog._variant_family(refresh)
    assert catalog._variant_family(catalog.get_model("qwen3-4b")) != catalog._variant_family(base)
    server = machine(128, 40, cores=32)
    picks = catalog.pick_shortlist(catalog.rank_models(server, list(catalog.MODEL_CATALOG) + [refresh]))
    families = [catalog._variant_family(f.model) for f in picks]
    assert len(families) == len(set(families))


def test_a_model_already_downloaded_needs_no_disk_space():
    nearly_full = machine(32, 40, disk=3.0)
    qwen = catalog.get_model("qwen3-4b")
    without = catalog.evaluate_fit(nearly_full, qwen)
    assert without.quant != "Q4_K_M" or without.verdict == "no"  # 2.5 GB + 1 GB spare doesn't fit on disk

    def have(model, quant):
        return model.key == "qwen3-4b" and quant == "Q4_K_M"

    fit = catalog.evaluate_fit(nearly_full, qwen, downloaded=have)
    assert fit.quant == "Q4_K_M" and fit.verdict != "no"
    assert "already downloaded" in fit.reason
    # ...and it's preferred even where a bigger version would fit: no surprise new download.
    roomy = machine(32, 40)
    assert catalog.evaluate_fit(roomy, qwen, downloaded=have).quant == "Q4_K_M"


def test_small_machines_get_a_shorter_context_instead_of_nothing():
    four_gb = machine(3.7, 15, cores=4)
    fit = catalog.evaluate_fit(four_gb, catalog.get_model("qwen3-0.6b"))
    assert fit.verdict != "no" and fit.context_tokens == catalog.MIN_CONTEXT_TOKENS
    assert "shorter conversation memory" in fit.reason
    assert catalog.recommend(four_gb) is not None
    # The usual context still wins wherever it fits.
    assert catalog.evaluate_fit(machine(16, 40), catalog.get_model("qwen3-0.6b")).context_tokens is None


def test_short_quant_spellings_are_understood():
    assert catalog.quant_bits("Q4") == catalog.quant_bits("Q4_K_M")
    assert catalog.quant_bits("Q4_K") == catalog.quant_bits("Q4_K_M")
    assert catalog.quant_bits("Q2_K_S") is not None


# ---------------------------------------------------------------------------
# Round 3: models that always think
# ---------------------------------------------------------------------------


def _always_thinker(base_key: str, *, repo: str, name: str) -> ModelEntry:
    seed = catalog.get_model(base_key)
    return dataclasses.replace(seed, key=repo, hf_repo=repo, display_name=name, source="huggingface",
                               base_model=repo.split("/")[1].removesuffix("-GGUF"), thinking="always",
                               reasoning=True, downloads=900_000)


RTX3060 = GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0, bandwidth_gbs=360.0)


def test_always_thinking_turns_count_the_thinking():
    plain = catalog.turn_tokens_per_s(30.0, "gpu")
    thinking = catalog.turn_tokens_per_s(30.0, "gpu", thinking_tokens=catalog.ALWAYS_THINKING_TOKENS)
    assert thinking < plain / 4


def test_always_thinkers_are_penalised_not_rewarded():
    qwen = catalog.get_model("qwen3-8b")
    r1 = dataclasses.replace(qwen, thinking="always")
    kwargs = dict(quant="Q4_K_M", verdict="great", tokens_per_s=40.0, ratio=0.3, download_gb=5.0, placement="gpu")
    assert catalog.score_fit(r1, **kwargs) < catalog.score_fit(qwen, **kwargs) - catalog.PENALTY_ALWAYS_THINKING


def test_always_thinkers_never_reach_the_short_menu():
    specs = machine(16, 40, (RTX3060,))
    r1 = _always_thinker("qwen3-8b", repo="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", name="DeepSeek R1 0528 Qwen3 8B")
    phi = _always_thinker("qwen3-14b", repo="unsloth/Phi-4-reasoning-plus-GGUF", name="Phi-4 reasoning plus")
    ranked = catalog.rank_models(specs, list(catalog.MODEL_CATALOG) + [r1, phi])
    menu = catalog.pick_shortlist(ranked)
    assert all(not catalog.always_thinks(f.model) for f in menu)
    assert any(catalog.always_thinks(f.model) for f in ranked)  # still in the full list ("more")
    fit = next(f for f in ranked if f.model is r1)
    assert "always thinks" in fit.reason


def test_a_thinking_twin_cant_take_its_instruct_twins_slot():
    specs = machine(16, 40, (RTX3060,))
    instruct = dataclasses.replace(catalog.get_model("qwen3-4b"), key="unsloth/Qwen3-4B-Instruct-2507-GGUF",
                                   hf_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF", base_model="Qwen/Qwen3-4B-Instruct-2507",
                                   source="huggingface", thinking="none", reasoning=False)
    thinking = _always_thinker("qwen3-4b", repo="unsloth/Qwen3-4B-Thinking-2507-GGUF", name="Qwen3 4B Thinking 2507")
    menu = catalog.pick_shortlist(catalog.rank_models(specs, [instruct, thinking]))
    assert [f.model.hf_repo for f in menu] == ["unsloth/Qwen3-4B-Instruct-2507-GGUF"]


def test_explain_fit_says_why_an_always_thinker_is_slow():
    specs = machine(16, 40, (RTX3060,))
    r1 = _always_thinker("qwen3-8b", repo="unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF", name="DeepSeek R1 0528 Qwen3 8B")
    text = catalog.explain_fit(specs, catalog.evaluate_fit(specs, r1))
    assert "thinking it always does first" in text


# ---------------------------------------------------------------------------
# Round 3: Apple overflow, placement-aware ladder, small machines, badges
# ---------------------------------------------------------------------------

def _gpu_bw(name: str, vendor: str, vram: float, bandwidth: float) -> GPUInfo:
    return GPUInfo(name=name, vendor=vendor, vram_gb=vram, bandwidth_gbs=bandwidth)


M2_PRO_16 = machine(16, 60, (_gpu_bw("Apple M2 Pro GPU", "apple", 11.2, 200.0),), unified=True)
M3_MAX_64 = machine(64, 80, (_gpu_bw("Apple M3 Max GPU", "apple", 48.0, 400.0),), unified=True)
M1_8 = machine(8, 50, (_gpu_bw("Apple M1 GPU", "apple", 5.6, 68.0),), unified=True)
GTX_1650 = machine(16, 30, (_gpu_bw("NVIDIA GeForce GTX 1650", "nvidia", 4.0, 128.0),))
CHROMEBOOK = dataclasses.replace(machine(3.8, 10, cores=2), cpu_flags=["sse4_2"])


def _hub(key: str, name: str, params: float, quants: tuple[tuple[str, float], ...], **kw) -> ModelEntry:
    return make_entry(key=key, hf_repo=key, display_name=name, params_b=params, quant=quants[0][0],
                      file_size_gb=quants[0][1], quant_options=quants, **kw)


def test_a_mac_never_gets_a_pretend_cpu_plan():
    dense = _hub("someone/Dense-14B-Instruct-GGUF", "Dense 14B", 14.0, (("Q4_K_M", 10.5), ("IQ4_XS", 7.6)))
    for mac in (M2_PRO_16, M3_MAX_64, M1_8):
        for model in list(catalog.MODEL_CATALOG) + [dense]:
            fit = catalog.evaluate_fit(mac, model)
            assert fit.placement != "cpu", (model.key, fit.placement)


def test_a_mac_prefers_a_version_that_fits_on_its_gpu_over_a_bigger_spilling_one():
    dense = _hub("someone/Dense-14B-Instruct-GGUF", "Dense 14B", 14.0,
                 (("Q4_K_L", 10.5), ("Q4_K_M", 9.0), ("IQ4_XS", 7.9)))
    fit = catalog.evaluate_fit(M2_PRO_16, dense)
    assert fit.placement == "unified", (fit.quant, fit.placement)


def test_overflow_on_a_mac_is_a_snug_split_with_honest_words():
    gpt = catalog.get_model("gpt-oss-20b")
    fit = catalog.evaluate_fit(M2_PRO_16, gpt)
    assert fit.placement == "partial" and fit.verdict == "tight"
    assert "runs on the GPU and the rest on the processor" in fit.reason
    unified_speed = perf.estimate_tokens_per_s(M2_PRO_16, active_gb=1.0, placement="unified")
    cpu_speed = perf.estimate_tokens_per_s(M2_PRO_16, active_gb=1.0, placement="cpu")
    assert cpu_speed < unified_speed


def test_a_split_is_never_labelled_a_great_fit():
    for model in catalog.MODEL_CATALOG:
        fit = catalog.evaluate_fit(GTX_1650, model)
        if fit.placement == "partial":
            assert fit.verdict in ("ok", "tight", "no"), model.key


def test_a_gpu_split_beats_a_slower_cpu_plan_for_half_a_percent_of_quality():
    nemo = _hub("someone/Mistral-Nemo-Instruct-2407-GGUF", "Mistral Nemo 12B", 12.2,
                (("Q8_0", 13.0), ("Q6_K", 10.1), ("Q5_K_M", 8.7), ("Q4_K_L", 7.98), ("Q4_K_M", 7.48),
                 ("IQ4_XS", 6.74), ("Q3_K_M", 6.08)))
    fit = catalog.evaluate_fit(GTX_1650, nemo)
    assert fit.placement == "partial", (fit.quant, fit.placement, fit.est_tokens_per_s)


def test_all_on_the_gpu_beats_a_split_for_a_marginally_better_quant():
    rtx = machine(64, 50, (gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0),), cores=16)
    seed = _hub("unsloth/Seed-OSS-36B-Instruct-GGUF", "Seed-OSS 36B", 36.2,
                (("Q8_0", 38.4), ("UD-Q4_K_XL", 22.9), ("Q4_K_M", 21.8), ("IQ4_XS", 19.6)))
    fit = catalog.evaluate_fit(rtx, seed)
    assert fit.placement == "gpu" and fit.quant != "UD-Q4_K_XL", (fit.quant, fit.placement)


def test_q8_0_beats_the_bigger_ud_q8_k_xl():
    qwen = _hub("unsloth/Qwen3-8B-GGUF", "Qwen3 8B", 8.19,
                (("UD-Q8_K_XL", 10.8), ("Q8_0", 8.71), ("UD-Q6_K_XL", 7.5), ("Q6_K", 6.73), ("Q4_K_M", 5.03)))
    rtx = machine(64, 50, (gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0),), cores=16)
    assert catalog.evaluate_fit(rtx, qwen).quant == "Q8_0"


def test_a_tiny_machine_gets_4_bits_at_a_shorter_context_not_2_bits():
    qwen = _hub("unsloth/Qwen3-0.6B-GGUF", "Qwen3 0.6B", 0.6,
                (("Q8_0", 0.64), ("UD-Q4_K_XL", 0.40), ("Q4_K_M", 0.40), ("UD-IQ2_M", 0.28), ("UD-IQ1_S", 0.22)),
                architecture="qwen3", native_context=32768)
    fit = catalog.evaluate_fit(CHROMEBOOK, qwen)
    assert fit.verdict != "no"
    assert catalog.quant_bits(fit.quant) >= 3.3, fit.quant
    assert fit.context_tokens == catalog.MIN_CONTEXT_TOKENS


def test_sub_2_bit_quants_are_never_suggested():
    qwen = _hub("unsloth/Qwen3-1.7B-GGUF", "Qwen3 1.7B", 1.72, (("UD-IQ1_S", 0.56), ("IQ2_M", 0.75)))
    fit = catalog.evaluate_fit(CHROMEBOOK, qwen)
    assert fit.verdict == "no" and "gibberish" in fit.reason
    # ...but a player who asks for that exact quant can still have it judged.
    assert catalog.evaluate_fit(LAPTOP_8GB, qwen, quant_floor=False).verdict != "no"


def test_fastest_goes_to_a_fast_moe_on_a_big_gpu():
    rtx = machine(64, 50, (gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0),), cores=16)
    picks = catalog.pick_shortlist(catalog.rank_models(rtx))
    fastest = next((f for f in picks if "fastest" in f.badges), None)
    assert fastest is not None and fastest.model.active_params_b


def test_fastest_is_never_a_toy_model():
    tiny_llama = make_entry(key="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", hf_repo="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                            display_name="TinyLlama 1.1B Chat", family="TinyLlama", params_b=1.1,
                            quant_options=(("Q4_K_M", 0.67), ("Q8_0", 1.17)), native_context=2048, downloads=900_000)
    for specs in (GTX_1650, M1_8):
        picks = catalog.pick_shortlist(catalog.rank_models(specs, list(catalog.MODEL_CATALOG) + [tiny_llama]))
        for f in picks:
            if "fastest" in f.badges and "recommended" not in f.badges:
                assert catalog._effective_params_b(f.model) >= catalog.FASTEST_MIN_EFFECTIVE_B, f.model.key


def test_fine_tunes_of_one_base_dont_crowd_the_menu():
    rtx = machine(64, 50, (gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0),), cores=16)
    base = catalog.get_model("mistral-small-3.2-24b")
    tunes = [dataclasses.replace(base, key=f"{who}/{name}-GGUF", hf_repo=f"{who}/{name}-GGUF", display_name=name,
                                 family=who, base_model=f"{who}/{name}", source="huggingface", downloads=500_000)
             for who, name in (("TheDrummer", "Cydonia-24B-v4.1"), ("mistralai", "Magistral-Small-2509"))]
    picks = catalog.pick_shortlist(catalog.rank_models(rtx, list(catalog.MODEL_CATALOG) + tunes))
    same_base = [f for f in picks if f.model.params_b == base.params_b and f.model.architecture == base.architecture]
    assert len(same_base) == 1  # one Mistral-Small-24B lineage; other base models fill the rest


def test_macs_read_prompts_about_ten_times_faster_than_they_write():
    assert catalog.PREFILL_SPEEDUP["unified"] == pytest.approx(10.0)
    assert catalog.turn_tokens_per_s(20.0, "unified") == pytest.approx(20.0 * 250 / (250 + 1200 / 10))


# ---------------------------------------------------------------------------
# Round 4: Smartest on Macs, splits, KV shapes, seeds and their successors
# ---------------------------------------------------------------------------

RTX_2060_16 = machine(16, 40, (gpu("NVIDIA GeForce RTX 2060", "nvidia", 6.0),))
RTX_4090_64 = machine(64, 80, (_gpu_bw("NVIDIA GeForce RTX 4090", "nvidia", 23.99, 1008.0),), cores=16)
MISTRAL_SMALL_HUB = _hub("unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF", "Mistral Small 3.2 24B", 24.0,
                         (("UD-Q4_K_XL", 14.5), ("Q4_K_M", 14.3), ("IQ4_XS", 12.8), ("UD-Q3_K_XL", 12.0),
                          ("UD-IQ3_XXS", 9.6), ("Q2_K", 8.9)), architecture="llama")
GRANITE_SMALL_HUB = _hub("unsloth/granite-4.0-h-small-GGUF", "granite-4.0-h-small", 32.2,
                         (("Q4_K_M", 19.5), ("IQ4_XS", 17.5), ("Q3_K_M", 15.6), ("UD-IQ3_XXS", 12.9),
                          ("Q2_K", 11.9)), active_params_b=9.0, architecture="granitehybrid")


def test_smartest_is_never_a_squeeze_of_system_ram_or_a_last_resort_quant():
    models = list(catalog.MODEL_CATALOG) + [MISTRAL_SMALL_HUB, GRANITE_SMALL_HUB]
    for specs in (M2_PRO_16, M3_MAX_64, M1_8, RTX_2060_16, GTX_1650, LAPTOP_8GB, DESKTOP_16GB, RTX_3060):
        for f in catalog.pick_shortlist(catalog.rank_models(specs, models)):
            if "smartest" in f.badges:
                assert not (f.verdict == "tight" and f.placement in ("cpu", "unified", "partial")), \
                    (specs.gpus, f.model.key, f.placement, f.verdict)
                assert catalog._quant_tier(f.quant) != "last", (specs.gpus, f.model.key, f.quant)


def test_an_apple_split_explains_its_budget_as_the_macs_own_memory():
    gpt = catalog.get_model("gpt-oss-20b")
    fit = catalog.evaluate_fit(M2_PRO_16, gpt)
    assert fit.placement == "partial" and fit.shares_system_ram
    text = catalog.explain_fit(M2_PRO_16, fit)
    assert "of your Mac's memory" in text and "keeping 2.5 GB for macOS" in text
    assert "video memory plus spare RAM" not in text


def test_a_small_spill_into_plenty_of_ram_is_a_good_fit_and_can_be_recommended():
    """16 GB + a 4 GB GTX 1650: Qwen3 4B keeps ~84% on the card and spills ~0.6 GB
    into ~13 GB of free RAM - that can't fail, so it isn't "snug", and it beats
    the much smaller Qwen3 1.7B."""
    fit = catalog.evaluate_fit(GTX_1650, catalog.get_model("qwen3-4b"))
    assert fit.placement == "partial" and fit.verdict == "ok" and fit.gpu_share > 0.75
    assert "snug" not in fit.reason and "of the 14 GB we can spare" in fit.reason
    rec = catalog.recommend(GTX_1650)
    assert rec is not None and rec.model.key == "qwen3-4b"


def test_a_split_with_little_on_the_gpu_is_never_the_first_choice():
    rtx4060 = machine(32, 50, (gpu("NVIDIA GeForce RTX 4060", "nvidia", 8.0),))
    moe = catalog.evaluate_fit(rtx4060, catalog.get_model("qwen3-30b-a3b"))
    assert moe.placement == "partial" and moe.gpu_share < catalog.SPLIT_RECOMMENDED_MIN_GPU_SHARE
    assert catalog.recommend(rtx4060).model.key != "qwen3-30b-a3b"


def test_a_model_the_processor_would_squeeze_is_split_with_the_graphics_card_instead():
    """64 GB + RTX 4090: gpt-oss-120b holds only ~38% on the card - below the usual
    40% - but the processor-only plan would fill the RAM, so it's split (as the
    engine would run it anyway)."""
    gpt120 = catalog.get_model("gpt-oss-120b") or _hub(
        "unsloth/gpt-oss-120b-GGUF", "gpt-oss 120b", 116.8, (("UD-Q4_K_XL", 63.4),), active_params_b=5.1)
    fit = catalog.evaluate_fit(RTX_4090_64, gpt120)
    assert fit.placement == "partial", (fit.placement, fit.reason)
    glm = _hub("unsloth/GLM-4.5-Air-GGUF", "GLM-4.5 Air", 106.0, (("Q4_K_M", 66.0), ("IQ4_XS", 60.2)),
               active_params_b=12.0, architecture="glm4moe")
    glm_fit = catalog.evaluate_fit(RTX_4090_64, glm)
    assert glm_fit.verdict != "no" and glm_fit.quant == "Q4_K_M"
    # ...and a "won't fit" never says it needs less than what you have to spare.
    squeezed = machine(60, 80, (_gpu_bw("NVIDIA GeForce RTX 4090", "nvidia", 23.99, 1008.0),), cores=16)
    fit2 = catalog.evaluate_fit(squeezed, gpt120)
    assert fit2.verdict != "no" or "you have about" not in fit2.reason


def test_kv_cache_is_right_for_models_without_grouped_query_attention():
    phi3 = make_entry(key="microsoft/Phi-3-mini-4k-instruct-gguf", hf_repo="microsoft/Phi-3-mini-4k-instruct-gguf",
                      display_name="Phi-3 mini 4k Instruct", params_b=3.82, architecture="phi3")
    olmo13 = make_entry(key="allenai/OLMo-2-1124-13B-Instruct-GGUF", hf_repo="allenai/OLMo-2-1124-13B-Instruct-GGUF",
                        display_name="OLMo-2 13B Instruct", params_b=13.7, architecture="olmo2")
    assert catalog.kv_cache_gb(phi3, 4096) == pytest.approx(2 * 32 * 32 * 96 * 2 * 4096 / 1e9)  # ~1.61 GB
    assert catalog.kv_cache_gb(olmo13, 4096) == pytest.approx(3.36, rel=0.01)
    # An unknown family: the shape read from its GGUF header wins over the rule of thumb.
    header_shape = make_entry(kv_shape=(40, 40, 128))
    assert catalog.kv_cache_gb(header_shape, 4096) == pytest.approx(3.36, rel=0.01)
    # No header, but an architecture known to lack grouped-query attention: estimated for that.
    no_gqa = make_entry(params_b=7.3, architecture="olmo2")
    assert catalog.kv_cache_gb(no_gqa, 4096) > 3 * catalog.kv_cache_gb(make_entry(params_b=7.3), 4096)
    fit = catalog.evaluate_fit(RTX_3060, dataclasses.replace(olmo13, quant_options=(("Q4_K_M", 8.95),),
                                                             file_size_gb=8.95))
    assert fit.placement != "gpu" or fit.verdict != "ok"  # ~11.8 GiB against an ~11.2 GiB card: no clean fit


def test_a_seeds_newer_release_from_the_same_publisher_isnt_outranked_by_the_frozen_seed():
    seed = catalog.get_model("qwen3-4b")
    successor = dataclasses.replace(
        seed, key="unsloth/Qwen3-4B-Instruct-2507-GGUF", hf_repo="unsloth/Qwen3-4B-Instruct-2507-GGUF",
        display_name="Qwen3 4B Instruct 2507", base_model="Qwen/Qwen3-4B-Instruct-2507", source="huggingface",
        reasoning=False, thinking="none", downloads=seed.downloads * 3 + 100_000)
    assert catalog._earns_curated_bonus(successor)
    stranger = dataclasses.replace(successor, key="someone/Qwen3-4B-Instruct-2507-GGUF",
                                   hf_repo="someone/Qwen3-4B-Instruct-2507-GGUF")
    assert not catalog._earns_curated_bonus(stranger)
    rec = catalog.recommend(M1_8, list(catalog.MODEL_CATALOG) + [successor])
    assert rec is not None and rec.model.key == successor.key


def test_fastest_near_ties_go_to_the_better_pick():
    a = dataclasses.replace(catalog.evaluate_fit(RTX_4090, catalog.get_model("qwen3-8b")), score=60.0,
                            est_tokens_per_s=100.0)
    b = dataclasses.replace(a, score=50.0, est_tokens_per_s=102.0)
    c = dataclasses.replace(a, score=70.0, est_tokens_per_s=80.0)
    assert catalog._fastest_order([b, a, c])[:2] == [a, b]


def test_an_unrecognised_fine_tune_never_takes_the_original_models_slot():
    base = catalog.get_model("qwen3-4b")
    jan = dataclasses.replace(base, key="Menlo/Jan-nano-gguf", hf_repo="Menlo/Jan-nano-gguf",
                              display_name="Jan nano", family="Jan", base_model="Menlo/Jan-nano",
                              source="huggingface", architecture=base.architecture or "qwen3",
                              downloads=base.downloads * 5)
    fits = catalog.rank_models(LAPTOP_8GB, list(catalog.MODEL_CATALOG) + [jan])
    picks = catalog.pick_shortlist(fits)
    keys = [f.model.key for f in picks]
    if "Menlo/Jan-nano-gguf" in keys and "qwen3-4b" in keys:
        assert keys.index("qwen3-4b") < keys.index("Menlo/Jan-nano-gguf")
    assert "qwen3-4b" in keys or all(f.model.key != "Menlo/Jan-nano-gguf" for f in picks)
