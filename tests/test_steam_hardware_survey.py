"""Steam Hardware Survey machines: one story pick and one referee per tier.

The rows are the common cards and RAM sizes on Steam, plus the cases that
have gone wrong before: memory already in use, a laptop with both a discrete
card and a built-in chip, and a Steam Deck APU whose "VRAM" is shared RAM.
Each row's expected pick is locked. The checks under the pick are the same
for every machine: the download and the memory need fit inside the budget,
the speed label matches the number, a recommendation is not a crawl, and a
local referee's wait matches where it actually runs.
"""

from __future__ import annotations

import io

import pytest
from rich.console import Console

from gettowork import catalog, perf
from gettowork.config import Settings
from gettowork.hf_discovery import DiscoveryResult
from gettowork.onboarding import run_jev_onboarding
from gettowork.setup_flow import ModelSearch, SetupServices, _SetupFlow
from gettowork.system_one import clef_client_timeout
from gettowork.types import GPUInfo, SystemSpecs
from gettowork.ui import UI

# story key, quant, verdict, placement, speed label
# referee key, quant, verdict, placement (None when local Clef does not fit)
# primary GPU name fragment, or None when the plan is CPU / shared memory
Expect = tuple


def _gpu(name: str, vendor: str, vram: float, used: float = 0.0) -> GPUInfo:
    info = GPUInfo(name=name, vendor=vendor, vram_gb=vram, vram_used_gb=used)
    info.bandwidth_gbs = perf.estimate_gpu_bandwidth(info)
    return info


def _machine(
    os_name: str,
    ram: float,
    cores: int,
    ram_bw: float,
    gpus: tuple[GPUInfo, ...] = (),
    *,
    unified: bool = False,
    vulkan: bool = False,
    os_version: str = "",
    arch: str = "",
) -> SystemSpecs:
    if not arch:
        arch = "arm64" if unified or os_name == "Darwin" else "x86_64"
    flags = ["neon"] if arch == "arm64" else ["avx2", "fma"]
    if vulkan or any(gpu.vendor in ("amd", "intel") and gpu.vram_gb > 0 for gpu in gpus):
        flags = [*flags, "vulkan"]
    if os_name == "Windows" and not os_version:
        os_version = "10"
    return SystemSpecs(
        os_name=os_name,
        os_version=os_version,
        arch=arch,
        cpu_name="Survey CPU",
        cpu_cores_physical=cores,
        cpu_cores_logical=cores,
        ram_total_gb=ram,
        ram_available_gb=ram * 0.6,
        disk_free_gb=200.0,
        gpus=list(gpus),
        unified_memory=unified,
        ram_bandwidth_gbs=ram_bw,
        cpu_flags=flags,
    )


def _row(name, specs, expect):
    return name, specs, expect


# Expected picks, reviewed against the fit engine. A missing referee means the
# menu says Clef does not fit and the story model decides. When both Clef
# files only fit on the CPU, the faster file is the pick.
SURVEY = [
    _row("gtx1650_w10", _machine("Windows", 16, 6, 40, (_gpu("NVIDIA GeForce GTX 1650", "nvidia", 4),)),
         ("qwen3-4b", "Q4_K_M", "ok", "partial", "fast", "clef-flash", "Q4_K_M", "ok", "cpu", "GTX 1650")),
    _row("rtx2060_w10", _machine("Windows", 16, 6, 40, (_gpu("NVIDIA GeForce RTX 2060", "nvidia", 6),)),
         ("qwen3-8b", "Q4_K_M", "ok", "partial", "fast", None, None, None, None, "RTX 2060")),
    _row("rtx3050_6_w11", _machine("Windows", 16, 6, 40, (_gpu("NVIDIA GeForce RTX 3050", "nvidia", 6),), os_version="11"),
         ("qwen3-8b", "Q4_K_M", "ok", "partial", "usable", None, None, None, None, "RTX 3050")),
    _row("rtx3060ti_w11", _machine("Windows", 16, 8, 45, (_gpu("NVIDIA GeForce RTX 3060 Ti", "nvidia", 8),), os_version="11"),
         ("qwen3-8b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3060 Ti")),
    _row("rtx4060_w11", _machine("Windows", 16, 6, 45, (_gpu("NVIDIA GeForce RTX 4060", "nvidia", 8),), os_version="11"),
         ("qwen3-8b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 4060")),
    _row("rtx4060_laptop_w11", _machine("Windows", 16, 8, 50, (_gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8),), os_version="11"),
         ("qwen3-8b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "4060 Laptop")),
    _row("rtx3060_12_w11", _machine("Windows", 32, 8, 45, (_gpu("NVIDIA GeForce RTX 3060", "nvidia", 12),), os_version="11"),
         ("qwen3-14b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3060")),
    _row("rtx3060_12_linux", _machine("Linux", 32, 8, 45, (_gpu("NVIDIA GeForce RTX 3060", "nvidia", 12),)),
         ("qwen3-14b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3060")),
    _row("rtx4070_w11", _machine("Windows", 32, 8, 50, (_gpu("NVIDIA GeForce RTX 4070", "nvidia", 12),), os_version="11"),
         ("qwen3-14b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 4070")),
    _row("rtx4060ti_16_w11", _machine("Windows", 32, 8, 50, (_gpu("NVIDIA GeForce RTX 4060 Ti", "nvidia", 16),), os_version="11"),
         ("qwen3-14b", "Q4_K_M", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 4060 Ti")),
    _row("rtx4080_w11", _machine("Windows", 32, 8, 50, (_gpu("NVIDIA GeForce RTX 4080", "nvidia", 16),), os_version="11"),
         ("qwen3-14b", "Q6_K", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 4080")),
    _row("rtx3090_w10", _machine("Windows", 64, 12, 50, (_gpu("NVIDIA GeForce RTX 3090", "nvidia", 24),)),
         ("qwen3-32b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3090")),
    _row("rtx4090_w11", _machine("Windows", 64, 16, 50, (_gpu("NVIDIA GeForce RTX 4090", "nvidia", 24),), os_version="11"),
         ("qwen3-32b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 4090")),
    _row("rtx5090_w11", _machine("Windows", 64, 16, 50, (_gpu("NVIDIA GeForce RTX 5090", "nvidia", 32),), os_version="11"),
         ("qwen3-32b", "Q5_K_M", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "ok", "partial", "RTX 5090")),
    _row("dual_rtx3060_w11", _machine("Windows", 32, 8, 45, (
        _gpu("NVIDIA GeForce RTX 3060", "nvidia", 12), _gpu("NVIDIA GeForce RTX 3060", "nvidia", 12),
    ), os_version="11"), ("qwen3-32b", "IQ4_XS", "ok", "gpu", "usable", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3060")),
    _row("rx6600_w11", _machine("Windows", 16, 6, 40, (_gpu("AMD Radeon RX 6600", "amd", 8),), os_version="11"),
         ("qwen3-8b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RX 6600")),
    _row("rx7800xt_linux", _machine("Linux", 32, 8, 50, (_gpu("AMD Radeon RX 7800 XT", "amd", 16),)),
         ("qwen3-14b", "Q6_K", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RX 7800 XT")),
    _row("arc_a770_w11", _machine("Windows", 32, 8, 45, (_gpu("Intel Arc A770", "intel", 16),), os_version="11"),
         ("qwen3-14b", "Q6_K", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "Arc A770")),
    _row("iris_xe_w11", _machine("Windows", 16, 4, 40, (_gpu("Intel Iris Xe Graphics", "intel", 0),), os_version="11"),
         ("qwen3-1.7b", "Q4_K_M", "great", "cpu", "fast", "clef-flash", "Q4_K_M", "ok", "cpu", None)),
    _row("amd_780m_w11", _machine("Windows", 16, 8, 70, (_gpu("AMD Radeon 780M", "amd", 0),), os_version="11"),
         ("qwen3-4b", "Q4_K_M", "great", "cpu", "usable", "clef-flash", "Q4_K_M", "ok", "cpu", None)),
    _row("cpu_8_w10", _machine("Windows", 8, 4, 25),
         ("qwen3-1.7b", "Q4_K_M", "great", "cpu", "usable", None, None, None, None, None)),
    _row("cpu_16_w11", _machine("Windows", 16, 6, 40, os_version="11"),
         ("qwen3-1.7b", "Q4_K_M", "great", "cpu", "fast", "clef-flash", "Q4_K_M", "ok", "cpu", None)),
    _row("cpu_32_linux", _machine("Linux", 32, 8, 45),
         ("qwen3-1.7b", "Q5_K_M", "great", "cpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", None)),
    _row("optimus_4060", _machine("Windows", 16, 8, 50, (
        _gpu("Intel Iris Xe Graphics", "intel", 0), _gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8),
    ), os_version="11"), ("qwen3-8b", "IQ4_XS", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "4060 Laptop")),
    _row("optimus_1650_igpu_reports_16gb", _machine("Windows", 16, 6, 40, (
        _gpu("Intel Iris Xe Graphics", "intel", 16), _gpu("NVIDIA GeForce GTX 1650", "nvidia", 4),
    ), os_version="11"), ("qwen3-4b", "Q4_K_M", "ok", "partial", "fast", "clef-flash", "Q4_K_M", "ok", "cpu", "GTX 1650")),
    _row("steam_deck", _machine("Linux", 16, 4, 88, (_gpu("AMD Custom GPU 0405", "amd", 8),), vulkan=True),
         ("qwen3-4b", "Q4_K_M", "great", "cpu", "fast", "clef-flash", "Q4_K_M", "ok", "cpu", None)),
    _row("mac_m2_16", _machine("Darwin", 16, 8, 100, (_gpu("Apple M2 GPU", "apple", 11.2),), unified=True),
         ("mistral-7b", "Q4_K_M", "great", "unified", "usable", "clef-flash", "Q4_K_M", "tight", "partial", "Apple M2")),
    _row("mac_m2_pro_16", _machine("Darwin", 16, 10, 200, (_gpu("Apple M2 Pro GPU", "apple", 11.2),), unified=True),
         ("qwen3-14b", "IQ4_XS", "ok", "unified", "usable", None, None, None, None, "M2 Pro")),
    _row("rtx3060_12_2p5gb_in_use", _machine("Windows", 32, 8, 45, (_gpu("NVIDIA GeForce RTX 3060", "nvidia", 12, 2.5),), os_version="11"),
         ("qwen3-8b", "Q5_K_M", "ok", "gpu", "fast", "clef-flash", "Q4_K_M", "great", "cpu", "RTX 3060")),
    _row("rtx4060_2gb_in_use", _machine("Windows", 16, 6, 45, (_gpu("NVIDIA GeForce RTX 4060", "nvidia", 8, 2.0),), os_version="11"),
         ("qwen3-8b", "Q4_K_M", "ok", "partial", "fast", None, None, None, None, "RTX 4060")),
    _row("gtx1650_1p5gb_in_use", _machine("Windows", 16, 6, 40, (_gpu("NVIDIA GeForce GTX 1650", "nvidia", 4, 1.5),), os_version="11"),
         ("qwen3-4b", "Q4_K_M", "ok", "partial", "usable", "clef-flash", "Q4_K_M", "ok", "cpu", "GTX 1650")),
    # Unified memory. The GPU figure is the share of RAM detection would
    # record (65% at 8 GB, 70% through 48 GB, 75% at 64 GB, 80% at 128 GB),
    # not the whole stick and not a second VRAM pool.
    _row("mac_m1_8", _machine("Darwin", 8, 8, 68, (_gpu("Apple M1 GPU", "apple", 5.2),), unified=True),
         ("qwen3-4b", "Q4_K_M", "ok", "unified", "usable", None, None, None, None, "Apple M1")),
    _row("mac_m4_24", _machine("Darwin", 24, 10, 120, (_gpu("Apple M4 GPU", "apple", 16.8),), unified=True),
         ("gpt-oss-20b", "MXFP4", "ok", "unified", "fast", "clef-flash", "Q4_K_M", "tight", "partial", "Apple M4 GPU")),
    _row("mac_m4_pro_36", _machine("Darwin", 36, 12, 273, (_gpu("Apple M4 Pro GPU", "apple", 25.2),), unified=True),
         ("qwen3-14b", "Q4_K_M", "great", "unified", "fast", "clef-flash", "Q8_0", "ok", "unified", "M4 Pro")),
    _row("mac_m4_max_64", _machine("Darwin", 64, 14, 546, (_gpu("Apple M4 Max GPU", "apple", 48.0),), unified=True),
         ("qwen3-32b", "Q4_K_M", "great", "unified", "fast", "clef", "Q4_K_M", "ok", "unified", "M4 Max")),
    _row("mac_m3_ultra_128", _machine("Darwin", 128, 16, 819, (_gpu("Apple M3 Ultra GPU", "apple", 102.4),), unified=True),
         ("qwen3-32b", "Q6_K", "great", "unified", "fast", "clef", "Q8_0", "great", "unified", "M3 Ultra")),
    _row("rtx_spark_32_w11", _machine("Windows", 32, 18, 200, (_gpu("NVIDIA RTX Spark", "nvidia", 22.4),), unified=True, os_version="11"),
         ("qwen3-14b", "Q5_K_M", "great", "unified", "fast", "clef-flash", "Q4_K_M", "tight", "unified", "RTX Spark")),
    _row("rtx_spark_64_w11", _machine("Windows", 64, 20, 250, (_gpu("NVIDIA RTX Spark", "nvidia", 48.0),), unified=True, os_version="11"),
         ("qwen3-14b", "Q5_K_M", "great", "unified", "fast", "clef", "Q4_K_M", "great", "unified", "RTX Spark")),
    _row("rtx_spark_128_w11", _machine("Windows", 128, 20, 300, (_gpu("NVIDIA RTX Spark", "nvidia", 102.4),), unified=True, os_version="11"),
         ("qwen3.8-27b", "Q4_K_M", "great", "unified", "usable", "clef", "Q4_K_M", "great", "unified", "RTX Spark")),
    _row("dgx_spark_128", _machine("Linux", 128, 20, 300, (_gpu("NVIDIA GB10", "nvidia", 102.4),), unified=True, vulkan=True),
         ("qwen3.8-27b", "Q4_K_M", "great", "unified", "usable", "clef", "Q4_K_M", "great", "unified", "GB10")),
    _row("strix_halo_64_w11", _machine("Windows", 64, 16, 150, (_gpu("AMD Radeon 8060S Graphics", "amd", 48.0),), unified=True, os_version="11", arch="x86_64"),
         ("qwen3-14b", "Q4_K_M", "great", "unified", "fast", "clef", "Q4_K_M", "great", "unified", "8060S")),
    _row("strix_halo_128", _machine("Linux", 128, 16, 200, (_gpu("AMD Radeon 8060S Graphics", "amd", 102.4),), unified=True, arch="x86_64"),
         ("qwen3-14b", "Q4_K_M", "great", "unified", "fast", "clef", "Q4_K_M", "great", "unified", "8060S")),
    _row("lunar_lake_32_w11", _machine("Windows", 32, 8, 80, (_gpu("Intel Arc 140V GPU", "intel", 22.4),), unified=True, os_version="11", arch="x86_64"),
         ("qwen3-30b-a3b", "Q4_K_M", "ok", "unified", "fast", None, None, None, None, "140V")),
    _row("below_minimum_2gb", _machine("Windows", 2, 2, 12),
         (None, None, None, None, None, None, None, None, None, None)),
]


def _fits(specs: SystemSpecs, fit) -> None:
    plan = catalog._plan(specs, fit.model, fit.quant, fit.download_gb, context=fit.context_tokens)
    assert plan.need_gb <= plan.budget_gb + 1e-6
    assert fit.est_memory_gb <= plan.budget_gb + 0.15
    assert fit.download_gb > 0
    assert fit.download_gb * catalog.GIB_PER_GB + catalog.DISK_SPARE_GB <= specs.disk_free_gb
    assert fit.est_speed == perf.speed_label(fit.est_tokens_per_s)
    assert fit.verdict in ("great", "ok", "tight")


@pytest.mark.parametrize("name,specs,expect", SURVEY, ids=[row[0] for row in SURVEY])
def test_survey_machine_gets_a_playable_story_and_referee(name, specs, expect):
    (story_key, story_quant, story_verdict, story_place, story_speed,
     ref_key, ref_quant, ref_verdict, ref_place, primary_bit) = expect
    if name.endswith("_w10"):
        assert specs.os_name == "Windows" and specs.os_version == "10"
    elif name.endswith("_w11"):
        assert specs.os_name == "Windows" and specs.os_version == "11"
    primary = perf.primary_gpu(specs)
    if primary_bit is None:
        assert primary is None
    else:
        assert primary is not None and primary_bit in primary.name
        if not specs.unified_memory:
            assert not __import__("gettowork.specs", fromlist=["_is_integrated"])._is_integrated(primary)

    story = catalog.recommend(specs)
    if story_key is None:
        assert story is None
        assert catalog.recommend_system_one(specs, None) is None
        return

    assert story.model.key == story_key
    assert story.quant == story_quant
    assert story.verdict == story_verdict
    assert story.placement == story_place
    assert story.est_speed == story_speed
    turn = catalog.model_turn_tokens_per_s(story.model, story.est_tokens_per_s, story.placement)
    assert turn >= 3
    assert story.est_speed != "very slow"
    _fits(specs, story)

    if specs.os_name == "Windows" and primary is not None and primary.vendor != "apple":
        gap = catalog.fit_target_mib(specs, decision=True) - catalog.fit_target_mib(specs)
        assert gap == round(catalog.WINDOWS_VRAM_MARGIN_GB * 1024)
    elif primary is not None:
        assert catalog.fit_target_mib(specs, decision=True) == catalog.fit_target_mib(specs)
    if any(gpu.vram_used_gb > 0 for gpu in specs.gpus):
        used = max(gpu.vram_used_gb for gpu in specs.gpus)
        assert catalog.fit_target_mib(specs) >= round((catalog.GPU_VRAM_RESERVE_GB + used) * 1024) - 1

    reserved = catalog.reserve_for_loaded_model(specs, story)
    referee = catalog.recommend_system_one(specs, story)
    if ref_key is None:
        assert referee is None
        for model in catalog.SYSTEM_ONE_CATALOG:
            assert catalog.evaluate_fit(reserved, model).verdict == "no"
        return

    assert referee.model.key == ref_key
    assert referee.quant == ref_quant
    assert referee.verdict == ref_verdict
    assert referee.placement == ref_place
    _fits(reserved, referee)
    assert "seconds per decision" in referee.reason
    assert "tokens/s" not in referee.reason
    seconds = catalog.decision_seconds(
        reserved, placement=referee.placement, gpu_share=referee.gpu_share or 0.0,
        weights_gb=float(referee.download_gb or 0.0),
    )
    timeout = clef_client_timeout(referee.placement)
    expected_timeout = {"cpu": 180.0, "partial": 120.0, "gpu": 30.0, "unified": 30.0}[referee.placement]
    assert timeout == expected_timeout
    assert seconds < timeout
    if name == "dual_rtx3060_w11":
        assert catalog._usable_vram_gb(specs) == pytest.approx(2 * (12.0 - catalog.GPU_VRAM_RESERVE_GB))


def test_a_tiny_pc_is_told_to_play_with_the_pretend_model():
    specs = next(s for n, s, _ in SURVEY if n == "below_minimum_2gb")
    assert catalog.recommend(specs) is None
    console = Console(file=io.StringIO(), width=200, color_system=None)
    ui = UI(console=console, input_fn=lambda prompt: "", secret_fn=lambda prompt: "")
    flow = _SetupFlow(ui, Settings(), object(), SetupServices())
    flow.specs = specs
    search = ModelSearch(discovery=DiscoveryResult(models=[], source="curated"), ranked=[], shortlist=[])
    flow._show_menu(search, [], False)
    text = console.file.getvalue()
    assert "too big or too slow" in text
    assert "pretend model" in text


def test_when_clef_does_not_fit_the_story_model_is_the_referee(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    specs = next(s for n, s, _ in SURVEY if n == "cpu_8_w10")
    story = catalog.recommend(specs)
    assert story is not None and catalog.recommend_system_one(specs, story) is None
    answers = iter(["no"])

    def answer(prompt):
        return next(answers)

    console = Console(file=io.StringIO(), width=200, color_system=None)
    ui = UI(console=console, input_fn=answer, secret_fn=answer)
    client = run_jev_onboarding(ui, Settings(), specs=specs, story_fit=story, engine_tag="b11485", env={})
    text = console.file.getvalue()
    assert client is None
    assert "Neither Clef model looks like it fits" in text
    assert "story model decides" in text
    assert "pretend model" not in text.lower()
