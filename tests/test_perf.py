"""Tests for gettowork.perf: the RAM benchmark, GPU bandwidth guesses and the speed model.

The calibration tests double as documentation: they pin the speed model to
numbers people commonly report for llama.cpp on real hardware.
"""

from __future__ import annotations

import time

import pytest

from gettowork import perf
from gettowork.types import GPUInfo, SystemSpecs


def make_specs(
    *,
    ram: float = 16.0,
    ram_bw: float | None = 40.0,
    gpus: list[GPUInfo] | None = None,
    unified: bool = False,
    cores: int | None = 8,
    arch: str = "x86_64",
    flags: tuple[str, ...] = ("avx", "avx2", "fma", "f16c"),
) -> SystemSpecs:
    return SystemSpecs(
        os_name="Darwin" if unified else "Linux",
        os_version="",
        arch=arch,
        cpu_name="Test CPU",
        cpu_cores_physical=cores,
        cpu_cores_logical=(cores or 4) * 2,
        ram_total_gb=ram,
        ram_available_gb=ram / 2,
        disk_free_gb=500.0,
        gpus=list(gpus or []),
        unified_memory=unified,
        ram_bandwidth_gbs=ram_bw,
        cpu_flags=list(flags),
    )


def nvidia(name: str, vram: float) -> GPUInfo:
    gpu = GPUInfo(name=name, vendor="nvidia", vram_gb=vram)
    gpu.bandwidth_gbs = perf.estimate_gpu_bandwidth(gpu)
    return gpu


def apple(name: str, vram: float) -> GPUInfo:
    gpu = GPUInfo(name=name, vendor="apple", vram_gb=vram)
    gpu.bandwidth_gbs = perf.estimate_gpu_bandwidth(gpu)
    return gpu


# ---------------------------------------------------------------------------
# measure_ram_bandwidth
# ---------------------------------------------------------------------------


def test_ram_benchmark_is_quick_and_plausible():
    # The very first big allocation in a fresh process can be slow on some
    # virtual machines (the host has to back the memory), so warm up once.
    assert perf.measure_ram_bandwidth(budget_s=0.05) is not None
    start = time.perf_counter()
    result = perf.measure_ram_bandwidth()
    elapsed = time.perf_counter() - start
    assert result is not None
    assert 0.5 < result < 5000
    assert elapsed < 2.5  # ~0.3 s of timing plus setting up ~512 MB: well under a second on a normal machine


def test_ram_benchmark_respects_a_tiny_budget():
    start = time.perf_counter()
    assert perf.measure_ram_bandwidth(budget_s=0.05) is not None
    assert time.perf_counter() - start < 2.0


def test_ram_benchmark_is_bigger_than_any_cache_but_memory_safe(monkeypatch):
    # Plenty of free RAM: ~512 MiB in all and >= 32 MiB per thread, so neither
    # the CPU cache (up to ~100s of MB on big chips) nor memcpy's store tricks
    # decide the number - the bug that made the old copy test 2x off.
    monkeypatch.setattr(perf, "_available_bytes", lambda: 32 * 1024**3)
    for threads in (1, 4, 8, 16):
        per_thread, reduced = perf._benchmark_plan(threads)
        assert per_thread >= 32 * 1024 * 1024
        assert per_thread * threads >= 512 * 1024 * 1024 and not reduced
        assert per_thread * threads <= 32 * 1024**3 // 8
    # Little free RAM: never more than 1/8 of it, and the result is flagged.
    monkeypatch.setattr(perf, "_available_bytes", lambda: 1 * 1024**3)
    per_thread, reduced = perf._benchmark_plan(4)
    assert per_thread * 4 <= 1024**3 // 8 and reduced


def test_ram_benchmark_notes_when_it_had_to_shrink(monkeypatch):
    monkeypatch.setattr(perf, "_available_bytes", lambda: 512 * 1024**2)
    assert perf.measure_ram_bandwidth(budget_s=0.05, threads=2) is not None
    assert perf.last_benchmark_note and "small" in perf.last_benchmark_note


def test_ram_benchmark_hardly_depends_on_buffer_size_once_past_the_cache():
    # Reading RAM (not copying small buffers) gives the same answer for any
    # buffer comfortably bigger than the CPU cache.
    small = perf._parallel_read_bandwidth(2, 96 * 1024 * 1024, 0.15)
    large = perf._parallel_read_bandwidth(2, 192 * 1024 * 1024, 0.15)
    assert small and large and 0.5 < small / large < 2.0


def test_ram_benchmark_reads_without_memchr_too(monkeypatch):
    monkeypatch.setattr(perf, "_libc_memchr", lambda: None)
    value = perf._parallel_read_bandwidth(2, 16 * 1024 * 1024, 0.05)
    assert value is not None and 0.5 < value < 5000


def test_silly_ram_readings_are_clamped_to_what_memory_can_do():
    specs = make_specs()
    specs.ram_bandwidth_gbs = 800.0  # a cache, not RAM
    assert perf.bandwidth_for(specs, "cpu")[0] == perf.MAX_PLAUSIBLE_RAM_GBS


def test_ram_benchmark_never_raises(monkeypatch):
    def boom(size):
        raise MemoryError("no memory for you")

    monkeypatch.setattr(perf, "_allocate", boom)
    assert perf.measure_ram_bandwidth() is None


def test_ram_benchmark_handles_garbage_budget():
    assert perf.measure_ram_bandwidth(budget_s="not a number") is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# estimate_gpu_bandwidth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, vendor, vram, expected",
    [
        ("NVIDIA GeForce RTX 3060", "nvidia", 12.0, 360),
        ("NVIDIA GeForce RTX 3060", "nvidia", 8.0, 240),  # the narrower 8 GB variant
        ("NVIDIA GeForce RTX 3060 Ti", "nvidia", 8.0, 448),
        ("NVIDIA GeForce RTX 3080", "nvidia", 12.0, 912),
        ("NVIDIA GeForce RTX 3090", "nvidia", 24.0, 936),
        ("NVIDIA GeForce RTX 4060", "nvidia", 8.0, 272),
        ("NVIDIA GeForce RTX 4070", "nvidia", 12.0, 504),
        ("NVIDIA GeForce RTX 4070 Ti", "nvidia", 12.0, 504),
        ("NVIDIA GeForce RTX 4070 Ti SUPER", "nvidia", 16.0, 672),
        ("NVIDIA GeForce RTX 4080 SUPER", "nvidia", 16.0, 736),
        ("NVIDIA GeForce RTX 4090", "nvidia", 24.0, 1008),
        ("NVIDIA GeForce RTX 5090", "nvidia", 32.0, 1792),
        ("NVIDIA GeForce RTX 5070 Ti", "nvidia", 16.0, 896),
        ("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8.0, 256),
        ("NVIDIA A100-SXM4-80GB", "nvidia", 80.0, 2039),
        ("AMD Radeon RX 7900 XTX", "amd", 24.0, 960),
        ("AMD Radeon RX 7900 XT", "amd", 20.0, 800),
        ("AMD Radeon RX 6700 XT", "amd", 12.0, 384),
        ("AMD Radeon RX 6800", "amd", 16.0, 512),
        ("AMD Radeon RX 9070 XT", "amd", 16.0, 640),
        ("Intel(R) Arc(TM) A770 Graphics", "intel", 16.0, 560),
        ("Intel(R) Arc(TM) B580 Graphics", "intel", 12.0, 456),
        ("Apple M1 GPU", "apple", 11.2, 68),
        ("Apple M2 GPU", "apple", 11.2, 100),
        ("Apple M1 Max GPU", "apple", 22.4, 400),
        ("Apple M2 Ultra GPU", "apple", 144.0, 800),
        ("Apple M4 Pro GPU", "apple", 33.6, 273),
    ],
)
def test_known_gpus(name, vendor, vram, expected):
    assert perf.estimate_gpu_bandwidth(GPUInfo(name=name, vendor=vendor, vram_gb=vram)) == pytest.approx(expected)


def test_unknown_cards_fall_back_to_vram_tiers():
    big = perf.estimate_gpu_bandwidth(GPUInfo(name="NVIDIA Mystery 9000", vendor="nvidia", vram_gb=24))
    small = perf.estimate_gpu_bandwidth(GPUInfo(name="NVIDIA Mystery 100", vendor="nvidia", vram_gb=4))
    assert big and small and big > small
    amd = perf.estimate_gpu_bandwidth(GPUInfo(name="AMD Radeon Something", vendor="amd", vram_gb=16))
    assert amd == pytest.approx(550)


def test_unknown_laptop_cards_are_slower():
    desktop = perf.estimate_gpu_bandwidth(GPUInfo(name="NVIDIA Mystery", vendor="nvidia", vram_gb=8))
    laptop = perf.estimate_gpu_bandwidth(GPUInfo(name="NVIDIA Mystery Laptop GPU", vendor="nvidia", vram_gb=8))
    assert laptop < desktop


def test_no_guess_without_name_or_vram():
    assert perf.estimate_gpu_bandwidth(GPUInfo(name="Weird adapter", vendor="unknown", vram_gb=0)) is None
    assert perf.estimate_gpu_bandwidth(GPUInfo(name="", vendor="amd", vram_gb=0)) is None


def test_gpu_guess_never_raises():
    assert perf.estimate_gpu_bandwidth(GPUInfo(name=None, vendor="nvidia", vram_gb="x")) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# estimate_tokens_per_s - calibration against real-world numbers
# ---------------------------------------------------------------------------

Q4_8B_GB = 4.9  # an 8B model at Q4_K_M
Q4_1_7B_GB = 1.1  # a 1.7B model at Q4_K_M


def test_rtx_3060_runs_8b_q4_at_40_to_60_tokens_per_s():
    specs = make_specs(gpus=[nvidia("NVIDIA GeForce RTX 3060", 12.0)])
    assert 40 <= perf.estimate_tokens_per_s(specs, active_gb=Q4_8B_GB, placement="gpu") <= 60


def test_rtx_4090_runs_8b_q4_at_100_to_140_tokens_per_s():
    specs = make_specs(ram=64, gpus=[nvidia("NVIDIA GeForce RTX 4090", 24.0)])
    assert 100 <= perf.estimate_tokens_per_s(specs, active_gb=Q4_8B_GB, placement="gpu") <= 140


def test_apple_m2_runs_8b_q4_at_15_to_25_tokens_per_s():
    specs = make_specs(unified=True, arch="arm64", flags=("neon",), gpus=[apple("Apple M2 GPU", 11.2)])
    assert 15 <= perf.estimate_tokens_per_s(specs, active_gb=Q4_8B_GB, placement="unified") <= 25


def test_ddr4_desktop_cpu_speeds():
    specs = make_specs(ram_bw=40.0, cores=6)
    assert 4 <= perf.estimate_tokens_per_s(specs, active_gb=Q4_8B_GB, placement="cpu") <= 8
    assert 20 <= perf.estimate_tokens_per_s(specs, active_gb=Q4_1_7B_GB, placement="cpu") <= 35


def test_mixture_of_experts_is_far_faster_than_dense_on_cpu():
    specs = make_specs(ram=64, ram_bw=40.0)
    # Qwen3-30B-A3B at Q4_K_M: 18.6 GB of weights but only ~3.3B of 30.5B params are read per token.
    moe_active = 18.6 * 3.3 / 30.5 * 1.15
    moe = perf.estimate_tokens_per_s(specs, active_gb=moe_active, placement="cpu")
    dense = perf.estimate_tokens_per_s(specs, active_gb=19.8, placement="cpu")  # Qwen3-32B Q4_K_M
    assert moe >= 8
    assert dense < 2
    assert moe > 5 * dense


def test_bigger_models_are_slower_and_tiny_ones_are_capped():
    specs = make_specs(ram=64, gpus=[nvidia("NVIDIA GeForce RTX 4090", 24.0)])
    speeds = [perf.estimate_tokens_per_s(specs, active_gb=gb, placement="gpu") for gb in (0.4, 2.5, 5, 9, 19)]
    assert speeds == sorted(speeds, reverse=True)
    assert speeds[0] < 1000  # the fixed per-token overhead keeps tiny models realistic


def test_partial_offload_is_a_harmonic_mix():
    specs = make_specs(gpus=[nvidia("NVIDIA GeForce RTX 3060", 12.0)])
    gpu = perf.estimate_tokens_per_s(specs, active_gb=10, placement="gpu")
    cpu = perf.estimate_tokens_per_s(specs, active_gb=10, placement="cpu")
    half = perf.estimate_tokens_per_s(specs, active_gb=10, placement="partial", offload_fraction=0.5)
    assert cpu < half < gpu
    assert half == pytest.approx(1 / (0.5 / gpu + 0.5 / cpu))
    assert half < (gpu + cpu) / 2  # harmonic mean: the slow half dominates
    assert perf.estimate_tokens_per_s(specs, active_gb=10, placement="partial", offload_fraction=1.0) == pytest.approx(gpu)
    assert perf.estimate_tokens_per_s(specs, active_gb=10, placement="partial", offload_fraction=0.0) == pytest.approx(cpu)
    more = perf.estimate_tokens_per_s(specs, active_gb=10, placement="partial", offload_fraction=0.8)
    assert more > half


def test_none_and_nonsense_inputs_give_zero():
    specs = make_specs()
    assert perf.estimate_tokens_per_s(specs, active_gb=5, placement="none") == 0.0
    assert perf.estimate_tokens_per_s(specs, active_gb=0, placement="cpu") == 0.0
    assert perf.estimate_tokens_per_s(specs, active_gb=-3, placement="cpu") == 0.0
    assert perf.estimate_tokens_per_s(specs, active_gb="lots", placement="cpu") == 0.0  # type: ignore[arg-type]


def test_gpu_placement_without_a_gpu_falls_back_to_cpu():
    specs = make_specs()
    assert perf.estimate_tokens_per_s(specs, active_gb=5, placement="gpu") == pytest.approx(
        perf.estimate_tokens_per_s(specs, active_gb=5, placement="cpu")
    )


def test_few_cores_and_no_avx2_slow_the_cpu_down():
    base = perf.estimate_tokens_per_s(make_specs(cores=8), active_gb=5, placement="cpu")
    two_cores = perf.estimate_tokens_per_s(make_specs(cores=2), active_gb=5, placement="cpu")
    no_avx2 = perf.estimate_tokens_per_s(make_specs(flags=("avx",)), active_gb=5, placement="cpu")
    unknown_flags = perf.estimate_tokens_per_s(make_specs(flags=()), active_gb=5, placement="cpu")
    assert two_cores < base
    assert no_avx2 < base
    assert unknown_flags == pytest.approx(base)  # unknown is not the same as missing


def test_missing_measurements_use_labelled_defaults():
    specs = make_specs(ram_bw=None)
    assert perf.bandwidth_for(specs, "cpu") == (perf.DEFAULT_CPU_BANDWIDTH_GBS, "default guess")
    assert perf.estimate_tokens_per_s(specs, active_gb=5, placement="cpu") > 0
    gpu = GPUInfo(name="Mystery", vendor="unknown", vram_gb=0.0)
    assert perf.bandwidth_for(make_specs(gpus=[gpu]), "gpu")[1] in ("measured", "default guess")


def test_bandwidth_sources_are_labelled():
    specs = make_specs(gpus=[nvidia("NVIDIA GeForce RTX 4090", 24.0)])
    assert perf.bandwidth_for(specs, "gpu") == (1008.0, "published spec, a rough guess")
    assert perf.bandwidth_for(specs, "cpu") == (40.0, "measured")
    silly = make_specs(ram_bw=0.001)
    assert perf.bandwidth_for(silly, "cpu")[0] >= 3.0  # clamped


def test_primary_gpu_prefers_the_biggest_dedicated_card():
    small = nvidia("NVIDIA GeForce GTX 1650", 4.0)
    big = nvidia("NVIDIA GeForce RTX 3090", 24.0)
    igpu = GPUInfo(name="Intel UHD Graphics", vendor="intel", vram_gb=0.0)
    assert perf.primary_gpu(make_specs(gpus=[igpu, small, big])) is big
    assert perf.primary_gpu(make_specs(gpus=[igpu])) is None
    mac = apple("Apple M2 GPU", 11.2)
    assert perf.primary_gpu(make_specs(unified=True, gpus=[mac])) is mac


@pytest.mark.parametrize(
    "tps, label",
    [(None, "n/a"), (0.5, "very slow"), (2.99, "very slow"), (3, "slow"), (7.9, "slow"), (8, "usable"),
     (19.9, "usable"), (20, "fast"), (500, "fast")],
)
def test_speed_labels(tps, label):
    assert perf.speed_label(tps) == label


def test_speed_explainer_is_short_friendly_and_honest():
    words = len(perf.SPEED_EXPLAINER.split())
    assert 60 <= words <= 180
    text = perf.SPEED_EXPLAINER.lower()
    assert "bandwidth" in text
    assert "estimate" in text
    assert "no warranty" in text


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_ram_benchmark_uses_every_core(monkeypatch):
    # llama.cpp reads with every core, and one core alone can't fill a
    # desktop's (let alone a server's) memory channels.
    calls = []

    def parallel(threads, per_thread_bytes, seconds):
        calls.append(threads)
        return 999.0

    monkeypatch.setattr(perf, "_parallel_read_bandwidth", parallel)
    assert perf.measure_ram_bandwidth(threads=8) == 999.0
    assert perf.measure_ram_bandwidth(threads=1) == 999.0
    assert calls == [8, 1]


def test_parallel_read_really_reads():
    value = perf._parallel_read_bandwidth(2, 32 * 1024 * 1024, 0.05)
    assert value is not None and 0.5 < value < 5000


def test_no_avx_penalty_depends_on_real_cpu_features_only():
    base = perf.estimate_tokens_per_s(make_specs(flags=()), active_gb=2, placement="cpu")
    vulkan_only = perf.estimate_tokens_per_s(make_specs(flags=("vulkan",)), active_gb=2, placement="cpu")
    assert vulkan_only == pytest.approx(base)  # a graphics library says nothing about the processor
    celeron = perf.estimate_tokens_per_s(make_specs(flags=("sse2", "sse4_2")), active_gb=2, placement="cpu")
    celeron_vulkan = perf.estimate_tokens_per_s(make_specs(flags=("sse2", "sse4_2", "vulkan")), active_gb=2,
                                                placement="cpu")
    assert celeron < base and celeron_vulkan == pytest.approx(celeron)
