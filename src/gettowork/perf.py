"""How fast will a model talk? Memory bandwidth, a micro-benchmark, and a speed model.

When a language model writes, it produces one *token* (a word-piece) at a
time, and for every token it has to read (almost) all of its weights from
memory. Doing the maths on those weights is quick; *fetching* them is the slow
part. So generation speed is set mostly by **memory bandwidth** (how many GB
per second your memory can move), not by how many cores or TFLOPS you have:

    tokens/sec ≈ efficiency × bandwidth ÷ gigabytes read per token

This module provides the three ingredients:

* `measure_ram_bandwidth` - a quick (well under a second, ~512 MB) *read*
  benchmark for system RAM, run on all cores at once (like llama.cpp itself).
* `estimate_gpu_bandwidth` - a look-up table of rough, published memory
  bandwidths for common graphics cards (a *guess*, clearly labelled).
* `estimate_tokens_per_s` - the speed model itself.

The efficiency and overhead constants below were calibrated by hand so that
the model lands inside real-world ranges people report for llama.cpp (see the
"Calibration" comment). They are our own rough heuristic, MIT licensed, with
no warranty: real speed depends on drivers, settings, context length, and
what else your computer is doing. The game measures the real speed once the
model is running.
"""

from __future__ import annotations

import contextlib
import math
import re
import time
from typing import Any, Optional

from .types import GPUInfo, SystemSpecs

__all__ = [
    "measure_ram_bandwidth",
    "estimate_gpu_bandwidth",
    "estimate_tokens_per_s",
    "primary_gpu",
    "bandwidth_for",
    "efficiency_for",
    "speed_label",
    "EFFICIENCY",
    "OVERHEAD_S_PER_TOKEN",
    "DEFAULT_CPU_BANDWIDTH_GBS",
    "MAX_PLAUSIBLE_RAM_GBS",
    "SPEED_EXPLAINER",
]

# ---------------------------------------------------------------------------
# The speed model's knobs (tweak these and re-run tests/test_perf.py!)
# ---------------------------------------------------------------------------

# Fraction of the raw bandwidth that llama.cpp turns into useful weight reads.
# GPU / unified figures are relative to the card's published bandwidth; the CPU
# figure is relative to OUR measured read bandwidth, which uses several cores
# at once - as llama.cpp does. (A single core can't come close to saturating
# a desktop's or server's memory channels.)
EFFICIENCY: dict[str, float] = {"gpu": 0.73, "unified": 0.85, "cpu": 0.70}

# A fixed cost per token (kernel launches, thread sync, sampling...). It hardly
# matters for big models but stops tiny ones from "running" at 5,000 tokens/s.
OVERHEAD_S_PER_TOKEN: dict[str, float] = {"gpu": 0.0013, "unified": 0.005, "cpu": 0.002}

# Used when the benchmark was skipped or failed: a typical dual-channel laptop.
DEFAULT_CPU_BANDWIDTH_GBS = 20.0
DEFAULT_GPU_BANDWIDTH_GBS = 200.0
DEFAULT_UNIFIED_BANDWIDTH_GBS = 100.0

# Calibration (plausibility checks, also asserted in tests/test_perf.py).
# Numbers people commonly report for llama.cpp text generation:
#   RTX 3060 12 GB (~360 GB/s), 8B at Q4_K_M (~4.9 GB)  -> ~40-60 tok/s  (model: ~50)
#   RTX 4090 (~1008 GB/s),      8B at Q4_K_M            -> ~100-140 tok/s (model: ~125)
#   Apple M2 (~100 GB/s),       8B at Q4_K_M            -> ~15-25 tok/s  (model: ~16)
#   DDR4 desktop (~40 GB/s measured), 8B at Q4_K_M      -> ~4-8 tok/s    (model: ~5.5)
#                                     1.7B at Q4_K_M    -> ~20-35 tok/s  (model: ~24)
#   Qwen3-30B-A3B (only ~3B active per token) on that same CPU -> ~10+ tok/s,
#   versus <1 tok/s for a dense 32B model. That's the magic of Mixture-of-Experts.
# The contract's first-draft efficiencies (0.55 / 0.45 / 0.35) under-shot the
# Apple and CPU ranges by ~2x, so these were re-fitted: two GPU data points give
# both the GPU efficiency and the fixed overhead, and likewise for Apple.


# ---------------------------------------------------------------------------
# RAM micro-benchmark
# ---------------------------------------------------------------------------

_MIB = 1024 * 1024
# The test *reads* memory, like token generation does: every thread scans its
# own big buffer with the C library's ``memchr`` (looking for a byte that
# isn't there, so it reads every byte - at full memory speed, no writes).
_TARGET_TOTAL_BYTES = 512 * _MIB  # all threads together: bigger than any realistic CPU cache (L3)
_MIN_TOTAL_BYTES = 64 * _MIB  # the least we'll use, on a machine with very little free RAM
_MIN_THREAD_BYTES = 32 * _MIB  # per thread: far past the point where memcpy's cache tricks kick in
_READ_SLICE_BYTES = 1024 * 1024  # fallback without memchr: copied per call into a small, cached scratch buffer
_RAM_SHARE = 8  # use at most 1/8 of the RAM that's free right now
_BENCH_ROUNDS = 3  # the median of a few short rounds, so one hiccup doesn't decide the number
_MAX_COPY_THREADS = 16  # enough to fill the memory channels of a big workstation
# No home or workstation memory reads faster than this (12 channels of DDR5 ~ 460 GB/s);
# a higher "measurement" means we timed a cache after all.
MAX_PLAUSIBLE_RAM_GBS = 460.0
# Set by measure_ram_bandwidth when it had to use a small test (little free RAM):
# the result may then be flattered by the CPU cache. specs.py shows it as a note.
last_benchmark_note: Optional[str] = None


def _allocate(size: int) -> bytearray:
    """Allocate a zeroed buffer (a separate function so tests can simulate failure)."""
    return bytearray(size)


def _available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except Exception:
        return 2 * 1024 * _MIB  # can't tell: assume a couple of GB are free


def _benchmark_plan(threads: int) -> tuple[int, bool]:
    """(bytes per thread, reduced?) for the read test.

    Normally 512 MiB in total and at least 32 MiB per thread - bigger than
    any CPU cache, so we time the RAM itself. On a machine with little free
    RAM we use less (at most 1/8 of what's free) and say so.
    """
    threads = max(1, int(threads))
    budget = max(_MIN_TOTAL_BYTES, min(_TARGET_TOTAL_BYTES, _available_bytes() // _RAM_SHARE))
    per_thread = max(_MIN_THREAD_BYTES, budget // threads)
    if per_thread * threads > budget:
        per_thread = max(_READ_SLICE_BYTES * 8, budget // threads)
    per_thread -= per_thread % _READ_SLICE_BYTES
    reduced = per_thread * threads < _TARGET_TOTAL_BYTES or per_thread < _MIN_THREAD_BYTES
    return per_thread, reduced


def _buffer_size() -> int:
    """Bytes per thread the read test uses on a single core (see `_benchmark_plan`)."""
    return _benchmark_plan(1)[0]


def _copy_threads() -> int:
    """How many cores to read with: the physical cores (llama.cpp's default), capped."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False) or psutil.cpu_count() or 1
    except Exception:
        import os

        cores = os.cpu_count() or 1
    return max(1, min(_MAX_COPY_THREADS, int(cores)))


def _libc_memchr() -> Optional[Any]:
    """The C library's ``memchr``, callable through ctypes (None if unavailable).

    ctypes releases Python's GIL around foreign calls, so several threads
    can scan memory on separate cores at the same time.
    """
    import ctypes
    import sys

    candidates: list[Any] = []
    try:
        if sys.platform == "win32":
            for name in ("ucrtbase", "msvcrt"):
                with contextlib.suppress(OSError, AttributeError):
                    candidates.append(getattr(ctypes.cdll, name))
        else:
            candidates.append(ctypes.CDLL(None))
    except OSError:
        return None
    for lib in candidates:
        try:
            func = lib.memchr
        except AttributeError:
            continue
        func.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t)
        func.restype = ctypes.c_void_p
        return func
    return None


def _parallel_read_bandwidth(threads: int, per_thread_bytes: int, seconds: float,
                             rounds: int = _BENCH_ROUNDS) -> Optional[float]:
    """GB/s *read* by `threads` threads streaming through their own buffers at once.

    Each thread scans its own buffer with ``memchr`` for a byte that isn't
    there, so it reads every byte - just what generating a token does with
    the model's weights. (Without ``memchr``, it copies the buffer a slice at
    a time into a small scratch buffer that stays in the CPU cache.) The
    calls release Python's GIL, so the threads really run on separate cores,
    like llama.cpp. Each thread's speed counts only its completed passes over
    the time they took; the result is the median of `rounds` short rounds.
    None if it can't run here.
    """
    import ctypes
    import statistics
    import threading

    threads = max(1, int(threads))
    memchr = _libc_memchr()
    slice_bytes = min(_READ_SLICE_BYTES, per_thread_bytes)
    size = max(slice_bytes, per_thread_bytes - per_thread_bytes % slice_bytes)
    sources: list[bytearray] = []
    addresses: list[int] = []
    for _ in range(threads):
        buf = _allocate(size)
        # Write every 4 KiB page, so the OS really hands us separate memory
        # (untouched pages all map to one shared zero page - which would be cached).
        buf[::4096] = b"\x01" * len(range(0, size, 4096))
        sources.append(buf)
        addresses.append(ctypes.addressof((ctypes.c_char * size).from_buffer(buf)))
    sinks = [ctypes.create_string_buffer(slice_bytes) for _ in range(threads)] if memchr is None else []

    def one_pass(index: int) -> None:
        base = addresses[index]
        if memchr is not None:
            memchr(base, 0xFF, size)  # 0xFF is never there: every byte is read
        else:
            for offset in range(0, size, slice_bytes):
                ctypes.memmove(sinks[index], base + offset, slice_bytes)

    for i in range(threads):  # warm-up pass: page faults happen here, not timed
        one_pass(i)

    results: list[float] = []
    per_round = max(0.02, float(seconds) / max(1, rounds))
    for _ in range(max(1, rounds)):
        start = threading.Barrier(threads + 1)
        stop = threading.Event()
        timing: list[tuple[int, float, float]] = [(0, 0.0, 0.0)] * threads

        def worker(index: int) -> None:
            start.wait()
            began = last = time.perf_counter()
            passes = 0
            while not stop.is_set():
                one_pass(index)
                passes += 1
                last = time.perf_counter()
            timing[index] = (passes, began, last)

        workers = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(threads)]
        for w in workers:
            w.start()
        start.wait()
        time.sleep(per_round)
        stop.set()
        for w in workers:
            w.join(timeout=5.0)
        speed = sum(passes * size / (last - began) for passes, began, last in timing if passes and last > began)
        if speed > 0:
            results.append(speed / 1e9)
    del sources
    return statistics.median(results) if results else None


def measure_ram_bandwidth(budget_s: float = 0.3, threads: Optional[int] = None) -> Optional[float]:
    """Measure how fast this computer can *read* its RAM, in GB/s, or None on failure.

    Token generation reads the model's weights from memory with every core,
    so that's what we time: all physical cores (`threads`) stream through
    their own buffers at once for about `budget_s` seconds, and we take the
    median of a few rounds. The buffers add up to ~512 MB and at least 32 MB
    per core - bigger than any CPU cache, so we time real memory, not cache.
    (Timing a *copy* of small buffers, as an earlier version did, measured
    the cache or memcpy's tricks instead, off by 2x either way.) On a
    machine with little free RAM the test shrinks and `last_benchmark_note`
    says so. Allocating and touching the memory takes a moment too, so the
    whole thing is well under a second. Never raises.
    """
    global last_benchmark_note
    last_benchmark_note = None
    try:
        budget_s = min(max(float(budget_s), 0.05), 0.6)
        threads = _copy_threads() if threads is None else max(1, int(threads))
        per_thread, reduced = _benchmark_plan(threads)
        value = _parallel_read_bandwidth(threads, per_thread, budget_s)
        if value is None or value <= 0 or not math.isfinite(value):
            return None
        if reduced:
            last_benchmark_note = (
                "There wasn't much free memory, so the RAM speed test was small and may read a little high."
            )
        return round(value, 1)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPU bandwidth: a rough look-up table
# ---------------------------------------------------------------------------

# (name pattern, desktop GB/s, laptop GB/s or None). Published spec-sheet
# figures, rounded. MORE SPECIFIC NAMES MUST COME FIRST ("4070 ti super" before
# "4070 ti" before "4070"). Spaces in a pattern match any whitespace, and the
# pattern must end on a word boundary, so "rx 7900 xt" does not match "7900 xtx".
_GPU_TABLE: list[tuple[str, float, Optional[float]]] = [
    # --- NVIDIA GeForce RTX 50 / 40 / 30 / 20 series ---
    ("rtx 5090", 1792, 896),
    ("rtx 5080", 960, 896),
    ("rtx 5070 ti", 896, 672),
    ("rtx 5070", 672, 384),
    ("rtx 5060 ti", 448, None),
    ("rtx 5060", 448, 384),
    ("rtx 5050", 320, 384),
    ("rtx 4090", 1008, 576),
    ("rtx 4080 super", 736, None),
    ("rtx 4080", 717, 432),
    ("rtx 4070 ti super", 672, None),
    ("rtx 4070 ti", 504, None),
    ("rtx 4070 super", 504, None),
    ("rtx 4070", 504, 256),
    ("rtx 4060 ti", 288, None),
    ("rtx 4060", 272, 256),
    ("rtx 4050", 192, 192),
    ("rtx 3090 ti", 1008, None),
    ("rtx 3090", 936, None),
    ("rtx 3080 ti", 912, 512),
    ("rtx 3080", 760, 448),
    ("rtx 3070 ti", 608, 448),
    ("rtx 3070", 448, 448),
    ("rtx 3060 ti", 448, None),
    ("rtx 3060", 360, 336),
    ("rtx 3050 ti", 192, 192),
    ("rtx 3050", 224, 192),
    ("rtx 2080 ti", 616, None),
    ("rtx 2080 super", 496, 448),
    ("rtx 2080", 448, 384),
    ("rtx 2070 super", 448, 448),
    ("rtx 2070", 448, 384),
    ("rtx 2060 super", 448, None),
    ("rtx 2060", 336, 336),
    ("titan rtx", 672, None),
    ("gtx 1660 ti", 288, 288),
    ("gtx 1660 super", 336, None),
    ("gtx 1660", 192, None),
    ("gtx 1650 super", 192, None),
    ("gtx 1650", 128, 128),
    ("gtx 1080 ti", 484, None),
    ("gtx 1080", 320, 320),
    ("gtx 1070 ti", 256, None),
    ("gtx 1070", 256, 256),
    ("gtx 1060", 192, 192),
    # --- NVIDIA workstation / data centre ---
    ("rtx pro 6000", 1792, None),
    ("rtx 6000 ada", 960, None),
    ("rtx 5000 ada", 576, None),
    ("rtx 4000 ada", 360, None),
    ("a6000", 768, None),
    ("a5000", 768, None),
    ("a4500", 640, None),
    ("a4000", 448, None),
    ("a2000", 288, None),
    ("h200", 4800, None),
    ("gh200", 4000, None),
    ("h100", 2000, None),
    ("a100", 1555, None),
    ("l40s", 864, None),
    ("l40", 864, None),
    ("l4", 300, None),
    ("a10g", 600, None),
    ("a10", 600, None),
    ("t4", 320, None),
    ("v100", 900, None),
    ("p100", 732, None),
    ("p40", 346, None),
    # --- AMD Radeon RX 9000 / 7000 / 6000 / 5000 series ---
    ("rx 9070 xt", 640, None),
    ("rx 9070", 640, None),
    ("rx 9060 xt", 320, None),
    ("rx 7900 xtx", 960, None),
    ("rx 7900 xt", 800, None),
    ("rx 7900 gre", 576, None),
    ("rx 7800 xt", 624, None),
    ("rx 7700 xt", 432, None),
    ("rx 7600 xt", 288, None),
    ("rx 7600", 288, None),
    ("rx 6950 xt", 576, None),
    ("rx 6900 xt", 512, None),
    ("rx 6800 xt", 512, None),
    ("rx 6800", 512, None),
    ("rx 6750 xt", 432, None),
    ("rx 6700 xt", 384, None),
    ("rx 6700", 320, None),
    ("rx 6650 xt", 280, None),
    ("rx 6600 xt", 256, None),
    ("rx 6600", 224, None),
    ("rx 6500 xt", 144, None),
    ("rx 6400", 128, None),
    ("rx 5700 xt", 448, None),
    ("rx 5700", 448, None),
    ("rx 5600 xt", 288, None),
    ("rx 5500 xt", 224, None),
    ("radeon vii", 1024, None),
    ("rx 590", 256, None),
    ("rx 580", 256, None),
    ("rx 570", 224, None),
    ("w7900", 864, None),
    ("w7800", 576, None),
    ("mi300x", 5300, None),
    ("mi250x", 3277, None),
    ("mi250", 3277, None),
    ("mi210", 1638, None),
    ("mi100", 1229, None),
    # AMD integrated graphics share (fast-ish) system RAM
    ("8060s", 256, None),
    ("8050s", 256, None),
    ("890m", 120, None),
    ("880m", 120, None),
    ("780m", 90, None),
    ("760m", 90, None),
    ("740m", 90, None),
    ("680m", 80, None),
    ("660m", 80, None),
    # --- Intel Arc ---
    ("arc pro b60", 456, None),
    ("arc b580", 456, None),
    ("arc b570", 380, None),
    ("arc a770m", 512, None),
    ("arc a770", 560, None),
    ("arc a750", 512, None),
    ("arc a580", 512, None),
    ("arc a730m", 336, None),
    ("arc a550m", 224, None),
    ("arc a380", 186, None),
    ("arc a370m", 112, None),
    ("arc a310", 124, None),
    ("arc 140v", 136, None),
    ("arc 130v", 136, None),
    # Intel integrated graphics (shared system RAM)
    ("arc graphics", 100, None),
    ("iris xe", 70, None),
    ("iris plus", 50, None),
    ("uhd graphics", 50, None),
]
_GPU_PATTERNS = [
    (re.compile(r"\b" + r"\s*".join(map(re.escape, key.split())) + r"\b"), key, desk, lap)
    for key, desk, lap in _GPU_TABLE
]

# Apple Silicon: the GPU reads the same unified memory as the CPU.
_APPLE_BANDWIDTH: dict[tuple[int, str], float] = {
    (1, ""): 68, (1, "pro"): 200, (1, "max"): 400, (1, "ultra"): 800,
    (2, ""): 100, (2, "pro"): 200, (2, "max"): 400, (2, "ultra"): 800,
    (3, ""): 100, (3, "pro"): 150, (3, "max"): 400, (3, "ultra"): 819,  # binned M3 Max: 300
    (4, ""): 120, (4, "pro"): 273, (4, "max"): 546, (4, "ultra"): 1092,  # binned M4 Max: 410
    (5, ""): 153, (5, "pro"): 300, (5, "max"): 600, (5, "ultra"): 1200,  # M5 family: best guesses
}
_APPLE_RE = re.compile(r"\bm(\d+)(?:\s*(pro|max|ultra))?\b")

_LAPTOP_RE = re.compile(r"\b(laptop|mobile|max-q|max-p)\b")


def _normalise_gpu_name(name: str) -> str:
    text = name.lower()
    for junk in ("(r)", "(tm)", "®", "™", "nvidia", "geforce", "corporation"):
        text = text.replace(junk, " ")
    return re.sub(r"\s+", " ", text).strip()


def estimate_gpu_bandwidth(gpu: GPUInfo) -> Optional[float]:
    """A ROUGH GUESS of a GPU's memory bandwidth in GB/s (None if we can't guess).

    Nobody's GPU reports its bandwidth in a portable way, so we look the name
    up in a small table of published spec-sheet figures for common cards
    (NVIDIA RTX 20-50 series, AMD RX 5000-9000, Intel Arc, Apple M-series).
    Unknown cards fall back to a guess from the vendor and amount of video
    memory (bigger cards usually have wider, faster memory buses). Laptop
    versions of a chip are usually slower than the desktop card of the same
    name, so we use the laptop figure when the name says "Laptop".
    """
    try:
        name = _normalise_gpu_name(gpu.name or "")
        vram = float(gpu.vram_gb or 0.0)

        if gpu.vendor == "apple" or name.startswith("apple"):
            match = _APPLE_RE.search(name)
            if match:
                gen = int(match.group(1))
                tier = match.group(2) or ""
                if (gen, tier) in _APPLE_BANDWIDTH:
                    return float(_APPLE_BANDWIDTH[(gen, tier)])
                return float(_APPLE_BANDWIDTH[(5, tier)]) if gen > 5 else DEFAULT_UNIFIED_BANDWIDTH_GBS
            return DEFAULT_UNIFIED_BANDWIDTH_GBS

        laptop = bool(_LAPTOP_RE.search(name))
        for pattern, key, desktop_gbs, laptop_gbs in _GPU_PATTERNS:
            if not pattern.search(name):
                continue
            value = float(desktop_gbs)
            if laptop:
                value = float(laptop_gbs) if laptop_gbs else value * 0.65
            # A few names cover two memory configurations.
            elif key == "rtx 3060" and 0 < vram < 10:
                value = 240.0  # the 8 GB desktop RTX 3060 has a narrower bus
            elif key == "rtx 3080" and vram >= 11:
                value = 912.0  # the 12 GB RTX 3080
            elif key == "a100" and vram >= 60:
                value = 2039.0  # A100 80 GB
            return value

        return _bandwidth_from_vram_tier(gpu.vendor, vram, laptop)
    except Exception:
        return None


def _bandwidth_from_vram_tier(vendor: str, vram: float, laptop: bool) -> Optional[float]:
    """Fallback guess: more video memory usually means a wider, faster bus."""
    if vram <= 0:
        return None
    tiers: dict[str, list[tuple[float, float]]] = {
        "nvidia": [(20, 900), (14, 600), (10, 400), (7, 300), (3.5, 190), (0, 100)],
        "amd": [(20, 800), (14, 550), (10, 400), (7, 250), (0, 120)],
        "intel": [(10, 450), (6, 300), (0, 70)],
    }
    for min_vram, gbs in tiers.get(vendor, [(8, 250), (0, 100)]):
        if vram >= min_vram:
            return gbs * (0.65 if laptop else 1.0)
    return None  # pragma: no cover - the last tier always matches


# ---------------------------------------------------------------------------
# The speed model
# ---------------------------------------------------------------------------


def primary_gpu(specs: SystemSpecs) -> Optional[GPUInfo]:
    """The GPU the game would use: the Apple GPU on unified-memory Macs, else
    the dedicated GPU with the most video memory (None if there isn't one, or
    if the engine can't use graphics cards on this computer - ``gpu_offload``)."""
    if specs.gpu_offload is False:
        return None
    if specs.unified_memory:
        for gpu in specs.gpus:
            if gpu.vendor == "apple":
                return gpu
    dedicated = [g for g in specs.gpus if g.vendor != "apple" and g.vram_gb > 0]
    return max(dedicated, key=lambda g: g.vram_gb, default=None)


def bandwidth_for(specs: SystemSpecs, placement: str) -> tuple[float, str]:
    """(GB/s, where the number came from) for a placement: "gpu", "unified" or "cpu".

    The source is one of "measured", "published spec, a rough guess" or
    "default guess", so the UI can be honest about how solid the number is.
    With no usable GPU, "gpu"/"unified" quietly fall back to the CPU numbers.
    """
    if placement in ("gpu", "unified"):
        gpu = primary_gpu(specs)
        if gpu is not None:
            value = gpu.bandwidth_gbs or estimate_gpu_bandwidth(gpu)
            if value:
                return float(value), "published spec, a rough guess"
            default = DEFAULT_UNIFIED_BANDWIDTH_GBS if gpu.vendor == "apple" else DEFAULT_GPU_BANDWIDTH_GBS
            return default, "default guess"
    measured = specs.ram_bandwidth_gbs
    if measured and measured > 0:
        # Clamp silly readings (VM hiccups, a laptop waking from sleep, a cache...).
        return float(min(max(measured, 3.0), MAX_PLAUSIBLE_RAM_GBS)), "measured"
    return DEFAULT_CPU_BANDWIDTH_GBS, "default guess"


def efficiency_for(specs: SystemSpecs, placement: str) -> float:
    """The efficiency factor the speed model uses for a placement on this machine."""
    if placement == "cpu" or placement not in EFFICIENCY:
        return _cpu_efficiency(specs)
    return EFFICIENCY[placement]


def _cpu_efficiency(specs: SystemSpecs) -> float:
    """CPU efficiency, reduced for machines with few cores or no AVX2.

    With only 1-3 cores llama.cpp spends much of its time on arithmetic
    rather than waiting for memory, and x86 CPUs without AVX2 unpack
    quantized weights much more slowly. Only real CPU features count here:
    "vulkan" (a graphics library we note in the same list) says nothing about
    the processor, and an empty list means "couldn't tell", not "none".
    """
    eff = EFFICIENCY["cpu"]
    cores = specs.cpu_cores_physical or specs.cpu_cores_logical
    if cores:
        eff *= min(1.0, 0.5 + 0.125 * cores)  # 2 cores -> 75 %, 4+ cores -> 100 %
    arch = (specs.arch or "").lower()
    is_x86 = arch in ("x86_64", "amd64", "x86", "i386", "i686")
    simd = {f for f in specs.cpu_flags if f != "vulkan"}
    if is_x86 and simd and "avx2" not in simd:
        eff *= 0.6
    return eff


def _speed_on(specs: SystemSpecs, placement: str, active_gb: float) -> float:
    """Tokens/s if the whole model ran with this placement ("gpu", "unified" or "cpu")."""
    if placement in ("gpu", "unified"):
        gpu = primary_gpu(specs)
        if gpu is None:
            placement = "cpu"
        else:  # an Apple GPU always behaves like "unified", any other like "gpu"
            placement = "unified" if gpu.vendor == "apple" else "gpu"
    bandwidth, _ = bandwidth_for(specs, placement)
    eff = efficiency_for(specs, placement)
    seconds_per_token = active_gb / (eff * bandwidth) + OVERHEAD_S_PER_TOKEN[placement]
    return 1.0 / seconds_per_token


def estimate_tokens_per_s(
    specs: SystemSpecs,
    *,
    active_gb: float,
    placement: str,
    offload_fraction: float = 1.0,
) -> float:
    """ESTIMATE generation speed (tokens/second) for a model on this machine.

    `active_gb` is how many gigabytes are read per token: the weights' size,
    or for a Mixture-of-Experts model only the active experts' share of it
    (the caller works that out). `placement` is "gpu", "unified", "partial",
    "cpu" or "none". For "partial", `offload_fraction` is the share of the
    model living on the GPU.

    The model: ``seconds per token = active_gb / (efficiency × bandwidth) +
    fixed overhead``. A partial offload is a *harmonic* mix, because each
    token has to pass through both halves one after the other:
    ``1 / speed = f / gpu_speed + (1 - f) / cpu_speed``.

    This is an estimate, not a promise. Returns 0.0 for "none".
    """
    try:
        active_gb = float(active_gb)
    except (TypeError, ValueError):
        return 0.0
    if placement == "none" or active_gb <= 0 or not math.isfinite(active_gb):
        return 0.0
    if placement == "partial":
        fraction = min(max(float(offload_fraction), 0.0), 1.0)
        gpu_speed = _speed_on(specs, "gpu", active_gb)
        cpu_speed = _speed_on(specs, "cpu", active_gb)
        return 1.0 / (fraction / gpu_speed + (1.0 - fraction) / cpu_speed)
    if placement not in EFFICIENCY:
        placement = "cpu"
    return _speed_on(specs, placement, active_gb)


def speed_label(tokens_per_s: Optional[float]) -> str:
    """Plain-English speed: "fast" (≥20 tok/s), "usable" (≥8), "slow" (≥3) or "very slow"."""
    if tokens_per_s is None:
        return "n/a"
    if tokens_per_s >= 20:
        return "fast"
    if tokens_per_s >= 8:
        return "usable"
    if tokens_per_s >= 3:
        return "slow"
    return "very slow"


SPEED_EXPLAINER = """\
**Why memory *speed* matters, not just size**

To write each new word-piece (a *token*), a model reads essentially all of its
weights from memory. Reading, not maths, is the bottleneck, so speed depends
mostly on **memory bandwidth**: how many gigabytes per second your memory can
move.

`tokens/sec ≈ efficiency × bandwidth ÷ GB read per token`

- Graphics-card memory is fast: an RTX 3060 moves ~360 GB/s, so a 5 GB model
  talks at roughly 50 tokens/s.
- Regular RAM is slower (often 30-60 GB/s), so the same model on a CPU manages
  about 5 tokens/s.
- *Mixture-of-Experts* models (like Qwen3-30B-A3B) only read a few "experts"
  per token, so they run far faster than their size suggests.

We timed your RAM with a quick read test (on all your cores at once, like
the engine) and looked up rough published numbers for your graphics card. It's a home-grown estimate with no warranty: real
speed varies, and the game measures the real thing once your model starts.
"""
