"""Which open-weight models fit this computer, and which should we suggest?

This is the "fit engine". For every candidate model it answers four questions:

1. **Which version (quantization) should we download?** GGUF repos offer the
   same model squeezed to different sizes: Q8_0 (~8.5 bits per weight, nearly
   perfect), Q4_K_M (~4.8 bits, the popular sweet spot), Q3_K_M (~3.9 bits,
   noticeably rougher)... `choose_quant` walks that ladder for *this* machine.
2. **How much memory will it need?** Weights + KV cache + overhead
   (`estimate_memory_gb`, explained in `MEMORY_FORMULA_EXPLAINER`).
3. **Where will it run, and how fast?** On the graphics card, in a Mac's
   unified memory, split between GPU and RAM, or on the CPU - with a speed
   estimate from `perf.estimate_tokens_per_s`.
4. **How good a pick is it overall?** `rank_models` scores every candidate
   (the formula is documented in `score_fit` so you can tweak it), and
   `pick_shortlist` chooses a handful of diverse picks with friendly badges.

It also holds `MODEL_CATALOG`, a small curated list of permissively licensed
models used when live Hugging Face discovery is unavailable (and as a "known
good" bonus when it is).

Everything here is our own home-grown heuristic, MIT licensed and provided
with no warranty. Real memory use and speed vary with drivers, settings and
whatever else your computer is doing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Optional

from . import perf
from .types import FitResult, ModelEntry, SystemSpecs

__all__ = [
    "MODEL_CATALOG",
    "TRUSTED_PUBLISHERS",
    "PERMISSIVE_LICENSES",
    "QUANT_BITS",
    "QUANT_PREFERENCE",
    "MEMORY_FORMULA_EXPLAINER",
    "get_model",
    "is_permissive",
    "quant_bits",
    "quant_quality",
    "estimate_quant_size_gb",
    "kv_cache_gb",
    "estimate_memory_gb",
    "turn_tokens_per_s",
    "model_turn_tokens_per_s",
    "thinking_mode",
    "always_thinks",
    "choose_quant",
    "size_unknown",
    "evaluate_fit",
    "score_fit",
    "rank_models",
    "pick_shortlist",
    "runnable_by_engine",
    "disk_space_needed",
    "recommend",
    "explain_fit",
    "speed_breakdown",
]

# ---------------------------------------------------------------------------
# Licenses and publishers
# ---------------------------------------------------------------------------

TRUSTED_PUBLISHERS: tuple[str, ...] = (
    "unsloth",
    "bartowski",
    "ggml-org",
    "lmstudio-community",
    "Qwen",
    "microsoft",
    "mistralai",
    "HuggingFaceTB",
    "ibm-granite",
    "NousResearch",
)

# Licenses that let anyone use, share and modify the model, commercially or not,
# with no extra conditions beyond keeping the notice. We only suggest these by default.
PERMISSIVE_LICENSES: frozenset[str] = frozenset({"apache-2.0", "mit"})


def is_permissive(license_id: Optional[str]) -> bool:
    """True for Apache-2.0 / MIT, however they are spelled ("Apache 2.0", "mit", ...)."""
    if not license_id:
        return False
    normalised = re.sub(r"[\s_]+", "-", license_id.strip().lower())
    normalised = {"apache-2": "apache-2.0", "apache2.0": "apache-2.0", "apache": "apache-2.0"}.get(normalised, normalised)
    return normalised in PERMISSIVE_LICENSES


# ---------------------------------------------------------------------------
# Quantization: bits per weight and how much quality survives
# ---------------------------------------------------------------------------

# Approximate average bits per weight of llama.cpp quant types (a few tensors
# are kept at higher precision, which is why Q4_K_M is ~4.8, not 4.0).
QUANT_BITS: dict[str, float] = {
    "F32": 32.0, "F16": 16.0, "BF16": 16.0,
    "Q8_0": 8.5, "Q8_K_XL": 9.0,
    "Q6_K": 6.6, "Q6_K_L": 6.8, "Q6_K_XL": 7.0,
    "Q5_K_M": 5.7, "Q5_K_S": 5.5, "Q5_K_L": 5.9, "Q5_K_XL": 6.0, "Q5_0": 5.5, "Q5_1": 6.0,
    "Q4_K_M": 4.8, "Q4_K_S": 4.6, "Q4_K_L": 5.0, "Q4_K_XL": 5.0, "Q4_0": 4.55, "Q4_1": 5.0,
    "IQ4_XS": 4.3, "IQ4_NL": 4.5, "MXFP4": 4.25,
    "Q3_K_XL": 4.2, "Q3_K_L": 4.3, "Q3_K_M": 3.9, "Q3_K_S": 3.5,
    "IQ3_M": 3.7, "IQ3_S": 3.45, "IQ3_XS": 3.3, "IQ3_XXS": 3.06,
    "Q2_K_XL": 3.3, "Q2_K_L": 3.2, "Q2_K": 3.0, "Q2_K_S": 2.9, "IQ2_M": 2.7, "IQ2_S": 2.5, "IQ2_XS": 2.3,
    "IQ2_XXS": 2.06, "IQ1_M": 1.75, "IQ1_S": 1.56, "TQ1_0": 1.7,
    # Short spellings some publishers use: "Q4_K" is Q4_K_M's family name, and a
    # bare "q4" / "q8" (e.g. Microsoft's "Phi-3-mini-4k-instruct-q4.gguf") is
    # usually the matching K-quant.
    "Q4_K": 4.8, "Q5_K": 5.7, "Q3_K": 3.9,
    "Q8": 8.5, "Q6": 6.6, "Q5": 5.7, "Q4": 4.8, "Q3": 3.9, "Q2": 3.0,
}

# Our ladder of acceptable quants, best -> smallest. We don't go below ~3.7 bits
# unless nothing else fits (and then the verdict says "tight").
QUANT_PREFERENCE: tuple[str, ...] = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q4_K_S", "Q3_K_M", "IQ3_M")

# Rough share of the full-precision model's quality that survives each quant
# (loosely based on published llama.cpp perplexity / KL-divergence tables).
_QUANT_QUALITY: dict[str, float] = {
    "F32": 1.0, "F16": 1.0, "BF16": 1.0,
    "Q8_K_XL": 0.997, "Q8_0": 0.995,
    "Q6_K_XL": 0.992, "Q6_K_L": 0.991, "Q6_K": 0.99,
    "Q5_K_XL": 0.985, "Q5_K_L": 0.982, "Q5_K_M": 0.98, "Q5_K_S": 0.975, "Q5_1": 0.972, "Q5_0": 0.97,
    "MXFP4": 0.97,  # gpt-oss was *trained* for MXFP4, so it loses very little
    "Q4_K_XL": 0.965, "Q4_K_L": 0.962, "Q4_K_M": 0.96, "IQ4_NL": 0.955, "IQ4_XS": 0.955, "Q4_K_S": 0.95,
    "Q4_1": 0.94, "Q4_0": 0.935,
    "Q3_K_XL": 0.925, "Q3_K_L": 0.915, "Q3_K_M": 0.90, "IQ3_M": 0.89,
    "IQ3_S": 0.865, "Q3_K_S": 0.86, "IQ3_XS": 0.855, "IQ3_XXS": 0.84,
    "Q2_K_XL": 0.85, "Q2_K_L": 0.82, "Q2_K": 0.80, "Q2_K_S": 0.78, "IQ2_M": 0.76, "IQ2_S": 0.72, "IQ2_XS": 0.70,
    "IQ2_XXS": 0.65, "IQ1_M": 0.55, "IQ1_S": 0.5, "TQ1_0": 0.5,
    "Q4_K": 0.96, "Q5_K": 0.98, "Q3_K": 0.90,
    "Q8": 0.995, "Q6": 0.99, "Q5": 0.98, "Q4": 0.955, "Q3": 0.89, "Q2": 0.80,
}


def _normalise_quant(tag: Optional[str]) -> str:
    """"ud-q4_k_xl" -> "Q4_K_XL" (Unsloth's "UD-" = dynamic variant of the same quant)."""
    text = (tag or "").strip().upper()
    return text[3:] if text.startswith("UD-") else text


def quant_bits(tag: Optional[str]) -> Optional[float]:
    """Approximate bits per weight for a quant tag, or None if we don't know it."""
    return QUANT_BITS.get(_normalise_quant(tag))


def quant_quality(tag: Optional[str]) -> float:
    """Rough quality retained (0-1) by a quant: 0.995 for Q8_0, 0.96 for Q4_K_M, 0.90 for Q3_K_M..."""
    key = _normalise_quant(tag)
    if key in _QUANT_QUALITY:
        return _QUANT_QUALITY[key]
    bits = QUANT_BITS.get(key)
    if bits is None:
        return 0.9  # unknown tag: assume a middling quant
    for min_bits, quality in ((8, 0.995), (6.5, 0.99), (5.5, 0.98), (4.7, 0.96), (4.2, 0.95), (3.8, 0.90), (3.4, 0.86), (2.9, 0.80), (2.2, 0.70)):
        if bits >= min_bits:
            return quality
    return 0.55


def estimate_quant_size_gb(params_b: float, quant: Optional[str]) -> Optional[float]:
    """Guess a quant's file size: parameters × bits ÷ 8, plus ~5% for metadata and
    the few tensors kept at higher precision. None if the quant is unknown."""
    bits = quant_bits(quant)
    if bits is None or not params_b or params_b <= 0:
        return None
    return round(params_b * bits / 8 * 1.05, 2)


def _quant_tier(tag: str) -> str:
    """Sort quants into rungs of the ladder: high / standard / low / last / full."""
    bits = quant_bits(tag) or 4.8
    if bits > 9:
        return "full"  # F16/BF16/F32: twice the size of Q8_0 for no visible gain
    quality = quant_quality(tag)
    if quality > 0.965:
        return "high"  # Q5_K_M and up
    if quality >= 0.93:
        return "standard"  # the ~4-bit sweet spot: Q4_K_M, IQ4_XS, Q4_K_S, Q4_0
    if quality >= 0.87:
        return "low"  # ~3.7-3.9 bits: Q3_K_M, IQ3_M
    return "last"  # below ~3.7 bits: only if nothing else fits


# ---------------------------------------------------------------------------
# The memory model (tweak these!)
# ---------------------------------------------------------------------------

# Units: Hugging Face (and file sizes) use decimal gigabytes (1 GB = 10^9
# bytes), while your operating system reports RAM, video memory and disk in
# binary gigabytes (1 GiB = 2^30 bytes, about 7% more), which specs.py
# records. Memory needs are converted to the same binary units before they
# are compared with what the computer has.
GIB_PER_GB = 1e9 / 2**30  # 0.931

OVERHEAD_GB = 0.6  # llama.cpp itself, scratch buffers, tokenizer...
GPU_COMPUTE_BUFFER_GB = 0.3  # extra scratch space when running on a GPU
GPU_VRAM_RESERVE_GB = 0.8  # leave room for your desktop, browser, etc.
OS_RAM_HEADROOM_GB = 2.5  # leave room for the operating system and other apps (macOS / Linux)
WINDOWS_RAM_HEADROOM_GB = 3.5  # Windows typically keeps 3-4 GB busy even when idle
DISK_SPARE_GB = 1.0  # never fill the disk to the brim
MIN_CONTEXT_TOKENS = 2048  # the shortest conversation memory the game can work with
PARTIAL_MIN_GPU_SHARE = 0.4  # split GPU+CPU when the GPU holds >= 40% of the model...
PARTIAL_LOW_GPU_SHARE = 0.15  # ...or >= 15% when the processor-only plan would be a snug squeeze
SPLIT_RECOMMENDED_MIN_GPU_SHARE = 0.75  # a split is Recommended (first choice) only with >= 75% on the GPU
MOE_ROUTING_OVERHEAD = 1.15  # Mixture-of-Experts: routing and small matmuls cost a bit extra
KV_READ_SHARE = 0.25  # per token we also read the KV cache (~a quarter-full context on average)
UPGRADE_MIN_TOKENS_PER_S = 20.0  # only pick Q5/Q6/Q8 if the model stays at least this fast
# Quants within this much quality of the best count as a tie, and the smaller
# file wins: Unsloth's UD-Q8_K_XL keeps some tensors in BF16 (~25% bigger and
# slower than Q8_0) for a difference nobody can see.
QUALITY_TIE = 0.0025
# A slower home (GPU+RAM split instead of all on the GPU, or the CPU instead of
# a split) is only worth it for a clearly better quant - not for half a percent.
PLACEMENT_QUALITY_MARGIN = 0.02
# Never suggest quants below these bits per weight: under ~1.8 bits (IQ1/TQ1)
# every model writes gibberish, and small models fall apart below ~3.3 bits.
MIN_QUANT_BITS = 1.8
SMALL_MODEL_B = 7.0
SMALL_MODEL_MIN_QUANT_BITS = 3.3
_PLACEMENT_RANK = {"gpu": 0, "unified": 0, "partial": 1, "cpu": 2, "none": 3}  # lower = faster home

VERDICT_THRESHOLDS = (("great", 0.60), ("ok", 0.85), ("tight", 1.00))  # need ÷ budget

# How long a whole story turn takes, not just how fast words come out. Each
# turn the model reads ~1,200 tokens of prompt (the story so far, the rules,
# your plan) and writes ~250 (the referee's verdict and the next scene).
# Reading a prompt ("prefill") is much faster than writing - by roughly these
# factors - but on a processor it still adds up.
TURN_ANSWER_TOKENS = 250
TURN_PROMPT_TOKENS = 1200
# (Apple Silicon reads prompts only ~8-13x faster than it writes - llama.cpp's
# own Mac benchmarks - because prompt reading is limited by its compute.)
PREFILL_SPEEDUP = {"gpu": 60.0, "unified": 10.0, "partial": 12.0, "cpu": 8.0}
# The game lets a "thinking" model think out loud only at this speed or faster
# (slower, it asks for the answer straight away), so a thinking model only
# earns its "shows its thinking" bonus when it's this quick.
THINKING_MIN_TOKENS_PER_S = 20.0
# A model whose chat template *forces* thinking (QwQ, DeepSeek-R1 distills,
# Phi-4-reasoning...) can't be asked to answer straight away: every turn it
# first writes this many tokens of reasoning (often far more). Its turn speed
# counts them, and it's kept off the short menu (it's still under "more").
ALWAYS_THINKING_TOKENS = 1000


def thinking_mode(model: ModelEntry) -> str:
    """"none", "switchable" or "always" (see ``ModelEntry.thinking``)."""
    mode = (getattr(model, "thinking", "") or "").strip().lower()
    if mode in ("none", "switchable", "always"):
        return mode
    return "switchable" if model.reasoning else "none"


def always_thinks(model: ModelEntry) -> bool:
    """True if the model thinks at length before every answer, whatever it's asked."""
    return thinking_mode(model) == "always"

# (layers, KV heads, head size) from each model's config.json, for exact
# KV-cache maths. Matched against the repo / base model / name, first hit wins.
_KV_SHAPES: tuple[tuple[str, tuple[int, int, int]], ...] = (
    (r"qwen3-30b-a3b", (48, 4, 128)),
    (r"qwen3-235b-a22b", (94, 4, 128)),
    (r"qwen3-0\.6b", (28, 8, 128)),
    (r"qwen3-1\.7b", (28, 8, 128)),
    (r"qwen3-4b", (36, 8, 128)),
    (r"qwen3-8b", (36, 8, 128)),
    (r"qwen3-14b", (40, 8, 128)),
    (r"qwen3-32b", (64, 8, 128)),
    (r"qwen2\.5-0\.5b", (24, 2, 64)),
    (r"qwen2\.5-1\.5b", (28, 2, 128)),
    (r"qwen2\.5-3b", (36, 2, 128)),
    (r"qwen2\.5-7b", (28, 4, 128)),
    (r"qwen2\.5-14b", (48, 8, 128)),
    (r"qwen2\.5-32b", (64, 8, 128)),
    (r"gpt-oss-20b", (24, 8, 64)),
    (r"gpt-oss-120b", (36, 8, 64)),
    (r"smollm2-1\.7b", (24, 32, 64)),
    (r"smollm2-360m", (32, 5, 64)),
    (r"smollm3-3b", (36, 4, 128)),
    (r"phi-4-mini", (32, 8, 128)),
    (r"phi-4", (40, 10, 128)),  # Phi-4 (14B) and its reasoning versions
    # No grouped-query attention: a KV head per attention head, so the cache is
    # 3-5x bigger than the rule of thumb (fitted to GQA models) would say.
    (r"phi-3(?:\.5)?-mini", (32, 32, 96)),
    (r"phi-3(?:\.5)?-medium", (40, 10, 128)),
    (r"olmo-2-(?:\d{4}-)?1b", (16, 16, 128)),
    (r"olmo-2-(?:\d{4}-)?7b", (32, 32, 128)),
    (r"olmo-2-(?:\d{4}-)?13b", (40, 40, 128)),
    (r"olmo-2-(?:\d{4}-)?32b", (64, 8, 128)),
    (r"mistral-7b", (32, 8, 128)),
    (r"mistral-small|magistral-small|mistral-nemo", (40, 8, 128)),
    (r"granite-3\.\d-8b", (40, 8, 128)),
    (r"granite-3\.\d-2b", (40, 8, 64)),
)


def known_kv_shape(*names: str) -> Optional[tuple[int, int, int]]:
    """The (layers, KV heads, head size) table entry matching any of `names`, or None."""
    haystack = re.sub(r"[\s_]+", "-", " ".join(n for n in names if n).lower())
    for pattern, shape in _KV_SHAPES:
        if re.search(pattern, haystack):
            return shape
    return None


def _kv_shape(model: ModelEntry) -> Optional[tuple[int, int, int]]:
    shape = known_kv_shape(model.key, model.hf_repo, model.base_model or "", model.display_name)
    if shape is None and len(model.kv_shape) == 3 and all(x > 0 for x in model.kv_shape):
        shape = (int(model.kv_shape[0]), int(model.kv_shape[1]), int(model.kv_shape[2]))  # from its GGUF header
    return shape


# Architectures without grouped-query attention (one KV head per attention
# head): when their exact shape is unknown, the cache is estimated for that.
# ("phi3" isn't here: Phi-3-mini has no GQA, but Phi-4-mini - same
# architecture name - does; both are in the table, and the header tells the rest.)
_NO_GQA_ARCHITECTURES = frozenset({"olmo", "olmo2", "stablelm", "gptneox", "gpt2", "baichuan", "orion",
                                   "mpt", "bloom", "gptj", "codeshell", "persimmon", "refact"})


def kv_cache_gb(model: ModelEntry, context_tokens: Optional[int] = None) -> float:
    """Memory for the KV cache: the model's "short-term memory" of the conversation.

    For every token in the context, every layer stores a Key and a Value
    vector per KV head, in 16-bit floats:
    ``2 × layers × kv_heads × head_size × 2 bytes × tokens``.
    We know those numbers for popular families (and read them from the GGUF
    header of others during discovery); otherwise a rule of thumb fitted to
    models with grouped-query attention (a few KV heads shared by many
    attention heads - most modern models):
    ``(0.1 + 0.006 × params_b) GB per 1,024 tokens`` - using the *active*
    parameters for Mixture-of-Experts models, whose attention layers are
    those of a much smaller model. Architectures known to have a KV head per
    attention head (OLMo-2, StableLM...) get ``0.18 × params_b^0.6`` GB per
    1,024 tokens instead, 3-5x more.
    """
    ctx = int(context_tokens or model.context_tokens or 4096)
    if model.native_context:
        ctx = min(ctx, int(model.native_context))
    shape = _kv_shape(model)
    if shape:
        layers, kv_heads, head_dim = shape
        return 2 * layers * kv_heads * head_dim * 2 * ctx / 1e9
    params = model.params_b
    if model.active_params_b and 0 < model.active_params_b < params:
        params = model.active_params_b
    if (model.architecture or "").lower() in _NO_GQA_ARCHITECTURES:
        return max(0.05, 0.18 * max(params, 0.0) ** 0.6 * ctx / 1024)
    return max(0.05, (0.1 + 0.006 * max(params, 0.0)) * ctx / 1024)


def turn_tokens_per_s(tokens_per_s: Optional[float], placement: str, *, thinking_tokens: int = 0) -> float:
    """Words-per-second as the player *feels* it over a whole turn, prompt reading included.

    ``seconds per turn = (answer + thinking) / speed + prompt / (speed x prefill speed-up)``,
    turned back into "tokens per second" of *answer* so it compares with the
    usual thresholds. On a graphics card it's close to the raw speed; on a
    processor, reading the prompt makes each turn noticeably longer - and a
    model that must think first (`thinking_tokens`) is slower still.
    """
    if not tokens_per_s or tokens_per_s <= 0:
        return 0.0
    speedup = PREFILL_SPEEDUP.get(placement, PREFILL_SPEEDUP["cpu"])
    written = TURN_ANSWER_TOKENS + max(0, int(thinking_tokens))
    seconds = written / tokens_per_s + TURN_PROMPT_TOKENS / (tokens_per_s * speedup)
    return TURN_ANSWER_TOKENS / seconds


def model_turn_tokens_per_s(model: ModelEntry, tokens_per_s: Optional[float], placement: str) -> float:
    """`turn_tokens_per_s` for this model, counting the thinking it can't skip."""
    return turn_tokens_per_s(tokens_per_s, placement,
                             thinking_tokens=ALWAYS_THINKING_TOKENS if always_thinks(model) else 0)


def estimate_memory_gb(
    model: ModelEntry, context_tokens: Optional[int] = None, *, weights_gb: Optional[float] = None
) -> float:
    """Memory needed to run `model`: weights + KV cache + overhead (GB).

    `weights_gb` defaults to the model's default quant file size. When the
    model runs on a GPU, add `GPU_COMPUTE_BUFFER_GB` on top (evaluate_fit does).
    """
    if weights_gb is None:
        weights_gb = model.file_size_gb if model.file_size_gb and model.file_size_gb > 0 else None
    if weights_gb is None:
        weights_gb = estimate_quant_size_gb(model.params_b, model.quant) or model.params_b * 0.6
    return float(weights_gb) + kv_cache_gb(model, context_tokens) + OVERHEAD_GB


# ---------------------------------------------------------------------------
# Placement + one "plan" per quant
# ---------------------------------------------------------------------------


@dataclass
class _Plan:
    """Everything we worked out for one (model, quant) pair on one machine."""

    quant: str
    size_gb: float  # download size
    kv_gb: float
    need_gb: float  # total memory needed at this placement
    placement: str
    budget_gb: float  # memory available for that placement
    ratio: float  # need ÷ budget
    verdict: str
    offload_fraction: float  # share on the GPU (1.0 gpu/unified, 0 cpu)
    active_gb: float  # GB read per generated token
    tokens_per_s: Optional[float]
    disk_ok: bool
    on_disk: bool = False  # this exact version is already downloaded
    context: Optional[int] = None  # conversation memory planned for (None = the model's usual)


def _dedicated_vram_gb(specs: SystemSpecs) -> float:
    """Usable dedicated VRAM. llama.cpp splits a model across several GPUs of the
    same kind, so we add up same-vendor cards with >= 4 GB each."""
    main = perf.primary_gpu(specs)
    if main is None or main.vendor == "apple":
        return 0.0
    same = [g.vram_gb for g in specs.gpus if g.vendor == main.vendor and g.vram_gb >= 4 and g is not main]
    return main.vram_gb + sum(same)


def _os_headroom_gb(specs: SystemSpecs) -> float:
    return WINDOWS_RAM_HEADROOM_GB if (specs.os_name or "").lower().startswith("win") else OS_RAM_HEADROOM_GB


def _ram_budget_gb(specs: SystemSpecs) -> float:
    return max(0.0, specs.ram_total_gb - _os_headroom_gb(specs))


def _place(specs: SystemSpecs, base_need_gb: float) -> tuple[str, float, float, float]:
    """Decide where a model needing `base_need_gb` runs.

    Returns (placement, need_gb, budget_gb, offload_fraction). Order of
    preference: whole model on the GPU, Apple unified memory, GPU+RAM split,
    CPU only, or "none" if it doesn't fit anywhere.
    """
    gpu_need = base_need_gb + GPU_COMPUTE_BUFFER_GB
    ram_budget = _ram_budget_gb(specs)
    gpu = perf.primary_gpu(specs)

    if gpu is not None and gpu.vendor == "apple":
        if gpu_need <= gpu.vram_gb:
            return "unified", gpu_need, gpu.vram_gb, 1.0
        # Past the share macOS lets the GPU use, llama.cpp's Metal build keeps
        # what fits on the GPU and runs the rest on the processor - in the same
        # memory. So it's a split ("partial"), never a separate CPU plan.
        if gpu.vram_gb >= PARTIAL_MIN_GPU_SHARE * gpu_need and gpu_need <= ram_budget:
            return "partial", gpu_need, ram_budget, gpu.vram_gb / gpu_need
        return "none", base_need_gb, max(ram_budget, gpu.vram_gb), 0.0
    elif gpu is not None:
        vram_budget = _dedicated_vram_gb(specs) - GPU_VRAM_RESERVE_GB
        if vram_budget > 0 and gpu_need <= vram_budget:
            return "gpu", gpu_need, vram_budget, 1.0
        share = vram_budget / gpu_need if vram_budget > 0 else 0.0
        split_fits = vram_budget > 0 and gpu_need - vram_budget <= ram_budget
        cpu_fits = base_need_gb <= ram_budget
        cpu_snug = cpu_fits and base_need_gb > VERDICT_THRESHOLDS[1][1] * ram_budget
        # A split usually needs the GPU to hold a fair share (PARTIAL_MIN_GPU_SHARE)
        # to be worth it. But when the processor-only plan would fill the RAM, a
        # smaller share still takes that pressure off (the engine uses the card
        # anyway), and when the model only fits across both, any share will do.
        if split_fits and (share >= PARTIAL_MIN_GPU_SHARE or (cpu_snug and share >= PARTIAL_LOW_GPU_SHARE)
                           or not cpu_fits):
            return "partial", gpu_need, vram_budget + ram_budget, share
    if base_need_gb <= ram_budget:
        return "cpu", base_need_gb, ram_budget, 0.0

    # Doesn't fit: report against the biggest budget we could have offered.
    budgets = [ram_budget]
    if gpu is not None and gpu.vendor == "apple":
        budgets.append(gpu.vram_gb)
    elif gpu is not None:
        budgets.append(max(0.0, _dedicated_vram_gb(specs) - GPU_VRAM_RESERVE_GB) + ram_budget)
    return "none", base_need_gb, max(budgets), 0.0


def _is_apple(specs: SystemSpecs) -> bool:
    gpu = perf.primary_gpu(specs)
    return gpu is not None and gpu.vendor == "apple"


def _shares_system_ram(specs: SystemSpecs, placement: str) -> bool:
    """Is the memory this placement fills the computer's own RAM?"""
    return placement in ("cpu", "unified") or (placement == "partial" and _is_apple(specs))


def _verdict_for(ratio: float) -> str:
    for verdict, limit in VERDICT_THRESHOLDS:
        if ratio <= limit:
            return verdict
    return "no"


def _active_share(model: ModelEntry) -> float:
    """Share of the weights read per token (1.0 for dense models)."""
    if model.active_params_b and model.params_b and 0 < model.active_params_b < model.params_b:
        return min(1.0, model.active_params_b / model.params_b * MOE_ROUTING_OVERHEAD)
    return 1.0


def _plan(specs: SystemSpecs, model: ModelEntry, quant: str, size_gb: float, *,
          on_disk: bool = False, context: Optional[int] = None) -> _Plan:
    """Where (quant, size) runs and how fast. `on_disk`: already downloaded (needs no disk space)."""
    kv = kv_cache_gb(model, context)
    base_need = (size_gb + kv) * GIB_PER_GB + OVERHEAD_GB  # in the computer's own (binary) units
    placement, need, budget, offload = _place(specs, base_need)
    ratio = need / budget if budget > 0 else math.inf
    if placement == "partial" and _is_apple(specs):
        # A Mac's split runs past the GPU's share into the rest of the same RAM,
        # so it's always a squeeze of the whole computer's memory: snug.
        ratio = max(1.0, ratio)
    elif placement == "partial":
        # A split fills the graphics card completely (that's what a split is), so
        # it's never "great"; how snug it is depends on the part that goes to
        # system RAM. A small spill into plenty of free RAM can't fail - it's only
        # a bit slower, and the speed estimate already allows for that.
        spill = need * (1.0 - offload)
        ram_budget = _ram_budget_gb(specs)
        ratio = max(VERDICT_THRESHOLDS[1][1], spill / ram_budget if ram_budget > 0 else math.inf)
    verdict = "no" if placement == "none" else _verdict_for(ratio)
    # < 0 = free space unknown; a file that's already here needs no room at all.
    disk_ok = on_disk or specs.disk_free_gb < 0 or specs.disk_free_gb >= size_gb * GIB_PER_GB + DISK_SPARE_GB
    if not disk_ok:
        verdict = "no"
    active = size_gb * _active_share(model) + KV_READ_SHARE * kv  # speed maths stays in decimal GB/s
    tps = None
    if placement != "none":
        tps = perf.estimate_tokens_per_s(specs, active_gb=active, placement=placement, offload_fraction=offload)
    return _Plan(quant, size_gb, kv, need, placement, budget, ratio, verdict, offload, active, tps, disk_ok,
                 on_disk, context)


def _quant_options(model: ModelEntry) -> list[tuple[str, float]]:
    """(quant, size GB) pairs for a model, estimating sizes we don't know."""
    raw = list(model.quant_options) or [(model.quant, model.file_size_gb)]
    seen: set[str] = set()
    options = []
    for quant, size in raw:
        if not quant or quant.upper() in seen:
            continue
        if not size or size <= 0:
            size = estimate_quant_size_gb(model.params_b, quant) or 0.0
        if size > 0:
            seen.add(quant.upper())
            options.append((quant, float(size)))
    return options


def _best(plans: Iterable[_Plan], keep: Callable[[_Plan], bool] = lambda p: True) -> Optional[_Plan]:
    """Highest-quality plan that passes `keep`. Quants within QUALITY_TIE of the
    best count as a tie, and the smallest file among them wins."""
    candidates = [p for p in plans if keep(p)]
    if not candidates:
        return None
    top = max(quant_quality(p.quant) for p in candidates)
    tied = [p for p in candidates if quant_quality(p.quant) >= top - QUALITY_TIE - 1e-9]
    return min(tied, key=lambda p: (p.size_gb, -quant_quality(p.quant)))


def _pick_home(plans: list[_Plan], keep: Callable[[_Plan], bool] = lambda p: True) -> Optional[_Plan]:
    """Like `_best`, but the fastest home comes first: all on the GPU, then a
    GPU+RAM split, then the CPU. A slower home only wins with a clearly
    better quant (PLACEMENT_QUALITY_MARGIN), and a comfortable fit beats a
    snug one in the same home."""
    candidates = [p for p in plans if keep(p)]
    if not candidates:
        return None
    top = min(_PLACEMENT_RANK[p.placement] for p in candidates)
    home = [p for p in candidates if _PLACEMENT_RANK[p.placement] == top]
    chosen = _best(home, _comfortable) or _best(home)
    assert chosen is not None
    slower = [p for p in candidates if _PLACEMENT_RANK[p.placement] > top
              and quant_quality(p.quant) >= quant_quality(chosen.quant) + PLACEMENT_QUALITY_MARGIN]
    return _best(slower, _comfortable) or _best(slower) or chosen


def _quant_allowed(model: ModelEntry, quant: str) -> bool:
    """Is this quant ever worth suggesting for this model? (See MIN_QUANT_BITS.)"""
    bits = quant_bits(quant)
    if bits is None:
        return True
    if bits < MIN_QUANT_BITS:
        return False
    return bits >= SMALL_MODEL_MIN_QUANT_BITS or _effective_params_b(model) >= SMALL_MODEL_B


def _comfortable(plan: _Plan) -> bool:
    """Fits with room to spare."""
    return plan.verdict in ("great", "ok")


def _spills_gracefully(plan: _Plan) -> bool:
    """On a GPU, a snug fit is fine: llama.cpp can move a layer or two to system RAM."""
    return plan.placement in ("gpu", "unified", "partial")


Downloaded = Callable[[ModelEntry, str], bool]  # "is this (model, quant) already on disk?"


def _choose_plan(specs: SystemSpecs, model: ModelEntry, downloaded: Optional[Downloaded] = None,
                 context: Optional[int] = None, *, floor: bool = True) -> Optional[_Plan]:
    """The quant ladder. See `choose_quant` for the plain-English version."""
    def have(quant: str) -> bool:
        try:
            return bool(downloaded and downloaded(model, quant))
        except Exception:
            return False

    plans = [_plan(specs, model, q, s, on_disk=have(q), context=context) for q, s in _quant_options(model)
             if not floor or _quant_allowed(model, q)]
    usable = [p for p in plans if p.verdict != "no"]  # fits in memory AND on disk
    tier = {p.quant: _quant_tier(p.quant) for p in usable}
    # 0. A version that's already downloaded (and isn't a last resort) wins:
    #    no new download, and no disk space needed.
    ready = [p for p in usable if p.on_disk and tier[p.quant] != "last"]
    if ready:
        return _pick_home(ready)
    standard = [p for p in usable if tier[p.quant] == "standard"]
    low = [p for p in usable if tier[p.quant] == "low"]
    high = [p for p in usable if tier[p.quant] == "high"]
    full = [p for p in usable if tier[p.quant] == "full"]
    last = [p for p in usable if tier[p.quant] == "last"]

    base = (
        # 1-2. a ~4-bit quant that fits with room to spare, or snugly on a GPU
        #      (llama.cpp can move a layer or two to RAM) - fastest home first
        _pick_home(standard, lambda p: _comfortable(p) or _spills_gracefully(p))
        or _pick_home(low, _comfortable)  # 3. ~3.7-3.9 bits with room to spare...
        or _pick_home(standard + low)  # ...or any snug fit at >= ~3.7 bits
        or _pick_home(high, _comfortable)  # 4. the repo only offers bigger quants
        or _pick_home(full, _comfortable)
        or _pick_home(high)
        or _pick_home(full)
        or _pick_home(last)  # 5. last resort: heavily compressed (evaluate_fit calls it "tight")
    )
    if base is None or not _comfortable(base):
        return base

    # 6. Upgrade to a higher-quality quant if it still fits comfortably, stays
    # fast (>= 20 tok/s and at least half the base pick's speed), and isn't
    # pushed off the GPU (e.g. from "all on the GPU" to a GPU+RAM split).
    base_speed = base.tokens_per_s or 0.0
    upgrade = _best(
        high,
        lambda p: _comfortable(p)
        and quant_quality(p.quant) > quant_quality(base.quant)
        and _PLACEMENT_RANK[p.placement] <= _PLACEMENT_RANK[base.placement]
        and (p.tokens_per_s or 0.0) >= max(UPGRADE_MIN_TOKENS_PER_S, 0.5 * base_speed),
    )
    return upgrade or base


def choose_quant(specs: SystemSpecs, model: ModelEntry, *, downloaded: Optional[Downloaded] = None) -> Optional[tuple[str, float]]:
    """Pick the best (quant, size_gb) of `model` for this machine, or None if nothing fits.

    The ladder, in plain English:

    0. A version that is already downloaded (`downloaded(model, quant)`) and
       fits wins outright: nothing new to fetch, no disk space needed.
    1. Start from the ~4-bit sweet spot (Q4_K_M, else IQ4_XS / Q4_K_S) if it fits
       with room to spare ("great"/"ok"), or snugly on a GPU - llama.cpp can
       move a layer or two to system RAM if it's short. The fastest home wins:
       all on the GPU, then a GPU+RAM split, then the CPU; a slower one only
       for a clearly better quant (2%+), never for half a percent.
    2. Else drop to ~3.7-3.9 bits (Q3_K_M, IQ3_M), comfortable first, then snug.
    3. If the repo only offers bigger quants (say, just Q8_0), use those.
    4. Last resort: anything smaller that fits at all (the verdict will say "tight").
    5. Finally, go *up* to Q5/Q6/Q8 only if that still fits comfortably and
       stays fast (≥ 20 tokens/s and at least half the speed of the 4-bit pick)
       without being pushed off the GPU.
       Big machines get sharper models; slow ones keep their speed.

    Quants within 0.25% quality of each other count as a tie and the smaller
    file wins (so Q8_0 beats the ~25% bigger UD-Q8_K_XL). Quants under 1.8
    bits, or under 3.3 bits for models below ~7B, are never suggested: they
    write gibberish. Sizes come from `model.quant_options` (or
    `model.quant`/`file_size_gb`); unknown sizes are estimated as params ×
    bits ÷ 8 × 1.05. "Fits" includes having the download size + 1 GB free on disk.
    """
    plan = _choose_plan(specs, model, downloaded)
    return (plan.quant, plan.size_gb) if plan else None


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

W_QUALITY = 12.0  # points per doubling of (effective) parameters, times quant quality
W_SPEED = 20.0  # full points at >= 15 tok/s (comfortable reading), zero at <= 3 tok/s, log scale between
SPEED_FULL_AT = 15.0
SPEED_ZERO_AT = 3.0
SLOW_PENALTY_PER_TOKEN = 2.5  # extra penalty per tok/s below 8
VERY_SLOW_PENALTY = 30.0  # below 3 tok/s: effectively unplayable
W_HEADROOM = 6.0  # up to 6 points for leaving memory free
TIGHT_PENALTY = 8.0  # a snug fit might fail if other apps grab memory
PARTIAL_PENALTY = 2.0  # GPU+RAM splits are more fragile and use lots of RAM
W_POPULARITY = 4.0  # up to 4 points for ~1M Hugging Face downloads
FAST_ENOUGH_TPS = 25.0  # a turn speed that feels quick (no big-download penalty; "fastest" must be 1.5x quicker still)
BIG_DOWNLOAD_GB = 8.0
BIG_DOWNLOAD_PENALTY_PER_GB = 0.2
BONUS_CURATED = 3.0  # in our hand-checked seed list
BONUS_TRUSTED = 1.5  # from a well-known publisher
BONUS_REASONING = 2.0  # shows its thinking (and can skip it when time is short) - great for the review
PENALTY_ALWAYS_THINKING = 10.0  # can't be asked to skip its thinking: slow turns, and the story may not fit
BONUS_INSTRUCT = 1.0  # tuned to follow instructions / chat
PENALTY_GATED = 3.0  # needs a Hugging Face login to download
PENALTY_NOT_PERMISSIVE = 2.0  # only shown with --all-licenses

_CURATED_REPOS: set[str] = set()  # filled after MODEL_CATALOG is defined
_INSTRUCT_RE = re.compile(r"instruct|chat|[-_]it\b|assistant", re.IGNORECASE)
_CHAT_FAMILIES = ("qwen3", "gpt-oss", "phi-4-mini", "smollm3", "granite", "magistral")


def _effective_params_b(model: ModelEntry) -> float:
    """Dense-equivalent size. Modern Mixture-of-Experts models perform roughly
    like a dense model of total^0.6 × active^0.4 parameters (e.g. 30B with
    3.3B active ≈ 12.5B) - a little better than the older sqrt(total × active) rule."""
    total = max(model.params_b, 0.01)
    if model.active_params_b and 0 < model.active_params_b < total:
        return total ** 0.6 * model.active_params_b ** 0.4
    return total


def _quality_points(model: ModelEntry, quant: Optional[str]) -> float:
    return W_QUALITY * math.log2(1 + _effective_params_b(model)) * quant_quality(quant or model.quant)


def _is_curated(model: ModelEntry) -> bool:
    return model.source == "curated" or model.hf_repo.lower() in _CURATED_REPOS


_CURATED_LINEAGES: Optional[frozenset[tuple[str, str]]] = None


def _earns_curated_bonus(model: ModelEntry) -> bool:
    """One of our hand-checked seeds - or a newer dated release of the same original
    from the same publisher (e.g. Qwen3 4B Instruct 2507 next to the Qwen3 4B seed),
    so the frozen seed list never outranks its own successor on the Hub."""
    global _CURATED_LINEAGES
    if _is_curated(model):
        return True
    if _CURATED_LINEAGES is None:
        _CURATED_LINEAGES = frozenset((m.hf_repo.split("/", 1)[0].lower(), _variant_family(m)) for m in MODEL_CATALOG)
    return (model.hf_repo.split("/", 1)[0].lower(), _variant_family(model)) in _CURATED_LINEAGES


def _is_trusted(model: ModelEntry) -> bool:
    publisher = model.hf_repo.split("/", 1)[0].lower()
    return publisher in {p.lower() for p in TRUSTED_PUBLISHERS}


def _is_instruction_tuned(model: ModelEntry) -> bool:
    if model.source == "curated":
        return True
    text = " ".join((model.hf_repo, model.display_name, model.base_model or ""))
    family = f"{model.family} {model.hf_repo}".lower()
    return bool(_INSTRUCT_RE.search(text)) or any(f in family for f in _CHAT_FAMILIES)


def score_fit(model: ModelEntry, *, quant: Optional[str], verdict: str, tokens_per_s: Optional[float],
              ratio: float, download_gb: Optional[float], placement: str = "gpu") -> float:
    """Overall "how good a pick is this?" score. Higher is better.

    score = quality + speed + headroom + popularity + bonuses - penalties

    - quality  = W_QUALITY × log2(1 + effective params in billions) × quant quality
                 (effective params = total^0.6 × active^0.4 for Mixture-of-Experts)
    - speed    = W_SPEED × position of the *turn* speed (tok/s including
                 prompt reading, see `turn_tokens_per_s`) between 3 and 15
                 on a log scale, minus SLOW_PENALTY_PER_TOKEN for every tok/s
                 below 8, and VERY_SLOW_PENALTY below 3 tok/s
    - headroom = W_HEADROOM × (1 - need/budget), or -TIGHT_PENALTY if tight
                 (and -PARTIAL_PENALTY when split between GPU and RAM)
    - popularity = W_POPULARITY × log10(1 + downloads) / 6 (capped at 1M)
    - bonuses: curated seed (or its newer dated release from the same
      publisher), trusted publisher, instruction-tuned, and shows
      reasoning (only when it's fast enough for the game to let it think, and
      only for models that can also be asked to skip it)
    - penalties: gated repo, non-permissive license, big downloads (> 8 GB,
      unless it runs at >= FAST_ENOUGH_TPS anyway),
      and a model that *always* thinks first (its turn speed also counts
      ALWAYS_THINKING_TOKENS of thinking per turn)

    Models that don't fit ("no") score below -100 (lower the further over
    budget they are), so they sort to the bottom. All weights are module constants: tweak and see!
    """
    if verdict == "no":
        return round(-100.0 - min(ratio, 50.0), 2)

    quality = _quality_points(model, quant)

    raw_tps = tokens_per_s or 0.0
    tps = max(model_turn_tokens_per_s(model, raw_tps, placement), 0.01)
    position = math.log(tps / SPEED_ZERO_AT) / math.log(SPEED_FULL_AT / SPEED_ZERO_AT)
    speed = W_SPEED * min(1.0, max(0.0, position))
    speed -= SLOW_PENALTY_PER_TOKEN * max(0.0, 8.0 - tps)
    if tps < 3.0:
        speed -= VERY_SLOW_PENALTY

    headroom = -TIGHT_PENALTY if verdict == "tight" else W_HEADROOM * max(0.0, 1.0 - ratio)
    if placement == "partial":
        headroom -= PARTIAL_PENALTY
    popularity = W_POPULARITY * min(1.0, math.log10(1 + max(model.downloads, 0)) / 6)

    bonus = 0.0
    bonus += BONUS_CURATED if _earns_curated_bonus(model) else 0.0
    bonus += BONUS_TRUSTED if _is_trusted(model) else 0.0
    mode = thinking_mode(model)
    bonus += BONUS_REASONING if mode == "switchable" and raw_tps >= THINKING_MIN_TOKENS_PER_S else 0.0
    bonus -= PENALTY_ALWAYS_THINKING if mode == "always" else 0.0
    bonus += BONUS_INSTRUCT if _is_instruction_tuned(model) else 0.0
    bonus -= PENALTY_GATED if model.gated else 0.0
    bonus -= 0.0 if is_permissive(model.license) else PENALTY_NOT_PERMISSIVE
    if tps < FAST_ENOUGH_TPS:  # a big download is worth it for a model that's quick anyway
        bonus -= BIG_DOWNLOAD_PENALTY_PER_GB * max(0.0, (download_gb or 0.0) - BIG_DOWNLOAD_GB)

    return round(quality + speed + headroom + popularity + bonus, 2)


# ---------------------------------------------------------------------------
# Fit, rank, shortlist
# ---------------------------------------------------------------------------


def _fmt_gb(value: float) -> str:
    return f"{value:.1f}" if value < 10 else f"{value:.0f}"


def _reason(specs: SystemSpecs, model: ModelEntry, plan: _Plan, verdict: str) -> str:
    """One friendly sentence explaining the verdict."""
    need, budget = _fmt_gb(plan.need_gb), _fmt_gb(plan.budget_gb)
    if not plan.disk_ok:
        return (
            f"The {_fmt_gb(plan.size_gb)} GB download needs more free disk space than you have "
            f"({specs.disk_free_gb:.0f} GB free; we keep 1 GB spare)."
        )
    if plan.placement == "none":
        return f"Too big for this computer: it needs about {need} GB of memory and you have about {budget} GB to spare."

    tps = plan.tokens_per_s or 0.0
    if tps >= 1.5:
        speed = f"roughly {tps:.0f} tokens/s"
    else:
        speed = "about 1 token/s" if tps >= 0.8 else "well under 1 token/s"
    gpu = perf.primary_gpu(specs)
    if plan.placement == "gpu":
        where = f"Fits on your {gpu.name if gpu else 'graphics card'} (needs ~{need} of {budget} GB video memory)"
    elif plan.placement == "unified":
        where = f"Fits in your Mac's unified memory (needs ~{need} of {budget} GB usable)"
    elif plan.placement == "partial" and gpu is not None and gpu.vendor == "apple":
        where = (
            f"Too big for the share of memory your Mac lets its GPU use, so ~{plan.offload_fraction:.0%} of it "
            f"runs on the GPU and the rest on the processor (needs ~{need} of {budget} GB)"
        )
    elif plan.placement == "partial":
        spill = plan.need_gb * (1.0 - plan.offload_fraction)
        where = (
            f"Splits between your graphics card (~{plan.offload_fraction:.0%} of the model, filling its memory) "
            f"and system RAM (~{_fmt_gb(spill)} of the {_fmt_gb(_ram_budget_gb(specs))} GB we can spare)"
        )
    else:
        where = f"Runs on your CPU using ~{need} of the {budget} GB of RAM we can spare"
    sentence = f"{where} — {speed}"
    if plan.context and plan.context < (model.context_tokens or plan.context):
        sentence += f", with a shorter conversation memory ({plan.context:,} tokens) so it fits"
    if plan.on_disk:
        sentence += " (already downloaded)"
    if always_thinks(model):
        sentence += (", but it always thinks at length before it answers (it can't be asked not to), "
                     "so every turn takes several times longer")
    elif _quant_tier(plan.quant) == "last":
        sentence += ", but only a heavily compressed version fits, so the writing may be rough"
    elif verdict == "tight":
        sentence += "; it's a snug fit, so close other big apps first"
    elif perf.speed_label(tps) in ("slow", "very slow"):
        sentence += ", so expect some waiting"
    return sentence + "."


def speed_breakdown(specs: SystemSpecs, fit: FitResult) -> Optional[tuple[float, float, float]]:
    """(GB read per token, efficiency x bandwidth in GB/s, fixed seconds per token)
    behind a fit's speed estimate - for fits all on one kind of memory
    ("gpu", "unified" or "cpu"); None for splits and non-fits."""
    if fit.placement not in ("gpu", "unified", "cpu") or not fit.download_gb:
        return None
    plan = _plan(specs, fit.model, fit.quant or fit.model.quant, fit.download_gb, context=fit.context_tokens)
    if plan.placement != fit.placement:
        return None
    placement = fit.placement
    if placement in ("gpu", "unified"):
        gpu = perf.primary_gpu(specs)
        if gpu is None:
            return None
        placement = "unified" if gpu.vendor == "apple" else "gpu"
    bandwidth, _source = perf.bandwidth_for(specs, placement)
    return plan.active_gb, perf.efficiency_for(specs, placement) * bandwidth, perf.OVERHEAD_S_PER_TOKEN[placement]


def size_unknown(model: ModelEntry) -> bool:
    """True when we know neither the parameter count nor any file size (so nothing can be estimated)."""
    return not _quant_options(model)


def evaluate_fit(specs: SystemSpecs, model: ModelEntry, *, downloaded: Optional[Downloaded] = None,
                 quant_floor: bool = True) -> FitResult:
    """How well does `model` fit this machine? Picks a quant, placement, speed and score.

    If no version fits with the model's usual conversation memory (context),
    or only a heavily compressed one does, a shorter one (MIN_CONTEXT_TOKENS)
    is tried first - on small computers that's the difference between a
    readable model and gibberish, or none at all.
    `downloaded(model, quant)` says which versions are already on disk.
    ``quant_floor=False`` also allows quants we'd never suggest (see
    MIN_QUANT_BITS) - for when the player asked for that exact quant.
    """
    if size_unknown(model):
        # Say so plainly, rather than estimating from a made-up size.
        return FitResult(
            model=model, verdict="no", placement="none", est_memory_gb=0.0, est_speed="n/a",
            reason="I couldn't work out how big this model is, so I can't say whether it fits.",
            quant=model.quant or None, download_gb=None, est_tokens_per_s=None, score=-150.0,
        )
    plan = _choose_plan(specs, model, downloaded, floor=quant_floor)
    if ((plan is None or _quant_tier(plan.quant) == "last")
            and (model.context_tokens or 4096) > MIN_CONTEXT_TOKENS):
        # A shorter conversation memory beats a heavily compressed model: try it
        # before settling for a last-resort (sub-3.7-bit) quant.
        short = _choose_plan(specs, model, downloaded, context=MIN_CONTEXT_TOKENS, floor=quant_floor)
        if short is not None and (plan is None or _quant_tier(short.quant) != "last"):
            plan = short
    if plan is None:
        # Nothing fits: explain using the smallest version worth running.
        all_options = _quant_options(model) or [(model.quant or "Q4_K_M", max(model.params_b * 0.6, 0.1))]
        options = [o for o in all_options if not quant_floor or _quant_allowed(model, o[0])]
        if not options:
            return FitResult(
                model=model, verdict="no", placement="none", est_memory_gb=0.0, est_speed="n/a",
                reason=("Only extremely compressed versions of this model are on offer, and at that size it "
                        "would write gibberish - so I won't suggest it."),
                quant=model.quant or None, download_gb=None, est_tokens_per_s=None, score=-150.0,
            )
        quant, size = min(options, key=lambda option: option[1])
        plan = _plan(specs, model, quant, size, context=MIN_CONTEXT_TOKENS
                     if (model.context_tokens or 4096) > MIN_CONTEXT_TOKENS else None)
        plan.verdict = "no"
    verdict = plan.verdict
    if verdict != "no" and _quant_tier(plan.quant) == "last":
        verdict = "tight"  # only a heavily-compressed version fits
    tps = round(plan.tokens_per_s, 1) if plan.tokens_per_s is not None else None
    return FitResult(
        model=model,
        verdict=verdict,  # type: ignore[arg-type]
        placement=plan.placement,  # type: ignore[arg-type]
        est_memory_gb=round(plan.need_gb, 1),
        est_speed=perf.speed_label(plan.tokens_per_s),
        reason=_reason(specs, model, plan, verdict),
        quant=plan.quant,
        download_gb=round(plan.size_gb, 2),
        est_tokens_per_s=tps,
        score=score_fit(
            model, quant=plan.quant, verdict=verdict, tokens_per_s=plan.tokens_per_s,
            ratio=plan.ratio, download_gb=0.0 if plan.on_disk else plan.size_gb, placement=plan.placement,
        ),
        context_tokens=plan.context,
        gpu_share=round(plan.offload_fraction, 3) if plan.placement != "none" else None,
        shares_system_ram=_shares_system_ram(specs, plan.placement),
    )


def rank_models(specs: SystemSpecs, catalog: Optional[list[ModelEntry]] = None, *,
                downloaded: Optional[Downloaded] = None) -> list[FitResult]:
    """Evaluate every model and sort: runnable ones by score (best first), then the rest."""
    models = MODEL_CATALOG if catalog is None else catalog
    fits = [evaluate_fit(specs, m, downloaded=downloaded) for m in models]
    runnable = sorted((f for f in fits if f.verdict != "no"), key=lambda f: f.score, reverse=True)
    too_big = sorted((f for f in fits if f.verdict == "no"), key=lambda f: f.est_memory_gb)
    return runnable + too_big


def _base_identity(model: ModelEntry) -> str:
    """A key that is the same for every GGUF conversion of one original model."""
    source = model.base_model or model.hf_repo or model.key
    name = source.split("/")[-1].lower()
    for publisher in (p.lower() for p in TRUSTED_PUBLISHERS + ("openai", "google", "meta-llama")):
        name = name.removeprefix(publisher + "_")  # bartowski's "mistralai_Mistral-..." style
    name = re.sub(r"[-_.](gguf|instruct|chat|it)\b", "", name)
    return re.sub(r"[\s_]+", "-", name).strip("-")


_DATE_SUFFIX_RE = re.compile(r"-(?:20)?\d{4}(?:\d{2})?(?=-|$)")  # "-2507", "-2506", "-20250514"
_VARIANT_WORD_RE = re.compile(r"-(?:instruct|thinking|reasoning|chat|it|hf)(?=-|$)")


def _variant_family(model: ModelEntry) -> str:
    """Like `_base_identity`, but also the same for a model's dated refreshes and its
    thinking / instruct twins ("Qwen3-4B", "Qwen3-4B-Instruct-2507", "Qwen3-4B-Thinking-2507"),
    so the short menu doesn't spend several of its few slots on near-identical rows."""
    name = _base_identity(model)
    for _ in range(3):  # "-instruct-2507" -> "-2507" -> ""
        name = _VARIANT_WORD_RE.sub("", _DATE_SUFFIX_RE.sub("", name))
    return name


def _lineage(model: ModelEntry) -> str:
    """Roughly "which base model is this built on": its architecture and size.
    Fine-tunes of one base (say three Mistral-Small-24B derivatives) share it."""
    arch = (model.architecture or model.family or "?").lower()
    return f"{arch}-{round(model.params_b)}"


def _turn_tps(f: FitResult) -> float:
    return model_turn_tokens_per_s(f.model, f.est_tokens_per_s, f.placement)


def _pick_recommended(viable: list[FitResult]) -> Optional[FitResult]:
    """Best score among comfortable, playable fits; never anything under 3 tok/s (turn speed)."""
    def tps(f: FitResult) -> float:
        return _turn_tps(f)

    def settled(f: FitResult) -> bool:
        # A split only counts as comfortable here when the card holds most of it.
        return f.placement != "partial" or (f.gpu_share or 0.0) >= SPLIT_RECOMMENDED_MIN_GPU_SHARE

    tiers: list[Callable[[FitResult], bool]] = [
        lambda f: f.verdict in ("great", "ok") and settled(f) and tps(f) >= 8,
        lambda f: f.verdict in ("great", "ok") and settled(f) and tps(f) >= 3,
        lambda f: tps(f) >= 3,
    ]
    for keep in tiers:
        candidates = [f for f in viable if keep(f)]
        if candidates:
            return max(candidates, key=lambda f: f.score)
    return None


DIVERSITY_MARGIN = 12.0  # how many score points we'll give up to show a different family
FASTEST_QUALITY_SHARE = 0.70  # "fastest" keeps at least this share of the recommended pick's quality points...
FASTEST_MIN_EFFECTIVE_B = 2.0  # ...and is never smaller than this (dense-equivalent billions): no toy models
FASTEST_MIN_SPEEDUP = 1.25  # ...and is clearly quicker than it


FASTEST_NEAR_TIE = 0.95  # speeds within 5% of the quickest count as a tie, broken by score


def _squeezed(f: FitResult) -> bool:
    """A snug fit that fills the computer's own RAM - on the processor, in a Mac's
    unified memory, or a split whose RAM side is the snug part (the only way a
    split on a graphics card is "tight"): the whole computer would be short of
    memory, and overflow means swapping."""
    return f.verdict == "tight" and (f.shares_system_ram or f.placement in ("cpu", "partial"))


def _fastest_order(pool: list[FitResult]) -> list[FitResult]:
    """Quickest first - but near-ties (within 5%, say two conversions of one model
    whose files differ by 2%) go to the better-scoring pick."""
    if not pool:
        return []
    top = max(_turn_tps(f) for f in pool)
    near = [f for f in pool if _turn_tps(f) >= FASTEST_NEAR_TIE * top]
    rest = [f for f in pool if _turn_tps(f) < FASTEST_NEAR_TIE * top]
    return sorted(near, key=lambda f: f.score, reverse=True) + sorted(rest, key=_turn_tps, reverse=True)


_CURATED_FAMILIES: Optional[frozenset[str]] = None


def _is_known_original(model: ModelEntry) -> bool:
    """One of our seed models, or another conversion / dated refresh of one."""
    global _CURATED_FAMILIES
    if _CURATED_FAMILIES is None:
        _CURATED_FAMILIES = frozenset(_variant_family(m) for m in MODEL_CATALOG)
    return _is_curated(model) or _variant_family(model) in _CURATED_FAMILIES


def _originals_first(fits: list[FitResult]) -> list[FitResult]:
    """`fits` in score order, except that a fine-tune we don't recognise never comes
    before the original model of its lineage (when that original is nearly as good):
    otherwise an agent or medical fine-tune of Qwen3-8B could claim the lineage's
    menu slot and push Qwen3-8B itself off the menu."""
    out: list[FitResult] = []
    placed: set[int] = set()
    for fit in fits:
        if id(fit) in placed:
            continue
        if not _is_known_original(fit.model):
            original = next((g for g in fits if id(g) not in placed and g is not fit
                             and _is_known_original(g.model) and _lineage(g.model) == _lineage(fit.model)
                             and g.score >= fit.score - DIVERSITY_MARGIN), None)
            if original is not None:
                out.append(original)
                placed.add(id(original))
        out.append(fit)
        placed.add(id(fit))
    return out


def disk_space_needed(specs: SystemSpecs, catalog: Optional[list[ModelEntry]] = None) -> Optional[float]:
    """When free disk space alone keeps every model out: the free space (GB, as the computer counts it)
    the smallest model that would otherwise run needs. None when some model fits, or when memory (not
    the disk) is what rules them all out - so "not enough memory" is never said about a full disk.
    """
    if specs.disk_free_gb < 0:
        return None  # free space unknown: never the reason
    ranked = rank_models(specs, catalog)
    if not ranked or any(f.verdict != "no" for f in ranked):
        return None
    roomy = replace(specs, disk_free_gb=-1.0)  # the same computer with room to spare
    fits = [f for f in rank_models(roomy, catalog) if f.verdict != "no" and f.download_gb]
    if not fits:
        return None
    smallest = min(f.download_gb for f in fits)
    return round(smallest * GIB_PER_GB + DISK_SPARE_GB, 1)


def runnable_by_engine(models: list[ModelEntry], architectures: Optional[frozenset[str]]
                       ) -> tuple[list[ModelEntry], list[ModelEntry]]:
    """Split `models` into ``(the engine can load them, it can't)`` by GGUF architecture.

    `architectures` is what the engine's llama.cpp release knows (see
    :func:`gettowork.runtime_install.engine_architectures`); None = no limit.
    A model whose architecture isn't known is kept: only a known name the
    engine lacks rules a model out.
    """
    if not architectures:
        return list(models), []
    kept: list[ModelEntry] = []
    left_out: list[ModelEntry] = []
    for model in models:
        arch = (model.architecture or "").strip().lower()
        (left_out if arch and arch not in architectures else kept).append(model)
    return kept, left_out


def pick_shortlist(ranked: list[FitResult], n: int = 6) -> list[FitResult]:
    """A short, diverse menu with badges, in display order.

    Speeds here are *turn* speeds (see `turn_tokens_per_s`).

    1. **recommended** - the best score among comfortable fits running at
       ≥ 8 tok/s (relaxing to ≥ 3 tok/s, then to snug fits; never below 3).
    2. **fastest** - the quickest "great"/"ok" fit that is clearly quicker
       than the recommended pick (1.25x, or 1.5x once that's already ≥ 25
       tok/s) *and* not much less capable: it keeps FASTEST_QUALITY_SHARE of
       the recommended pick's quality and has at least FASTEST_MIN_EFFECTIVE_B
       billion (dense-equivalent) parameters - so it's never a toy model, and
       fast Mixture-of-Experts models get their chance on big graphics cards.
    3. **smartest** - the most capable model (quality points) still running
       at ≥ 5 tok/s - never a snug fit in the computer's own RAM (processor,
       a Mac's unified memory or an Apple split: see `_squeezed`), nor a
       last-resort, heavily compressed quant.
    4. Then the next best scores, preferring families - and base models
       (`_lineage`: fine-tunes of one base share it) - not shown yet, as long
       as they score within DIVERSITY_MARGIN of the best. Models slower than
       3 tok/s are left for the full ranked list.

    Never two conversions of the same original model, nor two dated refreshes
    / thinking-vs-instruct twins of it. Models that *always* think first
    (see `always_thinks`) are never on this menu. A model that earns two
    badges shows both. Returns copies; `ranked` is not modified.
    """
    # Models that always think first can't be asked for a quick answer, so they
    # stay off the short menu (they're still in the full list, with a warning).
    viable = [f for f in ranked if f.verdict != "no" and not always_thinks(f.model)]
    picks: list[FitResult] = []
    badges: dict[int, list[str]] = {}

    def base(f: FitResult) -> str:
        return _variant_family(f.model)

    def try_add(fit: Optional[FitResult], badge: Optional[str] = None) -> bool:
        if fit is None:
            return False
        for i, chosen in enumerate(picks):
            if chosen is fit:
                if badge:
                    badges[i].append(badge)
                return True
            if base(chosen) == base(fit):
                return False
        if len(picks) >= n:
            return False
        picks.append(fit)
        badges[len(picks) - 1] = [badge] if badge else []
        return True

    def first_addable(candidates: list[FitResult], badge: str) -> None:
        for fit in candidates:
            if try_add(fit, badge):
                return

    recommended = _pick_recommended(viable)
    try_add(recommended, "recommended")
    fast_pool = [f for f in viable if f.verdict in ("great", "ok") and f.est_tokens_per_s]
    if recommended is not None:
        rec_tps = _turn_tps(recommended)
        speedup = 1.5 if rec_tps >= FAST_ENOUGH_TPS else FASTEST_MIN_SPEEDUP
        floor = FASTEST_QUALITY_SHARE * _quality_points(recommended.model, recommended.quant)
        quicker = [f for f in fast_pool
                   if _turn_tps(f) >= speedup * rec_tps and _quality_points(f.model, f.quant) >= floor
                   and _effective_params_b(f.model) >= FASTEST_MIN_EFFECTIVE_B]
        # No worthy quicker pick: the badge goes to the recommended one only if it
        # really is the quickest comfortable fit - otherwise nobody gets it.
        is_quickest = all(_turn_tps(f) <= rec_tps for f in fast_pool)
        fast_pool = quicker or ([recommended] if is_quickest and recommended in fast_pool else [])
    first_addable(_fastest_order(fast_pool), "fastest")
    smartest = sorted(
        (f for f in viable if _turn_tps(f) >= 5 and not _squeezed(f) and _quant_tier(f.quant or "") != "last"),
        key=lambda f: (_quality_points(f.model, f.quant), f.model.params_b),
        reverse=True,
    )
    first_addable(smartest, "smartest")

    # Fillers must be playable: painfully slow models stay in the full ranked list only.
    by_score = _originals_first(sorted((f for f in viable if _turn_tps(f) >= 3), key=lambda f: f.score, reverse=True))
    families = {p.model.family.lower() for p in picks}
    lineages = {_lineage(p.model) for p in picks}
    good_enough = (by_score[0].score - DIVERSITY_MARGIN) if by_score else 0.0
    for fit in by_score:  # first pass: new families, if they're nearly as good
        if len(picks) >= n or fit.score < good_enough:
            break
        # "New" means a new family *and* a new lineage: a fine-tune uploaded by
        # someone else (its "family" is the uploader) is still the same base model.
        family, lineage = fit.model.family.lower(), _lineage(fit.model)
        if family not in families and lineage not in lineages and try_add(fit):
            families.add(family)
            lineages.add(lineage)
    for fit in by_score:  # second pass: other sizes of families already shown (new base models first)
        if len(picks) >= n:
            break
        if _lineage(fit.model) not in lineages and try_add(fit):
            lineages.add(_lineage(fit.model))
    for fit in by_score:  # last: anything left, if there's still room
        if len(picks) >= n:
            break
        try_add(fit)

    return [replace(fit, badges=tuple(badges[i])) for i, fit in enumerate(picks)]


def recommend(specs: SystemSpecs, catalog: Optional[list[ModelEntry]] = None) -> Optional[FitResult]:  # noqa: D401
    """The single best pick for this machine (None if nothing is playable)."""
    for fit in pick_shortlist(rank_models(specs, catalog)):
        if "recommended" in fit.badges:
            return fit
    return None


# The same friendly words the model menu uses for each verdict.
_VERDICT_WORDS = {"great": "great fit", "ok": "good fit", "tight": "snug fit", "no": "won't fit"}


def explain_fit(specs: SystemSpecs, fit: FitResult) -> str:
    """Markdown showing the working behind a FitResult - for curious players."""
    model = fit.model
    quant = fit.quant or model.quant
    size = fit.download_gb or model.file_size_gb
    plan = _plan(specs, model, quant, size, context=fit.context_tokens)
    lines = [f"**{model.display_name} · {quant}** ({size:.1f} GB download, {model.license})"]
    gpu_extra = GPU_COMPUTE_BUFFER_GB if plan.placement in ("gpu", "unified", "partial") else 0.0
    context = fit.context_tokens or min(model.context_tokens, model.native_context or model.context_tokens)
    lines.append(
        f"- Memory: weights {size * GIB_PER_GB:.1f} GB + KV cache {plan.kv_gb * GIB_PER_GB:.2f} GB "
        f"({context:,} tokens) + overhead {OVERHEAD_GB + gpu_extra:.1f} GB = **{plan.need_gb:.1f} GB** "
        "(in the binary gigabytes your computer reports, so the download's 'GB' looks a little bigger)"
    )
    budget = f"{plan.budget_gb:.1f} GB"
    verdict_words = _VERDICT_WORDS.get(fit.verdict, fit.verdict)
    if plan.placement == "partial" and _is_apple(specs):
        lines.append(
            f"- Budget: {budget} of your Mac's memory ({specs.ram_total_gb:.0f} GB, keeping "
            f"{_os_headroom_gb(specs)} GB for macOS) → {plan.need_gb / max(plan.budget_gb, 0.01):.0%} used → "
            f"**{verdict_words}** (it runs past the share the GPU may use, into the rest of the same memory)"
        )
    elif plan.placement == "partial":
        vram = plan.need_gb * plan.offload_fraction
        spill = plan.need_gb - vram
        ram = _ram_budget_gb(specs)
        lines.append(
            f"- Budget: the graphics card's {vram:.1f} GB are filled, and the other {spill:.1f} GB go to system "
            f"RAM - {spill / max(ram, 0.01):.0%} of the {ram:.1f} GB we can spare → **{verdict_words}** "
            "(a split is never better than good: the card is full)"
        )
    else:
        where = {
            "gpu": f"{budget} of video memory ({_dedicated_vram_gb(specs):.0f} GB minus {GPU_VRAM_RESERVE_GB} GB kept free)",
            "unified": f"{budget} of unified memory that the Mac lets its GPU use",
            "cpu": f"{budget} of RAM ({specs.ram_total_gb:.0f} GB minus {_os_headroom_gb(specs)} GB for your system)",
            "none": f"{budget} at most",
        }[plan.placement]
        lines.append(f"- Budget: {where} → {plan.ratio:.0%} used → **{verdict_words}**")
    if plan.tokens_per_s:
        moe = " (only the active experts are read)" if _active_share(model) < 1 else ""
        if plan.placement == "partial":
            gpu_bw, _ = perf.bandwidth_for(specs, "gpu")
            cpu_bw, _ = perf.bandwidth_for(specs, "cpu")
            lines.append(
                f"- Speed: {plan.offload_fraction:.0%} on the GPU (~{gpu_bw:.0f} GB/s) and the rest on the CPU "
                f"(~{cpu_bw:.0f} GB/s), reading {plan.active_gb:.1f} GB per token{moe} ≈ "
                f"**{plan.tokens_per_s:.0f} tokens/s** ({fit.est_speed})"
            )
        else:
            bw, source = perf.bandwidth_for(specs, plan.placement)
            eff = perf.efficiency_for(specs, plan.placement)
            lines.append(
                f"- Speed: {eff:.2f} × {bw:.0f} GB/s ({source}) ÷ {plan.active_gb:.1f} GB read per token{moe}, "
                f"plus a tiny fixed cost per token ≈ **{plan.tokens_per_s:.0f} tokens/s** ({fit.est_speed})"
            )
        turn = TURN_ANSWER_TOKENS / max(model_turn_tokens_per_s(model, plan.tokens_per_s, plan.placement), 0.01)
        mode = thinking_mode(model)
        if mode == "always":
            note = (f" - including ~{ALWAYS_THINKING_TOKENS:,} tokens of thinking it always does first (it can't be "
                    "asked to skip it), which is why it isn't on the short menu")
        elif mode == "switchable" and plan.tokens_per_s >= THINKING_MIN_TOKENS_PER_S:
            note = "; it thinks out loud too, since it's quick enough"
        else:
            note = ""
        lines.append(
            f"- A story turn (reading ~{TURN_PROMPT_TOKENS:,} tokens of prompt, writing ~{TURN_ANSWER_TOKENS}) "
            f"takes roughly **{turn:.0f} seconds**{note}."
        )
    tier = "only" if len(_quant_options(model)) == 1 else _quant_tier(quant)
    if fit.verdict == "no":
        tier = "disk" if not plan.disk_ok else "none"
    why = {
        "high": "there's room to spare and it stays fast, so we picked a sharper, higher-precision version",
        "standard": "the ~4-bit sweet spot: small and fast with little quality loss",
        "low": "a more compressed version so it fits; a bit less sharp",
        "last": "a heavily compressed version, the only one that fits",
        "full": "a full-precision file (usually overkill, but it's what fits best here)",
        "only": "the only version on offer",
        "none": "even the smallest version on offer is too big for this computer",
        "disk": "even the smallest version needs more free disk space than you have",
    }[tier]
    lines.append(f"- Why {quant}: {why}.")
    lines.append("- *A home-grown estimate (MIT licensed), not a guarantee.*")
    return "\n".join(lines)


MEMORY_FORMULA_EXPLAINER = """\
**How we guess whether a model fits**

A model needs memory for three things:

1. **Weights** - the model file itself. *Quantization* shrinks it by storing
   each weight in fewer bits: Q8_0 uses ~8.5 bits, Q4_K_M ~4.8. Roughly,
   size ≈ parameters × bits ÷ 8, so an 8B model at Q4_K_M is about 5 GB.
2. **KV cache** - the model's short-term memory of the conversation. It grows
   with every token of context (about 0.6 GB for an 8B model at 4,096 tokens).
3. **Overhead** - about 0.6 GB for the engine, plus 0.3 GB scratch space on a GPU.

Then we pick where it runs: entirely on your graphics card if it fits (keeping
0.8 GB free), in a Mac's unified memory, split between GPU and RAM, or on the
CPU (leaving 2.5 GB for your system, 3.5 on Windows). Up to 60% of that
budget is *great*, 85% *ok*, 100% *tight*; if nothing fits, we try a
shorter conversation memory.

We choose the best version that fits comfortably and stays quick. A
home-grown estimate: handy, but no warranty!
"""


# ---------------------------------------------------------------------------
# The curated seed list
# ---------------------------------------------------------------------------

_STANDARD_QUANTS = ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "IQ4_XS", "Q3_K_M")


def _seed(
    key: str,
    display_name: str,
    family: str,
    params_b: float,
    repo: str,
    original: str,
    license_id: str,
    blurb: str,
    sizes: tuple[float, ...],
    *,
    active_params_b: Optional[float] = None,
    reasoning: bool = False,
    quants: tuple[str, ...] = _STANDARD_QUANTS,
    default_quant: str = "Q4_K_M",
    gguf_file: Optional[str] = None,
    ollama_ref: Optional[str] = None,
    architecture: Optional[str] = None,
    native_context: Optional[int] = None,
) -> ModelEntry:
    """Build one curated ModelEntry. `sizes` line up with `quants` (GB, approximate)."""
    options = tuple(zip(quants, sizes))
    default_size = dict(options)[default_quant]
    file_stem = repo.split("/")[1].removesuffix("-GGUF")
    return ModelEntry(
        key=key,
        display_name=display_name,
        family=family,
        params_b=params_b,
        active_params_b=active_params_b,
        license=license_id,
        license_url=f"https://huggingface.co/{original}",
        hf_repo=repo,
        quant=default_quant,
        file_size_gb=default_size,
        ollama_ref=ollama_ref or f"hf.co/{repo}:{default_quant}",
        reasoning=reasoning,
        thinking="switchable" if reasoning else "none",  # every curated thinker can be asked to skip it
        blurb=blurb,
        source="curated",
        quant_options=options,
        gguf_files=(gguf_file or f"{file_stem}-{default_quant}.gguf",),
        base_model=original,
        architecture=architecture,
        native_context=native_context,
    )


# Ordered small -> large. Only Apache-2.0 / MIT models. Sizes are approximate
# (Hugging Face shows the exact numbers); params_b is the *total* count.
MODEL_CATALOG: list[ModelEntry] = [
    _seed("qwen3-0.6b", "Qwen3 0.6B", "Qwen3", 0.6, "unsloth/Qwen3-0.6B-GGUF", "Qwen/Qwen3-0.6B", "Apache-2.0",
          "A pocket-sized thinker: tiny, speedy, and surprisingly chatty for its size.",
          (0.64, 0.50, 0.44, 0.40, 0.36, 0.35), reasoning=True, architecture="qwen3", native_context=32768),
    _seed("qwen3-1.7b", "Qwen3 1.7B", "Qwen3", 1.72, "unsloth/Qwen3-1.7B-GGUF", "Qwen/Qwen3-1.7B", "Apache-2.0",
          "Small but clever, and it shows its thinking - a great first model.",
          (1.83, 1.42, 1.26, 1.11, 1.01, 0.94), reasoning=True, architecture="qwen3", native_context=32768),
    _seed("smollm2-1.7b", "SmolLM2 1.7B Instruct", "SmolLM2", 1.71, "bartowski/SmolLM2-1.7B-Instruct-GGUF",
          "HuggingFaceTB/SmolLM2-1.7B-Instruct", "Apache-2.0",
          "Hugging Face's little all-rounder: light, quick and friendly.",
          (1.82, 1.41, 1.23, 1.06, 0.94, 0.86), architecture="llama", native_context=8192),
    _seed("phi-4-mini", "Phi-4 mini", "Phi-4", 3.84, "unsloth/Phi-4-mini-instruct-GGUF", "microsoft/Phi-4-mini-instruct",
          "MIT", "Microsoft's compact model with quick wits and a huge vocabulary.",
          (4.08, 3.16, 2.85, 2.49, 2.21, 2.02), architecture="phi3", native_context=131072),
    _seed("qwen3-4b", "Qwen3 4B", "Qwen3", 4.02, "unsloth/Qwen3-4B-GGUF", "Qwen/Qwen3-4B", "Apache-2.0",
          "The sweet spot for many laptops: quick, capable, and shows its reasoning.",
          (4.28, 3.31, 2.89, 2.50, 2.29, 2.08), reasoning=True, architecture="qwen3", native_context=32768),
    _seed("mistral-7b", "Mistral 7B Instruct v0.3", "Mistral", 7.25, "bartowski/Mistral-7B-Instruct-v0.3-GGUF",
          "mistralai/Mistral-7B-Instruct-v0.3", "Apache-2.0", "A classic, dependable storyteller from Mistral AI.",
          (7.70, 5.95, 5.14, 4.37, 3.91, 3.52), architecture="llama", native_context=32768),
    _seed("qwen3-8b", "Qwen3 8B", "Qwen3", 8.19, "unsloth/Qwen3-8B-GGUF", "Qwen/Qwen3-8B", "Apache-2.0",
          "A strong all-rounder with visible reasoning; lovely on a decent GPU or a Mac.",
          (8.71, 6.73, 5.85, 5.03, 4.59, 4.12), reasoning=True, architecture="qwen3", native_context=32768),
    _seed("qwen3-14b", "Qwen3 14B", "Qwen3", 14.8, "unsloth/Qwen3-14B-GGUF", "Qwen/Qwen3-14B", "Apache-2.0",
          "Noticeably smarter and funnier - happiest on a 12 GB+ graphics card.",
          (15.7, 12.1, 10.5, 9.00, 8.18, 7.32), reasoning=True, architecture="qwen3", native_context=32768),
    _seed("gpt-oss-20b", "gpt-oss 20B", "gpt-oss", 20.9, "ggml-org/gpt-oss-20b-GGUF", "openai/gpt-oss-20b", "Apache-2.0",
          "OpenAI's open-weight reasoning model; quick for its size thanks to Mixture-of-Experts.",
          (12.1,), active_params_b=3.6, reasoning=True, quants=("MXFP4",), default_quant="MXFP4",
          gguf_file="gpt-oss-20b-mxfp4.gguf", ollama_ref="gpt-oss:20b", architecture="gpt-oss",
          native_context=131072),
    _seed("mistral-small-3.2-24b", "Mistral Small 3.2 24B", "Mistral", 23.6,
          "bartowski/mistralai_Mistral-Small-3.2-24B-Instruct-2506-GGUF", "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
          "Apache-2.0", "A polished, imaginative writer for machines with plenty of memory.",
          (25.1, 19.3, 16.8, 14.3, 12.8, 11.5), architecture="llama", native_context=131072),
    _seed("qwen3-30b-a3b", "Qwen3 30B-A3B (MoE)", "Qwen3", 30.5, "unsloth/Qwen3-30B-A3B-GGUF", "Qwen/Qwen3-30B-A3B",
          "Apache-2.0", "A big Mixture-of-Experts brain that only wakes ~3B parameters per word, so it's quick.",
          (32.5, 25.1, 21.7, 18.6, 16.4, 14.7), active_params_b=3.3, reasoning=True, architecture="qwen3moe",
          native_context=32768),
    _seed("qwen3-32b", "Qwen3 32B", "Qwen3", 32.8, "unsloth/Qwen3-32B-GGUF", "Qwen/Qwen3-32B", "Apache-2.0",
          "The heavyweight: rich, witty storytelling if you have the memory for it.",
          (34.8, 26.9, 23.2, 19.8, 17.9, 16.0), reasoning=True, architecture="qwen3", native_context=32768),
]

_CURATED_REPOS.update(m.hf_repo.lower() for m in MODEL_CATALOG)


def get_model(key: str) -> Optional[ModelEntry]:
    """Find a curated model by its key ("qwen3-4b") or Hugging Face repo id (any case)."""
    wanted = (key or "").strip().lower()
    for model in MODEL_CATALOG:
        if wanted in (model.key.lower(), model.hf_repo.lower()):
            return model
    return None
