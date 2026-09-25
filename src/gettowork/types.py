"""Shared data types used across the game.

Every module talks to every other module through these small dataclasses, so
reading this file first is the fastest way to understand the codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional

# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------

GPUVendor = Literal["nvidia", "amd", "apple", "intel", "unknown"]


@dataclass
class GPUInfo:
    """One detected graphics processor."""

    name: str
    vendor: GPUVendor
    vram_gb: float  # dedicated VRAM; for Apple unified memory this is the usable share of RAM
    bandwidth_gbs: Optional[float] = None  # estimated memory bandwidth (GB/s), if known
    driver_version: Optional[str] = None  # e.g. "550.54.14" (NVIDIA, from nvidia-smi), if known
    # NVIDIA "compute capability" (the chip generation), e.g. 6.1 for a GTX 1080,
    # 8.6 for an RTX 3060 - newer CUDA builds leave out older generations.
    compute_capability: Optional[float] = None


@dataclass
class SystemSpecs:
    """A snapshot of the machine the game is running on."""

    os_name: str  # "Windows", "Darwin", "Linux", ...
    os_version: str
    arch: str  # "x86_64", "arm64", ...
    cpu_name: str
    cpu_cores_physical: Optional[int]
    cpu_cores_logical: Optional[int]
    ram_total_gb: float
    ram_available_gb: float
    disk_free_gb: float  # free space where models are stored
    gpus: list[GPUInfo] = field(default_factory=list)
    unified_memory: bool = False  # True on Apple Silicon: GPU shares system RAM
    notes: list[str] = field(default_factory=list)  # human-readable caveats from detection
    ram_bandwidth_gbs: Optional[float] = None  # measured by a quick read benchmark (perf.py)
    cpu_flags: list[str] = field(default_factory=list)  # e.g. ["avx2", "avx512f", "neon"]
    # False when the engine that will run the model can't use any graphics card
    # here (e.g. an AMD card on Linux without the Vulkan loader): the fit engine
    # then plans with the CPU. None = no limit known.
    gpu_offload: Optional[bool] = None

    @property
    def best_vram_gb(self) -> float:
        return max((g.vram_gb for g in self.gpus), default=0.0)


# ---------------------------------------------------------------------------
# Model catalog / fit
# ---------------------------------------------------------------------------

FitVerdict = Literal["great", "ok", "tight", "no"]
FitPlacement = Literal["gpu", "unified", "partial", "cpu", "none"]


@dataclass(frozen=True)
class ModelEntry:
    """A curated open-weight model the game knows how to download and run."""

    key: str  # short unique id, e.g. "qwen3-4b"
    display_name: str  # "Qwen3 4B"
    family: str  # "Qwen3"
    params_b: float  # total parameters, billions
    active_params_b: Optional[float]  # for Mixture-of-Experts models, else None
    license: str  # SPDX-ish id, e.g. "Apache-2.0"
    license_url: str
    hf_repo: str  # Hugging Face repo that hosts GGUF files
    quant: str  # quantization tag to look for in file names, e.g. "Q4_K_M"
    file_size_gb: float  # approximate download size of that quant
    ollama_ref: str  # what to `ollama pull`, e.g. "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"
    reasoning: bool  # emits visible chain-of-thought (e.g. <think> blocks)
    blurb: str  # one friendly sentence for the selection table
    context_tokens: int = 4096  # context window the game will request
    # --- fields filled in by live Hugging Face discovery (hf_discovery.py) ---
    source: str = "curated"  # "curated" (built-in seed list) or "huggingface" (live search)
    quant_options: tuple[tuple[str, float], ...] = ()  # (quant tag, total size GB) available in the repo
    gguf_files: tuple[str, ...] = ()  # exact file name(s) for `quant` (several if split into shards)
    downloads: int = 0  # Hugging Face download count (popularity signal)
    likes: int = 0
    base_model: Optional[str] = None  # the original (non-GGUF) model this was converted from
    architecture: Optional[str] = None  # e.g. "qwen3", "llama", "phi3" (from GGUF metadata)
    native_context: Optional[int] = None  # model's trained context length, if known
    gated: bool = False  # requires accepting terms / logging in on Hugging Face
    # How the model thinks out loud: "none" (never), "switchable" (thinks, but can
    # be asked to answer straight away - Qwen3, gpt-oss, SmolLM3...) or "always"
    # (its chat template forces thinking - QwQ, DeepSeek-R1 distills, Phi-4-reasoning...).
    # "" = not recorded (older saved lists): read as "switchable" if `reasoning`, else "none".
    # Use `catalog.thinking_mode(entry)` rather than reading it directly.
    thinking: str = ""
    # (layers, KV heads, head size) read from the GGUF header, for exact KV-cache
    # maths when `catalog` doesn't know the family; () = not read.
    kv_shape: tuple[int, ...] = ()


# How each saved ModelEntry field is read back from JSON (settings file, model
# list cache). Every field must be listed: tests check this table stays complete.
_ENTRY_TEXT = ("key", "display_name", "family", "license", "license_url", "hf_repo", "quant", "ollama_ref",
               "blurb", "source", "thinking")
_ENTRY_OPTIONAL_TEXT = ("base_model", "architecture")
_ENTRY_NUMBER = ("params_b", "file_size_gb")
_ENTRY_OPTIONAL_NUMBER = ("active_params_b",)
_ENTRY_COUNT = ("context_tokens", "downloads", "likes")
_ENTRY_OPTIONAL_COUNT = ("native_context",)
_ENTRY_FLAG = ("reasoning", "gated")
_ENTRY_LISTS = ("quant_options", "gguf_files", "kv_shape")


class _BadField(ValueError):
    pass


def _as_number(value: Any) -> float:
    if isinstance(value, bool):
        raise _BadField(value)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        raise _BadField(value) from None
    if number != number or number in (float("inf"), float("-inf")):
        raise _BadField(value)
    return number


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    raise _BadField(value)


def _as_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "yes", "no", "1", "0"):
        return value.strip().lower() in ("true", "yes", "1")
    raise _BadField(value)


def model_entry_from_json(data: Any) -> Optional[ModelEntry]:
    """Rebuild a ModelEntry from JSON (the settings file or the model-list cache).

    Both can be hand-edited or damaged, so every field is checked against its
    type (numbers written as text are accepted and converted; a missing name
    becomes ""), and an entry with a field that can't be read - or a required
    field missing - comes back as None instead of crashing the game later.
    """
    if not isinstance(data, dict):
        return None
    import dataclasses as _dc

    fields = {f.name: f for f in _dc.fields(ModelEntry)}
    values: dict[str, Any] = {}
    try:
        for name, raw in data.items():
            if name not in fields:
                continue
            if name in _ENTRY_TEXT:
                values[name] = _as_text(raw)
            elif name in _ENTRY_OPTIONAL_TEXT:
                values[name] = None if raw is None or raw == "" else _as_text(raw)
            elif name in _ENTRY_NUMBER:
                values[name] = _as_number(raw)
            elif name in _ENTRY_OPTIONAL_NUMBER:
                number = None if raw is None else _as_number(raw)
                values[name] = number if number is not None and number > 0 else None
            elif name in _ENTRY_COUNT:
                values[name] = max(0, int(_as_number(raw or 0)))
            elif name in _ENTRY_OPTIONAL_COUNT:
                count = None if raw is None else int(_as_number(raw))
                values[name] = count if count is not None and count > 0 else None
            elif name in _ENTRY_FLAG:
                values[name] = _as_flag(raw)
            elif name == "quant_options":
                values[name] = tuple((_as_text(q), _as_number(size)) for q, size in (raw or ()))
            elif name == "kv_shape":
                shape = tuple(int(_as_number(x)) for x in (raw or ()))
                values[name] = shape if len(shape) == 3 and all(x > 0 for x in shape) else ()
            elif name == "gguf_files":
                if isinstance(raw, str):
                    raise _BadField(raw)
                values[name] = tuple(_as_text(f) for f in (raw or ()))
        if values.get("context_tokens", 1) <= 0:
            values["context_tokens"] = 4096
        return ModelEntry(**values)
    except (_BadField, TypeError, ValueError, OverflowError):
        return None


@dataclass
class FitResult:
    """How well a ModelEntry fits a SystemSpecs."""

    model: ModelEntry
    verdict: FitVerdict
    placement: FitPlacement
    est_memory_gb: float  # estimated memory to run (weights + KV cache + overhead)
    est_speed: str  # rough, e.g. "fast", "usable", "slow", "very slow"
    reason: str  # one human-readable sentence explaining the verdict
    quant: Optional[str] = None  # the quantization chosen for this machine (may differ from model.quant)
    download_gb: Optional[float] = None  # size of the chosen quant's file(s)
    est_tokens_per_s: Optional[float] = None  # rough generation speed estimate
    score: float = 0.0  # overall ranking score (higher = better pick)
    badges: tuple[str, ...] = ()  # e.g. ("recommended",), ("fastest",), ("smartest",)
    context_tokens: Optional[int] = None  # a shorter context the fit engine chose so it fits (None = the model's own)
    gpu_share: Optional[float] = None  # share of the model on the graphics card (1.0 = all, 0 = none)
    # True when the memory this plan fills is the computer's own RAM - the
    # processor, a Mac's unified memory, or an Apple split - so a snug fit means
    # the whole computer is short of memory (and overflow means swapping).
    shares_system_ram: bool = False


# ---------------------------------------------------------------------------
# Local LLM
# ---------------------------------------------------------------------------


@dataclass
class LLMResult:
    """One completed local-LLM call, kept for the end-of-game transcript."""

    text: str  # the final answer with any reasoning stripped out
    reasoning: Optional[str]  # exposed chain-of-thought / thinking, if the model produced any
    model: str
    backend: str  # "ollama", "llamacpp", "mock"
    elapsed_s: float
    messages: list[dict[str, str]] = field(default_factory=list)  # the prompt that was sent
    raw: Optional[dict[str, Any]] = None  # backend-specific raw response (JSON-safe)
    truncated: bool = False  # the answer was cut off by the token limit (finish_reason "length")


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------


@dataclass
class JevExchange:
    """A raw HTTP round-trip to the Jev API, with secrets redacted."""

    url: str
    request_headers: dict[str, str]  # Authorization is redacted
    request_body: dict[str, Any]
    status: Optional[int]
    response_body: Optional[Any]
    error: Optional[str]
    elapsed_s: float


@dataclass
class JevVerdict:
    """The game's interpretation of one Jev judgment of a player's plan."""

    made_progress: bool
    progress_probability: float  # the Noul answer: P(yes, the plan made progress)
    outcome: str  # the Choice answer label
    outcome_confidence: float
    outcome_probabilities: dict[str, float]
    creativity: float  # the Score answer (expected level, 0..4)
    creativity_confidence: float
    creativity_legend: dict[str, Any]
    exchange: JevExchange


# ---------------------------------------------------------------------------
# Game
# ---------------------------------------------------------------------------

JudgeKind = Literal["jev", "local"]


@dataclass
class RoundRecord:
    """Everything that happened in one round, for the end-of-game review."""

    number: int  # 1-based
    challenge: str
    player_plan: str
    judge: JudgeKind
    made_progress: bool
    judge_explanation: str  # short text shown to the player
    progress_after: int
    jev: Optional[JevVerdict] = None
    llm_calls: list[tuple[str, LLMResult]] = field(default_factory=list)  # (purpose, result)
    failed_jev_exchange: Optional[JevExchange] = None  # a Jev call that failed this round (the local model judged instead)


@dataclass
class GameSummary:
    won: bool
    quit_early: bool
    progress: int
    target: int
    intro: str
    ending: str
    rounds: list[RoundRecord] = field(default_factory=list)
    intro_calls: list[tuple[str, LLMResult]] = field(default_factory=list)
    ending_calls: list[tuple[str, LLMResult]] = field(default_factory=list)
