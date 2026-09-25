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
    ram_bandwidth_gbs: Optional[float] = None  # measured by a quick copy benchmark (perf.py)
    cpu_flags: list[str] = field(default_factory=list)  # e.g. ["avx2", "avx512f", "neon"]

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
