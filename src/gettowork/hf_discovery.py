"""Live model discovery: ask Hugging Face which GGUF chat models exist *today*.

Instead of shipping a frozen list of models, the game searches the Hugging
Face Hub (https://huggingface.co) every few days:

1. **Search.** One `list_models` call per trusted publisher (unsloth,
   bartowski, ggml-org, ...) plus one global search, each asking for the most
   downloaded GGUF text-generation repos together with their metadata
   (GGUF header info, license, tags, downloads, gated flag).
2. **Screen.** Drop anything that isn't a family-friendly, instruction-tuned
   chat model with a permissive license (`rejection_reason` explains each
   decision), then keep one repo per original model (`base_model`).
3. **Measure.** For the most popular survivors, list the repo's files to get
   the *real* size of every quantization (`group_quant_files`), including
   models split into several shards.
4. **Cache.** Save the result as JSON so later launches are instant and work
   offline. If the Hub can't be reached, fall back to the saved list (even an
   old one) or, failing that, the curated seeds in `catalog.py`.

The filters are home-grown heuristics (MIT licensed, no warranty): always read
a model's card on Hugging Face before relying on it.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import os
import queue
import re
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from . import catalog, config
from .types import ModelEntry, model_entry_from_json

__all__ = [
    "DiscoveryResult",
    "discover_models",
    "entry_from_hub",
    "parse_quant",
    "group_quant_files",
    "shard_info",
    "rejection_reason",
    "not_family_friendly",
    "license_of",
    "params_from_name",
    "prettify_repo_name",
    "repo_gguf_files",
    "default_cache_path",
    "make_hub_api",
    "configure_hub_timeouts",
    "with_deadline",
    "read_gguf_header",
    "active_params_from_header",
    "FALLBACK_QUANT_ORDER",
    "CACHE_SCHEMA_VERSION",
    "RULES_VERSION",
    "rescreen_entry",
    "thinking_mode_for",
    "DISCOVERY_EXPLAINER",
]

# Ends the "left out N models" note; the game's window drops it (no command line there).
ALL_LICENSES_HINT = " (--all-licenses shows them)"

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

CACHE_FILENAME = "hf_models.json"
CACHE_SCHEMA_VERSION = 1  # bump when the cache layout changes; old caches are then ignored

PER_PUBLISHER_LIMIT = 30  # most-downloaded GGUF repos we look at per trusted publisher
GLOBAL_LIMIT = 60  # ...and in the one search across everybody
MAX_WORKERS = 6  # parallel requests to the Hub (be polite!)
MIN_PARAMS_B = 0.4  # smaller models can't keep a story straight
MAX_PARAMS_B = 130.0  # bigger ones won't fit any home computer
MIN_DOWNLOADS_UNTRUSTED = 1000  # a little popularity proof for lesser-known publishers
PARTIAL_CACHE_TTL_HOURS = 1.0  # an incomplete search is only trusted this long (then we ask again)

# Every request to the Hub gets a time limit: connecting, and each wait for
# data. (huggingface_hub itself waits forever by default, so one stalled
# connection - flaky Wi-Fi, a captive portal - could freeze the game.)
HUB_CONNECT_TIMEOUT_S = 10.0
HUB_READ_TIMEOUT_S = 30.0
GGUF_HEADER_BYTES = 512 * 1024  # enough of a GGUF file to read its settings (they come before the vocabulary)

# The metadata we ask the Hub to include with each search result. With `expand`,
# the Hub returns *only* these fields (plus the repo id). huggingface_hub refuses
# `expand` combined with `full=True` / `cardData=True`, so we only use expand.
EXPAND_FIELDS = ("gguf", "cardData", "tags", "downloads", "likes", "lastModified", "gated", "pipeline_tag")

# The quant we suggest by default when a repo has several (also download.py's fallback order).
FALLBACK_QUANT_ORDER = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q4_0", "IQ4_XS", "Q6_K", "Q8_0", "MXFP4")

# Mixture-of-Experts models whose names don't say how many parameters are
# active: name pattern -> (total, active) in billions, from their model cards.
# (Names like "30B-A3B" are read directly; the GGUF header is read when
# neither works - see `active_params_from_header`.)
_KNOWN_MOE: tuple[tuple[str, float, float], ...] = (
    (r"gpt-oss-120b", 116.8, 5.1),
    (r"gpt-oss-20b", 20.9, 3.6),
    (r"mixtral-8x22b", 141.0, 39.0),
    (r"mixtral-8x7b", 46.7, 12.9),
    (r"phi-3\.5-moe", 41.9, 6.6),
    (r"granite-4\.0-h-small", 32.2, 9.0),
    (r"granite-4\.0-h-tiny", 6.9, 1.0),
    (r"granite-3\.\d-3b-a800m", 3.3, 0.8),
    (r"granite-3\.\d-1b-a400m", 1.3, 0.4),
    (r"olmoe-1b-7b", 6.9, 1.3),
    (r"glm-4\.5-air", 106.0, 12.0),
    (r"glm-4\.[56](?!-air|v)", 355.0, 32.0),
    (r"ling-lite", 16.8, 2.75),
    (r"deepseek-v2-lite", 15.7, 2.4),
    (r"qwen1\.5-moe-a2\.7b", 14.3, 2.7),
)
# Architectures (GGUF "general.architecture") that are Mixture-of-Experts designs.
MOE_ARCHITECTURES = frozenset({
    "qwen2moe", "qwen3moe", "phimoe", "glm4moe", "olmoe", "granitemoe", "deepseek2",
    "bailingmoe", "bailingmoe2", "hunyuan-moe", "ernie4_5-moe", "dots1", "gpt-oss", "llama4", "arctic", "dbrx",
    "grok", "qwen3next", "smallthinker", "exaone-moe", "minimax-m2", "seed_oss_moe",
})
# Architectures that are *sometimes* Mixture-of-Experts: IBM's Granite-4.0-H
# "tiny"/"small" are, but "micro"/"1b" (and Bamba) are dense hybrids with the
# same architecture name, and so is a dense Jamba. Only the GGUF header's
# expert_count (> 1) says which - never the architecture name alone.
MAYBE_MOE_ARCHITECTURES = frozenset({"granitehybrid", "jamba"})
# When a Mixture-of-Experts model's active share can't be found at all, assume
# about a quarter is used per word (real ones range from ~10% to ~30%): far
# closer than treating it as a dense model.
MOE_UNKNOWN_ACTIVE_SHARE = 0.25

# Organisations whose name bartowski-style repos put in front ("Qwen_Qwen3-4B-GGUF").
_KNOWN_ORGS = frozenset(p.lower() for p in catalog.TRUSTED_PUBLISHERS) | {
    "google", "meta-llama", "openai", "deepseek-ai", "nvidia", "allenai", "thudm", "zai-org",
    "liquidai", "tiiuae", "cohereforai", "coherelabs", "01-ai", "internlm", "baidu", "tencent",
    "moonshotai", "lgai-exaone", "arcee-ai", "swiss-ai", "upstage", "stabilityai", "apple",
}

# ---------------------------------------------------------------------------
# The result
# ---------------------------------------------------------------------------


@dataclass
class DiscoveryResult:
    """What discovery found, and where it came from."""

    models: list[ModelEntry]
    source: str  # "live" (just asked the Hub) | "cache" (saved list) | "curated" (built-in seeds)
    fetched_at: Optional[float] = None  # when the Hub was asked (seconds since the epoch)
    notes: list[str] = field(default_factory=list)  # friendly remarks to show the player
    stale: bool = False  # True if we fell back to a cache older than the refresh interval


# ---------------------------------------------------------------------------
# Quantization tags in file names
# ---------------------------------------------------------------------------

# One llama.cpp quant tag, anywhere in a file name: Q4_K_M, IQ3_XXS, Q8_0, MXFP4,
# BF16... optionally with Unsloth's "UD-" (dynamic) prefix. The look-arounds stop
# us matching inside longer words, or the old ARM repacks like "Q4_0_4_4".
_QUANT_RE = re.compile(
    r"(?<![A-Z0-9])(UD-)?"
    r"(IQ[1-4]_(?:XXS|XS|NL|S|M)|Q[2-8]_K(?:_(?:XL|L|M|S))?|Q[4-8]_[01]|TQ[12]_0|MXFP4(?:_MOE)?|BF16|FP16|F16|FP32|F32"
    r"|Q[2-8](?=$|[-.]))"  # a bare "q4" / "q8", as in Microsoft's "Phi-3-mini-4k-instruct-q4.gguf"
    r"(?![A-Z0-9]|_\d)"
)
_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})$", re.IGNORECASE)  # "model-Q8_0-00001-of-00002"


def parse_quant(filename: str) -> Optional[str]:
    """Find the quantization tag in a GGUF file name, or None.

    "Qwen3-4B-Q4_K_M.gguf" -> "Q4_K_M"; "Qwen3-4B-UD-Q4_K_XL.gguf" -> "UD-Q4_K_XL"
    (Unsloth's dynamic quants keep their own tag); "gpt-oss-20b-mxfp4.gguf" -> "MXFP4";
    "model-Q8_0-00001-of-00002.gguf" -> "Q8_0"; "Q4_K_M/model-00001-of-00003.gguf"
    -> "Q4_K_M" (tag in the folder name). Vision projectors ("mmproj-*") and
    non-GGUF files -> None.
    """
    parts = filename.replace("\\", "/").split("/")
    base = parts[-1]
    if not base.lower().endswith(".gguf") or "mmproj" in base.lower():
        return None
    stem = _SHARD_RE.sub("", base[: -len(".gguf")])
    # The tag is normally in the file name; big split models sometimes keep it
    # only in their folder name. Look at the file first, then folders (nearest first).
    for text in [stem, *reversed(parts[:-1])]:
        matches = list(_QUANT_RE.finditer(text.upper()))
        if matches:
            last = matches[-1]  # the tag normally sits at the end of the name
            tag = {"FP16": "F16", "FP32": "F32", "MXFP4_MOE": "MXFP4"}.get(last.group(2), last.group(2))
            return f"UD-{tag}" if last.group(1) else tag
    return None


def shard_info(path: str) -> tuple[str, int, int]:
    """Split-model bookkeeping: "Q8_0/m-Q8_0-00002-of-00003.gguf" -> ("Q8_0/m-Q8_0", 2, 3);
    a single-file model -> (its path without ".gguf", 1, 1)."""
    stem = path[: -len(".gguf")] if path.lower().endswith(".gguf") else path
    match = _SHARD_RE.search(stem)
    if not match:
        return stem, 1, 1
    return stem[: match.start()], int(match.group(1)), int(match.group(2))


def group_quant_files(files: list[tuple[str, int]]) -> dict[str, tuple[tuple[str, ...], int]]:
    """Group a repo's files by quant: {quant: (file names incl. all shards, total bytes)}.

    Split models ("-00001-of-00003") are only kept when every part is present.
    If one quant appears twice (say at the repo root and in a sub-folder), the
    root copy wins. Sorted biggest (highest quality) first.
    """
    sets: dict[tuple[str, str], dict[str, Any]] = {}
    for path, size in files:
        quant = parse_quant(path)
        if quant is None:
            continue
        group, part, of = shard_info(path)
        entry = sets.setdefault((quant, group), {"parts": {}, "of": of})
        entry["parts"][part] = (path, int(size or 0))

    best: dict[str, tuple[tuple[int, int, str], tuple[str, ...], int]] = {}
    for (quant, group), entry in sets.items():
        parts = entry["parts"]
        if sorted(parts) != list(range(1, entry["of"] + 1)):
            continue  # a shard is missing: we couldn't load this set anyway
        names = tuple(parts[i][0] for i in sorted(parts))
        total = sum(size for _, size in parts.values())
        preference = (group.count("/"), len(group), group)  # root level first, then shorter paths
        if quant not in best or preference < best[quant][0]:
            best[quant] = (preference, names, total)
    ordered = sorted(best.items(), key=lambda item: -item[1][2])
    return {quant: (names, total) for quant, (_, names, total) in ordered}


# ---------------------------------------------------------------------------
# Reading Hub metadata (works with real ModelInfo objects, dicts or test fakes)
# ---------------------------------------------------------------------------


def _attr(obj: Any, *names: str, default: Any = None) -> Any:
    """First non-None attribute (or dict key) among `names`."""
    for name in names:
        value = obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)
        if value is not None:
            return value
    return default


def _card_value(info: Any, key: str) -> Any:
    card = _attr(info, "card_data", "cardData")
    if card is None:
        return None
    if isinstance(card, dict):
        return card.get(key)
    return getattr(card, key, None)


def _tags(info: Any) -> list[str]:
    return [str(t) for t in (_attr(info, "tags", default=[]) or [])]


def _gguf(info: Any) -> dict:
    value = _attr(info, "gguf")
    return value if isinstance(value, dict) else {}


def _repo_name(repo_id: str) -> str:
    """"bartowski/Qwen_Qwen3-4B-GGUF" -> "Qwen3-4B" (no publisher, no -GGUF)."""
    name = repo_id.rstrip("/").split("/")[-1]
    name = re.sub(r"[-_.]gguf$", "", name, flags=re.IGNORECASE)
    if "_" in name:
        prefix, rest = name.split("_", 1)
        if prefix.lower() in _KNOWN_ORGS or (rest.lower().startswith(prefix.lower()) and not re.search(r"\d", prefix)):
            name = rest
    return name


def _base_model(info: Any) -> Optional[str]:
    """The original model a GGUF repo was converted from, if the card or tags say."""
    tags = _tags(info)
    for tag in tags:
        if tag.startswith("base_model:quantized:"):
            return tag.split(":", 2)[2]
    value = _card_value(info, "base_model")
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if isinstance(value, str) and value.strip():
        return value.strip()
    for tag in tags:
        if tag.startswith("base_model:") and tag.count(":") == 1:
            return tag.split(":", 1)[1]
    return None


def _base_key(repo_id: str, base_model: Optional[str] = None) -> str:
    """Identical for every GGUF conversion of one original model ("qwen3-4b")."""
    return re.sub(r"[\s_]+", "-", _repo_name(base_model or repo_id).lower())


def license_of(info: Any) -> Optional[str]:
    """The model's license as shown to players ("Apache-2.0", "MIT", "llama3.2", ...), or None.

    Read from the Hub's `license:` tag, else the model card. A card that says
    "other" gets its `license_name` appended, e.g. "other (qwen-research)".
    """
    raw = next((t.split(":", 1)[1] for t in _tags(info) if t.startswith("license:")), None)
    if raw is None:
        raw = _card_value(info, "license")
        if isinstance(raw, (list, tuple)):
            raw = raw[0] if raw else None
    if not isinstance(raw, str) or not raw.strip():
        return None
    raw = raw.strip()
    if catalog.is_permissive(raw):
        return "MIT" if raw.lower() == "mit" else "Apache-2.0"
    if raw.lower() == "other":
        name = _card_value(info, "license_name")
        return f"other ({name})" if isinstance(name, str) and name.strip() else "other"
    return raw


# Families whose weights come with their own license terms (Llama Community
# License, Gemma Terms of Use). A GGUF re-upload - or a fine-tune - tagged
# "apache-2.0" or "mit" doesn't change the original model's license, so we
# don't take such a tag at face value.
_RESTRICTED_FAMILIES = (
    (re.compile(r"llama[-_.]?[234]|llama[-_.]?3\.\d|(?:^|[-_/])llama-?2|meta-llama", re.IGNORECASE), "Llama"),
    (re.compile(r"gemma|google/", re.IGNORECASE), "Gemma"),
)


def restricted_family(info: Any) -> Optional[str]:
    """"Llama" / "Gemma" if this repo or the model it came from belongs to a family
    with its own license terms (whatever license the uploader tagged), else None."""
    repo = str(_attr(info, "id", default=""))
    text = " ".join(x for x in (repo, _base_model(info) or "") if x)
    for pattern, family in _RESTRICTED_FAMILIES:
        if pattern.search(text):
            return family
    return None


# Sizes in names: "4B", "0.6B", "360M", "30B-A3B" (Mixture-of-Experts: 30B total, 3B active),
# "3b-a800m" (800M active), "8x7B".
_SIZE_RE = re.compile(
    r"(?:^|[-_/\s])(?:(\d+)x)?(\d+(?:\.\d+)?)([bm])(?:-a(\d+(?:\.\d+)?)([bm]))?(?=$|[-_.\s])", re.IGNORECASE
)


def params_from_name(name: str) -> tuple[Optional[float], Optional[float]]:
    """(total, active) parameters in billions parsed from a model name.

    "Qwen3-4B" -> (4.0, None); "Qwen3-30B-A3B" -> (30.0, 3.0); "SmolLM2-360M" -> (0.36, None);
    "granite-3.1-3b-a800m" -> (3.3, 0.8); "gpt-oss-20b" -> (20.9, 3.6) and "Mixtral-8x7B"
    -> (46.7, 12.9) from a small table of Mixture-of-Experts models; "Phi-4-mini" -> (None, None).
    Billions win over millions, so "Qwen2.5-7B-Instruct-1M" (a 1M-token context) is 7B.
    """
    lowered = name.lower()
    for pattern, known_total, known_active in _KNOWN_MOE:
        if re.search(pattern, lowered):
            return known_total, known_active
    matches = list(_SIZE_RE.finditer(name))
    match = next((m for m in matches if m.group(3).lower() == "b"), None) or next(iter(matches), None)
    if match is None:
        return None, None
    experts, size, unit, active, active_unit = match.groups()
    total = float(size) * (int(experts) if experts else 1) / (1000 if unit.lower() == "m" else 1)
    active_b = None
    if active:
        active_b = float(active) / (1000 if (active_unit or "b").lower() == "m" else 1)
    return round(total, 3), active_b


def _params(info: Any, files_bytes: dict[str, int] | None = None) -> tuple[Optional[float], Optional[float]]:
    """Total/active params: GGUF header first, then the name, then guessed from the Q4 file size."""
    repo = str(_attr(info, "id", default=""))
    name_total, active = params_from_name(_repo_name(repo))
    if name_total is None or active is None:
        base_total, base_active = params_from_name(_repo_name(_base_model(info) or repo))
        name_total = name_total if name_total is not None else base_total
        active = active if active is not None else base_active
    total = _gguf(info).get("total")
    if isinstance(total, (int, float)) and total > 0:
        return round(total / 1e9, 3), active
    if name_total:
        return name_total, active
    sized = {q: s for q, s in (files_bytes or {}).items() if s and catalog.quant_bits(q)}
    if sized:
        quant = _default_quant(sized)  # a ~4-bit file: what catalog's size formula is tuned for
        bits = catalog.quant_bits(quant) or 4.8
        return round(sized[quant] * 8 / bits / 1.05 / 1e9, 2), active  # inverse of catalog.estimate_quant_size_gb
    return None, active


# ---------------------------------------------------------------------------
# Screening: is this a model we'd happily put in front of a player?
# ---------------------------------------------------------------------------

# Not family-friendly: safety-removed ("abliterated", "uncensored"...) or adult roleplay tunes.
_UNSAFE_RE = re.compile(
    r"abliterat|obliterat|uncensor|decensor|unfiltered|unaligned|jailbr|heretic|nsfw|nsfl|"
    r"not-for-all-audiences|erotic|\berp\b|lewd|porn|hentai|smut|horny|roleplay|\brp\b|"
    r"dolphin|venice",  # the Dolphin / Venice tunes are marketed as uncensored
    re.IGNORECASE,
)
# Name words for models that aren't storytellers: coders, embedders, rerankers, vision, speech...
_NOT_CHAT_WORDS = frozenset(
    "coder code codestral devstral codellama starcoder starcoder2 codegemma embed embedding embeddings "
    "rerank reranker guard guardian shieldgemma math prover vl vision ocr tts asr audio speech whisper "
    "omni mmproj draft clip siglip reward prm classifier translate translation "
    # agents and tool users (coding, web research, function calling) - they follow
    # tool formats, not stories:
    "agent agentic swe dev openhands search research tool tools function functions "
    # narrow domains and pipelines:
    "medical med clinical medicine rag structure structured".split()
)
# ...and name parts that mark a specialist even inside a longer word
# ("WizardCoder", "OpenCoder", "Mathstral", "deepseek-math", "Qwen3Guard",
# "Tongyi-DeepResearch").
_NOT_CHAT_PARTS = ("coder", "math", "embed", "guard", "research")
_BASE_WORDS = frozenset({"base", "pretrain", "pretrained", "pt"})
# Families that also ship *base* checkpoints with a chat template (so the Hub
# tags them "conversational"): for these we want to see Instruct/Chat in the name.
_BASE_WITH_TEMPLATE_RE = re.compile(r"qwen2(?:\.5)?(?![\d.])|llama|mistral|gemma|olmo|falcon", re.IGNORECASE)
_NOT_CHAT_PIPELINES = frozenset({
    "feature-extraction", "sentence-similarity", "text-ranking", "text-classification", "token-classification",
    "fill-mask", "automatic-speech-recognition", "text-to-speech", "text-to-audio", "text-to-image",
    "image-to-text", "translation", "summarization", "zero-shot-classification",
})
_INSTRUCT_RE = re.compile(r"instruct|chat|(?:^|[-_])it(?:$|[-_])|assistant|thinking", re.IGNORECASE)
_CHAT_FAMILIES = ("qwen3", "qwq", "gpt-oss", "phi-4", "smollm3", "magistral", "deepseek-r1", "granite-3",
                  "granite-4", "exaone", "hermes", "openthinker")

# --- Does the model think out loud, and can it be asked not to? -----------------
#
# The game asks a thinking model to answer straight away when time is short
# (``think=False``: the chat template's ``enable_thinking=false`` /
# ``reasoning_effort`` switch). That only works if the template *has* such a
# switch. Some templates force thinking on - they end the prompt with
# ``<think>`` and ignore the switch - so the model spends the whole answer
# budget thinking, every time. We read the chat template the Hub returns with
# each search result (``gguf.chat_template``), and fall back to name rules.
THINKING_NONE, THINKING_SWITCHABLE, THINKING_ALWAYS = "none", "switchable", "always"
# A template that knows how to turn thinking off (Qwen3's enable_thinking,
# gpt-oss's reasoning_effort, Seed-OSS's thinking_budget, SmolLM3's /no_think...).
_THINK_SWITCH_RE = re.compile(
    r"enable_thinking|reasoning_effort|thinking_budget|/no_think|\bthinking\s+is\s+(?:defined|true|false)"
    r"|\bthinking_mode\b|\breasoning_mode\b",
    re.IGNORECASE,
)
# ...one that opens a thinking block itself when it hands over to the model.
_FORCED_THINK_RE = re.compile(
    r"add_generation_prompt[^%]*%\}(?:(?!\{%-?\s*(?:if|else|elif|endif)\b).){0,240}?(?:<think>|\[THINK\])",
    re.IGNORECASE | re.DOTALL,
)
# Names of models that always think, whatever you ask (used when there's no template).
_ALWAYS_THINKS_RE = re.compile(
    r"qwq|deepseek-r1|(?:^|[-_])r1(?:$|[-_])|thinking|thinker|(?:^|[-_])think(?:$|[-_])|[-_]reasoning"
    r"|exaone-deep|openreasoning|magistral",
    re.IGNORECASE,
)
# Names of models that think but can be switched off.
_SWITCHABLE_THINKS_RE = re.compile(
    r"qwen3(?!.*instruct)|gpt-oss|smollm3|seed-oss|glm-4\.[5-9]|hunyuan|granite-3\.[23]",
    re.IGNORECASE,
)


def thinking_mode_for(name: str, chat_template: Optional[str] = None) -> str:
    """"none", "switchable" or "always": how a model thinks out loud.

    From its chat template when we have it (a thinking switch = switchable;
    ``<think>`` opened for the model with no switch = always), else from its
    name. Names that always think win over a template that merely lacks a
    switch (some DeepSeek-R1 distills don't force the tag, but think anyway).
    """
    template = chat_template if isinstance(chat_template, str) else ""
    if template:
        if _THINK_SWITCH_RE.search(template):
            return THINKING_SWITCHABLE
        if _FORCED_THINK_RE.search(template):
            return THINKING_ALWAYS
    if _ALWAYS_THINKS_RE.search(name):
        return THINKING_ALWAYS
    if _SWITCHABLE_THINKS_RE.search(name):
        return THINKING_SWITCHABLE
    return THINKING_NONE


def _rules_fingerprint() -> str:
    """A short fingerprint of every screening and labelling rule (and the game's version).

    Saved with the model list: when a rule changes in an update, a list saved
    under the old rules is no longer "fresh", and is re-screened on load.
    """
    import hashlib

    from . import __version__

    parts = [
        __version__, _UNSAFE_RE.pattern, sorted(_NOT_CHAT_WORDS), _NOT_CHAT_PARTS, sorted(_BASE_WORDS),
        _BASE_WITH_TEMPLATE_RE.pattern, sorted(_NOT_CHAT_PIPELINES), _INSTRUCT_RE.pattern, _CHAT_FAMILIES,
        _THINK_SWITCH_RE.pattern, _FORCED_THINK_RE.pattern, _ALWAYS_THINKS_RE.pattern, _SWITCHABLE_THINKS_RE.pattern,
        [(p.pattern, f) for p, f in _RESTRICTED_FAMILIES], MIN_PARAMS_B, MAX_PARAMS_B, MIN_DOWNLOADS_UNTRUSTED,
        sorted(catalog.PERMISSIVE_LICENSES), catalog.TRUSTED_PUBLISHERS,
        _KNOWN_MOE, sorted(MOE_ARCHITECTURES), sorted(MAYBE_MOE_ARCHITECTURES), MOE_UNKNOWN_ACTIVE_SHARE,
        [(p, f) for p, f in _FAMILIES],
    ]
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:16]


def _words(name: str) -> list[str]:
    return [w for w in re.split(r"[-_.\s/]+", name.lower()) if w]


def _is_chat_model(info: Any, name: str) -> bool:
    """Instruction/chat tuned? "Instruct"/"Chat"/"-it" in the name, a family that only
    ships chat models, or the Hub's "conversational" tag / a chat template in the GGUF
    header - except for families whose *base* models also carry a chat template."""
    lowered = name.lower()
    if _INSTRUCT_RE.search(name) or any(f in lowered for f in _CHAT_FAMILIES):
        return True
    if _BASE_WITH_TEMPLATE_RE.search(lowered):
        return False  # e.g. "Qwen2.5-7B": a base model, despite its chat template
    return "conversational" in _tags(info) or bool(_gguf(info).get("chat_template"))


def _is_base_name(name: str) -> bool:
    words = _words(name)
    found = _BASE_WORDS.intersection(words)
    if found == {"pt"} and "ernie" in words:
        return False  # Baidu's "ERNIE-...-PT" means PyTorch weights of the chat model ("-Base-PT" is the base)
    return bool(found)


def _is_gated(info: Any) -> bool:
    gated = _attr(info, "gated")
    return gated not in (None, False, "false", "False", "")


def _publisher_rank(repo_id: str) -> int:
    """0 for the first trusted publisher, 1 for the next... len(TRUSTED) for everyone else."""
    publisher = repo_id.split("/", 1)[0].lower()
    ranks = [p.lower() for p in catalog.TRUSTED_PUBLISHERS]
    return ranks.index(publisher) if publisher in ranks else len(ranks)


def not_family_friendly(info: Any) -> bool:
    """Is this Hub model marked as uncensored, safety-removed ("abliterated") or adult content?

    Looks at the repo id and its tags (a GGUF of such a fine-tune names it in a
    ``base_model:`` tag). The model search leaves these out, and a model the
    player names themselves (the "custom" pick, ``--model``) is refused too.
    """
    repo = str(_attr(info, "id", default=""))
    return bool(_UNSAFE_RE.search(" ".join([repo, *_tags(info)])))


def rejection_reason(info: Any, *, allow_all_licenses: bool = False) -> Optional[str]:
    """Why we'd leave this Hub model out, in plain English - or None if it's a good candidate.

    Checks, in order: private/disabled, family-friendliness, "not a storyteller"
    (coder/maths/embedding/reranker/vision/speech...), base (not chat-tuned) models,
    license (Apache-2.0/MIT unless `allow_all_licenses`; unknown = excluded; a
    Llama or Gemma derivative re-labelled "apache-2.0" is not taken at face value),
    size, and popularity for publishers we don't know.
    """
    repo = str(_attr(info, "id", default=""))
    if "/" not in repo:
        return "not a normal Hugging Face repo id"
    if _attr(info, "private") or _attr(info, "disabled"):
        return "private or disabled"
    name = _repo_name(repo)
    if not_family_friendly(info):
        return "not family-friendly (uncensored, abliterated or adult content)"
    pipeline = _attr(info, "pipeline_tag")
    words = _words(name) + _words(_repo_name(_base_model(info) or ""))  # a GGUF of a specialist is one too
    if (pipeline in _NOT_CHAT_PIPELINES or _NOT_CHAT_WORDS.intersection(words)
            or any(part in word for word in words for part in _NOT_CHAT_PARTS)):
        return "a specialist model (coding, maths, embeddings, vision, speech...), not a storyteller"
    if _is_base_name(name) or not _is_chat_model(info, name):
        return "a base model that isn't tuned to chat or follow instructions"
    license_id = license_of(info)
    if not allow_all_licenses and not catalog.is_permissive(license_id):
        return f"its license ({license_id or 'unknown'}) isn't Apache-2.0 or MIT"
    family = restricted_family(info)
    if not allow_all_licenses and family:
        return (f"its license says {license_id}, but it's built on {family}, whose own license terms still apply "
                "- so it isn't Apache-2.0 or MIT")
    total, _ = _params(info)
    if total is not None and total < MIN_PARAMS_B:
        return "too small to tell a good story"
    if total is not None and total > MAX_PARAMS_B:
        return "far too big for a home computer"
    unknown_publisher = _publisher_rank(repo) >= len(catalog.TRUSTED_PUBLISHERS)
    if unknown_publisher and int(_attr(info, "downloads", default=0) or 0) < MIN_DOWNLOADS_UNTRUSTED:
        return "from a publisher we don't know, and not widely used yet"
    return None


def rescreen_entry(entry: ModelEntry, *, allow_all_licenses: bool = False) -> Optional[ModelEntry]:
    """Apply today's rules to a model saved in an older list: None if it's now left
    out, else the entry with its license label and thinking mode brought up to date.

    Only what can be judged from the saved entry is re-checked (name, family,
    license, size, popularity) - never a reason to go online.
    """
    if entry.source == "curated":
        return entry
    base = entry.base_model or ""
    declared = entry.license if catalog.is_permissive(entry.license) else "other"
    info = {
        "id": entry.hf_repo,
        "tags": ["conversational", f"license:{declared.lower()}"] + ([f"base_model:{base}"] if base else []),
        "cardData": {"base_model": base} if base else {},
        "downloads": entry.downloads,
        "gguf": {"total": entry.params_b * 1e9},
    }
    if rejection_reason(info, allow_all_licenses=True) is not None:
        return None  # e.g. now a specialist, not family-friendly, or too small
    label = entry.license
    family = restricted_family(info)
    if family and catalog.is_permissive(label):
        label = f"{family} license (tagged {label})"
    if not allow_all_licenses and not catalog.is_permissive(label):
        return None
    name_mode = thinking_mode_for(f"{_repo_name(entry.hf_repo)} {_repo_name(base)}")
    mode = catalog.thinking_mode(entry)
    if name_mode == THINKING_ALWAYS:
        mode = THINKING_ALWAYS  # the name rules know it always thinks, whatever was saved before
    active = _rescreened_active(entry)
    changes: dict[str, Any] = {"license": label, "thinking": mode, "reasoning": mode != THINKING_NONE,
                               "active_params_b": active,
                               "family": _family(entry.hf_repo, base or None, entry.architecture)}
    if active != entry.active_params_b and "Mixture-of-Experts" in (entry.blurb or ""):
        changes["blurb"] = _blurb(changes["family"], entry.params_b, active, mode, entry.hf_repo.split("/", 1)[0])
    return dataclasses.replace(entry, **changes)


def _rescreened_active(entry: ModelEntry) -> Optional[float]:
    """A saved model's active size under today's rules: the name tables win, and
    an old "a quarter of it, probably" guess is dropped for an architecture that
    isn't known to be Mixture-of-Experts (e.g. a dense Granite-4.0-H hybrid)."""
    for name in (_repo_name(entry.hf_repo), _repo_name(entry.base_model or "")):
        total, active = params_from_name(name) if name else (None, None)
        if active is not None and total is not None and 0 < active < entry.params_b * 1.1:
            return active
    active = entry.active_params_b
    guessed = active is not None and abs(active - round(entry.params_b * MOE_UNKNOWN_ACTIVE_SHARE, 2)) < 0.011
    if guessed and not _is_moe(entry.architecture, None):
        return None
    return active


# ---------------------------------------------------------------------------
# Hub model -> ModelEntry
# ---------------------------------------------------------------------------

_FAMILIES = (
    (r"deepseek", "DeepSeek"), (r"gpt-?oss", "gpt-oss"), (r"qwq", "QwQ"), (r"qwen3", "Qwen3"),
    (r"qwen2\.5", "Qwen2.5"), (r"qwen", "Qwen"), (r"phi-?4", "Phi-4"), (r"phi-?3", "Phi-3"),
    (r"hermes", "Hermes"), (r"magistral|ministral|mistral|mixtral", "Mistral"), (r"smollm3", "SmolLM3"),
    (r"smollm2", "SmolLM2"), (r"smollm", "SmolLM"), (r"granite", "Granite"), (r"llama", "Llama"),
    (r"gemma", "Gemma"), (r"olmo", "OLMo"), (r"exaone", "EXAONE"), (r"glm", "GLM"), (r"falcon", "Falcon"),
    (r"lfm", "LFM"),
)
_CAPITALISE = {"instruct": "Instruct", "chat": "Chat", "thinking": "Thinking", "reasoning": "Reasoning"}

RULES_VERSION = _rules_fingerprint()  # (after every table it fingerprints is defined)


def prettify_repo_name(repo_id: str) -> str:
    """A friendly display name: "unsloth/Qwen3-30B-A3B-GGUF" -> "Qwen3 30B-A3B",
    "bartowski/microsoft_Phi-4-mini-instruct-GGUF" -> "Phi-4 mini Instruct"."""
    words: list[str] = []
    for i, token in enumerate(t for t in re.split(r"[-_\s]+", _repo_name(repo_id)) if t):
        if re.fullmatch(r"\d+(\.\d+)?[bBmM]", token):
            token = token.upper()
        if words and re.fullmatch(r"[aA]\d+(\.\d+)?[bB]", token) and re.fullmatch(r"\d+(\.\d+)?B", words[-1]):
            words[-1] += "-" + token.upper()  # "30B" + "A3B" -> "30B-A3B"
        elif words and token.lower() == "oss" and words[-1].lower() == "gpt":
            words[-1] += "-oss"
        elif i == 1 and re.fullmatch(r"\d+(\.\d+)*", token) and words[0].isalpha():
            words[-1] += "-" + token  # "Phi" + "4" -> "Phi-4", "Llama" + "3.2" -> "Llama-3.2"
        else:
            words.append(_CAPITALISE.get(token.lower(), token))
    return " ".join(words) or repo_id


def _family(repo_id: str, base_model: Optional[str], architecture: Optional[str] = None) -> str:
    """The model family ("Qwen3", "Granite"...): from the names, else the GGUF
    architecture - a fine-tune of Qwen3 is a Qwen3 model, whoever uploaded it -
    and only then the first word of the name."""
    for text in (f"{_repo_name(repo_id)} {_repo_name(base_model or '')}", architecture or ""):
        for pattern, family in _FAMILIES:
            if re.search(pattern, text.lower()):
                return family
    if architecture and re.fullmatch(r"[a-z][a-z0-9_\-]*", architecture.lower()):
        return architecture.split("_")[0].split("-")[0].capitalize()
    return prettify_repo_name(repo_id).split(" ")[0]


def _default_quant(quants: Iterable[str]) -> str:
    """The quant a repo is listed with: our usual sweet spot, else the one nearest ~4.8 bits
    (Q4_K_M's size). The fit engine may still pick another one for each computer."""
    available = {q.upper(): q for q in quants}
    for wanted in FALLBACK_QUANT_ORDER:
        if wanted in available:
            return available[wanted]
    return min(available.values(), key=lambda q: (abs((catalog.quant_bits(q) or 99.0) - 4.8), q))


def _format_params(params_b: float) -> str:
    return f"{params_b * 1000:.0f}M" if params_b < 1 else f"{params_b:.3g}B"


def _blurb(family: str, params_b: float, active_b: Optional[float], thinking: str, publisher: str) -> str:
    """One friendly sentence for the selection table."""
    size = _format_params(params_b)
    if active_b:
        text = (f"A {size} Mixture-of-Experts {family} model that only wakes ~{_format_params(active_b)} "
                f"parameters per word, so it's quick for its size")
    else:
        text = f"A {size}-parameter {family} chat model, packaged by {publisher}"
    if thinking == THINKING_ALWAYS:
        return text + "; it always thinks at length before answering, so turns are slow."
    return text + ("; it shows its thinking." if thinking == THINKING_SWITCHABLE else ".")


def entry_from_hub(info: Any, files: list[tuple[str, int]], header: Optional[dict] = None) -> Optional[ModelEntry]:
    """Turn one Hub search result plus its (file name, bytes) list into a ModelEntry.

    Pure: no network, no filtering by license (see `rejection_reason`).
    `header` is the start of one GGUF file's settings (see `read_gguf_header`),
    if we read it: it tells a Mixture-of-Experts model's real active size.
    Returns None when the repo has no usable GGUF quant (e.g. only a vision
    projector) or its size can't be worked out.
    """
    repo = str(_attr(info, "id", default=""))
    if "/" not in repo:
        return None
    groups = {q: v for q, v in group_quant_files(files).items() if catalog.quant_bits(q) and v[1] > 0}
    if not groups:
        return None
    params_b, active_b = _params(info, {q: size for q, (_, size) in groups.items()})
    if not params_b:
        return None
    if active_b is None and header:
        active_b = active_params_from_header(header, params_b)
    architecture = _gguf(info).get("architecture") if isinstance(_gguf(info).get("architecture"), str) else None
    if active_b is None and _is_moe(architecture, header):
        # A Mixture-of-Experts model whose active size we couldn't find: much
        # better to guess than to rank it like a dense model of the same size.
        active_b = round(params_b * MOE_UNKNOWN_ACTIVE_SHARE, 2)
    if active_b is not None and not 0 < active_b < params_b:
        active_b = None

    base = _base_model(info)
    family = _family(repo, base, architecture or (header or {}).get("architecture"))
    quant = _default_quant(groups)
    quant_files, quant_bytes = groups[quant]
    gguf = _gguf(info)
    thinking = thinking_mode_for(f"{_repo_name(repo)} {_repo_name(base or '')}", gguf.get("chat_template"))
    reasoning = thinking != THINKING_NONE
    context = gguf.get("context_length")
    seed = _seed_for(_base_key(repo, base))
    publisher = repo.split("/", 1)[0]
    declared = license_of(info) or "unknown"
    restricted = restricted_family(info)
    license_label = f"{restricted} license (tagged {declared})" if restricted and catalog.is_permissive(declared) else declared
    # Link the original model's card when we know it: that's where its license lives.
    license_url = f"https://huggingface.co/{base}" if base and "/" in base else f"https://huggingface.co/{repo}"
    return ModelEntry(
        key=repo,
        display_name=prettify_repo_name(repo),
        family=family,
        params_b=round(params_b, 2),
        active_params_b=active_b,
        license=license_label,
        license_url=license_url,
        hf_repo=repo,
        quant=quant,
        file_size_gb=round(quant_bytes / 1e9, 2),
        ollama_ref=f"hf.co/{repo}:{quant}",
        reasoning=reasoning,
        thinking=thinking,
        blurb=seed.blurb if seed else _blurb(family, params_b, active_b, thinking, publisher),
        source="huggingface",
        quant_options=tuple((q, round(size / 1e9, 2)) for q, (_, size) in groups.items()),
        gguf_files=quant_files,
        downloads=int(_attr(info, "downloads", default=0) or 0),
        likes=int(_attr(info, "likes", default=0) or 0),
        base_model=base,
        architecture=architecture,
        native_context=int(context) if isinstance(context, (int, float)) and context > 0 else None,
        gated=_is_gated(info),
        kv_shape=kv_shape_from_header(header),
    )


def _is_moe(architecture: Optional[str], header: Optional[dict]) -> bool:
    """Is this a Mixture-of-Experts model? The GGUF header's expert count decides when
    we have it (0 or 1 expert = dense, whatever the architecture); otherwise the
    architecture name - except for those that come in both kinds."""
    arch = (architecture or "").lower()
    if header:
        experts = header.get("expert_count")
        if isinstance(experts, int) and not isinstance(experts, bool):
            return experts > 1
        arch = str(header.get("architecture") or arch).lower()
    if arch in MAYBE_MOE_ARCHITECTURES:
        return False  # can't tell without the header: dense is the safe guess here
    return arch in MOE_ARCHITECTURES or arch.endswith("moe")


def _maybe_moe(architecture: Optional[str]) -> bool:
    """Might this architecture be Mixture-of-Experts (so the header is worth reading)?"""
    arch = (architecture or "").lower()
    return arch in MOE_ARCHITECTURES or arch in MAYBE_MOE_ARCHITECTURES or arch.endswith("moe")


# ---------------------------------------------------------------------------
# Reading a GGUF file's header (its "label": architecture, sizes, experts...)
# ---------------------------------------------------------------------------

_GGUF_SCALAR = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_HEADER_KEYS = ("expert_count", "expert_used_count", "expert_shared_count", "block_count", "embedding_length",
                "feed_forward_length", "expert_feed_forward_length", "expert_shared_feed_forward_length",
                "leading_dense_block_count", "context_length", "attention.head_count_kv", "attention.head_count",
                "attention.key_length")


def read_gguf_header(data: bytes) -> dict:
    """Pull the model settings out of the first bytes of a GGUF file.

    A GGUF file starts with "GGUF", a version, the tensor count and a list of
    key/value settings ("qwen3moe.expert_count" = 128...). Those settings come
    before the big vocabulary arrays, so the first few hundred KB are enough.
    Returns {"architecture": ..., "expert_count": ..., ...} with whatever was
    found (keys without the architecture prefix), or {} if it isn't a GGUF.
    """
    if len(data) < 24 or data[:4] != b"GGUF":
        return {}
    pos = 4
    values: dict[str, Any] = {}

    def take(fmt: str) -> Any:
        nonlocal pos
        size = struct.calcsize(fmt)
        if pos + size > len(data):
            raise EOFError
        (value,) = struct.unpack_from(fmt, data, pos)
        pos += size
        return value

    def take_string() -> str:
        nonlocal pos
        length = take("<Q")
        if length > 1 << 20 or pos + length > len(data):
            raise EOFError
        raw = data[pos:pos + length]
        pos += length
        return raw.decode("utf-8", "replace")

    def take_value(kind: int) -> Any:
        nonlocal pos
        if kind in _GGUF_SCALAR:
            return take(_GGUF_SCALAR[kind])
        if kind == 8:
            return take_string()
        if kind == 9:
            item_kind, count = take("<I"), take("<Q")
            if item_kind in _GGUF_SCALAR:
                size = struct.calcsize(_GGUF_SCALAR[item_kind]) * count
                if pos + size > len(data):
                    raise EOFError
                pos += size  # we never need array contents: skip them
                return None
            for _ in range(count):
                take_value(item_kind)
            return None
        raise ValueError(f"unknown GGUF value type {kind}")

    try:
        version = take("<I")
        if version < 2:
            return {}
        take("<Q")  # tensor count
        kv_count = take("<Q")
        for _ in range(min(kv_count, 100_000)):
            key = take_string()
            value = take_value(take("<I"))
            if key == "general.architecture" and isinstance(value, str):
                values["architecture"] = value
            elif "." in key and isinstance(value, (int, float)) and not isinstance(value, bool):
                short = key.split(".", 1)[1]
                if short in _HEADER_KEYS:
                    values[short] = value
    except (EOFError, ValueError, struct.error):
        pass  # we read as far as the bytes we have allow
    return values


def kv_shape_from_header(header: Optional[dict]) -> tuple[int, ...]:
    """(layers, KV heads, head size) from a GGUF header, for exact KV-cache maths; () if unknown.

    A model without grouped-query attention has as many KV heads as attention
    heads (GGUF leaves ``head_count_kv`` out then). The head size is
    ``key_length`` if given, else ``embedding_length / head_count``.
    """
    if not header:
        return ()
    try:
        layers = int(header.get("block_count") or 0)
        heads = int(header.get("attention.head_count") or 0)
        kv_heads = int(header.get("attention.head_count_kv") or heads)
        head_size = int(header.get("attention.key_length") or 0)
        if not head_size and heads:
            head_size = int(header.get("embedding_length") or 0) // heads
    except (TypeError, ValueError):
        return ()
    if layers <= 0 or kv_heads <= 0 or head_size <= 0 or layers > 1000 or kv_heads > 1024 or head_size > 4096:
        return ()
    return layers, kv_heads, head_size


def active_params_from_header(header: dict, total_b: float) -> Optional[float]:
    """Parameters used per word by a Mixture-of-Experts model, from its GGUF settings.

    Each MoE layer holds `expert_count` small feed-forward "experts" but only
    runs `expert_used_count` of them per token. One expert has about
    3 x embedding_length x expert_feed_forward_length weights (gate, up and
    down projections), so the unused ones are subtracted from the total.
    None if this isn't an MoE model or a needed number is missing.
    """
    try:
        experts = int(header.get("expert_count") or 0)
        used = int(header.get("expert_used_count") or 0)
        layers = int(header.get("block_count") or 0) - int(header.get("leading_dense_block_count") or 0)
        width = int(header.get("embedding_length") or 0)
        expert_ff = int(header.get("expert_feed_forward_length") or header.get("feed_forward_length") or 0)
    except (TypeError, ValueError):
        return None
    if experts <= 1 or not 0 < used < experts or layers <= 0 or width <= 0 or expert_ff <= 0 or total_b <= 0:
        return None
    unused_b = (experts - used) * 3 * width * expert_ff * layers / 1e9
    active = total_b - unused_b
    if not 0 < active < total_b:
        return None
    return round(active, 2)


def _real_header_fetcher(api: Any) -> Optional[Callable[[str, str], dict]]:
    """Read the first bytes of a GGUF file on the Hub (an HTTP Range request).

    Only for the real ``HfApi`` (tests use fakes, which may offer their own
    ``gguf_header(repo, filename)``). The Hub's own file viewer does the same.
    """
    custom = getattr(api, "gguf_header", None)
    if callable(custom):
        return custom
    try:
        from huggingface_hub import HfApi, hf_hub_url
        from huggingface_hub.utils import build_hf_headers, get_session
    except ImportError:
        return None
    if not isinstance(api, HfApi):
        return None

    def fetch(repo_id: str, filename: str) -> dict:
        headers = build_hf_headers()
        headers["Range"] = f"bytes=0-{GGUF_HEADER_BYTES - 1}"
        resp = get_session().get(hf_hub_url(repo_id, filename), headers=headers)
        if resp.status_code not in (200, 206):
            return {}
        return read_gguf_header(resp.content[:GGUF_HEADER_BYTES])

    return fetch


def _seed_for(base_key: str) -> Optional[ModelEntry]:
    """The curated seed for the same original model, if we have one."""
    return next((m for m in catalog.MODEL_CATALOG if _base_key(m.hf_repo, m.base_model) == base_key), None)


# ---------------------------------------------------------------------------
# Talking to the Hub
# ---------------------------------------------------------------------------


def _file_size(item: Any) -> int:
    size = _attr(item, "size")
    if not size:
        lfs = _attr(item, "lfs")
        size = _attr(lfs, "size") if lfs is not None else None
    return int(size or 0)


_TIMEOUTS_CONFIGURED = False
_TIMEOUTS_LOCK = threading.Lock()


def configure_hub_timeouts(connect_s: float = HUB_CONNECT_TIMEOUT_S, read_s: float = HUB_READ_TIMEOUT_S) -> bool:
    """Give every huggingface_hub request a finite time limit (once per run).

    huggingface_hub's shared HTTP client waits forever by default. We wrap its
    own client factory and set a connect timeout and a per-read timeout, so a
    stalled connection raises an error instead of freezing the game. Requests
    that pass their own timeout (like file downloads) keep it. Returns True if
    the limit is in place.
    """
    global _TIMEOUTS_CONFIGURED
    with _TIMEOUTS_LOCK:
        if _TIMEOUTS_CONFIGURED:
            return True
        try:
            import huggingface_hub
            from huggingface_hub.utils import _http
        except ImportError:
            return False
        set_factory = getattr(huggingface_hub, "set_client_factory", None)
        default_factory = getattr(_http, "default_client_factory", None)
        if set_factory is None or default_factory is None:
            return False

        def factory() -> Any:
            client = default_factory()
            try:
                client.timeout = type(client.timeout)(read_s, connect=connect_s)
            except Exception:
                pass
            return client

        set_factory(factory)
        _TIMEOUTS_CONFIGURED = True
        return True


def make_hub_api() -> Any:
    """``huggingface_hub.HfApi()`` with finite request timeouts (see `configure_hub_timeouts`)."""
    configure_hub_timeouts()
    from huggingface_hub import HfApi  # imported lazily: only needed when going online

    return HfApi()


def with_deadline(func: Callable[[], Any], seconds: float, what: str = "Hugging Face") -> Any:
    """Run `func` on a background thread and wait at most `seconds` for it.

    Returns its result (or raises its exception). If time runs out, raises
    TimeoutError - and because the thread is a *daemon*, a request that is
    still hanging can never stop the game from exiting.
    """
    box: dict[str, Any] = {}
    done = threading.Event()

    def run() -> None:
        try:
            box["value"] = func()
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc
        finally:
            done.set()

    threading.Thread(target=run, name="hub-call", daemon=True).start()
    if not done.wait(timeout=max(0.0, seconds)):
        raise TimeoutError(f"{what} didn't answer within {int(seconds)} seconds")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def repo_gguf_files(api: Any, repo_id: str) -> list[tuple[str, int]]:
    """Every .gguf file in a repo with its size in bytes (0 = unknown). One Hub request.

    Uses `list_repo_tree(recursive=True)`, falling back to
    `model_info(files_metadata=True).siblings` for API objects without it.
    Network and "not found" errors are raised to the caller.
    """
    lister = getattr(api, "list_repo_tree", None)
    if lister is not None:
        items: Iterable[Any] = lister(repo_id, recursive=True, expand=False)
        pairs = ((_attr(item, "path"), item) for item in items)
    else:
        info = api.model_info(repo_id, files_metadata=True)
        pairs = ((_attr(s, "rfilename", "path"), s) for s in (_attr(info, "siblings", default=[]) or []))
    return [(str(path), _file_size(item)) for path, item in pairs if path and str(path).lower().endswith(".gguf")]


def _search_hub(api: Any, author: Optional[str], limit: int) -> list[Any]:
    """One `list_models` call: the most-downloaded GGUF text-generation repos (+ metadata)."""
    kwargs: dict[str, Any] = dict(filter="gguf", pipeline_tag="text-generation", sort="downloads",
                                  limit=limit, expand=list(EXPAND_FIELDS))
    if author:
        kwargs["author"] = author
    try:
        return list(itertools.islice(api.list_models(**kwargs), limit))
    except TypeError:
        # Older huggingface_hub without `expand`: ask for the model card data instead.
        kwargs.pop("expand")
        kwargs["cardData"] = True
        return list(itertools.islice(api.list_models(**kwargs), limit))


_SKIPPED = object()  # marks a job we never ran because time was up


def _run_parallel(jobs: list[Callable[[], Any]], deadline: float, clock: Callable[[], float]) -> list[Any]:
    """Run jobs on a few worker threads. Returns, in order, each job's result, the
    exception it raised, or `_SKIPPED` if the time budget ran out first.

    The workers are *daemon* threads: a request still hanging when time is up
    is simply abandoned, and can't keep the game from exiting afterwards (a
    ThreadPoolExecutor would make Python wait for it at exit).
    """
    if not jobs:
        return []
    results: list[Any] = [_SKIPPED] * len(jobs)
    todo: "queue.Queue[tuple[int, Callable[[], Any]]]" = queue.Queue()
    for index, job in enumerate(jobs):
        todo.put((index, job))
    lock = threading.Lock()
    left = [len(jobs)]
    finished = threading.Event()

    def worker() -> None:
        while True:
            try:
                index, job = todo.get_nowait()
            except queue.Empty:
                return
            if clock() >= deadline:
                value: Any = _SKIPPED  # don't start new requests after the deadline
            else:
                try:
                    value = job()
                except BaseException as exc:
                    value = exc
            with lock:
                results[index] = value
                left[0] -= 1
                if left[0] == 0:
                    finished.set()

    for _ in range(max(1, min(MAX_WORKERS, len(jobs)))):
        threading.Thread(target=worker, name="hf-discovery", daemon=True).start()
    finished.wait(timeout=max(0.0, deadline - clock()))
    with lock:
        return list(results)  # anything still running stays _SKIPPED


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _fetch_live(api: Any, *, allow_all_licenses: bool, max_candidates: int, timeout_s: float,
                clock: Callable[[], float]) -> tuple[list[ModelEntry], list[str], bool]:
    """Search, screen, measure. Returns (entries, notes, complete?).

    `complete` is False if any search or file listing failed or ran out of
    time - such a result is only cached briefly. Raises if the Hub couldn't
    be searched at all.
    """
    deadline = clock() + max(0.0, timeout_s)
    notes: list[str] = []

    # 1. Search: one call per trusted publisher + one global call.
    authors: list[Optional[str]] = [*catalog.TRUSTED_PUBLISHERS, None]
    searches = _run_parallel(
        [lambda a=a: _search_hub(api, a, PER_PUBLISHER_LIMIT if a else GLOBAL_LIMIT) for a in authors],
        deadline, clock,
    )
    failures = [r for r in searches if isinstance(r, BaseException)]
    if all(isinstance(r, BaseException) or r is _SKIPPED for r in searches):
        raise failures[0] if failures else TimeoutError("Hugging Face took too long to answer")
    infos: dict[str, Any] = {}
    for result in searches:
        if isinstance(result, list):
            for info in result:
                infos.setdefault(str(_attr(info, "id", default="")), info)

    # 2. Screen, then keep one repo per original model.
    license_skips = 0
    kept: list[Any] = []
    for info in infos.values():
        reason = rejection_reason(info, allow_all_licenses=allow_all_licenses)
        if reason is None:
            kept.append(info)
        elif reason.startswith("its license"):
            license_skips += 1
    if license_skips and not allow_all_licenses:
        notes.append(f"Left out {_plural(license_skips, 'model')} whose license isn't Apache-2.0 or MIT"
                     f"{ALL_LICENSES_HINT}.")
    open_models = [i for i in kept if not _is_gated(i)]
    if not open_models and kept:
        notes.append("Only models that need a Hugging Face login turned up, so those are listed.")
    candidates = _pick_candidates(open_models or kept, max_candidates)

    # 3. Measure: real file sizes for each candidate (in parallel, within the time budget).
    #    For Mixture-of-Experts models whose active size we can't tell from the
    #    name, also read the start of one GGUF file for its expert settings.
    header_fetcher = _real_header_fetcher(api)
    listings = _run_parallel([lambda i=i: _measure(api, i, header_fetcher) for i in candidates], deadline, clock)
    entries: list[ModelEntry] = []
    skipped = errors = 0
    for info, measured in zip(candidates, listings):
        if measured is _SKIPPED:
            skipped += 1
        elif isinstance(measured, BaseException):
            errors += 1
        else:
            files, header = measured
            entry = entry_from_hub(info, files, header)
            if entry is not None:
                entries.append(entry)
    if skipped:
        notes.append(f"Hugging Face was slow, so I skipped checking {_plural(skipped, 'model')} to save time.")
    if errors:
        notes.append(f"Couldn't read the file list of {_plural(errors, 'model')}, so I left them out.")
    if not entries:
        if skipped:
            raise TimeoutError("Hugging Face took too long to answer")
        raise LookupError("no suitable models turned up")
    entries.sort(key=lambda e: (e.params_b, e.hf_repo.lower()))
    notes.insert(0, f"Found {_plural(len(entries), 'model')} on Hugging Face that suit the game.")
    complete = not (skipped or errors or any(not isinstance(r, list) for r in searches))
    return entries, notes, complete


def _measure(api: Any, info: Any, header_fetcher: Optional[Callable[[str, str], dict]]) -> tuple[list, Optional[dict]]:
    """(file list, GGUF header or None) for one candidate repo."""
    repo = str(_attr(info, "id"))
    files = repo_gguf_files(api, repo)
    header: Optional[dict] = None
    if header_fetcher is not None and _needs_header(info):
        groups = {q: v for q, v in group_quant_files(files).items() if catalog.quant_bits(q) and v[1] > 0}
        if groups:
            first_file = groups[_default_quant(groups)][0][0]
            try:
                header = header_fetcher(repo, first_file) or None
            except Exception:
                header = None  # nice to have: the model is still listed without it
    return files, header


def _needs_header(info: Any) -> bool:
    """Is the start of a GGUF file worth reading for this candidate?

    Yes for a possible Mixture-of-Experts model whose active size its name
    doesn't tell, and for a model whose attention shape (for the KV-cache
    maths) isn't in `catalog`'s table - both come from the header.
    """
    architecture = _gguf(info).get("architecture")
    repo = str(_attr(info, "id", default=""))
    if not catalog.known_kv_shape(repo, _base_model(info) or ""):
        return True
    if not _maybe_moe(architecture if isinstance(architecture, str) else None):
        return False
    _total, active = _params(info)
    return active is None


def _pick_candidates(infos: list[Any], max_candidates: int) -> list[Any]:
    """One repo per original model (curated repo first, then trusted-publisher order,
    then downloads), most popular models first, at most `max_candidates`."""
    curated = {m.hf_repo.lower() for m in catalog.MODEL_CATALOG}

    def preference(info: Any) -> tuple:
        repo = str(_attr(info, "id"))
        return (repo.lower() not in curated, _publisher_rank(repo), -int(_attr(info, "downloads", default=0) or 0), repo)

    groups: dict[str, list[Any]] = {}
    for info in sorted(infos, key=preference):
        groups.setdefault(_base_key(str(_attr(info, "id")), _base_model(info)), []).append(info)

    def popularity(group: list[Any]) -> tuple:
        is_curated = any(str(_attr(i, "id")).lower() in curated for i in group)
        return (not is_curated, -sum(int(_attr(i, "downloads", default=0) or 0) for i in group))

    return [group[0] for group in sorted(groups.values(), key=popularity)][: max(0, max_candidates)]


# ---------------------------------------------------------------------------
# The on-disk cache
# ---------------------------------------------------------------------------


def default_cache_path() -> Path:
    """`config.cache_dir()/hf_models.json` (honours GETTOWORK_HOME)."""
    return config.cache_dir() / CACHE_FILENAME


@dataclass
class _Cache:
    models: list[ModelEntry]
    fetched_at: float
    allow_all_licenses: bool
    complete: bool = True  # False: saved from a search that was cut short (trusted only briefly)
    rules_version: str = ""  # RULES_VERSION when it was saved ("" = before rules were recorded)


def _entry_to_json(entry: ModelEntry) -> dict:
    return dataclasses.asdict(entry)


def _entry_from_json(data: Any) -> Optional[ModelEntry]:
    """Rebuild a ModelEntry from cached JSON; None if it looks wrong.

    Every field is type-checked (see :func:`~gettowork.types.model_entry_from_json`),
    so a damaged cache entry is dropped instead of crashing the ranking.
    """
    entry = model_entry_from_json(data)
    if entry is None:
        return None
    ok = "/" in entry.hf_repo and entry.params_b > 0 and bool(entry.quant) and bool(entry.key)
    return entry if ok else None


def _load_cache(path: Path) -> Optional[_Cache]:
    """The saved list, or None if missing, corrupt or from another schema version."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("schema_version") != CACHE_SCHEMA_VERSION:
            return None
        fetched_at = float(data["fetched_at"])
        raw_models = data["models"]
        if not isinstance(raw_models, list) or not math.isfinite(fetched_at):
            return None
    except (OSError, ValueError, KeyError, TypeError):
        return None
    models = [m for m in (_entry_from_json(item) for item in raw_models) if m is not None]
    if not models:
        return None
    return _Cache(models, fetched_at, bool(data.get("allow_all_licenses", False)), bool(data.get("complete", True)),
                  str(data.get("rules_version") or ""))


def _save_cache(path: Path, models: list[ModelEntry], fetched_at: float, allow_all_licenses: bool,
                complete: bool = True) -> Optional[str]:
    """Write the cache atomically. Returns a note if that failed (never raises)."""
    payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "fetched_at": fetched_at,
        "allow_all_licenses": allow_all_licenses,
        "complete": complete,
        "rules_version": RULES_VERSION,
        "models": [_entry_to_json(m) for m in models],
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        return f"Couldn't save the model list for next time ({exc.strerror or exc})."
    return None


def _describe_age(seconds: float) -> str:
    if seconds < 90:
        return "just now"
    if seconds < 90 * 60:
        return f"{seconds / 60:.0f} minutes ago"
    if seconds < 36 * 3600:
        return f"{seconds / 3600:.0f} hours ago"
    return f"{seconds / 86400:.0f} days ago"


# ---------------------------------------------------------------------------
# The main entry point
# ---------------------------------------------------------------------------


def _with_seeds(models: list[ModelEntry]) -> list[ModelEntry]:
    """Add curated seeds for original models the search didn't turn up (a safety net)."""
    present = {_base_key(m.hf_repo, m.base_model) for m in models}
    extra = [s for s in catalog.MODEL_CATALOG if _base_key(s.hf_repo, s.base_model) not in present]
    return list(models) + extra


def _license_ok(entry: ModelEntry, allow_all_licenses: bool) -> bool:
    return allow_all_licenses or catalog.is_permissive(entry.license)


def _why_offline(exc: BaseException) -> str:
    """A short, friendly reason the Hub couldn't be reached."""
    name = type(exc).__name__
    text = str(exc).lower()
    status = getattr(getattr(exc, "response", None), "status_code", None)
    if name == "OfflineModeIsEnabled":
        return "offline mode is switched on (HF_HUB_OFFLINE)"
    if isinstance(exc, TimeoutError) or "timeout" in name.lower() or "timed out" in text:
        return "it took too long to answer"
    if isinstance(status, int):
        return f"it answered with an error, HTTP {status}"
    if isinstance(exc, LookupError) and not isinstance(exc, KeyError):
        return "no suitable models turned up"
    if isinstance(exc, (ConnectionError, OSError)) or "connect" in name.lower() or "connect" in text:
        return "no internet connection?"
    return f"{name}: {exc}"[:120]


def discover_models(
    *,
    api: Any = None,
    cache_path: Optional[Path] = None,
    ttl_hours: float = 72,
    refresh: bool = False,
    offline: bool = False,
    allow_all_licenses: bool = False,
    max_candidates: int = 40,
    timeout_s: float = 20.0,
    clock: Callable[[], float] = time.time,
    offline_reason: Optional[str] = None,
) -> DiscoveryResult:
    """Find GGUF chat models that suit the game. Never raises for network trouble.

    Order of preference:

    1. A fresh cache (younger than `ttl_hours`), unless `refresh`.
    2. A live search of the Hub (skipped when `offline`), saved to the cache.
    3. The saved cache even if it's old ("stale"), with a note.
    4. The curated seeds from `catalog.py` (`source="curated"`), with a note.

    Live and cached results also include curated seeds for any original model
    the search missed. `api` defaults to `huggingface_hub.HfApi()`; tests pass
    a fake. `timeout_s` bounds the whole live search: slow lookups are skipped.
    `offline_reason` words the note when `offline` is set for another reason
    than being offline (e.g. "Pretend-model mode doesn't go online").
    """
    path = cache_path or default_cache_path()
    now = clock()
    cached = _load_cache(path)

    def from_cache(note: str, *, stale: bool) -> DiscoveryResult:
        assert cached is not None
        models = cached.models
        if cached.rules_version != RULES_VERSION:
            # Saved under older screening rules: apply today's rules to what we can.
            models = [e for e in (rescreen_entry(m, allow_all_licenses=allow_all_licenses) for m in models) if e]
        models = [m for m in models if _license_ok(m, allow_all_licenses)]
        return DiscoveryResult(_with_seeds(models), "cache", cached.fetched_at, [note], stale=stale)

    if cached is not None:
        age = now - cached.fetched_at
        # A list saved by a search that was cut short is only trusted for a
        # little while; a list saved in the other license mode was picked from
        # a different pool of models; and one saved under older screening
        # rules may keep models today's rules leave out: none is "fresh".
        ttl = ttl_hours if cached.complete else min(ttl_hours, PARTIAL_CACHE_TTL_HOURS)
        fresh = (0 <= age < ttl * 3600 and cached.allow_all_licenses == allow_all_licenses
                 and cached.rules_version == RULES_VERSION)
        if fresh and not refresh:
            return from_cache(f"Using the model list saved {_describe_age(age)} (it refreshes every few days).",
                              stale=False)

    if offline:
        reason = offline_reason or "You're offline"
        if cached is not None:
            age = _describe_age(now - cached.fetched_at)
            return from_cache(f"{reason}, so I'm using the model list saved {age}.", stale=True)
        return DiscoveryResult(list(catalog.MODEL_CATALOG), "curated", None,
                               [f"{reason} and there's no saved model list yet, so here are the built-in picks."])

    try:
        if api is None:
            api = make_hub_api()
        models, notes, complete = _fetch_live(api, allow_all_licenses=allow_all_licenses,
                                              max_candidates=max_candidates, timeout_s=timeout_s, clock=clock)
    except Exception as exc:  # network trouble of any kind: fall back, never crash the game
        why = _why_offline(exc)
        if cached is not None:
            age = _describe_age(now - cached.fetched_at)
            return from_cache(f"I couldn't reach Hugging Face ({why}), so I'm using the model list saved {age}.",
                              stale=True)
        return DiscoveryResult(list(catalog.MODEL_CATALOG), "curated", None,
                               [f"I couldn't reach Hugging Face ({why}), so here are the built-in picks - "
                                "all tried and tested."])

    fetched_at = clock()
    if not complete and cached is not None:
        # Part of the search failed: keep what we knew before, updated with what we just learned.
        fresh_repos = {m.hf_repo.lower() for m in models}
        older = cached.models
        if cached.rules_version != RULES_VERSION:
            older = [e for e in (rescreen_entry(m, allow_all_licenses=allow_all_licenses) for m in older) if e]
        models = models + [m for m in older
                           if m.hf_repo.lower() not in fresh_repos and _license_ok(m, allow_all_licenses)]
        models.sort(key=lambda e: (e.params_b, e.hf_repo.lower()))
    problem = _save_cache(path, models, fetched_at, allow_all_licenses, complete)
    if problem:
        notes.append(problem)
    return DiscoveryResult(_with_seeds(models), "live", fetched_at, notes)


DISCOVERY_EXPLAINER = """\
**How the game finds models on Hugging Face**

[Hugging Face](https://huggingface.co) is a huge public library of AI models.
Many are shared as **GGUF** files: one self-contained file (or a few "shards"
for big models) holding the model's weights, ready for the llama.cpp engine.

Here's what the game does, in about one second of network time:

1. **Search** the Hub's free public API for the most-downloaded GGUF chat
   models, from well-known publishers (unsloth, bartowski, ggml-org...) and
   from everyone.
2. **Read the labels.** Each GGUF file's header says how many parameters the
   model has, its architecture and how much text it can remember (context).
   We also read the *real* size of every quantization (Q4_K_M, Q8_0, ...).
3. **Filter.** We keep instruction-tuned chat models and skip coding,
   embedding, vision and speech specialists, base models, and anything
   marked uncensored or not family-friendly. By default we only show
   **Apache-2.0** and **MIT** models: licenses that let you use, share and
   modify them freely.
4. **Remember.** The list is saved on your computer for 3 days, so the next
   launch is instant and works offline.

These filters are our own home-grown guesses: no warranty or guarantee.
Always check a model's card on Hugging Face - that's where its authors
explain what it's for and how it may be used.
"""
