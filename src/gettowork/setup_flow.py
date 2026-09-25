"""From "what computer is this?" to "your model is ready!" - the setup flow.

The player makes exactly **one** real decision here - which model to play
with (Enter picks the recommended one) - plus one Y/n to approve the
downloads. Everything else is automatic and explained as it happens:

1. **Welcome back.** Returning players whose model and engine are still on
   disk get a single "Play with Qwen3 4B again? [Y/n]" and go straight in.
2. **Hardware check** (``specs.py`` + ``perf.py``): a warm one-line summary,
   a compact table and a short "Learn" panel on why memory size *and* speed
   matter.
3. **Live model discovery** (``hf_discovery.py``) and the **fit engine**
   (``catalog.py``): real models from Hugging Face, ranked for *this*
   computer, shown as a short numbered menu with friendly badges.
4. **One confirmation screen** listing exactly what will be downloaded, from
   where, under which license, and where it will be stored.
5. **Automatic install + start** with fallbacks: the managed llama.cpp
   engine -> Ollama (if it's already running) -> retry / pick another /
   pretend model / quit.
6. **Warm-up and speed test.** The real tokens/second is measured and fed
   back into the fit engine ("calibration"), so estimates get better the
   more you play. Painfully slow? You're offered a faster pick.
7. **Remember** the choice, so next time is one keypress.

Everything that touches the outside world (hardware detection, Hugging
Face, the engines) is reached through :class:`SetupServices`, so the tests
can swap in fakes and run without a network or a real model.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Union

from rich import box
from rich.markup import escape
from rich.table import Table

from . import catalog, config, download, hf_discovery, perf, runtime_install
from . import specs as specs_module
from .backends import ollama as ollama_backend
from .backends.base import BackendError, EngineStopped, LLMBackend
from .config import Settings
from .types import FitResult, ModelEntry, SystemSpecs, model_entry_from_json
from .ui import UI

__all__ = [
    "SetupResult",
    "SetupServices",
    "ModelSearch",
    "run_setup",
    "default_backend_factory",
    "find_models",
    "show_hardware",
    "model_table",
    "show_model_table",
    "discovery_summary",
    "entry_for_fit",
    "fit_for_quant",
    "calibrate_specs",
    "apply_saved_calibration",
    "apply_engine_limits",
    "hardware_fingerprint",
    "engine_key",
    "planned_engine_key",
    "entry_to_dict",
    "entry_from_dict",
    "format_size",
    "format_speed",
    "why_line",
    "symbols_for",
    "SLOW_TOKENS_PER_S",
    "QUICK_EXPLAINER",
]

# ---------------------------------------------------------------------------
# Tunables and friendly text
# ---------------------------------------------------------------------------

SLOW_TOKENS_PER_S = 3.0  # below this, a story turn takes minutes: offer a faster model
SHORTLIST_SIZE = 6  # how many picks the menu shows
FULL_LIST_LIMIT = 40  # the "more" list stops here (the rest are usually far too big)
WHY_MAX_FACTS = 3  # the menu's "Why" column stays short; `why N` shows the full working
WIDE_TABLE_MIN_COLUMNS = 140  # narrower terminals get the compact menu (no "Why" column)
CALIBRATION_KEY = "speed_calibration"  # where measured-speed corrections live in Settings.extra
CALIBRATION_RANGE = (0.25, 2.0)  # never trust one measurement more than this much...
CPU_CALIBRATION_MAX = 4.0  # ...except upwards on the processor, where a quick test can under-read a lot
# Only learn from a model whose speed is mostly set by memory bandwidth: for a
# tiny or Mixture-of-Experts model the fixed per-token cost (and our guess of
# its active share) dominates, and blaming bandwidth for that error would skew
# every bigger model's estimate by 30-50%.
CALIBRATION_MIN_BANDWIDTH_SHARE = 0.7
BACKEND_KINDS = ("managed", "ollama", "llamacpp")

OLLAMA_DOWNLOAD_URL = "https://ollama.com/download"
LLAMACPP_PIP_HINT = "pip install llama-cpp-python"

# Rough engine download sizes by build, for the confirmation screen. The real,
# exact size is printed by runtime_install.py when the download starts.
ENGINE_SIZE_HINTS = {
    "cpu": "roughly 20-50 MB",
    "metal": "roughly 20-50 MB",
    "vulkan": "roughly 30-60 MB",
    # Measured on real releases: 0.57-0.76 GB, most of it NVIDIA's CUDA runtime.
    "cuda-12": "roughly 0.6-0.8 GB (it includes NVIDIA's CUDA runtime)",
    "cuda-13": "roughly 0.6-0.8 GB (it includes NVIDIA's CUDA runtime)",
    "rocm": "a few hundred MB",
}

QUICK_EXPLAINER = """\
Two things decide which AI models your computer can run:

- **Memory size** - the whole model has to fit in memory (graphics-card
  memory is best, then regular RAM).
- **Memory speed** - to write each word, the model reads essentially *all*
  of itself, so faster memory means a faster storyteller.

I'll suggest models that fit comfortably *and* talk quickly. Curious about
the maths? Type `learn` at the model menu.
"""

VERDICT_LOOK = {  # verdict -> (colour, words)
    "great": ("green", "great fit"),
    "ok": ("cyan", "good fit"),
    "tight": ("yellow", "snug fit"),
    "no": ("red", "won't fit"),
}
SPEED_COLOURS = {"fast": "green", "usable": "cyan", "slow": "yellow", "very slow": "red", "n/a": "dim"}
PLACEMENT_WORDS = {
    "gpu": "Runs on your graphics card",
    "unified": "Runs in your Mac's shared memory",
    "partial": "Splits between graphics card and RAM",
    "cpu": "Runs on your processor",
    "none": "Too big for this computer",
}


# ---------------------------------------------------------------------------
# Symbols: pretty where the terminal can show them, plain ASCII elsewhere
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Symbols:
    """The little icons used in status lines and the model menu."""

    tick: str
    star: str
    bolt: str
    brain: str
    warn: str
    arrow: str


FANCY_SYMBOLS = Symbols(tick="✓", star="★", bolt="⚡", brain="🧠", warn="⚠", arrow="→")
PLAIN_SYMBOLS = Symbols(tick="OK", star="*", bolt="!", brain="+", warn="!", arrow="->")


def symbols_for(ui: UI) -> Symbols:
    """Fancy symbols when the console speaks UTF-8, ASCII otherwise.

    Some Windows terminals (or output redirected to a file) use an old code
    page that can't encode emoji; printing one there would crash.
    """
    encoding = (getattr(ui.console, "encoding", "") or "").lower().replace("-", "").replace("_", "")
    return FANCY_SYMBOLS if encoding.startswith("utf") else PLAIN_SYMBOLS


BADGE_FOOTNOTE = ("Badges judge whole story turns (reading your prompt as well as writing the answer) and skip "
                  "snug fits, so they can differ from the Speed column.")


def _badge_labels(sym: Symbols) -> dict[str, str]:
    return {
        "recommended": f"{sym.star} Recommended",
        # Named for what they really mean: both judge whole story turns (reading
        # the prompt too, not just the Speed column's writing speed) and leave
        # out snug fits - see catalog.pick_shortlist and BADGE_FOOTNOTE.
        "fastest": f"{sym.bolt} Fastest comfortable fit",
        "smartest": f"{sym.brain} Smartest at a playable pace",
    }


# ---------------------------------------------------------------------------
# Results and injectable services
# ---------------------------------------------------------------------------


@dataclass
class SetupResult:
    """What the rest of the game needs once setup is done."""

    backend: LLMBackend  # prepared and warmed up, ready to chat
    entry: Optional[ModelEntry]  # the chosen model (None for --gguf / --ollama-model / mock)
    specs: SystemSpecs
    fit: Optional[FitResult]  # how the fit engine rated the choice (None if unknown)
    tokens_per_s: Optional[float] = None  # the speed measured during warm-up (None if unknown)


def _model_context(entry: Optional[ModelEntry]) -> int:
    """The context window to request: the entry's wish, capped at what the model was trained for."""
    if entry is None:
        return 4096
    wanted = entry.context_tokens or 4096
    return min(wanted, entry.native_context) if entry.native_context else wanted


def default_backend_factory(
    kind: str,
    *,
    entry: Optional[ModelEntry] = None,
    specs: Optional[SystemSpecs] = None,
    quant: Optional[str] = None,
    model_path: Optional[Path] = None,
    server_exe: Optional[Path] = None,
    ollama_model: Optional[str] = None,
    gguf_path: Optional[Path] = None,
) -> LLMBackend:
    """Build (but don't start) a backend: ``kind`` is managed | ollama | llamacpp | mock.

    For Ollama, `gguf_path` is a model file the game already downloaded: it is
    handed to Ollama instead of Ollama downloading the same model again.

    Creating a backend is cheap and has no side effects; nothing is downloaded
    or started until its ``prepare()`` is called.
    """
    # Imported here so that only the engine actually used gets loaded.
    from . import backends

    if kind == "managed":
        return backends.LlamaServerBackend(
            entry, specs=specs, model_path=model_path, quant=quant, server_exe=server_exe, n_ctx=_model_context(entry)
        )
    if kind == "ollama":
        model = ollama_model or (entry.ollama_ref if entry is not None else "")
        return backends.OllamaBackend(model, n_ctx=_model_context(entry) if entry is not None else None,
                                      gguf_path=gguf_path)
    if kind == "llamacpp":
        return backends.LlamaCppBackend(model_path=model_path, entry=entry, quant=quant, n_ctx=_model_context(entry))
    if kind == "mock":
        return backends.MockBackend(seed=random.randrange(1_000_000))  # a different silly story each game
    raise ValueError(f"unknown backend kind: {kind!r}")


@dataclass
class SetupServices:
    """Everything the setup flow uses to reach the outside world.

    The defaults are the real thing; tests pass fakes (no network, no GPU,
    no downloads). This pattern is called *dependency injection*.
    """

    detect_specs: Callable[..., SystemSpecs] = specs_module.detect_specs
    discover_models: Callable[..., hf_discovery.DiscoveryResult] = hf_discovery.discover_models
    make_backend: Callable[..., LLMBackend] = default_backend_factory
    installed_runtimes: Callable[[], list] = runtime_install.installed_runtimes
    custom_entry: Callable[..., ModelEntry] = download.custom_entry


@dataclass
class ModelSearch:
    """Discovery results plus the fit engine's verdicts for this computer."""

    discovery: hf_discovery.DiscoveryResult
    ranked: list[FitResult]  # every candidate, best first (models that won't fit at the end)
    shortlist: list[FitResult]  # the short, diverse menu with badges

    @property
    def recommended(self) -> Optional[FitResult]:
        return next((f for f in self.shortlist if "recommended" in f.badges), None)


# ---------------------------------------------------------------------------
# Small pure helpers (easy to test, handy to read)
# ---------------------------------------------------------------------------


def format_size(gb: Optional[float]) -> str:
    """0.64 -> "640 MB", 2.5 -> "2.5 GB", 18.6 -> "19 GB"."""
    if gb is None or gb <= 0 or not math.isfinite(gb):
        return "?"
    if gb < 1:
        return f"{gb * 1000:.0f} MB"
    return f"{gb:.1f} GB" if gb < 10 else f"{gb:.0f} GB"


def format_speed(tokens_per_s: Optional[float], *, short: bool = False) -> str:
    """25.3 -> "~25 tokens/s" (or "~25 tok/s" with `short`); tiny or unknown speeds read naturally too."""
    unit = "tok/s" if short else "tokens/s"
    if tokens_per_s is None or tokens_per_s <= 0:
        return "n/a"
    if tokens_per_s < 1:
        return f"<1 {'tok/s' if short else 'token/s'}"
    return f"~{tokens_per_s:.0f} {unit}"


def _compact_count(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}K"
    return str(n)


def why_line(fit: FitResult) -> str:
    """One short line for the menu's "Why" column (``why N`` shows the full working)."""
    if fit.verdict == "no":
        return fit.reason
    model = fit.model
    bits = [PLACEMENT_WORDS.get(fit.placement, "Runs on this computer")]
    if fit.verdict == "tight":
        bits.append("snug fit, so close big apps first")
    if model.active_params_b:
        bits.append("Mixture-of-Experts: quick for its size")
    mode = catalog.thinking_mode(model)
    if mode == "always":
        bits.insert(0, "always thinks first, so turns are slow")  # the most important fact about it
    elif mode == "switchable" and (fit.est_tokens_per_s or 0.0) >= catalog.THINKING_MIN_TOKENS_PER_S:
        bits.append("shows its thinking")  # slower than that, the game asks it to skip the thinking
    if model.gated:
        bits.append("needs a Hugging Face login")
    if model.downloads >= 100_000:
        bits.append(f"popular ({_compact_count(model.downloads)} downloads)")
    elif model.source == "curated":
        bits.append("hand-checked")
    return "; ".join(bits[:WHY_MAX_FACTS])


def entry_for_fit(fit: FitResult) -> ModelEntry:
    """The fit's model, adjusted to the quantization the fit engine chose.

    The fit engine may pick e.g. Q8_0 on a big GPU while the repo's default is
    Q4_K_M. Returning an entry whose ``quant``, size and Ollama reference all
    match the choice keeps every later step (download, Ollama, labels) consistent.
    """
    entry = fit.model
    if fit.context_tokens and fit.context_tokens < (entry.context_tokens or fit.context_tokens):
        # The fit engine had to plan with a shorter conversation memory: run it that way too.
        entry = dataclasses.replace(entry, context_tokens=fit.context_tokens)
    quant = fit.quant or entry.quant
    if not quant or quant.upper() == (entry.quant or "").upper():
        return entry
    ref = entry.ollama_ref
    if ref.startswith("hf.co/"):
        ref = f"hf.co/{entry.hf_repo}:{quant}"
    return dataclasses.replace(
        entry,
        quant=quant,
        file_size_gb=fit.download_gb or entry.file_size_gb,
        gguf_files=(),  # the exact file names for this quant are looked up at download time
        ollama_ref=ref,
    )


def downloaded_checker() -> Callable[[ModelEntry, str], bool]:
    """``(model, quant) -> already downloaded?`` for the fit engine (looks at each model's folder once)."""
    seen: dict[str, set[str]] = {}

    def check(model: ModelEntry, quant: str) -> bool:
        if model.hf_repo not in seen:
            try:
                seen[model.hf_repo] = download.downloaded_quants(model.hf_repo)
            except Exception:
                seen[model.hf_repo] = set()
        return (quant or "").upper() in seen[model.hf_repo]

    return check


def fit_for_quant(specs: SystemSpecs, entry: ModelEntry, quant: str) -> FitResult:
    """Evaluate `entry` at one specific quantization (for ``--quant`` or a saved choice)."""
    quant = quant.strip()
    sizes = {q.upper(): size for q, size in entry.quant_options}
    size = sizes.get(quant.upper())
    same_as_default = quant.upper() == (entry.quant or "").upper()
    if size is None and same_as_default and entry.file_size_gb > 0:
        size = entry.file_size_gb
    if not size:
        size = catalog.estimate_quant_size_gb(entry.params_b, quant) or entry.file_size_gb
    ref = entry.ollama_ref
    if ref.startswith("hf.co/") and not same_as_default:
        ref = f"hf.co/{entry.hf_repo}:{quant}"
    pinned = dataclasses.replace(
        entry,
        quant=quant,
        file_size_gb=size,
        quant_options=((quant, size),),
        gguf_files=entry.gguf_files if same_as_default else (),
        ollama_ref=ref,
    )
    # The player asked for this exact quant: judge it, even one we'd never suggest ourselves.
    return catalog.evaluate_fit(specs, pinned, downloaded=downloaded_checker(), quant_floor=False)


def hardware_fingerprint(specs: SystemSpecs) -> str:
    """A short id for "this computer", so a saved speed calibration is only reused on it."""
    gpus = ",".join(sorted(g.name for g in specs.gpus))
    return f"{specs.os_name}|{specs.arch}|{specs.cpu_name}|{round(specs.ram_total_gb)}|{gpus}"


def engine_key(kind: Optional[str], backend: Any = None) -> Optional[str]:
    """Which engine produced a measurement: "managed:cuda-12", "managed:cpu", "ollama"...

    A speed correction learned with one engine build (say CUDA) isn't
    applied when another (Vulkan, CPU mode, Ollama) will run the model.
    """
    if not kind:
        return None
    if kind == "llamacpp-server":  # the managed backend's class name, not its setup kind
        kind = "managed"
    if kind != "managed":
        return kind
    variant = getattr(getattr(backend, "variant", None), "name", None)
    if getattr(backend, "cpu_only", False):
        variant = "cpu-mode" if variant and variant != "cpu" else "cpu"
    return f"managed:{variant}" if variant else "managed"


def planned_engine_key(specs: SystemSpecs, settings: Optional[Settings] = None, *,
                       server_exe: Optional[Path] = None) -> Optional[str]:
    """The engine the next game will most likely use (for applying saved corrections).

    With `server_exe` (a returning player's saved engine) that exact build is
    what will run, so its own variant counts - not the first build on the
    plan, which may differ (e.g. a CUDA 13 build that fell back to CUDA 12).
    """
    kind = getattr(settings, "backend", None) if settings is not None else None
    if kind and kind not in ("managed", "auto"):
        return str(kind)
    if server_exe is not None:
        try:
            variant = (runtime_install.install_info(Path(server_exe)) or {}).get("variant")
        except Exception:
            variant = None
        if isinstance(variant, str) and variant:
            return f"managed:{variant}"
    try:
        plan = runtime_install.usable_plan(specs)
    except Exception:
        return None
    return f"managed:{plan[0].name}" if plan else None


def _calibration_range(placement: str) -> tuple[float, float]:
    """How far one placement's speed correction may go.

    Processor speed is estimated from a quick read test, which can still land
    under what a many-channel desktop or server delivers to llama.cpp, so that
    correction may grow further (up to CPU_CALIBRATION_MAX).
    """
    low, high = CALIBRATION_RANGE
    return (low, CPU_CALIBRATION_MAX) if placement == "cpu" else (low, high)


def calibrate_specs(specs: SystemSpecs, placement: str, factor: float) -> SystemSpecs:
    """Scale the memory bandwidth behind `placement` by `factor`; returns a new SystemSpecs.

    Why: the fit engine predicts speed from memory bandwidth (see perf.py).
    Once we've *measured* a real model, ``measured / estimated`` tells us how
    far off that prediction was on this computer. Scaling the bandwidth by
    the same factor makes every other estimate more realistic too. It's
    approximate (speed isn't perfectly proportional to bandwidth) and clamped,
    so one odd measurement can't wreck the rankings.
    """
    low, high = _calibration_range(placement)
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return specs
    if not math.isfinite(factor) or factor <= 0:
        return specs
    factor = min(max(factor, low), high)
    if placement == "cpu":
        bandwidth, _source = perf.bandwidth_for(specs, "cpu")
        return dataclasses.replace(specs, ram_bandwidth_gbs=round(bandwidth * factor, 1), notes=list(specs.notes))
    if placement in ("gpu", "unified"):
        gpu = perf.primary_gpu(specs)
        if gpu is None:
            return specs
        bandwidth, _source = perf.bandwidth_for(specs, placement)
        tuned = dataclasses.replace(gpu, bandwidth_gbs=round(bandwidth * factor, 1))
        gpus = [tuned if g is gpu else g for g in specs.gpus]
        return dataclasses.replace(specs, gpus=gpus, notes=list(specs.notes))
    return specs  # "partial" / "none": too tangled to calibrate honestly


def saved_calibration(specs: SystemSpecs, settings: Settings, *,
                      engine: Optional[str] = None) -> dict[str, float]:
    """The saved speed corrections that apply here, as ``{placement: factor}``.

    Corrections saved on different hardware are ignored, and so is one learned
    with a different engine build than `engine` (when both are known): a CUDA
    measurement says little about Vulkan. Factors are clamped like
    :func:`calibrate_specs` clamps them.
    """
    extra = settings.extra if isinstance(settings.extra, dict) else {}
    saved = extra.get(CALIBRATION_KEY)
    if not isinstance(saved, dict) or saved.get("fingerprint") != hardware_fingerprint(specs):
        return {}
    factors = saved.get("factors")
    engines = saved.get("engines") if isinstance(saved.get("engines"), dict) else {}
    usable: dict[str, float] = {}
    if isinstance(factors, dict):
        for placement, factor in factors.items():
            learned_with = engines.get(placement)
            if engine and learned_with and learned_with != engine:
                continue
            if isinstance(factor, bool) or not isinstance(factor, (int, float)):
                continue
            if not math.isfinite(factor) or factor <= 0:
                continue
            low, high = _calibration_range(str(placement))
            usable[str(placement)] = min(max(float(factor), low), high)
    return usable


def apply_saved_calibration(specs: SystemSpecs, settings: Settings, *,
                            engine: Optional[str] = None) -> tuple[SystemSpecs, bool]:
    """Apply speed corrections measured on this same computer in earlier games.

    Returns ``(specs, applied?)``; see :func:`saved_calibration` for which apply.
    """
    tuned, applied = _apply_factors(specs, saved_calibration(specs, settings, engine=engine))
    return tuned, bool(applied)


def _apply_factors(specs: SystemSpecs, factors: Mapping[str, float]) -> tuple[SystemSpecs, dict[str, float]]:
    """Apply ``{placement: factor}``; returns the new specs and the factors that changed them."""
    applied: dict[str, float] = {}
    for placement, factor in factors.items():
        tuned = calibrate_specs(specs, placement, factor)
        if tuned is not specs:
            applied[placement] = factor
        specs = tuned
    return specs, applied


def apply_engine_limits(specs: SystemSpecs, backend: str = "auto") -> SystemSpecs:
    """Plan with the CPU when the built-in engine can't use this computer's graphics card.

    With the built-in llama.cpp engine (``backend`` "auto" or "managed"),
    some GPUs have no usable build - e.g. an AMD card on Linux without the
    Vulkan loader. The fit engine must not promise graphics-card speed there,
    so ``gpu_offload`` is switched off and a note explains how to fix it.
    (Ollama brings its own GPU support, so ``--backend ollama`` is left alone.)
    """
    if backend not in ("auto", "managed") or specs.gpu_offload is not None:
        return specs
    if not any(g.vendor != "apple" and g.vram_gb > 0 for g in specs.gpus):
        return specs
    if runtime_install.engine_can_use_gpu(specs):
        return specs
    note = "The built-in engine can't use your graphics card on this computer, so I'm planning with the CPU."
    if specs.os_name == "Linux":
        note += " (Installing the Vulkan loader - e.g. 'sudo apt install libvulkan1' - lets it use the card.)"
    return dataclasses.replace(specs, gpu_offload=False, notes=list(specs.notes) + [note])


def entry_to_dict(entry: ModelEntry) -> dict:
    """A JSON-safe copy of a ModelEntry (for the settings file)."""
    data = dataclasses.asdict(entry)
    data["quant_options"] = [[q, s] for q, s in entry.quant_options]
    data["gguf_files"] = list(entry.gguf_files)
    return data


# Saved ModelEntry fields that must be non-empty text / positive numbers for the entry to be usable.
_ENTRY_TEXT_FIELDS = ("key", "display_name", "hf_repo", "quant")
_ENTRY_NUMBER_FIELDS = ("params_b", "file_size_gb")


def entry_from_dict(data: Any) -> Optional[ModelEntry]:
    """Rebuild a ModelEntry saved by :func:`entry_to_dict` (None if it's unusable).

    The settings file can be hand-edited or damaged, so every field is
    checked against its type (:func:`~gettowork.types.model_entry_from_json`),
    and the fields the game relies on must be filled in: a missing name,
    ``"quant": null`` or ``"native_context": "lots"`` means the saved choice is
    ignored (and the player picks again) rather than crashing later.
    """
    if not isinstance(data, dict):
        return None
    if not all(isinstance(data.get(k), str) and data[k].strip() for k in _ENTRY_TEXT_FIELDS):
        return None
    entry = model_entry_from_json(data)
    if entry is None or not all(getattr(entry, k) > 0 for k in _ENTRY_NUMBER_FIELDS):
        return None
    return entry


def _with_measured_speeds(ranked: list[FitResult], measured: Mapping[tuple[str, str], float]) -> list[FitResult]:
    """`ranked` with each model the player already tried showing its *measured* speed.

    Its score is adjusted by the same amount the speed change is worth, and
    the list is re-sorted, so a model just measured as painfully slow isn't
    recommended again on the strength of a hopeful estimate.
    """
    if not measured:
        return ranked
    out: list[FitResult] = []
    for fit in ranked:
        real = measured.get((fit.model.key, (fit.quant or fit.model.quant or "").upper()))
        if real is None or fit.verdict == "no" or not fit.est_tokens_per_s:
            out.append(fit)
            continue

        def score(tps: float, fit: FitResult = fit) -> float:
            return catalog.score_fit(fit.model, quant=fit.quant, verdict=fit.verdict, tokens_per_s=tps,
                                     ratio=0.5, download_gb=fit.download_gb, placement=fit.placement)

        out.append(dataclasses.replace(
            fit, est_tokens_per_s=round(real, 1), est_speed=perf.speed_label(real),
            score=round(fit.score + score(real) - score(fit.est_tokens_per_s), 2), badges=()))
    runnable = sorted((f for f in out if f.verdict != "no"), key=lambda f: f.score, reverse=True)
    return runnable + [f for f in out if f.verdict == "no"]


def _absolute(path: Path) -> Path:
    """`path` with ``~`` expanded and made absolute (saved paths must work from any folder)."""
    path = Path(path).expanduser()
    try:
        return path.resolve()
    except (OSError, RuntimeError):
        return path.absolute()


def _ollama_import_name(entry: ModelEntry) -> str:
    """A tidy local Ollama name for a model file we hand over, e.g. "gettowork-qwen3-4b:q4_k_m"."""
    base = re.sub(r"[^a-z0-9._-]+", "-", (entry.key or entry.hf_repo).lower()).strip("-.") or "model"
    tag = re.sub(r"[^a-z0-9._-]+", "-", (entry.quant or "latest").lower()).strip("-.") or "latest"
    return f"gettowork-{base}:{tag}"


def discovery_summary(result: hf_discovery.DiscoveryResult, ranked: list[FitResult]) -> str:
    """One status line, e.g. "Found 34 models on Hugging Face - 6 of them fit your computer nicely"."""
    count = len(result.models)
    # Live and cached lists also carry built-in seeds for models the search missed:
    # count those separately so "Found N on Hugging Face" is literally true.
    from_hub = sum(1 for m in result.models if m.source != "curated")
    if result.source in ("live", "cache") and 0 < from_hub < count:
        seeds = count - from_hub
        count = from_hub
        extra = f" (plus {seeds} of my hand-checked favourite{'s' if seeds != 1 else ''})"
    else:
        extra = ""
    plural = "model" if count == 1 else "models"
    if result.source == "live":
        where = f"Found {count} {plural} on Hugging Face{extra}"
    elif result.source == "cache":
        where = f"Loaded {count} {plural} from my saved Hugging Face list{extra}"
    else:
        where = f"Using my {count} built-in, hand-checked {plural}"
    playable = [f for f in ranked if f.verdict != "no" and (f.est_tokens_per_s or 0.0) >= SLOW_TOKENS_PER_S]
    comfy = [f for f in playable if f.verdict in ("great", "ok")]
    if comfy:
        return f"{where} - {len(comfy)} of them fit your computer nicely"
    if playable:
        return f"{where} - a few will squeeze in, though none fit comfortably"
    return f"{where} - sadly, none of them run well on this computer"


# ---------------------------------------------------------------------------
# Reusable screens (also used by `gettowork --specs` / `--list-models`)
# ---------------------------------------------------------------------------


def show_hardware(ui: UI, specs: SystemSpecs, *, teach: bool = True) -> None:
    """The friendly hardware summary, a compact details table and (optionally) a short lesson."""
    sym = symbols_for(ui)
    ui.heading("Your computer")
    ui.say(f"[green]{sym.tick}[/green] {escape(specs_module.friendly_summary(specs))}")
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in specs_module.describe_specs(specs):
        table.add_row(escape(label), escape(value))
    ui.console.print(table)
    if teach:
        ui.teach("memory size and speed, in 20 seconds", QUICK_EXPLAINER)


_FOUND_NOTE_RE = re.compile(r"^Found \d+ models? on Hugging Face")


def find_models(
    ui: UI,
    specs: SystemSpecs,
    services: Optional[SetupServices] = None,
    *,
    refresh: bool = False,
    offline: bool = False,
    allow_all_licenses: bool = False,
    offline_reason: Optional[str] = None,
) -> ModelSearch:
    """Discover models (live, cached or curated), rank them for `specs`, report in one line.

    `offline_reason` explains *why* we're not searching when it isn't that
    the player is offline (e.g. pretend-model mode).
    """
    svc = services or SetupServices()
    sym = symbols_for(ui)
    message = (
        "Looking through my saved model list..."
        if offline
        else "Searching Hugging Face for models that fit your computer..."
    )
    extra: dict[str, Any] = {"offline_reason": offline_reason} if offline_reason else {}
    with ui.status(message):
        result = svc.discover_models(refresh=refresh, offline=offline, allow_all_licenses=allow_all_licenses, **extra)
    models = list(result.models) or list(catalog.MODEL_CATALOG)
    ranked = catalog.rank_models(specs, models, downloaded=downloaded_checker())
    shortlist = catalog.pick_shortlist(ranked, SHORTLIST_SIZE)
    ui.say(f"[green]{sym.tick}[/green] {escape(discovery_summary(result, ranked))}")
    for note in result.notes:
        if _FOUND_NOTE_RE.match(note):
            continue  # the summary line above already says how many were found
        ui.say(f"  [dim]{escape(note)}[/dim]")
    return ModelSearch(discovery=result, ranked=ranked, shortlist=shortlist)


def _speed_cell(fit: FitResult, *, short: bool = False) -> str:
    colour = SPEED_COLOURS.get(fit.est_speed, "white")
    return f"[{colour}]{format_speed(fit.est_tokens_per_s, short=short)}[/{colour}]"


def _verdict_cell(fit: FitResult) -> str:
    colour, words = VERDICT_LOOK.get(fit.verdict, ("white", fit.verdict))
    return f"[{colour}]{words}[/{colour}]"


def _license_cell(entry: ModelEntry, sym: Symbols) -> str:
    if catalog.is_permissive(entry.license):
        return escape(entry.license)
    return f"[yellow]{sym.warn} {escape(entry.license or 'unknown')}[/yellow]"


def model_table(
    fits: list[FitResult],
    sym: Symbols,
    *,
    title: Optional[str] = None,
    start: int = 1,
    show_repo: bool = False,
    compact: bool = False,
) -> Table:
    """The numbered model menu as a rich Table (colour-coded speeds and verdicts).

    `compact` suits narrow terminals: badges move under the model name, the
    "Why" column is dropped (``why N`` still explains any pick) and speeds
    are abbreviated, so nothing important gets squeezed out.
    """
    labels = _badge_labels(sym)
    table = Table(title=title, box=box.ROUNDED, header_style="bold", show_lines=compact)
    table.add_column("#", justify="right", style="bold cyan", no_wrap=True)
    if not compact:
        table.add_column("Pick", no_wrap=True)
    table.add_column("Model", overflow="fold")
    table.add_column("Size" if compact else "Download", justify="right", no_wrap=True)
    # In compact mode these may wrap, so the model name keeps enough room on small screens.
    table.add_column("Speed" if compact else "Speed (est.)", no_wrap=not compact)
    table.add_column("Fit", no_wrap=not compact)
    table.add_column("License", no_wrap=not compact, overflow="fold")
    if not compact:
        table.add_column("Why", overflow="fold")
    for number, fit in enumerate(fits, start):
        entry = fit.model
        badges = [labels.get(b, b) for b in fit.badges]
        name = f"{escape(entry.display_name)} [dim]{escape(fit.quant or entry.quant)}[/dim]"
        if compact and badges:
            name += "\n[bold]" + escape(" · ".join(badges)) + "[/bold]"
        if show_repo:
            name += f"\n[dim]{escape(entry.hf_repo)}[/dim]"
        row = [str(number)]
        if not compact:
            row.append("\n".join(badges))
        row += [
            name,
            format_size(fit.download_gb or entry.file_size_gb),
            _speed_cell(fit, short=compact),
            _verdict_cell(fit),
            _license_cell(entry, sym),
        ]
        if not compact:
            row.append(escape(why_line(fit)))
        table.add_row(*row)
    return table


def show_model_table(ui: UI, fits: list[FitResult], *, title: Optional[str] = None, start: int = 1,
                     show_repo: bool = False, interactive: bool = True) -> None:
    """Print :func:`model_table` through the UI's console, compact on narrow terminals.

    `interactive` is False when nothing will be asked afterwards (``--list-models``),
    so the hint points to the menu instead of offering a command that can't be typed.
    """
    compact = ui.console.width < WIDE_TABLE_MIN_COLUMNS
    ui.console.print(model_table(fits, symbols_for(ui), title=title, start=start, show_repo=show_repo,
                                 compact=compact))
    if any(b in ("fastest", "smartest") for f in fits for b in f.badges):
        ui.say(f"[dim]{escape(BADGE_FOOTNOTE)}[/dim]")
    if compact and interactive:
        ui.say("[dim](Estimated speeds. Type [bold]why 1[/bold] to see why I rated pick 1 that way.)[/dim]")
    elif compact:
        ui.say("[dim](Estimated speeds. Run [bold]gettowork[/bold] and type [bold]why N[/bold] at the model menu "
               "to see how I rated a pick.)[/dim]")


# ---------------------------------------------------------------------------
# The interactive flow
# ---------------------------------------------------------------------------

# What a step can ask the main loop to do next.
_BACK, _MOCK, _QUIT = "back", "mock", "quit"

# Failures the player should hear about in one friendly sentence (anything
# else is a bug, and cli.py's "rerun with --debug" handler takes over).
_EXPECTED_ERRORS = (BackendError, runtime_install.RuntimeInstallError, download.DownloadError, OSError)

_WHY_RE = re.compile(r"^(?:why|explain|\?)\s*#?\s*(\d{1,6})$")


@dataclass
class _Choice:
    """What the player picked: a model to download, a local file, or an Ollama tag."""

    entry: Optional[ModelEntry] = None  # already adjusted to the chosen quant
    fit: Optional[FitResult] = None
    gguf: Optional[Path] = None  # a GGUF file already on disk (--gguf, or a returning player's)
    server_exe: Optional[Path] = None  # a returning player's installed llama-server
    ollama_model: Optional[str] = None  # --ollama-model, or a returning player's Ollama tag
    import_gguf: Optional[Path] = None  # an already-downloaded GGUF to hand to Ollama (no second download)

    @property
    def label(self) -> str:
        if self.entry is not None:
            return self.entry.display_name
        if self.ollama_model:
            return self.ollama_model
        if self.gguf is not None:
            return self.gguf.name
        return "your model"


class _StartFailed(Exception):
    """A backend couldn't be prepared; the message is already friendly."""


StepOutcome = Union[SetupResult, str]


def run_setup(ui: UI, settings: Settings, *, args: Any, services: Optional[SetupServices] = None) -> Optional[SetupResult]:
    """Hardware -> discovery -> pick -> install -> warm-up. Returns None if the player quits.

    Args:
        ui: Where all questions and messages go.
        settings: Loaded settings; the choice is saved back for next time.
        args: The parsed command line (see ``cli.py``). Missing attributes
            fall back to defaults, so a bare ``argparse.Namespace()`` works.
        services: Stand-ins for the outside world (tests); defaults are real.

    Ctrl+C at a prompt raises :class:`~gettowork.ui.UserQuit`, which cli.py
    turns into a friendly goodbye. Any backend created here but not returned
    is always closed first.
    """
    return _SetupFlow(ui, settings, args, services or SetupServices()).run()


class _SetupFlow:
    """One run of the setup conversation, one small method per step."""

    def __init__(self, ui: UI, settings: Settings, args: Any, services: SetupServices) -> None:
        self.ui = ui
        self.settings = settings
        self.args = args
        self.svc = services
        self.sym = symbols_for(ui)
        self.specs: Optional[SystemSpecs] = None
        self.search: Optional[ModelSearch] = None
        self._last_model_file: Optional[Path] = None  # what the built-in engine had downloaded before it failed
        # {placement: factor} of the speed corrections folded into self.specs so far
        # (saved ones applied at start-up, then any learned this session).
        self._specs_factors: dict[str, float] = {}
        # Real speeds measured this session, by (model key, QUANT): the menu shows
        # these instead of an estimate for a model the player has just tried.
        self._measured: dict[tuple[str, str], float] = {}
        self._last_learning: Optional[str] = None  # "learned" / "clamped" / None: what the last warm-up taught

    # -- the overall flow ---------------------------------------------------------

    def run(self) -> Optional[SetupResult]:
        resumed = self._welcome_back()
        if resumed is not None:
            return resumed
        self._check_hardware()
        if self._opt("mock"):
            return self._mock_mode()

        choice: Union[_Choice, str, None] = self._direct_choice()
        while True:
            if choice is None:
                choice = self._pick_model()
            if choice == _QUIT:
                return None
            if choice == _MOCK:
                return self._start_mock()
            assert isinstance(choice, _Choice)

            kind = self._plan_backend(choice)
            if kind is None:
                outcome: StepOutcome = self._failure_menu(None)
            elif not self._confirm(choice, kind):
                self.ui.info("No problem - nothing was downloaded. Pick whichever you like:")
                outcome = _BACK
            else:
                outcome = self._install_and_start(choice, kind)

            if isinstance(outcome, SetupResult):
                return outcome
            if outcome == _QUIT:
                return None
            if outcome == _MOCK:
                return self._start_mock()
            choice = None  # _BACK: show the menu again

    def _opt(self, name: str, default: Any = None) -> Any:
        """An argparse option, tolerating a Namespace that lacks it."""
        value = getattr(self.args, name, default)
        return default if value is None else value

    def _auto_backend(self) -> bool:
        return (self._opt("backend", "auto") or "auto") == "auto"

    # -- step 1: welcome back -----------------------------------------------------

    def _saved_choice(self) -> Optional[tuple[str, _Choice]]:
        """The returning player's previous (backend, choice), if it can still start as-is."""
        if any(self._opt(flag) for flag in ("mock", "model", "gguf", "ollama_model", "quant",
                                           "refresh_models", "all_licenses")):
            return None  # the command line asks for something specific
        s = self.settings
        kind = s.backend
        if kind not in BACKEND_KINDS or self._opt("backend", "auto") not in ("auto", kind):
            return None
        extra = s.extra if isinstance(s.extra, dict) else {}
        entry = entry_from_dict(extra.get("model_entry"))
        if entry is not None and s.model_quant and s.model_quant.upper() != entry.quant.upper():
            entry = dataclasses.replace(entry, quant=s.model_quant, gguf_files=())

        if kind == "ollama":
            if not s.ollama_model or not self._probe("ollama", _Choice(entry=entry, ollama_model=s.ollama_model))[0]:
                return None
            return kind, _Choice(entry=entry, ollama_model=s.ollama_model)

        path = Path(s.model_path).expanduser() if s.model_path else None
        if path is None:
            return None
        if not path.is_file():
            self.ui.info(f"The model file you played with last time ({escape(str(path))}) isn't there any more, "
                         "so let's pick a model again.")
            return None
        exe = Path(s.server_exe).expanduser() if s.server_exe else None
        if kind == "managed":
            if exe is not None and not exe.is_file():
                exe = None
            if exe is None and not self._installed_runtimes():
                return None
        return kind, _Choice(entry=entry, gguf=path, server_exe=exe)

    def _welcome_back(self) -> Optional[SetupResult]:
        saved = self._saved_choice()
        if saved is None:
            return None
        kind, choice = saved
        extra = self.settings.extra if isinstance(self.settings.extra, dict) else {}
        name = extra.get("model_label") or choice.label
        last_speed = extra.get("last_tokens_per_s")
        if isinstance(last_speed, (int, float)) and last_speed > 0:
            self.ui.say(f"[dim]Last time it talked at {format_speed(last_speed)}. Say n to pick a different model.[/dim]")
        if not self.ui.confirm(f"Welcome back! Play with {escape(str(name))} again?", default=True):
            self.ui.info("Sure thing - let's find you a model.")
            return None

        with self.ui.status("Getting everything ready..."):
            self.specs = self._detect(server_exe=choice.server_exe)
        if choice.entry is not None and choice.entry.quant:
            choice.fit = fit_for_quant(self.specs, choice.entry, choice.entry.quant)
        try:
            outcome = self._start(kind, choice)
        except _StartFailed:
            self.ui.info("No worries - let's set things up again from the top.")
            return None
        return outcome if isinstance(outcome, SetupResult) else None

    # -- step 2: hardware -----------------------------------------------------------

    def _detect(self, *, server_exe: Optional[Path] = None) -> SystemSpecs:
        specs = self.svc.detect_specs()
        specs = apply_engine_limits(specs, self._opt("backend", "auto") or "auto")
        wanted = self._opt("backend", "auto") or "auto"
        engine = (wanted if wanted not in ("auto", "managed")
                  else planned_engine_key(specs, self.settings, server_exe=server_exe))
        specs, applied = _apply_factors(specs, saved_calibration(specs, self.settings, engine=engine))
        self._specs_factors = dict(applied)
        if applied:
            specs.notes.append("Speed estimates fine-tuned with the real speed measured in an earlier game.")
        return specs

    def _check_hardware(self) -> None:
        if self.specs is None:
            with self.ui.status("Let me take a look at your computer..."):
                self.specs = self._detect()
        show_hardware(self.ui, self.specs)

    # -- step 3: discovery + the model menu -------------------------------------------

    def _find(self, *, refresh: bool = False, offline: Optional[bool] = None,
              offline_reason: Optional[str] = None) -> ModelSearch:
        assert self.specs is not None
        user_offline = bool(self._opt("offline"))
        self.search = find_models(
            self.ui,
            self.specs,
            self.svc,
            refresh=refresh or bool(self._opt("refresh_models")),
            offline=user_offline if offline is None else offline,
            allow_all_licenses=bool(self._opt("all_licenses")),
            offline_reason=None if user_offline else offline_reason,
        )
        return self.search

    def _rerank(self) -> None:
        """Re-run the fit engine (e.g. after a speed calibration), without searching again."""
        if self.search is None or self.specs is None:
            return
        ranked = catalog.rank_models(self.specs, list(self.search.discovery.models) or list(catalog.MODEL_CATALOG),
                                     downloaded=downloaded_checker())
        ranked = _with_measured_speeds(ranked, self._measured)
        self.search = ModelSearch(self.search.discovery, ranked, catalog.pick_shortlist(ranked, SHORTLIST_SIZE))

    def _mock_mode(self) -> SetupResult:
        """--mock: show what we'd recommend, clearly labelled, then use the pretend model."""
        ui = self.ui
        ui.heading("Pretend-model mode (--mock)")
        ui.info("You asked for the pretend model, so nothing will be downloaded.")
        # No downloads, no waiting: the saved list or the built-in picks.
        search = self._find(offline=True, offline_reason="Pretend-model mode doesn't go online")
        rec = search.recommended
        if rec is not None:
            ui.say(
                f"[bold]For a real game on this computer I'd recommend:[/bold] {escape(rec.model.display_name)} "
                f"({format_size(rec.download_gb)}, {format_speed(rec.est_tokens_per_s)}) - "
                "just run [bold]gettowork[/bold] without --mock whenever you're ready."
            )
        return self._start_mock()

    def _start_mock(self) -> SetupResult:
        if self.specs is None:
            self.specs = self._detect()
        backend = self.svc.make_backend("mock")
        backend.prepare(self.ui)
        self.ui.say(
            f"[green]{self.sym.tick}[/green] The pretend model is ready! It's scripted and offline, so it answers "
            "instantly - but it isn't a real AI."
        )
        return SetupResult(backend=backend, entry=None, specs=self.specs, fit=None)

    def _pick_model(self) -> Union[_Choice, str]:
        """The one real decision: which model? Enter = the recommended pick."""
        ui = self.ui
        search = self.search or self._find()
        showing_all = False
        redraw = True
        while True:
            search = self.search or search
            fits = search.ranked[:FULL_LIST_LIMIT] if showing_all else search.shortlist
            default = self._default_answer(search, fits)
            if redraw:
                self._show_menu(search, fits, showing_all)
                redraw = False
            answer = ui.ask("Your pick", default=default).strip()
            word = answer.lower()

            if word.isdecimal():
                index = int(word) - 1 if len(word) <= 6 else -1  # (int() refuses 4,300+ digits)
                if 0 <= index < len(fits):
                    choice = self._choose_fit(fits[index])
                    if choice is not None:
                        return choice
                    continue
                shown = word if len(word) <= 12 else word[:12] + "..."
                ui.warn(f"There's no number {escape(shown)} in the list - try 1 to {len(fits)}.")
                continue
            if word in ("mock", "pretend"):
                return _MOCK
            if word in ("quit", "q", "exit"):
                return _QUIT
            if word in ("more", "all", "m"):
                showing_all, redraw = True, True
                continue
            if word in ("back", "b", "less", "short"):
                showing_all, redraw = False, True
                continue
            if word in ("refresh", "r"):
                if self._opt("offline"):
                    ui.warn("You're in offline mode, so I can't search Hugging Face right now.")
                else:
                    self._find(refresh=True, offline=False)
                    showing_all, redraw = False, True
                continue
            if word in ("custom", "c"):
                choice = self._custom()
                if choice is not None:
                    return choice
                redraw = True
                continue
            if word in ("learn", "l", "help", "?"):
                self._teach_all()
                # The lessons are long: let them be read, then show the menu again
                # below them, so the numbered list isn't scrolled out of sight.
                ui.pause("Press Enter to see the model list again")
                redraw = True
                continue
            match = _WHY_RE.match(word)
            if match:
                index = int(match.group(1)) - 1
                if 0 <= index < len(fits):
                    assert self.specs is not None
                    ui.teach(f"why pick {index + 1}?", catalog.explain_fit(self.specs, fits[index]))
                else:
                    ui.warn(f"There's no number {index + 1} in the list.")
                continue
            by_name = next((f for f in fits if word in (f.model.display_name.lower(), f.model.key.lower(),
                                                        f.model.hf_repo.lower())), None)
            if by_name is not None:
                choice = self._choose_fit(by_name)
                if choice is not None:
                    return choice
                continue
            ui.warn("Type a number from the list, or just press Enter for my recommendation.")

    def _default_answer(self, search: ModelSearch, fits: list[FitResult]) -> str:
        rec = search.recommended
        for i, fit in enumerate(fits, 1):
            if rec is not None and fit.model.key == rec.model.key and fit.quant == rec.quant:
                return str(i)
        return "mock" if rec is None else "back"

    def _show_menu(self, search: ModelSearch, fits: list[FitResult], showing_all: bool) -> None:
        ui, sym = self.ui, self.sym
        if showing_all:
            ui.heading("Every model I found, best first")
            show_model_table(ui, fits)
            hidden = len(search.ranked) - len(fits)
            if hidden > 0:
                ui.say(f"[dim]...plus {hidden} more that are far too big for this computer.[/dim]")
            ui.say("[dim]Type a number to choose, [bold]back[/bold] for the short list, or [bold]why N[/bold] "
                   "to see how I rated one.[/dim]")
            return
        ui.heading("Models that fit your computer")
        if not fits:
            ui.warn("I couldn't find a model that runs comfortably here - they're all too big or too slow. "
                    "You can still look at [bold]more[/bold], paste a [bold]custom[/bold] model, or play with "
                    "the pretend model ([bold]mock[/bold]).")
        else:
            show_model_table(ui, fits)
            if search.recommended is not None:
                ui.say(f"Press [bold]Enter[/bold] for my {sym.star} recommendation, or type a number. "
                       "That's the only big choice - I'll show you the plan to confirm, then handle the rest.")
            else:
                ui.say("Type a number to choose.")
        ui.say("[dim]Also: [bold]more[/bold] (every model I found) · [bold]refresh[/bold] (search Hugging Face "
               "again) · [bold]custom[/bold] (paste any Hugging Face GGUF) · [bold]mock[/bold] (play offline, "
               "no download) · [bold]learn[/bold] (how I chose) · [bold]quit[/bold][/dim]")

    def _teach_all(self) -> None:
        ui = self.ui
        ui.teach("how I check whether a model fits", catalog.MEMORY_FORMULA_EXPLAINER)
        ui.teach("why speed is all about memory bandwidth", perf.SPEED_EXPLAINER)
        ui.teach("where the models come from", hf_discovery.DISCOVERY_EXPLAINER)
        ui.teach("the engine that runs your model", runtime_install.RUNTIME_EXPLAINER)
        ui.say("[dim]Tip: type [bold]why 2[/bold] to see the exact working behind pick number 2.[/dim]")

    def _choose_fit(self, fit: FitResult) -> Optional[_Choice]:
        """Turn a menu row into a choice, double-checking picks that won't run well."""
        quant = self._opt("quant")
        if quant:
            assert self.specs is not None
            fit = fit_for_quant(self.specs, fit.model, quant)
        if not self._ok_despite_warnings(fit):
            return None
        return _Choice(entry=entry_for_fit(fit), fit=fit)

    def _ok_despite_warnings(self, fit: FitResult) -> bool:
        ui = self.ui
        if fit.verdict == "no":
            ui.warn(f"Heads-up: {escape(fit.reason)}")
            return ui.confirm("Try it anyway?", default=False)
        tps = fit.est_tokens_per_s
        if tps is not None and tps < SLOW_TOKENS_PER_S:
            ui.warn(f"Heads-up: this one would be very slow here ({format_speed(tps)}), so each turn could "
                    "take minutes.")
            return ui.confirm("Try it anyway?", default=False)
        return True

    def _custom(self) -> Optional[_Choice]:
        """Let the player paste any Hugging Face GGUF repo; check its fit before anything downloads."""
        ui = self.ui
        assert self.specs is not None
        ui.say("Paste a Hugging Face model id or link - it needs GGUF files, for example "
               "[bold]unsloth/Qwen3-4B-GGUF[/bold]. Press Enter to go back.")
        text = ui.ask("Hugging Face model", default="")
        if not text:
            return None
        repo, quant_in_ref = download.normalize_repo_id(text)
        if not repo:
            ui.warn("That doesn't look like a Hugging Face model id. It should look like owner/name.")
            return None
        quant = quant_in_ref or self._opt("quant")
        try:
            with ui.status(f"Looking up {escape(repo)} on Hugging Face..."):
                entry = self.svc.custom_entry(repo, quant)
        except download.DownloadError as exc:
            ui.warn(escape(str(exc)))
            return None
        except Exception as exc:  # anything odd from the Hub: explain, don't crash
            ui.warn(f"I couldn't look that model up ({escape(str(exc))}).")
            return None

        if catalog.size_unknown(entry):
            self._license_warning(entry)
            ui.warn("I couldn't work out how big this model is, so I can't check whether it fits.")
            if not ui.confirm("Try it anyway?", default=False):
                return None
            return _Choice(entry=entry, fit=None)  # no estimate: nothing to compare the real speed with
        fit = (fit_for_quant(self.specs, entry, quant) if quant
               else catalog.evaluate_fit(self.specs, entry, downloaded=downloaded_checker()))
        show_model_table(ui, [fit], title="Your custom pick")
        ui.say(f"[dim]{escape(fit.reason)}[/dim]")
        self._license_warning(entry)
        if not self._ok_despite_warnings(fit):
            return None
        return _Choice(entry=entry_for_fit(fit), fit=fit)

    def _license_warning(self, entry: ModelEntry) -> None:
        """The same yellow license warning, however the player named the model
        (the custom menu, or --model): only Apache-2.0 / MIT are suggested by default."""
        if not catalog.is_permissive(entry.license):
            self.ui.warn(f"This model's license is {escape(entry.license or 'unknown')}, not Apache-2.0 or MIT - "
                         f"please read it on https://huggingface.co/{escape(entry.hf_repo)} before using it.")

    def _direct_choice(self) -> Optional[_Choice]:
        """--gguf / --ollama-model / --model skip the menu."""
        gguf = self._opt("gguf")
        if gguf:
            # Absolute, so the saved choice still works from any other folder next time.
            return _Choice(gguf=_absolute(Path(gguf)))
        ollama_model = self._opt("ollama_model")
        if ollama_model:
            return _Choice(ollama_model=str(ollama_model))
        wanted = self._opt("model")
        return self._resolve_model_arg(str(wanted)) if wanted else None

    def _resolve_model_arg(self, wanted: str) -> Optional[_Choice]:
        """--model: a curated key, a discovered repo, or any Hugging Face GGUF repo."""
        ui = self.ui
        assert self.specs is not None
        entry = catalog.get_model(wanted)
        if entry is None:
            search = self.search or self._find()
            lowered = wanted.strip().lower()
            entry = next((m for m in search.discovery.models if lowered in (m.key.lower(), m.hf_repo.lower())), None)
        if entry is None:
            repo, quant_in_ref = download.normalize_repo_id(wanted)
            try:
                with ui.status(f"Looking up {escape(repo or wanted)} on Hugging Face..."):
                    entry = self.svc.custom_entry(repo or wanted, quant_in_ref or self._opt("quant"))
            except Exception as exc:  # DownloadError is friendly; anything else gets a short note
                ui.warn(f"I couldn't find the model '{escape(wanted)}': {escape(str(exc))}")
                ui.info("Let's pick one from the list instead.")
                return None
        quant = self._opt("quant")
        if catalog.size_unknown(entry):
            ui.say(f"You asked for [bold]{escape(entry.display_name)}[/bold].")
            self._license_warning(entry)
            ui.warn("I couldn't work out how big this model is, so I can't check whether it fits - "
                    "I'll give it a go since you asked for it.")
            return _Choice(entry=entry, fit=None)  # no made-up estimate (and nothing wrong to learn from)
        fit = (fit_for_quant(self.specs, entry, quant) if quant
               else catalog.evaluate_fit(self.specs, entry, downloaded=downloaded_checker()))
        colour, words = VERDICT_LOOK.get(fit.verdict, ("white", fit.verdict))
        ui.say(f"You asked for [bold]{escape(entry.display_name)}[/bold] - [{colour}]{words}[/{colour}]: "
               f"{escape(fit.reason)}")
        self._license_warning(entry)
        if fit.verdict == "no":
            ui.warn("It probably won't run well here, but I'll give it a go since you asked for it.")
        return _Choice(entry=entry_for_fit(fit), fit=fit)

    # -- step 4: one confirmation ------------------------------------------------------

    def _installed_runtimes(self) -> list:
        try:
            return list(self.svc.installed_runtimes())
        except Exception:
            return []

    def _probe(self, kind: str, choice: _Choice) -> tuple[bool, str]:
        """Could this kind of backend run here? (Fast; nothing is downloaded or started.)"""
        try:
            return self.svc.make_backend(kind, **self._backend_kwargs(choice)).is_available()
        except Exception as exc:
            return False, str(exc)

    def _plan_backend(self, choice: _Choice) -> Optional[str]:
        """Which engine to use. Automatic unless --backend says otherwise."""
        if choice.ollama_model and choice.gguf is None:
            return "ollama"
        wanted = self._opt("backend", "auto") or "auto"
        if wanted != "auto":
            if wanted == "ollama" and choice.gguf is not None:
                self.ui.warn("Ollama can't load a GGUF file directly, so I'll use the built-in engine for --gguf.")
                return "managed"
            return wanted
        ok, why = self._probe("managed", choice)
        if ok:
            return "managed"
        self.ui.warn(escape(why))
        if choice.gguf is None and self._probe("ollama", choice)[0]:
            remote = self._ollama_elsewhere(choice)
            if remote is None:
                self.ui.info("Ollama is running on your computer, so I'll use that instead.")
                return "ollama"
            self.ui.info(f"Ollama is running at {escape(ollama_backend.display_host(remote))} - another computer "
                         "(your OLLAMA_HOST setting).")
            self.ui.say(f"[dim]{escape(self._ollama_privacy_note(remote))}[/dim]")
            if self.ui.confirm("Use that Ollama for this game?", default=True):
                return "ollama"
        if self._probe("llamacpp", choice)[0]:
            self.ui.info("llama-cpp-python is installed, so I'll use that instead.")
            return "llamacpp"
        return None

    def _engine_step(self, choice: _Choice) -> tuple[str, list[str], bool]:
        """(title, lines, needs a download?) for the llama.cpp engine."""
        title = "The llama.cpp engine - the program that runs the model"
        if choice.server_exe is not None and choice.server_exe.is_file():
            return title, ["Already installed - nothing to download."], False
        assert self.specs is not None
        try:
            plan = runtime_install.usable_plan(self.specs) or [runtime_install.CPU]  # skips builds known not to run here
        except Exception:
            plan = [runtime_install.CPU]
        best = plan[0]
        installed = {variant: tag for _exe, tag, variant in self._installed_runtimes()}
        if best.name in installed:
            return title, [f"Already installed ({escape(installed[best.name])}, {escape(best.display)} build) - "
                           "nothing to download."], False
        lines = [
            f"Official {escape(best.display)} build for your computer - {ENGINE_SIZE_HINTS.get(best.name, 'a small download')}, "
            f"{escape(runtime_install.license_text(best))}",
            f"From: {runtime_install.LLAMA_CPP_URL}/releases",
            f"Into: {escape(str(config.runtime_dir() / 'llama.cpp'))}",
        ]
        if best.name == runtime_install.VULKAN.name and specs_module.uses_built_in_graphics(self.specs):
            lines.insert(1, "It will use your built-in graphics, which share your RAM - so the speeds I showed "
                            "still apply (if they can't help, it simply runs on the processor).")
        if len(plan) > 1:
            def backup(v: runtime_install.RuntimeVariant) -> str:
                # Say up front which backups are big downloads (the CUDA builds).
                return f"{v.display} ({ENGINE_SIZE_HINTS[v.name]})" if v.needs_cudart else v.display

            backups = f" {self.sym.arrow} ".join(backup(v) for v in plan[1:])
            lines.append(f"[dim]If that build can't start, I'll try the next one instead (and say so): "
                         f"{escape(backups)}[/dim]")
        return title, lines, True

    def _model_step(self, choice: _Choice, kind: str) -> tuple[str, list[str], bool]:
        """(title, lines, needs a download?) for the model itself."""
        if choice.gguf is not None and kind != "ollama":
            return "Your own model file", [escape(str(choice.gguf)), "Already on your computer - nothing to download."], False
        if choice.entry is None:  # --ollama-model TAG
            tag = escape(choice.ollama_model or "")
            return f"The model {tag}", [
                "Ollama downloads it once (if it doesn't have it yet) into its own model folder.",
                "Please check the model's license on its page before using it.",
            ], True

        entry, fit = choice.entry, choice.fit
        size = format_size(fit.download_gb if fit and fit.download_gb else entry.file_size_gb)
        page = f"https://huggingface.co/{entry.hf_repo}"
        name = entry.license or "unknown"
        license_text = escape(name) + ("" if "license" in name.lower() else " license")
        if entry.source != "curated":
            license_text += " as declared on Hugging Face"  # we read the tag; the model card has the details
        if entry.license_url and entry.license_url.rstrip("/") != page:
            license_text += f" (read it: {escape(entry.license_url)})"
        title = f"{escape(entry.display_name)} ({escape(entry.quant)}) - the model itself"
        note = ("[dim]The weights come from Hugging Face, shared by their authors under that license - they "
                "aren't part of this game.[/dim]")
        if kind == "ollama":
            return title, [
                f"{size} · {license_text}",
                f"Ollama pulls it from Hugging Face: {escape(entry.ollama_ref)}",
                "Into: Ollama's own model folder",
                note,
            ], True

        dest = download.model_folder(entry.hf_repo)
        files = entry.gguf_files
        source = f"From: {escape(page)}"
        if files:
            first = files[0].replace("\\", "/").rsplit("/", 1)[-1]
            more = f" + {len(files) - 1} more part{'s' if len(files) > 2 else ''}" if len(files) > 1 else ""
            source += f" (file: {escape(first)}{more})"
        lines = [f"{size} · {license_text}", source, f"Into: {escape(str(dest))}", note]
        already = bool(files) and all(
            dest.joinpath(*[p for p in f.replace("\\", "/").split("/") if p]).is_file() for f in files
        )
        if not files and entry.quant:  # exact names unknown: look for a finished copy of this quant
            already = download.find_local_copy(entry.hf_repo, entry.quant, dest) is not None
        if already:
            lines.insert(0, "[green]Already downloaded - nothing to fetch![/green]")
        return title, lines, not already

    def _confirm(self, choice: _Choice, kind: str) -> bool:
        """The one Y/n: exactly what will be downloaded, from where, and where it goes."""
        ui = self.ui
        ui.heading("Here's the plan")
        steps: list[tuple[str, list[str]]] = []
        needs_download = False
        if kind == "managed":
            title, lines, dl = self._engine_step(choice)
            steps.append((title, lines))
            needs_download |= dl
        elif kind == "ollama":
            steps.append(("Ollama - the app that will fetch and run the model", ["Nothing else to install."]))
        elif kind == "llamacpp":
            steps.append(("llama-cpp-python - the Python package that will run the model", ["Nothing else to install."]))
        title, lines, dl = self._model_step(choice, kind)
        steps.append((title, lines))
        needs_download |= dl
        steps.append(("Start it privately on your computer", ["...and time a quick test sentence to see how fast it talks."]))

        for number, (title, lines) in enumerate(steps, 1):
            ui.say(f"  [bold cyan]{number}.[/bold cyan] [bold]{title}[/bold]")
            for line in lines:
                ui.say(f"     {line}")
        ui.say()
        if kind == "ollama":
            ui.say("[dim]Ollama keeps models in its own folder ('ollama rm <name>' removes one). Nothing else is "
                   "installed.[/dim]")
        else:
            ui.say(f"[dim]Nothing is installed system-wide and no admin rights are needed - everything lives in "
                   f"{escape(str(config.config_dir()))}. Delete that folder any time to remove it all.[/dim]")
        if needs_download and self._opt("offline"):
            ui.warn("You're in offline mode, so this only works if these files are already on your computer.")
        return ui.confirm("Shall I go ahead?", default=True)

    # -- step 5: install and start, with automatic fallbacks ------------------------------

    def _backend_kwargs(self, choice: _Choice) -> dict[str, Any]:
        return {
            "entry": choice.entry,
            "specs": self.specs,
            "quant": choice.entry.quant if choice.entry is not None else None,
            "model_path": choice.gguf,
            "server_exe": choice.server_exe,
            "ollama_model": choice.ollama_model,
            "gguf_path": choice.import_gguf,
        }

    def _install_and_start(self, choice: _Choice, kind: str) -> StepOutcome:
        first_kind, first_choice = kind, choice
        tried_ollama = kind == "ollama"
        while True:
            self._last_model_file = None
            try:
                return self._start(kind, choice)
            except _StartFailed:
                pass
            if (kind == "managed" and self._auto_backend() and choice.entry is not None and not tried_ollama
                    and self._probe("ollama", choice)[0]):
                tried_ollama = True
                ollama_choice = self._offer_ollama(choice)
                if ollama_choice is not None:
                    kind, choice = "ollama", ollama_choice
                    continue
            action = self._failure_menu(kind, after_fallback=kind != first_kind)
            if action != "retry":
                return action
            kind, choice, tried_ollama = first_kind, first_choice, first_kind == "ollama"

    def _offer_ollama(self, choice: _Choice) -> Optional[_Choice]:
        """The built-in engine failed but Ollama is running: ask before using it.

        If the model file was already downloaded, it's handed to Ollama (it
        keeps its own copy, but nothing is downloaded again). Otherwise Ollama
        would download the model itself - never without saying how big it is.
        """
        ui, entry = self.ui, choice.entry
        assert entry is not None
        local = self._last_model_file
        if local is not None and (not local.is_file() or re.search(r"-\d{5}-of-\d{5}\.gguf$", local.name, re.I)):
            local = None  # gone, or split into parts (Ollama can only take a single file)
        remote = self._ollama_elsewhere(choice)
        shown = ollama_backend.display_host(remote) if remote else ""
        where = "on your computer" if remote is None else f"at {escape(shown)} (another computer - your OLLAMA_HOST)"
        if local is not None:
            size = format_size(local.stat().st_size / 1e9)
            ui.info(f"Good news: Ollama is running {where}, so it could run this model instead.")
            if remote is None:
                ui.say(f"[dim]I'd hand it the file I already downloaded - nothing new to download. Ollama keeps its "
                       f"own copy ({size}) in its own folder; the game's copy stays in "
                       f"{escape(str(local.parent))}.[/dim]")
            else:
                ui.say(f"[dim]I'd send it the file I already downloaded ({size}) over your network - nothing new "
                       f"from the internet. {escape(self._ollama_privacy_note(remote))}[/dim]")
            prompt = "Shall I hand the model to Ollama?" if remote is None else \
                f"Shall I send the model ({size}) to the Ollama at {escape(shown)}?"
            if not ui.confirm(prompt, default=True):
                return None
            name = _ollama_import_name(entry)
            return dataclasses.replace(choice, ollama_model=name, import_gguf=local, gguf=None, server_exe=None)
        fit = choice.fit
        size = format_size(fit.download_gb if fit and fit.download_gb else entry.file_size_gb)
        ui.info(f"Good news: Ollama is running {where}, so it could fetch and run the same model instead.")
        ui.say(f"[dim]Ollama would download it itself ({size}) from Hugging Face into its own model folder.[/dim]")
        if remote is not None:
            ui.say(f"[dim]{escape(self._ollama_privacy_note(remote))}[/dim]")
        if not ui.confirm(f"Shall I ask Ollama to download it ({size})?", default=True):
            return None
        return choice

    def _ollama_elsewhere(self, choice: _Choice) -> Optional[str]:
        """Where Ollama runs when it isn't this computer ("gpu-box.lan:11434"), else None."""
        try:
            backend = self.svc.make_backend("ollama", **self._backend_kwargs(choice))
        except Exception:
            return None
        host = getattr(backend, "host", None)
        with contextlib.suppress(Exception):
            backend.close()
        if not isinstance(host, str) or not host or ollama_backend.is_local_host(host):
            return None
        return host

    @staticmethod
    def _ollama_privacy_note(remote: str) -> str:
        return (f"Your plans and the story would be sent to {ollama_backend.display_host(remote)} over your network"
                + ("." if remote.lower().startswith("https://") else " without encryption (plain http).")
                + " (Unset OLLAMA_HOST to use an Ollama on this computer instead.)")

    def _start(self, kind: str, choice: _Choice) -> StepOutcome:
        """One attempt: prepare the backend, warm it up, save the choice.

        Returns a SetupResult, or _BACK if the model is too slow and the player
        wants a faster one. Raises _StartFailed (already explained to the player).
        """
        ui = self.ui
        backend = self.svc.make_backend(kind, **self._backend_kwargs(choice))
        try:
            ui.heading(f"Getting {escape(choice.label)} ready")
            try:
                backend.prepare(ui, choice.entry)
            except _EXPECTED_ERRORS as exc:
                ui.warn(f"Hmm, that didn't work: {escape(str(exc))}")
                path = getattr(backend, "model_path", None)
                self._last_model_file = Path(path) if kind == "managed" and path else None
                raise _StartFailed(str(exc)) from exc
            try:
                tps = self._warm_up(backend, choice, kind)
            except EngineStopped as exc:  # it started, then broke on real work: a failed start after all
                ui.warn(f"Hmm, that didn't work: {escape(str(exc))}")
                path = getattr(backend, "model_path", None)
                self._last_model_file = Path(path) if kind == "managed" and path else None
                raise _StartFailed(str(exc)) from exc
            if tps is not None and tps < SLOW_TOKENS_PER_S and self._wants_faster(tps, choice):
                backend.close()
                return _BACK
        except BaseException:  # failures *and* Ctrl+C: never leave an engine running behind us
            backend.close()
            raise
        self._save(kind, backend, choice, tps)
        return SetupResult(backend=backend, entry=choice.entry, specs=self.specs or self._detect(), fit=choice.fit,
                           tokens_per_s=tps)

    def _failure_menu(self, kind: Optional[str], *, after_fallback: bool = False) -> str:
        """Friendly guidance plus retry / pick another / pretend model / quit.

        `kind` is the engine that just failed (None = no engine can run here at
        all); `after_fallback` means we'd already switched to it automatically.
        """
        ui = self.ui
        if kind is None:
            ui.warn("I can't run a real model on this computer automatically.")
        if after_fallback:
            ui.info("Neither the built-in engine nor Ollama could start this model. A smaller model often helps "
                    "(choose 'pick'), or try again in a bit if the internet is being flaky.")
        elif kind in (None, "managed"):
            ui.info(f"Plan B: the free Ollama app works on most computers - install it from {OLLAMA_DOWNLOAD_URL}, "
                    "start it, and I'll use it automatically. (Tinkerers: "
                    f"'{LLAMACPP_PIP_HINT}', then --backend llamacpp.)")
        elif kind == "ollama":
            ui.info(f"Ollama needs to be installed and running: get it from {OLLAMA_DOWNLOAD_URL}, then start the app "
                    "(or run 'ollama serve'). Without --backend ollama I'd use the built-in engine instead.")
        elif kind == "llamacpp":
            ui.info(f"This needs the llama-cpp-python package: '{LLAMACPP_PIP_HINT}'. Without --backend llamacpp "
                    "I'd use the built-in engine instead.")
        options = [
            ("retry", "Try again (handy if the internet hiccupped or you just started Ollama)"),
            ("pick", "Choose a different model"),
            ("mock", "Play now with the pretend model (offline, nothing to download)"),
            ("quit", "Stop for now - anything already downloaded is kept for next time"),
        ]
        if kind is None:  # no engine at all: another model wouldn't help
            options = [o for o in options if o[0] != "pick"]
        choice = ui.choose("What would you like to do?", options, default="retry")
        return _BACK if choice == "pick" else choice

    # -- step 6: warm-up, speed test and calibration --------------------------------------

    def _warm_up(self, backend: LLMBackend, choice: _Choice, kind: Optional[str] = None) -> Optional[float]:
        ui, sym = self.ui, self.sym
        try:
            tps = backend.benchmark(ui)
        except EngineStopped:
            raise  # not "up and running" after all: _start treats it as a failed start
        except _EXPECTED_ERRORS as exc:
            ui.warn(f"I couldn't time the model ({escape(str(exc))}), but it's up and running.")
            return None
        if not tps or tps <= 0:
            ui.say(f"[green]{sym.tick}[/green] Your model is up and running!")
            return None
        speed = f"~{tps:.0f} tokens/sec" if tps >= 10 else f"~{tps:.1f} tokens/sec"
        if tps >= 20:
            mood = "faster than you can read!"
        elif tps >= 8:
            mood = "nice and comfortable for a story."
        elif tps >= SLOW_TOKENS_PER_S:
            mood = "a little leisurely, but fine - maybe grab a coffee between turns."
        else:
            mood = "that's quite slow: each story turn could take a minute or more."
        icon = f"[green]{sym.tick}[/green]" if tps >= SLOW_TOKENS_PER_S else f"[yellow]{sym.warn}[/yellow]"
        ui.say(f"{icon} Your model is talking at [bold]{speed}[/bold] - {mood}")
        entry = choice.entry
        if entry is not None and catalog.thinking_mode(entry) == "switchable" and tps < catalog.THINKING_MIN_TOKENS_PER_S:
            ui.say("[dim]It can think out loud, but at this speed I'll ask it to answer straight away so turns stay "
                   "quick (start the game with --think to see its thinking anyway).[/dim]")
        if entry is not None:
            self._measured[(entry.key, (entry.quant or "").upper())] = float(tps)
        cpu_only = getattr(backend, "cpu_only", None)
        self._compare_with_estimate(choice.fit, tps, engine_on_cpu=cpu_only if isinstance(cpu_only, bool) else None,
                                    engine=engine_key(kind or getattr(backend, "name", None), backend))
        return tps

    def _compare_with_estimate(self, fit: Optional[FitResult], measured: float, *,
                               engine_on_cpu: Optional[bool] = None, engine: Optional[str] = None) -> None:
        """Tell the player how good the guess was, and learn from the measurement.

        We only learn when the engine really ran where the estimate assumed:

        * `engine_on_cpu` True (e.g. its GPU build wouldn't start, or its
          driver didn't answer) while the estimate was for the graphics card:
          the numbers can't be compared, so nothing is learned.
        * A processor-only estimate, but the engine may have used a graphics
          card anyway (GPU builds and Ollama offload layers by themselves):
          the speed isn't a pure processor measurement, so nothing is learned.
        """
        self._last_learning = None
        if fit is None or not fit.est_tokens_per_s or self.specs is None:
            return
        estimate = fit.est_tokens_per_s
        if engine_on_cpu and fit.placement in ("gpu", "unified", "partial"):
            self.ui.say(f"[dim]I estimated {format_speed(estimate)} on your graphics card, but the engine is running "
                        "on the processor instead, so it's slower than that - which is expected.[/dim]")
            return
        ratio = measured / estimate
        if 0.67 <= ratio <= 1.5:
            self.ui.say(f"[dim]I estimated {format_speed(estimate)}, so the fit engine was close![/dim]")
        else:
            self.ui.say(f"[dim]I estimated {format_speed(estimate)} - real speed depends on drivers and whatever else "
                        "is running.[/dim]")
        if fit.placement not in ("gpu", "unified", "cpu"):
            return
        if fit.placement == "cpu" and engine_on_cpu is not True and self.specs.gpus:
            # A graphics-card build (CUDA, Vulkan - or Metal on a Mac, where the
            # "processor" plan really runs on the GPU) may have helped: not a
            # clean processor-only measurement.
            return
        factor = self._bandwidth_factor(fit, measured)
        if factor is None:
            return  # mostly fixed per-token cost (a tiny or Mixture-of-Experts model): nothing to learn
        # Learn: nudge this computer's bandwidth numbers so future estimates match reality.
        # `factor` is measured against self.specs, which already includes any
        # correction applied at start-up (or learned earlier this session), so
        # the total correction is that one times `factor`. A saved correction
        # that was *not* applied (another engine build) plays no part here.
        low, high = _calibration_range(fit.placement)
        previous = self._specs_factors.get(fit.placement, 1.0)
        total = min(max(previous * factor, low), high)
        self._last_learning = "learned" if low <= previous * factor <= high else "clamped"
        self.specs = calibrate_specs(self.specs, fit.placement, total / previous)
        self._specs_factors[fit.placement] = total
        extra = dict(self.settings.extra) if isinstance(self.settings.extra, dict) else {}
        saved = extra.get(CALIBRATION_KEY)
        fingerprint = hardware_fingerprint(self.specs)
        factors = dict(saved.get("factors") or {}) if isinstance(saved, dict) and saved.get("fingerprint") == fingerprint else {}
        engines = dict(saved.get("engines") or {}) if factors and isinstance(saved, dict) else {}
        factors[fit.placement] = round(total, 3)
        if engine:
            engines[fit.placement] = engine
        else:
            engines.pop(fit.placement, None)
        extra[CALIBRATION_KEY] = {"fingerprint": fingerprint, "factors": factors, "engines": engines}
        self.settings.extra = extra
        if not 0.67 <= ratio <= 1.5:
            self.ui.say("[dim]I'll remember this measurement to make better guesses from now on.[/dim]")

    def _bandwidth_factor(self, fit: FitResult, measured: float) -> Optional[float]:
        """How far off the bandwidth behind `fit`'s estimate was, or None if this
        measurement can't tell us (see CALIBRATION_MIN_BANDWIDTH_SHARE).

        The speed model is ``seconds per token = GB read / (efficiency x
        bandwidth) + a fixed cost``. We solve it for the bandwidth that gives
        the measured speed, instead of scaling bandwidth by measured/estimated
        (which would blame bandwidth for errors in the fixed cost too).
        """
        assert self.specs is not None
        if fit.model.active_params_b or measured <= 0:
            return None  # Mixture-of-Experts: our guess of the active share would get the blame
        terms = catalog.speed_breakdown(self.specs, fit)
        if terms is None:
            return None
        active_gb, effective_bw, overhead = terms
        bandwidth_s = active_gb / effective_bw
        if bandwidth_s / (bandwidth_s + overhead) < CALIBRATION_MIN_BANDWIDTH_SHARE:
            return None
        measured_bandwidth_s = 1.0 / measured - overhead
        if measured_bandwidth_s <= 0:
            return None
        return bandwidth_s / measured_bandwidth_s

    def _wants_faster(self, tps: float, choice: Optional[_Choice] = None) -> bool:
        ui = self.ui
        # A menu, not a yes/no question: at a [Y/n] prompt "back" means "no",
        # which here would keep the slow model - the opposite of what's meant.
        answer = ui.choose(
            "Pick a faster model, or play with this one?",
            [("pick", "Choose a faster model from the list (recommended)"),
             ("play", "Play with this one anyway - each turn will just take a while")],
            default="pick",
            aliases={"back": "pick", "y": "pick", "yes": "pick", "faster": "pick", "menu": "pick",
                     "n": "play", "no": "play", "keep": "play", "anyway": "play"},
        )
        if answer != "pick":
            return False
        self._persist()  # keep the speed calibration even though we're not keeping the model
        self._rerank()
        if self.search is not None and not self._faster_options(tps, choice):
            # Don't send the player back to a menu whose best pick is the model they just tried.
            ui.warn("Hmm - that was already about the quickest model I could find for this computer, so another "
                    "pick won't be much faster.")
            ui.info("You can play with it anyway (each turn just takes a while), or go back and choose "
                    "[bold]mock[/bold] for the instant pretend model.")
            return not ui.confirm("Play with this model anyway?", default=True)
        if self._last_learning == "learned":
            ui.info("I've updated my speed estimates with that real measurement, so the list should be more "
                    "realistic now.")
        elif self._last_learning == "clamped":
            ui.info("That was far slower than I expected, so I've turned my speed estimates down as far as one "
                    "measurement safely allows - the other models may still be a bit slower than the list says. "
                    "The model you just tried shows its real, measured speed.")
        else:
            ui.info("The model you just tried now shows its real, measured speed. The other speeds are still "
                    "estimates, so treat them as rough guesses.")
        return True

    def _faster_options(self, tps: float, choice: Optional[_Choice]) -> list[FitResult]:
        """Menu candidates expected to be clearly faster than the `tps` just measured."""
        assert self.search is not None
        tried = (choice.entry.key, (choice.entry.quant or "").upper()) if choice and choice.entry else None
        return [
            f for f in self.search.ranked
            if f.verdict != "no"
            and (f.est_tokens_per_s or 0.0) >= max(1.5 * tps, SLOW_TOKENS_PER_S)
            and (f.model.key, (f.quant or f.model.quant or "").upper()) != tried
        ]

    # -- step 7: remember for next time ----------------------------------------------------

    def _persist(self) -> bool:
        try:
            self.settings.save()
        except OSError as exc:
            self.ui.warn(f"I couldn't save your settings for next time ({escape(str(exc))}) - no harm done.")
            return False
        return True

    def _save(self, kind: str, backend: LLMBackend, choice: _Choice, tps: Optional[float]) -> None:
        s = self.settings
        entry = choice.entry
        s.backend = kind
        s.model_key = entry.key if entry is not None else None
        s.model_quant = entry.quant if entry is not None else None
        if kind == "ollama":
            s.model_path = None
            s.server_exe = None
            s.ollama_model = str(getattr(backend, "model", "") or choice.ollama_model or "") or None
        else:
            path = getattr(backend, "model_path", None) or choice.gguf
            s.model_path = str(_absolute(Path(path))) if path else None
            exe = getattr(backend, "server_exe", None) if kind == "managed" else None
            s.server_exe = str(_absolute(Path(exe))) if exe else None
            s.ollama_model = None
        extra = dict(s.extra) if isinstance(s.extra, dict) else {}
        if entry is not None:
            extra["model_entry"] = entry_to_dict(entry)
        else:
            extra.pop("model_entry", None)
        extra["model_label"] = choice.label
        if tps:
            extra["last_tokens_per_s"] = round(tps, 1)
        else:
            extra.pop("last_tokens_per_s", None)
        s.extra = extra
        if self._persist():
            self.ui.say("[dim]Saved your choice - next time it's just one keypress.[/dim]")
