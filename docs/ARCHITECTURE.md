# Get To Work — Architecture & Module Contracts

This document is both a guide for learners and the **binding contract** the
modules are built against. If code and this document disagree, fix one of them.

## The player's journey

The design goal: **the player only ever picks a model.** Everything else —
finding models that fit, downloading the engine, downloading the weights,
starting the model — is automatic, explained in friendly language, and
reversible.

1. `gettowork` launches → banner. Returning players: "Welcome back! Play again
   with Qwen3 4B? [Y/n]" skips straight to the game.
2. **Hardware check** (`specs.py` + `perf.py`): OS, CPU (+ SIMD flags), RAM,
   GPU(s)/VRAM, Apple unified memory, free disk, plus a ~0.3 s memory-bandwidth
   micro-benchmark. Summarised in plain English ("16 GB of RAM and an NVIDIA
   RTX 3060 with 12 GB of video memory — a solid setup for local AI!"), with a
   details table and a "Learn" panel on why memory size *and* bandwidth matter.
3. **Live model discovery** (`hf_discovery.py`): searches the Hugging Face Hub
   API for popular instruction-tuned GGUF models from trusted publishers,
   reads their real GGUF metadata (parameter count, architecture, context
   length), real per-file sizes, and licenses. Results are cached on disk
   (default 3 days) so later launches are instant and work offline. If the Hub
   is unreachable and there is no cache, the built-in curated seed list
   (`catalog.py`) is used instead, with a note.
4. **Fit engine** (`catalog.py` + `perf.py`): for every candidate, picks the
   best quantization that fits *this* machine (e.g. Q8_0 on a big GPU, Q4_K_M
   on a laptop, IQ3 when tight), estimates memory (weights + KV cache +
   overhead) and generation speed (bandwidth ÷ bytes-per-token), and scores it
   on fit, speed, quality and popularity. The player sees ~6 picks with plain
   badges — **Recommended**, **Fastest**, **Smartest that fits** — and just types
   a number (Enter = recommended). By default only permissively licensed
   (Apache-2.0 / MIT) models are shown.
5. **One confirmation, then automatic** (`setup_flow.py`): "Here's what will
   happen: ① download the llama.cpp engine (~40 MB, MIT, from GitHub)
   ② download Qwen3 4B Q4_K_M (2.5 GB, Apache-2.0, from Hugging Face)
   ③ start it on your computer. OK? [Y/n]". Then:
   - **Managed llama.cpp (default)** — `runtime_install.py` downloads the
     official prebuilt `llama-server` for this OS/CPU/GPU from the
     ggml-org/llama.cpp GitHub releases (CUDA / Vulkan / Metal / CPU), verifies
     and unpacks it into the app's data folder; `download.py` fetches the GGUF;
     `backends/llamaserver.py` launches `llama-server` on a free localhost port
     and talks to its OpenAI-compatible API. No compilers, no admin rights.
   - **Ollama** (if the player already runs it, or `--backend ollama`) —
     `ollama pull hf.co/<repo>:<quant>` pulls the same GGUF from Hugging Face.
   - **llama-cpp-python** (`--backend llamacpp`, for tinkerers).
   - **Mock** (`--mock`) — scripted, offline, instant.
   If anything fails (no GPU driver, blocked network), the game explains in
   one sentence and falls back automatically (GPU build → CPU build; managed →
   Ollama if running → offer mock).
6. **Warm-up & speed test**: a tiny generation measures real tokens/second and
   reports it ("Your model is talking at ~18 tokens/sec!"). If it's painfully
   slow (< 3 tok/s) the game offers to switch to a smaller pick.
7. **Jev onboarding** (`onboarding.py`): "Do you want to enable Jev?" with a
   plain-language explanation. Yes → paste API key (hidden input), or "help me
   get one" (opens the TypeSafe website/docs in a browser, step-by-step), or
   back out to local-only at *any* prompt. Keys are validated with
   `GET /v1/models`. Saving the key to disk is opt-in only.
8. **The game** (`game.py`, `prompts.py`): the local LLM narrates a farcical
   "you're about to be late for work" intro and the first absurd challenge.
   Each round the player types how they'll get past it. A judge decides whether
   they made progress:
   - **Jev enabled** → one `POST /v1/systemone` call with three questions:
     a **Noul** (did they make progress?), a **Choice** (what kind of outcome?),
     and a **Score** (how creative was it, 0–4). The Noul decides progress.
     The answers are shown with "Learn" panels the first time each type appears.
   - **Local only** → the local LLM is asked for a small JSON verdict.
   Progress +1 on success. The LLM narrates the result and invents the next,
   ever-more-ridiculous challenge. Reaching **5** wins: the LLM narrates a
   triumphant arrival at work. Typing `quit` ends early.
9. **Review** (`review.py`): two *independent* yes/no questions — show the Jev
   request/response JSON per round? show the local model's exposed reasoning
   (chain-of-thought) per round? — plus an optional transcript export
   (JSON + Markdown, API key never included).
10. On exit, the managed `llama-server` process is always stopped.

## Package layout

```
src/gettowork/
  __init__.py        version
  __main__.py        `python -m gettowork` -> cli.main()
  types.py           shared dataclasses (read this first)
  ui.py              rich-based UI; all input/output goes through UI
  config.py          settings file + data dirs
  specs.py           hardware detection
  perf.py            bandwidth micro-benchmark + tokens/sec estimates
  catalog.py         curated seed models + the fit/ranking engine
  hf_discovery.py    live Hugging Face search, GGUF metadata, disk cache
  download.py        Hugging Face GGUF download (exact files / shards)
  runtime_install.py fetch + unpack official prebuilt llama.cpp binaries
  reasoning.py       split chain-of-thought from answers
  backends/
    __init__.py      exports + detect_backends()
    base.py          LLMBackend ABC, BackendError
    llamaserver.py   LlamaServerBackend (managed llama-server subprocess) — default
    ollama.py        OllamaBackend (HTTP to localhost:11434)
    llamacpp.py      LlamaCppBackend (llama-cpp-python, optional)
    mock.py          MockBackend (scripted, offline, deterministic)
  jev.py             Jev HTTP client + game questions + verdict parsing
  onboarding.py      Jev opt-in / API-key flow
  prompts.py         all LLM prompt text
  game.py            core game loop
  review.py          end-of-game review + transcript export
  setup_flow.py      hardware -> discovery -> pick -> install -> warm-up
  cli.py             argparse entry point
tests/               pytest, no network, no real models
```

Runtime deps: `rich` (MIT), `psutil` (BSD-3), `huggingface_hub` (Apache-2.0).
Optional: `llama-cpp-python` (MIT). HTTP to GitHub, llama-server, Ollama and
Jev uses only the Python standard library (`urllib.request`) so learners can
see exactly what is sent. Python ≥ 3.10. Must run on Windows, macOS, Linux.

Data locations (`config.py`): settings in `config_dir()`; models in
`models_dir()`; add `runtime_dir()` = `config_dir()/runtime` (llama.cpp
builds) and `cache_dir()` = `config_dir()/cache` (discovery cache). Both honour
`GETTOWORK_HOME`.

## Module contracts

Signatures below are binding. Private helpers are free-form.

### types.py, ui.py, config.py, backends/base.py
Already written — read them. Do not change existing signatures; additive
changes only if truly necessary (and then update this doc). `ModelEntry` and
`FitResult` have extra optional fields for live discovery (quant options,
exact files, downloads, speed estimate, badges...).

### specs.py
```python
def detect_specs(models_path: Path | None = None) -> SystemSpecs
def describe_specs(specs: SystemSpecs) -> list[tuple[str, str]]   # rows for a 2-col table
def friendly_summary(specs: SystemSpecs) -> str                     # 1–2 warm plain-English sentences
```
- psutil for RAM/CPU/disk (disk = free space of `models_path` or its nearest
  existing parent, defaulting to `config.models_dir()`).
- CPU name: `platform.processor()`, falling back to `/proc/cpuinfo` "model name"
  (Linux), `sysctl -n machdep.cpu.brand_string` (macOS), `wmic`/PowerShell
  best effort on Windows; never fail — "Unknown CPU". CPU flags: `/proc/cpuinfo`
  flags (avx, avx2, avx512f, f16c, fma), `sysctl hw.optional` on macOS (neon on
  arm64), best effort elsewhere.
- NVIDIA: `nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits`
  (MiB). AMD (Linux): `rocm-smi --showmeminfo vram --json` best effort; also
  detect AMD/Intel GPUs by name via `lspci` / Windows `Win32_VideoController`
  (VRAM unknown → 0, noted). Apple Silicon (`Darwin` + `arm64`): one
  GPUInfo(vendor="apple", vram_gb≈0.70 × RAM (0.75 if RAM ≥ 64 GB)),
  `unified_memory=True`. Vulkan loader present? (Linux: `libvulkan.so.1`
  findable via ctypes.util.find_library("vulkan"); Windows: vulkan-1.dll in
  System32) — record in notes/flags as "vulkan" for runtime selection.
- Every subprocess: timeout ≤ 5 s, list args (no shell), catch everything,
  append a note on failure. `detect_specs()` must never raise.
- Calls `perf.measure_ram_bandwidth()` and `perf.estimate_gpu_bandwidth()` to
  fill the bandwidth fields (skippable via parameter `benchmark: bool = True`).

### perf.py
```python
def measure_ram_bandwidth(budget_s: float = 0.3) -> float | None   # GB/s via large bytearray/memoryview copies; None on failure
def estimate_gpu_bandwidth(gpu: GPUInfo) -> float | None           # rough GB/s by vendor + VRAM tier (+ name keywords); documented as a guess
def estimate_tokens_per_s(specs: SystemSpecs, *, active_gb: float, placement: str, offload_fraction: float = 1.0) -> float
SPEED_EXPLAINER: str   # Markdown: generation is memory-bandwidth bound; tok/s ≈ efficiency × bandwidth ÷ bytes read per token
```
- Tokens/s model: `eff × bandwidth / active_weight_bytes`, eff≈0.55 GPU,
  0.45 unified, 0.35 CPU (clamped by core count); partial offload = harmonic
  mix of GPU and CPU speed by offload fraction. For MoE use active params'
  share of the weights. Always label as an estimate.
- Micro-benchmark must stay under ~0.5 s and ~256 MB, and never raise.

### catalog.py
```python
MODEL_CATALOG: list[ModelEntry]          # curated seeds, ordered small -> large (offline fallback + trust bonus)
TRUSTED_PUBLISHERS: tuple[str, ...]      # e.g. ("unsloth", "bartowski", "ggml-org", "lmstudio-community", "Qwen", "microsoft", "mistralai", "HuggingFaceTB", "ibm-granite", "NousResearch")
PERMISSIVE_LICENSES: frozenset[str]      # {"apache-2.0", "mit"}
QUANT_BITS: dict[str, float]             # approx bits/weight per quant tag (Q8_0 8.5, Q6_K 6.6, Q5_K_M 5.7, Q4_K_M 4.8, IQ4_XS 4.3, Q3_K_M 3.9, IQ3_M 3.7, Q2_K 3.0, MXFP4 4.25, F16 16, BF16 16 ...)
QUANT_PREFERENCE: tuple[str, ...]        # best -> smallest acceptable: Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q4_K_S, Q3_K_M, IQ3_M (never below 3 bits by default)
def get_model(key: str) -> ModelEntry | None
def estimate_memory_gb(model: ModelEntry, context_tokens: int | None = None, *, weights_gb: float | None = None) -> float
def choose_quant(specs: SystemSpecs, model: ModelEntry) -> tuple[str, float] | None   # best (quant, size_gb) that fits, using model.quant_options (or model.quant/file_size_gb)
def evaluate_fit(specs: SystemSpecs, model: ModelEntry) -> FitResult
def rank_models(specs: SystemSpecs, catalog: list[ModelEntry] | None = None) -> list[FitResult]
def pick_shortlist(ranked: list[FitResult], n: int = 6) -> list[FitResult]   # diverse picks with badges
def recommend(specs: SystemSpecs, catalog: list[ModelEntry] | None = None) -> FitResult | None
MEMORY_FORMULA_EXPLAINER: str   # Markdown for a UI.teach panel
```
- Curated seeds: only permissively licensed (Apache-2.0 or MIT) GGUF repos
  from well-known publishers, small → large (Qwen3 0.6B/1.7B/4B/8B/14B/32B,
  Qwen3-30B-A3B MoE, SmolLM2 1.7B, Phi-4-mini (MIT), Mistral 7B Instruct v0.3,
  Mistral Small 3.2 24B, gpt-oss-20b with `ollama_ref="gpt-oss:20b"`). The seed
  list is a fallback and a "known good" bonus; live discovery is primary.
- Memory estimate: `weights = chosen quant size`; KV cache ≈
  `0.00012 GB × params_b × context_tokens/1024` (floor 0.05) — or computed
  exactly when layer/head metadata is available; `overhead = 0.6 GB` (+ ~0.3 GB
  compute buffer for GPU). Documented in MEMORY_FORMULA_EXPLAINER.
- Placement: gpu if fits VRAM (keep ~0.8 GB free); unified if Apple and fits
  usable unified memory; partial if VRAM ≥ 40% of need and RAM covers the
  rest; cpu if RAM (total − ~2.5 GB OS headroom) fits; none otherwise.
  Verdict thresholds on need/budget: ≤0.6 great, ≤0.85 ok, ≤1.0 tight, else
  no; "no" also if disk free < download + 1 GB (reason says so).
- Speed from `perf.estimate_tokens_per_s`; labels: ≥20 tok/s "fast",
  ≥8 "usable", ≥3 "slow", else "very slow".
- Score (documented in code): quality (log params × quant quality factor) +
  speed (penalise < 8 tok/s hard, < 3 tok/s disqualify from Recommended) +
  headroom + popularity (log downloads) + small bonus for curated/trusted,
  reasoning-capable, and instruction-tuned. `pick_shortlist`: Recommended
  (best score), Fastest (highest tok/s among great/ok), Smartest that fits
  (largest params with ≥ 5 tok/s), then fill with the next best distinct
  families; never two variants of the same base model.
- The heuristic is our own, MIT licensed, provided with no warranty.

### hf_discovery.py
```python
class DiscoveryResult: models: list[ModelEntry]; source: str  # "live" | "cache" | "curated"; fetched_at: float | None; notes: list[str]
def discover_models(*, api=None, cache_path: Path | None = None, ttl_hours: float = 72,
                    refresh: bool = False, offline: bool = False, allow_all_licenses: bool = False,
                    max_candidates: int = 40, timeout_s: float = 20.0, clock=time.time) -> DiscoveryResult
def entry_from_hub(info, files: list[tuple[str, int]]) -> ModelEntry | None   # pure; used by tests
def parse_quant(filename: str) -> str | None       # "Qwen3-4B-Q4_K_M.gguf" -> "Q4_K_M"; handles IQ*, MXFP4, F16/BF16, UD- prefixes, shards
def group_quant_files(files: list[tuple[str, int]]) -> dict[str, tuple[tuple[str, ...], int]]   # quant -> (files incl. all shards, total bytes)
DISCOVERY_EXPLAINER: str   # Markdown: what the Hub is, GGUF, how we filter, licenses, caching
```
- Uses `huggingface_hub.HfApi` (injectable `api` for tests):
  `list_models(filter="gguf", pipeline_tag="text-generation", author=<publisher>, sort="downloads", limit=…, expand=["gguf","cardData","tags","downloads","likes","lastModified","gated"])`
  for each trusted publisher (plus one global `filter="gguf"` query), then
  `list_repo_tree(repo, recursive=True, expand=False)` or
  `model_info(repo, files_metadata=True)` for real file sizes of the top
  `max_candidates` after pre-filtering. `ModelInfo.gguf` gives
  `{"total": <params>, "architecture": ..., "context_length": ...}` when present.
- Filters: instruction/chat models only (tags `conversational`, names with
  instruct/chat/-it, or known chat families like Qwen3 / gpt-oss / Phi-4-mini);
  exclude base/embedding/reranker/vision-projector-only/coder-only?(keep coder
  out: not great storytellers), "abliterated"/"uncensored"/NSFW/
  `not-for-all-audiences`; exclude gated unless nothing else; licenses
  permissive only unless `allow_all_licenses` (license read from `license:`
  tags or cardData; unknown license → excluded by default). De-duplicate by
  base model (prefer trusted publisher order, then downloads).
- ModelEntry from hub: key = repo id, display_name prettified from repo
  (strip "-GGUF", publisher), params from gguf.total (fallback: parse "4B",
  "0.6B", "30B-A3B" from the name, or infer from Q4 size), MoE active params
  from name ("A3B"), reasoning=True for known thinking families (qwen3,
  gpt-oss, deepseek-r1 distills, phi-4-mini-reasoning, magistral, …),
  ollama_ref "hf.co/<repo>:<quant>", license_url
  `https://huggingface.co/<repo>` (model card), quant_options from files.
- Cache: JSON at `cache_dir()/hf_models.json` with `fetched_at`; used when
  fresh, or when offline / the Hub fails (stale cache is fine with a note).
  If live fails and no cache: return curated seeds (`source="curated"`).
- Never raise for network errors; put a friendly note in `notes`.

### reasoning.py
```python
def split_reasoning(text: str) -> tuple[str, str | None]
```
Handles `<think>...</think>` (also `<thinking>`, `<reasoning>`), multiple
blocks, an unclosed leading `<think>` (everything after it is reasoning if no
close tag; answer empty), stray `</think>` with no opener (text before it is
reasoning), and gpt-oss/harmony style `<|channel|>analysis<|message|>...<|end|>
... <|channel|>final<|message|>...`. Returns (answer.strip(), reasoning or None).

### download.py
```python
class DownloadError(RuntimeError)
def pick_gguf_file(filenames: list[str], quant: str) -> str | None
def resolve_files(entry: ModelEntry, quant: str | None = None, *, hf_api=None) -> tuple[str, ...]   # exact file(s) incl. shards
def download_gguf(entry: ModelEntry, ui: UI, dest_dir: Path | None = None, *, quant: str | None = None,
                  hf_api=None, hf_download=None) -> Path          # returns path of the first shard / the file
def download_custom_gguf(repo_id: str, quant: str, ui: UI, dest_dir: Path | None = None,
                         *, hf_api=None, hf_download=None) -> Path
```
- Uses `entry.gguf_files` when present and matching the quant; otherwise lists
  the repo and uses `pick_gguf_file` (case-insensitive; `.gguf` only; skip
  `mmproj`; prefer exact quant; fallback order Q4_K_M, Q4_K_S, Q5_K_M, Q4_0,
  IQ4_XS, Q6_K, Q8_0, MXFP4, anything; prefer repo root). Downloads **all
  shards** of split models (`-00001-of-0000N`) into the same folder.
- `hf_hub_download(repo_id, filename, local_dir=...)` (injectable). Skip if the
  complete file already exists (size matches when known). Check disk space
  first (`shutil.disk_usage`). Clear errors: repo not found, no GGUF, network
  failure, gated repo (needs HF login: `hf auth login`), disk full.
- Before downloading, print file name(s), size, license, model page URL and
  destination, and remind that weights come from Hugging Face under the
  author's license.

### runtime_install.py
```python
class RuntimeInstallError(RuntimeError)
@dataclass
class RuntimeVariant: name: str  # "cuda-12.4", "cuda-13", "vulkan", "metal", "cpu", "rocm", ...
                      asset_patterns: tuple[str, ...]; needs_cudart: bool; gpu: bool
def plan_variants(specs: SystemSpecs) -> list[RuntimeVariant]   # ordered best -> safest (always ends with cpu)
def select_assets(assets: list[dict], variant: RuntimeVariant, os_name: str, arch: str) -> list[dict]   # pure; [] if no match
def fetch_releases(*, http=None, limit: int = 8) -> list[dict]   # GET https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=N (NOT /latest: builds are published as prereleases)
def ensure_llama_server(ui: UI, specs: SystemSpecs, *, variant: RuntimeVariant | None = None,
                        http=None, runtime_root: Path | None = None) -> tuple[Path, RuntimeVariant]   # path to llama-server executable
def installed_runtimes(runtime_root: Path | None = None) -> list[tuple[Path, str, str]]   # (exe, tag, variant)
RUNTIME_EXPLAINER: str   # Markdown: what llama.cpp / llama-server is, why prebuilt, what CUDA/Vulkan/Metal mean
```
- Official release asset names (from ggml-org/llama.cpp `.github/workflows/release.yml`):
  `llama-<tag>-bin-macos-arm64.tar.gz`, `llama-<tag>-bin-macos-x64.tar.gz`,
  `llama-<tag>-bin-ubuntu-x64.tar.gz`, `llama-<tag>-bin-ubuntu-arm64.tar.gz`,
  `llama-<tag>-bin-ubuntu-vulkan-x64.tar.gz`, `llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`
  (+ `cudart-llama-<tag>-bin-ubuntu-cuda-12.8-x64.tar.gz`), `…-ubuntu-cuda-13.4-x64…`,
  `llama-<tag>-bin-win-cpu-x64.zip`, `llama-<tag>-bin-win-cpu-arm64.zip`,
  `llama-<tag>-bin-win-vulkan-x64.zip`, `llama-<tag>-bin-win-cuda-12.4-x64.zip`
  (+ `cudart-llama-bin-win-cuda-12.4-x64.zip`), `llama-<tag>-bin-win-cuda-13.4-x64.zip`,
  `llama-<tag>-bin-win-rocm-*-x64.zip`. Tarballs contain a top-level
  `llama-<tag>/` folder; Windows zips are flat. Names drift over time → match
  with tolerant regexes on (os, accel, arch), never exact strings; pick the
  newest release that has a matching asset.
- Variant plan: macOS arm64 → metal build (the macos-arm64 asset); macOS x64 →
  cpu; Windows + NVIDIA → cuda-12.4 (driver ≥ 525-ish; if driver_version known
  and ≥ 580 prefer cuda-13) + cudart, then vulkan, then cpu; Windows AMD/Intel
  → vulkan, then cpu; Linux + NVIDIA → cuda-12.8 (+cudart) then vulkan (if
  loader present) then cpu; Linux AMD/Intel with vulkan loader → vulkan then
  cpu; else cpu. Windows arm64 → win-cpu-arm64.
- Download via urllib to a temp file with a `ui.download_progress` bar,
  verify size matches the asset `size` (and the GitHub `digest` sha256 field
  if present), extract safely (reject absolute paths / `..` / symlinks
  escaping the target — zip-slip/tar-slip protection; use
  `tarfile` `filter="data"` when available), flatten the top folder, `chmod +x`
  executables on POSIX, write a small `install.json` marker
  (tag, variant, asset names). Install dir:
  `runtime_dir()/llama.cpp/<tag>-<variant>/`. Reuse an existing install.
- Respect `GITHUB_TOKEN` env if set (higher rate limit) but never print it.
  GitHub API rate limit / offline → RuntimeInstallError with a friendly
  message suggesting Ollama or retry later.

### backends/
```python
# backends/__init__.py
from .base import LLMBackend, BackendError
from .llamaserver import LlamaServerBackend
from .ollama import OllamaBackend
from .llamacpp import LlamaCppBackend
from .mock import MockBackend
def detect_backends() -> dict[str, tuple[bool, str]]   # {"managed": (ok, why), "ollama": (ok, why), "llamacpp": (ok, why)}

# llamaserver.py  — the default, fully automatic backend
class LlamaServerBackend(LLMBackend):
    name = "llamacpp-server"
    def __init__(self, entry: ModelEntry | None = None, *, specs: SystemSpecs | None = None,
                 model_path: Path | None = None, quant: str | None = None, n_ctx: int = 4096,
                 server_exe: Path | None = None, http=None, popen=None, port: int | None = None)
    def is_available(self) -> tuple[bool, str]      # True unless platform totally unsupported
    def prepare(self, ui, entry=None) -> None       # ensure_llama_server + download_gguf + start + wait healthy (+ GPU->CPU fallback)
    def chat(...) -> LLMResult                      # POST /v1/chat/completions
    def benchmark(self, ui=None) -> float | None    # tokens/s from a short generation (use `timings.predicted_per_second` if returned, else usage/elapsed)
    def close(self) -> None                         # terminate the subprocess (also registered with atexit); idempotent
    @property model_label
```
  - Launch: `[exe, "-m", gguf, "--host", "127.0.0.1", "--port", str(port),
    "-c", str(n_ctx), "--reasoning-format", "deepseek", "--no-webui", "-np", "1"]`
    (`-ngl` defaults to auto / `--fit` on in current builds; on the CPU variant
    pass `-ngl 0`). Free port chosen via a bound socket. stdout/stderr to a log
    file in `runtime_dir()/logs/` (show its tail on failure). Windows:
    `creationflags=CREATE_NO_WINDOW`; Linux: `LD_LIBRARY_PATH` += exe dir.
    If the process exits with an unknown-argument error, retry once with the
    minimal args (`-m`, `--host`, `--port`, `-c`).
  - Health: poll `GET /health` until 200 `{"status":"ok"}` (503 = still
    loading), up to ~180 s with a friendly spinner ("Waking up the model…");
    if the process dies, read the log tail: CUDA/Vulkan errors → retry with
    the next variant from `plan_variants` (download it) → finally CPU.
  - Chat: `{"messages", "temperature", "max_tokens", "stream": false}` plus
    `"response_format": {"type": "json_object"}` when json_mode. Reasoning =
    `choices[0].message.reasoning_content` if present, else
    `split_reasoning(content)`. Empty answer after stripping → retry once with
    `"chat_template_kwargs": {"enable_thinking": false}` and more max_tokens.
  - Timeouts: chat 300 s. Every error surfaces as BackendError with a
    friendly message.
```python
# ollama.py
class OllamaBackend(LLMBackend):
    name = "ollama"
    def __init__(self, model: str, host: str | None = None, *, http=None, think: bool | None = None)
    def is_available(self) -> tuple[bool, str]          # GET /api/version
    def has_model(self) -> bool                         # GET /api/tags
    def prepare(self, ui, entry=None) -> None           # POST /api/pull {"model":..., "stream": true}; progress from streamed JSON lines (completed/total)
    def chat(...) -> LLMResult                          # POST /api/chat, stream false, options {temperature, num_predict, num_ctx}; format "json" when json_mode
    @property model_label
```
  - host default: env OLLAMA_HOST (normalize "0.0.0.0:11434" / no scheme) or
    http://127.0.0.1:11434; `http` injectable for tests.
  - Thinking: send `"think": true` when `think` is True (default: the catalog
    entry's `reasoning`). If Ollama returns an error mentioning "think", retry
    once without it. Reasoning = `message.thinking` if present, else
    `split_reasoning(message.content)`. Empty answer → retry once with
    `"think": false` and larger `num_predict`. Timeouts: chat 300 s, pull
    streaming per-read 600 s.
```python
# llamacpp.py
class LlamaCppBackend(LLMBackend):
    name = "llamacpp"
    def __init__(self, model_path: Path | None = None, entry: ModelEntry | None = None,
                 *, n_ctx: int = 4096, n_gpu_layers: int = -1, llama_factory=None)
```
  - Lazy-import `llama_cpp`; `Llama(model_path, n_ctx, n_gpu_layers, verbose=False)`;
    `create_chat_completion(... response_format={"type":"json_object"} when json_mode)`;
    GPU load failure → retry n_gpu_layers=0; same empty-answer retry (append
    " /no_think" for Qwen3).
```python
# mock.py
class MockBackend(LLMBackend):
    name = "mock"
    def __init__(self, seed: int = 0, *, think: bool = True)
```
  - Deterministic, offline. Recognises what it's being asked via the
    `TASK: <purpose>` line the game places in the system prompt (purpose ∈
    {intro, outcome, judge, victory, ending_quit}). Farcical text; `judge`
    returns JSON `{"made_progress": ..., "explanation": ...}` (progress = plan
    has ≥ 4 words and isn't "nothing"/"give up"). Emits fake `reasoning`.
    Intro/outcome always contain a `CHALLENGE:` line.

All backends that own resources implement `close()` (no-op default is fine
for others); cli.py calls it in a `finally`.

### jev.py
Wire format (from TypeSafe's official MIT-licensed SDK, `typesafe-sdk` 0.7.1):
- Base URL `https://api.typesafe.ai` (env `TYPESAFE_BASE_URL`), model
  `jev-latest` (env `TYPESAFE_DEFAULT_MODEL`), key env `TYPESAFE_API_KEY`.
- `POST /v1/systemone`, headers `Authorization: Bearer <key>`,
  `Content-Type: application/json`, `Accept: application/json`.
  Body: `{"state": <str|object|array>, "model": "jev-latest", "questions": {name: question}}`.
  - Noul: `{"type":"noul","instructions":..., "criteria": {"true": ..., "false": ...}}` (criteria optional)
  - Choice: `{"type":"choice","instructions":..., "criteria": {"label": "description"|null, ...}}` (≤255 options)
  - Score: `{"type":"score","instructions":..., "criteria": ["level 0 desc", "level 1 desc", ...]}` (non-empty; index = score)
- Response: `{"model": str, "answers": {name: answer}, "usage": {"input_tokens": int, "output_tokens": int}}`
  - Noul answer: `{"type":"noul","noul": 0.98}` — probability of yes; no confidence field.
  - Choice answer: `{"type":"choice","choice":"label","confidence":0.9,"probabilities":{"label":0.8,...}}`
  - Score answer: `{"type":"score","score":1.7,"confidence":0.9,"legend":{"0":"...",...},"probabilities":{"0":0.1,...}}`
- `GET /v1/models` → `{"models":[{"name","description","release_date"}]}` (key validation).
- Errors: non-2xx; body may have `error` (str or {message}), `message`, or
  `detail` (str or list of {loc,msg,type}). 401/403 = bad key, 402 = billing,
  422 = validation, 429 = rate limit (honour `retry-after`), 5xx = retry.

```python
JEV_DEFAULT_BASE_URL = "https://api.typesafe.ai"
JEV_DEFAULT_MODEL = "jev-latest"
JEV_API_KEY_ENV = "TYPESAFE_API_KEY"
JEV_HOME_URL = "https://typesafe.ai"
JEV_DOCS_URL = "https://docs.typesafe.ai/"
class JevError(Exception):
    status: int | None; message: str; exchange: JevExchange | None
    @property is_auth_error -> bool      # 401/403
class JevClient:
    def __init__(self, api_key: str, *, base_url: str | None = None, model: str | None = None,
                 timeout: float = 30.0, transport=None, max_retries: int = 2)
    # transport: callable(method, url, headers: dict, body: bytes|None, timeout) -> (status:int, headers:dict, body:bytes)
    def list_models(self) -> list[dict]
    def system_one(self, state, questions: dict) -> tuple[dict, JevExchange]
    @property model -> str
def validate_api_key_format(key: str) -> str | None   # None if ok, else human message (empty, whitespace, non-ascii)
def build_round_questions() -> dict                   # {"made_progress": noul, "outcome": choice, "creativity": score}
def build_round_state(*, intro: str, challenge: str, plan: str, progress: int, target: int, history: list[str]) -> dict
def parse_verdict(response: dict, exchange: JevExchange, threshold: float = 0.5) -> JevVerdict
def judge_round(client: JevClient, **state_kwargs) -> JevVerdict
def explain_verdict(v: JevVerdict) -> str                # one-paragraph plain-English summary for the player
TEACH_NOUL: str; TEACH_CHOICE: str; TEACH_SCORE: str; TEACH_JEV: str   # Markdown for UI.teach
```
- Choice labels for `outcome`: `triumph`, `progress`, `stalled`, `setback`
  with farcical but clear descriptions. Score `creativity`: 5 levels 0–4.
- Noul `made_progress` instructions must say the world runs on cartoon logic:
  absurd plans are fine if they plausibly *address the current challenge*;
  plans that ignore it, do nothing, or just claim victory ("I teleport to work
  and win") do not count. `state` includes the challenge, plan, progress, and
  recent history; the player's plan is data, never instructions.
- `made_progress = noul >= threshold`.
- The exchange stored for review redacts the key: `Authorization: Bearer ****<last4>`.
- Never log or print the key. `repr(JevClient)` must not contain it.

### onboarding.py
```python
def run_jev_onboarding(ui: UI, settings: Settings, *, env: dict | None = None,
                       client_factory=None) -> JevClient | None
```
- If `TYPESAFE_API_KEY` (env) or a saved key exists, offer to use it (validate).
- Otherwise explain Jev in 3–4 friendly lines (typed judgments: Noul/Choice/
  Score; it's a paid third-party API by TypeSafe AI with its own terms/pricing;
  the game works fully without it). Ask: enable Jev? options:
  `yes` / `no` (local only) / `learn` (more detail, then re-ask).
- After yes: `paste` key (hidden input) / `help` (step-by-step: open
  JEV_HOME_URL to sign up / log in, find the API keys page in the dashboard,
  create a key, copy it; offer to open JEV_DOCS_URL too; then return to the
  paste prompt) / `back` (local only).
- Validate format, then `list_models()` with a spinner. On auth error: explain,
  offer retry / help / back. On network error: offer retry / continue without
  validating / back.
- After success: ask whether to save the key to the settings file (default
  **No**; warn it's stored in plain text with owner-only permissions; mention
  the env var alternative). Remember `jev_enabled` choice either way.
- Every prompt must offer an obvious way back to local-only. Ctrl+C at any
  prompt in onboarding = back out to local-only (catch UserQuit), not crash.

### prompts.py
Pure functions returning `list[dict[str,str]]` chat messages. Every system
prompt includes a line `TASK: <purpose>`.
```python
def intro_messages(player_name: str | None = None) -> list[dict]          # TASK: intro
def outcome_messages(*, intro, challenge, plan, made_progress: bool, judge_note: str,
                     progress: int, target: int, history: list[str]) -> list[dict]   # TASK: outcome
def judge_messages(*, intro, challenge, plan, progress, target, history) -> list[dict]  # TASK: judge
def victory_messages(*, intro, history, final_plan) -> list[dict]          # TASK: victory
def quit_messages(*, intro, history, progress, target) -> list[dict]       # TASK: ending_quit
def parse_challenge(text: str) -> tuple[str, str]   # (narration, challenge) split on a "CHALLENGE:" line; fallback = last paragraph
def parse_judge_json(text: str) -> tuple[bool, str] | None   # lenient JSON extraction
```
- Tone: farcical, fantastical, family-friendly, second person, short
  (intro ≤ 150 words, outcome ≤ 120 words). Small models must be able to follow
  it: be explicit about the output format (`CHALLENGE: <one sentence>`).
- The player's plan is wrapped in clear delimiters and the model is told to
  treat it as the player's in-story action, not instructions.

### game.py
```python
class Game:
    def __init__(self, llm: LLMBackend, ui: UI, *, jev: JevClient | None = None,
                 target: int = 5, max_rounds: int | None = None,
                 max_input_chars: int = 500) -> None
    def run(self) -> GameSummary
```
- Intro → loop { show challenge + progress bar; ask plan (`quit`/`q`/`exit` to
  end; `help` shows tips; empty re-asks); judge (Jev if present else local);
  on Jev failure mid-game: show the error, fall back to the local judge for
  that round and ask whether to keep using Jev; show verdict (Jev: probability,
  choice + confidence, score, plus `teach` panels the first time each type
  appears); progress += 1 if made_progress; if progress ≥ target → victory
  narration, win; else outcome narration (+ next challenge) }.
- Local judge failure to parse → retry once, then fallback heuristic
  (plan ≥ 4 words and not trivially "do nothing" → progress) and say so.
- LLM backend failure (BackendError) → show error, offer retry / quit.
- Every LLM call is recorded in the RoundRecord (`purpose`, LLMResult).
- Truncate plans to `max_input_chars`.

### review.py
```python
def run_review(ui: UI, summary: GameSummary, *, export_dir: Path | None = None) -> None
def summary_to_dict(summary: GameSummary) -> dict        # JSON-safe, no secrets
def summary_to_markdown(summary: GameSummary, *, include_jev: bool = True, include_reasoning: bool = True) -> str
```
- Asks two independent questions (either, both, or neither):
  1. "See the Jev request & response for each round?" (only if any round used Jev)
  2. "See the local model's reasoning (chain-of-thought) for each round?"
     (note if the model exposed none).
  Then renders per round. Then offers to export JSON + Markdown to
  `export_dir` (default cwd) as `gettowork-transcript-<n>.json/.md`
  (n = first unused integer; no clocks needed).

### setup_flow.py
```python
@dataclass
class SetupResult: backend: LLMBackend; entry: ModelEntry | None; specs: SystemSpecs; fit: FitResult | None
def run_setup(ui: UI, settings: Settings, *, args) -> SetupResult | None   # None = user quit
```
Super-friendly, few decisions:
1. Returning player with saved settings whose model file / runtime still
   exists → "Welcome back! Play with <model> again? [Y/n]" → start it
   (warm-up) and return.
2. "Let me take a look at your computer…" spinner → `specs.friendly_summary`
   + compact details table; `teach` panel (condensed memory + speed
   explainers) — offer "details?" rather than dumping walls of text.
3. "Searching Hugging Face for models that fit your computer…" spinner →
   `discover_models` → `rank_models` → `pick_shortlist` → a clean numbered
   table: #, badge, model, download size, est. speed ("~25 tokens/s"),
   license, one-line why. Enter = Recommended. Extra options: `more`
   (show the full ranked list), `refresh` (re-query the Hub), `custom`
   (paste any HF GGUF repo id), `mock` (play offline without a model).
4. One confirmation screen listing exactly what will be downloaded (engine
   size/source/license if not yet installed; model files/size/source/license)
   and where it will be stored. Y → everything automatic with progress bars.
5. Backend choice is automatic: `--backend auto` → managed llama-server;
   if its install fails and Ollama is running → Ollama (explain); else offer
   retry / mock / quit with friendly guidance (Ollama download link
   https://ollama.com/download, `pip install llama-cpp-python`).
6. Warm-up + `benchmark()` → "Your model is talking at ~N tokens/sec"; if
   < 3 tok/s offer to go back and pick a faster model.
7. Save settings (backend, model repo/quant/path, server exe) for next time.

### cli.py
```python
def main(argv: list[str] | None = None) -> int
```
Flags: `--mock` (offline scripted model), `--backend {auto,managed,ollama,llamacpp}`
(default auto), `--model REPO_OR_KEY` (skip the picker), `--quant TAG`,
`--ollama-model TAG`, `--gguf PATH` (use a local GGUF with the managed
runtime), `--list-models` (print the ranked shortlist for this machine and
exit), `--specs` (print hardware + bandwidth and exit), `--refresh-models`
(ignore discovery cache), `--offline` (no network: cache or curated seeds),
`--all-licenses` (include non-permissive licenses in discovery; shows each
license prominently), `--no-jev`, `--target N` (default 5), `--reset` (forget
saved settings), `--export-dir DIR`, `--debug` (tracebacks), `--version`.
Ctrl+C anywhere exits cleanly with a friendly line (exit code 130); the
backend is always `close()`d. Flow: banner → setup → Jev onboarding (skipped
by `--no-jev`) → Game.run() → run_review().

## Testing rules
- `pytest` only, no network, no real models, no real browser. Use MockBackend,
  fake Jev transports, fake `hf_api`/`hf_download`, and `UI(console=Console(file=io.StringIO()), input_fn=scripted)`.
- Each module has its own `tests/test_<module>.py`.

## Legal / safety rules for the codebase
- No model weights are bundled. Models are downloaded by the player from
  Hugging Face under each model's own license; the game shows that license.
- Curated seeds and default discovery results: Apache-2.0 / MIT only. `--all-licenses` shows others, always with the license visible.
- The llama.cpp engine (MIT) is downloaded from the official ggml-org GitHub releases at the player's request; it is not bundled.
- The hardware-fit heuristic is original code in this repo (MIT, no warranty).
- Not affiliated with TypeSafe AI, Hugging Face, Ollama, or any model author;
  names are used only to identify their products.
- API keys: never printed, never logged, never exported, redacted in review.
