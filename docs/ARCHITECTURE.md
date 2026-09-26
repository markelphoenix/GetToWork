# Get To Work — Architecture & Module Contracts

This document is both a guide for learners and the **binding contract** the
modules are built against. If code and this document disagree, fix one of them.

## The player's journey

The design goal: **the player only ever picks a model.** Everything else —
finding models that fit, getting the engine ready, downloading the weights,
starting the model — is automatic, explained in friendly language, and
reversible.

There are two front ends and one game. Players launch the **built game**
(Steam, or a double-clicked test build): its windowed program runs
`launcher.gui_main()` → `gui.app.run_gui()`, which opens the game's own Tk
window and runs the unchanged `cli.main()` on a worker thread, talking to the
window through `UI`. Developers run `gettowork` in a terminal (`cli.main()`
directly). The built game ships the llama.cpp engine inside it
(`distribution.py`), so it never downloads programs; a developer copy
downloads the engine as before. See the
[Distribution section](#distribution-the-game-window-the-built-game-and-safety-as-built)
below and [docs/DISTRIBUTION.md](DISTRIBUTION.md) for the build itself.

1. The game launches → banner. On the very first launch, a short note says
   the story is written live by an AI and how to report problems
   (`notices.AI_CONTENT_NOTICE`, shown once). Returning players: "Welcome
   back! Play with Qwen3 4B again? [Y/n]" skips straight to the game.
2. **Hardware check** (`specs.py` + `perf.py`): OS, CPU (+ SIMD flags), RAM,
   GPU(s)/VRAM, Apple unified memory, free disk, plus a quick (< 1 s)
   multi-core memory *read* benchmark (~512 MB, bigger than any CPU cache). Summarised in plain English ("16 GB of RAM and an NVIDIA
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
   badges — **Recommended**, **Fastest comfortable fit**, **Smartest at a playable pace** — and just types
   a number (Enter = recommended). By default only permissively licensed
   (Apache-2.0 / MIT) models are shown. Models whose chat template *forces*
   thinking (QwQ, DeepSeek-R1 distills, Phi-4-reasoning...) can't be asked for
   a quick answer, so they never reach this short menu (they stay under
   `more`, with a warning).
5. **One confirmation, then automatic** (`setup_flow.py`): "Here's what will
   happen: ① download the llama.cpp engine (~40 MB, MIT, from GitHub)
   ② download Qwen3 4B Q4_K_M (2.5 GB, Apache-2.0, from Hugging Face)
   ③ start it on your computer. OK? [Y/n]". In the built game step ① reads
   "Built into the game (llama.cpp <tag>, Vulkan + CPU) - nothing to
   download". Then:
   - **Managed llama.cpp (default)** — the built game uses the engine builds
     it ships with (`distribution.py`, read-only). A developer copy's
     `runtime_install.py` downloads the
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
   one sentence and falls back (GPU build → CPU build; a model architecture
   the installed engine doesn't know → update the engine; managed → Ollama if
   running, *after asking*, handing it the already-downloaded file instead of
   downloading again → offer mock). A computer no official build can run on
   (too old a Linux/macOS, a build already marked unusable) is recognised
   *before* anything is downloaded, and every fresh engine is test-run
   (`llama-server --version`) before the model download starts. A GPU build
   that passes start-up but crashes on real work (or hangs setting up the
   device) falls back the same way.
6. **Warm-up & speed test**: a tiny generation measures real tokens/second and
   reports it ("Your model is talking at ~18 tokens/sec!"). If it's painfully
   slow (< 3 tok/s) the game offers to switch to a smaller pick.
7. **Jev onboarding** (`onboarding.py`): "Do you want to enable Jev?" with a
   plain-language explanation, including a privacy line saying exactly what
   Jev receives (the typed plan, the challenge, a story summary, progress) and
   where it goes (TypeSafe AI's host). Yes → paste API key (hidden input; if
   the window can't hide input the player is told first and pointed to
   `TYPESAFE_API_KEY`), or "help me get one" (opens the TypeSafe website/docs
   in a browser, step-by-step), or back out to local-only at *any* step
   (including Ctrl+C while the key is being checked). Keys are validated with
   `GET /v1/models`. Saving the key to disk is opt-in only.
8. **The game** (`game.py`, `prompts.py`): the local LLM narrates a farcical
   "you're about to be late for work" intro (no obstacle yet). **Round 1 asks
   "How do you plan to get to work?"** (on foot, bike, bus, broomstick...); a
   real way of travelling counts as the first step, and the LLM then narrates
   them setting off and invents the first obstacle to fit that choice. Every
   later round shows an obstacle and the player types how they'll get past it.
   A judge decides whether they made progress:
   - **Jev enabled** → one `POST /v1/systemone` call with three questions:
     a **Noul** (did they make progress?), a **Choice** (what kind of outcome?),
     and a **Score** (how creative was it, 0–4). The Noul decides progress.
     The answers are shown with "Learn" panels the first time each type appears.
   - **Local only** → the local LLM is asked for a small JSON verdict.
   Progress +1 on success. The LLM narrates the result and invents the next,
   ever-more-ridiculous challenge (the absurdity level comes from the progress,
   and the last step is always the finale at the office). Reaching **5** (the
   commute plus four obstacles) wins: the LLM narrates a triumphant arrival at
   work. Typing `quit` ends early.
   Every piece of model text is checked by the family-friendly filter
   (`safety.py`) before it is shown, and every typed plan before it reaches
   the model or Jev (see the Distribution section).
9. **Review** (`review.py`): two *independent* yes/no questions — show the Jev
   request/response JSON per round? show the local model's exposed reasoning
   (chain-of-thought) per round? — each asked only when there is something to
   show (plus the local referee's JSON verdicts when it didn't think out loud),
   then an optional transcript export (JSON + Markdown, API key never included).
10. **"Play again? [Y/n]"** (only when a person is playing): a new game with the
    same running model and Jev client, no setup. After a game with the pretend
    model picked from the menu the question is a menu - play again with it,
    **Pick a real AI model** (back to setup's model menu) or quit.
11. On exit, the managed `llama-server` process is always stopped.

## Package layout

```
src/gettowork/
  __init__.py        version (the one place it is set; pyproject reads it)
  __main__.py        `python -m gettowork` -> cli.main()
  launcher.py        gui_main(): the windowed entry (Steam / double-click / `gettowork-gui`)
  gui/
    __init__.py      exports run_gui, GuiBridge, TerminalBuffer
    terminal.py      TerminalBuffer: pure-Python ANSI/VT screen model (no Tk)
    bridge.py        GuiBridge: thread-safe game thread <-> window plumbing (no Tk)
    app.py           run_gui(), GameWindow: the Tk window
  assets/icon.png    window icon (drawn by packaging/make_icon.py)
  distribution.py    built game or developer copy? reads distribution.json
  types.py           shared dataclasses (read this first)
  ui.py              rich-based UI; all input/output goes through UI
  config.py          settings file + data dirs (+ --models-dir, command_name())
  crashlog.py        write_crash_report(): logs/crash.txt (game errors), logs/gui-crash.txt (window)
  specs.py           hardware detection
  perf.py            bandwidth micro-benchmark + tokens/sec estimates
  catalog.py         curated seed models + the fit/ranking engine
  hf_discovery.py    live Hugging Face search, GGUF metadata, disk cache
  download.py        Hugging Face GGUF download (exact files / shards)
  runtime_install.py fetch + unpack official prebuilt llama.cpp binaries
  tls.py             HTTPS trust store (truststore -> certifi -> default) + cert-error detection
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
  safety.py          family-friendly filter (check_text / soften / check_player_input)
  safety_terms.py    its word lists, ROT13-scrambled
  notices.py         AI_CONTENT_NOTICE (first launch) + STEAM_AI_DISCLOSURE (store page)
  game.py            core game loop
  review.py          end-of-game review + transcript export
  setup_flow.py      hardware -> discovery -> pick -> install -> warm-up
  cli.py             argparse entry point
packaging/           PyInstaller spec, entry scripts, fetch_engine / collect_licenses /
                     assemble / make_icon, smoke_test.sh, steam/ (see DISTRIBUTION.md)
tests/               pytest, no network, no real models
```

Runtime deps: `rich` (MIT), `psutil` (BSD-3), `huggingface_hub` (Apache-2.0),
plus the standard library's `tkinter` (Tcl/Tk, BSD-style) for the window only -
imported lazily, so the terminal version works on a Python without Tk.
Optional: `llama-cpp-python` (MIT). HTTP to GitHub, llama-server, Ollama and
Jev uses only the Python standard library (`urllib.request`) so learners can
see exactly what is sent. Python ≥ 3.10. Must run on Windows, macOS, Linux
(including Steam Deck).

Data locations (`config.py`): settings in `config_dir()`; models in
`models_dir()`; add `runtime_dir()` = `config_dir()/runtime` (llama.cpp
builds) and `cache_dir()` = `config_dir()/cache` (discovery cache). Both honour
`GETTOWORK_HOME`. On Windows `config_dir()` is `%LOCALAPPDATA%\GetToWork`
(multi-GB models must not ride along in a roaming profile); a folder an older
version created in `%APPDATA%\GetToWork` keeps being used.

## Module contracts

Signatures below are binding. Private helpers are free-form.

### types.py, ui.py, config.py, backends/base.py
Already written — read them. Do not change existing signatures; additive
changes only if truly necessary (and then update this doc). `ModelEntry` and
`FitResult` have extra optional fields for live discovery (quant options,
exact files, downloads, speed estimate, badges...). `GPUInfo.driver_version`
(optional, e.g. "550.54.14" from nvidia-smi) feeds the CUDA 12/13 choice in
`runtime_install.py`.
*As built (integration):* `UI.confirm` escapes its `[y/N]` hint (rich would
otherwise read it as a style tag and hide it); `UI.download_progress` keeps
one line at 80 columns (long file names are shortened with "…"); additive
`ui.make_stream_safe(stream)` makes a non-UTF-8 stdout (legacy Windows code
pages, output redirected to a cp1252 file) print plain look-alikes ("→" → "->",
"✓" → "OK") instead of raising UnicodeEncodeError — cli.py applies it.

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
- RAM/VRAM are GiB rounded to 0.1; `arch` is normalised to `x86_64`/`arm64`;
  `disk_free_gb = -1.0` means "unknown" (the fit engine then skips the disk
  check); a found Vulkan loader is recorded as `"vulkan"` in `cpu_flags`
  (`specs.has_vulkan(specs)`); integrated GPUs are listed with `vram_gb=0`.

### perf.py
```python
def measure_ram_bandwidth(budget_s: float = 0.3, threads: int | None = None) -> float | None   # GB/s *read*, all cores, median of 3 rounds; None on failure
def estimate_gpu_bandwidth(gpu: GPUInfo) -> float | None           # rough GB/s by vendor + VRAM tier (+ name keywords); documented as a guess
def estimate_tokens_per_s(specs: SystemSpecs, *, active_gb: float, placement: str, offload_fraction: float = 1.0) -> float
SPEED_EXPLAINER: str   # Markdown: generation is memory-bandwidth bound; tok/s ≈ efficiency × bandwidth ÷ bytes read per token
```
- Tokens/s model: `eff × bandwidth / active_weight_bytes`, eff≈0.55 GPU,
  0.45 unified, 0.35 CPU (clamped by core count); partial offload = harmonic
  mix of GPU and CPU speed by offload fraction. For MoE use active params'
  share of the weights. Always label as an estimate.
  *As built:* re-calibrated against real llama.cpp numbers (see perf.py) to
  `seconds/token = active_gb / (eff × bandwidth) + overhead` with eff 0.73 GPU,
  0.85 unified, 0.70 CPU (relative to the *measured* single-thread copy
  bandwidth; × core-count and no-AVX2 factors) and overhead 1.3 / 5 / 2 ms.
  Extra helpers: `primary_gpu(specs)`, `bandwidth_for(specs, placement) ->
  (GB/s, source)`, `speed_label(tps)`.
- Micro-benchmark must stay well under a second, use at most 1/8 of the free
  RAM (~512 MB, ≥ 32 MB per thread, so neither the CPU cache nor memcpy's
  store tricks decide the number), and never raise. *Round 3:* it *reads*
  (libc `memchr` over each thread's buffer, GIL-free via ctypes; fallback:
  slice copies into a cached scratch buffer), because token generation reads;
  `last_benchmark_note` flags a shrunk test on low-RAM machines; readings
  are clamped to `MAX_PLAUSIBLE_RAM_GBS` (460).

### catalog.py
```python
MODEL_CATALOG: list[ModelEntry]          # curated seeds, ordered small -> large (offline fallback + trust bonus)
TRUSTED_PUBLISHERS: tuple[str, ...]      # e.g. ("unsloth", "bartowski", "ggml-org", "lmstudio-community", "Qwen", "microsoft", "mistralai", "HuggingFaceTB", "ibm-granite", "NousResearch")
PERMISSIVE_LICENSES: frozenset[str]      # {"apache-2.0", "mit"}
QUANT_BITS: dict[str, float]             # approx bits/weight per quant tag (Q8_0 8.5, Q6_K 6.6, Q5_K_M 5.7, Q4_K_M 4.8, IQ4_XS 4.3, Q3_K_M 3.9, IQ3_M 3.7, Q2_K 3.0, MXFP4 4.25, F16 16, BF16 16 ...)
QUANT_PREFERENCE: tuple[str, ...]        # best -> smallest acceptable: Q8_0, Q6_K, Q5_K_M, Q4_K_M, IQ4_XS, Q4_K_S, Q3_K_M, IQ3_M (never below 3 bits by default)
def get_model(key: str) -> ModelEntry | None
def estimate_memory_gb(model: ModelEntry, context_tokens: int | None = None, *, weights_gb: float | None = None) -> float
def choose_quant(specs: SystemSpecs, model: ModelEntry, *, downloaded: Downloaded | None = None) -> tuple[str, float] | None   # best (quant, size_gb) that fits, using model.quant_options (or model.quant/file_size_gb); an already-downloaded quant that fits wins
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
  *As built:* KV is exact (`2 × layers × kv_heads × head_dim × 2 B × ctx`)
  for families in a known-shapes table (Qwen3/2.5, gpt-oss, Mistral, Phi-4-mini,
  SmolLM2/3, Granite); otherwise `(0.1 + 0.006 × params_b) GB per 1024 tokens`
  (active params for MoE) — the formula above under-estimated real KV caches
  ~100×. Context is capped at `native_context`. `choose_quant` also requires
  disk space and only upgrades above ~4-bit if the model stays ≥ 20 tok/s and
  on the same placement. FitResult for placement "none" has
  `est_tokens_per_s=None`, `est_speed="n/a"`. Shortlist fillers must run at
  ≥ 3 tok/s and prefer new families only within 12 score points of the best.
  Extra public helpers:
  `is_permissive`, `quant_bits`, `quant_quality`, `estimate_quant_size_gb`,
  `kv_cache_gb`, `score_fit`, `explain_fit(specs, fit) -> Markdown`.
- Placement: gpu if fits VRAM (keep ~0.8 GB free); unified if Apple and fits
  usable unified memory; partial if VRAM ≥ 40% of need and RAM covers the
  rest (*Round 4:* also at ≥ 15% when the CPU-only plan would be tight, and at
  any share when only VRAM + RAM together fit - llama.cpp's auto-fit uses the
  card anyway); cpu if RAM (total − ~2.5 GB OS headroom) fits; none otherwise.
  *Round 4:* a non-Apple split is graded on its RAM side (spill ÷ RAM budget,
  never better than "ok" because the card is full); the Recommended tiers
  that need a comfortable fit only take a split with ≥ 75% on the GPU
  (`SPLIT_RECOMMENDED_MIN_GPU_SHARE`). `FitResult.gpu_share` and
  `FitResult.shares_system_ram` (cpu, unified, Apple split) record this.
  *Round 3:* on Apple Silicon an overflow past the unified budget is a Metal
  "partial" (GPU share at unified speed, the rest at CPU speed, within RAM −
  2.5 GB) - never a separate "cpu" plan. A partial plan fills the card's
  memory by definition, so its verdict is at best "tight".
  Verdict thresholds on need/budget: ≤0.6 great, ≤0.85 ok, ≤1.0 tight, else
  no; "no" also if disk free < download + 1 GB (reason says so).
- Speed from `perf.estimate_tokens_per_s`; labels: ≥20 tok/s "fast",
  ≥8 "usable", ≥3 "slow", else "very slow".
- Score (documented in code): quality (log params × quant quality factor) +
  speed (penalise < 8 tok/s hard, < 3 tok/s disqualify from Recommended) +
  headroom + popularity (log downloads) + small bonus for curated/trusted,
  reasoning-capable, and instruction-tuned. `pick_shortlist`: Recommended
  (best score), Fastest comfortable fit (highest turn tok/s among great/ok;
  near-ties within 5% go to the better score), Smartest at a playable pace
  (most quality with ≥ 5 turn tok/s; never a tight fit in system RAM - cpu,
  unified, or a split - nor a last-tier quant), then fill with the next best distinct
  families; never two variants of the same base model.
  *Round 3:* within a quant tier the ladder prefers the fastest home (gpu /
  unified, then partial, then cpu) and only takes a slower one for ≥ 2%
  quality (`_pick_home`); quants within 0.25% quality tie and the smaller
  file wins (Q8_0 over UD-Q8_K_XL); quants under 1.8 bits (or under 3.3 bits
  below ~7B) are never suggested; `evaluate_fit` tries the 2,048-token
  context *before* accepting a last-resort (< 3.7-bit) quant. Fastest keeps
  ≥ 70% of Recommended's quality points and ≥ 2B effective params (MoE
  effective size = total^0.6 × active^0.4); fillers prefer new families *and*
  new lineages (architecture + size), so fine-tunes of one base share a
  slot. Always-thinking models are never on the short menu and their turn
  speed counts `ALWAYS_THINKING_TOKENS`. Apple prefill speed-up is 10x.
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
  from name ("A3B"), `thinking` = "none" / "switchable" / "always" from the
  GGUF `chat_template` (a thinking switch → switchable; `<think>` opened for
  the model with no switch → always), else name rules (`thinking_mode_for`);
  `reasoning = thinking != "none"`,
  ollama_ref "hf.co/<repo>:<quant>", license_url
  `https://huggingface.co/<repo>` (model card), quant_options from files.
- Cache: JSON at `cache_dir()/hf_models.json` with `fetched_at`; used when
  fresh, or when offline / the Hub fails (stale cache is fine with a note).
  If live fails and no cache: return curated seeds (`source="curated"`).
- Never raise for network errors; put a friendly note in `notes`.
- *As built:* `DiscoveryResult.stale: bool = False` (True when an expired
  cache was used as a fallback). `expand` also requests `pipeline_tag` (with
  `expand` the Hub returns only the listed fields). Live and cached results
  also include curated seeds for original models the search missed
  (`source="curated"` on those entries; the cache stores live entries only,
  with `schema_version`, `fetched_at`, `allow_all_licenses`). Lookups run on
  ≤ 6 threads; jobs not started before `timeout_s` are skipped with a note.
  Extra public helpers: `rejection_reason(info, *, allow_all_licenses)` (plain-
  English reason or None), `license_of`, `params_from_name` ("30B-A3B" →
  (30, 3)), `prettify_repo_name`, `shard_info`, `repo_gguf_files(api, repo)`,
  `default_cache_path()`, `FALLBACK_QUANT_ORDER`, `CACHE_SCHEMA_VERSION`.
  *Round 3:* the cache also stores `rules_version` (`RULES_VERSION`, a
  fingerprint of every screening/labelling rule plus the game version); a
  cache saved under other rules is never "fresh", and when it's used as a
  fallback each entry is re-screened (`rescreen_entry`: name/family/license/
  size/popularity, restricted-family relabel, thinking mode).

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
- *As built:* `DownloadError(message, kind="other")` — `kind` ∈ not_found,
  gated, no_gguf, missing_file, network, offline_mode, server, disk,
  incomplete, bad_repo_id, not_family_friendly (a player-named repo - the
  `custom` pick, `--model` - whose id or tags mark it uncensored, safety-removed
  or adult, `hf_discovery.not_family_friendly`: refused before anything is
  downloaded, since the Steam AI disclosure promises only family-friendly
  models). Default folder `model_folder(repo)` =
  `models_dir()/<owner>--<name>`. Already-downloaded files are returned
  without any network call; offline, an existing local copy of the quant is
  reused. Progress: a tqdm-compatible bridge is passed as `tqdm_class` when the
  downloader accepts it, feeding `ui.download_progress`. Extra public helpers:
  `custom_entry(repo_id, quant=None, *, hf_api=None) -> ModelEntry` (describe a
  pasted repo so the fit engine can check it before downloading) and
  `normalize_repo_id(text) -> (repo_id, quant | None)` (accepts owner/name,
  huggingface.co links, `hf.co/owner/name:QUANT`).
  `find_local_copy(repo_id, quant, dest_dir=None) -> Path | None` finds a
  finished local download of a quant without the network; `download_gguf`
  uses it first when the entry has no exact file names for the chosen quant.

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
RUNTIME_EXPLAINER_BUILT_IN: str   # the same for a built game: the engine ships inside it, nothing downloaded
def runtime_explainer() -> str    # whichever of the two fits this copy (the model menu's `learn`)
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
  cpu; else cpu. Windows arm64 → win-cuda-13.x-arm64 (+cudart) when an NVIDIA
  GPU with driver ≥ 580 (or unknown) is present, then win-cpu-arm64. CUDA 13
  is skipped when every NVIDIA GPU reports a compute capability below 7.5
  (`GPUInfo.compute_capability`, from `nvidia-smi --query-gpu=…,compute_cap`,
  retried without it on drivers that don't know the field): CUDA 13 builds
  have no code for Maxwell/Pascal/Volta.
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
def parse_challenge(text: str, *, current: str | None = None, truncated: bool = False) -> tuple[str, str]   # (narration, challenge) split on the first usable "CHALLENGE:" line (skips echoes of `current`); fallback = last paragraph, or "" if `truncated`
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
  *As built:* a Jev call that failed is kept in the additive field
  `RoundRecord.failed_jev_exchange` so the review can show it; the error menu
  also offers a "skip" (built-in intro/challenge, or the backup rule for a
  verdict) so an LLM hiccup never costs the player their round.
  The Jev "Learn" panels are staggered: one new answer type per Jev verdict
  (Noul, then Choice, then Score), so round 1 isn't three lessons in a row.
- Truncate plans to `max_input_chars`.

### review.py
```python
def run_review(ui: UI, summary: GameSummary, *, export_dir: Path | None = None,
               secrets: Iterable[str] = (), thinking_skipped_note: str | None = None) -> None
def export_transcript(summary: GameSummary, export_dir: Path | None = None, *,
                      secrets: Iterable[str] = ()) -> tuple[Path, Path]
def summary_to_dict(summary: GameSummary, *, secrets: Iterable[str] = ()) -> dict   # JSON-safe, no secrets
def summary_to_markdown(summary: GameSummary, *, include_jev: bool = True, include_reasoning: bool = True,
                        secrets: Iterable[str] = ()) -> str
```
- Asks independent questions (either, both, or neither), each only when there
  is something to show:
  1. "See the Jev request & response for each round?" (only if any round used Jev)
  2. "See the local model's reasoning (chain-of-thought) for each round?"
     (only if it exposed some; otherwise `thinking_skipped_note` or a short
     note that the model didn't think out loud).
  3. "See the local model's verdict (its JSON answer) for each round?" (only
     when the local model refereed and there was no reasoning to show - the
     reasoning view already includes those answers).
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
   (warm-up) and return. If Jev was turned down last time, that question has a
   third choice, `jev` - "Play with Jev on this time (the optional AI
   referee)" (`WELCOME_BACK_JEV_OPTIONS`; `SetupResult.ask_jev` →
   `run_jev_onboarding(ask_again=True)`): the window has no command line for
   `--jev`. A pretend-model game passes `remember_no=False`, so a "no" to Jev
   there isn't saved.
2. "Let me take a look at your computer…" spinner → `specs.friendly_summary`
   + compact details table; `teach` panel (condensed memory + speed
   explainers) — offer "details?" rather than dumping walls of text.
3. "Searching Hugging Face for models that fit your computer…" spinner →
   `discover_models` → `rank_models` → `pick_shortlist` (the "more" list
   keeps the short list's numbers first, then the rest best first, so a number
   picks the same model in both views) → a clean numbered
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

*As built:* `run_setup(..., services: SetupServices | None = None)` — a
dataclass of injectable callables (`detect_specs`, `discover_models`,
`make_backend(kind, **kw)`, `installed_runtimes`, `custom_entry`) so tests use
fakes. Extra menu words: `learn` (full explainers), `why N`
(`catalog.explain_fit`), `back`, `quit`; the table switches to a compact
layout below 140 columns. The failure menu also offers `pick` (choose another
model). The measured tokens/s is fed back into the fit engine: the memory
bandwidth behind the chosen placement is scaled by measured ÷ estimated
(clamped 0.25–2×), re-ranking the menu in-session and saved in
`Settings.extra["speed_calibration"]` keyed by a hardware fingerprint for later
launches (*Round 3:* only from bandwidth-dominated dense models, solved for
the bandwidth, tagged with the engine build - see the Round 3 notes).
`--mock` runs discovery with `offline=True`. In a built game (downloads off,
managed engine) `find_models(..., engine_limits=True)` first drops discovered
models whose GGUF architecture the bundled llama.cpp release doesn't know
(`models_for_engine`: `SetupServices.engine_architectures` =
`runtime_install.engine_architectures()`, the `architectures` list
`fetch_engine.py` writes into each bundled `install.json`; unknown
architectures are kept; a note says how many were left out), since the game
can't update its engine and they would download and then fail; a custom repo
or `--model` of such an architecture is warned about first
(`engine_lacks_message`). When free disk space alone rules every model out
(`catalog.disk_space_needed`), the empty menu, `--list-models` and
`specs.friendly_summary` say so - how much the smallest model needs, how much
is free, and `--models-dir` (`disk_space_warning` / `specs.disk_space_advice`,
worded per `ui.option_hint`) - instead of blaming memory. Extra public helpers:
`find_models`, `ModelSearch`, `show_hardware`, `model_table`/`show_model_table`,
`entry_for_fit`, `fit_for_quant`, `calibrate_specs`, `apply_saved_calibration`,
`entry_to_dict`/`entry_from_dict`, `default_backend_factory`.
Integration notes: if the managed engine had to fall back to CPU mode
(`backend.cpu_only`) for a GPU/unified estimate, the measurement is *not* used
for calibration (it says so instead). Asking for a faster model when none is
clearly faster (≥ 1.5× the measured speed) says so and offers to play anyway
rather than re-offering the same pick. The discovery line counts Hub results
and built-in seeds separately ("Found 11 models on Hugging Face (plus 3 of my
hand-checked favourites)"). The confirmation screen recognises an
already-downloaded quant via `download.find_local_copy`.

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
license prominently), `--no-jev`, `--jev` (ask about Jev again after
choosing the local model only), `--models-dir DIR` (keep models in this
folder; remembered), `--think` (let a thinking model think out
loud even when it's slow), `--target N` (default 5), `--reset` (forget saved
settings), `--export-dir DIR`, `--debug` (tracebacks), `--version`.
Ctrl+C anywhere exits cleanly with a friendly line (exit code 130); the
backend is always `close()`d. Flow: banner → setup → Jev onboarding (skipped
by `--no-jev`) → Game.run() → run_review().
*As built:* `main(argv=None, *, ui=None, services=None)` (test hooks). Exit
codes: 0 ok (including choosing `quit` - at a menu, or typed at a yes/no
question, which raises `UserChoseQuit`, a `UserQuit` subclass; in the game it
still leads to the quit ending and the review), 1 unexpected error (friendly line;
the traceback is saved to `logs/crash.txt` via `crashlog.write_crash_report`
and its path shown; `--debug` also prints it; the terminal names
`<command_name()> --reset`, the window offers to forget the settings right
away), 2 bad option, a missing `--gguf` file or an unusable `--models-dir`,
130 Ctrl+C. `-h` and `--version` return 0 instead of
raising `SystemExit`; `--quant` is upper-cased; `--target` accepts 1–50.
SIGTERM, SIGHUP and (Windows) SIGBREAK are handled like Ctrl+C for the
duration of main(); the previous handlers are restored on return. The engine
is not orphaned even when no Python code gets to run (console window closed,
game killed): on Windows it is assigned to a Job Object with
`KILL_ON_JOB_CLOSE`, on Linux it gets `PR_SET_PDEATHSIG`, and on POSIX it runs
in its own session (so Ctrl+C reaches only the game, which stops it itself).
Each launch also writes an owner record, and the next launch stops any engine
whose game is gone (`reap_orphaned_servers`). `main()` also makes stdin
tolerant of undecodable bytes (`ui.make_input_safe`) and passes the warm-up
speed and context window to `Game`.
*Distribution (as built):* the first launch shows `notices.AI_CONTENT_NOTICE`
once (`show_ai_notice_once`, remembered as `Settings.extra["ai_notice_seen"]`);
after each review `ask_to_play_again` offers "Play again? [Y/n]" and loops with
the same backend and Jev client - or, after a pretend model picked from the
menu (not `--mock`), `after_pretend_game` offers `AFTER_PRETEND_OPTIONS`
(again / **Pick a real AI model** / quit): "real" closes the pretend model and
runs `run_setup` again (the model menu; Jev is asked again, since the trial's
"no" wasn't saved), and quitting names the way to a real model. Both happen only when a person is playing
(`player_is_present`: a real terminal or the game's window), so piped input
and scripted tests see exactly the old flow. The games of one session share one
set of "Learn" panels already shown (`Game(taught=...)`). A returning player
who chose the local model only isn't asked about Jev again unless `--jev`
(`run_jev_onboarding(..., ask_again=...)`). `--specs` shows `available_plan`
(only the builds a built game ships) and `runtime_install.engine_summary()`.
The built game's console program
double-clicked on Windows waits for Enter before its window closes
(`launcher.console_closes_on_exit` / `wait_before_closing`), including after
errors. The window runs this same `main(argv, ui=...)` on a worker thread.

## Distribution: the game window, the built game and safety (as built)

Get To Work ships as a free double-click / Steam game. This section records
the modules added for that; [docs/DISTRIBUTION.md](DISTRIBUTION.md) is the
detailed contract (build layout, packaging scripts, CI, Steam) and has the
same authority as this document.

### distribution.py
```python
@dataclass(frozen=True)
class Distribution:
    channel: str = "dev"                 # "dev" (no distribution.json) or "release"
    engine_downloads: bool = True        # may the game download llama.cpp at run time?
    engine_dirs: tuple[Path, ...] = ()   # folders holding bundled engine builds
    llama_cpp_tag: str | None = None
    app_version: str = __version__
    root: Path | None = None             # folder holding distribution.json
    built_from: str | None = None        # git commit of the build
    notes: tuple[str, ...] = ()          # anything odd noticed while reading the file
    @property
    def bundled(self) -> bool            # at least one engine dir exists
def find_distribution_file() -> Path | None
def load(*, refresh: bool = False) -> Distribution   # cached, thread-safe, never raises
```
- `distribution.json` is looked for at `$GETTOWORK_DISTRIBUTION`, next to
  `sys.executable`, in `../Resources/` (macOS .app), then in `sys._MEIPASS`.
  None found → developer copy (downloads allowed, no engine dirs) - but a
  built game (`sys.frozen`) keeps downloads off and looks in the `engine/`
  folder next to its executable. A damaged file → the `engine/` folder beside
  it and a note; downloads stay on only in a developer copy.
- `GETTOWORK_ENGINE_DIR` (one or more folders, `os.pathsep`-separated) is
  checked first; `GETTOWORK_ALLOW_ENGINE_DOWNLOAD=1`/`0` forces downloads on or
  off. The live-check workflow uses both to play through a real engine exactly
  like the Steam build does.

### The bundled engine (runtime_install.py, backends/llamaserver.py, setup_flow.py)
- `installed_runtimes()` also lists the builds in `distribution.load().engine_dirs`
  (each `<tag>-<variant>/` folder with an `install.json` carrying
  `"bundled": true`). Bundled builds are **read-only**: never pruned, never
  written to; `mark_unusable()` for one records the verdict in
  `runtime_dir()/bundled-unusable.json` (`BUNDLED_UNUSABLE_FILE`, keyed
  `"<tag>-<variant>"`, plus the program's size and mtime: a repaired or
  different copy of the file gets a fresh check), which `unusable_reasons()` merges.
- `downloads_allowed()` (False in a built game) makes `ensure_llama_server`
  work offline: `choose_installed(plan, specs)` takes the best planned build
  that is here (CUDA isn't bundled → Vulkan/Metal) - this copy's own builds
  first, of any type or release, before any other install sharing the
  settings folder (`own_builds_first`: a developer copy's newer download or
  its CUDA build never beats the engine that ships; other installs count only
  when no own build will do) - CPU mode on a bundled
  build is the last resort, and with nothing installed it raises
  `RuntimeInstallError(ENGINE_MISSING_MESSAGE)` ("The game's built-in engine is
  missing. On Steam: ... Verify integrity. Otherwise re-download the game.").
  `available_plan(specs)` is the plan a built game can really try (used by
  `--specs`); `find_installed(variant, specs)` finds a build that can serve
  as a variant (on Apple Silicon the Metal build *is* the CPU build: it runs
  on the processor with `--device none`); `is_bundled(exe)` tells bundled
  builds apart.
- `relocate_engine(saved_exe)`: settings store the full path of the engine
  that worked; a built game's engine moves with the game (another Steam
  library, the app dragged elsewhere), so the same `<tag>-<variant>` build -
  or else the newest of the same variant - is looked up again (setup's
  "Welcome back" and `LlamaServerBackend.prepare` both use it). In a built
  game a saved engine that still exists but belongs to another copy (an older
  download, a developer copy sharing the settings folder) is swapped for this
  copy's own build of the same type (`own_build_instead`), so the engine that
  ships is the one that runs. A moved game's saved build is looked up among
  its own builds first too.
- `engine_architectures()`: in a built game, the model architectures its own
  builds' `install.json` files list (`architectures`, from the pinned
  release's `src/llama-arch.cpp`); None in a developer copy or for builds
  that recorded none (then nothing is filtered).
- `bundled_builds_problem()`: every built-in build present but noted as
  unable to run here (on a Mac, the one Metal build) → the built-in
  "can't run on this computer" wording instead of `ENGINE_MISSING_MESSAGE`.
  `engine_problem`, `is_available`, `_use_builds_here` and setup's engine step
  use it.
- `engine_summary()` (for `--specs` and the build's smoke test): "built into
  the game: llama.cpp <tag> (CPU, Vulkan); engine downloads off", or
  "... engine downloads on" in a developer copy. `runtime_explainer()` picks
  `RUNTIME_EXPLAINER_BUILT_IN` in a built game (no downloads, no CUDA).
  `other_engines_hint()` names Ollama, plus `pip install llama-cpp-python`
  only where that can work (`llama_cpp_python_possible()`: not frozen, and
  downloads allowed).
- `LlamaServerBackend`: the GPU → CPU fallback only considers builds that are
  here when downloads are off (a failing Vulkan/Metal build falls back to the
  bundled CPU build or `--device none` on the same exe). A model whose
  architecture the bundled engine doesn't know gets a "builtin" explanation
  (pick another model; game updates bring newer engines) instead of an
  engine download - after trying the newest build of that type already on
  disk, if the one that failed was older. `is_available()` reports "The
  game's built-in llama.cpp engine is ready (<tag>, <variant>)". `close()` is
  final until the next `prepare()` (`_closed`, guarded by a lock around
  launching): once the game is quitting, `_launch`, `_ensure_running` and
  `_switch_after_crash` refuse to start an engine, so a request failing
  because `atexit` stopped the engine never starts a CPU-mode one that nothing
  would stop (macOS has no parent-death signal).
- `setup_flow`'s confirmation screen shows "Built into the game (llama.cpp
  <tag>, Vulkan + CPU) - nothing to download" (Metal + CPU on a Mac) with a
  line saying llama.cpp's MIT license text ships as THIRD_PARTY_LICENSES.txt;
  `planned_engine_key` uses the build a built game will really run, so speed
  calibration is keyed correctly.
- Helpers `fetch_release(tag)`, `download_asset`, `unpack_archive`,
  `finish_unpacked`, `install_marker(..., bundled=True, license_files=...)`
  and `releases_newest_first` are public so `packaging/fetch_engine.py`
  reuses the game's own verified download and safe-extract code.

### gui/ (terminal.py, bridge.py, app.py)
- `terminal.TerminalBuffer(*, max_lines=10_000, rows=24)`: pure-Python ANSI/VT screen
  model. `feed(text)`, `lines` (runs of `(text, Style)`), `take_dirty()`,
  `cursor`, `text()`. Wide and zero-width characters, `\n \r \b \t`, CSI
  cursor movement / erase / SGR (16, 256 and truecolor), OSC 8 hyperlinks
  (runs carry the URL), everything else dropped safely, even when split
  across chunks. Rich's `Live`/`Status`/`Progress` redraws work as in a
  terminal.
- `bridge.GuiBridge(columns=100, rows=32, opener=None)`: `.stream` (what the
  game's rich Console writes to; `isatty()` True, UTF-8), `request_line(prompt,
  *, secret=False)` (blocks the game thread; `EOFError` once closed),
  `show_choices(options)`, `open_url(url)` (http/https only), `closed`,
  `columns`; the window side calls `poll()` and `submit(text, prompt_id)`,
  and `close()` / `detach()` when the window goes.
  Answers name the prompt they answer, so a late click can't answer the next
  question. No Tk import.
- `app.run_gui(argv=None, *, selftest=False, game_main=None) -> int` and
  `GameWindow(root, argv, *, game_main, selftest, opener, env, ...)`: the Tk
  window on the main thread (macOS requires it), `cli.main(argv, ui=...)` on a
  daemon worker thread with `UI(console=Console(file=bridge.stream,
  force_terminal=True, color_system="truecolor", width=<cols>, ...),
  choices_fn=bridge.show_choices, hides_input=True, window=True, pauses=True, ...)`. Dark theme,
  monospace font from `FONT_CANDIDATES`, clickable links, menu buttons, input
  history (Up/Down), Ctrl+= / Ctrl+- / Ctrl+0 zoom saved as
  `Settings.extra["gui_font_size"]`, F11 full screen. Full screen with a bigger
  font and a **Keyboard** button (`steam://open/keyboard`) on a Steam Deck / in
  Big Picture (`SteamDeck`, `SteamGamepadUI`, `GAMESCOPE_WAYLAND_DISPLAY`;
  `GETTOWORK_FULLSCREEN=1/0` overrides); full screen keeps the 16-point font
  while 80×24 characters fit (`FULLSCREEN_MIN_COLUMNS/ROWS`), so it stays big on
  the Deck's 1280×800 screen. A **Report a problem** button opens
  `notices.report_url()` (Steam's overlay can't open over a Tk window) - and
  always writes that link into the transcript first (a browser may never
  appear), adding a "copy the link above" line if opening failed; on a Steam
  Deck / in Big Picture it goes through `steam://openurl/` (Steam's own
  browser, over the game). Menu buttons show each option's short label, never
  two that could be mistaken for each other (`button_labels` /
  `labels_look_alike`: "Yes" next to "Yes, play" gets its full label). Pastes
  always land in the input bar (a click on the transcript gives focus back;
  Ctrl/Cmd+V there is redirected; Ctrl+Shift+V; a right-click menu). Keys that
  aren't typing - a modifier on its own (the Ctrl of Ctrl+C), F-keys, Escape -
  are left alone on the transcript, so Ctrl+C copies a selection and F11 works
  first time; the transcript has its own right-click Copy / Select all. Lone
  surrogates are replaced before they reach Tk (`bridge.clean_text`); a failed
  redraw is logged once and the transcript redrawn, while the question still
  appears. Transcripts default to
  `default_export_dir()` (`~/Documents/Get To Work`, else
  `config_dir()/transcripts`) unless the player passes `--export-dir`; macOS
  `-psn_...` arguments are dropped. When the game ends: "Press Enter or close
  the window to exit". Closing: mark closed → `EOFError` → `WindowClosed` (a
  `UserQuit` the Jev setup and the review don't swallow) → the game's goodbye
  and cleanup with no more model calls, wait ≤ 8 s, put stdout/stderr back
  whatever replaced them (the window's spinners and progress bars don't
  redirect them), return so `atexit` stops llama-server. A window that can't open writes `config_dir()/logs/gui-crash.txt`
  and, from a terminal, plays there instead; otherwise a native message box
  (`show_error_dialog`) says why and how to repair the install. Long answers
  are echoed together with their question so rich wraps them inside the
  window; key-shaped answers (`ui.looks_like_secret`, or one given at a key
  question) are echoed as "(hidden)" and kept out of the history; the Jev key
  menus take a key pasted straight into them (`UI.choose(accept=...)`). On a
  Steam Deck the input area is at the top (Steam's keyboard covers the bottom).
- Self-test (`--gui-selftest`, CI): `SelftestPlayer` answers by *what* is
  asked (Enter at pauses and menus, plans, "n" to yes/no), the transcript must
  contain "YOU GOT TO WORK", `$GETTOWORK_SELFTEST_OUT` receives it; exit 0 /
  1 / 2 (timeout, default 120 s, `GETTOWORK_SELFTEST_TIMEOUT`). With no other
  options it plays `--mock --no-jev`, in a throwaway `GETTOWORK_HOME`.

### ui.py (additive)
`UI.__init__(..., choices_fn=None, hides_input=False, window=False)`. With
`window=True` (`ui.in_window`), an `EOFError` from the input function raises
`WindowClosed` (subclass of `UserQuit`), hints are worded for the window (no
commands or Ctrl+C), `status()` / `download_progress()` leave stdout/stderr
alone, and the model menu's buttons come from setup's `_menu_buttons`.
`choose()`, `confirm()`
(`[("y", "Yes"), ("n", "No")]`) and `pause()` (`[("", "Continue")]`) call
`choices_fn(options)` before asking and `choices_fn([])` afterwards (errors
in it are ignored); `can_hide_input()` is True with `hides_input`. No change
for the terminal.

### launcher.py
`gui_main(argv=None) -> int` (the windowed executable, `gettowork-gui`,
`python -m gettowork.launcher`): strips `--gui-selftest`, runs
`gui.app.run_gui`. `console_closes_on_exit()` (Windows + frozen + the console
belongs to this process alone + stdin is a TTY) and
`wait_before_closing(input_fn=None)` keep a double-clicked console window
open. `restore_system_library_path()` (called first by both frozen entry
scripts) undoes PyInstaller's `LD_LIBRARY_PATH` for the programs the game
starts on Linux (engine, browser, hardware checks), so they load the system's
libraries, not the game's bundled copies. Imports neither Tk nor the game at
import time.

### safety.py, safety_terms.py, notices.py (and game.py / prompts.py)
```python
@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool; category: str | None = None; matched: str | None = None
    @property
    def label(self) -> str               # "graphic gore", or ""
CATEGORY_LABELS: dict[str, str]          # sexual, hate, self_harm, gore, drugs
def check_text(text) -> SafetyVerdict    # hard blocks; never raises
def check_player_input(text) -> SafetyVerdict   # same lists, for typed plans
def soften(text) -> str                  # masks mild swearing: "damn" -> "d***"; idempotent
def normalize(text) -> str               # folding, lowercase, invisible chars, leetspeak inside words
def hidden_note(verdict) -> str          # what the review/transcripts keep instead of blocked text
```
- Word lists are ROT13 in `safety_terms.py` (`BLOCKED` by category,
  `MILD_PROFANITY`, `EXEMPT_PHRASES`); whole-word matching on several
  readings of the normalised text (joined single letters, symbols inside a
  word removed, stretched letters), phrases only within one clause. The
  self-harm phrases are generated for every person and tense (`_phrases` over
  small ROT13 building blocks), and figures of speech made from the same words
  ("killing myself laughing"), compound nouns ("my own life jacket") and the
  British crop are exempt.
- `game.py`: every model text the player sees - intro, outcome narration and
  challenge, victory/quit endings, local referee explanations, Jev's labels,
  exposed reasoning - goes through `check_text`; a flagged story reply is
  asked for once more with `prompts.safety_retry_messages(messages)` (adds
  `SAFETY_REMINDER`, never repeats the rejected text), then replaced by a
  built-in line from `backends/mock.py`; shown text is always `soften()`ed.
  The kept `LLMResult.text`/`reasoning` becomes `hidden_note(...)` when
  blocked, and `LLMResult.raw` (the engine's JSON, saved in exported
  transcripts) goes through the same filter string by string
  (`_screened_raw`), or is dropped when the reasoning was blocked. Notes (categories only) go to `RoundRecord.safety_notes` and
  `Game.safety_notes`. A flagged plan is refused with
  `FAMILY_FRIENDLY_REFUSAL` ("Let's keep it family-friendly - try another
  plan!"), never sent to the model or Jev, and costs no round.
- `notices.AI_CONTENT_NOTICE` (Markdown, first launch) and
  `notices.STEAM_AI_DISCLOSURE` (the Steam content-survey text, quoted
  verbatim in `packaging/steam/STORE_PAGE.md` and checked by a test);
  `REPORT_HOW` says how to report problems: the window's **Report a problem**
  button (`REPORT_BUTTON`, opening `report_url()` - the game's Steam
  Discussions once `STEAM_APP_ID` is set, a store search before) or the store
  page's Discussions. Steam's Shift+Tab overlay can't open over the Tk window
  outside a Deck's Game Mode, so it isn't promised.

## Testing rules
- `pytest` only, no network, no real models, no real browser. Use MockBackend,
  fake Jev transports, fake `hf_api`/`hf_download`, and `UI(console=Console(file=io.StringIO()), input_fn=scripted)`.
- Each module has its own `tests/test_<module>.py`.
- Tests that open real Tk windows (`tests/test_gui_app.py`) skip cleanly
  without `tkinter` or a display; Linux CI runs the suite under `xvfb-run`.
  `GuiBridge` and `TerminalBuffer` are tested with plain threads and strings.
- Built-game behaviour is tested with `GETTOWORK_DISTRIBUTION` /
  `GETTOWORK_ENGINE_DIR` / `GETTOWORK_ALLOW_ENGINE_DOWNLOAD` and
  `distribution.load(refresh=True)`; the build scripts in `packaging/` are
  tested with fake archives and fake `runner`s (`tests/test_packaging.py`,
  `tests/test_fetch_engine.py`).

## Legal / safety rules for the codebase
- No model weights are bundled. Models are downloaded by the player from
  Hugging Face under each model's own license; the game shows that license.
- Curated seeds and default discovery results: Apache-2.0 / MIT only. `--all-licenses` shows others, and so does naming a model yourself (`custom`, `--model`) - always with the license visible and the same yellow warning.
- The llama.cpp engine (MIT) is **bundled in the game builds** (official
  ggml-org release builds fetched and digest-checked in CI by
  `packaging/fetch_engine.py`, used unmodified, license files kept -
  llama.cpp's own MIT text and the texts of what's compiled into it
  (cpp-httplib, jsonhpp, BoringSSL, LLVM OpenMP), which the build requires per
  OS - and the Visual C++ runtime DLLs next to the Windows engine); a
  developer copy downloads it from the official GitHub releases at the
  player's request. It is never committed to the repository.
- Every game build ships `THIRD_PARTY_LICENSES.txt` (`packaging/collect_licenses.py`):
  every bundled Python distribution's license files, Python, Tcl/Tk, the
  PyInstaller bootloader note (GPL with the bootloader exception) and the
  llama.cpp licenses. Anything new that ships in the build must appear there
  and in NOTICE.md.
- Live-generated AI text has guardrails (Steam requires them): prompts ask for
  family-friendly slapstick, `safety.py` checks every model text and typed
  plan, the first launch discloses AI content, and `notices.STEAM_AI_DISCLOSURE`
  must describe the guardrails accurately.
- The hardware-fit heuristic is original code in this repo (MIT, no warranty).
- Not affiliated with TypeSafe AI, Hugging Face, Ollama, or any model author;
  names are used only to identify their products.
- API keys: never printed, never logged, never exported, redacted in review,
  never in `repr()` (`JevClient` and `Settings`), never sent to another host
  on a redirect, and never over plain http (except to localhost). The same
  goes for `GITHUB_TOKEN` (the engine installer drops credential headers on a
  redirect to another host and refuses https → http). A key pasted by
  accident as a plan is refused, and the review/transcripts mask the live key
  wherever it appears. Hardware-detection tools and the engine never receive
  the player's API tokens in their environment.

## Review fixes (as built)

These notes record behaviour added after the first full review. They are
binding in the same way as the contracts above.

**backends/base.py.** `chat(..., think: bool | None = None, stop: list[str] | None = None)`:
`think=False` asks a thinking model to answer straight away (llama-server:
`chat_template_kwargs {"enable_thinking": false, "reasoning_effort": "low"}`;
Ollama: `"think": false`; llama-cpp-python: `/no_think` for Qwen3), `stop`
ends the answer early. `LLMResult.truncated` is True when the answer hit
`max_tokens` (`finish_reason`/`done_reason` "length"). `supported_chat_options(backend)`
tells the game which of these a backend accepts (older backends get neither).
`LLMBackend.on_notice` is set by the game while a call runs; a backend calls
`self._notice(text)` for a mid-call status, e.g. `RETRY_WITHOUT_THINKING_NOTICE`
before the empty-answer retry, and the game shows it in the spinner.

**llamaserver.py.** Each launch gets a random API key, passed as
`LLAMA_API_KEY` in the child's environment (never on the command line) and sent
as a Bearer header; the child environment drops the player's `LLAMA_*`
variables and common secrets. CPU mode passes `--device none` (and `-ngl 0`).
The bad-arguments check is anchored to llama.cpp's own messages (an
`Invalid argument` in a model-load error is not a bad flag). The health
timeout scales with the size of the whole model - every part of a split
model (`health_timeout_for`, `_model_total_bytes`) - and is extended while
the log is still growing *or the engine is still reading from disk* (psutil
`io_counters`, Linux/Windows; current engines log nothing between "loading
model" and "model loaded"); the chat timeout scales with the measured tok/s.
Before the model is loaded, a GPU build is asked which devices it can use
(`llama-server --list-devices`, `gpu_devices_from_listing`): an empty list
("(none)" - no Vulkan driver, a CUDA library that won't load) moves to the
next build, or runs this one in CPU mode with a driver hint. After start-up,
`gpu_offload_from_log` decides GPU use from where the weights went
(`CUDA0`/`Vulkan0`/`MTL0` model buffers vs `CPU_Mapped`) when the log shows
it, never from "offloaded N/M layers to GPU" alone (every official build has
the RPC backend, which reports layers "offloaded" with no GPU at all). "unknown model
architecture" means the engine is too old: the engine is updated once
(`ensure_llama_server(update=True)`), otherwise the error says so. The log
file is locked while in use (a second copy of the game uses its own log).

**runtime_install.py.** Linux CUDA builds need glibc ≥ 2.38
(`CUDA_LINUX_MIN_GLIBC`); older systems skip them, and a build that failed for a
permanent reason (`PERMANENT_FAILURES`) is marked unusable so the next launch
doesn't retry it. The confirmation screen shows `license_text(variant)`: CUDA
builds include NVIDIA's CUDA runtime under NVIDIA's terms, not only MIT. A
stale `GITHUB_TOKEN` (401) is retried without the token. Finishing an install
that another copy of the game completed first reuses that install.

**hf_discovery.py / download.py.** Every Hub call has a finite timeout (an
httpx client factory with a timeout) and runs under a deadline on daemon
threads, so a stalled connection can't block exit. MoE active parameters are
read from the GGUF header (expert counts), not only from "A3B"-style names.
A partial or slow Hub answer is cached with a short TTL and never served as
fresh for the full 72 h; a cache written by `--all-licenses` is not reused by
a normal launch. Licences are shown "as declared on Hugging Face", with the
original model's card linked, and families with their own licence are labelled
as such. `download._hub_errors()` looks up each error class in
`huggingface_hub.errors` and then `huggingface_hub.utils`. The dependency floor
is `huggingface_hub>=1.1` (the first version with `tqdm_class`, which drives
the download bar).

**catalog.py / perf.py (fit engine).** RAM and VRAM are GiB and file sizes
are GB, so sizes are converted (`GIB_PER_GB`) before comparing. OS headroom is
3.5 GiB on Windows (2.5 elsewhere). Speed thresholds use the *turn* speed
(`turn_tokens_per_s`: reading a ~1,200-token prompt plus writing a ~250-token
answer); the reasoning bonus only applies when the model is fast enough for
the game to let it think (≥ `THINKING_MIN_TOKENS_PER_S`), which is the same
threshold the game uses. A quant that is already downloaded needs no
disk space; a model that fits only with a shorter context gets
`FitResult.context_tokens = 2048` rather than being dropped. The Fastest badge
needs a minimum quality (no sub-1B toy unless nothing else fits); Smartest
excludes tight fits in system RAM and last-tier quants; near-identical
variants of one family take one slot, and in the filler passes an
unrecognised fine-tune never comes before the original model of its lineage
(`_originals_first`). The curated bonus also goes to a seed's newer dated
release from the same publisher (`_earns_curated_bonus`).
RAM bandwidth is measured with parallel `memmove` copies on several threads
(`measure_ram_bandwidth(budget_s, threads=None)`), and the no-AVX penalty
depends only on the CPU's SIMD flags. `specs.friendly_summary` uses the same
ranking as the menu, so the two never disagree.

**setup_flow.py.** The engine's actual limits (e.g. a CPU-only engine) are
applied to the specs before ranking (`apply_engine_limits`), so the menu never
promises GPU speed the engine can't deliver. The managed → Ollama fallback asks
first and hands over the downloaded file with `/api/blobs` + `/api/create`
(`OllamaBackend(gguf_path=...)`) instead of pulling it again. `--model` with a
repo whose size can't be read says so instead of "won't fit". `SetupResult`
gains `tokens_per_s` (the warm-up measurement).

**prompts.py.** `COMMUTE_CHALLENGE` is round 1's question and
`COMMUTE_JUDGE_CHALLENGE` is what referees (local or Jev) are asked in that
round. `intro_messages` asks for no CHALLENGE line; `outcome_messages(...,
commute=...)` narrates setting off (or a comic failure to set off, with no
CHALLENGE line) and passes the chosen way of travelling to later rounds.
`absurdity_index(progress, target)` spreads the levels end to end over the
obstacles after the commute (the first is always mild, the last one before the
finale always the fantastical, magical tier - a default 5-step game goes mild,
surreal, fantastical) and always uses the finale for the last step.
`parse_challenge(text, *, current=None, truncated=False)` uses the **first**
usable CHALLENGE line (skipping placeholders, prompt examples, echoed
"CURRENT CHALLENGE" lines and, after a success, a repeat of `current`), drops
everything after it (a rambling model playing the player's part), accepts
numbered, quoted and inline labels, and never turns a cut-off sentence into a
challenge. Echoed prompt labels and `<player_plan>` blocks are removed from
stories. The judge template uses placeholders (`<true or false>`), a copied
template is unreadable, and when several verdicts appear the last one wins.
`screen_plan(plan, *, commute=False)` is the one shared heuristic for the
backup rule and the pretend model (empty, gave up, waits, claims victory,
orders the referee, too short; any short answer counts in round 1).
`quote_plan` quotes earlier plans inside `<player_plan>` tags in the round
history, and the referee rules (and Jev's `state.note`) say earlier quoted
plans are player data too.

**game.py.** `Game(..., tokens_per_s=None, context_tokens=None)`. Story
narration (outcome, victory, quit) never thinks and uses stop sequences
(`STORY_STOPS`); the intro and the local referee may think only when the model
is fast enough and has room; models squeezed into a 2048-token context get
shorter answers. A truncated or unusable challenge is replaced by a built-in
one from `FALLBACK_CHALLENGE_TIERS` matching the progress. The meter reads
"Progress to your desk"; round 1 is titled "Round 1: the journey" and later
panels "Challenge N". In an interactive terminal the game pauses ("Press
Enter...") after the opening and after a verdict with a lesson. Jev's Choice
label reaches the narrator only when it agrees with the Noul, and the verdict
panel explains a disagreement as a close call. With the pretend model the
chain-of-thought lesson (and the review) call its thinking "scripted example"
text.

**backends/mock.py.** The intro has no CHALLENGE line; the commute outcome
sets off (or not); the next obstacle's tier comes from the progress stated in
the prompt (or its own count), never goes down, and a failed round keeps the
same obstacle.

**jev.py.** `normalize_base_url` adds `https://` to a bare host and refuses
non-https addresses (except localhost) with a `kind="config"` error. The
default transport never follows redirects (urllib would forward the
Authorization header); a 3xx becomes a `JevError` explaining why. An address
urllib can't use is reported like any unreachable address.

**onboarding.py.** Ctrl+C while the key is being checked skips Jev like Ctrl+C
at a prompt. Navigation words typed at the key prompt ("back", "help"...) go
back instead of being sent as a key, and the prompt says Enter goes back.
Choosing a new key without saving it, or having the saved key rejected,
removes the old saved key; a saved key can also be forgotten from the menu.
Any unexpected error during the key check becomes the retry/back menu.

**config.py.** `Settings.load` ignores a file that isn't a JSON object and any
setting of the wrong type; `setup_flow.entry_from_dict` rejects saved entries
without the fields the game relies on. `Settings.save` creates the temporary
file owner-only from the start and removes it if the save fails; `reset()`
also removes a leftover `settings.tmp`. `jev_api_key` is excluded from
`repr()`.

**ui.py / review.py.** `safe_text` strips terminal control sequences from all
printed text; `UI.ask`/`secret` survive undecodable input; `UI.choose`
accepts `y`/`n` for yes/no menus and unique prefixes; `UI.can_hide_input()`
tells callers whether hidden input is really hidden; `UI.status` shows elapsed
seconds and yields `update(text)`; `UI.pause` only waits in an interactive
terminal (or with `UI(pauses=True)`). Transcripts replace the home folder with
`~` (so the OS user name isn't shared) and never fail on broken characters.

## Round 3 review fixes (as built)

**Thinking models (hf_discovery.py, catalog.py, game.py, llamaserver.py).**
`ModelEntry.thinking` is "none", "switchable" or "always" (read through
`catalog.thinking_mode`, which falls back to `reasoning` for old entries).
Always-thinkers are penalised (`PENALTY_ALWAYS_THINKING`), their turn speed
counts `ALWAYS_THINKING_TOKENS`, only switchable models earn
`BONUS_REASONING`, and they never reach the short menu. When the game asks for
no thinking, llama-server gets every known switch: `chat_template_kwargs`
`enable_thinking=false`, `reasoning_effort="low"`, `thinking_budget=0`
(Seed-OSS), plus the top-level `reasoning_budget_tokens=0` (current llama.cpp
closes a forced `<think>` straight away; older builds ignore it). `Game` takes
`thinking=` and gives an always-thinker `ALWAYS_THINKING_EXTRA_TOKENS` on top
of each answer budget, in case an engine ignores the switches.

**Showing the thinking (game.py, review.py, setup_flow.py, cli.py).** When the
game keeps a switchable model from thinking (slow, or a short context),
`Game.thinking_note` says why and `run_review(thinking_skipped_note=...)`
shows it instead of "your model doesn't think out loud". `--think` lets a slow
model think anyway (`Game(force_think=True)`); the warm-up mentions it through
`ui.option_hint` (the terminal command; in the window, Steam's Launch Options
when Steam started the game, otherwise nothing - there's no command line).

**Engine lifecycle (runtime_install.py, llamaserver.py, tls.py).** HTTPS
(GitHub and Jev) uses `tls.https_context()`: truststore (the OS store), else
the default store plus certifi. A failed certificate check is explained
(`tls.CERTIFICATE_HELP`, "Install Certificates.command") instead of "are you
offline?", and Jev doesn't retry it. `UrllibHttp` drops credential headers on
a redirect to another host and refuses https → http. `platform_problem` (per
build glibc minimums: x64 CPU/Vulkan 2.35, arm64 and CUDA 2.38; macOS 13.3)
and builds marked unusable (`unusable_reasons`; `cant_execute` and
`gpu_arch` join `glibc`/`cpu_unsupported`) make `is_available()` say no - and
`ensure_llama_server` refuse - before anything is downloaded; an install
marked unusable is never downloaded again. The note belongs to one install
(one `<tag>-<build>` folder): a build *type* only counts as unusable while it
has no working install left and the note is under `UNUSABLE_RETRY_DAYS` (30)
old, so a broken engine update never blocks the older install that works (a
model that needs the newer engine is told exactly that:
`newer_engine_unusable_message`), and a later release gets another chance.
An engine fetched by an update is checked with `--version` before the model
is loaded with it. Right after installing,
`llama-server --version` runs (`LlamaServerBackend(runner=...)`); a build that
can't run moves to the next one before the model download. The `.part`
rename waits out antivirus locks (`_rename_with_retry`, ~15 s), deleting the
archive may fail harmlessly, other disk errors become friendly
`RuntimeInstallError`s, and `LlamaServerBackend._install` turns an `OSError`
into a `BackendError` so the fallback chain continues. After a successful
start, `prune_old_installs` deletes same-build installs an update replaced
(unless a running game uses them, per the owner records) and the payload of
unusable builds (keeping their `install.json` note). A GPU build that dies or
answers 5xx on its first real work (the warm-up now reads a ~1,000-token
prompt) or during a chat switches to the next build (warm-up only) or to CPU
mode, with a notice; if nothing works, `benchmark` raises `EngineStopped` and
setup treats it as a failed start. A quiet restart happens once; a second
stop switches setup. A GPU start that times out with no load progress at all
- nothing in its log and no disk reads - may be a stuck device (`gpu_hang`):
the player is asked whether to keep waiting or try the next build (the build
that works is saved for next time, so this is never switched silently). A
graphics build that gives way to the separate CPU build (Windows/Linux) for a
reason that may not last - no graphics card found, a driver hiccup, graphics
memory running out, a crash on the warm-up, a hang - is noted in
`runtime_dir()/gpu-switch.json` (`GPU_SWITCH_FILE`: from, to, reason, time, a
game-version + GPU-driver fingerprint). When a later launch's saved engine is
that CPU build, `_back_to_gpu_build` tries the graphics build again: for "no
graphics card" by asking `--list-devices` every launch; for the others once
the graphics driver or the game changed, or after `GPU_RETRY_AFTER_S` (7 days;
a hang only after a change) - otherwise it says why the game is still on the
processor. (On a Mac the fallback is the same Metal program in CPU mode, so
the saved engine stays the GPU build.) Windows refusing to start the engine -
Smart App Control / an app-control policy (WinError 4551) or an antivirus
quarantine (225) - is explained (`windows_block_message`), and a missing
library under Steam asks for Steam's file check, never `apt install`
(`missing_library_hint(env=..., bundled=...)`). On
Windows (no `flock`) each game writes its own `llama-server-<pid>-<id>.log`.

**Fit engine (catalog.py, perf.py, setup_flow.py).** See the catalog/perf
contract notes above. Calibration only learns from dense models whose
estimated time per token is ≥ 70% bandwidth (`CALIBRATION_MIN_BANDWIDTH_SHARE`),
solves `seconds/token = GB / (eff × bw) + overhead` for the bandwidth instead
of scaling by measured/estimated, never learns "cpu" when a GPU build
(including Metal) ran, and records the engine build (`engine_key`, e.g.
"managed:cuda-12") so a correction isn't applied to another build.
`fit_for_quant` (an explicit `--quant`) bypasses the quant floor.

**Parsing and prompts (prompts.py, mock.py).** `parse_judge_json` never
raises, decodes only top-level objects (braces inside strings are skipped),
ignores verdict dicts the player typed (`plan=`) and prompt few-shot copies
unless that's all there is. `defang_plan` (used for plans in prompts and the
Jev state) NFKC-folds, removes invisible characters and breaks up
chat-template tokens (`<|...|>`, `[INST]`, `</s>`, `<think>`...).
`screen_plan` reads any script (CJK counted by characters) and catches a typed
`made_progress: true`. The mock reads the official `REFEREE'S VERDICT` line
(never the player's text), picks obstacles that fit the journey
(`Challenge.modes`, `travel_mode`), and its scripted round-1 reasoning no
longer claims "at least four words". Challenges are numbered by progress.

**UI and review (ui.py, review.py, onboarding.py, specs.py).** Menus accept
only decimal digits (`isdecimal`); `safe_text` never removes a backslash at
all (*Round 4*: removing one of `escape()`'s doubled backslashes exposed a live
`[/]`; review text now goes through `ui.plain()` = sanitise then escape);
`UI.table` cells and `UI.json` C1 characters are sanitised; review reasoning
and answers go through `safe_text`. `UI.choose(aliases=...)` and
`UI.confirm` understand back/cancel/skip (no) and quit (UserQuit); onboarding
menus accept the back-out words they advertise, a returning local-only player
gets one short question, `privacy_notice` survives a malformed
`TYPESAFE_BASE_URL`, key tails are escaped, and any unexpected error means
"local only". The Jev review shows the three question definitions once,
then only each round's state, pausing between rounds in a real terminal.
`learn` at the model menu redraws the menu after the lessons. On Windows,
`specs._run` starts powershell/wmic/nvidia-smi from their real system paths
(missing = not installed), and child programs get `config.child_env()`.

## Round 4 review fixes (as built)

**Speed calibration that lasts (setup_flow.py).** The warm-up records the
engine by its *setup kind* (`engine_key("managed", backend)` →
"managed:cuda-12"; the class name "llamacpp-server" also maps to "managed"),
so the key saved matches the key the next launch plans with
(`planned_engine_key`, which for a returning player reads the saved
`server_exe`'s install.json variant). `_detect` records which saved factors it
applied (`saved_calibration` + `_apply_factors` → `_specs_factors`); a new
measurement is relative to those, so the saved total is
`applied × factor` - a factor that wasn't applied (another build) is replaced,
never compounded. After a slow warm-up the menu shows the model just tried at
its *measured* speed (`_with_measured_speeds`, score adjusted), the message
admits when the correction hit its clamp, and the question is a menu
(`pick` / `play`; "back" = pick) instead of a [Y/n] where "back" meant no.
`--gguf` paths (and every saved path) are stored absolute; a vanished saved
model file is mentioned. Saved and cached ModelEntry JSON goes through one
type-checking loader (`types.model_entry_from_json`: numbers written as text
are converted, anything unreadable drops the entry). Badges are named for what
they mean - **Fastest comfortable fit**, **Smartest at a playable pace** - with
a one-line footnote (`BADGE_FOOTNOTE`).

**Discovery (hf_discovery.py).** `granitehybrid` and `jamba` are
`MAYBE_MOE_ARCHITECTURES`: Mixture-of-Experts only when the GGUF header says
`expert_count > 1` (a header with 0 or 1 expert means dense for any
architecture), so dense Granite-4.0-H micro / 1B aren't ranked with a 25%
active-share guess. `rescreen_entry` recomputes the active size (dropping an
old guess) and the family; the MoE tables and `_FAMILIES` are part of
`RULES_VERSION`. The specialist screen also rejects agents and domain
fine-tunes (agent, swe, dev, openhands, search, research, tool(s),
function(s), medical/med/clinical, rag, structure(d); "guard" and "research"
also inside a word) and checks the base model's name too. A family comes from
the names, then the GGUF architecture, never the uploader's name. Candidates
whose attention shape isn't in `catalog._KV_SHAPES` get their GGUF header
read (`_needs_header`), and `kv_shape_from_header` stores
`ModelEntry.kv_shape` = (layers, KV heads, head size) for exact KV maths.

**Fit engine (catalog.py).** KV table entries for Phi-4, Phi-3/3.5-mini and
-medium and OLMo-2 (no grouped-query attention: 3-5x the rule of thumb); a
stored `kv_shape` wins next; architectures known to lack GQA get
`0.18 × params^0.6` GB per 1,024 tokens. Smartest never goes to a tight fit in
system RAM (cpu, unified, Apple split, or a split whose RAM side is snug) or a
last-tier quant; `explain_fit` words an Apple split's budget as the Mac's own
memory. In the filler passes `_originals_first` keeps an unrecognised
fine-tune from claiming its lineage before the original; Fastest near-ties
(within 5%) go to the better score; the curated bonus also covers a seed's
newer dated release from the same publisher.

**Engine lifecycle (llamaserver.py, runtime_install.py, specs.py).** See the
engine-lifecycle notes above for `--list-devices`, buffer-based GPU detection,
disk-read progress, per-install unusable notes and the update check. Also:
`gpu_arch` ("no kernel image is available") is a permanent failure of that
install; `GPUInfo.compute_capability` keeps CUDA 13 away from cards below 7.5;
Windows on ARM with an NVIDIA GPU tries the CUDA 13 arm64 build; the
"10.16" macOS compatibility answer is seen through (`specs.macos_release`:
`sysctl kern.osproductversion`, or a fresh Python with
`SYSTEM_VERSION_COMPAT=0`) and never treated as a real version;
`engine_can_use_gpu` uses `usable_plan`; a missing library is named with the
fix for this OS (`missing_library_hint`: the apt package on Linux, the VC++
Redistributable on Windows); CUDA build sizes say 0.6-0.8 GB and big backups
are priced on the confirmation screen; built-in graphics are described as
"Vulkan on the built-in graphics" in the hardware table and on the
confirmation screen (`specs.uses_built_in_graphics`).

**Game, prompts and review.** `screen_plan` judges what the player does:
give-up phrases count only at the start of a clause after the player's own
subject, with nothing afterwards that does something else (`_gives_up`); "I
win" only as a claim at the end or with "the game/this round"; orders need an
instruction to the referee, not a bare "you must". The mock has separate
give-up lines for the commute and for obstacles. `defang_plan` also breaks up
`<name:name>` / `<name_with_underscore>` tokens (Seed-OSS, Nemotron) and
EXAONE's `[|...|]`. A typed "quit" at a yes/no question raises
`UserChoseQuit` (a `UserQuit`): in the game it ends with the quit ending and
review, and the program exits 0. Menu numbers are capped at 6 digits (`int()`
refuses 4,300+). A disagreeing Choice is called a close call only for a Noul
between 0.35 and 0.65. With `--mock` the referee is labelled "the pretend
model (a simple scripted rule)" everywhere (welcome, verdict panel, lesson,
review, transcript). The review offers each local verdict (its JSON answer)
when there's no reasoning to show, and says "pick either, both or neither"
only when two or more questions follow.

**Privacy and keys.** A plain-http localhost Jev address is reached without
any `http_proxy` (`jev._LOCAL_OPENER`). An `OLLAMA_HOST` on another computer
is named ("another computer - your OLLAMA_HOST"), the player is told plans
and the model file would go there (over plain http if so) and asked before
the upload, and onboarding stops promising "nothing leaves this computer".
On Windows a saved key's `settings.json` gets an owner-only access list
(`config.restrict_to_owner_windows`, `icacls /inheritance:r /grant:r`); if
that fails, onboarding says so.

