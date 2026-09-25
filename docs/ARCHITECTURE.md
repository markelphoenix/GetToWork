# Get To Work — Architecture & Module Contracts

This document is both a guide for learners and the **binding contract** the
modules are built against. If code and this document disagree, fix one of them.

## The player's journey

1. `gettowork` launches → banner.
2. **Hardware check** (`specs.py`): OS, CPU, RAM, GPU(s)/VRAM, Apple unified
   memory, free disk. Shown in a table, with a "Learn" panel explaining why RAM
   and VRAM matter for LLMs.
3. **Model picker** (`catalog.py`): every curated open-weight model is rated
   `great / ok / tight / no` for *this* machine using our own transparent
   heuristic (no third-party data or paid services). The best fit is
   pre-selected. A "Learn" panel explains parameters, quantization (Q4_K_M), and
   the memory formula.
4. **Backend + download** (`setup_flow.py`, `backends/`, `download.py`):
   - **Ollama** (if its local server is running): `ollama pull hf.co/<repo>:<quant>`
     via its HTTP API — Ollama downloads the GGUF straight from Hugging Face.
   - **llama.cpp** (if `llama-cpp-python` is installed): the game downloads the
     GGUF from Hugging Face with `huggingface_hub`, then loads it in-process.
   - If neither is available, the game explains both options with copy-paste
     install commands and lets the player try again, or play in `--mock` mode
     (a scripted stand-in model, great for a quick look / classrooms / tests).
5. **Jev onboarding** (`onboarding.py`): "Do you want to enable Jev?" with a
   plain-language explanation. Yes → paste API key (hidden input), or "help me
   get one" (opens the TypeSafe website/docs in a browser, step-by-step), or
   back out to local-only at *any* prompt. Keys are validated with
   `GET /v1/models`. Saving the key to disk is opt-in only.
6. **The game** (`game.py`, `prompts.py`): the local LLM narrates a farcical
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
7. **Review** (`review.py`): two *independent* yes/no questions — show the Jev
   request/response JSON per round? show the local model's exposed reasoning
   (chain-of-thought) per round? — plus an optional transcript export
   (JSON + Markdown, API key never included).

## Package layout

```
src/gettowork/
  __init__.py        version
  __main__.py        `python -m gettowork` -> cli.main()
  types.py           shared dataclasses (read this first)
  ui.py              rich-based UI; all input/output goes through UI
  config.py          settings file + models dir
  specs.py           hardware detection
  catalog.py         curated model list + fit heuristic
  download.py        Hugging Face GGUF download
  reasoning.py       split chain-of-thought from answers
  backends/
    __init__.py      exports + detect_backends()
    base.py          LLMBackend ABC, BackendError
    ollama.py        OllamaBackend (HTTP to localhost:11434)
    llamacpp.py      LlamaCppBackend (llama-cpp-python, optional)
    mock.py          MockBackend (scripted, offline, deterministic)
  jev.py             Jev HTTP client + game questions + verdict parsing
  onboarding.py      Jev opt-in / API-key flow
  prompts.py         all LLM prompt text
  game.py            core game loop
  review.py          end-of-game review + transcript export
  setup_flow.py      hardware -> model -> backend -> download orchestration
  cli.py             argparse entry point
tests/               pytest, no network, no real models
```

Runtime deps: `rich` (MIT), `psutil` (BSD-3), `huggingface_hub` (Apache-2.0).
Optional: `llama-cpp-python` (MIT). HTTP to Ollama and Jev uses only the Python
standard library (`urllib.request`) so learners can see exactly what is sent.
Python ≥ 3.10. Must run on Windows, macOS, Linux.

## Module contracts

Signatures below are binding. Private helpers are free-form.

### types.py, ui.py, config.py, backends/base.py
Already written — read them. Do not change existing signatures; additive
changes only if truly necessary (and then update this doc).

### specs.py
```python
def detect_specs(models_path: Path | None = None) -> SystemSpecs
```
- psutil for RAM/CPU/disk (disk = free space of `models_path` or its nearest
  existing parent, defaulting to `config.models_dir()`).
- CPU name: `platform.processor()`, falling back to `/proc/cpuinfo` "model name"
  (Linux), `sysctl -n machdep.cpu.brand_string` (macOS), registry/`wmic` optional
  on Windows; never fail — "Unknown CPU".
- NVIDIA: `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits`
  (MiB). AMD (Linux): `rocm-smi --showmeminfo vram --json` best effort. Apple
  Silicon (`Darwin` + `arm64`): one GPUInfo(vendor="apple",
  vram_gb≈ 0.70 × RAM (0.75 if RAM ≥ 64 GB)), `unified_memory=True`.
- Every subprocess: timeout ≤ 5 s, catch everything, append a note on failure.
  `detect_specs()` must never raise.
```python
def describe_specs(specs: SystemSpecs) -> list[tuple[str, str]]   # rows for a 2-col table
```

### catalog.py
```python
MODEL_CATALOG: list[ModelEntry]          # ordered small -> large
def get_model(key: str) -> ModelEntry | None
def estimate_memory_gb(model: ModelEntry, context_tokens: int | None = None) -> float
def evaluate_fit(specs: SystemSpecs, model: ModelEntry) -> FitResult
def rank_models(specs: SystemSpecs, catalog: list[ModelEntry] | None = None) -> list[FitResult]
def recommend(specs: SystemSpecs, catalog: list[ModelEntry] | None = None) -> FitResult | None
MEMORY_FORMULA_EXPLAINER: str   # Markdown for a UI.teach panel
```
- Only permissively licensed (Apache-2.0 or MIT) models, GGUF repos on HF.
  Include small → large so every machine gets something, e.g. Qwen3 0.6B/1.7B/
  4B/8B/14B/32B + Qwen3-30B-A3B (MoE), SmolLM2 1.7B, Phi-4-mini (MIT),
  Mistral 7B Instruct v0.3, Mistral Small 3.2 24B, gpt-oss-20b (Apache-2.0,
  `ollama_ref="gpt-oss:20b"`). Prefer well-known GGUF publishers (unsloth,
  bartowski, ggml-org). Exact filenames are **not** hardcoded; download.py
  discovers them by `quant` tag.
- Memory estimate (document it in MEMORY_FORMULA_EXPLAINER):
  `weights = file_size_gb`; `kv = params-based estimate for context_tokens`
  (e.g. ≈ 0.00012 GB × params_b × context_tokens/1024, floor 0.05);
  `overhead = 0.6 GB`; total = weights + kv + overhead.
- Fit: compare to VRAM (dedicated), unified usable memory, or RAM
  (use `ram_total_gb` minus ~2.5 GB OS headroom, and consider available RAM for
  a caveat). Placement: gpu if fits VRAM; partial if VRAM ≥ 40% of need and
  RAM covers the rest; cpu if RAM fits; none otherwise. Verdict thresholds on
  headroom ratio (need / budget): ≤0.6 great, ≤0.85 ok, ≤1.0 tight, else no.
  Speed: gpu/unified → "fast"; partial → "usable"; cpu → by active params
  (≤4B "usable", ≤9B "slow", else "very slow"). Also "no" if disk free <
  file_size_gb + 1 (reason says so).
- `rank_models`: sorted with fitting models first, preferring the **largest
  model rated great/ok that isn't very slow**; `recommend` returns the first,
  or the smallest model if nothing fits (with verdict "no").
- The heuristic is our own, MIT licensed, provided with no warranty.

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
def download_gguf(entry: ModelEntry, ui: UI, dest_dir: Path | None = None,
                  *, hf_api=None, hf_download=None) -> Path
def download_custom_gguf(repo_id: str, quant: str, ui: UI, dest_dir: Path | None = None,
                         *, hf_api=None, hf_download=None) -> Path
```
- `pick_gguf_file`: case-insensitive; `.gguf` only; skip `mmproj` and split
  shards other than `-00001-of-`; prefer exact quant tag match, then a
  closest-quality fallback order (Q4_K_M, Q4_K_S, Q5_K_M, Q4_0, IQ4_XS, Q6_K,
  Q8_0, MXFP4, anything). Prefer files at repo root over subfolders.
- Uses `huggingface_hub.HfApi().list_repo_files` and `hf_hub_download`
  (injectable for tests). Skips download if the file already exists locally.
  Checks disk space first. Clear errors: repo not found, no GGUF, network
  failure, gated repo (tell user it needs HF login — `hf auth login`).
- Prints license + model page URL before downloading and reminds the user the
  weights come from Hugging Face under the author's license.

### backends/
```python
# backends/__init__.py
from .base import LLMBackend, BackendError
from .ollama import OllamaBackend
from .llamacpp import LlamaCppBackend
from .mock import MockBackend
def detect_backends() -> dict[str, tuple[bool, str]]   # {"ollama": (ok, why), "llamacpp": (ok, why)}

# ollama.py
class OllamaBackend(LLMBackend):
    name = "ollama"
    def __init__(self, model: str, host: str | None = None, *, http=None, think: bool | None = None)
    # host default: env OLLAMA_HOST (normalize "0.0.0.0:11434" / no scheme) or http://127.0.0.1:11434
    # http: injectable callable(method, url, json_body|None, timeout, stream: bool) for tests
    def is_available(self) -> tuple[bool, str]          # GET /api/version
    def has_model(self) -> bool                         # GET /api/tags
    def prepare(self, ui, entry=None) -> None           # POST /api/pull {"model":..., "stream": true}; show progress from streamed JSON lines (completed/total)
    def chat(...) -> LLMResult                          # POST /api/chat, stream false, options {temperature, num_predict, num_ctx}; format "json" when json_mode
    @property model_label
```
  - Thinking: send `"think": true` when `think` is True (default: True when the
    catalog entry is a reasoning model). If Ollama returns an error mentioning
    "think", retry once without it. Reasoning = `message.thinking` if present,
    else `split_reasoning(message.content)`.
  - If the answer is empty after stripping reasoning (thinking ate the token
    budget), retry once with `"think": false` and a larger `num_predict`.
  - Timeouts: chat 300 s, pull streaming per-read 600 s.
```python
# llamacpp.py
class LlamaCppBackend(LLMBackend):
    name = "llamacpp"
    def __init__(self, model_path: Path | None = None, entry: ModelEntry | None = None,
                 *, n_ctx: int = 4096, n_gpu_layers: int = -1, llama_factory=None)
    def is_available(self) -> tuple[bool, str]          # importlib.util.find_spec("llama_cpp")
    def prepare(self, ui, entry=None) -> None           # download_gguf if no path, then load Llama(model_path, n_ctx, n_gpu_layers, verbose=False)
    def chat(...) -> LLMResult                          # create_chat_completion; response_format {"type":"json_object"} when json_mode; split_reasoning
```
  - Lazy-import `llama_cpp` inside methods (optional dependency).
  - Same empty-answer retry: for Qwen3 append " /no_think" to the last user
    message and retry once.
```python
# mock.py
class MockBackend(LLMBackend):
    name = "mock"
    def __init__(self, seed: int = 0, *, think: bool = True)
```
  - Deterministic, offline. Recognises what it's being asked via the
    `purpose` marker the game places in the system prompt (see prompts.py:
    every system prompt contains a line `TASK: <purpose>` where purpose ∈
    {intro, outcome, judge, victory, ending_quit}). Returns plausible farcical
    text; for `judge` returns JSON `{"made_progress": ..., "explanation": ...}`
    (progress = plan has ≥ 4 words and isn't "nothing"/"give up").
  - Emits fake `reasoning` ("(mock reasoning) ...") so the review feature works.
  - Outcome/intro responses must contain a `CHALLENGE:` line (see prompts.py).

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
class SetupResult: backend: LLMBackend; entry: ModelEntry | None; specs: SystemSpecs
def run_setup(ui: UI, settings: Settings, *, args) -> SetupResult | None   # None = user quit
```
Hardware table → Learn panel → ranked model table (#, model, size, license,
verdict, where, speed, reason) → pick (default = recommended; also allow
`custom` HF GGUF repo for llama.cpp or any Ollama tag) → backend choice
(auto-detect; explain install steps for Ollama — https://ollama.com/download —
and `pip install llama-cpp-python` if neither is available; offer retry or
mock) → `backend.prepare(ui, entry)` → remember choices in settings (offer to
reuse next launch).

### cli.py
```python
def main(argv: list[str] | None = None) -> int
```
Flags: `--mock` (offline scripted model), `--backend {auto,ollama,llamacpp}`,
`--model KEY`, `--ollama-model TAG`, `--gguf PATH`, `--list-models`
(print catalog + fit and exit), `--specs` (print hardware and exit),
`--no-jev`, `--target N` (default 5), `--reset` (forget saved settings),
`--export-dir DIR`, `--version`. Ctrl+C anywhere exits cleanly with a friendly
line (exit code 130). Flow: banner → setup → Jev onboarding (skipped by
`--no-jev`) → Game.run() → run_review().

## Testing rules
- `pytest` only, no network, no real models, no real browser. Use MockBackend,
  fake Jev transports, fake `hf_api`/`hf_download`, and `UI(console=Console(file=io.StringIO()), input_fn=scripted)`.
- Each module has its own `tests/test_<module>.py`.

## Legal / safety rules for the codebase
- No model weights are bundled. Models are downloaded by the player from
  Hugging Face under each model's own license; the game shows that license.
- Only list Apache-2.0 / MIT models in the catalog.
- The hardware-fit heuristic is original code in this repo (MIT, no warranty).
- Not affiliated with TypeSafe AI, Hugging Face, Ollama, or any model author;
  names are used only to identify their products.
- API keys: never printed, never logged, never exported, redacted in review.
