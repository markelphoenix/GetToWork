# Distribution: double-click / Steam builds — contract

Goal: a player launches **Get To Work** from Steam (or double-clicks the test
build) and lands straight in the guided journey — hardware check → pick/download
a model if none is set up → optional Jev setup → the game loop — in the game's
**own window**. No terminal, no Python, no manual installs. Developers keep the
terminal version (`gettowork` in a shell).

This document is the binding contract for the modules below. The existing
contract (docs/ARCHITECTURE.md) still applies to everything else.

## Two front ends, one game

| How it's started | What runs |
|---|---|
| Steam / double-click `GetToWork(.exe/.app)` | `gettowork.launcher.gui_main()` → `gettowork.gui.app.run_gui()` opens a Tk window; the unchanged game (`cli.main`) runs in a worker thread and talks to the window through `UI`. |
| `gettowork` in a terminal (pip install, or the bundled console exe) | `gettowork.cli.main()` exactly as today. |

Tkinter is used because it ships with Python (python.org / actions/setup-python
builds and PyInstaller bundle Tcl/Tk), needs no extra dependency, and works in
Steam Deck Game Mode (any X11/XWayland window does). Tcl/Tk is BSD-licensed.

## Build layout (what ships)

PyInstaller **one-folder** build from `packaging/gettowork.spec`, one Analysis,
two executables in one folder:

```
Windows / Linux                        macOS
GetToWork/                             Get To Work.app/
  GetToWork(.exe)   windowed (GUI)       Contents/MacOS/GetToWork     (GUI, CFBundleExecutable)
  gettowork(.exe)   console  (CLI)       Contents/MacOS/gettowork     (CLI)
  _internal/        Python + libs        Contents/Frameworks/...      (Python + libs)
  engine/           bundled llama.cpp    Contents/Resources/engine/   (bundled llama.cpp)
  distribution.json                      Contents/Resources/distribution.json
  THIRD_PARTY_LICENSES.txt               Contents/Resources/THIRD_PARTY_LICENSES.txt
  README.txt                             (README.txt next to the .app in the artifact)
```

`engine/` holds one sub-folder per bundled llama.cpp build, each with the same
`install.json` marker format `runtime_install` already writes, plus
`"bundled": true`:

| OS | bundled engine builds |
|---|---|
| Windows x64 | `vulkan` (the official win-vulkan zip already contains the CPU backends), `cpu` |
| Linux x64 | `vulkan`, `cpu` |
| macOS arm64 | `metal` (the official macos-arm64 build; runs on CPU with `--device none`) |

`distribution.json`:
```json
{"schema": 1, "channel": "release", "engine_downloads": false,
 "engine_dir": "engine", "llama_cpp_tag": "b7xxx", "app_version": "0.2.0",
 "built_from": "<git sha>"}
```
`engine_dir` is relative to the folder containing `distribution.json`.

## Module contracts

### `src/gettowork/distribution.py` (new)
```python
@dataclass(frozen=True)
class Distribution:
    channel: str                     # "dev" (no distribution.json found), "release"
    engine_downloads: bool           # may the game download llama.cpp at runtime?
    engine_dirs: tuple[Path, ...]    # folders to scan for bundled engine builds
    llama_cpp_tag: Optional[str]
    app_version: str
    root: Optional[Path]             # folder holding distribution.json, if any
    @property
    def bundled(self) -> bool        # True when at least one engine dir exists
def find_distribution_file() -> Optional[Path]   # candidates below, first that exists
def load(*, refresh: bool = False) -> Distribution   # cached; never raises
```
Candidate locations, in order: `$GETTOWORK_DISTRIBUTION` (a file path);
`Path(sys.executable).parent / "distribution.json"`;
`Path(sys.executable).parent.parent / "Resources" / "distribution.json"` (macOS
.app); `Path(getattr(sys, "_MEIPASS", "")) / "distribution.json"`.
No file → `Distribution(channel="dev", engine_downloads=True, engine_dirs=(), ...)`.
Env overrides: `GETTOWORK_ENGINE_DIR` (extra engine dir, prepended; for tests and
the CI live check), `GETTOWORK_ALLOW_ENGINE_DOWNLOAD=1` (forces downloads on).
Malformed JSON → dev defaults + a logged note, never a crash.

### `runtime_install.py` + `backends/llamaserver.py` + `setup_flow.py` (changes)
- `installed_runtimes()` also returns bundled builds from
  `distribution.load().engine_dirs` (each sub-folder with an `install.json`).
  Bundled builds are **read-only**: never pruned (`prune_old_installs`), never
  modified. `mark_unusable()` for a bundled exe records the verdict in
  `config.runtime_dir()/bundled-unusable.json` (keyed by variant + tag) instead
  of writing into the build folder; `unusable_reasons()` merges it.
- `ensure_llama_server()`: when `distribution.load().engine_downloads` is False,
  never touch the network — candidates are the planned variants that are
  installed (bundled or earlier installs), in plan order; if the plan's GPU
  builds (e.g. CUDA) aren't bundled, use the best bundled one (Vulkan/Metal),
  and the bundled CPU build / CPU mode as the fallback. Nothing installed →
  `RuntimeInstallError` with: "The game's built-in engine is missing. On Steam:
  right-click Get To Work → Properties → Installed Files → Verify integrity.
  Otherwise re-download the game."
- The GPU→CPU fallback in `LlamaServerBackend` keeps working with bundled
  builds only (a failing `vulkan`/`metal` build falls back to the bundled `cpu`
  build or `--device none` on the same exe) and never downloads when downloads
  are off.
- `setup_flow`'s plan/confirmation screen shows the engine as
  "Built into the game (llama.cpp <tag>, <Vulkan/Metal> + CPU) — nothing to
  download" when a bundled build will be used.

### `packaging/fetch_engine.py` (new, used by CI; stdlib + `gettowork.runtime_install`)
```
python packaging/fetch_engine.py --os windows|linux|macos --arch x64|arm64 \
    --variants vulkan,cpu --dest build/engine [--tag auto|b7xxx] [--verify]
```
Picks the newest release that has **all** requested variants for that OS/arch
(or the given tag), downloads with size + sha256 digest checks (reuse
`runtime_install` helpers: `fetch_releases`, `select_assets`, download,
`safe_extract`, `find_server_executable`), extracts each variant to
`<dest>/<tag>-<variant>/`, writes `install.json` (`tag`, `variant`, `exe`,
`assets`, `licenses`, `"bundled": true`), copies each archive's LICENSE files,
prints the chosen tag on the last line as `LLAMA_CPP_TAG=<tag>`. `--verify` runs
`<exe> --version` for each extracted build that can run on the host and fails
if it doesn't exit 0. Honours `GITHUB_TOKEN` (never printed). Exit code 0/1.

### `src/gettowork/gui/` (new package)
- `terminal.py` — `class TerminalBuffer`: a pure-Python ANSI/VT screen model
  (no Tk import). `feed(text)`, `lines` (list of lines; each a list of
  `(text, Style)` runs), `dirty` line indices + `take_dirty()`, `cursor`.
  Supports: printable text incl. wide/emoji chars, `\n`, `\r`, `\b`, `\t`;
  CSI `A B C D E F G H/f(row;col) J(0,1,2) K(0,1,2) m`; SGR 0/1/2/3/4/7/9/
  22/23/24/27/29/30–37/38;5;n/38;2;r;g;b/39/40–47/48;…/49/90–97/100–107;
  `?25l/h` and other private modes ignored; OSC 8 hyperlinks (runs carry the
  URL) and other OSC ignored; unknown sequences dropped safely; bounded
  scrollback (default 10 000 lines). Must correctly render what rich's
  `Live`/`Status`/`Progress` emit (cursor-up + erase-line redraws).
- `bridge.py` — thread-safe plumbing between the game thread and Tk:
  `class GuiBridge` with `.stream` (writable text file object: `write`,
  `flush`, `isatty() -> True`, `encoding = "utf-8"`), `request_line(prompt,
  *, secret=False) -> str` (blocks the game thread; raises `EOFError` once the
  window closed), `show_choices(options: list[tuple[str, str]])`,
  `open_url(url) -> bool`, `closed` flag, `columns` (current width in chars).
  No Tk import (the Tk side polls a queue).
- `app.py` — `run_gui(argv: list[str] | None = None, *, selftest: bool = False)
  -> int`. Builds the window: dark theme, monospace font picked from
  installed fonts (Cascadia Mono, Consolas, SF Mono, Menlo, DejaVu Sans Mono,
  Noto Sans Mono, Liberation Mono, Courier New, TkFixedFont), read-only
  transcript `Text` with scrollbar rendering `TerminalBuffer` (colours, bold,
  underline, clickable OSC-8 links), an input bar (prompt label + entry;
  masked for secrets), a row of big buttons for the current menu choices
  (from `UI.choose` / `UI.confirm`, for mouse/touch/Steam Deck), Ctrl+= / Ctrl+-
  / Ctrl+0 font zoom (remembered in settings), window title "Get To Work",
  icon from `gettowork/assets/icon.png` if present, sensible default size
  (≥ 100×32 chars, fits 1280×800). Runs `cli.main(argv, ui=gui_ui)` in a
  daemon worker thread with `UI(console=Console(file=bridge.stream,
  force_terminal=True, color_system="truecolor", width=<cols>,
  legacy_windows=False, soft_wrap=False), input_fn=…, secret_fn=…,
  open_url_fn=…, choices_fn=bridge.show_choices)`; keeps the console width in
  sync on resize. When the game ends it shows "Press Enter or close the window
  to exit" instead of vanishing. Closing the window: mark closed, unblock
  input (EOF → `UserQuit` → the game's normal goodbye/cleanup), wait ≤ 8 s for
  the worker, then return so interpreter shutdown runs atexit (which stops
  llama-server). Exceptions before the window can open → write
  `config_dir()/logs/gui-crash.txt` and, if a terminal is attached, fall back
  to `cli.main`.
  `selftest=True` (flag `--gui-selftest`, used by CI): answers every prompt
  from a scripted list (Enter, Enter, five plans, then "n"s), requires the
  transcript to contain "YOU GOT TO WORK", writes the plain transcript to
  `$GETTOWORK_SELFTEST_OUT` if set, closes itself, exit code 0 on success /
  1 on failure / 2 on timeout (default 120 s).

### `ui.py` (additive change, owned by the GUI builder)
`UI.__init__(..., choices_fn: Optional[Callable[[list[tuple[str, str]]], None]] = None)`.
`choose()` calls `choices_fn(options)` before asking and `choices_fn([])` after
an answer; `confirm()` does the same with `[("y", "Yes"), ("n", "No")]`.
`can_hide_input()` returns True when a `secret_fn` was injected by the GUI
(add `hides_input: bool = False` init flag). No behaviour change for the terminal.

### `src/gettowork/launcher.py` (new) + `cli.py` changes
- `gui_main(argv=None) -> int`: entry for the windowed executable and the
  `gettowork-gui` script. Handles `--gui-selftest`; otherwise `run_gui(argv)`.
- `cli.main`:
  - after the review, **"Play again? [Y/n]"** loop reusing the running backend
    and Jev client (no re-setup, engine keeps running); "n" → goodbye.
  - first launch shows `notices.AI_CONTENT_NOTICE` once (remember
    `settings.extra["ai_notice_seen"] = True`).
  - Windows + frozen + the console belongs only to this process (double-clicked
    `gettowork.exe`): "Press Enter to close this window" before exiting,
    including after errors.
  - unchanged behaviour for tests/piped input.
- `pyproject.toml`: `[project.gui-scripts] gettowork-gui = "gettowork.launcher:gui_main"`;
  version bump to 0.2.0.

### `src/gettowork/safety.py` + `src/gettowork/notices.py` (new) + `game.py`/`prompts.py` changes
Steam requires guardrails for live-generated AI content.
```python
@dataclass(frozen=True)
class SafetyVerdict: ok: bool; category: Optional[str]; matched: Optional[str]
def check_text(text: str) -> SafetyVerdict          # hard blocks: sexual content, hate/slurs, self-harm, graphic gore, drugs
def soften(text: str) -> str                         # masks mild profanity ("d***"), keeps the story
def check_player_input(text: str) -> SafetyVerdict   # same lists, for what the player types
```
Word lists stored obfuscated (ROT13) in `safety_terms.py`, matched on
normalised text (case, leetspeak digits, spacing/punctuation tricks, accents),
whole-word so "Scunthorpe"/"assassin"/"classic" don't trip. Game: every model
narration/challenge/victory/judge explanation goes through `check_text` →
flagged → retry once with a stricter reminder → still flagged → a safe canned
line (reuse `backends/mock` content) and a note in the round record; always
`soften()` before display. Flagged player input is refused with a friendly
"Let's keep it family-friendly — try another plan!" and never sent to the
model or Jev. `notices.AI_CONTENT_NOTICE` (Markdown, shown once) and
`notices.STEAM_AI_DISCLOSURE` (store-page text) describe the local model, the
filter, and how to report problems.

### Packaging & CI
- `packaging/gettowork.spec`, `packaging/gui_entry.py`, `packaging/cli_entry.py`
  (replace `gettowork_entry.py`), `packaging/make_icon.py` →
  `src/gettowork/assets/icon.png` (committed; stdlib-only PNG writer),
  `packaging/collect_licenses.py` → `THIRD_PARTY_LICENSES.txt` (every bundled
  Python distribution's license files + Python + Tcl/Tk + PyInstaller
  bootloader note + llama.cpp licenses from the engine dir),
  `packaging/assemble.py` (copies engine + distribution.json + licenses +
  README.txt into the dist folder at the right per-OS place, then zips:
  `.zip` on Windows, `ditto`-made `.zip` of the .app on macOS, `.tar.gz` on
  Linux so the executable bit survives), `packaging/smoke_test.sh` (CLI) and
  `--gui-selftest` for the GUI.
- `.github/workflows/build.yml` (push to main, workflow_dispatch, PRs touching
  packaging/gui/launcher/distribution/workflows): per OS fetch engine →
  collect licenses → PyInstaller spec → assemble → smoke CLI → engine
  `--version` → GUI self-test (xvfb-run on Linux) → upload one archive per OS
  (retention 7 days). Plus the wheel/sdist job.
- `.github/workflows/ci.yml`: tests on PRs: ubuntu (3.10, 3.12) + windows
  3.12; on push to main / dispatch also macOS 3.12 + windows 3.10 (private-repo
  minutes: macOS counts 10×). Linux runs pytest under `xvfb-run` so Tk tests run.
- `.github/workflows/live-check.yml` (dispatch + weekly): real Hugging Face
  discovery (`gettowork --list-models`), real engine end-to-end (fetch CPU
  engine → download a tiny real GGUF → play a scripted game through the real
  `llama-server` → assert clean exit and no leftover process), optional real
  Jev round when the `TYPESAFE_API_KEY` secret is set.
- `packaging/steam/`: SteamPipe `app_build` / `depot_build` templates per OS
  with placeholders, `README.md` (upload + launch options: Windows
  `GetToWork/GetToWork.exe`, macOS `Get To Work.app`, Linux `GetToWork/GetToWork`,
  Steam Deck notes), `STORE_PAGE.md` (AI-content disclosure answers,
  third-party account notice for Jev, privacy, system requirements).
