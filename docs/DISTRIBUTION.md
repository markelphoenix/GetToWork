# Distribution: double-click / Steam builds

*As built for version 0.2.0. This is the binding contract for the modules,
scripts and workflows below; docs/ARCHITECTURE.md still applies to everything
else (its "Distribution" section summarises the same modules). If code and
this document disagree, fix one of them.*

Goal: a player launches **Get To Work** from Steam (or double-clicks the test
build) and lands straight in the guided journey - hardware check → pick and
download a model if none is set up → optional Jev setup → the game loop - in
the game's **own window**. No terminal, no Python, no manual installs, and no
programs downloaded while playing (the llama.cpp engine ships inside the
build; only the AI model is downloaded, once, after the player confirms).
Developers keep the terminal version (`gettowork` in a shell).

Get To Work is a free game. The source repository is private; the owner may
later sell a bundle or DLC with their other game, Gridfall, which needs no
change here (a bundle groups separate Steam apps; a DLC is its own app ID with
its own depots).

## Two front ends, one game

| How it's started | What runs |
|---|---|
| Steam / double-click `GetToWork(.exe)` / `Get To Work.app`; `gettowork-gui` after `pip install`; `python -m gettowork.launcher` | `gettowork.launcher.gui_main()` → `gettowork.gui.app.run_gui()` opens a Tk window; the unchanged game (`cli.main`) runs in a worker thread and talks to the window through `UI`. |
| `gettowork` in a terminal (pip install); `gettowork-cli(.exe)` inside a build | `gettowork.cli.main()` exactly as before (plus "Play again?" and the first-launch AI note, both only when a person is playing). |

Tkinter is used because it ships with Python (python.org / actions/setup-python
builds and PyInstaller bundle Tcl/Tk), needs no extra dependency, and works in
Steam Deck Game Mode (any X11/XWayland window does). Tcl/Tk is BSD-licensed.
The game's options work in both: `gettowork-gui --mock`, or Steam's
**Properties > General > Launch Options**.

## Build layout (what ships)

PyInstaller **one-folder** build from `packaging/gettowork.spec`: one Analysis,
two executables sharing it in one folder. The console program is called
`gettowork-cli`, not `gettowork`: Windows and macOS file systems ignore
upper/lower case, so `gettowork.exe` and `GetToWork.exe` would be one file.

```
Windows / Linux                           macOS
GetToWork/                                GetToWork/
  GetToWork(.exe)      windowed (GUI)       README.txt
  gettowork-cli(.exe)  console (CLI)        Get To Work.app/Contents/
  _internal/           Python + libs          MacOS/GetToWork       (GUI, CFBundleExecutable)
  engine/              bundled llama.cpp      MacOS/gettowork-cli   (CLI)
  distribution.json                           Frameworks/...        (Python + libs)
  THIRD_PARTY_LICENSES.txt                    Resources/engine/     (bundled llama.cpp)
  README.txt                                  Resources/distribution.json
                                              Resources/THIRD_PARTY_LICENSES.txt
```

On macOS the terminal version is `Get To Work.app/Contents/MacOS/gettowork-cli`.

Archives (one per OS, top folder `GetToWork/`), made by `packaging/assemble.py`:

| OS | Archive | Why this format |
|---|---|---|
| Windows x64 | `GetToWork-<version>-windows-x64.zip` | the usual format |
| macOS arm64 | `GetToWork-<version>-macos-arm64.zip`, made with `ditto` | keeps the app's symbolic links, permissions and signature |
| Linux x64 | `GetToWork-<version>-linux-x64.tar.gz` | keeps the executable bits (no `chmod` for players) |

On macOS the app is re-signed ad-hoc (`codesign --force --deep --sign -`)
after the engine and files are added, because adding files breaks the seal.
`LSMinimumSystemVersion` is 13.3, the minimum of the bundled engine. The
Linux build is made on ubuntu-22.04, so it needs glibc 2.35+ (SteamOS 3.5+,
Ubuntu 22.04+), like the official llama.cpp Linux builds.

`engine/` holds one `<tag>-<variant>/` folder per bundled llama.cpp build,
each with the same `install.json` marker format `runtime_install` writes
(`tag`, `variant`, `label`, `assets`, `exe`, `source`, `license`, `licenses`,
`installed_at`), plus `"bundled": true`, `license_files` (the build's
license texts, copied into its `licenses/` folder), `license_components`,
`architectures` (the model architectures that llama.cpp release knows - the
game offers only models it can load), and `vc_runtime` (Windows) or
`openssl` + `openssl_version` (Linux: the OpenSSL 3 libraries copied next to
`llama-server`, which Steam's Linux runtime lacks):

| OS | bundled engine builds |
|---|---|
| Windows x64 | `vulkan` (the official win-vulkan zip already contains the CPU backends), `cpu` |
| Linux x64 | `vulkan`, `cpu` |
| macOS arm64 | `metal` (the official macos-arm64 build; runs on the CPU with `--device none`) |

`distribution.json` (written by `assemble.py`):
```json
{"schema": 1, "channel": "release", "engine_downloads": false,
 "engine_dir": "engine", "llama_cpp_tag": "b7xxx", "app_version": "0.2.0",
 "built_from": "<git sha>"}
```
`engine_dir` is relative to the folder containing `distribution.json` (a list
of folders is accepted too); `built_from` is `$GITHUB_SHA` (or null).

## Module contracts

### `src/gettowork/distribution.py`
```python
@dataclass(frozen=True)
class Distribution:
    channel: str = "dev"                 # "dev" (no distribution.json found), "release"
    engine_downloads: bool = True        # may the game download llama.cpp at runtime?
    engine_dirs: tuple[Path, ...] = ()   # folders to scan for bundled engine builds
    llama_cpp_tag: Optional[str] = None
    app_version: str = __version__
    root: Optional[Path] = None          # folder holding distribution.json, if any
    built_from: Optional[str] = None     # git commit the build was made from
    notes: tuple[str, ...] = ()          # anything odd noticed while reading the file
    @property
    def bundled(self) -> bool            # True when at least one engine dir exists
def find_distribution_file() -> Optional[Path]   # candidates below, first that exists
def load(*, refresh: bool = False) -> Distribution   # cached, thread-safe; never raises
```
Candidate locations, in order: `$GETTOWORK_DISTRIBUTION` (a file path);
`Path(sys.executable).parent / "distribution.json"`;
`Path(sys.executable).parent.parent / "Resources" / "distribution.json"` (macOS
.app); `Path(sys._MEIPASS) / "distribution.json"` (only when set).
No file → `Distribution(channel="dev", engine_downloads=True, engine_dirs=())` -
except in a built game (`sys.frozen`), which then keeps downloads **off** and
looks in the `engine/` folder next to its executable (and
`Contents/Resources/engine` for a Mac app), with a logged note: a built game
never downloads programs, whatever happened to the file.
A file without a valid `engine_downloads` defaults to downloads off unless its
channel is "dev". Env overrides: `GETTOWORK_ENGINE_DIR` (extra engine dirs,
`os.pathsep`-separated, prepended; for tests and the live check),
`GETTOWORK_ALLOW_ENGINE_DOWNLOAD=1` forces downloads on, `=0` forces them off.
Paths from the environment (`GETTOWORK_ENGINE_DIR`, `GETTOWORK_DISTRIBUTION`)
are made absolute (the engine runs with its own folder as the working
directory); an entry that can't be expanded (`~nobody/...`) is skipped with a
logged note. Even `load()`'s last-resort fallback (an unexpected error) keeps a
built game's downloads off and its `engine/` folder.
Malformed JSON → the `engine/` folder next to the file (so a built game keeps
its engine), a logged note, never a crash; engine downloads stay on only in a
developer copy (off when `sys.frozen`).

### `runtime_install.py` + `backends/llamaserver.py` + `setup_flow.py`
- `installed_runtimes()` also returns bundled builds from
  `distribution.load().engine_dirs` (each sub-folder with an `install.json`
  naming an `exe` that exists; an engine dir that holds an `install.json`
  itself counts too). Bundled builds are **read-only**: never pruned
  (`prune_old_installs`), never modified. `mark_unusable()` for a bundled exe
  records the verdict in `config.runtime_dir()/bundled-unusable.json`
  (`BUNDLED_UNUSABLE_FILE`, keyed `"<tag>-<variant>"`, with the program's size
  and modification time as `"file"`) instead of writing into the build folder;
  `unusable_reasons()` merges it. A note only counts while the program is that
  very file: a repaired one (Steam's Verify integrity, a re-extracted test
  build) or another copy of the game with the same release gets a fresh check.
  The engine checks (`--version`, `--list-devices`) always run the program by
  its absolute path.
- New public helpers: `downloads_allowed()`, `is_bundled(exe)`,
  `find_installed(variant, specs)`, `choose_installed(plan, specs)`,
  `own_builds_first(runtimes)`, `engine_architectures()` (a built game's
  bundled builds' `architectures`, or None),
  `available_plan(specs)` (the builds a built game can really try; `--specs`
  shows it), `engine_summary()` (one line on where the engine comes from -
  `built into the game: llama.cpp <tag> (CPU, Vulkan); engine downloads off`
  in a built game; `--specs` prints it and the build check reads it),
  `relocate_engine(saved_exe)`, `own_build_instead(saved_exe)`,
  `bundled_builds_problem()`, `runtime_explainer()`, `other_engines_hint()`,
  `llama_cpp_python_possible()`, `ENGINE_MISSING_MESSAGE`, and - for
  `packaging/fetch_engine.py` - `releases_newest_first`, `fetch_release(tag)`,
  `download_asset`, `unpack_archive`, `finish_unpacked`,
  `install_marker(..., bundled=False, license_files=None)`.
- `ensure_llama_server()`: when downloads are off it never touches the
  network - candidates are the planned variants that are installed (bundled or
  earlier installs), in plan order - this copy's own builds first, of any type
  or release, and other installs (a developer copy's downloads sharing the
  settings folder) only when none of its own will do (`own_builds_first`); if
  the plan's GPU builds (e.g. CUDA) aren't bundled, the best bundled one
  (Vulkan/Metal) is used, and the bundled CPU build / CPU mode is the
  fallback. Nothing installed → `RuntimeInstallError`
  with `ENGINE_MISSING_MESSAGE`: "The game's built-in engine is missing. On
  Steam: right-click Get To Work → Properties → Installed Files → Verify
  integrity. Otherwise re-download the game."
- `relocate_engine(saved_exe)`: the saved engine path goes stale when a built
  game moves (another Steam library, the app dragged elsewhere); the same
  `<tag>-<variant>` build, or else the newest of that variant, is found again.
  A built game also swaps a saved engine that still exists but belongs to
  another copy (an older download still on disk, a developer copy sharing the
  settings folder) for its own build of the same type (`own_build_instead`;
  an engine the player set up themselves - no `install.json` - is never
  swapped). Setup's "Welcome back" and `LlamaServerBackend.prepare` use it.
- Every built-in build present but noted as unable to run here (on a Mac the
  one Metal build is also the CPU build): `bundled_builds_problem()` says so,
  with the built-in wording and the alternatives, instead of
  `ENGINE_MISSING_MESSAGE` (verifying the game's files can't fix that).
- Built-game wording: the `learn` page about the engine is
  `RUNTIME_EXPLAINER_BUILT_IN` (the engine ships inside the game; nothing is
  downloaded; no CUDA builds), and failure messages never suggest
  `pip install llama-cpp-python` or `--backend llamacpp` in a built game (it
  has no pip and leaves `llama_cpp` out) - Ollama is the alternative named.
  After a failed start, setup's advice follows the cause: a model download
  that didn't finish gets "check your internet connection and free disk
  space, then Try again", not another engine.
- The GPU→CPU fallback in `LlamaServerBackend` keeps working with bundled
  builds only (a failing `vulkan`/`metal` build falls back to the bundled
  `cpu` build or `--device none` on the same exe) and never downloads when
  downloads are off. An "unknown model architecture" in a built game says the
  built-in engine doesn't know that model yet (pick another; game updates
  bring newer engines) instead of downloading an engine - after first trying
  the newest build of that type already on disk, when the one that failed was
  older (another copy's). `LlamaServerBackend.close()` is final until the next
  `prepare()`: once the game is quitting (the window closed while the model
  was writing, and `atexit` runs `close()` while the game thread still waits),
  no crash fallback may start a new engine - there's no parent-death signal
  on macOS to stop it.
- Child programs of a built game get the system's libraries, not the game's
  bundled copies: on Linux the entry scripts restore `LD_LIBRARY_PATH`
  (`launcher.restore_system_library_path`); on Windows
  `llamaserver.windows_system_dll_search()` clears PyInstaller's inherited
  `SetDllDirectory(_internal)` while `llama-server.exe` starts (and while its
  `--version` / `--list-devices` checks run), then puts it back.
- `setup_flow`'s confirmation screen shows the engine as
  "Built into the game (llama.cpp <tag>, Vulkan + CPU) - nothing to download"
  (Metal + CPU on a Mac; just CPU when the plan has no graphics build), plus
  a line that llama.cpp's MIT license text ships as THIRD_PARTY_LICENSES.txt. `planned_engine_key` names the build a built
  game will really run, so speed calibration is keyed correctly.

### `packaging/fetch_engine.py` (used by CI; stdlib + `gettowork.runtime_install`)
```
python packaging/fetch_engine.py --os windows|linux|macos --arch x64|arm64 \
    --variants vulkan,cpu --dest build/engine [--tag auto|pinned|b7xxx] [--verify] \
    [--allow-missing-digest] [--vc-runtime auto|none|<folder>] [--linux-openssl auto|none|<folder>]
```
Picks the newest release that has **all** requested variants for that OS/arch
(`--tag auto`, the default), or the given tag via `GET /releases/tags/<tag>`.
`--tag pinned` is the release in `packaging/llama_cpp_tag.txt` (a `b<N>`
line, then one `<sha256>  <archive name>` line per archive the builds use -
`sha256sum`'s format; `#` comments allowed): the game builds use it, so every
OS of one commit - and every re-run - ships the same, reviewed engine, byte
for byte. An archive with no pinned fingerprint, one GitHub now reports a
different SHA-256 for (re-uploaded after the pin was reviewed), or bytes that
don't match the pin fail the build; moving the pin (tag and fingerprints) is a
deliberate, reviewed change. Downloads with size +
SHA-256 digest checks (GitHub's digest, or the pinned one with `--tag pinned`;
a missing digest is an error unless `--allow-missing-digest`), extracts each
variant safely to
`<dest>/<tag>-<variant>/` (`--dest` is made absolute first: CI passes a
relative one), writes `install.json` with `"bundled": true`, and gathers every
license text the build needs into `licenses/`:
1. the archive's own license files (the Linux/macOS tarballs carry llama.cpp's
   `LICENSE`; the Windows zips carry only `LICENSE-LLVM-OpenMP`, for
   `libomp.dll`);
2. the texts llama.cpp embeds in its programs (`cmake/license.cmake`: one
   `License for <name>` C string each - what `llama licenses` prints:
   llama.cpp, cpp-httplib, jsonhpp, and BoringSSL on Windows/macOS, whose
   official builds link it statically);
3. anything still missing from `REQUIRED_LICENSES[os]` (Windows/macOS:
   llama.cpp, cpp-httplib, jsonhpp, BoringSSL; Linux: the first three), plus
   BoringSSL whenever its code is found in a build and LLVM OpenMP whenever
   libomp ships - fetched from the same tag's source on
   raw.githubusercontent.com (BoringSSL's from the version the tag's
   `vendor/cpp-httplib/CMakeLists.txt` names). A copy of llama.cpp's own MIT
   text (its "The ggml authors" copyright line) is always required: another
   license file never stands in for it. OpenSSL's (from its own release's
   `LICENSE.txt`, the version read from the bundled libcrypto) is required
   whenever a build carries libssl/libcrypto. If a text can't be had, the
   build fails - a build is never bundled without its licenses.
`install.json` records `license_files` and `license_components`
(`{part: file}`). Windows builds also get Microsoft's Visual C++ runtime
(`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`) copied next to
`llama-server.exe` (`--vc-runtime auto` = the Windows build machine's
System32; `none` skips it with a warning; recorded as `vc_runtime`), so the
engine starts on a PC without the Visual C++ Redistributable; then every DLL
each `.exe`/`.dll` imports (read from its PE import tables, delay-loaded
ones included - works on any OS) must ship in the folder or be a Windows DLL
(`WINDOWS_SYSTEM_DLLS`, `api-ms-win-*`), or the build fails. Linux builds
get OpenSSL 3 (`libssl.so.3`, `libcrypto.so.3`, which the official Ubuntu
builds link but don't include) copied next to `llama-server` when a program
needs it (`--linux-openssl auto` = the Linux build machine's system library
folder for that processor - ubuntu-22.04 in CI; `none` skips it with a
warning; recorded as `openssl`/`openssl_version`): Steam's Linux runtimes
(1.0 = a Debian 10 "soldier" container, 3.0 "sniper" = Debian 11) have only
OpenSSL 1.1 and don't show the computer's `/usr` to the game. Then every
library each ELF program needs (its `NEEDED` entries, read from the file -
works on any OS) must be in the folder or on `LINUX_RUNTIME_LIBS` (glibc's,
the C++/GCC runtime, OpenMP, the Vulkan loader: what every supported Linux
and every Steam runtime has), or the build fails. The model architectures the
release knows are read from its `src/llama-arch.cpp` (`LLM_ARCH_NAMES`) and
recorded as `architectures` (the build fails if they can't be read). It prints
`LLAMA_CPP_TAG=<tag>` as the last line. `--verify` runs `<exe> --version`
(by its absolute path) for each build that can run on the host. Honours
`GITHUB_TOKEN` (never printed). Exit code 0/1.

### `packaging/engine_isolation_check.sh <engine-dir> [image]` (CI, Linux)
Starts every bundled engine build (`llama-server --version`, the way the game
starts it) inside a container without OpenSSL 3 (default
`debian:bookworm-slim`, plus the OpenMP and Vulkan loader libraries every
Steam runtime has) - proof that the build doesn't lean on the build machine's
libraries, as it can't under Steam. `--inside <engine-dir>` is the check
itself (bash only); it refuses to run on a system that has OpenSSL 3.

### `packaging/sign_windows.py <folders...> [--trusted-signing-dir DIR]` (CI, Windows)
Code-signs every `.exe` and `.dll` under the folders that isn't validly
signed already (`signtool verify /pa`; Microsoft's Visual C++ runtime keeps
its signature), SHA-256 with an RFC 3161 timestamp - via Azure Trusted Signing
(`AZURE_TRUSTED_SIGNING_ENDPOINT/ACCOUNT/PROFILE` + `AZURE_TENANT_ID`,
`AZURE_CLIENT_ID`, `AZURE_CLIENT_SECRET`; Microsoft's
`Azure.CodeSigning.Dlib.dll` from the NuGet package the workflow installs) or a
`.pfx` (`WINDOWS_SIGNING_PFX_BASE64`, `WINDOWS_SIGNING_PFX_PASSWORD`). Secrets
only ever come from the environment, the certificate file is deleted
afterwards, and a password never reaches the log. The build workflow runs it
only when those secrets exist and never for pull requests: Windows 11's Smart
App Control blocks unsigned programs outright, Steam installs included.

### `src/gettowork/gui/`
- `terminal.py` - `TerminalBuffer(*, max_lines=10_000, rows=24)`: a
  pure-Python ANSI/VT screen model (no Tk). `feed(text)`, `lines` (each a list
  of `(text, Style)` runs), `take_dirty()`, `cursor`, `text()`. Supports
  printable text incl. wide/emoji and zero-width characters, `\n` (new line
  and carriage return, like a cooked terminal), `\r`, `\b`, `\t`; CSI
  `A B C D E F G H f d` (cursor), `J K X` (erase), `m` (SGR: bold, dim,
  italic, underline, reverse, strike-through and their resets, 16 colours,
  256-colour palette, truecolor); private modes ignored; OSC 8 hyperlinks
  (runs carry the URL; long URLs capped) and other OSC ignored; unknown or
  over-long sequences dropped safely, even when split across chunks; bounded
  scrollback. Renders rich's `Live`/`Status`/`Progress` redraws correctly.
- `bridge.py` - `GuiBridge(*, columns=100, rows=32, opener=None)`: `.stream`
  (writable text file object: `write`, `flush`, `isatty() -> True`,
  `encoding = "utf-8"`), `request_line(prompt, *, secret=False) -> str`
  (blocks the game thread; `EOFError` once the window closed),
  `show_choices(options)`, `open_url(url) -> bool` (http/https only),
  `closed`, `columns`; the window side uses `poll()` (events `OUTPUT`,
  `PROMPT`, `CHOICES`, `FINISHED`), `submit(text, prompt_id=None)` (an answer
  names its prompt, so a late click can't answer the next question),
  `close()` and `detach()`. No Tk import.
- `app.py` - `run_gui(argv=None, *, selftest=False, game_main=None) -> int`
  and `GameWindow(root, argv, *, game_main, selftest, opener, env, font_size,
  ...)`. Dark theme; monospace font from `FONT_CANDIDATES` (Cascadia Mono,
  Consolas, SF Mono, Menlo, DejaVu Sans Mono, Noto Sans Mono, Liberation Mono,
  Courier New, then TkFixedFont; box-drawing look-alikes on Tk builds without
  Xft); read-only transcript with scrollbar (colours, bold, italic,
  underline, clickable OSC-8 links); input bar (prompt label + entry, masked
  for secrets, Up/Down history, Page Up/Down scrolls); an answer is echoed
  after its question and a long one rewrites the question's line so the two
  wrap together inside the window (`_echo_answer`); an answer that is - or
  contains - an API key (one given at a key question this session, kept only
  as a SHA-256, or anything `ui.looks_like_secret` matches) is echoed as
  "(hidden)" and never goes into the Up/Down history; a row of big buttons
  for the current choices (`UI.choose` / `UI.confirm` / `UI.pause`);
  Ctrl+= / Ctrl+- / Ctrl+0 zoom (also Cmd on macOS), saved as
  `Settings.extra["gui_font_size"]` when the window closes; F11 full screen,
  Escape leaves it; window title "Get To Work", icon from
  `gettowork/assets/icon.png`; default size ≥ 100×32 characters, fitted to the
  screen. **Steam Deck / Big Picture** (`SteamDeck=1`, `SteamGamepadUI=1`, or
  `GAMESCOPE_WAYLAND_DISPLAY`; `GETTOWORK_FULLSCREEN=1/0` overrides): full
  screen with a 16-point font, kept as long as 80×24 characters fit (on the
  Deck's 1280×800 screen it stays 16 points), plus a **Keyboard** button that
  asks Steam for its on-screen keyboard (`steam://open/keyboard`) on Deck /
  Big Picture. There the input area (question, buttons, input bar) sits at the
  **top** of the window, since Steam's keyboard covers the lower part of the
  screen without resizing the game; while the keyboard is up (from the
  Keyboard button until the answer is sent) the transcript gives up the bottom
  `KEYBOARD_SHARE` (45%) so its newest lines stay above it. A **Report a problem** button opens `notices.report_url()` -
  the game's Steam Discussions (`notices.STEAM_APP_ID`), or a Steam store
  search before the app ID is set - because Steam's overlay can't open over a
  Tk window on Windows, macOS or a Linux desktop. A click on the read-only
  transcript hands the focus back to the input bar, a paste there (Ctrl/Cmd+V)
  goes into the input bar, Ctrl+Shift+V pastes too, and the input bar has a
  right-click Paste / Copy / Cut menu. The model menu also gets buttons (its
  first picks, More models, Pretend model, How I chose, Quit).
  Windows: DPI-aware, own taskbar identity. Runs `cli.main(argv, ui=gui_ui)`
  in a daemon worker thread with `UI(console=Console(file=bridge.stream,
  force_terminal=True, color_system="truecolor", width=<cols>,
  legacy_windows=False, soft_wrap=False), input_fn=…, secret_fn=…,
  open_url_fn=bridge.open_url, choices_fn=bridge.show_choices,
  hides_input=True, window=True, pauses=True)`; keeps the console width in sync
  on resize. The bridge replaces lone UTF-16 surrogates (a model's JSON can
  carry half an emoji) with U+FFFD, since Tk on macOS and Linux refuses them; a
  redraw that fails anyway is logged, the question and buttons still appear,
  and the transcript is redrawn once (the crash report is written for the
  first of each kind of error, at most 5).
  Transcripts default to `default_export_dir()` (`~/Documents/Get To Work`,
  else `config_dir()/transcripts`) unless `--export-dir` is given; macOS
  `-psn_…` arguments are dropped. When the game ends it shows "Press Enter or
  close the window to exit". Closing the window (or SIGINT/SIGTERM/SIGHUP/
  SIGBREAK, or Cmd+Q): mark closed, unblock input (EOF → `WindowClosed`, a
  `UserQuit` that the optional Jev setup and the review let through, unlike
  Ctrl+C → the game's normal goodbye/cleanup, with no more model calls), wait
  ≤ 8 s for the worker, then return so interpreter shutdown runs atexit (which
  stops llama-server; the backend then refuses to start another). stdout and
  stderr are put back whatever replaced them, and in the window rich's
  spinners and progress bars don't redirect them at all. The window
  must open on the main thread (macOS). On X11 (Linux, Steam Deck) emoji are
  drawn as plain stand-ins exactly as many cells wide (`emoji_stand_in`): Tk
  there can't draw colour emoji, and with a libXft older than 2.3.5 trying
  ends the program with an X error. When a long question or a row of buttons
  makes the input area taller, the transcript keeps its newest lines in view
  (unless the player scrolled up). Exceptions before the window can open
  → `config_dir()/logs/gui-crash.txt` and, if a terminal is attached, fall
  back to `cli.main`; without a terminal (a double-click, Steam) a native
  message box (`show_error_dialog`: `MessageBoxW` on Windows, `osascript` on
  macOS, zenity/kdialog/xmessage on Linux; never in the self-test) gives the
  reason, the crash file's path and Steam's "Verify integrity of game files"
  step, so a failed start never looks like nothing happened; errors inside Tk callbacks are logged there too and the
  game carries on. An unexpected error inside the game itself is caught by
  `cli.main`, which writes the traceback to `config_dir()/logs/crash.txt`
  (`gettowork.crashlog`) and says where; in the window it offers to forget
  the saved settings right there instead of naming command-line options.
  `selftest=True` (flag `--gui-selftest`, used by CI): plays `--mock --no-jev`
  when no other options are given, answers every prompt with
  `SelftestPlayer` (by *what* is asked: Enter at pauses and menus, a plan
  whenever one is asked for, "n" to every yes/no; gives up after 120
  answers), also exercises zoom, requires the transcript to contain
  "YOU GOT TO WORK", writes the plain transcript to `$GETTOWORK_SELFTEST_OUT`
  if set, closes itself, exit code 0 on success / 1 on failure / 2 on timeout
  (default 120 s, `$GETTOWORK_SELFTEST_TIMEOUT`).

### `ui.py` (additive)
`UI.__init__(..., choices_fn: Optional[Callable[[list[tuple[str, str]]], None]] = None,
hides_input: bool = False, window: bool = False)`. `window=True` (the game's
window: `ui.in_window`) words hints for the window instead of naming commands
to type, and turns an `EOFError` from the input function into `WindowClosed`
(a `UserQuit` subclass). `choose()` calls `choices_fn(options)` before
asking and `choices_fn([])` after an answer; `confirm()` does the same with
`[("y", "Yes"), ("n", "No")]`, and `pause()` with `[("", "Continue")]`. A
failing `choices_fn` never breaks the question. `can_hide_input()` returns
True when `hides_input` is set. No behaviour change for the terminal.

### `src/gettowork/launcher.py` + `cli.py` changes
- `gui_main(argv=None) -> int`: entry for the windowed executable, the
  `gettowork-gui` script and `python -m gettowork.launcher`. Handles
  `--gui-selftest` (in a throwaway `GETTOWORK_HOME` unless one is set);
  otherwise `run_gui(argv)`. Imports neither Tk nor the game at import time.
- `restore_system_library_path()`: called first by both frozen entry scripts.
  PyInstaller's bootloader points `LD_LIBRARY_PATH` at the bundled `_internal`
  libraries; on Linux this puts the player's original value back (from
  `LD_LIBRARY_PATH_ORIG`, or removes it), so llama-server (and the graphics
  driver it loads), the browser opened for links and hardware checks use the
  system's own libraries. Only when frozen; never on Windows or macOS.
- `console_closes_on_exit()` / `wait_before_closing(input_fn=None)`: Windows
  + frozen + stdin is a TTY + the console belongs only to this process
  (double-clicked `gettowork-cli.exe`) → "Press Enter to close this window"
  before exiting, including after errors. Never true from source, in tests or
  with piped input.
- `cli.main`:
  - after the review, **"Play again? [Y/n]"** (`ask_to_play_again`) loops
    reusing the running backend and Jev client (no re-setup, engine keeps
    running); "n" → goodbye. Not asked after `quit`. The games of one session
    share one record of the "Learn" panels already shown (`Game(taught=...)`).
  - first launch shows `notices.AI_CONTENT_NOTICE` once
    (`show_ai_notice_once`, remembers `settings.extra["ai_notice_seen"] = True`).
  - both only when a person is playing (`player_is_present`: an interactive
    terminal or the game's window), so tests and piped input keep the old
    flow exactly.
  - a returning player who chose the local model only isn't asked about Jev
    again (one line says how: `--jev`); `--jev` asks again.
  - `--models-dir DIR` keeps models in another folder (a bigger drive) and is
    remembered (`Settings.models_dir`, applied with `config.use_models_dir`);
    it works from Steam's launch options on every OS.
  - hints name the terminal program as it is called in a build
    (`config.command_name()`: `gettowork-cli` when frozen).
- `pyproject.toml`: `[project.gui-scripts] gettowork-gui = "gettowork.launcher:gui_main"`;
  the version is dynamic, read from `gettowork.__version__` (0.2.0); the icon
  ships as package data.

### `src/gettowork/safety.py` + `safety_terms.py` + `notices.py` + `game.py`/`prompts.py` changes
Steam requires guardrails for live-generated AI content.
```python
@dataclass(frozen=True)
class SafetyVerdict:
    ok: bool; category: Optional[str] = None; matched: Optional[str] = None
    label: str   # property: "graphic gore", ...
CATEGORY_LABELS: dict[str, str]                      # sexual, hate, self_harm, gore, drugs
def check_text(text) -> SafetyVerdict                # hard blocks: sexual content, hate/slurs, self-harm, graphic gore, drugs
def soften(text) -> str                              # masks mild profanity ("d***"), keeps the story; idempotent
def check_player_input(text) -> SafetyVerdict        # same lists, for what the player types
def normalize(text) -> str; def hidden_note(verdict) -> str
```
Word lists stored obfuscated (ROT13) in `safety_terms.py` (`BLOCKED`,
`MILD_PROFANITY`, `EXEMPT_PHRASES`), matched on normalised text (case,
leetspeak digits inside words, spacing/punctuation tricks, accents,
look-alike letters, invisible characters, stretched letters), whole-word, so
"Scunthorpe"/"assassin"/"classic" don't trip. Game: every model
narration/challenge/victory/quit ending, local judge explanation, Jev label
and exposed reasoning goes through `check_text` → flagged story → retry once
with `prompts.safety_retry_messages` (adds `SAFETY_REMINDER`; the rejected
text is never repeated to the model) → still flagged → a safe canned line
(reusing `backends/mock` content); always `soften()` before display. The kept
reply in the round record becomes `hidden_note(...)`, and notes (categories
only, never the words) go to `RoundRecord.safety_notes` / `Game.safety_notes`.
Flagged player input is refused with "Let's keep it family-friendly - try
another plan!" and never sent to the model or Jev (no round used). A model the
player names themselves (the `custom` pick, `--model`) whose repo id or tags
mark it uncensored, safety-removed or adult (`hf_discovery.not_family_friendly`,
the same rule the model search uses) is refused before anything downloads
(`DownloadError` kind `not_family_friendly`), as `STEAM_AI_DISCLOSURE` says.
`notices.AI_CONTENT_NOTICE` (Markdown, shown once) and
`notices.STEAM_AI_DISCLOSURE` (store-page text, quoted verbatim in
`packaging/steam/STORE_PAGE.md`; a test keeps them equal) describe the local
model, the filter, and how to report problems (`REPORT_HOW`: the window's
**Report a problem** button, which opens the game's Steam Discussions, or the
store page's Discussions; Steam's overlay only in a Deck's Game Mode). The
engine's raw answer kept for the saved transcript goes through the same filter
(blocked thinking drops it; everything else is softened).

## Packaging & CI

- `packaging/gettowork.spec` (one Analysis, `GetToWork` windowed +
  `gettowork-cli` console, `BUNDLE` on macOS with bundle id
  `com.markelphoenix.gettowork` and `LSBackgroundOnly: false` - PyInstaller
  would otherwise make the app background-only (no Dock icon, no keyboard
  focus) because COLLECT's last program is the console one; hidden imports for rich, truststore,
  huggingface_hub, the whole game and tkinter; excludes tests, Pillow,
  llama_cpp and `readline` - GNU readline is GPL-3.0 and would bring
  libreadline/libtinfo into the Linux build; no UPX; `optimize=0` keeps
  docstrings), `packaging/gui_entry.py`
  and `packaging/cli_entry.py` (replacing `gettowork_entry.py`),
  `packaging/make_icon.py` → `src/gettowork/assets/icon.png` (committed;
  stdlib-only PNG writer; the spec turns it into `.ico`/`.icns` with Pillow),
  `packaging/collect_licenses.py --engine-dir DIR --app-dir dist/GetToWork --out FILE` →
  `THIRD_PARTY_LICENSES.txt` (every bundled Python distribution's license
  files, Python, Tcl/Tk, PyInstaller bootloader note, llama.cpp licenses from
  the engine dir, and every native library PyInstaller copied into
  `_internal/` - OpenSSL, libffi, and on Linux X11, Xft, fontconfig, FreeType,
  libpng, ncurses, libuuid, ... - with the package's own
  `/usr/share/doc/<package>/copyright` on a Debian/Ubuntu runner, else a
  maintained notice; a library with no license entry, or GNU readline/gdbm,
  fails the build), `packaging/assemble.py --dist dist --engine-dir DIR --os
  windows|linux|macos [--arch] --out DIR [--licenses FILE]` (copies engine +
  distribution.json + licenses + README.txt into the right per-OS place,
  re-signs the Mac app inside out - each engine Mach-O file ad-hoc, then the
  app with `--deep` - packs the archive, prints and writes `ARCHIVE`,
  `APP_DIR`, `GUI`, `CLI`, `ENGINE` step outputs),
  `packaging/smoke_test.sh [--engine-dir DIR] <cli>` (bundled engines'
  `--version`, info commands, a whole pretend-model game; with `--engine-dir`
  it also checks that `--specs` says `built into the game: llama.cpp <tag>`
  with engine downloads off, so a build that lost its `distribution.json`
  fails) and `--gui-selftest` for the window.
- `.github/workflows/build.yml` (push to main, workflow_dispatch, and pull
  requests touching `packaging/**`, `src/gettowork/gui/**`, `launcher.py`,
  `distribution.py` or the workflow): per OS (windows-latest, ubuntu-22.04,
  macos-latest) fetch the pinned engine (`--tag pinned --verify`) →
  PyInstaller spec → collect licenses (`--app-dir`) → (Windows, once signing
  secrets exist) `sign_windows.py` → assemble → smoke CLI + bundled engine →
  (Linux) `engine_isolation_check.sh` → GUI self-test (xvfb-run on Linux; in a `GETTOWORK_HOME`
  under `runner.temp`, so its crash report is printed when the window fails)
  → a live Hugging Face search (an outage only warns; a crash of the built
  game fails the build) → upload each game archive as it is
  (`actions/upload-artifact@v7` with `archive: false`: the artifact is the
  archive, named after it, so there is one layer to unpack; not on pull
  requests; retention 3 days). Pushes that only change docs or tests
  (`paths-ignore`) build nothing. Plus the wheel/sdist job (twine check, icon
  in the wheel, smoke test and GUI self-test of the installed wheel; uploaded
  too, 7 days).
- `.github/workflows/ci.yml`: pytest on pull requests on ubuntu (3.10, 3.12)
  + windows 3.12; on push to main also windows 3.10; manual runs add macOS 3.12
  (the Mac app is still smoke- and self-tested by build.yml on every push to main)
  (private-repo minutes: macOS counts 10×, Windows 2×). Linux runs pytest
  under `xvfb-run` so the Tk tests run.
- `.github/workflows/live-check.yml` (dispatch + weekly, Linux only): real
  Hugging Face discovery (`gettowork --list-models --refresh-models`), real
  engine end-to-end (fetch the CPU engine - the pinned release and the newest
  one - and play a scripted game with a tiny real GGUF through each real
  `llama-server`, with `GETTOWORK_ENGINE_DIR` and downloads off like the Steam
  build → assert a clean exit and no leftover process; the newest release
  failing while the pinned one works means "don't move the pin yet"),
  optional real Jev round when dispatched with the
  `TYPESAFE_API_KEY` secret set.
- `packaging/steam/`: SteamPipe `app_build_*.vdf` / `depot_build_*.vdf`
  templates per OS with placeholders (`<APP_ID>`, `<DEPOT_ID_WIN>`,
  `<DEPOT_ID_MAC>`, `<DEPOT_ID_LINUX>`, `<VERSION>`), `README.md` (the
  release checklist: depots, launch options - Windows
  `GetToWork\GetToWork.exe`, macOS `Get To Work.app`, Linux
  `GetToWork/GetToWork` - the Visual C++ 2015-2022 x64 redistributable,
  code-signing the Windows build (the secrets `sign_windows.py` reads),
  uploading from macOS/Linux, setting `notices.STEAM_APP_ID`, Steam Deck
  notes - the Linux runtime (4.0 if offered; the engine carries its own
  OpenSSL 3, so 1.0/3.0 work too) - the Gridfall bundle note),
  `STORE_PAGE.md` (AI-content disclosure answers, optional third-party
  service notice for Jev, privacy, system requirements).

## Changes from the first draft of this contract

- The console program is `gettowork-cli(.exe)` (not `gettowork(.exe)`), so it
  can sit next to `GetToWork(.exe)` on case-insensitive file systems.
- The macOS archive's top folder is `GetToWork/` holding the app and
  `README.txt` (the draft left the README's place open); the app is re-signed
  ad-hoc after assembly.
- `Distribution` gained `built_from` and `notes`; `engine_dir` may be a list;
  `GETTOWORK_ENGINE_DIR` takes several folders and
  `GETTOWORK_ALLOW_ENGINE_DOWNLOAD=0` forces downloads off.
- `UI.pause()` also offers a button ("Continue"); `run_gui` gained
  `game_main=` (tests), a default transcript folder, full screen and a
  Keyboard button on Steam Deck, and F11.
- `runtime_install` gained `relocate_engine` (a built game that moved finds
  its engine again) and `available_plan`; a built game never updates its
  engine for a new model architecture (it explains instead).
- `fetch_engine.py` gained `--allow-missing-digest`; it always makes sure
  llama.cpp's own LICENSE ships (the Windows zips carry only
  LICENSE-LLVM-OpenMP), plus the embedded texts of the parts compiled into the
  engine (BoringSSL, cpp-httplib, jsonhpp), and adds the Visual C++ runtime to
  the Windows builds (`--vc-runtime`).
- The game builds are not uploaded for pull requests, and the Linux build
  runs on ubuntu-22.04 for glibc 2.35 compatibility.
- Found while testing a real build: the frozen entry scripts restore
  `LD_LIBRARY_PATH` for child programs (`launcher.restore_system_library_path`),
  the Windows engine starts without the inherited DLL directory,
  the Mac app sets `LSBackgroundOnly: false`, the engine's Mac binaries are
  signed before the app, and the X11 window swaps emoji for stand-ins and keeps
  the newest transcript lines visible when the input area grows.
- After review: the game builds pin their llama.cpp release
  (`packaging/llama_cpp_tag.txt`, `fetch_engine.py --tag pinned`); game
  artifacts are uploaded as the archive itself and kept 3 days; the license
  file covers the native libraries PyInstaller bundles, and `readline` is
  excluded; `--specs` reports the built-in engine and the smoke test checks
  it; a built game without a readable `distribution.json` still never
  downloads; the window has a Report a problem button (Steam's overlay can't
  open over it), keeps the Deck's 16-point font, pastes into the input bar
  from anywhere, and gives the model menu buttons; closing the window ends the
  game at once (`WindowClosed`) and the engine is never restarted while the
  game quits; game errors are saved to `logs/crash.txt`.
- After the Steam-readiness review: the pin also fixes each archive's
  SHA-256; the Linux engine carries its own OpenSSL 3 (Steam's Linux runtime
  has none) and every engine program's needed libraries are checked, in CI
  also by starting the engine in a container without OpenSSL 3
  (`engine_isolation_check.sh`); each bundled build records the model
  architectures its release knows, and a built game leaves models it couldn't
  load out of the menu; a built game prefers its own engine builds over any
  other install; a switch from a graphics build to the CPU build is retried
  on later launches when its cause may have gone; the Windows build is
  code-signed once signing secrets exist (`sign_windows.py`), and Smart App
  Control / antivirus blocks of the engine are explained; the window's Report a
  problem button always shows its link (through Steam's browser on a Deck),
  the transcript copies with Ctrl+C or its right-click menu, menu buttons never
  read alike, and the window lets go of every Tk object when it closes.
