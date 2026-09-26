# Contributing to Get To Work

Thank you for helping! Get To Work is a learning project first, so clear,
friendly code and docs matter as much as features. Bug reports, typo fixes,
new "Try this" exercises and better explanations are all very welcome.

The source repository is private, so contributing means being given access by
the maintainer first. Everything below assumes you have it.

## Development setup

You need Python 3.10 or newer and git.

```bash
git clone https://github.com/markelphoenix/GetToWork
cd GetToWork
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
```

The `-e` ("editable") install means your code changes take effect
immediately, with no reinstall needed.

Run the tests:

```bash
python -m pytest -q
```

The tests of the game's window open real Tk windows. They skip themselves
cleanly on a Python without Tk or a computer without a screen; on a Linux
machine without a desktop, run them on a virtual screen:
`xvfb-run -a python -m pytest -q` (the `xvfb` package; CI does the same).

Play without downloading anything:

```bash
gettowork --mock            # scripted offline model, in this terminal
gettowork-gui --mock        # the same, in the game's own window
gettowork --mock --target 2 # an even quicker game
gettowork --specs           # hardware detection only
gettowork --list-models     # discovery + fit engine only
```

Optional extra: `pip install -e ".[dev,llamacpp]"` installs `llama-cpp-python`
for `--backend llamacpp`. CI doesn't install it, so tests must never need it.

## Ground rules for code

- **Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.** It is the
  binding contract between modules; [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md)
  covers the game window, the built game and its packaging. If you change a
  public signature or behaviour, update the right document in the same pull
  request.
- **Python 3.10+** with `from __future__ import annotations`. No APIs that only
  exist in 3.11 or later (such as `tomllib`) outside of tests with a fallback.
- **Windows, macOS and Linux** (and so Steam Deck) must all work. Use
  `pathlib`, list-style subprocess arguments (never `shell=True`), and give
  every external command a short timeout.
- **Dependencies stay small**: `rich`, `psutil` and `huggingface_hub`, plus
  Python's own `tkinter` for the window. HTTP to GitHub, llama-server, Ollama
  and Jev uses only the standard library (`urllib.request`), so learners can
  see exactly what is sent. Please open an issue before adding a dependency;
  any new one must be free and permissively licensed (it ends up inside the
  game builds, and in `THIRD_PARTY_LICENSES.txt`).
- **One game, two front ends.** The game talks to the player only through
  `UI` (`ui.py`), so the same code runs in a terminal and in the window
  (`gui/app.py` runs `cli.main` on a worker thread). Game code never imports
  Tk, and nothing imports Tk at import time.
- **Written for learners**: clear names, a docstring on every public function,
  and short comments where a concept isn't obvious (not on every line).
- **Warm, plain-English messages.** Every error the player sees should say what
  happened and what to do next, in one or two sentences, without jargon.
- **Escape untrusted text.** Anything from a model, from Jev or from the player
  is printed with `rich.markup.escape` (or as a `rich.text.Text`).
- **Family-friendly, always.** Every piece of model text the player can see
  goes through `safety.check_text` (then `safety.soften`), and every typed plan
  through `safety.check_player_input` before it reaches a model or Jev. See
  [The family-friendly filter](#the-family-friendly-filter) below.
- **Secrets are sacred.** API keys are never printed, logged, exported or put in
  `repr()`. They're shown only as `****abcd`.
- **No telemetry**, analytics or "phone home" features, ever.

## Tests: no network, no real models

The whole suite must run offline in a few seconds. Tests must never touch the
network, real model files, real subprocess downloads or a real browser. Every
module that talks to the outside world accepts a fake:

| Module | What to inject |
|--------|----------------|
| `ui.py` | `UI(console=Console(file=io.StringIO()), input_fn=..., secret_fn=..., open_url_fn=...)`; the window also passes `choices_fn=...` (menu buttons) and `hides_input=True` |
| `hf_discovery.py` | `discover_models(api=FakeHfApi(), cache_path=tmp_path / "c.json", clock=...)` |
| `download.py` | `download_gguf(..., hf_api=fake_api, hf_download=fake_download)` |
| `runtime_install.py` | `http=` (an object with `request(method, url, *, headers, body, timeout)`), `runtime_root=tmp_path` |
| `distribution.py` | environment variables `GETTOWORK_DISTRIBUTION` (a `distribution.json`), `GETTOWORK_ENGINE_DIR` (a folder of engine builds), `GETTOWORK_ALLOW_ENGINE_DOWNLOAD`, then `distribution.load(refresh=True)` |
| `backends/llamaserver.py` | `http=`, `popen=`, `installer=`, `downloader=`, `sleep=`, `clock=`, `log_dir=` |
| `backends/ollama.py` | `http=` |
| `backends/llamacpp.py` | `llama_factory=`, `downloader=` |
| `jev.py` | `JevClient(key, transport=fake_transport)` |
| `setup_flow.py` | `SetupServices(detect_specs=..., discover_models=..., make_backend=..., installed_runtimes=..., custom_entry=...)` |
| `cli.py` | `main(argv, ui=..., services=SetupServices(...))` |
| `gui/app.py` | `run_gui(argv, game_main=fake_main)` or `GameWindow(root, argv, game_main=..., opener=..., env=...)`; `GuiBridge` and `TerminalBuffer` need no Tk at all |
| game logic | `MockBackend(seed=...)` |

`tests/test_e2e.py` drives the real `cli.main()` from banner to review with
these fakes. It is the one place that starts a subprocess: a tiny Python
stand-in for `llama-server` (localhost only, POSIX only), so process start-up,
health polling, chat, the speed test and shutdown (including Ctrl+D and
SIGTERM mid-game) are exercised for real.

A tiny example:

```python
import io
from rich.console import Console
from gettowork.ui import UI

def scripted(*answers):
    replies = iter(answers)
    return lambda prompt: next(replies)

def test_something(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))  # never touch the real settings
    out = io.StringIO()
    ui = UI(console=Console(file=out, width=100), input_fn=scripted("yes", "quit"))
    ...
    assert "Welcome" in out.getvalue()
```

Each module has its own `tests/test_<module>.py` (the window's are
`tests/test_gui_app.py`, `tests/test_gui_bridge.py` and
`tests/test_gui_terminal.py`; the build scripts' are `tests/test_packaging.py`
and `tests/test_fetch_engine.py`). `tests/test_docs.py` checks that the
documentation still matches the code (formulas, numbers, flags, links and the
Jev wire-format examples), so if you change a constant it may ask you to
update [docs/LEARN.md](docs/LEARN.md) too. That's on purpose.

## How the model list works

There are two sources of models, and it helps to know both.

### The curated seed list (`catalog.py`)

`MODEL_CATALOG` is a small, hand-checked list of models. It is the offline
fallback when Hugging Face can't be reached, and seeds earn a small "known
good" bonus in scoring. Live discovery is the main source. Rules for adding
or changing a seed:

1. **License: Apache-2.0 or MIT only.** Check the *original* model's card, not
   just the GGUF repo.
2. **A GGUF repo from a trusted publisher** (see `TRUSTED_PUBLISHERS`), with
   the file names matching `<Name>-<QUANT>.gguf`, or pass `gguf_file=`.
3. **Real sizes.** Copy each quant's file size (GB) from the repo's file list
   into `sizes`, in the same order as `quants`.
4. **Keep the list ordered small to large** by `params_b` (total parameters).
5. Set `reasoning=True` for models that show their thinking, `active_params_b`
   for Mixture-of-Experts models, and `native_context` from the model card.
6. If you know the model's layer count, KV heads and head size (from its
   `config.json`), add them to `_KV_SHAPES` for exact KV-cache maths.
7. Update `EXPECTED_REPOS` in `tests/test_catalog.py`, then run the tests.

### Live discovery filters (`hf_discovery.py`)

Discovery searches the Hub once per trusted publisher plus once globally, then
screens every result with `rejection_reason()`, which returns a plain-English
reason (or `None` for a good candidate). The knobs, all near the top of the
file:

- **Family-friendly**: `_UNSAFE_RE` (uncensored, abliterated, NSFW, adult
  roleplay...).
- **Storytellers only**: `_NOT_CHAT_WORDS` and `_NOT_CHAT_PIPELINES` (coders,
  embedders, rerankers, vision, speech...).
- **Chat-tuned only**: `_BASE_WORDS`, `_INSTRUCT_RE` and `_CHAT_FAMILIES`.
- **Licenses**: `catalog.PERMISSIVE_LICENSES` (Apache-2.0, MIT) unless the
  player passes `--all-licenses`; an unknown license is excluded.
- **Size and popularity**: `MIN_PARAMS_B`, `MAX_PARAMS_B`,
  `MIN_DOWNLOADS_UNTRUSTED`.
- **Thinking models**: `thinking_mode_for` reads each model's chat template
  (`_THINK_SWITCH_RE`: it can be asked to skip thinking; `_FORCED_THINK_RE`:
  it always thinks), falling back to the names in `_ALWAYS_THINKS_RE` and
  `_SWITCHABLE_THINKS_RE`. Models that always think stay off the short menu.
- **Family names**: `_FAMILIES`.

Every filter change needs a test in `tests/test_hf_discovery.py` showing a
model that is now kept or left out, with a fake Hub response. The saved model
list records `RULES_VERSION` (a fingerprint of all these filters), so after a
filter change old lists are refreshed - and re-screened when offline -
without you doing anything. If you change *what gets saved* in the cache,
bump `CACHE_SCHEMA_VERSION` so old caches are ignored rather than misread.

### The fit engine (`catalog.py`, `perf.py`)

The memory, speed and scoring constants are module-level names (`OVERHEAD_GB`,
`EFFICIENCY`, `W_SPEED`...) with comments explaining them. If you recalibrate
them, explain your evidence in the pull request (for example, measured speeds
on your hardware), update the calibration notes in `perf.py`, and keep
`tests/test_perf.py`, `tests/test_catalog.py` and [docs/LEARN.md](docs/LEARN.md)
in sync.

## The family-friendly filter

Steam asks games with live-generated AI content for guardrails, and the game
is for all ages, so the filter is part of the game, not an add-on:

- `safety.py` holds the logic: `check_text` (hard blocks), `soften` (masks
  mild swearing) and `check_player_input` (typed plans), all on normalised
  text (case, accents, look-alike letters, leetspeak, spacing and symbol
  tricks), matching whole words only.
- `safety_terms.py` holds the word lists, **scrambled with ROT13** so nobody
  has to read them. To add a term, run
  `python -c "import codecs; print(codecs.encode('new term', 'rot13'))"` and
  paste the result into the right list. Never add a word with a common
  innocent meaning in a story about getting to work (the file's docstring
  explains why).
- Every change needs tests in `tests/test_safety.py`: the new term is caught
  (including a disguised spelling), and innocent words that contain it are
  not.
- `game.py` decides what happens when text is blocked (ask once more with
  `prompts.safety_retry_messages`, then a built-in line). If you change that,
  or what the filter covers, update `notices.STEAM_AI_DISCLOSURE` too: it is
  the text on the Steam store page, so it must stay accurate
  (`packaging/steam/STORE_PAGE.md` quotes it, and a test checks the two
  match).

## License rules

- Get To Work's code is currently **MIT licensed** (see [LICENSE](LICENSE)).
  By contributing, you agree that your contribution is licensed under the
  license in the LICENSE file at the time you contribute (today: MIT).
- Only contribute code you wrote yourself or that comes from a compatible
  permissive license (MIT, BSD, Apache-2.0), with attribution added to
  [NOTICE.md](NOTICE.md). Don't paste code from sources with unclear or
  incompatible licenses.
- **Never commit model weights, engine binaries or other large downloads.** The
  game builds fetch the official llama.cpp engine in CI
  (`packaging/fetch_engine.py`), and players download models at their request;
  `.gitignore` already blocks `*.gguf`.
- Anything new that ships inside the game builds must have its license text
  collected into `THIRD_PARTY_LICENSES.txt` (`packaging/collect_licenses.py`)
  and be listed in [NOTICE.md](NOTICE.md).
- Seeds and default discovery results stay **Apache-2.0 / MIT only**.
- The hardware-fit heuristic is original code. Keep it that way: don't copy
  tables or code from other projects without checking their license first.
- Product names (Hugging Face, llama.cpp, Ollama, TypeSafe AI, Jev, Steam...)
  are used only to identify those products. Don't add logos or anything
  implying endorsement.

### A note for the maintainer: the license of future versions

The repository is now private and the game is heading to Steam as a free
game, but the [LICENSE](LICENSE) file is unchanged: the code is still MIT
licensed, and everyone who received a copy under MIT keeps those rights. Please
decide, before the next public release, whether future versions stay MIT or
move to another license (for example "all rights reserved" for the game while
keeping the learning material open). If it changes, update LICENSE, the
README's license section, this file and the `license` field in
`pyproject.toml` together - and remember that code contributed under MIT
stays available under MIT.

## Pull requests

1. Open an issue first for big changes, so we can agree on the approach.
2. Keep pull requests focused; one topic each.
3. Make sure `python -m pytest -q` passes. On pull requests CI runs it on
   Linux (Python 3.10 and 3.12) and Windows (3.12); once a change lands on
   `main` it also runs on Windows (3.10), and the `build` workflow smoke- and
   self-tests the Mac app; the macOS unit tests run on manual runs ("Run
   workflow"). (The repository is private, where GitHub counts macOS minutes
   10x and Windows minutes 2x.)
4. Update the docs (README, LEARN, ARCHITECTURE, DISTRIBUTION) when behaviour
   changes.

Found a security problem, such as an API key leak or an unsafe download?
Please contact the maintainer privately rather than opening an issue.

## Code of conduct

Be kind, patient and welcoming, especially to beginners: this project exists
so people can learn. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Harassment or disrespect isn't tolerated. If something goes wrong, contact the
maintainer through GitHub.

## Test builds

`.github/workflows/build.yml` makes the double-click / Steam builds for
Windows, macOS (Apple Silicon) and Linux, plus the wheel and source archive,
on every push to `main` (on pull requests only when they touch the packaging,
and without uploading the game builds). For each OS it:

1. fetches the official llama.cpp engine builds that ship inside the game -
   the release pinned in `packaging/llama_cpp_tag.txt`, whose archives must
   match the SHA-256 fingerprints pinned there too - and checks them
   (`packaging/fetch_engine.py --tag pinned --verify`; Linux builds also get
   OpenSSL 3 copied in, since Steam's Linux runtime has none, and every
   program's needed libraries are checked);
2. builds the two programs, `GetToWork` (the window) and `gettowork-cli`
   (the terminal version), with PyInstaller (`packaging/gettowork.spec`);
3. gathers the license texts (`packaging/collect_licenses.py`, including every
   native library PyInstaller bundled - a library with no license entry fails
   the build);
4. on Windows, once code signing is set up (repository secrets - see
   `packaging/steam/README.md`), signs every `.exe` and `.dll`
   (`packaging/sign_windows.py`);
5. adds the engine, `distribution.json`, the licenses and a README, and packs
   the archive (`packaging/assemble.py`);
6. smoke-tests the terminal version and the bundled engine
   (`packaging/smoke_test.sh`, which also checks the game reports its built-in
   engine with downloads off), on Linux starts the engine in a container
   without OpenSSL 3, as under Steam (`packaging/engine_isolation_check.sh`),
   then lets the window play a scripted game (`GetToWork --gui-selftest`);
7. uploads the archive itself as a workflow artifact (kept 3 days).

To move the game to a newer llama.cpp release, change the tag in
`packaging/llama_cpp_tag.txt` and the archives' SHA-256 lines under it (the
weekly live check says whether the newest release plays a real game), let CI
build it, and play a test build.

[docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) explains the layout and every
script. To make a build on your own computer (for your own OS):

```bash
pip install . pyinstaller pillow
python packaging/fetch_engine.py --os linux --arch x64 --variants vulkan,cpu --dest build/engine --tag pinned --verify
pyinstaller packaging/gettowork.spec --noconfirm --clean
python packaging/collect_licenses.py --engine-dir build/engine --app-dir dist/GetToWork --out build/THIRD_PARTY_LICENSES.txt
python packaging/assemble.py --dist dist --engine-dir build/engine --os linux --out out \
  --licenses build/THIRD_PARTY_LICENSES.txt
packaging/smoke_test.sh --engine-dir dist/GetToWork/engine dist/GetToWork/gettowork-cli
dist/GetToWork/GetToWork --gui-selftest
```

(Use `--os windows --variants vulkan,cpu` on Windows, or
`--os macos --arch arm64 --variants metal` on a Mac, where the programs are
inside `dist/macos-package/GetToWork/Get To Work.app`.) Releasing on Steam is
covered in [packaging/steam/README.md](packaging/steam/README.md).

`.github/workflows/live-check.yml` (weekly, or by hand) is the one workflow
that uses the real internet: a real Hugging Face search, the real engine
playing a scripted game with a tiny real model, and - when you start it by
hand with the `TYPESAFE_API_KEY` secret set - a real Jev round.
