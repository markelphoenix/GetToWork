# Contributing to Get To Work

Thank you for helping! Get To Work is a learning project first, so clear,
friendly code and docs matter as much as features. Bug reports, typo fixes,
new "Try this" exercises and better explanations are all very welcome.

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

Play without downloading anything:

```bash
gettowork --mock            # scripted offline model
gettowork --mock --target 2 # an even quicker game
gettowork --specs           # hardware detection only
gettowork --list-models     # discovery + fit engine only
```

Optional extra: `pip install -e ".[dev,llamacpp]"` installs `llama-cpp-python`
for `--backend llamacpp`. CI doesn't install it, so tests must never need it.

## Ground rules for code

- **Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) first.** It is the
  binding contract between modules. If you change a public signature or
  behaviour, update that document in the same pull request.
- **Python 3.10+** with `from __future__ import annotations`. No APIs that only
  exist in 3.11 or later (such as `tomllib`) outside of tests with a fallback.
- **Windows, macOS and Linux** must all work. Use `pathlib`, list-style
  subprocess arguments (never `shell=True`), and give every external command a
  short timeout.
- **Dependencies stay small**: `rich`, `psutil` and `huggingface_hub`. HTTP to
  GitHub, llama-server, Ollama and Jev uses only the standard library
  (`urllib.request`), so learners can see exactly what is sent. Please open an
  issue before adding a dependency; any new one must be free and permissively
  licensed.
- **Written for learners**: clear names, a docstring on every public function,
  and short comments where a concept isn't obvious (not on every line).
- **Warm, plain-English messages.** Every error the player sees should say what
  happened and what to do next, in one or two sentences, without jargon.
- **Escape untrusted text.** Anything from a model, from Jev or from the player
  is printed with `rich.markup.escape` (or as a `rich.text.Text`).
- **Secrets are sacred.** API keys are never printed, logged, exported or put in
  `repr()`. They're shown only as `****abcd`.
- **No telemetry**, analytics or "phone home" features, ever.

## Tests: no network, no real models

The whole suite must run offline in a few seconds. Tests must never touch the
network, real model files, real subprocess downloads or a real browser. Every
module that talks to the outside world accepts a fake:

| Module | What to inject |
|--------|----------------|
| `ui.py` | `UI(console=Console(file=io.StringIO()), input_fn=..., secret_fn=..., open_url_fn=...)` |
| `hf_discovery.py` | `discover_models(api=FakeHfApi(), cache_path=tmp_path / "c.json", clock=...)` |
| `download.py` | `download_gguf(..., hf_api=fake_api, hf_download=fake_download)` |
| `runtime_install.py` | `http=` (an object with `request(method, url, *, headers, body, timeout)`), `runtime_root=tmp_path` |
| `backends/llamaserver.py` | `http=`, `popen=`, `installer=`, `downloader=`, `sleep=`, `clock=`, `log_dir=` |
| `backends/ollama.py` | `http=` |
| `backends/llamacpp.py` | `llama_factory=`, `downloader=` |
| `jev.py` | `JevClient(key, transport=fake_transport)` |
| `setup_flow.py` | `SetupServices(detect_specs=..., discover_models=..., make_backend=..., installed_runtimes=..., custom_entry=...)` |
| `cli.py` | `main(argv, ui=..., services=SetupServices(...))` |
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

Each module has its own `tests/test_<module>.py`. `tests/test_docs.py` checks
that the documentation still matches the code (formulas, numbers, flags,
links and the Jev wire-format examples), so if you change a constant it may
ask you to update [docs/LEARN.md](docs/LEARN.md) too. That's on purpose.

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

## License rules

- Get To Work is **MIT licensed**. By contributing, you agree that your
  contribution is licensed under the MIT License too.
- Only contribute code you wrote yourself or that comes from a compatible
  permissive license (MIT, BSD, Apache-2.0), with attribution added to
  [NOTICE.md](NOTICE.md). Don't paste code from sources with unclear or
  incompatible licenses.
- **Never commit model weights, engine binaries or other large downloads.** The
  game downloads them at the player's request; `.gitignore` already blocks
  `*.gguf`.
- Seeds and default discovery results stay **Apache-2.0 / MIT only**.
- The hardware-fit heuristic is original code. Keep it that way: don't copy
  tables or code from other projects without checking their license first.
- Product names (Hugging Face, llama.cpp, Ollama, TypeSafe AI, Jev...) are used
  only to identify those products. Don't add logos or anything implying
  endorsement.

## Pull requests

1. Open an issue first for big changes, so we can agree on the approach.
2. Keep pull requests focused; one topic each.
3. Make sure `python -m pytest -q` passes. CI runs it on Windows, macOS and
   Linux with Python 3.10 and 3.12.
4. Update the docs (README, LEARN, ARCHITECTURE) when behaviour changes.

Found a security problem, such as an API key leak or an unsafe download? Please
don't open a public issue. Use GitHub's private vulnerability reporting on the
repository (Security tab), or contact the maintainer privately first.

## Code of conduct

Be kind, patient and welcoming, especially to beginners: this project exists
so people can learn. We follow the spirit of the
[Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/).
Harassment or disrespect isn't tolerated. If something goes wrong, contact the
maintainer through GitHub.

## Test builds

`.github/workflows/build.yml` builds the wheel/sdist and standalone
executables (PyInstaller, via `packaging/gettowork_entry.py`) for Windows,
macOS and Linux on every push to `main` and every pull request, smoke-tests
each with `packaging/smoke_test.sh`, and uploads them as workflow artifacts.
To check a build locally:

```bash
pip install . pyinstaller
pyinstaller --onefile --name gettowork --collect-submodules rich \
  --collect-submodules truststore --collect-data certifi packaging/gettowork_entry.py
packaging/smoke_test.sh ./dist/gettowork
```

