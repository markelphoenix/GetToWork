# Get To Work

[![tests](https://github.com/markelphoenix/GetToWork/actions/workflows/ci.yml/badge.svg)](https://github.com/markelphoenix/GetToWork/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A silly text adventure that runs a real AI model on your own computer and
teaches you how it works along the way.**

It's 8:45. Work starts at 9:00. Between you and your desk: squirrels with a
toll booth, a bicycle that joined a jazz band, and a polite dragon sunbathing
on the zebra crossing. You type what you do. An open-weight language model,
running privately on *your* machine, tells the story. A referee decides if
your plan worked. Five good plans and you're at work.

You don't need to know anything about AI to play. The game checks your
computer, finds models on Hugging Face that will actually run well on it, and
installs everything for you. **All you do is pick a model.**

---

## Contents

- [A taste of the game](#a-taste-of-the-game)
- [What you'll learn](#what-youll-learn)
- [Quickstart in 3 steps](#quickstart-in-3-steps)
- [What happens automatically](#what-happens-automatically)
- [Where files are stored (and how to delete them)](#where-files-are-stored-and-how-to-delete-them)
- [Other ways to run the model (optional)](#other-ways-to-run-the-model-optional)
- [Jev (optional)](#jev-optional)
- [How to play](#how-to-play)
- [How does it know which models fit?](#how-does-it-know-which-models-fit)
- [Command-line options](#command-line-options)
- [Privacy: what leaves your computer](#privacy-what-leaves-your-computer)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [License and disclaimer](#license-and-disclaimer)

---

## A taste of the game

*(An illustrative sample. Every game is different: the model makes it up as
it goes.)*

```text
BRRRING! You wake up with your face in a bowl of cereal and your alarm clock
humming a sad little tune. It is 8:45. Work starts at 9:00...

Progress to your desk  [----------] 0/5
Round 1: the journey
How do you plan to get to work? > I ride my rusty bicycle, ringing the bell.

You wobble off down the lane at a heroic 6 miles an hour...

Progress to your desk  [##--------] 1/5
Challenge 1: A committee of squirrels has declared your front path a
nut-storage zone and demands a toll of three acorns.
What do you do? > I hold a snap election and run on a platform of free peanuts.

Jev's verdict
  Noul    made_progress  [###################-] 93% chance of yes
                         => that counts as progress!
  Choice  outcome        triumph  (81% confident)
  Score   creativity     3.4 out of 4  (74% confident)

The squirrels erupt into tiny applause and carry you down the path on a
wave of fluffy tails...

Progress to your desk  [####------] 2/5
Challenge 2: Your bicycle has joined a jazz band and refuses to move unless
someone plays the tambourine.
```

When the game ends you can peek behind the curtain: the exact requests sent to
the referee, and the model's private "thinking" (when it's quick enough to
think out loud - or when you start the game with `--think`).

## What you'll learn

- **How much memory a model needs, and why**: parameters, quantization
  (what "Q4_K_M" means), and the KV cache.
- **Why memory *speed* decides how fast a model talks**, with a worked example
  you can check on a calculator.
- **What GGUF, llama.cpp, CUDA, Vulkan and Metal are**, and why your graphics
  card matters.
- **How "thinking" models show their chain-of-thought**, and how to read it.
- **Two ways to judge text with AI**: asking a chat model to "reply in JSON"
  versus asking a typed-judgment API (Jev) for calibrated probabilities.
- **What prompt injection is**, and one simple way to defend against it.

Curious already? The mini-course is in [docs/LEARN.md](docs/LEARN.md).

## Quickstart in 3 steps

**1. Install Python 3.10 or newer** from [python.org](https://www.python.org/downloads/).
On Windows, tick **"Add python.exe to PATH"** in the installer. Most Macs and
Linux computers can use the python.org installer, Homebrew or their package
manager.

**2. Install the game.** The easiest way is [pipx](https://pipx.pypa.io/),
which keeps the game in its own tidy box:

```bash
pipx install git+https://github.com/markelphoenix/GetToWork
```

<details>
<summary>Don't have pipx? Or prefer plain pip?</summary>

- Get pipx: `python -m pip install --user pipx` then `python -m pipx ensurepath`
  (on Windows you can type `py` instead of `python`; on macOS `brew install pipx`
  also works). Open a new terminal afterwards.
- Or use pip from a downloaded copy of this repository, ideally inside a
  virtual environment:

  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
  pip install .
  ```

</details>

**3. Run it:**

```bash
gettowork
```

Then just pick a model from the list (press Enter for the recommended one)
and confirm. The game does the rest.

> **Want to try it right now, with no downloads at all?**
> Run `gettowork --mock`. A pretend, scripted "model" plays along instantly and
> offline, so you can see how everything works before downloading anything.

If your terminal says `gettowork` isn't found: after pipx, run
`pipx ensurepath` and open a new terminal; after pip, `python -m gettowork`
(with your virtual environment active) does the same thing.

## What happens automatically

The design goal is simple: **you only ever pick a model.** Everything else is
automatic, explained in plain English, and reversible.

1. **Hardware check (about a second).** Your operating system, CPU (and which
   speed-up instructions it has), RAM, graphics card(s) and video memory,
   Apple Silicon unified memory, and free disk space. A 0.3-second memory
   speed test measures how fast your RAM is. Nothing is sent anywhere.
2. **Live Hugging Face search.** The game asks the free
   [Hugging Face Hub](https://huggingface.co) API for today's most popular
   GGUF chat models from well-known publishers, reads their real metadata
   (size, architecture, context length) and the real size of every file. It
   keeps family-friendly, instruction-tuned models with permissive licenses
   (Apache-2.0 or MIT). The results are **cached for 3 days**, so the next
   launch is instant and works offline. No internet and no cache? A built-in
   list of tried-and-tested models is used instead.
3. **The fit engine.** For every candidate, the game picks the best version
   (quantization) for *your* machine, estimates memory use and speed, and
   scores it. You see about six picks with plain badges: **Recommended**,
   **Fastest comfortable fit** and **Smartest at a playable pace** (both judge
   whole story turns - reading your prompt as well as writing - and skip snug
   fits, so they can differ from the Speed column). Press Enter for the recommended
   one, or type `more` (full list), `refresh` (search again), `custom` (paste
   any Hugging Face GGUF repo), `mock` (play offline), `learn` (how it chose)
   or `why 2` (the working behind pick number 2).
4. **One confirmation.** A single screen lists exactly what will be downloaded,
   how big it is, where it comes from, its license, and where it will be saved.
5. **The engine.** The game downloads the official prebuilt **llama.cpp**
   engine (`llama-server`) for your computer from the
   [ggml-org/llama.cpp GitHub releases](https://github.com/ggml-org/llama.cpp/releases):
   an NVIDIA CUDA, Vulkan, Apple Metal or plain CPU build (usually a few tens
   of MB; CUDA builds are about half a gigabyte because they include NVIDIA's
   CUDA runtime). It checks the file's size and SHA-256 fingerprint and
   unpacks it into the game's own folder. No compilers, no admin rights,
   nothing installed system-wide.
6. **The model.** The model file(s) download from Hugging Face with a progress
   bar. Files that are already complete are never downloaded twice.
7. **Start and speed test.** `llama-server` starts on your computer, listening
   only on `127.0.0.1` (your own machine), and a short test measures the real
   speed: *"Your model is talking at ~18 tokens/sec!"* If it's painfully slow
   (under 3 tokens/sec), the game offers to switch to a smaller model.
   **It also learns from the measurement:** it compares the real speed with
   its estimate and remembers the difference for this computer, so its
   recommendations get more realistic the more you play.
   If a graphics-card build won't start (an old driver, say) - or starts but
   stumbles on its first real work - the game explains in one sentence and
   falls back automatically to the next build, ending with the CPU build,
   which works everywhere. On a computer where the ready-made engine can't
   run at all (a very old Linux, say), it tells you *before* downloading
   anything.
8. **Next time:** *"Welcome back! Play again with Qwen3 4B? [Y/n]"* and you're
   straight into the game (if you chose to play without Jev last time, it
   just checks once, in one line). When you quit, the engine is always stopped.

## Where files are stored (and how to delete them)

Everything lives in one folder:

| System  | Folder |
|---------|--------|
| Windows | `%LOCALAPPDATA%\GetToWork` (for example `C:\Users\you\AppData\Local\GetToWork`; kept out of your roaming profile because models are big. A folder an older version made in `%APPDATA%\GetToWork` keeps being used.) |
| macOS   | `~/Library/Application Support/GetToWork` |
| Linux   | `~/.config/gettowork` (or `$XDG_CONFIG_HOME/gettowork`) |

Inside it:

| Path | What it is |
|------|------------|
| `settings.json` | Your choices (model, backend, whether Jev is on) and the speed calibration measured on this computer. Your Jev key is stored here **only** if you said yes to saving it. |
| `models/` | Downloaded model files, one folder per Hugging Face repo, e.g. `models/unsloth--Qwen3-4B-GGUF/`. These are the big ones (1-20+ GB). |
| `runtime/llama.cpp/` | The llama.cpp engine, one folder per build, e.g. `runtime/llama.cpp/b6000-vulkan/`. When an update replaces a build, the old copy is tidied away once the new one works. |
| `runtime/logs/` | The engine's log (`llama-server.log`; on Windows each game writes its own `llama-server-<number>.log`), handy when something goes wrong. |
| `cache/hf_models.json` | The saved Hugging Face search results. |

**Moving or deleting things:**

- Set `GETTOWORK_HOME` to use a different folder for everything, or
  `GETTOWORK_MODELS_DIR` to keep just the models somewhere else (a bigger
  drive, for example).
- `gettowork --reset` forgets your saved settings: your choices, the speed
  calibration and any saved key. It doesn't delete models.
- To remove a model, delete its folder under `models/`. To remove everything,
  delete the whole folder above, then uninstall the game with
  `pipx uninstall gettowork` (or `pip uninstall gettowork`).
- Transcripts you choose to save (`gettowork-transcript-1.json` / `.md`) go in
  the folder you ran the game from, or `--export-dir`.
- If you use Ollama, its models are managed by Ollama: `ollama list` and
  `ollama rm <name>`.
- If you logged in to Hugging Face yourself (`hf auth login`), that token is
  kept by `huggingface_hub` in its own folder (`~/.cache/huggingface` by default).

## Other ways to run the model (optional)

The built-in llama.cpp engine is the default and needs nothing else. If you
like tinkering, there are alternatives:

- **[Ollama](https://ollama.com/download)** (a separate, free app). If it's
  running, `gettowork --backend ollama` uses it: the game asks Ollama to pull
  the same GGUF straight from Hugging Face (`hf.co/<repo>:<quant>`). Already
  have an Ollama model you like? `--ollama-model TAG`.
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)**: install
  it with `pip install llama-cpp-python` (or `pip install ".[llamacpp]"` from
  this repo), then run `gettowork --backend llamacpp`. On some systems pip has
  to compile it, which needs a C/C++ compiler.
- **Mock**: `gettowork --mock` for the scripted offline pretend model.
- **Your own GGUF file**: `gettowork --gguf path/to/model.gguf` runs it with
  the built-in engine.

With the default `--backend auto`, if the built-in engine can't be installed
or started, the game offers to use Ollama (if it's running; it asks first, and
hands Ollama the model file you already downloaded instead of downloading it
again) or llama-cpp-python (if it's installed), and tells you what it's doing.
If a brand-new model needs a newer engine, the game updates the engine once by
itself. If nothing works, it offers to try again, pick another model, play with
the pretend model, or stop, and anything already downloaded is kept.

## Jev (optional)

**Jev** is a *typed-judgment* model from [TypeSafe AI](https://typesafe.ai).
Instead of chatting, it answers questions with values code can use directly:

- a **Noul** (yes/no) answer is the *probability of yes*, e.g. `0.93`;
- a **Choice** answer picks one of your labels, with a probability for each;
- a **Score** answer rates on your scale, e.g. `2.6` out of `4`.

With Jev turned on, it referees every plan with one question of each type.
The Noul decides whether you made progress. The game shows each answer and
explains it the first time you see it.

**The game never needs Jev.** Without it, your local model referees for free.

How to turn it on:

1. After your model is ready, the game asks *"Enable Jev for this game?"*
   Choose `yes`, `no` (local only) or `learn` (tell me more first). The screen
   also says what Jev receives each round (your plan, the challenge and a short
   story summary; see [Privacy](#privacy-what-leaves-your-computer)).
2. Paste your API key (it stays hidden while you type; if your window can't
   hide it, for example an IDE's Run console, the game warns you first and
   suggests the environment variable below), **or** choose
   *"walk me through getting one"*: the game opens https://typesafe.ai in your
   browser and lists the steps (sign up or log in, open the API keys page in
   your dashboard, create a key, copy it). The documentation is at
   https://docs.typesafe.ai/.
3. The key is checked with a quick request. Saving it for next time is
   **opt-in** (default: no). It would be stored in plain text in
   `settings.json` (readable only by you: file permissions on macOS and Linux, an owner-only access list on Windows). If you switch to
   a new key without saving it, or Jev stops accepting the saved one, the old
   key is removed from the file.

You can back out to local-only at **every** step: choose `no` or `back`,
press Enter at the key prompt, or press Ctrl+C (even while the key is being
checked). `gettowork --no-jev` skips the question entirely.

Prefer not to paste the key each time? Set the `TYPESAFE_API_KEY` environment
variable and the game will offer to use it:

```text
export TYPESAFE_API_KEY="your-key"        # macOS / Linux
$env:TYPESAFE_API_KEY = "your-key"        # Windows PowerShell
set TYPESAFE_API_KEY=your-key             # Windows Command Prompt
```

> **Costs and terms are TypeSafe's.** Jev is a paid third-party service with
> its own pricing and terms of service. Check them on their website before
> signing up. Get To Work is not affiliated with TypeSafe AI, and any charges
> for using Jev are between you and TypeSafe.

## How to play

- **First, say how you'll get to work**: on foot, by bike, bus, broomstick, a
  borrowed ostrich... The obstacles that follow fit your choice.
- **Then read each challenge and type what you do**, in your own words:
  *"I bribe the geese with a bagel and tiptoe past."*
- **Cartoon logic rules.** Silly, magical and impossible ideas are welcome,
  as long as they deal with *this* challenge.
- **Doing nothing, giving up, or "I teleport to work and win" won't count.**
  The referee wants to see you tackle the obstacle.
- **Reach 5 steps to get to work**: setting off counts as the first, then
  four ever-sillier obstacles, the last one right at the office (change it
  with `--target N`).
- Type `help` for tips, or `quit` (also `q` or `exit`) to stop early. You
  still get the behind-the-scenes review.

**Who's the referee?** With Jev on, one request per round asks the Noul,
Choice and Score questions, and progress counts when the Noul's probability is
at least 0.5. Without Jev, your local model is asked for a tiny JSON verdict.
If anything goes wrong mid-game (a network hiccup, an unreadable answer), the
game falls back gracefully so you never lose a round to a glitch.

**After the game** you can look behind the scenes. Each question appears only
when there is something to show:

1. See Jev's request and response for each round? (if Jev refereed)
2. See the local model's reasoning (chain-of-thought) for each round? (if it
   thought out loud)
3. See the local model's verdict (its JSON answer) for each round? (when it
   refereed but didn't think out loud - so you can compare it with Jev's numbers)

Then you can save a transcript as JSON and Markdown. Your API key is never
included.

## How does it know which models fit?

No magic, just a few honest formulas (all in [catalog.py](src/gettowork/catalog.py)
and [perf.py](src/gettowork/perf.py), and explained step by step in
[docs/LEARN.md](docs/LEARN.md#part-7-inside-the-games-fit-engine)):

- **Memory needed** = the model file's size + the KV cache (the model's
  short-term memory of the conversation) + about 0.6 GB of overhead (plus
  0.3 GB on a graphics card).
- **Memory available** = your video memory minus 0.8 GB, *or* the share of a
  Mac's unified memory its GPU may use, *or* your RAM minus 2.5 GB (3.5 GB on
  Windows) for your system. Using up to 60% of it is *great*, up to 85% *ok*,
  up to 100% *tight*.
- **Speed**: writing each word-piece means reading the model's weights from
  memory, so `tokens/sec ≈ efficiency × memory bandwidth ÷ GB read per token`.
  That's why the game times your RAM and looks up your graphics card.
- **Which version**: start from the popular ~4-bit sweet spot (Q4_K_M), go
  smaller only if needed, and go bigger (Q5, Q6, Q8) only if it still fits
  comfortably and stays fast.
- **Score** = quality (size and precision) + speed + headroom + popularity,
  plus small bonuses (for example, models that show their thinking).
- **Learning from reality**: after the speed test, the game divides the real
  speed by its estimate and scales its memory-speed numbers for this computer
  by that factor (within limits), so the next ranking is closer to the truth.

See it for your own computer:

```bash
gettowork --specs         # what the game found, including memory speed
gettowork --list-models   # the ranked shortlist for this machine
```

These are home-grown estimates: useful, but not guarantees. The game measures
the real speed once your model is running.

## Command-line options

| Option | What it does |
|--------|--------------|
| `--mock` | Play with a scripted, offline pretend model. No downloads. |
| `--backend {auto,managed,ollama,llamacpp}` | Which engine runs the model. `auto` (default) uses the built-in llama.cpp engine (`managed`), falling back to Ollama if it's running. |
| `--model REPO_OR_KEY` | Skip the picker and use this model, e.g. `unsloth/Qwen3-4B-GGUF` or `qwen3-4b`. |
| `--quant TAG` | Use this quantization, e.g. `Q4_K_M` or `Q8_0`. |
| `--ollama-model TAG` | Use a model you already have in Ollama. |
| `--gguf PATH` | Run a GGUF file you already have, with the built-in engine. |
| `--list-models` | Print the ranked shortlist for this computer, then exit. |
| `--specs` | Print what the game found about your hardware (including memory speed), then exit. |
| `--refresh-models` | Ignore the saved Hugging Face results and search again. |
| `--offline` | Don't go online for the model list: use the saved results or the built-in list. Downloads then only work for files you already have. |
| `--all-licenses` | Also show models whose licenses aren't Apache-2.0 or MIT. Each license is shown clearly; complying with it is up to you. (A model you name yourself with `custom` or `--model` can have any license too, always with a warning.) |
| `--no-jev` | Don't ask about Jev; the local model referees. |
| `--think` | Let a "thinking" model think out loud even on a slow computer (each turn takes longer). |
| `--target N` | Steps needed to win (default 5). |
| `--reset` | Forget saved settings: your choices, speed calibration and any saved key (models stay on disk). |
| `--export-dir DIR` | Where to save transcripts (default: the current folder). |
| `--debug` | Show full error details (tracebacks). Useful for bug reports. |
| `--version` | Print the version and exit. |

Press Ctrl+C at any time to leave cleanly. The engine is always stopped on the
way out.

## Privacy: what leaves your computer

The story and your plans stay on your computer unless you turn on Jev. The
model runs locally and is only reachable at `127.0.0.1`.

| When | Sent to | What |
|------|---------|------|
| Model search (at most every 3 days, or `refresh`) | Hugging Face (`huggingface.co`) | Search requests for public GGUF models, and file-list requests for the most promising ones. |
| Downloading a model | Hugging Face | Requests for the model file(s) you confirmed. |
| Downloading the engine (first time, or a new build) | GitHub (`api.github.com`, `github.com` and its download servers) | A request for the list of recent llama.cpp releases, then the download of one archive. If you set `GITHUB_TOKEN`, it's sent only to `api.github.com`. |
| Each round, **only if Jev is enabled** | TypeSafe AI (`api.typesafe.ai`) | This round's state: a trimmed "story so far", the current challenge, your plan, your progress and a short summary of recent rounds (including your earlier plans), plus the three questions and your API key (in the `Authorization` header). The key is only ever sent over https to that address: the game never follows a redirect elsewhere. |
| Checking your Jev key | TypeSafe AI | One `GET /v1/models` request with your key. |
| "Open the website" in the Jev help | Your web browser | Only when you ask; it opens typesafe.ai or its docs. |

That's all. **The game has no telemetry, analytics or crash reporting.** Like
any website, these services see your IP address and basic technical details
such as the app's user-agent. If you've logged in with `hf auth login`,
`huggingface_hub` includes your Hugging Face token in its requests (that's how
gated models work). With Ollama, requests go to your local Ollama app (unless
you point `OLLAMA_HOST` at another machine), and Ollama downloads the model
from Hugging Face.

## Troubleshooting

Run with `--debug` for full details, and look at the engine's log in
`runtime/logs/llama-server.log` (on Windows, `llama-server-<number>.log`; see
[where files are stored](#where-files-are-stored-and-how-to-delete-them)).

### It's very slow

- Pick a smaller model: the one with the **Fastest comfortable fit** badge, or a 0.6B-4B model.
  The game offers this automatically below 3 tokens/sec.
- Close big apps (browsers with many tabs, games, video calls) and plug your
  laptop in: many laptops slow down on battery.
- "Thinking" models spend a while reasoning before they answer. Below about 20
  tokens/sec the game asks them to answer straight away (the review tells you
  it did); start with `--think` to see their thinking anyway. Models that
  *always* think at length (QwQ, DeepSeek-R1 distills, Phi-4-reasoning...)
  can't be asked not to, so they're kept off the short menu.
- Speed estimates are rough guesses. The measured number after the speed test
  is the real one.

### It ran out of memory, or the engine stopped suddenly

- The fit engine leaves headroom, but other apps can use more memory than when
  you started. Close them, or pick a smaller model or quantization
  (`--quant Q3_K_M`).
- A "tight" verdict means it barely fits. Choose a model marked *great* or *ok*.

### Graphics card (GPU) problems

- NVIDIA CUDA builds need a reasonably recent driver (about version 525 or newer
  for CUDA 12, 580 or newer for CUDA 13). Updating your driver from NVIDIA's
  website often helps. Older cards (GTX 9xx/10xx, Titan V) get the CUDA 12 build,
  because CUDA 13 no longer supports them.
- If a GPU build won't start, the game reads the engine's log, explains, and
  **falls back automatically**: CUDA, then Vulkan, then CPU. Everything still
  works on the CPU, just slower.
- On Linux, AMD and Intel graphics use the Vulkan build, which needs the Vulkan
  loader and a Vulkan driver (for example the `libvulkan1` and
  `mesa-vulkan-drivers` packages on Ubuntu). Without them, the CPU build is used.
- Integrated graphics share your system RAM, so the game plans (and estimates
  speed) as if the model runs from RAM. That's expected.

### Corporate networks and proxies

- Set the usual `HTTPS_PROXY` / `HTTP_PROXY` (and `NO_PROXY`) environment
  variables; Python and `huggingface_hub` honour them. Requests to your own
  engine on `127.0.0.1` never go through a proxy.
- If your company inspects encrypted traffic, Python needs your company's
  certificate: point `SSL_CERT_FILE` at it (your IT team can provide it).
- The game checks secure connections against your operating system's own list
  of trusted certificates, so the Python installer from python.org on a Mac
  works even if you never ran its *Install Certificates* helper. If you ever
  see "your Python couldn't check the server's security certificate", that
  helper (double-click *Install Certificates.command* in your Python folder in
  Applications) or `SSL_CERT_FILE` fixes it - it isn't a network problem.
- If large model downloads fail but small requests work, try setting
  `HF_HUB_DISABLE_XET=1` (it makes `huggingface_hub` use plain HTTPS downloads).
  Your company's Hugging Face mirror, if any, goes in `HF_ENDPOINT`.
- GitHub limits anonymous requests to its API to 60 an hour. If you hit that
  limit, wait a bit or set a `GITHUB_TOKEN`.
- If GitHub or Hugging Face is blocked entirely, try `gettowork --offline`,
  Ollama, or `--mock`.

### "Gated" models

Some models ask you to accept their authors' terms before downloading. The
game avoids them by default. If you pick one (for example with `custom`),
open the model's page on Hugging Face while logged in, accept the terms, run
`hf auth login` in a terminal, and try again. Or simply pick another model.

### Windows: SmartScreen or antivirus warnings about llama-server.exe

- The engine is the official build from the llama.cpp project. Like many
  open-source tools it may not carry a commercial code-signing certificate, so
  Windows doesn't "know" it yet. The game checks its size and SHA-256
  fingerprint against what GitHub reports before unpacking it.
- Some antivirus tools flag or quarantine new, unsigned programs. If yours does,
  you can compare the file with the
  [official releases page](https://github.com/ggml-org/llama.cpp/releases)
  and, if you're comfortable, allow the game's `runtime` folder. Or use Ollama.
- If Windows Firewall asks about `llama-server.exe`, you can choose not to allow
  network access. The game only talks to it on `127.0.0.1`, your own computer.

### macOS

- On Apple Silicon (M1 and later) the Metal build uses the GPU and unified
  memory automatically. Intel Macs use the CPU build, which is slower, so
  smaller models are best there.
- If macOS ever says `llama-server` "can't be opened" or "cannot be verified",
  the engine's folder has been marked as downloaded from the internet. You can
  clear that mark for the game's runtime folder only with
  `xattr -dr com.apple.quarantine ~/Library/Application\ Support/GetToWork/runtime`,
  or use Ollama instead.
- Very old macOS versions may not run the current prebuilt engine; Ollama or
  `--mock` still work.

### Jev says my key doesn't work

Check that you copied the whole key (no spaces), that it hasn't been deleted in
your TypeSafe dashboard, and that your account is in good standing. A "payment
required" message comes from TypeSafe's billing. You can always choose `back`
and play locally.

### Start fresh

`gettowork --reset` forgets your saved choices. Deleting the whole data folder
starts completely from scratch.

## Project layout

```text
src/gettowork/
  __main__.py        `python -m gettowork` starts the game
  cli.py             command-line options, then setup -> Jev -> game -> review
  setup_flow.py      hardware -> discovery -> pick -> install -> warm-up
  types.py           shared data classes (read this first!)
  ui.py              all terminal input/output (built on rich)
  config.py          settings file and data folders
  specs.py           hardware detection
  perf.py            memory-speed test and tokens/sec estimates
  catalog.py         built-in model list and the fit/ranking engine
  hf_discovery.py    live Hugging Face search, GGUF metadata, cache
  download.py        model downloads from Hugging Face
  runtime_install.py official prebuilt llama.cpp engine download
  tls.py             HTTPS certificate checks that work on every computer
  reasoning.py       separates chain-of-thought from answers
  backends/          llamaserver.py (default), ollama.py, llamacpp.py, mock.py
  jev.py             Jev API client, the game's three questions, verdicts
  onboarding.py      the friendly "enable Jev?" flow
  prompts.py         every word the game says to the local model
  game.py            the core game loop
  review.py          end-of-game review and transcript export
tests/               pytest suite: no network, no real models
docs/ARCHITECTURE.md how the modules fit together (the contract)
docs/LEARN.md        the mini-course
```

## Contributing

Contributions, bug reports and ideas are very welcome. See
[CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, how the model
list and filters work, and the (short) rules.

## License and disclaimer

Get To Work is free, open-source software under the [MIT License](LICENSE).
Third-party components and services are listed in [NOTICE.md](NOTICE.md).

> **DISCLAIMER**
>
> - **No warranty.** This software is provided **"AS IS"**, without warranty
>   of any kind, express or implied. See the [LICENSE](LICENSE). You use it at
>   your own risk.
> - **Estimates can be wrong.** Hardware detection, memory and speed estimates,
>   and model recommendations are home-grown heuristics. They may be inaccurate
>   for your computer, and a recommended model may still run slowly or fail.
> - **Not affiliated.** This project is not affiliated with, endorsed by or
>   sponsored by TypeSafe AI, Hugging Face, ggml-org / the llama.cpp project,
>   Ollama, or any model author or publisher. Product and company names are
>   trademarks of their respective owners and are used only to identify their
>   products and services.
> - **Third-party downloads are your responsibility.** No model weights and no
>   llama.cpp engine are included in this project. When you confirm, you
>   download them yourself from third parties (Hugging Face, GitHub) under their
>   own licenses and terms, and you are responsible for complying with them.
>   The game shows each model's license before downloading; `--all-licenses`
>   can show licenses with extra conditions.
> - **Jev may cost money.** Using Jev may incur charges under TypeSafe AI's own
>   pricing and terms. The game works fully without it.
> - **AI output is unpredictable.** The prompts ask for farcical, family-friendly
>   stories, but AI models can still produce odd, wrong or inappropriate text.
>   Nothing the game or a model says is advice.
