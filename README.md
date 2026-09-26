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
sets everything up for you. **All you do is pick a model.**

> **Where this is heading:** Get To Work is a **free game, planned for Steam**.
> Players press **Play** and the game opens its own window, walks them through
> a one-time setup and starts the story. The source repository is private;
> test builds come from its GitHub Actions (see
> [Playing the test builds](#playing-the-test-builds)).

---

## Contents

- [What it is](#what-it-is)
- [Playing the test builds](#playing-the-test-builds)
- [Playing from source](#playing-from-source)
- [What happens when you play](#what-happens-when-you-play)
- [What's bundled and what's downloaded](#whats-bundled-and-whats-downloaded)
- [Where files are stored (and how to delete them)](#where-files-are-stored-and-how-to-delete-them)
- [Other ways to run the model (optional)](#other-ways-to-run-the-model-optional)
- [Jev (optional)](#jev-optional)
- [How to play](#how-to-play)
- [How does it know which models fit?](#how-does-it-know-which-models-fit)
- [Command-line options](#command-line-options)
- [AI content and the family-friendly filter](#ai-content-and-the-family-friendly-filter)
- [Privacy: what leaves your computer](#privacy-what-leaves-your-computer)
- [Troubleshooting](#troubleshooting)
- [Releasing on Steam](#releasing-on-steam)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [License and disclaimer](#license-and-disclaimer)

---

## What it is

A short, replayable comedy: every morning goes wrong in a new way, the story
is written live by an AI running on your computer, and the game explains the
technology as it goes. It's a learning project first, so the code and docs are
written to be read.

### A taste of the game

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
think out loud - or when you start the game with `--think`). Then it asks
**"Play again?"** and a brand-new morning starts straight away, with the model
still loaded.

### What you'll learn

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

## Playing the test builds

Every push to `main` is built by the [`build` workflow](.github/workflows/build.yml)
into a ready-to-play game for each operating system - the same files that get
uploaded to Steam. No Python, no installs: the AI engine (llama.cpp) is built
in, and only the AI model itself is downloaded, once, during the game's guided
setup.

**1. Download.** Open the repository's **Actions** tab, pick the latest green
**build** run on `main`, and download your system's artifact from its
**Artifacts** list (you need to be signed in to GitHub with access to the
repository). Each game artifact is the game archive itself:

| Artifact | For |
|---|---|
| `GetToWork-<version>-windows-x64.zip` | Windows 10/11, 64-bit |
| `GetToWork-<version>-macos-arm64.zip` | Macs with Apple Silicon (M1 or later), macOS 13.3+ |
| `GetToWork-<version>-linux-x64.tar.gz` | 64-bit Linux with glibc 2.35+ (Ubuntu 22.04+, SteamOS 3.5+) |
| `gettowork-python-package` (the wheel and source archive, zipped by GitHub) | `pip install gettowork-*.whl` |

Game builds are kept for **3 days**, the Python package for 7 (the repository
is private, so storage is tight). Need a fresh one? Open the **build** workflow
and press **Run workflow**. Pushes that only change docs or tests don't make
new builds, and game builds for pull requests (made only when a pull request
touches the packaging) are tested but not uploaded - run the workflow by hand
on the branch to get them. All three builds of a run carry the same llama.cpp
release, the one pinned in `packaging/llama_cpp_tag.txt`.

**2. Unpack** the archive (one layer):

- **Windows:** right-click the zip, **Extract All...** - the game can't start
  from inside the zip (double-clicking `GetToWork.exe` in Explorer's zip view
  fails), so extract it first.
- **macOS:** double-click the zip (Safari may already have unpacked it for you).
- **Linux:** `tar xzf GetToWork-*-linux-x64.tar.gz` (the `.tar.gz` keeps the
  programs executable, so no `chmod` is needed).

You get a `GetToWork` folder with a `README.txt` and the game.

**3. Double-click the game** - `GetToWork.exe` (Windows), `Get To Work.app`
(macOS) or `GetToWork` (Linux). The game opens its own window and walks you
through everything:

1. a short note that the story is written by an AI (first launch only);
2. a quick **hardware check** of your computer;
3. **picking a model** - press Enter for the recommended one - and one
   confirmation screen, then the **model download** from Hugging Face with a
   progress bar (skipped when you already have one);
4. a speed test, then the optional **Jev** question (say `no` to play with
   your local model only);
5. **the game** - and "Play again?" at the end.

Next time it's one question: *"Welcome back! Play with Qwen3 4B again?"* and
you're in. (If you said no to Jev, it isn't asked again - choose **Play with
Jev on this time** (type `jev`) at that Welcome back question to set it up
after all. A "no" while trying the pretend model isn't remembered: your first
game with a real model still asks. Tried the pretend model first? After its
game, choose **Pick a real AI model** to go back to the model menu.)

**The builds aren't code-signed yet**, so the first launch may need an extra
step:

- **Windows:** SmartScreen may say "Windows protected your PC": choose
  **More info**, then **Run anyway** (builds installed through Steam don't get
  this prompt). **Smart App Control** is different: on a Windows 11 PC where
  it's switched on, it blocks unsigned programs outright - test builds *and*
  Steam builds - with no Run anyway ("Smart App Control blocked an app that
  may be unsafe", or Steam error **0x11C7**), and the game's engine
  (`llama-server.exe`) is blocked the same way when the game starts it. Until
  the builds are code-signed (the `build` workflow signs them once a
  certificate is added - see [Releasing on Steam](#releasing-on-steam)), such
  a PC can play with the free, signed [Ollama](https://ollama.com/download)
  app as the engine, or switch Smart App Control off (Windows Security > App &
  browser control > Smart App Control) - but Windows can only switch it back
  on with a reset, so that's the player's call.
- **macOS:** Gatekeeper refuses apps from unidentified developers. Right-click
  the app and choose **Open**; on macOS 15 or later, try to open it once, then
  go to **System Settings > Privacy & Security** and click **Open Anyway**.
  Or, in Terminal: `xattr -dr com.apple.quarantine "Get To Work.app"`.
- **Linux:** if your file manager won't run programs by double-click, open a
  terminal in the folder and type `./GetToWork`.

Each build also contains the **terminal version**, `gettowork-cli`
(`gettowork-cli.exe` on Windows; on macOS it's
`Get To Work.app/Contents/MacOS/gettowork-cli`). It takes every
[command-line option](#command-line-options), for example
`./gettowork-cli --mock` (macOS/Linux) or `.\gettowork-cli.exe --mock`
(Windows) to try the game instantly with a pretend model.
To give the **window** an option (say `--jev` or `--think`), start it from a
terminal in the game's folder:

- **Windows:** `.\GetToWork.exe --jev` (works in both PowerShell - Windows
  Terminal's default - and Command Prompt)
- **macOS:** `open "Get To Work.app" --args --jev`
- **Linux:** `./GetToWork --jev`
- **Steam:** right-click Get To Work > **Properties > General > Launch Options**.

## Playing from source

For development and tinkering. You need **Python 3.10 or newer** (from
[python.org](https://www.python.org/downloads/); on Windows tick
**"Add python.exe to PATH"**) and a copy of this repository (it's private, so
you need access to clone it).

```bash
git clone https://github.com/markelphoenix/GetToWork
cd GetToWork
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -e .
```

Then start it either way:

```bash
gettowork         # in this terminal
gettowork-gui     # in the game's own window, like the Steam build
```

`python -m gettowork` does the same as `gettowork` (handy if your terminal
says `gettowork` isn't found), and `python -m gettowork.launcher` opens the
window. (`pipx install .` from the folder also works, if you prefer pipx.)

> **Want to try it right now, with no downloads at all?**
> Run `gettowork --mock`. A pretend, scripted "model" plays along instantly and
> offline, so you can see how everything works before downloading anything.

A copy run from source doesn't have the engine built in, so the first real
game also downloads the official llama.cpp engine from GitHub (see
[What's bundled and what's downloaded](#whats-bundled-and-whats-downloaded)).
The window needs Python's Tk toolkit: the python.org installers include it,
and on Linux it's a separate package (for example
`sudo apt install python3-tk`).

## What happens when you play

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
   any family-friendly Hugging Face GGUF repo), `mock` (play offline), `learn` (how it chose)
   or `why 2` (the working behind pick number 2).
4. **One confirmation.** A single screen lists exactly what will be downloaded,
   how big it is, where it comes from, its license, and where it will be saved.
   In the Steam and test builds the engine line reads *"Built into the game
   (llama.cpp ..., Vulkan + CPU) - nothing to download"* (just "CPU" on a
   computer without a graphics card the engine can use; "Metal + CPU" on a Mac).
5. **The engine.** The Steam and test builds use the **llama.cpp** engine
   (`llama-server`) they ship with: Vulkan and CPU builds on Windows and
   Linux, the Metal build on Apple Silicon. A copy run from source downloads
   the official prebuilt engine for your computer from the
   [ggml-org/llama.cpp GitHub releases](https://github.com/ggml-org/llama.cpp/releases)
   instead: an NVIDIA CUDA, Vulkan, Apple Metal or plain CPU build (usually a
   few tens of MB; CUDA builds are about half a gigabyte because they include
   NVIDIA's CUDA runtime). It checks the file's size and SHA-256 fingerprint
   and unpacks it into the game's own folder. No compilers, no admin rights,
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
   falls back automatically to the next build, ending with the CPU build (or
   CPU mode), which works everywhere. On a computer where the ready-made
   engine can't run at all (a very old Linux, say), it tells you *before*
   downloading anything.
8. **Jev (optional)**, then the game, the review, and "Play again?".
9. **Next time:** *"Welcome back! Play with Qwen3 4B again? [Y/n]"* and you're
   straight into the game (if you chose to play without Jev last time, that
   question has a **jev** choice to turn it on after all, and `--jev` asks
   again too). Each "Learn" panel appears
   once per session, not again after "Play again". When you quit, the engine
   is always stopped.

## What's bundled and what's downloaded

| Part | Steam and test builds | Copy run from source |
|---|---|---|
| The game, Python and its libraries, the window toolkit (Tcl/Tk) | built in | installed by `pip` |
| The **llama.cpp engine** | **built in** - never downloaded while you play | downloaded from GitHub on first use (after you confirm) |
| An **AI model** (a GGUF file, from under 1 GB to 20+ GB) | downloaded from Hugging Face on first run, after you pick one and confirm | the same |
| **Jev** (an online referee) | optional, off unless you turn it on | the same |

The license texts of everything built into the game ship with it in
`THIRD_PARTY_LICENSES.txt` (see [NOTICE.md](NOTICE.md)).

## Where files are stored (and how to delete them)

Everything the game saves lives in one folder, outside the game itself (so
updating, verifying or uninstalling the game on Steam never deletes a model
you downloaded):

| System  | Folder |
|---------|--------|
| Windows | `%LOCALAPPDATA%\GetToWork` (for example `C:\Users\you\AppData\Local\GetToWork`; kept out of your roaming profile because models are big. A folder an older version made in `%APPDATA%\GetToWork` keeps being used.) |
| macOS   | `~/Library/Application Support/GetToWork` |
| Linux   | `~/.config/gettowork` (or `$XDG_CONFIG_HOME/gettowork`) |

Inside it:

| Path | What it is |
|------|------------|
| `settings.json` | Your choices (model, backend, whether Jev is on, the window's font size) and the speed calibration measured on this computer. Your Jev key is stored here **only** if you said yes to saving it. |
| `models/` | Downloaded model files, one folder per Hugging Face repo, e.g. `models/unsloth--Qwen3-4B-GGUF/`. These are the big ones (1-20+ GB). |
| `runtime/llama.cpp/` | The llama.cpp engine a copy run from source downloaded, one folder per build, e.g. `runtime/llama.cpp/b6000-vulkan/`. When an update replaces a build, the old copy is tidied away once the new one works. (The Steam and test builds keep their engine inside the game instead.) |
| `runtime/logs/` | The engine's log (`llama-server.log`; on Windows each game writes its own `llama-server-<number>.log`), handy when something goes wrong. |
| `cache/hf_models.json` | The saved Hugging Face search results. |
| `logs/crash.txt` | Only if the game ever hit an unexpected error: the technical details, for a bug report (the game says so when it happens). |
| `logs/gui-crash.txt` | Only if the game's window ever failed: what went wrong, for a bug report. |

**Moving or deleting things:**

- To keep the models on another drive (a bigger one, or a Steam Deck's microSD
  card), start the game once with `--models-dir <folder>` - on Steam, add it
  under **Properties > General > Launch Options**, which works on Windows too
  - and it's remembered (the free-space check then looks at that drive). Or
  set `GETTOWORK_HOME` to use a different folder for everything, or
  `GETTOWORK_MODELS_DIR` for just the models.
- `gettowork --reset` forgets your saved settings: your choices, the speed
  calibration and any saved key. It doesn't delete models.
- To remove a model, delete its folder under `models/`. To remove everything,
  delete the whole folder above, then uninstall the game (on Steam, or by
  deleting the `GetToWork` folder / `Get To Work.app`; a copy from source:
  `pip uninstall gettowork`).
- Transcripts you choose to save (`gettowork-transcript-1.json` / `.md`) go in
  `Documents/Get To Work` when you play in the game's window, and in the
  folder you ran the terminal version from otherwise - or wherever
  `--export-dir` says.
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
- **[llama-cpp-python](https://github.com/abetlen/llama-cpp-python)** (copies
  run from source only): install it with `pip install llama-cpp-python` (or
  `pip install -e ".[llamacpp]"` from this repo), then run
  `gettowork --backend llamacpp`. On some systems pip has to compile it,
  which needs a C/C++ compiler.
- **Mock**: `gettowork --mock` for the scripted offline pretend model.
- **Your own GGUF file**: `gettowork --gguf path/to/model.gguf` runs it with
  the built-in engine.

With the default `--backend auto`, if the built-in engine can't be installed
or started, the game offers to use Ollama (if it's running; it asks first, and
hands Ollama the model file you already downloaded instead of downloading it
again) or llama-cpp-python (if it's installed), and tells you what it's doing.
If a brand-new model needs a newer engine, a copy run from source updates the
engine once by itself (the Steam build gets newer engines with game updates,
and says so). If nothing works, it offers to try again, pick another model,
play with the pretend model, or stop, and anything already downloaded is kept.

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
2. Paste your API key (it stays hidden while you type - the game's window
   masks it; if a terminal can't hide it, for example an IDE's Run console,
   the game warns you first and suggests the environment variable below),
   **or** choose *"walk me through getting one"*: the game opens
   https://typesafe.ai in your browser and lists the steps (sign up or log in,
   open the API keys page in your dashboard, create a key, copy it). The
   documentation is at https://docs.typesafe.ai/.
3. The key is checked with a quick request. Saving it for next time is
   **opt-in** (default: no). It would be stored in plain text in
   `settings.json` (readable only by you: file permissions on macOS and Linux, an owner-only access list on Windows). If you switch to
   a new key without saving it, or Jev stops accepting the saved one, the old
   key is removed from the file.

You can back out to local-only at **every** step: choose `no` or `back`,
press Enter at the key prompt, or press Ctrl+C (even while the key is being
checked). `gettowork --no-jev` skips the question entirely. Once you've
chosen to play with your local model only, the game doesn't ask again: choose
**Play with Jev on this time** (type `jev`) at the *Welcome back* question,
or start it with `--jev` (on Steam:
Properties > General > Launch Options), to set Jev up later. (A "no" given
while trying the pretend model isn't remembered.)

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
  as long as they deal with *this* challenge - and stay family-friendly.
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

Then you can save a transcript as JSON and Markdown (your API key is never
included), and the game asks **"Play again?"**.

### In the game's window

The window shows the same game as the terminal, with a few extras:

- **Menus and yes/no questions get big buttons** under the story, for a mouse,
  a touchscreen or a Steam Deck - the model menu too (its first picks, More
  models, Pretend model, How I chose, Quit). Typing works everywhere too;
  **Enter** answers.
- **Pasting** (Ctrl+V, Cmd+V on a Mac, Ctrl+Shift+V or right-click > Paste)
  always goes into the answer box, even after a click on the story. An API
  key shows as dots.
- **Report a problem** opens the game's Steam Discussions in your browser.
- **Up / Down** bring back what you typed before; **Page Up / Page Down**
  scroll the story. Links are clickable.
- **Ctrl + = / Ctrl + - / Ctrl + 0** make the text bigger, smaller or
  normal again (also Cmd on a Mac); the size is remembered.
- **F11** switches full screen on and off (**Escape** leaves it). On a Steam
  Deck, and in Steam's Big Picture mode, the window starts full screen with
  bigger text (16 points on the Deck's own screen) and a **Keyboard** button
  that opens Steam's on-screen keyboard. On a Deck the question and the input
  bar sit at the top of the screen, so the keyboard never covers them.
- An API key typed or pasted anywhere but the key question shows as
  "(hidden)" and is never offered again with Up. Pasting your Jev key straight
  into the "How would you like to add your Jev API key?" or "What next?" menu
  just works.
- **Closing the window** ends the game politely: it says goodbye and stops the
  engine. When a game is over, press Enter or close the window to exit.

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

These work with `gettowork` (from source), with `gettowork-cli` in a built
game, and with the game's window too: `gettowork-gui --mock`, or on Steam
through **Properties > General > Launch Options**.

| Option | What it does |
|--------|--------------|
| `--mock` | Play with a scripted, offline pretend model. No downloads. |
| `--backend {auto,managed,ollama,llamacpp}` | Which engine runs the model. `auto` (default) uses the built-in llama.cpp engine (`managed`), falling back to Ollama if it's running. |
| `--model REPO_OR_KEY` | Skip the picker and use this model, e.g. `unsloth/Qwen3-4B-GGUF` or `qwen3-4b`. |
| `--quant TAG` | Use this quantization, e.g. `Q4_K_M` or `Q8_0`. |
| `--ollama-model TAG` | Use a model you already have in Ollama. |
| `--gguf PATH` | Run a GGUF file you already have, with the built-in engine. |
| `--list-models` | Print the ranked shortlist for this computer, then exit. |
| `--specs` | Print what the game found about your hardware (including memory speed), the engine builds it would try and where its engine comes from, then exit. |
| `--refresh-models` | Ignore the saved Hugging Face results and search again. |
| `--offline` | Don't go online for the model list: use the saved results or the built-in list. Downloads then only work for files you already have. |
| `--all-licenses` | Also show models whose licenses aren't Apache-2.0 or MIT. Each license is shown clearly; complying with it is up to you. (A model you name yourself with `custom` or `--model` can have any license too, always with a warning.) |
| `--no-jev` | Don't ask about Jev; the local model referees. |
| `--jev` | Ask about Jev again, after you chose to play with the local model only. |
| `--models-dir DIR` | Keep downloaded models in this folder from now on (a bigger drive, say); remembered. |
| `--think` | Let a "thinking" model think out loud even on a slow computer (each turn takes longer). |
| `--target N` | Steps needed to win (default 5). |
| `--reset` | Forget saved settings: your choices, speed calibration and any saved key (models stay on disk). |
| `--export-dir DIR` | Where to save transcripts (default: the current folder; `Documents/Get To Work` in the game's window). |
| `--debug` | Show full error details (tracebacks). Useful for bug reports. |
| `--version` | Print the version and exit. |

Press Ctrl+C at any time to leave cleanly. The engine is always stopped on the
way out.

## AI content and the family-friendly filter

The story is **written live by an AI** on your computer, so no two mornings
are alike - and nobody reviewed the words before you see them. The game says
so in a short note the first time it starts, and it has guardrails (Steam asks
games with live-generated AI content for exactly this):

- **Friendly instructions.** Every request asks the model for farcical,
  family-friendly slapstick in which nobody gets hurt, and your plan is always
  quoted as an in-story action, never followed as an instruction.
- **A filter on everything the model writes** ([safety.py](src/gettowork/safety.py)):
  story text, challenges, endings, the referee's explanations and any
  "thinking" shown in the review. It blocks sexual content, slurs and hate
  speech, self-harm, graphic gore and hard drugs, and sees through common
  disguises (capitals, accents, look-alike letters, leetspeak, s p a c e d or
  broken-up words). Whole words only, so "Scunthorpe", "assassin" and
  "classic" are fine. Milder swearing is masked, like "d***".
- **Blocked means never shown.** The game asks the model once more with a
  firmer reminder; if that fails too, a pre-written line takes its place. In
  the review and saved transcripts a blocked reply appears only as a short
  "hidden by the family-friendly filter" note, never the words.
- **Your plans are checked too.** A plan that doesn't pass is refused with
  *"Let's keep it family-friendly - try another plan!"* before it reaches the
  model or Jev, and it doesn't cost you a round.
- **Only family-friendly models are suggested**: the model search leaves out
  models marketed as uncensored, "abliterated" or made for adult content, and
  one you name yourself (`custom` or `--model`) is refused too.

No filter is perfect. If you ever see something that shouldn't be there,
please report it: click **Report a problem** in the game's window (it opens
the game's Steam Discussions in your browser), or open the Discussions from the
game's Steam store page; while testing, tell the maintainer. (Steam's
Shift+Tab overlay can't open over the game's window on Windows, macOS or a
Linux desktop; on a Steam Deck in Game Mode the Steam button's menu works.) The exact wording the
Steam store page uses is in [`notices.py`](src/gettowork/notices.py)
(`STEAM_AI_DISCLOSURE`).

## Privacy: what leaves your computer

The story and your plans stay on your computer unless you turn on Jev. The
model runs locally and is only reachable at `127.0.0.1`.

| When | Sent to | What |
|------|---------|------|
| Model search (at most every 3 days, or `refresh`) | Hugging Face (`huggingface.co`) | Search requests for public GGUF models, and file-list requests for the most promising ones. |
| Downloading a model | Hugging Face | Requests for the model file(s) you confirmed. |
| Downloading the engine (copies run from source only: first time, or a new build) | GitHub (`api.github.com`, `github.com` and its download servers) | A request for the list of recent llama.cpp releases, then the download of one archive. If you set `GITHUB_TOKEN`, it's sent only to `api.github.com`. The Steam and test builds never do this: their engine is built in. |
| Each round, **only if Jev is enabled** | TypeSafe AI (`api.typesafe.ai`) | This round's state: a trimmed "story so far", the current challenge, your plan, your progress and a short summary of recent rounds (including your earlier plans), plus the three questions and your API key (in the `Authorization` header). The key is only ever sent over https to that address: the game never follows a redirect elsewhere. |
| Checking your Jev key | TypeSafe AI | One `GET /v1/models` request with your key. |
| "Open the website" in the Jev help, or a link you click | Your web browser | Only when you ask; it opens typesafe.ai, its docs, or the link. |
| The **Keyboard** button (Steam Deck) | The Steam app on your computer | A `steam://open/keyboard` request, so Steam shows its on-screen keyboard. |
| **Report a problem** in the game's window | Your web browser | Only when you click it: it opens the game's Steam Discussions. |

That's all. **The game has no telemetry, analytics or crash reporting** (an
unexpected error or a window problem is written to a file on your computer,
`logs/crash.txt` or `logs/gui-crash.txt`, and stays there). Like any website, these services see your IP address and
basic technical details such as the app's user-agent. If you've logged in with
`hf auth login`, `huggingface_hub` includes your Hugging Face token in its
requests (that's how gated models work). With Ollama, requests go to your
local Ollama app (unless you point `OLLAMA_HOST` at another machine), and
Ollama downloads the model from Hugging Face.

## Troubleshooting

After an unexpected error the technical details are saved in
`logs/crash.txt` (the game says where); the terminal version's `--debug`
shows them on screen. Look at the engine's log in `runtime/logs/llama-server.log` (on Windows,
`llama-server-<number>.log`; see
[where files are stored](#where-files-are-stored-and-how-to-delete-them)).

### The game's window won't open

- **Linux:** the window needs a desktop session - X11, or Wayland with
  XWayland (every mainstream desktop and Steam Deck's Game Mode have it). Over
  SSH or on a server without a display, use the terminal version
  (`./gettowork-cli`, or `gettowork` from source).
- **From source:** the window needs Python's Tk toolkit. On Linux install it
  with your package manager (`sudo apt install python3-tk`, or `python3-tkinter`
  on Fedora); the python.org installers for Windows and macOS include it.
- Started from a terminal, the game plays right there in the terminal if the
  window can't open; started by a double-click or from Steam, a message box
  says why (and, on Steam, suggests **Properties > Installed Files > Verify
  integrity of game files**). Either way, the reason is saved in `logs/gui-crash.txt`
  in the game's folder (see
  [where files are stored](#where-files-are-stored-and-how-to-delete-them)) -
  please include it in a bug report.

### Steam Deck

- **Typing:** press **STEAM + X** for the on-screen keyboard at any time, or
  tap the game's **Keyboard** button. The question and the input bar are at the
  top of the screen, above the keyboard. Menus also have big buttons you can tap.
- **Full screen:** in Game Mode the game fills the screen by itself; **F11**
  toggles it in Desktop Mode.
- The Deck runs the native Linux build - no Proton needed - and uses its
  graphics through the built-in Vulkan engine (falling back to the CPU).

### "The game's built-in engine is missing"

The Steam or test build lost part of its files (an antivirus quarantine, an
interrupted update). On Steam: right-click Get To Work, then
**Properties > Installed Files > Verify integrity of game files**. Otherwise
download and unpack the build again.

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

- The Steam and test builds use Vulkan on Windows and Linux (NVIDIA, AMD and
  Intel graphics alike) and Metal on Apple Silicon, with a CPU fallback. Up to
  date graphics drivers help most.
- A copy run from source can also download NVIDIA's CUDA builds, which need a
  reasonably recent driver (about version 525 or newer for CUDA 12, 580 or
  newer for CUDA 13). Updating your driver from NVIDIA's website often helps.
  Older cards (GTX 9xx/10xx, Titan V) get the CUDA 12 build, because CUDA 13
  no longer supports them.
- If a GPU build won't start, the game reads the engine's log, explains, and
  **falls back automatically**: CUDA, then Vulkan, then CPU. Everything still
  works on the CPU, just slower.
- On Linux, the Vulkan build needs the Vulkan loader and a Vulkan driver (for
  example the `libvulkan1` and `mesa-vulkan-drivers` packages on Ubuntu;
  SteamOS has them). Without them, the CPU build is used.
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
- GitHub limits anonymous requests to its API to 60 an hour (this only
  matters for copies run from source, which download the engine). If you hit
  that limit, wait a bit or set a `GITHUB_TOKEN`.
- If Hugging Face is blocked entirely, try `--offline` with a model you
  already have, Ollama, or `--mock`.

### "Gated" models

Some models ask you to accept their authors' terms before downloading. The
game avoids them by default. If you pick one (for example with `custom`),
open the model's page on Hugging Face while logged in, accept the terms, run
`hf auth login` in a terminal, and try again. Or simply pick another model.

### Windows: SmartScreen or antivirus warnings

- The test builds aren't code-signed yet, so SmartScreen may warn on first
  launch (choose **More info**, then **Run anyway**). Builds installed through
  Steam don't get this prompt - but on a Windows 11 PC with **Smart App
  Control** on, unsigned programs are blocked outright, from Steam too (Steam
  error 0x11C7; if the game itself starts, it says so when Windows blocks its
  engine). See [Playing the test builds](#playing-the-test-builds) for what
  works there.
- The engine (`llama-server.exe`) is the official build from the llama.cpp
  project. Like many open-source tools it may not carry a commercial
  code-signing certificate, so Windows doesn't "know" it yet. Every copy the
  game uses was checked against the size and SHA-256 fingerprint GitHub
  reports before it was unpacked (by the build, or by the game itself).
- Some antivirus tools flag or quarantine new, unsigned programs. If yours does,
  you can compare the file with the
  [official releases page](https://github.com/ggml-org/llama.cpp/releases)
  and, if you're comfortable, allow the game's folder. Or use Ollama.
- If Windows Firewall asks about `llama-server.exe`, you can choose not to allow
  network access. The game only talks to it on `127.0.0.1`, your own computer.
- The game's engine brings its own copy of the Microsoft Visual C++ runtime.
  If the engine still says a system library is missing, install the free
  Microsoft Visual C++ Redistributable (x64) - Steam installs it for you; the
  game names the download link.

### macOS

- The test build is for Apple Silicon (M1 and later) and uses the Metal
  build, the GPU and unified memory automatically. On an Intel Mac, play from
  source: it uses the CPU build, which is slower, so smaller models are best.
- If macOS says the app "can't be opened" or "cannot be verified", see the
  unsigned-build steps in [Playing the test builds](#playing-the-test-builds).
- Playing from source and macOS says `llama-server` "can't be opened"? The
  engine's folder has been marked as downloaded from the internet. You can
  clear that mark for the game's runtime folder only with
  `xattr -dr com.apple.quarantine ~/Library/Application\ Support/GetToWork/runtime`,
  or use Ollama instead.
- macOS older than 13.3 can't run the current prebuilt engine; Ollama or
  `--mock` still work.

### Jev says my key doesn't work

Check that you copied the whole key (no spaces), that it hasn't been deleted in
your TypeSafe dashboard, and that your account is in good standing. A "payment
required" message comes from TypeSafe's billing. You can always choose `back`
and play locally.

### Start fresh

`gettowork --reset` (`gettowork-cli --reset` in a built game) forgets your
saved choices; after an unexpected error the game's window offers to do the
same right there. Deleting the whole data folder starts completely from
scratch.

## Releasing on Steam

Get To Work is planned as a **free** Steam game. The builds the `build`
workflow makes are exactly what gets uploaded, and every step is written
down:

- [packaging/steam/README.md](packaging/steam/README.md) - the release
  checklist: Steamworks setup (depots, launch options, the Visual C++
  redistributable, the app ID for the Report a problem button, code-signing
  the Windows build for Smart App Control), uploading with SteamPipe, Steam
  Deck notes (the Linux runtime: the engine carries its own OpenSSL 3);
- [packaging/steam/STORE_PAGE.md](packaging/steam/STORE_PAGE.md) - store page
  text, the AI-content disclosure answers, the Jev notice, privacy and system
  requirements;
- [docs/DISTRIBUTION.md](docs/DISTRIBUTION.md) - how a build is put together
  (bundled engine, `distribution.json`, license texts) and how CI tests it.

The owner may later offer their other game, Gridfall, together with Get To
Work (for example as a paid DLC or bundle). That needs no change to these
builds: a bundle groups separate Steam apps.

## Project layout

```text
src/gettowork/
  __main__.py        `python -m gettowork` starts the game (terminal version)
  cli.py             command-line options, then setup -> Jev -> game -> review -> play again
  launcher.py        the game's window as Steam / a double-click starts it (`gettowork-gui`)
  gui/               the game's own window: app.py (Tk), bridge.py, terminal.py
  setup_flow.py      hardware -> discovery -> pick -> install -> warm-up
  types.py           shared data classes (read this first!)
  ui.py              all input/output (built on rich), in a terminal or the window
  config.py          settings file and data folders
  crashlog.py        saves what went wrong to logs/ (for a bug report)
  specs.py           hardware detection
  perf.py            memory-speed test and tokens/sec estimates
  catalog.py         built-in model list and the fit/ranking engine
  hf_discovery.py    live Hugging Face search, GGUF metadata, cache
  download.py        model downloads from Hugging Face
  runtime_install.py the llama.cpp engine: built-in builds, or the official download
  distribution.py    built game or developer copy? (a built game ships its engine)
  tls.py             HTTPS certificate checks that work on every computer
  reasoning.py       separates chain-of-thought from answers
  backends/          llamaserver.py (default), ollama.py, llamacpp.py, mock.py
  jev.py             Jev API client, the game's three questions, verdicts
  onboarding.py      the friendly "enable Jev?" flow
  prompts.py         every word the game says to the local model
  safety.py          the family-friendly filter for AI text and typed plans
  safety_terms.py    its word lists (scrambled with ROT13, so they're not on show)
  notices.py         the AI-content notice and the Steam AI disclosure text
  game.py            the core game loop
  review.py          end-of-game review and transcript export
  assets/icon.png    the window icon (drawn by packaging/make_icon.py)
packaging/           how the double-click / Steam builds are made (see docs/DISTRIBUTION.md)
  steam/             SteamPipe templates, the release checklist, store page text
tests/               pytest suite: no network, no real models
docs/ARCHITECTURE.md how the modules fit together (the contract)
docs/DISTRIBUTION.md how the builds are made, tested and shipped
docs/LEARN.md        the mini-course
```

## Contributing

Contributions, bug reports and ideas are very welcome from everyone with
access to the repository. See [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, how the model list and filters work, and the (short)
rules.

## License and disclaimer

Get To Work is **free**: the game costs nothing, on Steam or anywhere else.
Its source code is licensed under the [MIT License](LICENSE); the source
repository itself is private. Third-party components and services - including
the software built into the game builds, whose license texts ship with every
build in `THIRD_PARTY_LICENSES.txt` - are listed in [NOTICE.md](NOTICE.md).

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
>   Ollama, Valve / Steam, or any model author or publisher. Product and
>   company names are trademarks of their respective owners and are used only
>   to identify their products and services.
> - **Third-party downloads are your responsibility.** No model weights are
>   included in the game or this project. When you confirm, you download them
>   yourself from Hugging Face under their authors' own licenses and terms, and
>   you are responsible for complying with them. The game shows each model's
>   license before downloading; `--all-licenses` can show licenses with extra
>   conditions. The game builds include the llama.cpp engine under its MIT
>   license; a copy run from source downloads it from GitHub under the same
>   license (and, for NVIDIA CUDA builds, NVIDIA's terms).
> - **Jev may cost money.** Using Jev may incur charges under TypeSafe AI's own
>   pricing and terms. The game works fully without it.
> - **AI output is unpredictable.** The prompts ask for farcical, family-friendly
>   stories and a filter checks what the model writes, but AI models can still
>   produce odd, wrong or inappropriate text. Nothing the game or a model says
>   is advice.
