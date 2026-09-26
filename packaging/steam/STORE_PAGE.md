# Store page and Steamworks answers

Copy-ready text for Get To Work's Steam store page and the Steamworks forms.
Anything marked **Owner: confirm** is a draft only the owner can finish.

## Basics

- **Price:** Free to Play (no in-game purchases).
- **Genres:** Casual, Indie, Simulation. **Suggested tags:** Education, Text-Based,
  Comedy, Choose Your Own Adventure, AI, Family Friendly, Singleplayer.
- **Short description** (at most 300 characters):

  > It's 8:45 and work starts at 9:00 - but squirrels run a toll booth on your
  > path. Type your way to work in a silly text adventure told live by an AI that
  > runs on your own computer, and learn how local AI models work along the way.

- **About this game** (outline): every morning goes wrong in a new way; the
  player types plans, a local AI narrates what happens and judges the plan; the
  game explains the tech as it goes (picking a model that fits the computer,
  what quantization and video memory mean, how "thinking" models reason); an
  end-of-game review shows what the model did. Setup is guided: the game checks
  the computer, suggests a model that fits, downloads it once, and plays -
  no accounts needed.

## AI-generated content disclosure (Content Survey)

**Pre-generated content** - Owner: confirm. Draft: "The game's fixed text
(menus, tutorials, the pre-written fallback lines) was written by the developer.
AI coding assistants were used while developing the game's code." (Adjust this
to what was actually used, and list any AI-made art, music or store images.)

**Live-generated content: Yes.** Paste this text exactly (it lives in
`src/gettowork/notices.py` as `STEAM_AI_DISCLOSURE`; a test keeps this copy in
step with it):

```text
Get To Work uses generative AI to write story text live while you play. The text is written by an open-weight language model that runs locally on the player's own computer, through the llama.cpp engine that ships with the game, so the story is written on the player's machine rather than on a server. The in-game model menu only offers popular, instruction-tuned chat models, and it leaves out models that are marketed as uncensored, have had their safety training removed, or are made for adult content; such a model is refused even when a player names it themselves. Optionally, players can connect their own account for TypeSafe AI's Jev referee service: it then receives the player's plan and a short summary of the story, and returns only numbers and labels that decide each round - it writes no text that appears in the story. The game creates no AI images, audio or voices.

Guardrails for the live-generated text:
- Every request tells the model to write farcical, family-friendly slapstick in which nobody gets hurt, and the player's typed plan is always quoted as an in-story action, never followed as an instruction.
- Everything the model writes - story narration, challenges, endings, the referee's explanations, and any "thinking" shown in the end-of-game review - is checked by a built-in filter before it is shown. The filter blocks sexual content, slurs and hate speech, self-harm, graphic gore and hard drugs, and it sees through common disguises (capital letters, accents, look-alike characters, leetspeak, and letters spaced out or broken up with symbols). Milder swearing is masked (for example "d***").
- A blocked reply is never shown: the game asks the model once more with a stricter family-friendly instruction, and if the new reply is also blocked, it shows a pre-written line instead.
- What the player types is checked with the same filter; a blocked plan is refused with a friendly message before it reaches any AI model or service.
- On first launch the game explains that its story is AI-generated and how to report anything inappropriate: a "Report a problem" button in the game's window opens the game's Steam Discussions, which are also reachable from the store page.
```

**Adult Only Sexual Content:** not applicable - the game has none, and the
filter above blocks it.

## Optional third-party service: Jev

The game works fully offline with the player's local model. Jev is optional, so
don't tick "requires a third-party account". Put this notice in "About this
game":

> **Optional:** Get To Work can use **Jev**, TypeSafe AI's typed-judgment API,
> as an extra referee. Jev is a paid third-party service: it needs your own
> TypeSafe AI account and API key and has its own pricing, terms and privacy
> policy. It is off unless you turn it on, and the whole game is free to play
> without it. Get To Work isn't affiliated with TypeSafe AI.

## Privacy summary

- **No account, no tracking.** The game has no sign-in, no analytics and no ads.
- **The hardware check stays on the computer.** Operating system, processor,
  memory, graphics card and free disk space are only used to pick a model that
  fits.
- **Downloads.** The game searches Hugging Face for suitable models and
  downloads the one the player picks (a normal, anonymous download). The AI
  engine ships inside the game, so no programs are downloaded.
- **The story is written on the player's computer.** With the local model only,
  nothing the player types leaves the computer.
- **Jev, only when switched on:** each round sends the typed plan, the current
  challenge, a short summary of the story and the player's progress to TypeSafe
  AI (api.typesafe.ai), under their terms and privacy policy. The API key is
  stored on the computer only if the player asks the game to remember it.
- **Settings and models** are saved in the game's own folder on the player's
  computer (`%LOCALAPPDATA%\GetToWork` on Windows,
  `~/Library/Application Support/GetToWork` on macOS, `~/.config/gettowork` on
  Linux).

## System requirements

These follow the game's own fit engine (`src/gettowork/catalog.py`), which
picks a model for each computer; `tests/test_packaging.py` checks the memory
figures against it. With the minimum memory it recommends a small model that
runs comfortably on the processor alone; with the recommended setup it picks
4B-8B-class models that tell a richer story. Everything runs locally, so more
memory and a graphics card mean smarter and faster storytelling.

### Windows

| | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 64-bit | Windows 11 64-bit |
| Processor | 64-bit (x86-64) processor | 4 or more cores with AVX2 (2015 or newer) |
| Memory | 8 GB RAM | 16 GB RAM |
| Graphics | Not required - the AI can run on the processor | Vulkan-capable graphics card with 6 GB+ video memory (NVIDIA, AMD or Intel) |
| Storage | 2 GB available space | 10 GB available space |
| Additional notes | Internet connection for the one-time model download | Bigger models download 5-10 GB once |

### macOS

| | Minimum | Recommended |
|---|---|---|
| OS | macOS 13.3 Ventura | macOS 14 Sonoma or later |
| Processor | Apple Silicon (M1) - Intel Macs aren't supported | Apple M2 or newer |
| Memory | 8 GB RAM | 16 GB RAM |
| Graphics | Built in (Apple Silicon, Metal) | Built in (Apple Silicon, Metal) |
| Storage | 2 GB available space | 10 GB available space |
| Additional notes | Internet connection for the one-time model download | Bigger models download 5-10 GB once |

### SteamOS + Linux

| | Minimum | Recommended |
|---|---|---|
| OS | SteamOS 3.5 or Ubuntu 22.04 (64-bit, glibc 2.35 or newer) | SteamOS 3.6 or a current 64-bit distribution |
| Processor | 64-bit (x86-64) processor | 4 or more cores with AVX2 (2015 or newer) |
| Memory | 8 GB RAM | 16 GB RAM |
| Graphics | Not required - the AI can run on the processor | Vulkan-capable graphics card with 6 GB+ video memory (Steam Deck: built in) |
| Storage | 2 GB available space | 10 GB available space |
| Additional notes | Runs natively on Steam Deck (no Proton); X11 or XWayland desktop | Internet connection for the one-time model download |

The game itself takes well under 1 GB; the rest of the space is for the AI
model it downloads (0.5-2 GB on minimum computers).
