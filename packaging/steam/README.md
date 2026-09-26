# Putting Get To Work on Steam

Get To Work is a **free** Steam game. Each push to `main` makes the three builds
Steam needs (the [`build` workflow](../../.github/workflows/build.yml)), already
tested: the terminal version plays a whole game, the bundled llama.cpp engine
starts, and the game window plays a scripted game (`--gui-selftest`). This
folder has what's needed to upload them with SteamPipe:

| File | What it's for |
|---|---|
| `app_build_all.vdf` | uploads all three depots at once (run it on macOS or Linux) |
| `app_build_windows.vdf`, `app_build_macos.vdf`, `app_build_linux.vdf` | upload one depot |
| `depot_build_windows.vdf`, `depot_build_macos.vdf`, `depot_build_linux.vdf` | which files go into each depot |
| [`STORE_PAGE.md`](STORE_PAGE.md) | store page text: AI disclosure, Jev notice, privacy, system requirements |

The templates contain placeholders: `<APP_ID>`, `<DEPOT_ID_WIN>`,
`<DEPOT_ID_MAC>`, `<DEPOT_ID_LINUX>` and `<VERSION>`. Steamworks shows the IDs
under **App Admin > SteamPipe > Depots**.

## One-time setup in Steamworks

1. **Depots** (SteamPipe > Depots): three depots, one per operating system:
   Windows (64-bit), macOS, and Linux + SteamOS. Publish the change.
2. **Launch options** (Installation > General Installation) - one per OS,
   no arguments, working directory left empty (the game finds its own files):

   | Operating system | Executable | Notes |
   |---|---|---|
   | Windows | `GetToWork\GetToWork.exe` | CPU architecture: 64-bit |
   | macOS | `Get To Work.app` | Apple Silicon only (M1 or later) |
   | Linux + SteamOS | `GetToWork/GetToWork` | also what Steam Deck runs |

   Each build also contains the terminal version (`gettowork-cli`); it doesn't
   need a launch option.
3. **Redistributables** (Installation > Redistributables): tick
   **Visual C++ Redist 2015-2022 (x64)**. The bundled llama.cpp engine needs it
   on Windows. Each engine folder already carries its own copy of the three
   runtime files (`fetch_engine.py` adds them), so this is a safety net: Steam
   installs the full redistributable for players who don't have it yet.
4. **Content survey** (the AI-generated content questions) and the store page:
   copy the answers from [`STORE_PAGE.md`](STORE_PAGE.md).
5. **The app ID in the game.** Put the game's app ID in `STEAM_APP_ID` in
   [`src/gettowork/notices.py`](../../src/gettowork/notices.py). The window's
   **Report a problem** button then opens the game's own Steam Discussions
   (until then it opens a Steam store search for the game). Steam's overlay
   can't open over the game's window on Windows, macOS or a Linux desktop, so
   that button - and the store page's Discussions - is how players report
   something the AI shouldn't have written.
6. **Code-sign the Windows build** before a public release. On Windows 11 PCs
   with **Smart App Control** on, Windows blocks unsigned programs outright -
   even ones Steam installed (Steam shows error **0x11C7**), with no "Run
   anyway" - and that includes the bundled `llama-server.exe` when the game
   starts it. The `build` workflow signs every `.exe` and `.dll` (the game's
   and the engine's) with [`packaging/sign_windows.py`](../sign_windows.py) as
   soon as the repository has these secrets (Settings > Secrets and variables >
   Actions); pull requests stay unsigned:
   - **Azure Trusted Signing** (Microsoft's signing service; about $10 a month
     after an identity check): `AZURE_TRUSTED_SIGNING_ENDPOINT`,
     `AZURE_TRUSTED_SIGNING_ACCOUNT`, `AZURE_TRUSTED_SIGNING_PROFILE`, and a
     service principal allowed to sign: `AZURE_TENANT_ID`, `AZURE_CLIENT_ID`,
     `AZURE_CLIENT_SECRET`;
   - or a code-signing certificate you can export as a `.pfx` file:
     `WINDOWS_SIGNING_PFX_BASE64` (`base64 -w0 certificate.pfx`) and
     `WINDOWS_SIGNING_PFX_PASSWORD`. (Most certificates sold today keep their
     key on a hardware token or in a cloud service instead - then use that
     service's signtool integration.)

   Check a signed build on Windows: right-click `GetToWork.exe` > Properties >
   Digital Signatures.

## Uploading a build

You need Steamworks' SDK (its `tools/ContentBuilder` folder, with SteamCMD) and
a Steam account allowed to upload builds for the app.

1. **Download the builds.** Open the repository's **Actions** tab, pick the
   latest green **build** run on `main`, and download the three game archives
   from its **Artifacts** list: `GetToWork-<version>-windows-x64.zip`,
   `GetToWork-<version>-macos-arm64.zip` and
   `GetToWork-<version>-linux-x64.tar.gz` (each artifact is the archive itself -
   there's no extra zip around it). All three carry the same llama.cpp release,
   the one pinned in [`packaging/llama_cpp_tag.txt`](../llama_cpp_tag.txt).
   (Game builds are kept for 3 days; run the workflow again by hand -
   "Run workflow" - to make fresh ones.)
2. **Unpack them into ContentBuilder's `content` folder**, each into its own
   sub-folder, keeping permissions and links. On a Mac:

   ```bash
   cd ContentBuilder
   mkdir -p content/windows content/macos content/linux
   unzip -q GetToWork-*-windows-x64.zip -d content/windows
   ditto -x -k GetToWork-*-macos-arm64.zip content/macos
   tar xzf GetToWork-*-linux-x64.tar.gz -C content/linux
   ```

   (On Linux use `unzip -q GetToWork-*-macos-arm64.zip -d content/macos` for the
   Mac build; Info-ZIP's unzip restores the app's symbolic links too.) You should
   now have `content/windows/GetToWork/GetToWork.exe`,
   `content/macos/GetToWork/Get To Work.app` and
   `content/linux/GetToWork/GetToWork`.
3. **Copy the `.vdf` files** from this folder into `ContentBuilder/scripts/` and
   fill in the placeholders.
4. **Upload from macOS or Linux**, not Windows: SteamCMD on Windows can't record
   the "executable" permission of the Linux and macOS programs (including the
   bundled `llama-server`), or the symbolic links inside the Mac app.

   The SDK's own SteamCMD sits in ContentBuilder (it isn't on your `PATH`):
   `builder_osx/steamcmd.sh` on a Mac, `builder_linux/steamcmd.sh` on Linux.

   ```bash
   cd ContentBuilder
   ./builder_osx/steamcmd.sh +login <builder account> +run_app_build "$PWD/scripts/app_build_all.vdf" +quit
   # on Linux:
   ./builder_linux/steamcmd.sh +login <builder account> +run_app_build "$PWD/scripts/app_build_all.vdf" +quit
   ```

   (A separately installed `steamcmd` - Homebrew, your Linux package manager
   or Valve's standalone download - works the same way.)

5. **Set the build live** on the Builds page: first on a password-protected
   beta branch to try it through Steam (on a Deck too), then on the default
   branch. (Or put a branch name in `SetLive` in the app build file.)

Players then just press **Play**: the game opens its own window and walks them
through checking their computer, picking and downloading an AI model, and the
optional Jev setup - then straight into the game. The engine ships inside the
build, so only the model is ever downloaded.

## Steam Deck

- **Native Linux build - no Proton needed.** Steam Deck runs the
  `Linux + SteamOS` launch option. Don't force a Proton version for the game.
- **Full screen by itself.** In Game Mode (and Big Picture) the window opens
  full screen with a bigger font; F11 toggles full screen elsewhere.
- **Typing.** Plans are typed, so players need the on-screen keyboard:
  **STEAM + X** opens it at any time, and the game's **Keyboard** button (shown on a Deck) asks
  Steam to show it. On a Deck the question and input bar are at the top of the
  window, above the keyboard (check this on real hardware before each release:
  the keyboard is Steam's own overlay). Menus also have big buttons for the
  touchscreen and trackpads.
- **Controller layout.** In Steamworks > Steam Input, set the default
  configuration to a template that moves the mouse with the trackpad (for
  example "Web Browser"), so the buttons and the transcript can be used without
  touching the screen.
- **Graphics.** The bundled Vulkan engine uses the Deck's AMD graphics through
  SteamOS's Vulkan driver; the game falls back to its CPU engine if that fails.
- **Requirements.** The Linux build needs glibc 2.35 or newer: SteamOS 3.5 or
  later (any up-to-date Deck), Ubuntu 22.04 or later.
- **Linux runtime.** Under Installation > Linux Runtime, select
  **Steam Linux Runtime 4.0** (steamrt4, Debian 13 based - the newest) if
  Steamworks offers it for the app. The game doesn't depend on it: Steam's
  older runtimes - the default 1.0 (a Debian 10 "soldier" container) and 3.0
  (sniper, Debian 11) - have only OpenSSL 1.1 and hide the computer's own
  libraries, so the Linux engine carries its own OpenSSL 3 (`fetch_engine.py`
  copies it in, and the `build` workflow starts the engine in a container
  without OpenSSL 3 to prove it works). Whichever runtime you pick, test on a
  Deck after changing it. Players can't fix a missing library themselves
  (SteamOS is read-only, and the runtime never sees their system's
  libraries): the game asks them to verify the game's files instead.
- **Storage.** Models are saved in the game's settings folder
  (`~/.config/gettowork` on Linux), not in the Steam install folder, so
  verifying or updating the game never deletes a downloaded model. A player
  who wants them elsewhere (a bigger drive, a microSD card) can add
  `--models-dir <folder>` to the game's launch options once: the game
  remembers it. This works on Windows too, where launch options can't set
  environment variables.

## Later: a Gridfall bundle

The owner may later offer their other game, Gridfall, together with Get To Work
as a paid DLC or bundle. That needs nothing in these builds: a bundle groups
separate Steam apps, and a DLC is its own app ID with its own depots, so this
app's depots and launch options stay as they are.
