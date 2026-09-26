# Third-party notices

Get To Work's own code is licensed under the [MIT License](LICENSE). It builds
on, talks to, or downloads (at your request) the third-party software and
services below, and each keeps its own license and terms.

- **The source repository** contains only Get To Work's own code (plus the
  window icon it draws itself). No third-party programs, engine binaries or
  model weights are committed to it; pip installs the Python packages from
  PyPI.
- **The game builds** (the Steam build and the double-click test builds from
  GitHub Actions) bundle the software listed under
  [Inside the game builds](#inside-the-game-builds): Python, its libraries,
  Tcl/Tk, PyInstaller's start-up program and the llama.cpp engine. Every
  build ships their license texts in **`THIRD_PARTY_LICENSES.txt`** (next to
  the game on Windows and Linux; in `Get To Work.app/Contents/Resources` on
  macOS), made by `packaging/collect_licenses.py` from the exact packages in
  that build.
- **Model weights** are never bundled: you download them from Hugging Face,
  when you confirm, under each model's own license.

## Python packages

Installed automatically with the game (`pip install`), and bundled in the game
builds:

| Package | License | Copyright / author | Link |
|---------|---------|--------------------|------|
| rich | MIT | Will McGugan and contributors | https://github.com/Textualize/rich |
| psutil | BSD-3-Clause | Giampaolo Rodola and contributors | https://github.com/giampaolo/psutil |
| huggingface_hub | Apache-2.0 | Hugging Face, Inc. | https://github.com/huggingface/huggingface_hub |

These packages have dependencies of their own (for example tqdm, filelock,
fsspec, PyYAML, packaging, Pygments, markdown-it-py and the Hugging Face HTTP
and Xet download libraries), which pip installs with their own licenses. The
game builds bundle whichever versions pip chose for that build, and
`THIRD_PARTY_LICENSES.txt` lists each one with its license text.

Optional, only if you install it yourself (`pip install llama-cpp-python`, or
the `llamacpp` extra) - never part of the game builds:

| Package | License | Copyright / author | Link |
|---------|---------|--------------------|------|
| llama-cpp-python | MIT | Andrei Betlen and contributors | https://github.com/abetlen/llama-cpp-python |

For development only (the `dev` extra):

| Package | License | Link |
|---------|---------|------|
| pytest | MIT | https://github.com/pytest-dev/pytest |

## Inside the game builds

Besides the Python packages above, every game build contains:

| Component | License | Notes | Source |
|-----------|---------|-------|--------|
| **Python** (the interpreter and its standard library) | PSF-2.0, the Python Software Foundation License | Bundled by PyInstaller. | https://www.python.org |
| **Tcl/Tk** (the toolkit behind the game's window, used through Python's `tkinter`) | Tcl/Tk license (BSD-style), copyright the Regents of the University of California, Sun Microsystems, Inc. and others | Free to use, copy and distribute for any purpose as long as the copyright notices are kept. | https://www.tcl-lang.org/software/tcltk/license.html |
| **PyInstaller** bootloader (the small start-up program inside `GetToWork` and `gettowork-cli`) | GPL-2.0-or-later with the PyInstaller bootloader exception | The exception explicitly allows shipping the bootloader inside programs under any license, so distributing the built game - free on Steam or anywhere else - is fine, and Get To Work's own code stays MIT. | https://github.com/pyinstaller/pyinstaller |
| **llama.cpp** (`llama-server` and its libraries), official prebuilt release builds | MIT, copyright the ggml authors | **Bundled** in the game builds: Vulkan and CPU builds on Windows and Linux, the Metal build on macOS (Apple Silicon). Each build's license texts - llama.cpp's own and those of the code compiled into it: cpp-httplib (MIT), nlohmann/json (MIT), BoringSSL (Apache-2.0) in the Windows and macOS builds, and LLVM OpenMP (Apache-2.0 with LLVM exceptions, `libomp.dll`) on Windows - are taken from the official archive and from the texts llama.cpp embeds in its programs (or, when missing, fetched from the same release's source) into `engine/<build>/licenses/`, and included in `THIRD_PARTY_LICENSES.txt`. | https://github.com/ggml-org/llama.cpp/releases |
| **OpenSSL 3** (`libssl.so.3`, `libcrypto.so.3`), Linux engine builds only | Apache-2.0, copyright the OpenSSL Project Authors | Copied from the build machine (Ubuntu 22.04) next to the bundled llama.cpp engine, which links it: Steam's Linux runtime has no OpenSSL 3. Its license text ships in `engine/<build>/licenses/LICENSE-OpenSSL` and `THIRD_PARTY_LICENSES.txt`. | https://www.openssl.org/source/license.html |
| **Microsoft Visual C++ runtime** (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`), Windows builds only | Microsoft Visual C++ Redistributable terms | Copied next to the bundled llama.cpp engine so it starts on a PC without the Visual C++ Redistributable; Microsoft allows applications to redistribute these files. | https://visualstudio.microsoft.com/license-terms/ |
| truststore, certifi (when present on the build machine) | MIT (truststore), MPL-2.0 (certifi) | Used for HTTPS certificate checks (`tls.py`). | https://github.com/sethmlarson/truststore, https://github.com/certifi/python-certifi |
| **Native libraries** PyInstaller copies in with Python and Tk: OpenSSL (Apache-2.0), libffi, Expat, Brotli (MIT), zlib, bzip2, XZ/liblzma (0BSD), mpdecimal, SQLite (public domain), and on Linux the system libraries Tk and Python link against - X11/Xft/Xrender/Xss (MIT/X11), Fontconfig, FreeType (FTL), libpng, ncurses/libtinfo, util-linux libuuid, libbsd/libmd, the GCC runtime (with the GCC Runtime Library Exception) | each its own permissive license | `collect_licenses.py --app-dir` lists every one the build really contains, with the package's own copyright file where the build machine has it, and fails the build for a library with no license entry. GNU Readline (GPL-3.0) is deliberately left out of the builds: the game doesn't need it. | see `THIRD_PARTY_LICENSES.txt` |

The builds are made by the repository's GitHub Actions workflow
(`.github/workflows/build.yml`): `packaging/fetch_engine.py` downloads the
official llama.cpp archives and checks their size and SHA-256 digest (the one
pinned in `packaging/llama_cpp_tag.txt`) before unpacking them, and the game's
own code is packed by PyInstaller (`packaging/gettowork.spec`). The engine's
code is used unmodified, as published by the llama.cpp project (on Windows,
once code signing is set up, the build adds its Authenticode signature to the
programs, which changes no code).

## Downloaded at runtime, at your request

| Component | License | Source |
|-----------|---------|--------|
| **Model weights** (GGUF files) | Each model's own license, shown before download. Only Apache-2.0 or MIT models are suggested by default; other licenses appear with `--all-licenses`, or when you name a model yourself (`custom`, `--model`) - always with a warning to read its license first. | https://huggingface.co (the model page is shown before download) |
| **llama.cpp** official prebuilt release archives - **only for a copy of the game run from source** (the game builds have the engine built in and never download it) | MIT, copyright the ggml authors | https://github.com/ggml-org/llama.cpp/releases |

Notes:

- A copy run from source downloads the official llama.cpp archive from the
  project's GitHub releases only after you confirm, verifies its size and
  SHA-256 digest, and unpacks it into your data folder. The engine is not
  bundled in the source repository.
- Some prebuilt archives include third-party runtime libraries with their own
  licenses. In particular, the NVIDIA CUDA builds (which only a copy run from
  source downloads; the game builds don't include them) come with NVIDIA's
  CUDA runtime libraries, which are distributed under NVIDIA's license terms,
  not MIT. The unpacked folder contains the files as published by the
  llama.cpp project.
- Model weights are never part of this project or the game builds. You
  download them from Hugging Face, where they are shared by their authors
  under the license shown on each model's page. You are responsible for
  complying with it.

## Separate software the game can use

| Software | License | Link |
|----------|---------|------|
| Ollama (a separate application you install yourself; optional, not bundled) | MIT | https://github.com/ollama/ollama, downloads at https://ollama.com/download |

## Online services

| Service | Used for | Terms |
|---------|----------|-------|
| Hugging Face Hub | Searching for models and downloading model files | Hugging Face's terms of service, https://huggingface.co/terms-of-service, and each model's license |
| GitHub | Hosting the (private) source repository and its test builds; for copies run from source, listing llama.cpp releases (`api.github.com`) and downloading the engine | GitHub's terms of service, https://docs.github.com/en/site-policy/github-terms/github-terms-of-service |
| TypeSafe AI, Jev API (optional, paid) | Judging each round when you enable Jev | TypeSafe AI's own terms and pricing, https://typesafe.ai |
| Steam (Valve) | Distributing the free game, when released there | The Steam Subscriber Agreement, https://store.steampowered.com/subscriber_agreement/ |

## Referenced, not bundled

- **TypeSafe SDK** (`typesafe-sdk`, MIT License, by TypeSafe AI). Get To Work's
  `jev.py` is an independent client written with the Python standard library.
  It follows the public wire format (endpoints, JSON shapes, headers and error
  conventions) used by TypeSafe's official MIT-licensed SDK, so learners can
  see every byte that is sent. No SDK code is bundled, and the SDK isn't
  required.
- **llama.cpp release workflow.** The names of the prebuilt archives were taken
  from the llama.cpp project's public release workflow, so the installer can
  find the right file.
- **Published hardware specifications.** The rough GPU memory-bandwidth table in
  `perf.py` lists figures from manufacturers' public specification sheets.

## Not affiliated

Get To Work is an independent project. It is **not affiliated with, endorsed
by or sponsored by** TypeSafe AI, Hugging Face, ggml-org / the llama.cpp
project, Ollama, Valve / Steam, the Python Software Foundation, the Tcl/Tk
or PyInstaller projects, NVIDIA, AMD, Intel, Apple, Microsoft, GitHub, or any
model author or publisher. All product names, logos and trademarks belong to
their respective owners and are used here only to identify their products and
services.

The software is provided "AS IS", without warranty of any kind. See
[LICENSE](LICENSE) and the disclaimer in the [README](README.md#license-and-disclaimer).
