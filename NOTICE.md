# Third-party notices

Get To Work is licensed under the [MIT License](LICENSE). It builds on, talks
to, or downloads (at your request) the third-party software and services
below. **None of them is bundled in this repository.** Python packages are
installed by pip from PyPI, and everything else is downloaded by you, when you
confirm, from its official source. Each keeps its own license and terms.

## Python packages

Installed automatically with the game (`pip install`):

| Package | License | Copyright / author | Link |
|---------|---------|--------------------|------|
| rich | MIT | Will McGugan and contributors | https://github.com/Textualize/rich |
| psutil | BSD-3-Clause | Giampaolo Rodola and contributors | https://github.com/giampaolo/psutil |
| huggingface_hub | Apache-2.0 | Hugging Face, Inc. | https://github.com/huggingface/huggingface_hub |

These packages have dependencies of their own (for example tqdm, filelock,
fsspec, PyYAML, packaging and the Hugging Face HTTP and Xet download
libraries), which pip installs with their own licenses.

Optional, only if you install it yourself (`pip install llama-cpp-python`, or
the `llamacpp` extra):

| Package | License | Copyright / author | Link |
|---------|---------|--------------------|------|
| llama-cpp-python | MIT | Andrei Betlen and contributors | https://github.com/abetlen/llama-cpp-python |

For development only (the `dev` extra):

| Package | License | Link |
|---------|---------|------|
| pytest | MIT | https://github.com/pytest-dev/pytest |

## Downloaded at runtime, at your request

| Component | License | Source |
|-----------|---------|--------|
| **llama.cpp** (`llama-server` and its libraries), official prebuilt release archives | MIT, copyright the ggml authors | https://github.com/ggml-org/llama.cpp/releases |
| **Model weights** (GGUF files) | Each model's own license, shown before download. Only Apache-2.0 or MIT models are suggested by default; other licenses appear with `--all-licenses`, or when you name a model yourself (`custom`, `--model`) - always with a warning to read its license first. | https://huggingface.co (the model page is shown before download) |

Notes:

- The llama.cpp engine is **not bundled**. The game downloads the official
  archive from the llama.cpp project's GitHub releases only after you confirm,
  verifies its size and SHA-256 digest, and unpacks it into your data folder.
- Some prebuilt archives include third-party runtime libraries with their own
  licenses. In particular, the NVIDIA CUDA builds come with NVIDIA's CUDA
  runtime libraries, which are distributed under NVIDIA's license terms, not
  MIT. The unpacked folder contains the files as published by the llama.cpp
  project.
- Model weights are never part of this project. You download them from Hugging
  Face, where they are shared by their authors under the license shown on each
  model's page. You are responsible for complying with it.

## Separate software the game can use

| Software | License | Link |
|----------|---------|------|
| Ollama (a separate application you install yourself; optional) | MIT | https://github.com/ollama/ollama, downloads at https://ollama.com/download |

## Online services

| Service | Used for | Terms |
|---------|----------|-------|
| Hugging Face Hub | Searching for models and downloading model files | Hugging Face's terms of service, https://huggingface.co/terms-of-service, and each model's license |
| GitHub | Listing llama.cpp releases (`api.github.com`) and downloading the engine | GitHub's terms of service, https://docs.github.com/en/site-policy/github-terms/github-terms-of-service |
| TypeSafe AI, Jev API (optional, paid) | Judging each round when you enable Jev | TypeSafe AI's own terms and pricing, https://typesafe.ai |

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

Get To Work is an independent open-source project. It is **not affiliated
with, endorsed by or sponsored by** TypeSafe AI, Hugging Face, ggml-org / the
llama.cpp project, Ollama, NVIDIA, AMD, Intel, Apple, Microsoft, GitHub, or
any model author or publisher. All product names, logos and trademarks belong
to their respective owners and are used here only to identify their products
and services.

The software is provided "AS IS", without warranty of any kind. See
[LICENSE](LICENSE) and the disclaimer in the [README](README.md#license-and-disclaimer).
