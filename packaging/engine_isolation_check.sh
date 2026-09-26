#!/usr/bin/env bash
# Checks that the bundled Linux engine starts on a system without OpenSSL 3 - as under Steam.
#
#   packaging/engine_isolation_check.sh <engine-dir> [image]    # runs the check in a container (docker)
#   packaging/engine_isolation_check.sh --inside <engine-dir>   # the check itself (what runs in the container)
#
# Steam starts native Linux games inside its Linux runtime: by default "Steam
# Linux Runtime 1.0" (a Debian 10 "soldier" container) or 3.0 "sniper"
# (Debian 11). Both have only OpenSSL 1.1, and the computer's own /usr isn't
# visible inside them. The official llama.cpp Linux builds link OpenSSL 3, so
# packaging/fetch_engine.py copies libssl.so.3 and libcrypto.so.3 into each
# engine build; this proves the builds really start without the system's copy.
#
# The first form runs the second one in a container of `image` (default
# debian:bookworm-slim: new enough for the engine's glibc 2.35+, and without
# OpenSSL 3) after installing what every Steam runtime has (the OpenMP and
# Vulkan loader libraries). The second form fails if the system it runs on has
# OpenSSL 3 (the check would prove nothing there), then runs
# `llama-server --version` for every build in <engine-dir> the way the game
# starts it (LD_LIBRARY_PATH = the build's folder). It needs nothing but bash.
set -euo pipefail

if [ "${1:-}" = "--inside" ]; then
  engine_dir="${2:?--inside needs the engine folder}"
  # The folders a system keeps its libraries in (SYSTEM_LIB_DIRS replaces the list - for the tests).
  lib_dirs="${SYSTEM_LIB_DIRS:-/lib /lib64 /usr/lib /usr/lib64 /usr/local/lib /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu /lib/aarch64-linux-gnu /usr/lib/aarch64-linux-gnu}"
  for dir in $lib_dirs; do
    if [ -e "$dir/libssl.so.3" ] || [ -e "$dir/libcrypto.so.3" ]; then
      echo "::error::this system has OpenSSL 3 (in $dir), so the check would prove nothing - use an image without it"
      exit 1
    fi
  done
  found=0
  for marker in "$engine_dir"/*/install.json; do
    [ -f "$marker" ] || continue
    build="${marker%/install.json}"
    exe=""
    if [[ "$(<"$marker")" =~ \"exe\":[[:space:]]*\"([^\"]*)\" ]]; then
      exe="${BASH_REMATCH[1]}"
    fi
    if [ -z "$exe" ] || [ ! -f "$build/$exe" ]; then
      echo "::error::$marker names an engine program that isn't there: '$exe'"
      exit 1
    fi
    folder="$build/$exe"
    folder="${folder%/*}"
    echo "${build##*/}: ${exe##*/} --version (no OpenSSL 3 on this system)"
    if ! (cd "$folder" && LD_LIBRARY_PATH="$PWD" "./${exe##*/}" --version); then
      echo "::error::the bundled engine build ${build##*/} doesn't start without the system's OpenSSL 3 (as under Steam's Linux runtime) - see LINUX_OPENSSL_LIBS in packaging/fetch_engine.py"
      exit 1
    fi
    found=$((found + 1))
  done
  if [ "$found" -eq 0 ]; then
    echo "::error::no engine builds (folders with install.json) in $engine_dir"
    exit 1
  fi
  echo "Every bundled engine build starts without the system's OpenSSL 3."
  exit 0
fi

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "usage: $0 <engine-dir> [image]   or   $0 --inside <engine-dir>" >&2
  exit 2
fi
engine_abs="$(cd "$1" && pwd)"
image="${2:-debian:bookworm-slim}"
script_abs="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
echo "Starting every engine build in $engine_abs inside $image (no OpenSSL 3 there)..."
docker run --rm -v "$engine_abs:/engine:ro" -v "$script_abs:/engine-isolation-check.sh:ro" "$image" bash -c '
  set -e
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -qq >/dev/null
  apt-get install -y -qq --no-install-recommends libgomp1 libvulkan1 >/dev/null
  bash /engine-isolation-check.sh --inside /engine'
