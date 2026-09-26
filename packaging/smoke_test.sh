#!/usr/bin/env bash
# Smoke-tests a built copy of the game without any network, model or GPU.
#
#   packaging/smoke_test.sh [--engine-dir DIR] <command...>
#   e.g. packaging/smoke_test.sh gettowork
#        packaging/smoke_test.sh ./dist/GetToWork/gettowork-cli
#        packaging/smoke_test.sh --engine-dir dist/GetToWork/engine dist/GetToWork/gettowork-cli
#
# Runs the info commands, then plays a whole game with the scripted offline
# "pretend" model and checks that it is won with no Python traceback.
# With --engine-dir it first checks that every bundled llama.cpp engine build
# in DIR starts (`llama-server --version`), e.g. after packaging/assemble.py
# copied them into the game - and that the game itself says it uses that
# built-in engine with engine downloads off (`--specs`), so a build that lost
# its distribution.json (and would download llama.cpp while playing) fails.
#
# (The game's window has its own check: `GetToWork --gui-selftest`.)
set -euo pipefail

engine_dir=""
if [ "${1:-}" = "--engine-dir" ]; then
  engine_dir="${2:?--engine-dir needs a folder}"
  shift 2
fi
if [ "$#" -eq 0 ]; then
  echo "usage: $0 [--engine-dir DIR] <gettowork command...>" >&2
  exit 2
fi

GETTOWORK_HOME="$(mktemp -d)"
export GETTOWORK_HOME
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
trap 'rm -rf "$GETTOWORK_HOME"' EXIT

engine_tag=""
if [ -n "$engine_dir" ]; then
  echo "::group::bundled engine builds"
  found=0
  for marker in "$engine_dir"/*/install.json; do
    [ -f "$marker" ] || continue
    build="$(dirname "$marker")"
    engine_tag="$(sed -n 's/.*"tag":[[:space:]]*"\([^"]*\)".*/\1/p' "$marker" | head -n 1)"
    # Pull  "exe": "path/to/llama-server"  out of install.json (no JSON tool needed).
    exe="$(sed -n 's/.*"exe":[[:space:]]*"\([^"]*\)".*/\1/p' "$marker" | head -n 1)"
    if [ -z "$exe" ] || [ ! -f "$build/$exe" ]; then
      echo "::error::$marker names an engine program that isn't there: '$exe'"
      exit 1
    fi
    echo "$(basename "$build"): $exe --version"
    # The engine finds its libraries next to itself (RPATH / DLL search); LD_LIBRARY_PATH is belt and braces.
    if ! (cd "$build/$(dirname "$exe")" && LD_LIBRARY_PATH="$PWD${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "./$(basename "$exe")" --version); then
      echo "::error::the bundled engine build $(basename "$build") didn't start"
      exit 1
    fi
    found=$((found + 1))
  done
  echo "::endgroup::"
  if [ "$found" -eq 0 ]; then
    echo "::error::no engine builds (folders with install.json) in $engine_dir"
    exit 1
  fi
fi

echo "::group::--version"
"$@" --version
echo "::endgroup::"

echo "::group::--specs --offline"
# Wide output: the engine line checked below must not be wrapped.
specs="$(COLUMNS=200 "$@" --specs --offline 2>&1)"
printf '%s\n' "$specs"
echo "::endgroup::"
if [ -n "$engine_dir" ]; then
  if ! grep -q "built into the game: llama.cpp $engine_tag " <<<"$specs" || ! grep -q "engine downloads off" <<<"$specs"; then
    echo "::error::the game doesn't say it uses its built-in llama.cpp $engine_tag with engine downloads off (is distribution.json in place?)"
    exit 1
  fi
fi

echo "::group::--list-models --offline"
"$@" --list-models --offline
echo "::endgroup::"

echo "::group::mock game"
# Enter at the first two prompts, five plans, then "no" to every end-of-game question.
input=$'\n\nI ride a llama to the bus stop while singing sea shanties\nI bribe the geese with artisanal breadcrumbs and a heartfelt speech\nI politely ask gravity to clock back in by offering it a coffee\nI glue my shoes back together and sing them a reconciliation ballad\nI summon a tiny parade of accountants to carry me through the lobby\nn\nn\nn\nn\n'
status=0
out="$(printf '%s' "$input" | "$@" --mock --no-jev 2>&1)" || status=$?
printf '%s\n' "$out"
echo "::endgroup::"

if [ "$status" -ne 0 ]; then
  echo "::error::the mock game exited with status $status"
  exit 1
fi
if grep -qi "traceback" <<<"$out"; then
  echo "::error::the mock game printed a Python traceback"
  exit 1
fi
if ! grep -q "YOU GOT TO WORK" <<<"$out"; then
  echo "::error::the mock game did not reach the victory screen"
  exit 1
fi
echo "Smoke test passed: info commands work and the mock game was won."
