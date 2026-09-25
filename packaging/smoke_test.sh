#!/usr/bin/env bash
# Smoke-tests a built copy of the game without any network, model or GPU.
#
#   packaging/smoke_test.sh <command...>
#   e.g. packaging/smoke_test.sh gettowork
#        packaging/smoke_test.sh ./dist/gettowork.exe
#
# Runs the info commands, then plays a whole game with the scripted offline
# "pretend" model and checks that it is won with no Python traceback.
set -euo pipefail

if [ "$#" -eq 0 ]; then
  echo "usage: $0 <gettowork command...>" >&2
  exit 2
fi

GETTOWORK_HOME="$(mktemp -d)"
export GETTOWORK_HOME
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
trap 'rm -rf "$GETTOWORK_HOME"' EXIT

echo "::group::--version"
"$@" --version
echo "::endgroup::"

echo "::group::--specs --offline"
"$@" --specs --offline
echo "::endgroup::"

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
