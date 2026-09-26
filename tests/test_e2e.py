"""End-to-end tests: the real ``cli.main()`` driven like a player would drive it.

Unlike the per-module tests, nothing inside the game is swapped out here:
setup, discovery + the fit engine, Jev onboarding, the game loop, the review
and the export all run for real. Only the outside world is faked:

* hardware detection returns a fixed machine (no benchmark),
* the Hugging Face Hub is a fake ``HfApi`` object (no network),
* Jev is a fake *transport* that checks every request against the official
  wire format (the same shape the TypeSafe SDK sends) and answers realistically,
* the managed llama.cpp engine is a tiny Python "llama-server" started as a
  real subprocess, so process start-up, health polling, chat, the speed test
  and shutdown are all exercised (POSIX only: the launcher is a shell script).
"""

from __future__ import annotations

import functools
import io
import json
import os
import sys
import textwrap
from pathlib import Path

import psutil
import pytest
from rich.console import Console

from gettowork import hf_discovery, jev
from gettowork.cli import EXIT_INTERRUPTED, EXIT_OK, main
from gettowork.setup_flow import SetupServices
from gettowork.types import SystemSpecs
from gettowork.ui import UI

KEY = "tsk-e2e-0123456789abcdef-SECRET"
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
PLAN_PROMPTS = ("How do you plan to get to work?", "What do you do?")  # round 1, then every obstacle
PLANS = [
    "I ride my bicycle very fast, ringing the bell the whole way",
    "I bribe the geese with a basket of warm bread rolls and a tiny union contract",
    "I build a ramp out of cereal boxes and jump clean over the problem",
    "I recite the office safety manual so loudly that the obstacle gives up",
    "I disguise myself as the manager and stroll straight past",
    "I sing the company anthem while riding the escalator backwards",
    "I tip my hat, bow politely and dance past with great confidence",
]


# ---------------------------------------------------------------------------
# Shared fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    for var in ("GETTOWORK_MODELS_DIR", "TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    return home


def laptop() -> SystemSpecs:
    """A 16 GB laptop with no graphics card (and a measured RAM speed)."""
    return SystemSpecs(
        os_name="Linux", os_version="Test 1.0", arch="x86_64", cpu_name="Test CPU 5000",
        cpu_cores_physical=8, cpu_cores_logical=16, ram_total_gb=16.0, ram_available_gb=12.0,
        disk_free_gb=200.0, ram_bandwidth_gbs=40.0, cpu_flags=["avx", "avx2", "fma"],
    )


class Player:
    """Answers prompts by what they ask; records every prompt; never loops forever."""

    def __init__(self, answers: dict[str, list[str]], *, plans=PLANS, eof_after_plans: int | None = None) -> None:
        self.answers = {k: list(v) for k, v in answers.items()}
        self.plans = list(plans)
        self.prompts: list[str] = []
        self.eof_after_plans = eof_after_plans
        self.plans_given = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        assert len(self.prompts) < 80, f"the game kept asking questions; last: {prompt!r}"
        if any(question in prompt for question in PLAN_PROMPTS):
            if self.eof_after_plans is not None and self.plans_given >= self.eof_after_plans:
                raise EOFError  # stdin closed mid-game (Ctrl+D / a closed pipe)
            self.plans_given += 1
            return self.plans.pop(0)
        for needle, queue in self.answers.items():
            if needle in prompt and queue:
                return queue.pop(0)
        raise AssertionError(f"unexpected prompt: {prompt!r}")


def make_ui(player: Player) -> UI:
    return UI(console=Console(file=io.StringIO(), width=80), input_fn=player,
              secret_fn=lambda prompt: pytest.fail("the key should come from the environment"),
              open_url_fn=lambda url: True)


def output(ui: UI) -> str:
    return ui.console.file.getvalue()


def services(**overrides) -> SetupServices:
    return SetupServices(detect_specs=lambda **kw: laptop(), **overrides)


# ---------------------------------------------------------------------------
# A fake Jev that checks the wire format
# ---------------------------------------------------------------------------


class FakeJev:
    """A Jev transport: ``(method, url, headers, body, timeout) -> (status, headers, body)``."""

    def __init__(self, *, fail_systemone: int = 0) -> None:
        self.calls: list[tuple[str, str, dict, dict | None]] = []
        self.fail_systemone = fail_systemone  # this many /v1/systemone calls answer HTTP 500

    def __call__(self, method, url, headers, body, timeout):
        payload = json.loads(body) if body else None
        self.calls.append((method, url, dict(headers), payload))
        assert url.startswith("https://api.typesafe.ai/"), url
        assert headers["Authorization"] == f"Bearer {KEY}"
        assert headers["Accept"] == "application/json"
        if url.endswith("/v1/models"):
            assert method == "GET" and body is None and "Content-Type" not in headers
            return 200, {"content-type": "application/json"}, json.dumps({"models": [
                {"name": "jev-latest", "description": "General-purpose system one model.",
                 "release_date": "2026-09-15"}]}).encode()
        assert method == "POST" and url.endswith("/v1/systemone")
        assert headers["Content-Type"] == "application/json"
        self._check_request(payload)
        if self.fail_systemone:
            self.fail_systemone -= 1
            return 500, {}, b'{"detail": "Internal Server Error"}'
        return 200, {"content-type": "application/json"}, json.dumps(self._answer(payload)).encode()

    @staticmethod
    def _check_request(payload: dict) -> None:
        """The SystemOneRequest schema: state + model + named, typed questions (no extra fields)."""
        assert set(payload) == {"state", "model", "questions"}
        assert payload["model"] == "jev-latest"
        assert isinstance(payload["state"], (str, dict, list))
        assert payload["questions"]
        for name, q in payload["questions"].items():
            assert isinstance(name, str) and q["type"] in ("noul", "choice", "score")
            assert set(q) <= {"type", "instructions", "criteria"}
            if q["type"] == "noul":
                assert set(q.get("criteria") or {}) <= {"true", "false"}
            elif q["type"] == "choice":
                assert isinstance(q["criteria"], dict) and 0 < len(q["criteria"]) <= 255
            else:
                assert isinstance(q["criteria"], list) and q["criteria"]
        # The player's text is data inside the state, never part of the questions.
        plan = payload["state"]["player_plan"]
        assert plan and plan not in json.dumps(payload["questions"])

    @staticmethod
    def _answer(payload: dict) -> dict:
        answers = {}
        for name, q in payload["questions"].items():
            if q["type"] == "noul":
                answers[name] = {"type": "noul", "noul": 0.91}
            elif q["type"] == "choice":
                labels = list(q["criteria"])
                probs = {label: (0.7 if i == 1 else 0.3 / (len(labels) - 1)) for i, label in enumerate(labels)}
                answers[name] = {"type": "choice", "choice": labels[1], "confidence": 0.8, "probabilities": probs}
            else:
                levels = {str(i): (0.6 if i == 3 else 0.4 / (len(q["criteria"]) - 1)) for i in range(len(q["criteria"]))}
                answers[name] = {"type": "score", "score": round(sum(int(k) * v for k, v in levels.items()), 3),
                                 "confidence": 0.6, "legend": {str(i): c for i, c in enumerate(q["criteria"])},
                                 "probabilities": levels}
        return {"model": "jev-1", "answers": answers, "usage": {"input_tokens": 900, "output_tokens": 9}}


# ---------------------------------------------------------------------------
# Pretend model + Jev: a whole game, review and export
# ---------------------------------------------------------------------------


def test_mock_game_with_jev_over_the_wire_win_review_and_export(tmp_path, monkeypatch):
    fake = FakeJev()
    monkeypatch.setattr(jev, "urllib_transport", fake)
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    export_dir = tmp_path / "exports"
    player = Player({
        "Use it?": ["use"],  # the key found in TYPESAFE_API_KEY
        "Jev request & response": ["y"],
        "reasoning (chain-of-thought)": ["y"],
        "Save a transcript": ["y"],
    })
    ui = make_ui(player)

    code = main(["--mock", "--export-dir", str(export_dir)], ui=ui, services=services())

    assert code == EXIT_OK
    text = output(ui)
    assert "YOU GOT TO WORK!" in text
    assert "Referee: Jev (jev-latest)" in text
    assert text.count("Jev's verdict") == 5
    # One Jev lesson per round, in order - never all three at once.
    assert text.index("Learn: Noul") < text.index("Learn: Choice") < text.index("Learn: Score")
    assert "request to Jev" in text and "Bearer ****" in text  # the review shows the redacted exchange
    assert "The pretend model's scripted example reasoning" in text
    # One key check + one judgment per round; the key never leaks anywhere.
    assert [c[1].rsplit("/", 1)[-1] for c in fake.calls] == ["models"] + ["systemone"] * 5
    files = sorted(p.name for p in export_dir.iterdir())
    assert files == ["gettowork-transcript-1.json", "gettowork-transcript-1.md"]
    exported = "".join((export_dir / f).read_text(encoding="utf-8") for f in files)
    for haystack in (text, exported, "\n".join(player.prompts)):
        assert KEY not in haystack and KEY[-8:] not in haystack
    data = json.loads((export_dir / "gettowork-transcript-1.json").read_text(encoding="utf-8"))
    assert data["won"] is True and len(data["rounds"]) == 5
    assert all(r["judge"] == "jev" for r in data["rounds"])


def test_jev_outage_mid_game_falls_back_to_the_local_referee(monkeypatch):
    fake = FakeJev(fail_systemone=3)  # the first round's call + its 2 retries all fail
    monkeypatch.setattr(jev, "urllib_transport", fake)
    monkeypatch.setattr(jev.time, "sleep", lambda s: None)  # skip the retry back-off waits
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    player = Player({
        "Use it?": ["use"],
        "Keep asking Jev": ["y"],  # ...and Jev is back for round 2
        "Jev request & response": ["y"],
        "reasoning (chain-of-thought)": ["n"],
        "Save a transcript": ["n"],
    }, plans=PLANS[:2] + ["quit"])
    ui = make_ui(player)

    code = main(["--mock"], ui=ui, services=services())

    assert code == EXIT_OK
    text = " ".join(output(ui).split())
    assert "Jev couldn't referee this round" in text
    # --mock: the stand-in referee is honestly called the pretend model, not "your local model".
    assert "the pretend model (a simple scripted rule) will referee this round instead" in text
    assert "Referee's verdict (the pretend model)" in text and "Jev's verdict" in text
    assert "This Jev call failed" in text  # the review still shows what was sent
    assert KEY not in text


# ---------------------------------------------------------------------------
# Live discovery (fake Hub) -> fit engine -> menu
# ---------------------------------------------------------------------------


class FakeHub:
    """Just enough of ``huggingface_hub.HfApi``: search results + file listings."""

    MODELS = {
        # repo: (params, license, downloads, tags)
        "unsloth/Qwen3-4B-GGUF": (4.0, "apache-2.0", 500_000, ["conversational"]),
        "unsloth/Qwen3-1.7B-GGUF": (2.0, "apache-2.0", 380_000, ["conversational"]),
        "unsloth/Qwen3-30B-A3B-GGUF": (30.5, "apache-2.0", 600_000, ["conversational"]),
        "bartowski/Llama-3.2-3B-Instruct-GGUF": (3.2, "llama3.2", 900_000, ["conversational"]),
        "unsloth/Qwen2.5-Coder-7B-Instruct-GGUF": (7.6, "apache-2.0", 300_000, ["conversational", "code"]),
    }

    def __init__(self) -> None:
        self.searches = 0

    def list_models(self, **kwargs):
        self.searches += 1
        author = kwargs.get("author")
        for repo, (params, lic, downloads, tags) in self.MODELS.items():
            if author in (None, repo.split("/")[0]):
                base = repo.split("/")[1].replace("-GGUF", "")
                yield {"id": repo, "downloads": downloads, "likes": 10, "pipeline_tag": "text-generation",
                       "tags": ["gguf", "text-generation", f"license:{lic}", *tags], "gated": False,
                       "gguf": {"total": int(params * 1e9), "architecture": "qwen3", "context_length": 40960},
                       "cardData": {"license": lic, "base_model": f"org/{base}"}}

    def list_repo_tree(self, repo_id, *, recursive=False, expand=False, **kwargs):
        params = self.MODELS[repo_id][0]
        stem = repo_id.split("/")[1].replace("-GGUF", "")
        for quant, bits in (("Q3_K_M", 3.9), ("Q4_K_M", 4.8), ("Q8_0", 8.5)):
            yield {"path": f"{stem}-{quant}.gguf", "size": int(params * 1e9 * bits / 8)}


def test_list_models_with_live_discovery_shows_sensible_picks(tmp_path):
    hub = FakeHub()
    discover = functools.partial(hf_discovery.discover_models, api=hub, cache_path=tmp_path / "cache.json")
    ui = make_ui(Player({}))

    code = main(["--list-models"], ui=ui, services=services(discover_models=discover))

    assert code == EXIT_OK and hub.searches > 0
    text = output(ui)
    assert "Found 3 models on Hugging Face" in text  # Llama (license) and the coder model are left out
    assert "Left out 1 model whose license isn't Apache-2.0 or MIT" in text
    assert "Recommended" in text and "Qwen3 4B" in text
    assert "Llama" not in text and "Coder" not in text
    assert all(len(line) <= 80 for line in text.splitlines())  # nothing spills past 80 columns


# ---------------------------------------------------------------------------
# The managed engine, as a real subprocess
# ---------------------------------------------------------------------------

FAKE_SERVER = textwrap.dedent('''
    """A tiny stand-in for llama-server: same flags, same HTTP API, scripted words."""
    import json, os, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from gettowork.backends.mock import MockBackend

    args = sys.argv[1:]
    opt = lambda flag: args[args.index(flag) + 1]
    model, port = opt("-m"), int(opt("--port"))
    assert opt("--host") == "127.0.0.1" and os.path.isfile(model)
    with open(os.environ["FAKE_PID_FILE"], "w") as fh:
        fh.write(str(os.getpid()))
    started, mock = time.monotonic(), MockBackend(seed=3)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, status, payload):
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if time.monotonic() - started < 0.3:
                return self.send(503, {"error": {"code": 503, "message": "Loading model"}})
            self.send(200, {"status": "ok"})

        def do_POST(self):
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            system = " ".join(m["content"] for m in req["messages"] if m["role"] == "system")
            if "TASK:" in system:
                r = mock.chat(req["messages"], json_mode="response_format" in req)
                message = {"role": "assistant", "content": r.text, "reasoning_content": r.reasoning}
            else:
                message = {"role": "assistant", "content": "1, 2, 3, 4, 5"}
            self.send(200, {"choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                            "usage": {"completion_tokens": 40}, "timings": {"predicted_per_second": 18.5}})

    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
''')


@pytest.fixture
def managed_engine(tmp_path, isolated_home, monkeypatch):
    """A pretend llama.cpp install (the same layout runtime_install.py creates) + a GGUF file."""
    if os.name == "nt":
        pytest.skip("the fake llama-server launcher is a POSIX shell script")
    folder = isolated_home / "runtime" / "llama.cpp" / "b1-cpu"
    folder.mkdir(parents=True)
    script = tmp_path / "fake_llama_server.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    launcher = folder / "llama-server"
    launcher.write_text(
        f'#!/bin/sh\nPYTHONPATH="{SRC_DIR}${{PYTHONPATH:+:$PYTHONPATH}}" exec "{sys.executable}" "{script}" "$@"\n',
        encoding="utf-8")
    launcher.chmod(0o755)
    (folder / "install.json").write_text(json.dumps(
        {"tag": "b1", "variant": "cpu", "label": "CPU", "assets": [], "exe": "llama-server"}), encoding="utf-8")
    gguf = tmp_path / "tiny-model-Q4_K_M.gguf"
    gguf.write_bytes(b"GGUF" + b"\0" * 1024)
    pid_file = tmp_path / "server.pid"
    monkeypatch.setenv("FAKE_PID_FILE", str(pid_file))
    return gguf, pid_file


def server_gone(pid_file: Path) -> bool:
    pid = int(pid_file.read_text())
    try:
        return psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return True


def test_managed_engine_starts_plays_and_is_stopped_on_exit(managed_engine):
    gguf, pid_file = managed_engine
    player = Player({"Shall I go ahead?": ["y"], "reasoning (chain-of-thought)": ["n"], "Save a transcript": ["n"]})
    ui = make_ui(player)

    code = main(["--gguf", str(gguf), "--no-jev", "--target", "2"], ui=ui, services=services())

    assert code == EXIT_OK
    text = " ".join(output(ui).split())
    assert "Already installed" in text  # the confirmation screen noticed the engine
    assert "Your model is awake and ready!" in text
    assert "Your model is talking at ~18 tokens/sec" in text  # the server's own timing
    assert "YOU GOT TO WORK!" in text
    assert pid_file.exists() and server_gone(pid_file)


def test_managed_engine_is_stopped_when_stdin_closes_mid_game(managed_engine):
    gguf, pid_file = managed_engine
    player = Player({"Shall I go ahead?": ["y"]}, eof_after_plans=1)
    ui = make_ui(player)

    code = main(["--gguf", str(gguf), "--no-jev"], ui=ui, services=services())

    assert code == EXIT_INTERRUPTED
    assert "Traceback" not in output(ui)
    assert pid_file.exists() and server_gone(pid_file)


def test_managed_engine_is_stopped_when_the_game_is_terminated(managed_engine):
    """SIGTERM / SIGHUP (e.g. the terminal window closed) must not orphan llama-server."""
    import signal

    gguf, pid_file = managed_engine
    before = signal.getsignal(signal.SIGTERM)

    class Terminator(Player):
        def __call__(self, prompt: str) -> str:
            if any(question in prompt for question in PLAN_PROMPTS):
                os.kill(os.getpid(), signal.SIGTERM)  # arrives while we wait for the player
            return super().__call__(prompt)

    ui = make_ui(Terminator({"Shall I go ahead?": ["y"]}))

    code = main(["--gguf", str(gguf), "--no-jev"], ui=ui, services=services())

    assert code == EXIT_INTERRUPTED
    assert pid_file.exists() and server_gone(pid_file)
    assert signal.getsignal(signal.SIGTERM) == before  # main() puts the old handler back


# ---------------------------------------------------------------------------
# Play again: a second morning with the same model and the same Jev client
# ---------------------------------------------------------------------------


def test_play_again_reuses_the_model_and_jev_for_a_second_game(tmp_path, monkeypatch, isolated_home):
    fake = FakeJev()
    monkeypatch.setattr(jev, "urllib_transport", fake)
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    closed: list[str] = []
    from gettowork.backends.mock import MockBackend

    monkeypatch.setattr(MockBackend, "close", lambda self: closed.append("closed"))
    player = Player({
        "Press Enter": [""] * 60,  # a person is playing: pauses between the long stretches of text
        "Use it?": ["use"],
        "Jev request & response": ["n", "n"],
        "reasoning (chain-of-thought)": ["n", "n"],
        "Save a transcript": ["n", "n"],
        "Play again?": ["y", "n"],
    }, plans=PLANS * 2)
    ui = UI(console=Console(file=io.StringIO(), width=80), input_fn=player,
            secret_fn=lambda prompt: pytest.fail("the key should come from the environment"),
            open_url_fn=lambda url: True, pauses=True)

    code = main(["--mock", "--export-dir", str(tmp_path / "exports")], ui=ui, services=services())

    assert code == EXIT_OK
    text = output(ui)
    assert "Before you play" in text  # the first-launch note about AI-written content...
    assert sum("Press Enter to start" in p for p in player.prompts) == 1  # ...shown once, with a pause
    assert text.count("Behind the scenes") == 2  # two games, two reviews
    assert text.count("Here comes a brand-new morning!") == 1
    assert text.count("Jev's verdict") == 10  # five rounds per game, all judged by Jev
    # The key is checked once; the same Jev client referees both games.
    assert [c[1].rsplit("/", 1)[-1] for c in fake.calls] == ["models"] + ["systemone"] * 10
    assert sum("Play again?" in p for p in player.prompts) == 2
    assert closed == ["closed"]  # the model is stopped once, at the very end
    from gettowork.config import Settings

    assert Settings.load().extra.get("ai_notice_seen") is True
    for haystack in (text, "\n".join(player.prompts)):
        assert KEY not in haystack


# ---------------------------------------------------------------------------
# The game's own window: the --gui-selftest that CI runs on every build
# ---------------------------------------------------------------------------


def _skip_without_a_window() -> None:
    try:
        import tkinter
    except ImportError:
        pytest.skip("this Python has no tkinter")
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        pytest.skip("no display (run under xvfb-run)")
    try:
        root = tkinter.Tk()
    except tkinter.TclError as exc:
        pytest.skip(f"Tk can't open a window: {exc}")
    root.destroy()


def test_gui_selftest_plays_the_real_game_in_its_window(tmp_path, monkeypatch, isolated_home):
    """What double-clicking the built game (or Steam) runs, with the self-test player at the keyboard."""
    import gc

    from gettowork import launcher
    from gettowork.config import Settings

    _skip_without_a_window()
    out = tmp_path / "selftest.txt"
    monkeypatch.setenv("GETTOWORK_SELFTEST_OUT", str(out))
    monkeypatch.delenv("GETTOWORK_SELFTEST_TIMEOUT", raising=False)
    gc.collect()
    try:
        code = launcher.gui_main(["--gui-selftest"])
    finally:
        gc.collect()  # free leftover Tk objects here, on the main thread

    transcript = out.read_text(encoding="utf-8")
    assert code == 0, transcript[-2000:]
    assert "Before you play" in transcript  # the first-launch AI note, with its pause...
    assert "Press Enter to start" in transcript
    assert "How do you plan to get to work?" in transcript
    assert "YOU GOT TO WORK" in transcript
    assert "Play again?" in transcript  # ...the self-test says no...
    assert "Thanks for playing" in transcript  # ...and the game ends normally
    assert Settings.load().extra.get("ai_notice_seen") is True  # (GETTOWORK_HOME was set: that folder is used)
