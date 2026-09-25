"""Tests for gettowork.cli (and ``python -m gettowork``).

No network, no real hardware probing, no model downloads: `main()` accepts a
scripted UI and fake `SetupServices`, and the game / review / Jev onboarding
are swapped for recorders where a test only cares about the wiring.
"""

from __future__ import annotations

import io
import json
import runpy
import sys

import pytest
from rich.console import Console

from gettowork import __version__, catalog, game, onboarding, review
from gettowork.backends.base import BackendError, LLMBackend
from gettowork.backends.mock import MockBackend
from gettowork.cli import EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, build_parser, main
from gettowork.config import Settings
from gettowork.hf_discovery import DiscoveryResult
from gettowork.setup_flow import SetupServices
from gettowork.types import GameSummary, GPUInfo, LLMResult, SystemSpecs
from gettowork.ui import UI, UserQuit

MODELS = list(catalog.MODEL_CATALOG)


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    monkeypatch.delenv("GETTOWORK_MODELS_DIR", raising=False)
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return home


def make_specs(ram: float = 32.0, gpu: bool = True) -> SystemSpecs:
    gpus = [GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0, bandwidth_gbs=360.0)] if gpu else []
    return SystemSpecs(
        os_name="Linux", os_version="Test 1.0", arch="x86_64", cpu_name="Test CPU 9000",
        cpu_cores_physical=8, cpu_cores_logical=16, ram_total_gb=ram, ram_available_gb=ram * 0.75,
        disk_free_gb=500.0, gpus=gpus, ram_bandwidth_gbs=40.0, cpu_flags=["avx2"],
    )


class Script:
    """Scripted answers; fails loudly on an unexpected extra prompt."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"Unexpected extra prompt: {prompt!r}")
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        return answer


def make_ui(input_fn) -> UI:
    return UI(console=Console(file=io.StringIO(), width=200), input_fn=input_fn, open_url_fn=lambda url: True)


def output(ui: UI) -> str:
    return ui.console.file.getvalue()


class FakeBackend(LLMBackend):
    def __init__(self, kind: str, fail: bool = False) -> None:
        self.name = kind
        self.fail = fail
        self.close_calls = 0
        self.model_path = None
        self.server_exe = None

    @property
    def model_label(self) -> str:
        return f"fake {self.name}"

    def is_available(self):
        return (self.name != "ollama"), "fake"

    def prepare(self, ui, entry=None):
        if self.fail:
            raise BackendError("fake failure")

    def chat(self, messages, *, temperature=0.9, max_tokens=700, json_mode=False):
        return LLMResult(text="ok", reasoning=None, model="fake", backend=self.name, elapsed_s=0.0)

    def benchmark(self, ui=None):
        return 30.0

    def close(self):
        self.close_calls += 1


class CountingMock(MockBackend):
    """The real pretend model, counting close() calls."""

    close_calls = 0

    def close(self) -> None:
        CountingMock.close_calls += 1


def make_services(*, specs_fn=make_specs, backends=None, source="live", mock_cls=CountingMock) -> SetupServices:
    calls: dict = {"discover": [], "detect": 0, "backends": []}

    def detect(**kwargs):
        calls["detect"] += 1
        return specs_fn()

    def discover(**kwargs):
        calls["discover"].append(kwargs)
        return DiscoveryResult(models=list(MODELS), source=source, fetched_at=0.0, notes=[])

    def make_backend(kind, **kwargs):
        if kind == "mock":
            backend = mock_cls(seed=0)
        else:
            backend = (backends or {}).get(kind) or FakeBackend(kind)
        calls["backends"].append((kind, backend))
        return backend

    services = SetupServices(detect_specs=detect, discover_models=discover, make_backend=make_backend,
                             installed_runtimes=lambda: [], custom_entry=lambda *a, **k: None)
    services.calls = calls  # type: ignore[attr-defined]
    return services


class GameRecorder:
    """Stands in for game.Game: records how it was built; `behaviour` decides what run() does."""

    instances: list["GameRecorder"] = []
    behaviour = None  # None -> return a summary; an exception instance -> raise it

    def __init__(self, llm, ui, *, jev=None, target=5, **kwargs):
        self.llm, self.ui, self.jev, self.target = llm, ui, jev, target
        self.kwargs = kwargs
        GameRecorder.instances.append(self)

    def run(self) -> GameSummary:
        if GameRecorder.behaviour is not None:
            raise GameRecorder.behaviour
        return GameSummary(won=True, quit_early=False, progress=self.target, target=self.target, intro="", ending="")


@pytest.fixture
def recorders(monkeypatch):
    """Replace the game, review and Jev onboarding with recorders."""
    GameRecorder.instances = []
    GameRecorder.behaviour = None
    seen: dict = {"review": [], "onboarding": []}
    monkeypatch.setattr(game, "Game", GameRecorder)
    monkeypatch.setattr(review, "run_review",
                        lambda ui, summary, *, export_dir=None, **kw: seen["review"].append((summary, export_dir)))

    def fake_onboarding(ui, settings, **kwargs):
        seen["onboarding"].append(settings)
        return "JEV-CLIENT"

    monkeypatch.setattr(onboarding, "run_jev_onboarding", fake_onboarding)
    return seen


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def test_parser_defaults():
    args = build_parser().parse_args([])
    assert args.backend == "auto" and args.target == 5
    assert not any([args.mock, args.no_jev, args.list_models, args.specs, args.refresh_models, args.offline,
                    args.all_licenses, args.reset, args.debug])
    assert args.model is None and args.quant is None and args.gguf is None and args.ollama_model is None
    assert args.export_dir is None


def test_parser_accepts_every_flag(tmp_path):
    args = build_parser().parse_args([
        "--mock", "--backend", "ollama", "--model", "unsloth/Qwen3-4B-GGUF", "--quant", "Q8_0",
        "--ollama-model", "llama3.2", "--gguf", "x.gguf", "--list-models", "--specs", "--refresh-models",
        "--offline", "--all-licenses", "--no-jev", "--target", "3", "--reset", "--export-dir", str(tmp_path),
        "--debug",
    ])
    assert args.mock and args.backend == "ollama" and args.model == "unsloth/Qwen3-4B-GGUF"
    assert args.quant == "Q8_0" and args.ollama_model == "llama3.2" and args.gguf == "x.gguf"
    assert args.list_models and args.specs and args.refresh_models and args.offline and args.all_licenses
    assert args.no_jev and args.target == 3 and args.reset and args.debug
    assert args.export_dir == tmp_path


@pytest.mark.parametrize("argv", [["--target", "0"], ["--target", "many"], ["--target", "999"],
                                  ["--backend", "teleport"], ["--no-such-flag"]])
def test_bad_options_exit_with_usage_code(argv, capsys):
    assert main(argv, ui=make_ui(Script())) == EXIT_USAGE
    assert "usage: gettowork" in capsys.readouterr().err


def test_version_and_help(capsys):
    assert main(["--version"]) == EXIT_OK
    assert f"gettowork {__version__}" in capsys.readouterr().out
    assert main(["--help"]) == EXIT_OK
    help_text = capsys.readouterr().out
    assert "--mock" in help_text and "--list-models" in help_text and "examples:" in help_text


def test_quant_is_normalised(recorders):
    services = make_services()
    ui = make_ui(Script(""))
    assert main(["--model", "qwen3-4b", "--quant", "q8_0", "--no-jev"], ui=ui, services=services) == EXIT_OK
    assert GameRecorder.instances[0].llm.name == "managed"
    assert "Qwen3 4B (Q8_0)" in output(ui)


def test_missing_gguf_file_is_a_usage_error(tmp_path):
    ui = make_ui(Script())
    assert main(["--gguf", str(tmp_path / "nope.gguf")], ui=ui, services=make_services()) == EXIT_USAGE
    assert "can't find the model file" in output(ui)


# ---------------------------------------------------------------------------
# --specs / --list-models
# ---------------------------------------------------------------------------


def test_specs_prints_hardware_and_exits(isolated_home):
    services = make_services()
    ui = make_ui(Script())
    assert main(["--specs"], ui=ui, services=services) == EXIT_OK
    text = output(ui)
    assert "RTX 3060 with 12 GB of video memory" in text
    assert "RAM bandwidth" in text and "~40 GB/s (measured)" in text
    assert "Graphics memory bandwidth" in text and "~360 GB/s" in text
    assert "NVIDIA CUDA 12" in text and "CPU" in text
    assert "Learn: why speed is all about memory bandwidth" in text
    assert services.calls["discover"] == [] and services.calls["backends"] == []
    assert not (isolated_home / "settings.json").exists()


def test_list_models_prints_the_shortlist_and_honours_offline():
    services = make_services(source="cache")
    ui = make_ui(Script())
    assert main(["--list-models", "--offline"], ui=ui, services=services) == EXIT_OK
    assert services.calls["discover"] == [{"refresh": False, "offline": True, "allow_all_licenses": False}]
    rec = catalog.recommend(make_specs(), MODELS)
    text = output(ui)
    assert "Best picks for this computer" in text
    assert rec.model.display_name in text and rec.model.hf_repo in text
    assert f"gettowork --model {rec.model.hf_repo}" in text
    assert services.calls["backends"] == []  # nothing is started or downloaded


def test_list_models_passes_refresh_and_license_flags():
    services = make_services()
    assert main(["--list-models", "--refresh-models", "--all-licenses"], ui=make_ui(Script()), services=services) == 0
    assert services.calls["discover"] == [{"refresh": True, "offline": False, "allow_all_licenses": True}]


def test_list_models_when_nothing_fits():
    tiny = lambda: make_specs(ram=1.0, gpu=False)  # noqa: E731
    ui = make_ui(Script())
    assert main(["--list-models"], ui=ui, services=make_services(specs_fn=tiny)) == EXIT_OK
    assert "--mock" in output(ui)


# ---------------------------------------------------------------------------
# The whole game
# ---------------------------------------------------------------------------


def test_full_mock_game_with_scripted_input(tmp_path):
    """The real pretend model, game loop and review, driven like a player would."""
    CountingMock.close_calls = 0
    asked: list[str] = []

    def player(prompt: str) -> str:
        asked.append(prompt)
        if len(asked) > 60:
            raise AssertionError("the game kept asking questions")
        if "How do you plan to get to work?" in prompt:
            return "I ride a giant snail"
        if "What do you do?" in prompt:
            return "I tickle the obstacle until it laughs and lets me pass"
        return "n"  # no to the optional review extras and the export

    ui = make_ui(player)
    code = main(["--mock", "--no-jev", "--target", "2", "--export-dir", str(tmp_path)], ui=ui, services=make_services())
    assert code == EXIT_OK
    # Round 1 asks how you'll get to work; with a target of 2 the obstacle comes next.
    assert sum("How do you plan to get to work?" in p for p in asked) == 1
    assert sum("What do you do?" in p for p in asked) >= 1
    text = output(ui)
    assert "GET TO WORK" in text  # the banner
    assert "Pretend-model mode" in text
    assert "Behind the scenes" in text  # the review ran
    assert "Thanks for playing" in text
    assert CountingMock.close_calls == 1
    assert list(tmp_path.iterdir()) == []  # the player declined the export


def test_main_wires_setup_jev_game_and_review(tmp_path, recorders):
    services = make_services()
    ui = make_ui(Script("", ""))
    code = main(["--target", "3", "--export-dir", str(tmp_path)], ui=ui, services=services)
    assert code == EXIT_OK
    [played] = GameRecorder.instances
    assert played.llm.name == "managed" and played.jev == "JEV-CLIENT" and played.target == 3
    # The warm-up speed and the context window reach the game (they decide whether a model may think).
    assert set(played.kwargs) == {"tokens_per_s", "context_tokens", "secrets", "thinking", "force_think"}
    assert len(recorders["onboarding"]) == 1
    [(summary, export_dir)] = recorders["review"]
    assert summary.won and export_dir == tmp_path
    assert played.llm.close_calls == 1  # always shut down on the way out


def test_no_jev_skips_onboarding(recorders):
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert recorders["onboarding"] == []
    assert GameRecorder.instances[0].jev is None


def test_quitting_at_the_model_menu_says_goodbye(recorders):
    ui = make_ui(Script("quit"))
    assert main([], ui=ui, services=make_services()) == EXIT_OK
    assert GameRecorder.instances == []
    assert "see you next time" in output(ui)


def test_saved_choice_is_offered_next_time(tmp_path, recorders, isolated_home):
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"GGUF")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    class Managed(FakeBackend):
        def prepare(self, ui, entry=None):
            self.model_path, self.server_exe = model_file, exe

    services = make_services(backends={"managed": Managed("managed")})
    assert main(["--no-jev"], ui=make_ui(Script("", "")), services=services) == EXIT_OK
    saved = json.loads((isolated_home / "settings.json").read_text(encoding="utf-8"))
    assert saved["backend"] == "managed" and saved["model_path"] == str(model_file)

    script = Script("")
    services = make_services(backends={"managed": Managed("managed")})
    assert main(["--no-jev"], ui=make_ui(script), services=services) == EXIT_OK
    assert "Welcome back! Play with" in script.prompts[0]
    assert services.calls["discover"] == []


# ---------------------------------------------------------------------------
# Leaving: Ctrl+C, errors, reset
# ---------------------------------------------------------------------------


def test_ctrl_c_during_the_game_exits_130_and_closes_the_backend(recorders):
    GameRecorder.behaviour = UserQuit()
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_INTERRUPTED
    assert GameRecorder.instances[0].llm.close_calls == 1
    assert "see you next time" in output(ui)


def test_keyboard_interrupt_during_the_game_exits_130(recorders):
    GameRecorder.behaviour = KeyboardInterrupt()
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_INTERRUPTED
    assert GameRecorder.instances[0].llm.close_calls == 1


def test_ctrl_c_at_a_setup_prompt_exits_130():
    ui = make_ui(Script(KeyboardInterrupt()))
    assert main([], ui=ui, services=make_services()) == EXIT_INTERRUPTED
    assert "see you next time" in output(ui)


def test_ctrl_c_while_checking_hardware_exits_130():
    def interrupted(**kwargs):
        raise KeyboardInterrupt

    services = make_services(specs_fn=lambda: interrupted())
    assert main([], ui=make_ui(Script()), services=services) == EXIT_INTERRUPTED


def test_unexpected_errors_get_a_friendly_message(recorders):
    GameRecorder.behaviour = RuntimeError("kaboom")
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_ERROR
    text = output(ui)
    assert "something unexpected went wrong" in text and "kaboom" in text and "--debug" in text
    assert "Traceback" not in text
    assert GameRecorder.instances[0].llm.close_calls == 1


def test_debug_shows_the_traceback(recorders):
    GameRecorder.behaviour = RuntimeError("kaboom")
    ui = make_ui(Script("", ""))
    assert main(["--no-jev", "--debug"], ui=ui, services=make_services()) == EXIT_ERROR
    text = output(ui)
    assert "Traceback" in text and "kaboom" in text


def test_reset_forgets_saved_settings(isolated_home):
    Settings(backend="managed", model_key="qwen3-4b").save()
    assert (isolated_home / "settings.json").exists()
    ui = make_ui(Script())
    assert main(["--reset", "--specs"], ui=ui, services=make_services()) == EXIT_OK
    assert not (isolated_home / "settings.json").exists()
    assert "fresh start" in output(ui)


def test_python_dash_m_runs_main(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["gettowork", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("gettowork", run_name="__main__")
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_typing_quit_at_a_yes_no_question_is_a_choice_and_exits_0(recorders):
    """'quit' at "Shall I go ahead? [Y/n]" means what it means at the menu: exit 0."""
    ui = make_ui(Script("", "quit"))
    assert main([], ui=ui, services=make_services()) == EXIT_OK
    assert "see you next time" in output(ui)
