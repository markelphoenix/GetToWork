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

from gettowork import __version__, catalog, cli, game, launcher, onboarding, review
from gettowork.backends.base import BackendError, LLMBackend
from gettowork.backends.mock import MockBackend
from gettowork.cli import AI_NOTICE_SETTING, EXIT_ERROR, EXIT_INTERRUPTED, EXIT_OK, EXIT_USAGE, build_parser, main
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
        seen.setdefault("onboarding_kwargs", []).append(kwargs)
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


def _fake_built_game(tmp_path, monkeypatch, variants=("cpu", "vulkan"), tag="b7000"):
    """A built game's distribution.json + engine folder (downloads off), found via GETTOWORK_DISTRIBUTION."""
    from gettowork import distribution

    folder = tmp_path / "GetToWork"
    assets = {"cpu": "llama-{t}-bin-ubuntu-x64.tar.gz", "vulkan": "llama-{t}-bin-ubuntu-vulkan-x64.tar.gz"}
    for variant in variants:
        build = folder / "engine" / f"{tag}-{variant}"
        build.mkdir(parents=True)
        (build / "llama-server").write_bytes(b"#!engine")
        (build / "install.json").write_text(json.dumps({
            "tag": tag, "variant": variant, "exe": "llama-server", "bundled": True,
            "assets": [assets[variant].format(t=tag)]}))
    (folder / "distribution.json").write_text(json.dumps(
        {"channel": "release", "engine_downloads": False, "engine_dir": "engine", "llama_cpp_tag": tag}))
    for var in ("GETTOWORK_ENGINE_DIR", "GETTOWORK_ALLOW_ENGINE_DOWNLOAD"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("GETTOWORK_DISTRIBUTION", str(folder / "distribution.json"))
    monkeypatch.setattr(distribution, "_cache", None)
    return folder


def test_specs_in_a_built_game_shows_only_the_builds_it_ships_and_where_its_engine_comes_from(
        tmp_path, monkeypatch):
    """--specs is the build check's window into the built game (packaging/smoke_test.sh): it must
    show the release and builds the game ships, and never CUDA builds it doesn't have."""
    _fake_built_game(tmp_path, monkeypatch)
    ui = make_ui(Script())
    assert main(["--specs"], ui=ui, services=make_services()) == EXIT_OK
    text = output(ui)
    [plan] = [line for line in text.splitlines() if "llama.cpp build I'd use" in line]
    assert "CUDA" not in plan and "CPU" in plan  # an RTX 3060 - but the game ships only Vulkan and CPU builds
    assert "built into the game: llama.cpp b7000 (CPU, Vulkan); engine downloads off" in text


def test_specs_in_a_developer_copy_says_the_engine_is_downloaded():
    ui = make_ui(Script())
    assert main(["--specs"], ui=ui, services=make_services()) == EXIT_OK
    assert "engine downloads on" in output(ui)


def test_models_dir_option_is_used_and_remembered(tmp_path, isolated_home):
    from gettowork import config

    folder = tmp_path / "BigDrive" / "Models"
    ui = make_ui(Script())
    try:
        assert main(["--models-dir", str(folder), "--specs"], ui=ui, services=make_services()) == EXIT_OK
        assert folder.is_dir() and config.models_dir() == folder.resolve()
        assert Settings.load().models_dir == str(folder.resolve())
        assert "Models are kept in" in output(ui)
        config.use_models_dir(None)
        # Next launch, with no option: the remembered folder is used again.
        assert main(["--specs"], ui=make_ui(Script()), services=make_services()) == EXIT_OK
        assert config.models_dir() == folder.resolve()
    finally:
        config.use_models_dir(None)


def test_models_dir_that_cant_be_made_is_a_usage_error(tmp_path):
    from gettowork import config

    blocker = tmp_path / "a-file"
    blocker.write_text("x")
    ui = make_ui(Script())
    try:
        assert main(["--models-dir", str(blocker / "sub"), "--specs"], ui=ui, services=make_services()) == EXIT_USAGE
        assert "can't use" in output(ui)
    finally:
        config.use_models_dir(None)


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


def test_list_models_on_a_full_disk_names_the_disk_not_the_computer():
    import dataclasses

    full = lambda: dataclasses.replace(make_specs(ram=16.0, gpu=False), disk_free_gb=0.3)  # noqa: E731
    ui = make_ui(Script())
    assert main(["--list-models"], ui=ui, services=make_services(specs_fn=full)) == EXIT_OK
    said = " ".join(output(ui).split())
    assert "Not enough free disk space for any model" in said and "--models-dir" in said and "--mock" in said
    assert "run comfortably on this computer" not in said


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
    assert set(played.kwargs) == {"tokens_per_s", "context_tokens", "secrets", "thinking", "force_think", "taught"}
    assert len(recorders["onboarding"]) == 1
    [(summary, export_dir)] = recorders["review"]
    assert summary.won and export_dir == tmp_path
    assert played.llm.close_calls == 1  # always shut down on the way out


def test_jev_option_asks_about_jev_again(recorders):
    assert main(["--jev"], ui=make_ui(Script("", "")), services=make_services()) == EXIT_OK
    assert recorders["onboarding_kwargs"][-1]["ask_again"] is True
    assert main([], ui=make_ui(Script("", "")), services=make_services()) == EXIT_OK
    assert recorders["onboarding_kwargs"][-1]["ask_again"] is False


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


def test_a_pretend_model_game_never_remembers_a_no_to_jev(recorders):
    main(["--mock", "--target", "2"], ui=make_ui(Script("")), services=make_services())
    assert recorders["onboarding_kwargs"][-1]["remember_no"] is False
    main(["--target", "2"], ui=make_ui(Script("", "", "n")), services=make_services())
    assert recorders["onboarding_kwargs"][-1]["remember_no"] is True  # a real model: remembered as before


def test_choosing_jev_at_welcome_back_offers_jev_again(tmp_path, recorders, isolated_home):
    model_file = tmp_path / "m.gguf"
    model_file.write_bytes(b"GGUF")
    exe = tmp_path / "llama-server"
    exe.write_bytes(b"x")

    class Managed(FakeBackend):
        def prepare(self, ui, entry=None):
            self.model_path, self.server_exe = model_file, exe

    assert main([], ui=make_ui(Script("", "")), services=make_services(backends={"managed": Managed("managed")})) == 0
    settings = Settings.load()
    settings.jev_enabled = False
    settings.save()
    script = Script("jev")
    assert main([], ui=make_ui(script), services=make_services(backends={"managed": Managed("managed")})) == 0
    assert "Welcome back! Play with" in script.prompts[0]
    assert recorders["onboarding_kwargs"][-1]["ask_again"] is True


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


def test_unexpected_errors_get_a_friendly_message(recorders, isolated_home):
    GameRecorder.behaviour = RuntimeError("kaboom")
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_ERROR
    text = output(ui)
    assert "something unexpected went wrong" in text and "kaboom" in text
    assert "Traceback" not in text
    assert GameRecorder.instances[0].llm.close_calls == 1
    # The details are saved for a bug report (a closed window would otherwise lose them)...
    report = isolated_home / "logs" / "crash.txt"
    assert str(report) in text
    saved = report.read_text(encoding="utf-8")
    assert "Traceback" in saved and "RuntimeError: kaboom" in saved
    # ...and the terminal hint names the program to run.
    assert "gettowork --reset" in text


def test_unexpected_errors_in_a_built_game_name_its_terminal_program(recorders, monkeypatch):
    GameRecorder.behaviour = RuntimeError("kaboom")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    ui = make_ui(Script("", ""))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_ERROR
    assert "gettowork-cli --reset" in output(ui)


def test_unexpected_errors_in_the_window_offer_a_fresh_start_instead_of_commands(recorders, isolated_home):
    """The game's window has no command line: no "--debug" / "gettowork --reset" advice,
    a question instead - and the details are in a file that outlives the window."""
    GameRecorder.behaviour = RuntimeError("kaboom")
    script = Script("", "", "y")
    ui = UI(console=Console(file=io.StringIO(), width=200), input_fn=script, open_url_fn=lambda url: True,
            window=True)
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_ERROR
    text = output(ui)
    assert "--debug" not in text and "--reset" not in text and "gettowork " not in text
    assert "crash.txt" in text and (isolated_home / "logs" / "crash.txt").is_file()
    assert "Forget your saved settings now?" in script.prompts[-1]
    assert not (isolated_home / "settings.json").exists()  # said yes: a fresh start next time


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


# ---------------------------------------------------------------------------
# Play again, the first-launch AI note, and the game's window
# ---------------------------------------------------------------------------


NOTICE_TITLE = "Before you play"


def make_player_ui(input_fn) -> UI:
    """A UI like a real terminal or the game's window: a person is reading, so it pauses and asks."""
    return UI(console=Console(file=io.StringIO(), width=200), input_fn=input_fn, open_url_fn=lambda url: True,
              pauses=True)


class PreparingBackend(FakeBackend):
    """A fake engine that counts how often it was set up."""

    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.prepare_calls = 0

    def prepare(self, ui, entry=None):
        self.prepare_calls += 1


def seen_notice() -> bool:
    return bool(Settings.load().extra.get(AI_NOTICE_SETTING))


def test_play_again_twice_then_no_reuses_the_engine_and_jev(recorders):
    engine = PreparingBackend("managed")
    services = make_services(backends={"managed": engine})
    # the AI note's pause, the model menu, "Shall I go ahead?", then yes, yes, no to "Play again?"
    script = Script("", "", "", "y", "", "n")
    ui = make_player_ui(script)

    assert main(["--target", "3"], ui=ui, services=services) == EXIT_OK

    games = GameRecorder.instances
    assert len(games) == 3 and len({id(g) for g in games}) == 3  # a brand-new Game every time...
    assert all(g.llm is engine and g.jev == "JEV-CLIENT" and g.target == 3 for g in games)  # ...same engine, same Jev
    # ...and one record of the "Learn" panels already shown, so they aren't taught again.
    assert all(g.kwargs["taught"] is games[0].kwargs["taught"] for g in games)
    assert len(recorders["review"]) == 3  # each game gets its own review
    assert len(recorders["onboarding"]) == 1  # Jev is set up once
    assert services.calls["detect"] == 1  # the hardware check and setup ran once
    assert engine.prepare_calls == 1 and engine.close_calls == 1  # set up once, stopped once
    assert sum("Play again?" in p for p in script.prompts) == 3
    text = output(ui)
    assert text.count("Here comes a brand-new morning!") == 2
    assert text.count("Thanks for playing") == 1
    assert text.index("Thanks for playing") > text.rindex("Here comes a brand-new morning!")


def test_play_again_defaults_to_yes(recorders):
    script = Script("", "", "", "", "no")  # Enter at the first "Play again? [Y/n]" means yes
    assert main(["--no-jev"], ui=make_player_ui(script), services=make_services()) == EXIT_OK
    assert len(GameRecorder.instances) == 2
    assert "Play again? [Y/n]" in script.prompts[3].replace("\\", "")


def test_no_play_again_after_typing_quit_in_the_game(recorders, monkeypatch):
    def quitter(self):
        return GameSummary(won=False, quit_early=True, progress=1, target=self.target, intro="", ending="")

    monkeypatch.setattr(GameRecorder, "run", quitter)
    script = Script("", "", "")  # no "Play again?" at all (Script fails on an extra prompt)
    ui = make_player_ui(script)
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert len(GameRecorder.instances) == 1 and len(recorders["review"]) == 1
    assert "Thanks for playing" in output(ui)


def test_piped_and_scripted_runs_are_never_asked_to_play_again(recorders):
    script = Script("", "")  # the model menu and "Shall I go ahead?" - nothing else
    ui = make_ui(script)
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert len(GameRecorder.instances) == 1
    assert not any("Play again" in p for p in script.prompts)
    assert NOTICE_TITLE not in output(ui)  # nor shown the first-launch note (nobody is reading)


def _answering(rules: list[tuple[str, list[str]]], asked: list[str]):
    """An input function answering by prompt text (each rule's answers used in turn); Enter otherwise."""
    queues = [(words, list(answers)) for words, answers in rules]

    def answer(prompt: str) -> str:
        asked.append(prompt)
        assert len(asked) < 60, "the game kept asking questions"
        for words, answers in queues:
            if words in prompt and answers:
                return answers.pop(0)
        return ""

    return answer


def test_after_a_pretend_model_game_the_player_can_go_on_to_a_real_model(recorders):
    """The pretend model is a big button on the first menu. After its game, "Play again?" also offers a real
    AI model - back to the model menu - instead of only replaying the scripted one."""
    engine = PreparingBackend("managed")
    services = make_services(backends={"managed": engine})
    asked: list[str] = []
    ui = make_player_ui(_answering([("Your pick", ["mock", ""]), ("Play again?", ["real", "n"])], asked))
    assert main([], ui=ui, services=services) == EXIT_OK
    kinds = [kind for kind, _backend in services.calls["backends"] if kind in ("mock", "managed")]
    assert "mock" in kinds and kinds[-1] == "managed"
    games = GameRecorder.instances
    assert len(games) == 2 and games[0].llm.name == "mock" and games[1].llm is engine
    assert len(recorders["onboarding"]) == 2  # the trial's Jev "no" wasn't remembered: asked again for real
    assert recorders["onboarding_kwargs"][0]["remember_no"] is False
    assert recorders["onboarding_kwargs"][1]["remember_no"] is True
    text = " ".join(output(ui).split())
    assert "Play again with the pretend model" in text and "Pick a real AI model" in text
    assert "Let's find you a real AI model!" in text
    assert text.count("Thanks for playing") == 1 and "tucked back into bed" in text  # (the real model's goodbye)
    assert engine.close_calls == 1


def test_after_a_pretend_model_game_quitting_says_how_to_play_for_real(recorders):
    asked: list[str] = []
    ui = make_player_ui(_answering([("Your pick", ["mock"]), ("Play again?", ["quit"])], asked))
    assert main([], ui=ui, services=make_services()) == EXIT_OK
    text = " ".join(output(ui).split())
    assert "Thanks for playing Get To Work!" in text and "tucked back into bed" not in text
    assert "pick a real AI model from the menu" in text
    assert len(GameRecorder.instances) == 1
    # "again" replays the pretend model, as "Play again? [Y/n]" did.
    GameRecorder.instances = []
    asked = []
    ui = make_player_ui(_answering([("Your pick", ["mock"]), ("Play again?", ["", "n"])], asked))
    assert main([], ui=ui, services=make_services()) == EXIT_OK
    assert len(GameRecorder.instances) == 2 and all(g.llm.name == "mock" for g in GameRecorder.instances)


def test_quit_at_play_again_says_goodbye_and_exits_0(recorders):
    engine = PreparingBackend("managed")
    ui = make_player_ui(Script("", "", "", "quit"))
    assert main(["--no-jev"], ui=ui, services=make_services(backends={"managed": engine})) == EXIT_OK
    assert "see you next time" in output(ui)
    assert engine.close_calls == 1


def test_ctrl_c_at_play_again_exits_130_and_stops_the_engine(recorders):
    engine = PreparingBackend("managed")
    ui = make_player_ui(Script("", "", "", "y", KeyboardInterrupt()))
    code = main(["--no-jev"], ui=ui, services=make_services(backends={"managed": engine}))
    assert code == EXIT_INTERRUPTED
    assert len(GameRecorder.instances) == 2 and engine.close_calls == 1


def test_an_error_in_a_second_game_still_stops_the_engine_once(recorders, monkeypatch):
    engine = PreparingBackend("managed")
    real_run = GameRecorder.run

    def second_game_breaks(self):
        if len(GameRecorder.instances) == 2:
            raise RuntimeError("kaboom")
        return real_run(self)

    monkeypatch.setattr(GameRecorder, "run", second_game_breaks)
    ui = make_player_ui(Script("", "", "", "y"))
    assert main(["--no-jev"], ui=ui, services=make_services(backends={"managed": engine})) == EXIT_ERROR
    assert "kaboom" in output(ui) and engine.close_calls == 1


def test_the_ai_note_is_shown_only_on_the_first_launch(recorders, isolated_home):
    script = Script("", "", "", "n")
    ui = make_player_ui(script)
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    text = output(ui)
    assert NOTICE_TITLE in text and "written live by an AI" in text
    assert "Press Enter to start" in script.prompts[0]  # a moment to read it
    # Shown before anything else happens (the hardware check comes next).
    assert text.index("GET TO WORK") < text.index(NOTICE_TITLE) < text.index("RTX 3060")
    assert seen_notice()

    script = Script("", "", "n")  # no pause this time
    ui = make_player_ui(script)
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert NOTICE_TITLE not in output(ui)
    assert not any("Press Enter to start" in p for p in script.prompts)


def test_the_ai_note_keeps_the_other_settings(recorders, isolated_home):
    Settings(backend="ollama", ollama_model="llama3.2", extra={"gui_font_size": 14}).save()
    ui = make_player_ui(Script("", "n", "", "", "n"))  # pause, "no" to "Welcome back?", menu, go ahead, no
    main(["--no-jev"], ui=ui, services=make_services())
    saved = Settings.load()
    assert saved.extra[AI_NOTICE_SETTING] is True and saved.extra["gui_font_size"] == 14


def test_reset_shows_the_ai_note_again(recorders):
    Settings(extra={AI_NOTICE_SETTING: True}).save()
    ui = make_player_ui(Script("", "", "", "n"))
    assert main(["--reset", "--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert NOTICE_TITLE in output(ui) and seen_notice()


def test_the_ai_note_is_remembered_even_if_the_player_leaves_right_away(recorders):
    ui = make_player_ui(Script(EOFError()))  # the window closed at the note
    assert main([], ui=ui, services=make_services()) == EXIT_INTERRUPTED
    assert seen_notice() and GameRecorder.instances == []


def test_an_unsaveable_settings_file_does_not_stop_the_game(recorders, monkeypatch):
    def refuse(self):
        raise PermissionError("read-only folder")

    monkeypatch.setattr(Settings, "save", refuse)
    ui = make_player_ui(Script("", "", "", "n"))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert NOTICE_TITLE in output(ui) and len(GameRecorder.instances) == 1


def test_the_ai_note_falls_back_to_built_in_text(recorders, monkeypatch):
    monkeypatch.setitem(sys.modules, "gettowork.notices", None)  # "import gettowork.notices" now fails
    assert cli.ai_notice_text() == cli.FALLBACK_AI_NOTICE
    ui = make_player_ui(Script("", "", "", "n"))
    assert main(["--no-jev"], ui=ui, services=make_services()) == EXIT_OK
    assert "written live by an AI" in " ".join(output(ui).split())


def test_the_ai_note_uses_the_notices_module():
    from gettowork import notices

    assert cli.ai_notice_text() == notices.AI_CONTENT_NOTICE


def test_info_commands_never_show_the_ai_note(isolated_home):
    ui = make_player_ui(Script())
    assert main(["--specs"], ui=ui, services=make_services()) == EXIT_OK
    assert NOTICE_TITLE not in output(ui)
    assert not (isolated_home / "settings.json").exists()


def test_a_real_game_plays_twice_from_a_worker_thread_like_the_window_does(tmp_path):
    """The game's window runs main() off the main thread, with its own secret input: it must just work."""
    import signal
    import threading

    CountingMock.close_calls = 0
    asked: list[str] = []

    def player(prompt: str) -> str:
        asked.append(prompt)
        assert len(asked) < 120, "the game kept asking questions"
        if "Play again?" in prompt:
            return "y" if sum("Play again?" in p for p in asked) == 1 else "n"
        if "How do you plan to get to work?" in prompt:
            return "I ride a giant snail"
        if "What do you do?" in prompt:
            return "I tickle the obstacle until it laughs and lets me pass"
        if "Press Enter" in prompt:
            return ""
        return "n"

    ui = UI(console=Console(file=io.StringIO(), width=100, force_terminal=True), input_fn=player,
            secret_fn=lambda prompt: pytest.fail("no secret is asked with --no-jev"),
            open_url_fn=lambda url: True, pauses=True, hides_input=True, choices_fn=lambda options: None)
    before = signal.getsignal(signal.SIGTERM)
    result: list = []
    worker = threading.Thread(target=lambda: result.append(main(
        ["--mock", "--no-jev", "--target", "2", "--export-dir", str(tmp_path)], ui=ui, services=make_services())))
    worker.start()
    worker.join(60)
    assert not worker.is_alive() and result == [EXIT_OK]
    assert signal.getsignal(signal.SIGTERM) == before  # a worker thread leaves the signal handlers alone
    text = ui.console.file.getvalue()
    assert text.count("Behind the scenes") == 2 and text.count("Thanks for playing") == 1  # two games, two reviews
    first_game, second_game = text.split("Here comes a brand-new morning!")
    assert "YOU GOT TO WORK" in first_game.split("Behind the scenes")[0]  # both games were played to the end
    assert "YOU GOT TO WORK" in second_game.split("Behind the scenes")[0]
    assert sum("Play again?" in p for p in asked) == 2
    assert CountingMock.close_calls == 1


# ---------------------------------------------------------------------------
# The console program double-clicked on Windows: "Press Enter to close this window"
# ---------------------------------------------------------------------------


@pytest.fixture
def double_clicked(monkeypatch):
    """Pretend the console window closes when the game exits; records each wait for Enter."""
    waits: list[int] = []
    monkeypatch.setattr(launcher, "console_closes_on_exit", lambda: True)
    monkeypatch.setattr(launcher, "wait_before_closing", lambda: waits.append(1))
    return waits


def test_a_double_clicked_console_waits_for_enter_before_closing(double_clicked, capsys):
    assert main(["--version"]) == EXIT_OK
    assert double_clicked == [1]


def test_a_double_clicked_console_waits_after_errors_too(double_clicked, tmp_path, capsys):
    assert main(["--no-jev", "--gguf", str(tmp_path / "missing.gguf")]) == EXIT_USAGE
    assert "can't find the model file" in capsys.readouterr().out
    assert double_clicked == [1]


def test_a_double_clicked_console_waits_even_after_a_crash(double_clicked, monkeypatch):
    def crash(argv, *, ui, services):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(cli, "_run", crash)
    with pytest.raises(RuntimeError):
        main([])
    assert double_clicked == [1]


def test_the_window_and_tests_never_wait_for_enter(monkeypatch):
    """An injected UI (the game's window, a test) never checks the console at all."""
    monkeypatch.setattr(launcher, "console_closes_on_exit", lambda: pytest.fail("must not be asked"))
    monkeypatch.setattr(launcher, "wait_before_closing", lambda: pytest.fail("must not wait"))
    assert main(["--version"], ui=make_ui(Script())) == EXIT_OK


def test_a_terminal_run_never_waits_for_enter(monkeypatch, capsys):
    monkeypatch.setattr(launcher, "wait_before_closing", lambda: pytest.fail("must not wait"))
    assert main(["--version"]) == EXIT_OK  # the real check: not a frozen Windows exe
