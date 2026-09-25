"""Tests for gettowork.setup_flow.

No network, no real hardware probing, no downloads, no processes: the flow
reaches the outside world only through `SetupServices`, so every test hands
it fake hardware, a fake Hugging Face search and fake engines (`FakeFactory`),
and drives the conversation with scripted answers.
"""

from __future__ import annotations

import dataclasses
import io
import json
from pathlib import Path

import pytest
from rich.console import Console

from gettowork import catalog, runtime_install, setup_flow
from gettowork.backends.base import BackendError, LLMBackend
from gettowork.cli import build_parser
from gettowork.config import Settings
from gettowork.download import DownloadError
from gettowork.hf_discovery import DiscoveryResult
from gettowork.setup_flow import (
    CALIBRATION_KEY,
    FANCY_SYMBOLS,
    PLAIN_SYMBOLS,
    SetupResult,
    SetupServices,
    apply_saved_calibration,
    calibrate_specs,
    discovery_summary,
    entry_for_fit,
    entry_from_dict,
    entry_to_dict,
    fit_for_quant,
    format_size,
    format_speed,
    hardware_fingerprint,
    model_table,
    run_setup,
    symbols_for,
    why_line,
)
from gettowork.types import GPUInfo, LLMResult, SystemSpecs
from gettowork.ui import UI, UserQuit

QWEN4B = catalog.get_model("qwen3-4b")
MODELS = list(catalog.MODEL_CATALOG)


# ---------------------------------------------------------------------------
# Fixtures and fakes
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Settings, models and runtimes all live in a throwaway folder."""
    home = tmp_path / "home"
    monkeypatch.setenv("GETTOWORK_HOME", str(home))
    monkeypatch.delenv("GETTOWORK_MODELS_DIR", raising=False)
    return home


def make_specs(*, gpu: bool = True, ram: float = 32.0) -> SystemSpecs:
    gpus = [GPUInfo(name="NVIDIA GeForce RTX 3060", vendor="nvidia", vram_gb=12.0, bandwidth_gbs=360.0,
                    driver_version="550.54")] if gpu else []
    return SystemSpecs(
        os_name="Linux", os_version="Test 1.0", arch="x86_64", cpu_name="Test CPU 9000",
        cpu_cores_physical=8, cpu_cores_logical=16, ram_total_gb=ram, ram_available_gb=ram * 0.75,
        disk_free_gb=500.0, gpus=gpus, ram_bandwidth_gbs=40.0, cpu_flags=["avx", "avx2"],
    )


def tiny_specs() -> SystemSpecs:
    """A computer too small for any model."""
    return dataclasses.replace(make_specs(gpu=False, ram=1.5), ram_available_gb=0.5, ram_bandwidth_gbs=5.0)


class Script:
    """Scripted answers for UI prompts; records every prompt it was asked."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"Unexpected extra prompt: {prompt!r}")
        return self.answers.pop(0)


def make_ui(script: Script) -> UI:
    return UI(console=Console(file=io.StringIO(), width=220), input_fn=script, open_url_fn=lambda url: True)


def output(ui: UI) -> str:
    return ui.console.file.getvalue()


class FakeBackend(LLMBackend):
    """Pretends to be any engine; behaviour is set by its FakeFactory."""

    def __init__(self, factory: "FakeFactory", kind: str, kwargs: dict) -> None:
        self.factory = factory
        self.name = kind
        self.kwargs = kwargs
        self.prepare_calls = 0
        self.close_calls = 0
        self.benchmarked = False
        self.model_path = kwargs.get("model_path")
        self.server_exe = kwargs.get("server_exe")
        entry = kwargs.get("entry")
        self.model = kwargs.get("ollama_model") or (entry.ollama_ref if entry is not None else "")

    @property
    def model_label(self) -> str:
        return f"fake {self.name}"

    def is_available(self) -> tuple[bool, str]:
        return self.factory.available[self.name]

    def prepare(self, ui, entry=None) -> None:
        self.prepare_calls += 1
        if self.name in self.factory.quit_on_prepare:
            raise UserQuit()
        if self.factory.failures.get(self.name, 0) > 0:
            self.factory.failures[self.name] -= 1
            raise BackendError(f"The {self.name} engine exploded politely.")
        if self.name in ("managed", "llamacpp"):
            self.model_path = self.model_path or self.factory.model_file
        if self.name == "managed":
            self.server_exe = self.server_exe or self.factory.exe_file
        ui.info(f"fake {self.name} is ready")

    def chat(self, messages, *, temperature=0.9, max_tokens=700, json_mode=False) -> LLMResult:
        return LLMResult(text="ok", reasoning=None, model="fake", backend=self.name, elapsed_s=0.0)

    def benchmark(self, ui=None):
        self.benchmarked = True
        speeds = self.factory.speeds.get(self.name, 25.0)
        if isinstance(speeds, list):
            return speeds.pop(0) if len(speeds) > 1 else speeds[0]
        return speeds

    def close(self) -> None:
        self.close_calls += 1


class FakeFactory:
    """Stands in for `default_backend_factory`: records every backend it builds."""

    def __init__(self, tmp_path: Path, *, available=None, failures=None, speeds=None, quit_on_prepare=()) -> None:
        self.available = {
            "managed": (True, "The llama.cpp engine can be set up automatically."),
            "ollama": (False, "Ollama isn't running."),
            "llamacpp": (False, "llama-cpp-python isn't installed."),
            "mock": (True, "Built in."),
        }
        self.available.update(available or {})
        self.failures = dict(failures or {})
        self.speeds = dict(speeds or {})
        self.quit_on_prepare = set(quit_on_prepare)
        self.model_file = tmp_path / "downloaded" / "model-Q4_K_M.gguf"
        self.exe_file = tmp_path / "engine" / "llama-server"
        for path in (self.model_file, self.exe_file):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"x")
        self.created: list[FakeBackend] = []

    def __call__(self, kind: str, **kwargs) -> FakeBackend:
        backend = FakeBackend(self, kind, kwargs)
        self.created.append(backend)
        return backend

    def prepared(self, kind: str | None = None) -> list[FakeBackend]:
        return [b for b in self.created if b.prepare_calls and (kind is None or b.name == kind)]


def make_services(factory: FakeFactory, *, specs_fn=make_specs, models=None, source="live", runtimes=(),
                  custom=None) -> SetupServices:
    calls: dict = {"discover": [], "detect": 0, "custom": []}

    def detect(**kwargs):
        calls["detect"] += 1
        return specs_fn()

    def discover(**kwargs):
        calls["discover"].append(kwargs)
        return DiscoveryResult(models=list(MODELS if models is None else models), source=source, fetched_at=0.0,
                               notes=["A friendly note from discovery."])

    def custom_entry(repo, quant=None, **kwargs):
        calls["custom"].append((repo, quant))
        if custom is None:
            raise AssertionError("custom_entry should not be called in this test")
        if isinstance(custom, Exception):
            raise custom
        return custom

    services = SetupServices(detect_specs=detect, discover_models=discover, make_backend=factory,
                             installed_runtimes=lambda: list(runtimes), custom_entry=custom_entry)
    services.calls = calls  # type: ignore[attr-defined]  # handy for assertions
    return services


def parse(*argv: str):
    return build_parser().parse_args(list(argv))


def shortlist_for(specs: SystemSpecs, models=None) -> list:
    return catalog.pick_shortlist(catalog.rank_models(specs, list(MODELS if models is None else models)), 6)


def recommended_for(specs: SystemSpecs):
    return next(f for f in shortlist_for(specs) if "recommended" in f.badges)


def settings_file(home: Path) -> dict:
    return json.loads((home / "settings.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_format_size_and_speed():
    assert format_size(0.64) == "640 MB"
    assert format_size(2.5) == "2.5 GB"
    assert format_size(18.6) == "19 GB"
    assert format_size(None) == "?"
    assert format_size(0) == "?"
    assert format_speed(25.3) == "~25 tokens/s"
    assert format_speed(0.4) == "<1 token/s"
    assert format_speed(None) == "n/a"


def test_entry_for_fit_matches_the_chosen_quant():
    fit = catalog.evaluate_fit(make_specs(), QWEN4B)
    fit = dataclasses.replace(fit, quant="Q8_0", download_gb=4.28)
    entry = entry_for_fit(fit)
    assert entry.quant == "Q8_0"
    assert entry.file_size_gb == 4.28
    assert entry.ollama_ref == "hf.co/unsloth/Qwen3-4B-GGUF:Q8_0"
    assert entry.gguf_files == ()  # exact names are looked up at download time
    same = dataclasses.replace(fit, quant=QWEN4B.quant)
    assert entry_for_fit(same) is QWEN4B


def test_entry_for_fit_keeps_non_hub_ollama_refs():
    gpt = catalog.get_model("gpt-oss-20b")
    fit = dataclasses.replace(catalog.evaluate_fit(make_specs(), gpt), quant="Q4_K_M", download_gb=11.0)
    assert entry_for_fit(fit).ollama_ref == "gpt-oss:20b"


def test_fit_for_quant_pins_one_quantization():
    fit = fit_for_quant(make_specs(), QWEN4B, "Q8_0")
    assert fit.quant == "Q8_0"
    assert fit.download_gb == pytest.approx(4.28)
    assert fit.model.quant == "Q8_0"
    assert fit.model.ollama_ref.endswith(":Q8_0")
    # A quant the repo doesn't list is estimated from the parameter count.
    odd = fit_for_quant(make_specs(), QWEN4B, "Q2_K")
    assert odd.quant == "Q2_K" and odd.download_gb > 0


def test_entry_dict_round_trip_is_json_safe():
    data = json.loads(json.dumps(entry_to_dict(QWEN4B)))
    assert entry_from_dict(data) == QWEN4B
    assert entry_from_dict(None) is None
    assert entry_from_dict({"key": "only-a-key"}) is None
    assert entry_from_dict({**data, "surprise": 1}) == QWEN4B  # unknown fields are ignored


def test_calibrate_specs_scales_the_right_bandwidth():
    specs = make_specs()
    slower_cpu = calibrate_specs(specs, "cpu", 0.5)
    assert slower_cpu.ram_bandwidth_gbs == pytest.approx(20.0)
    assert specs.ram_bandwidth_gbs == 40.0  # the original is untouched
    faster_gpu = calibrate_specs(specs, "gpu", 1.5)
    assert faster_gpu.gpus[0].bandwidth_gbs == pytest.approx(540.0)
    assert specs.gpus[0].bandwidth_gbs == 360.0
    clamped = calibrate_specs(specs, "cpu", 0.01)
    assert clamped.ram_bandwidth_gbs == pytest.approx(40.0 * setup_flow.CALIBRATION_RANGE[0])
    assert calibrate_specs(specs, "partial", 0.5) is specs
    assert calibrate_specs(specs, "cpu", float("nan")) is specs
    assert calibrate_specs(specs, "cpu", "fast") is specs
    assert calibrate_specs(make_specs(gpu=False), "gpu", 0.5).gpus == []


def test_calibration_makes_estimates_follow_reality():
    specs = make_specs()
    fit = catalog.evaluate_fit(specs, QWEN4B)
    slower = catalog.evaluate_fit(calibrate_specs(specs, fit.placement, 0.5), QWEN4B)
    assert slower.est_tokens_per_s < fit.est_tokens_per_s


def test_apply_saved_calibration_only_on_the_same_computer():
    specs = make_specs()
    settings = Settings(extra={CALIBRATION_KEY: {"fingerprint": hardware_fingerprint(specs), "factors": {"cpu": 0.5}}})
    tuned, applied = apply_saved_calibration(specs, settings)
    assert applied and tuned.ram_bandwidth_gbs == pytest.approx(20.0)
    other = Settings(extra={CALIBRATION_KEY: {"fingerprint": "another|computer", "factors": {"cpu": 0.5}}})
    same, applied = apply_saved_calibration(specs, other)
    assert not applied and same is specs
    assert apply_saved_calibration(specs, Settings()) == (specs, False)


def test_symbols_fall_back_to_ascii_on_old_code_pages():
    utf = UI(console=Console(file=io.StringIO()))
    assert symbols_for(utf) == FANCY_SYMBOLS
    legacy_file = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")
    legacy = UI(console=Console(file=legacy_file))
    assert symbols_for(legacy) == PLAIN_SYMBOLS


def test_why_line_is_short_and_specific():
    specs = make_specs()
    fit = catalog.evaluate_fit(specs, QWEN4B)
    line = why_line(fit)
    assert line.startswith("Runs on your graphics card")
    assert "shows its thinking" in line and "hand-checked" in line
    moe = why_line(catalog.evaluate_fit(specs, catalog.get_model("qwen3-30b-a3b")))
    assert "Mixture-of-Experts" in moe
    too_big = catalog.evaluate_fit(tiny_specs(), catalog.get_model("qwen3-32b"))
    assert why_line(too_big) == too_big.reason


def test_discovery_summary_wording():
    specs = make_specs()
    ranked = catalog.rank_models(specs, MODELS)
    live = discovery_summary(DiscoveryResult(MODELS, "live"), ranked)
    assert live.startswith(f"Found {len(MODELS)} models on Hugging Face - ")
    assert "fit your computer nicely" in live
    assert discovery_summary(DiscoveryResult(MODELS, "cache"), ranked).startswith("Loaded")
    assert discovery_summary(DiscoveryResult(MODELS, "curated"), ranked).startswith("Using my")
    none_fit = discovery_summary(DiscoveryResult(MODELS, "live"), catalog.rank_models(tiny_specs(), MODELS))
    assert "none of them run well" in none_fit


def test_model_table_shows_badges_speed_and_license_warnings():
    specs = make_specs()
    picks = shortlist_for(specs)
    odd_license = dataclasses.replace(picks[-1], model=dataclasses.replace(picks[-1].model, license="llama3.2"))
    console = Console(file=io.StringIO(), width=220)
    console.print(model_table(picks[:-1] + [odd_license], FANCY_SYMBOLS))
    text = console.file.getvalue()
    assert "★ Recommended" in text and "tokens/s" in text
    assert "⚠ llama3.2" in text
    assert picks[0].model.display_name in text


def test_compact_model_table_keeps_names_readable_on_narrow_terminals():
    picks = shortlist_for(make_specs())
    for width in (60, 80, 100):
        console = Console(file=io.StringIO(), width=width)
        console.print(model_table(picks, FANCY_SYMBOLS, compact=True))
        lines = console.file.getvalue().splitlines()
        assert lines[0].endswith("╮")  # nothing was cropped
        assert "Why" not in lines[1]
    assert "tok/s" in console.file.getvalue() and "★ Recommended" in console.file.getvalue()


def test_show_model_table_picks_the_layout_from_the_terminal_width():
    picks = shortlist_for(make_specs())
    narrow = UI(console=Console(file=io.StringIO(), width=80))
    setup_flow.show_model_table(narrow, picks)
    assert "why 1" in output(narrow) and "Speed (est.)" not in output(narrow)
    wide = UI(console=Console(file=io.StringIO(), width=200))
    setup_flow.show_model_table(wide, picks)
    assert "Speed (est.)" in output(wide) and "Why" in output(wide)


def test_default_backend_factory_builds_each_engine_without_side_effects(tmp_path):
    from gettowork.backends import LlamaCppBackend, LlamaServerBackend, MockBackend, OllamaBackend

    managed = setup_flow.default_backend_factory("managed", entry=QWEN4B, quant="Q4_K_M")
    assert isinstance(managed, LlamaServerBackend) and managed.n_ctx == 4096
    ollama = setup_flow.default_backend_factory("ollama", entry=QWEN4B)
    assert isinstance(ollama, OllamaBackend) and ollama.model == QWEN4B.ollama_ref
    assert setup_flow.default_backend_factory("ollama", ollama_model="llama3.2").model == "llama3.2"
    llama = setup_flow.default_backend_factory("llamacpp", model_path=tmp_path / "m.gguf")
    assert isinstance(llama, LlamaCppBackend)
    assert isinstance(setup_flow.default_backend_factory("mock"), MockBackend)
    with pytest.raises(ValueError):
        setup_flow.default_backend_factory("teleporter")


# ---------------------------------------------------------------------------
# The flow: mock mode
# ---------------------------------------------------------------------------


def test_mock_mode_shows_hardware_and_recommendation_without_downloads(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path)
    services = make_services(factory)
    ui = make_ui(Script())  # no questions at all
    result = run_setup(ui, Settings(), args=parse("--mock"), services=services)

    assert isinstance(result, SetupResult)
    assert result.backend.name == "mock" and result.entry is None and result.fit is None
    # It doesn't go online - and says why (not "you're offline", which may be untrue).
    assert services.calls["discover"] == [{"refresh": False, "offline": True, "allow_all_licenses": False,
                                           "offline_reason": "Pretend-model mode doesn't go online"}]
    assert factory.prepared("managed") == []
    text = output(ui)
    assert "You've got 32 GB of RAM" in text  # the friendly hardware summary
    assert "Learn: memory size and speed" in text
    assert "Pretend-model mode" in text
    assert recommended_for(make_specs()).model.display_name in text
    assert "without --mock" in text
    assert not (isolated_home / "settings.json").exists()  # mock never overwrites saved choices


# ---------------------------------------------------------------------------
# The flow: the managed happy path
# ---------------------------------------------------------------------------


def test_managed_path_enter_enter_downloads_starts_and_saves(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, speeds={"managed": 25.0})
    services = make_services(factory)
    script = Script("", "")  # Enter = recommended, Enter = yes
    ui = make_ui(script)
    settings = Settings()
    result = run_setup(ui, settings, args=parse(), services=services)

    rec = recommended_for(make_specs())
    assert result.backend.name == "managed"
    assert result.entry == entry_for_fit(rec)
    assert result.fit.quant == rec.quant
    [backend] = factory.prepared()
    assert backend is result.backend and backend.benchmarked and backend.close_calls == 0
    assert backend.kwargs["entry"] == entry_for_fit(rec) and backend.kwargs["quant"] == rec.quant
    assert "Your pick" in script.prompts[0] and "(1)" in script.prompts[0]
    assert "Shall I go ahead?" in script.prompts[1] and "Y/n" in script.prompts[1]

    text = output(ui)
    assert f"Found {len(MODELS)} models on Hugging Face" in text
    assert "A friendly note from discovery." in text
    assert "★ Recommended" in text and "Speed (est.)" in text
    assert "more" in text and "refresh" in text and "custom" in text and "mock" in text and "quit" in text
    assert "Here's the plan" in text
    assert "NVIDIA CUDA 12 build" in text and "github.com/ggml-org/llama.cpp/releases" in text
    assert f"https://huggingface.co/{rec.model.hf_repo}" in text
    assert "Apache-2.0 license" in text
    assert "Nothing is installed system-wide" in text
    assert "talking at ~25 tokens/sec" in text

    saved = settings_file(isolated_home)
    assert saved["backend"] == "managed"
    assert saved["model_key"] == rec.model.key
    assert saved["model_quant"] == rec.quant
    assert saved["model_path"] == str(factory.model_file)
    assert saved["server_exe"] == str(factory.exe_file)
    assert entry_from_dict(saved["extra"]["model_entry"]) == entry_for_fit(rec)
    assert saved["extra"]["last_tokens_per_s"] == 25.0


def test_confirmation_decline_goes_back_to_the_menu(tmp_path):
    factory = FakeFactory(tmp_path)
    ui = make_ui(Script("", "n", "2", "y"))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))

    second = shortlist_for(make_specs())[1]
    assert result.entry == entry_for_fit(second)
    assert len(factory.prepared()) == 1  # nothing was started for the declined pick
    text = output(ui)
    assert "nothing was downloaded" in text
    assert text.count("Models that fit your computer") == 2


def test_confirmation_notes_an_engine_that_is_already_installed(tmp_path):
    factory = FakeFactory(tmp_path)
    best = runtime_install.plan_variants(make_specs())[0]
    runtimes = [(factory.exe_file, "b7000", best.name)]
    ui = make_ui(Script("", ""))
    run_setup(ui, Settings(), args=parse(), services=make_services(factory, runtimes=runtimes))
    assert "Already installed (b7000" in output(ui)


# ---------------------------------------------------------------------------
# The flow: automatic fallbacks
# ---------------------------------------------------------------------------


def test_managed_failure_falls_back_to_a_running_ollama(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "Ollama 0.12 is running.")}, failures={"managed": 1})
    script = Script("", "", "")  # pick, confirm the plan, and OK Ollama downloading it
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))

    assert result.backend.name == "ollama"
    [managed] = factory.prepared("managed")
    assert managed.close_calls >= 1  # the failed engine was cleaned up
    text = output(ui)
    assert "exploded politely" in text
    assert "Ollama is running on your computer" in text
    # Never a surprise multi-GB download: the size is shown and the player is asked.
    size = format_size(recommended_for(make_specs()).download_gb)
    assert f"Shall I ask Ollama to download it ({size})?" in script.prompts[-1]
    saved = settings_file(isolated_home)
    assert saved["backend"] == "ollama"
    assert saved["ollama_model"] == entry_for_fit(recommended_for(make_specs())).ollama_ref
    assert saved["model_path"] is None and saved["server_exe"] is None


def test_managed_and_ollama_failures_offer_the_pretend_model(tmp_path):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")}, failures={"managed": 1, "ollama": 1})
    script = Script("", "", "", "mock")
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))

    assert result.backend.name == "mock"
    assert all(b.close_calls >= 1 for b in factory.prepared() if b.name != "mock")
    assert "What would you like to do?" in script.prompts[-1]
    text = output(ui)
    assert "Neither the built-in engine nor Ollama could start this model" in text
    assert "Without --backend ollama" not in text  # the player never asked for Ollama


def test_managed_failure_without_ollama_can_be_retried(tmp_path):
    factory = FakeFactory(tmp_path, failures={"managed": 1})
    ui = make_ui(Script("", "", "retry"))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))

    first, second = factory.prepared("managed")
    assert result.backend is second
    assert first.close_calls >= 1 and second.close_calls == 0
    text = output(ui)
    assert "Plan B" in text and "ollama.com/download" in text and "pip install llama-cpp-python" in text


def test_failure_menu_quit_returns_none_and_cleans_up(tmp_path):
    factory = FakeFactory(tmp_path, failures={"managed": 1})
    ui = make_ui(Script("", "", "quit"))
    assert run_setup(ui, Settings(), args=parse(), services=make_services(factory)) is None
    assert all(b.close_calls >= 1 for b in factory.prepared())


def test_failure_menu_pick_returns_to_the_menu(tmp_path):
    factory = FakeFactory(tmp_path, failures={"managed": 1})
    ui = make_ui(Script("", "", "pick", "2", ""))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert result.entry == entry_for_fit(shortlist_for(make_specs())[1])


def test_managed_unavailable_uses_a_running_ollama_up_front(tmp_path):
    factory = FakeFactory(tmp_path, available={"managed": (False, "No prebuilt engine for this CPU."),
                                               "ollama": (True, "running")})
    ui = make_ui(Script("", ""))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert result.backend.name == "ollama"
    assert factory.prepared("managed") == []
    text = output(ui)
    assert "No prebuilt engine for this CPU." in text and "Ollama pulls it from Hugging Face" in text


def test_no_engine_at_all_offers_mock(tmp_path):
    factory = FakeFactory(tmp_path, available={"managed": (False, "No prebuilt engine for this CPU.")})
    ui = make_ui(Script("", "mock"))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert result.backend.name == "mock"
    text = output(ui)
    assert "can't run a real model on this computer automatically" in text
    assert "ollama.com/download" in text
    assert "Choose a different model" not in text  # another model wouldn't help


def test_benchmark_errors_do_not_stop_the_game(tmp_path):
    class Grumpy(FakeBackend):
        def benchmark(self, ui=None):
            raise BackendError("the stopwatch broke")

    factory = FakeFactory(tmp_path)
    services = make_services(factory)
    services.make_backend = lambda kind, **kw: Grumpy(factory, kind, kw)
    ui = make_ui(Script("", ""))
    result = run_setup(ui, Settings(), args=parse(), services=services)
    assert result.backend.name == "managed"
    assert "couldn't time the model (the stopwatch broke)" in output(ui)


def test_ctrl_c_during_prepare_closes_the_backend(tmp_path):
    factory = FakeFactory(tmp_path, quit_on_prepare={"managed"})
    ui = make_ui(Script("", ""))
    with pytest.raises(UserQuit):
        run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    [backend] = factory.prepared("managed")
    assert backend.close_calls == 1


# ---------------------------------------------------------------------------
# The flow: warm-up and speed
# ---------------------------------------------------------------------------


def test_slow_benchmark_offers_a_faster_model_and_learns(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, speeds={"managed": [1.5, 30.0]})
    script = Script("", "", "y", "", "")
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))

    first, second = factory.prepared("managed")
    assert first.close_calls == 1 and result.backend is second
    text = output(ui)
    assert "that's quite slow" in text
    assert "Pick a faster model, or play with this one?" in script.prompts[2]
    assert "speed estimates" in text
    assert text.count("Models that fit your computer") == 2
    calibration = settings_file(isolated_home)["extra"][CALIBRATION_KEY]
    assert calibration["fingerprint"] == hardware_fingerprint(make_specs())
    assert 0 < calibration["factors"]["gpu"] < 1


def test_slow_benchmark_can_be_accepted(tmp_path):
    factory = FakeFactory(tmp_path, speeds={"managed": 2.0})
    ui = make_ui(Script("", "", "n"))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert result.backend is factory.prepared("managed")[0]
    assert result.backend.close_calls == 0


def test_unknown_speed_still_starts(tmp_path):
    factory = FakeFactory(tmp_path, speeds={"managed": None})
    ui = make_ui(Script("", ""))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert result.backend.name == "managed"
    assert "up and running" in output(ui)


def test_accurate_estimate_is_praised(tmp_path):
    rec = recommended_for(make_specs())
    factory = FakeFactory(tmp_path, speeds={"managed": rec.est_tokens_per_s})
    ui = make_ui(Script("", ""))
    run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    assert "the fit engine was close" in output(ui)


# ---------------------------------------------------------------------------
# The flow: returning players
# ---------------------------------------------------------------------------


def saved_managed_settings(factory: FakeFactory) -> Settings:
    settings = Settings(
        backend="managed", model_key=QWEN4B.key, model_quant="Q4_K_M", model_path=str(factory.model_file),
        server_exe=str(factory.exe_file),
        extra={"model_entry": entry_to_dict(QWEN4B), "model_label": "Qwen3 4B", "last_tokens_per_s": 18.0},
    )
    settings.save()
    return Settings.load()


def test_returning_player_fast_path(tmp_path):
    factory = FakeFactory(tmp_path)
    services = make_services(factory)
    script = Script("")
    ui = make_ui(script)
    result = run_setup(ui, saved_managed_settings(factory), args=parse(), services=services)

    assert "Welcome back! Play with Qwen3 4B again?" in script.prompts[0]
    assert services.calls["discover"] == []  # no searching, no menu
    [backend] = factory.prepared()
    assert backend.kwargs["model_path"] == factory.model_file
    assert backend.kwargs["server_exe"] == factory.exe_file
    assert backend.kwargs["entry"] == QWEN4B
    assert result.entry == QWEN4B and result.fit.quant == "Q4_K_M"
    assert "Last time it talked at ~18 tokens/s" in output(ui)


def test_returning_player_can_choose_something_else(tmp_path):
    factory = FakeFactory(tmp_path)
    services = make_services(factory)
    ui = make_ui(Script("n", "", ""))
    run_setup(ui, saved_managed_settings(factory), args=parse(), services=services)
    assert len(services.calls["discover"]) == 1
    assert "Models that fit your computer" in output(ui)


def test_no_fast_path_when_the_model_file_is_gone(tmp_path):
    factory = FakeFactory(tmp_path)
    settings = saved_managed_settings(factory)
    factory.model_file.unlink()
    script = Script("", "")
    run_setup(make_ui(script), settings, args=parse(), services=make_services(factory))
    assert not any("Welcome back" in p for p in script.prompts)


def test_no_fast_path_when_the_command_line_asks_for_something_else(tmp_path):
    factory = FakeFactory(tmp_path)
    settings = saved_managed_settings(factory)
    script = Script("")
    run_setup(make_ui(script), settings, args=parse("--mock"), services=make_services(factory))
    assert not any("Welcome back" in p for p in script.prompts)


def test_fast_path_failure_falls_back_to_full_setup(tmp_path):
    factory = FakeFactory(tmp_path, failures={"managed": 1})
    services = make_services(factory)
    ui = make_ui(Script("", "", ""))
    result = run_setup(ui, saved_managed_settings(factory), args=parse(), services=services)
    assert result.backend.name == "managed"
    assert len(services.calls["discover"]) == 1
    assert "let's set things up again" in output(ui)


def test_fast_path_with_ollama(tmp_path):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")})
    settings = Settings(backend="ollama", ollama_model="hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M",
                        extra={"model_label": "Qwen3 4B (via Ollama)"})
    script = Script("")
    result = run_setup(make_ui(script), settings, args=parse(), services=make_services(factory))
    assert "Welcome back! Play with Qwen3 4B (via Ollama) again?" in script.prompts[0]
    assert result.backend.name == "ollama"
    assert result.backend.kwargs["ollama_model"] == "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"


def test_fast_path_skipped_when_ollama_is_not_running(tmp_path):
    factory = FakeFactory(tmp_path)
    settings = Settings(backend="ollama", ollama_model="llama3.2")
    script = Script("", "")
    result = run_setup(make_ui(script), settings, args=parse(), services=make_services(factory))
    assert not any("Welcome back" in p for p in script.prompts)
    assert result.backend.name == "managed"


# ---------------------------------------------------------------------------
# The flow: menu options
# ---------------------------------------------------------------------------


def test_custom_repo_is_checked_then_used(tmp_path):
    custom = dataclasses.replace(QWEN4B, key="someone/Cool-Model-GGUF", hf_repo="someone/Cool-Model-GGUF",
                                 display_name="Cool Model", source="huggingface")
    factory = FakeFactory(tmp_path)
    services = make_services(factory, custom=custom)
    ui = make_ui(Script("custom", "https://huggingface.co/someone/Cool-Model-GGUF", ""))
    result = run_setup(ui, Settings(), args=parse(), services=services)

    assert services.calls["custom"] == [("someone/Cool-Model-GGUF", None)]
    assert result.entry.hf_repo == "someone/Cool-Model-GGUF"
    assert "Your custom pick" in output(ui)


def test_custom_repo_errors_return_to_the_menu(tmp_path):
    services = make_services(FakeFactory(tmp_path), custom=DownloadError("That model doesn't exist.", "not_found"))
    ui = make_ui(Script("custom", "nobody/nothing", "custom", "not a repo!", "custom", "", "quit"))
    assert run_setup(ui, Settings(), args=parse(), services=services) is None
    text = output(ui)
    assert "That model doesn't exist." in text
    assert "doesn't look like a Hugging Face model id" in text


def test_custom_repo_with_a_restrictive_license_is_flagged(tmp_path):
    custom = dataclasses.replace(QWEN4B, key="meta/L-GGUF", hf_repo="meta/L-GGUF", display_name="L", license="llama3.2")
    ui = make_ui(Script("custom", "meta/L-GGUF", ""))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(FakeFactory(tmp_path), custom=custom))
    assert result.entry.hf_repo == "meta/L-GGUF"
    assert "not Apache-2.0 or MIT" in output(ui)


def test_more_shows_every_model_and_accepts_its_numbers(tmp_path):
    specs = make_specs()
    ranked = catalog.rank_models(specs, MODELS)
    index = next(i for i, f in enumerate(ranked) if i > 0 and f.verdict != "no" and (f.est_tokens_per_s or 0) >= 3)
    ui = make_ui(Script("more", str(index + 1), ""))
    result = run_setup(ui, Settings(), args=parse(), services=make_services(FakeFactory(tmp_path)))
    assert result.entry == entry_for_fit(ranked[index])
    assert "Every model I found" in output(ui)


def test_picking_a_model_that_wont_fit_asks_first(tmp_path):
    specs = make_specs(gpu=False, ram=8.0)
    ranked = catalog.rank_models(specs, MODELS)
    too_big = next(i for i, f in enumerate(ranked) if f.verdict == "no")
    ui = make_ui(Script("more", str(too_big + 1), "n", "back", "quit"))
    services = make_services(FakeFactory(tmp_path), specs_fn=lambda: make_specs(gpu=False, ram=8.0))
    assert run_setup(ui, Settings(), args=parse(), services=services) is None
    assert "Heads-up" in output(ui)


def test_refresh_searches_hugging_face_again(tmp_path):
    services = make_services(FakeFactory(tmp_path))
    ui = make_ui(Script("refresh", "", ""))
    run_setup(ui, Settings(), args=parse(), services=services)
    assert [c["refresh"] for c in services.calls["discover"]] == [False, True]


def test_refresh_is_refused_offline(tmp_path):
    services = make_services(FakeFactory(tmp_path), source="cache")
    ui = make_ui(Script("refresh", "quit"))
    run_setup(ui, Settings(), args=parse("--offline"), services=services)
    assert len(services.calls["discover"]) == 1 and services.calls["discover"][0]["offline"] is True
    assert "offline mode" in output(ui)


def test_quit_at_the_menu(tmp_path):
    factory = FakeFactory(tmp_path)
    assert run_setup(make_ui(Script("quit")), Settings(), args=parse(), services=make_services(factory)) is None
    assert factory.prepared() == []


def test_mock_from_the_menu(tmp_path):
    result = run_setup(make_ui(Script("mock")), Settings(), args=parse(), services=make_services(FakeFactory(tmp_path)))
    assert result.backend.name == "mock"


def test_why_teaches_without_redrawing_the_menu_but_learn_redraws_it(tmp_path):
    ui = make_ui(Script("why 1", "why 99", "99", "banana", "learn", "quit"))
    run_setup(ui, Settings(), args=parse(), services=make_services(FakeFactory(tmp_path)))
    text = output(ui)
    assert "Learn: why pick 1?" in text and "Memory: weights" in text
    assert "Learn: how I check whether a model fits" in text
    assert "Learn: where the models come from" in text
    assert "There's no number 99" in text
    assert "Type a number from the list" in text
    # "why N" is short, so the menu stays put; "learn" is ~90 lines of lessons,
    # so the numbered list is shown again below them.
    assert text.count("Models that fit your computer") == 2
    assert text.rindex("Models that fit your computer") > text.index("Learn: where the models come from")


def test_nothing_fits_makes_mock_the_default(tmp_path):
    services = make_services(FakeFactory(tmp_path), specs_fn=tiny_specs)
    script = Script("")
    result = run_setup(make_ui(script), Settings(), args=parse(), services=services)
    assert "(mock)" in script.prompts[0]
    assert result.backend.name == "mock"


# ---------------------------------------------------------------------------
# The flow: command-line shortcuts
# ---------------------------------------------------------------------------


def test_model_flag_skips_the_menu(tmp_path):
    services = make_services(FakeFactory(tmp_path))
    script = Script("")
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse("--model", "qwen3-4b"), services=services)
    assert result.entry.key == "qwen3-4b"
    assert services.calls["discover"] == []
    assert len(script.prompts) == 1 and "Shall I go ahead?" in script.prompts[0]
    assert "You asked for" in output(ui)


def test_model_flag_with_quant(tmp_path):
    ui = make_ui(Script(""))
    result = run_setup(ui, Settings(), args=parse("--model", "unsloth/Qwen3-4B-GGUF", "--quant", "Q8_0"),
                       services=make_services(FakeFactory(tmp_path)))
    assert result.entry.quant == "Q8_0" and result.fit.quant == "Q8_0"
    assert result.backend.kwargs["quant"] == "Q8_0"


def test_model_flag_finds_discovered_and_custom_repos(tmp_path):
    discovered = dataclasses.replace(QWEN4B, key="bartowski/Found-GGUF", hf_repo="bartowski/Found-GGUF",
                                     display_name="Found", source="huggingface")
    services = make_services(FakeFactory(tmp_path), models=MODELS + [discovered])
    result = run_setup(make_ui(Script("")), Settings(), args=parse("--model", "bartowski/found-gguf"), services=services)
    assert result.entry.hf_repo == "bartowski/Found-GGUF"

    custom = dataclasses.replace(QWEN4B, key="x/Y-GGUF", hf_repo="x/Y-GGUF", display_name="Y")
    services = make_services(FakeFactory(tmp_path), custom=custom)
    result = run_setup(make_ui(Script("")), Settings(), args=parse("--model", "hf.co/x/Y-GGUF:Q4_K_M"),
                       services=services)
    assert services.calls["custom"] == [("x/Y-GGUF", "Q4_K_M")]
    assert result.entry.hf_repo == "x/Y-GGUF"


def test_unknown_model_flag_falls_back_to_the_menu(tmp_path):
    services = make_services(FakeFactory(tmp_path), custom=DownloadError("Not on Hugging Face.", "not_found"))
    ui = make_ui(Script("quit"))
    assert run_setup(ui, Settings(), args=parse("--model", "nobody/nothing"), services=services) is None
    assert "Let's pick one from the list instead." in output(ui)


def test_gguf_flag_uses_the_local_file(tmp_path):
    gguf = tmp_path / "mine.gguf"
    gguf.write_bytes(b"GGUF")
    factory = FakeFactory(tmp_path)
    ui = make_ui(Script(""))
    result = run_setup(ui, Settings(), args=parse("--gguf", str(gguf)), services=make_services(factory))
    assert result.backend.name == "managed" and result.entry is None
    assert result.backend.kwargs["model_path"] == gguf
    assert "Your own model file" in output(ui)


def test_ollama_model_flag(tmp_path):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")})
    ui = make_ui(Script(""))
    result = run_setup(ui, Settings(), args=parse("--ollama-model", "llama3.2"), services=make_services(factory))
    assert result.backend.name == "ollama" and result.backend.kwargs["ollama_model"] == "llama3.2"
    assert factory.prepared("managed") == []


def test_explicit_backend_is_used_without_probing_others(tmp_path):
    factory = FakeFactory(tmp_path, available={"llamacpp": (True, "installed")})
    ui = make_ui(Script("", ""))
    result = run_setup(ui, Settings(), args=parse("--backend", "llamacpp"), services=make_services(factory))
    assert result.backend.name == "llamacpp"
    assert [b.name for b in factory.created] == ["llamacpp"]
    assert "llama-cpp-python" in output(ui)


def test_explicit_backend_failure_explains_how_to_fix_it(tmp_path):
    factory = FakeFactory(tmp_path, failures={"ollama": 1})
    ui = make_ui(Script("", "", "quit"))
    assert run_setup(ui, Settings(), args=parse("--backend", "ollama"), services=make_services(factory)) is None
    text = output(ui)
    assert "Ollama needs to be installed and running" in text
    assert factory.prepared("managed") == []  # no silent switch when the player chose an engine


def test_all_licenses_flag_reaches_discovery(tmp_path):
    services = make_services(FakeFactory(tmp_path))
    run_setup(make_ui(Script("quit")), Settings(), args=parse("--all-licenses", "--refresh-models"), services=services)
    assert services.calls["discover"] == [{"refresh": True, "offline": False, "allow_all_licenses": True}]


def test_gpu_estimate_is_not_learned_from_when_the_engine_fell_back_to_the_cpu(tmp_path, isolated_home):
    """A GPU build that wouldn't start runs on the CPU: that's no reason to distrust the GPU estimate."""
    factory = FakeFactory(tmp_path, speeds={"managed": 4.0})
    original = factory.__call__

    def cpu_fallback_backend(kind, **kwargs):
        backend = original(kind, **kwargs)
        backend.cpu_only = True  # what LlamaServerBackend reports after its GPU -> CPU fallback
        return backend

    services = make_services(factory)
    services.make_backend = cpu_fallback_backend
    ui = make_ui(Script("", ""))
    rec = recommended_for(make_specs())
    assert rec.placement == "gpu"
    run_setup(ui, Settings(), args=parse(), services=services)
    text = " ".join(output(ui).split())
    assert "running on the processor instead" in text
    assert CALIBRATION_KEY not in settings_file(isolated_home)["extra"]


def test_discovery_summary_counts_hub_models_and_built_in_seeds_separately():
    specs = make_specs()
    hub = [dataclasses.replace(m, source="huggingface") for m in MODELS[:4]]
    models = hub + MODELS[4:]
    ranked = catalog.rank_models(specs, models)
    line = discovery_summary(DiscoveryResult(models, "live"), ranked)
    assert line.startswith(f"Found 4 models on Hugging Face (plus {len(MODELS) - 4} of my hand-checked favourites)")


def test_asking_for_a_faster_model_when_there_is_none_says_so(tmp_path):
    only = catalog.get_model("qwen3-0.6b")
    factory = FakeFactory(tmp_path, speeds={"managed": 0.5})
    script = Script("", "", "y", "")  # pick, go ahead, "faster please", then "play anyway" (default yes)
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory, models=[only]))
    text = " ".join(output(ui).split())
    assert "already about the quickest model" in text
    assert "Play with this model anyway?" in script.prompts[3]
    assert result is not None and result.backend is factory.prepared("managed")[0]
    assert result.backend.close_calls == 0


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_declining_the_ollama_fallback_goes_to_the_failure_menu(tmp_path):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")}, failures={"managed": 1})
    script = Script("", "", "n", "quit")
    ui = make_ui(script)
    assert run_setup(ui, Settings(), args=parse(), services=make_services(factory)) is None
    assert factory.prepared("ollama") == []  # nothing was downloaded behind the player's back


def test_ollama_fallback_reuses_the_model_file_already_downloaded(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")}, failures={"managed": 1})
    real_call = factory.__call__

    def build(kind, **kwargs):
        backend = real_call(kind, **kwargs)
        if kind == "managed":
            backend.model_path = factory.model_file  # the download finished, then the start failed
        return backend

    services = make_services(factory)
    services.make_backend = build
    script = Script("", "", "")
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=services)

    assert result.backend.name == "ollama"
    assert result.backend.kwargs["gguf_path"] == factory.model_file  # handed over, not downloaded again
    assert "nothing new to download" in output(ui)
    assert "Shall I hand the model to Ollama?" in script.prompts[-1]
    saved = settings_file(isolated_home)
    assert saved["backend"] == "ollama" and saved["ollama_model"].startswith("gettowork-qwen")


def test_model_arg_with_unknown_size_makes_no_made_up_estimate(tmp_path, isolated_home):
    unknown = dataclasses.replace(QWEN4B, key="someone/mystery-gguf", hf_repo="someone/mystery-GGUF",
                                  display_name="Mystery", params_b=0.0, file_size_gb=0.0, quant_options=(),
                                  source="huggingface")
    factory = FakeFactory(tmp_path, speeds={"managed": 25.0})
    ui = make_ui(Script(""))
    result = run_setup(ui, Settings(), args=parse("--model", "someone/mystery-GGUF"),
                       services=make_services(factory, custom=unknown))
    text = output(ui)
    assert "couldn't work out how big this model is" in text
    assert "won't fit" not in text and "tokens/s" not in text.split("You asked for")[1].split("\n")[0]
    assert result.fit is None
    saved = settings_file(isolated_home)
    assert CALIBRATION_KEY not in saved["extra"]  # nothing learned from a non-estimate


def test_evaluate_fit_says_when_the_size_is_unknown():
    unknown = dataclasses.replace(QWEN4B, params_b=0.0, file_size_gb=0.0, quant_options=())
    fit = catalog.evaluate_fit(make_specs(), unknown)
    assert fit.verdict == "no" and fit.est_tokens_per_s is None
    assert "couldn't work out how big" in fit.reason and "tokens/s" not in fit.reason


def test_cpu_estimate_is_not_calibrated_when_a_gpu_build_may_have_helped(tmp_path, isolated_home):
    # An integrated-GPU laptop: the fit engine plans "cpu", but a Vulkan build
    # offloads to the iGPU by itself - that speed isn't a processor-only measurement.
    igpu = GPUInfo(name="Intel Iris Xe Graphics", vendor="intel", vram_gb=0.0)

    def specs():
        return dataclasses.replace(make_specs(gpu=False, ram=16.0), gpus=[igpu])

    factory = FakeFactory(tmp_path, speeds={"managed": 40.0})
    run_setup(make_ui(Script("", "")), Settings(), args=parse(), services=make_services(factory, specs_fn=specs))
    assert CALIBRATION_KEY not in settings_file(isolated_home)["extra"]


def test_cpu_estimate_is_calibrated_when_the_engine_ran_on_the_cpu(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, speeds={"managed": 40.0})

    def build(kind, **kwargs):
        backend = FakeBackend(factory, kind, kwargs)
        backend.cpu_only = True
        factory.created.append(backend)
        return backend

    services = make_services(factory, specs_fn=lambda: make_specs(gpu=False, ram=16.0))
    services.make_backend = build
    run_setup(make_ui(Script("", "")), Settings(), args=parse(), services=services)
    assert "cpu" in settings_file(isolated_home)["extra"][CALIBRATION_KEY]["factors"]


def test_gpu_the_engine_cannot_use_is_not_promised(tmp_path, monkeypatch):
    # Linux + AMD card, no Vulkan loader: the built-in engine only has a CPU build.
    monkeypatch.setattr(runtime_install, "_system_has_vulkan_loader", lambda: False)
    amd = GPUInfo(name="AMD Radeon RX 6700 XT", vendor="amd", vram_gb=12.0, bandwidth_gbs=384.0)
    specs = dataclasses.replace(make_specs(gpu=False, ram=16.0), gpus=[amd], cpu_flags=["avx2"])
    limited = setup_flow.apply_engine_limits(specs, "auto")
    assert limited.gpu_offload is False and any("Vulkan loader" in n for n in limited.notes)
    fits = catalog.rank_models(limited, MODELS)
    assert all(f.placement in ("cpu", "none") for f in fits)
    # Ollama brings its own GPU support: no limit there.
    assert setup_flow.apply_engine_limits(specs, "ollama").gpu_offload is None
    with_vulkan = dataclasses.replace(specs, cpu_flags=["avx2", "vulkan"])
    assert setup_flow.apply_engine_limits(with_vulkan, "auto").gpu_offload is None


def test_list_models_hint_does_not_offer_a_command_that_cannot_be_typed():
    ui = UI(console=Console(file=io.StringIO(), width=80), input_fn=lambda p: "")
    fits = shortlist_for(make_specs())
    setup_flow.show_model_table(ui, fits, interactive=False)
    text = " ".join(output(ui).split())
    assert "Type why 1" not in text and "Run gettowork and type why N at the model menu" in text


def test_a_model_already_on_disk_is_not_marked_wont_fit_on_a_full_disk(tmp_path):
    qwen = catalog.get_model("qwen3-4b")
    folder = setup_flow.download.model_folder(qwen.hf_repo)
    folder.mkdir(parents=True)
    (folder / "Qwen3-4B-Q4_K_M.gguf").write_bytes(b"GGUF")
    nearly_full = dataclasses.replace(make_specs(gpu=False, ram=32.0), disk_free_gb=3.0)
    fit = fit_for_quant(nearly_full, qwen, "Q4_K_M")
    assert fit.verdict != "no" and "already downloaded" in fit.reason


def test_shorter_context_is_carried_to_the_engine():
    fit = catalog.evaluate_fit(dataclasses.replace(make_specs(gpu=False, ram=3.7), ram_bandwidth_gbs=15.0),
                               catalog.get_model("qwen3-0.6b"))
    assert fit.context_tokens == catalog.MIN_CONTEXT_TOKENS
    entry = entry_for_fit(fit)
    assert entry.context_tokens == catalog.MIN_CONTEXT_TOKENS
    assert setup_flow._model_context(entry) == catalog.MIN_CONTEXT_TOKENS


def test_an_engine_that_breaks_on_the_warm_up_is_a_failed_start_not_up_and_running(tmp_path):
    from gettowork.backends.base import EngineStopped

    class Crashy(FakeBackend):
        def benchmark(self, ui=None):
            raise EngineStopped("the engine stopped as soon as it was given real work")

    factory = FakeFactory(tmp_path)
    services = make_services(factory)
    services.make_backend = lambda kind, **kw: (Crashy if kind == "managed" else FakeBackend)(factory, kind, kw)
    ui = make_ui(Script("", "", "mock"))  # pick the recommended model, confirm, then the failure menu: mock
    result = run_setup(ui, Settings(), args=parse(), services=services)
    text = output(ui)
    assert "Hmm, that didn't work" in text and "up and running" not in text
    assert result.backend.name == "mock"
    saved = tmp_path / "home" / "settings.json"
    assert not saved.exists() or settings_file(tmp_path / "home").get("backend") != "managed"


def test_the_confirmation_screen_skips_builds_known_not_to_run(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_install, "unusable_reasons", lambda root=None: {"cuda-12": "glibc"})
    specs = make_specs()
    plan = runtime_install.usable_plan(specs)
    assert all(v.name != "cuda-12" for v in plan)


# ---------------------------------------------------------------------------
# Round 3: learning from a measurement only when it's about bandwidth
# ---------------------------------------------------------------------------


def _flow(tmp_path, specs):
    flow = setup_flow._SetupFlow(make_ui(Script()), Settings(), parse(), make_services(FakeFactory(tmp_path)))
    flow.specs = specs
    return flow


def test_a_moe_or_tiny_model_measurement_teaches_nothing(tmp_path):
    rtx = make_specs()
    flow = _flow(tmp_path, rtx)
    gpt = catalog.evaluate_fit(rtx, catalog.get_model("gpt-oss-20b"))
    flow._compare_with_estimate(gpt, gpt.est_tokens_per_s * 0.6, engine="managed:cuda-12")
    tiny = catalog.evaluate_fit(rtx, catalog.get_model("qwen3-0.6b"))
    flow._compare_with_estimate(tiny, tiny.est_tokens_per_s * 0.6, engine="managed:cuda-12")
    assert CALIBRATION_KEY not in (flow.settings.extra or {})
    assert flow.specs is rtx


def test_a_dense_model_measurement_solves_for_the_bandwidth(tmp_path):
    rtx = make_specs()
    flow = _flow(tmp_path, rtx)
    fit = catalog.evaluate_fit(rtx, catalog.get_model("qwen3-14b"))
    assert flow._bandwidth_factor(fit, fit.est_tokens_per_s) == pytest.approx(1.0, rel=0.02)
    slower = flow._bandwidth_factor(fit, fit.est_tokens_per_s * 0.8)
    assert slower < 0.8  # the fixed per-token cost isn't blamed on the bandwidth
    flow._compare_with_estimate(fit, fit.est_tokens_per_s * 0.5, engine="managed:cuda-12")
    saved = flow.settings.extra[CALIBRATION_KEY]
    assert saved["engines"] == {"gpu": "managed:cuda-12"} and saved["factors"]["gpu"] < 0.5


def test_a_correction_from_another_engine_build_isnt_applied(tmp_path):
    rtx = make_specs()
    settings = Settings(extra={CALIBRATION_KEY: {"fingerprint": hardware_fingerprint(rtx), "factors": {"gpu": 0.5},
                                                 "engines": {"gpu": "managed:cuda-12"}}})
    same, applied = apply_saved_calibration(rtx, settings, engine="managed:cuda-12")
    assert applied and same.gpus[0].bandwidth_gbs == pytest.approx(180.0)
    other, applied = apply_saved_calibration(rtx, settings, engine="managed:vulkan")
    assert not applied and other is rtx


def test_a_metal_run_is_never_learned_as_processor_speed(tmp_path):
    mac = dataclasses.replace(make_specs(gpu=False, ram=16.0), os_name="Darwin", arch="arm64", unified_memory=True,
                              gpus=[GPUInfo(name="Apple M2 GPU", vendor="apple", vram_gb=11.2, bandwidth_gbs=100.0)])
    flow = _flow(tmp_path, mac)
    fit = dataclasses.replace(catalog.evaluate_fit(mac, catalog.get_model("qwen3-14b")), placement="cpu")
    flow._compare_with_estimate(fit, 5.0, engine_on_cpu=False, engine="managed:metal")
    assert CALIBRATION_KEY not in (flow.settings.extra or {})


def test_model_flag_warns_about_a_non_permissive_license_like_custom_does(tmp_path):
    custom = dataclasses.replace(QWEN4B, key="bartowski/Llama-3.2-3B-Instruct-GGUF",
                                 hf_repo="bartowski/Llama-3.2-3B-Instruct-GGUF", display_name="Llama 3.2 3B Instruct",
                                 license="llama3.2", source="huggingface")
    services = make_services(FakeFactory(tmp_path), custom=custom)
    ui = make_ui(Script(""))
    run_setup(ui, Settings(), args=parse("--model", "bartowski/Llama-3.2-3B-Instruct-GGUF"), services=services)
    assert "not Apache-2.0 or MIT - please read it" in output(ui)


def test_model_flag_with_unknown_size_still_warns_about_the_license(tmp_path):
    custom = dataclasses.replace(QWEN4B, key="x/Odd-GGUF", hf_repo="x/Odd-GGUF", display_name="Odd", license="other",
                                 params_b=0.0, file_size_gb=0.0, quant_options=(), source="huggingface")
    services = make_services(FakeFactory(tmp_path), custom=custom)
    ui = make_ui(Script(""))
    run_setup(ui, Settings(), args=parse("--model", "x/Odd-GGUF"), services=services)
    assert "not Apache-2.0 or MIT - please read it" in output(ui)


# ---------------------------------------------------------------------------
# Round 4: the managed engine's saved speed correction is really reused
# ---------------------------------------------------------------------------


class _RealNameEngine:
    """Looks like the real managed backend where the warm-up can tell: its class
    name ("llamacpp-server", not the setup kind "managed"), build and CPU mode."""

    name = __import__("gettowork.backends.llamaserver", fromlist=["LlamaServerBackend"]).LlamaServerBackend.name

    def __init__(self, speed: float, variant=runtime_install.CPU, cpu_only: bool = True) -> None:
        self.speed, self.variant, self.cpu_only = speed, variant, cpu_only

    def benchmark(self, ui=None):
        return self.speed


def test_the_warm_up_records_the_managed_engine_by_its_setup_kind(tmp_path):
    cpu_box = make_specs(gpu=False)
    flow = _flow(tmp_path, cpu_box)
    model = catalog.get_model("qwen3-8b")
    fit = catalog.evaluate_fit(cpu_box, model)
    assert fit.placement == "cpu"
    assert _RealNameEngine.name == "llamacpp-server"
    flow._warm_up(_RealNameEngine(fit.est_tokens_per_s * 0.5), setup_flow._Choice(entry=model, fit=fit), "managed")
    saved = flow.settings.extra[CALIBRATION_KEY]
    assert saved["engines"] == {"cpu": "managed:cpu"}
    # ...which is exactly the key the next launch plans with:
    assert setup_flow.planned_engine_key(cpu_box, Settings(backend="managed")) == "managed:cpu"
    assert setup_flow.engine_key("llamacpp-server", _RealNameEngine(1.0)) == "managed:cpu"


def test_a_saved_correction_is_applied_next_time_and_never_compounds(tmp_path):
    """Same computer, same real speed every game: the estimate should settle on
    that speed, and the saved factor should stay put (not drift to the clamp)."""
    cpu_box = make_specs(gpu=False)
    model = catalog.get_model("qwen3-8b")
    real = catalog.evaluate_fit(cpu_box, model).est_tokens_per_s * 0.5
    settings = Settings(backend="managed")
    factors, estimates = [], []
    for _launch in range(5):
        flow = setup_flow._SetupFlow(make_ui(Script()), settings, parse(),
                                     make_services(FakeFactory(tmp_path), specs_fn=lambda: cpu_box))
        flow.specs = flow._detect()
        fit = catalog.evaluate_fit(flow.specs, model)
        estimates.append(fit.est_tokens_per_s)
        flow._warm_up(_RealNameEngine(real), setup_flow._Choice(entry=model, fit=fit), "managed")
        factors.append(settings.extra[CALIBRATION_KEY]["factors"]["cpu"])
    assert estimates[1] == pytest.approx(real, rel=0.1)  # applied from the second game on
    assert all(f == pytest.approx(factors[0], rel=0.05) for f in factors)
    assert factors[-1] < setup_flow.CPU_CALIBRATION_MAX


def test_a_correction_that_wasnt_applied_is_not_compounded(tmp_path):
    """A factor saved for another engine build isn't applied, so the new one
    replaces it instead of multiplying it."""
    cpu_box = make_specs(gpu=False)
    model = catalog.get_model("qwen3-8b")
    settings = Settings(backend="managed", extra={CALIBRATION_KEY: {
        "fingerprint": hardware_fingerprint(cpu_box), "factors": {"cpu": 3.0}, "engines": {"cpu": "ollama"}}})
    flow = setup_flow._SetupFlow(make_ui(Script()), settings, parse(),
                                 make_services(FakeFactory(tmp_path), specs_fn=lambda: cpu_box))
    flow.specs = flow._detect()
    fit = catalog.evaluate_fit(flow.specs, model)
    assert fit.est_tokens_per_s == pytest.approx(catalog.evaluate_fit(cpu_box, model).est_tokens_per_s)
    flow._warm_up(_RealNameEngine(fit.est_tokens_per_s * 0.5), setup_flow._Choice(entry=model, fit=fit), "managed")
    assert settings.extra[CALIBRATION_KEY]["factors"]["cpu"] < 1.0
    assert settings.extra[CALIBRATION_KEY]["engines"]["cpu"] == "managed:cpu"


def test_a_returning_player_plans_with_the_build_they_saved(tmp_path, isolated_home):
    rtx = make_specs()
    folder = runtime_install._llama_root(None) / "b100-cuda-12"
    folder.mkdir(parents=True)
    exe = folder / "llama-server"
    exe.write_bytes(b"x")
    (folder / runtime_install.INSTALL_MARKER).write_text(json.dumps(
        {"tag": "b100", "variant": "cuda-12", "exe": "llama-server", "assets": []}), encoding="utf-8")
    settings = Settings(backend="managed")
    assert setup_flow.planned_engine_key(rtx, settings, server_exe=exe) == "managed:cuda-12"


def test_a_relative_gguf_path_is_saved_absolute_so_welcome_back_works_anywhere(tmp_path, isolated_home, monkeypatch):
    work = tmp_path / "work"
    (work / "fake").mkdir(parents=True)
    (work / "fake" / "mine-Q4_K_M.gguf").write_bytes(b"GGUF")
    monkeypatch.chdir(work)
    factory = FakeFactory(tmp_path)
    run_setup(make_ui(Script("")), Settings(), args=parse("--gguf", "fake/mine-Q4_K_M.gguf"),
              services=make_services(factory))
    saved = settings_file(isolated_home)
    assert Path(saved["model_path"]).is_absolute()
    assert Path(saved["model_path"]) == (work / "fake" / "mine-Q4_K_M.gguf").resolve()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    script = Script("y")
    ui = make_ui(script)
    result = run_setup(ui, Settings.load(), args=parse(), services=make_services(factory))
    assert "Welcome back" in script.prompts[0]
    assert result is not None and result.backend.kwargs["model_path"] == Path(saved["model_path"])


def test_a_saved_model_file_that_vanished_is_mentioned_not_silently_skipped(tmp_path, isolated_home):
    gone = tmp_path / "gone-Q4_K_M.gguf"
    settings = Settings(backend="managed", model_path=str(gone))
    ui = make_ui(Script("quit"))
    run_setup(ui, settings, args=parse(), services=make_services(FakeFactory(tmp_path)))
    assert "isn't there any more" in output(ui)


def test_a_hand_edited_saved_entry_with_wrong_types_never_crashes(tmp_path):
    data = entry_to_dict(QWEN4B)
    data["native_context"] = "40960"
    fixed = entry_from_dict(data)
    assert fixed is not None and fixed.native_context == 40960
    assert setup_flow._model_context(fixed) > 0
    data["native_context"] = "lots"
    assert entry_from_dict(data) is None
    data = entry_to_dict(QWEN4B)
    data["family"] = None
    assert entry_from_dict(data).family == ""


def test_typing_back_at_the_slow_model_question_goes_back_to_the_menu(tmp_path, isolated_home):
    factory = FakeFactory(tmp_path, speeds={"managed": [1.5, 30.0]})
    script = Script("", "", "back", "", "")
    ui = make_ui(script)
    result = run_setup(ui, Settings(), args=parse(), services=make_services(factory))
    first, second = factory.prepared("managed")
    assert first.close_calls == 1 and result.backend is second
    assert output(ui).count("Models that fit your computer") == 2


def test_after_a_slow_warm_up_the_menu_shows_the_measured_speed_and_is_honest_about_the_clamp(tmp_path):
    """Measured 2 tok/s against a ~20 tok/s guess: the learned correction hits its
    floor, so the game must not claim the list is now realistic, and the model
    just tried shows 2 tok/s rather than a still-hopeful estimate."""
    cpu_box = make_specs(gpu=False, ram=8.0)
    factory = FakeFactory(tmp_path, speeds={"managed": [2.0, 30.0]})
    script = Script("", "", "", "", "")
    ui = make_ui(script)
    run_setup(ui, Settings(), args=parse(), services=make_services(factory, specs_fn=lambda: cpu_box))
    text = output(ui)
    assert "as far as one measurement safely allows" in text
    assert "should be more realistic now" not in text
    tried = factory.prepared("managed")[0].kwargs["entry"]
    second_menu = text.split("Models that fit your computer")[2]
    rows = [line for line in second_menu.splitlines() if tried.display_name in line]
    assert all("~2.0 tokens/s" in row and "Recommended" not in row for row in rows)


def test_measured_speeds_replace_estimates_and_reorder_the_list():
    cpu_box = make_specs(gpu=False, ram=8.0)
    ranked = catalog.rank_models(cpu_box, MODELS)
    top = ranked[0]
    patched = setup_flow._with_measured_speeds(ranked, {(top.model.key, top.quant.upper()): 2.0})
    mine = next(f for f in patched if f.model.key == top.model.key)
    assert mine.est_tokens_per_s == 2.0 and mine.score < top.score
    assert patched[0].model.key != top.model.key
    assert setup_flow._with_measured_speeds(ranked, {}) is ranked


def test_badges_say_what_they_mean_and_the_menu_explains_them(tmp_path):
    ui = make_ui(Script("quit"))
    run_setup(ui, Settings(), args=parse(), services=make_services(FakeFactory(tmp_path)))
    text = output(ui)
    assert "Smartest that fits" not in text
    assert "Badges judge whole story turns" in text
    assert "only choice you need to make" not in text and "show you the plan to confirm" in text
    labels = setup_flow._badge_labels(PLAIN_SYMBOLS)
    assert "comfortable" in labels["fastest"] and "playable pace" in labels["smartest"]


def test_the_confirmation_screen_prices_cuda_builds_and_their_backups_honestly(tmp_path):
    rtx = dataclasses.replace(make_specs(), gpus=[dataclasses.replace(make_specs().gpus[0], driver_version="581.29")])
    flow = _flow(tmp_path, rtx)
    _title, lines, _needs = flow._engine_step(setup_flow._Choice(entry=QWEN4B))
    text = " ".join(lines)
    assert "0.6-0.8 GB" in lines[0]
    assert "NVIDIA CUDA 12 (roughly 0.6-0.8 GB" in text  # the backup is a big download too
    assert "quietly" not in text


def test_built_in_graphics_are_described_the_same_way_everywhere(tmp_path):
    from gettowork import specs as specs_module

    laptop = dataclasses.replace(
        make_specs(gpu=False, ram=8.0), os_name="Windows",
        gpus=[GPUInfo(name="Intel Iris Xe Graphics", vendor="intel", vram_gb=0.0)])
    assert specs_module.uses_built_in_graphics(laptop)
    rows = dict(specs_module.describe_specs(laptop))
    assert "Vulkan on the built-in graphics" in rows["GPU acceleration"]
    flow = _flow(tmp_path, laptop)
    _title, lines, _needs = flow._engine_step(setup_flow._Choice(entry=QWEN4B))
    assert "Vulkan" in lines[0] and "built-in graphics" in lines[1]
    assert not specs_module.uses_built_in_graphics(make_specs())


def test_a_pasted_wall_of_digits_at_the_model_menu_is_just_a_wrong_number(tmp_path):
    ui = make_ui(Script("9" * 5000, "why " + "9" * 5000, "quit"))
    assert run_setup(ui, Settings(), args=parse(), services=make_services(FakeFactory(tmp_path))) is None
    assert "There's no number 999999999999..." in output(ui)


def test_an_ollama_on_another_computer_is_named_and_asked_about_not_called_local(tmp_path):
    """OLLAMA_HOST=gpu-box.lan: the fallback must say where the model and the plans
    would go - not "Ollama is running on your computer"."""
    factory = FakeFactory(tmp_path, available={"ollama": (True, "running")})
    real_call = factory.__call__

    def make(kind, **kwargs):
        backend = real_call(kind, **kwargs)
        if kind == "ollama":
            backend.host = "http://gpu-box.lan:11434"
        return backend

    services = make_services(factory)
    services.make_backend = make
    model_file = tmp_path / "downloaded" / "model-Q4_K_M.gguf"
    script = Script("n")
    flow = setup_flow._SetupFlow(make_ui(script), Settings(), parse(), services)
    flow.specs = make_specs()
    flow._last_model_file = model_file
    assert flow._offer_ollama(setup_flow._Choice(entry=QWEN4B)) is None  # the player said no
    text = " ".join(output(flow.ui).split())
    assert "gpu-box.lan:11434" in text and "another computer" in text and "on your computer" not in text
    assert "without encryption" in text
    assert "gpu-box.lan" in script.prompts[0]


def test_onboarding_doesnt_promise_nothing_leaves_this_computer_with_a_remote_ollama():
    from gettowork import onboarding

    flow = onboarding._JevOnboarding(make_ui(Script()), Settings(), {}, lambda key: None,
                                     local_model_elsewhere="gpu-box.lan:11434")
    assert "nothing leaves this computer" not in flow._local_only_label
    assert "gpu-box.lan:11434" in flow._local_note
    from gettowork.cli import _remote_ollama

    class Remote:
        name, host = "ollama", "http://gpu-box.lan:11434"

    class Local:
        name, host = "ollama", "http://127.0.0.1:11434"

    assert _remote_ollama(Remote()) == "gpu-box.lan:11434" and _remote_ollama(Local()) is None
