"""Clef as a System One referee, and the hardware match around it.

Story models stay in MODEL_CATALOG. Clef is scored with the same memory
formula, without the chat-speed gate that picks a storyteller.
"""

from __future__ import annotations

import dataclasses
import io

import pytest
from rich.console import Console

from gettowork import catalog, hf_discovery, notices
from gettowork.config import Settings
from gettowork.onboarding import run_jev_onboarding
from gettowork.perf import estimate_gpu_bandwidth
from gettowork.backends.llamaserver import build_server_args
from gettowork.system_one import (
    CLEF_TEXT_MIN_BUILD,
    CLEF_TIMEOUT_CPU_S,
    CLEF_TIMEOUT_PARTIAL_S,
    LocalClefUnavailable,
    clef_batch_tokens,
    clef_client_timeout,
    clef_engine_status,
    clef_gguf_name,
    clef_server_args,
    engine_block_message,
    launch_local_clef,
    pinned_engine_tag,
)
from gettowork.types import GPUInfo, SystemSpecs
from gettowork.ui import UI

# Hugging Face GGUF byte lengths. Decimal GB in the catalog is bytes / 1e9.
CLEF_BYTES = {"Q4_K_M": 19232219200, "Q8_0": 28732215360, "BF16": 54064664640}
FLASH_BYTES = {"Q4_K_M": 6486448288, "Q8_0": 9657260192, "BF16": 18164488352}


def gpu(name, vendor, vram):
    info = GPUInfo(name=name, vendor=vendor, vram_gb=vram)
    info.bandwidth_gbs = estimate_gpu_bandwidth(info)
    return info


def machine(ram, ram_bw, gpus=(), *, unified=False, disk=500.0, cores=8, gpu_offload=None):
    return SystemSpecs(
        os_name="Darwin" if unified else "Linux",
        os_version="",
        arch="arm64" if unified else "x86_64",
        cpu_name="Test CPU",
        cpu_cores_physical=cores,
        cpu_cores_logical=cores * 2,
        ram_total_gb=ram,
        ram_available_gb=ram / 2,
        disk_free_gb=disk,
        gpus=list(gpus),
        unified_memory=unified,
        ram_bandwidth_gbs=ram_bw,
        cpu_flags=["neon"] if unified else ["avx2", "fma"],
        gpu_offload=gpu_offload,
    )


RTX_5090 = machine(64, 50, (gpu("NVIDIA GeForce RTX 5090", "nvidia", 32.0),), cores=16)
LAPTOP_8 = machine(16, 30, (gpu("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia", 8.0),))
MAC_16 = machine(16, 60, (gpu("Apple M2 GPU", "apple", 11.2),), unified=True)
CPU_16 = machine(16, 40)
AMD_16 = machine(32, 40, (gpu("AMD Radeon RX 7800 XT", "amd", 16.0),))
AMD_8 = machine(16, 40, (gpu("AMD Radeon RX 7600", "amd", 8.0),))
TWO_3060 = machine(32, 40, (
    gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0),
    gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0),
))


def _fits_somewhere(specs, fit):
    """The chosen quant's own plan is not an overflow."""
    assert fit is not None
    assert fit.verdict != "no"
    again = catalog.evaluate_fit(specs, fit.model)
    assert again.verdict != "no"
    assert again.quant == fit.quant
    assert again.est_memory_gb > 0


# ---------------------------------------------------------------------------
# Catalog facts (verified sizes; nothing invented)
# ---------------------------------------------------------------------------


def test_clef_is_not_a_story_model():
    keys = {model.key for model in catalog.MODEL_CATALOG}
    repos = {model.hf_repo.lower() for model in catalog.MODEL_CATALOG}
    assert "clef" not in keys and "clef-flash" not in keys
    assert "ggml-org/clef-gguf" not in repos
    assert catalog.recommend(RTX_5090).model.key != "clef"


def test_system_one_entries_match_published_files():
    assert [model.key for model in catalog.SYSTEM_ONE_CATALOG] == ["clef-flash", "clef"]
    flash = catalog.get_system_one("clef-flash")
    clef = catalog.get_system_one("ggml-org/Clef-GGUF")
    assert flash is not None and clef is not None
    for model, params, origin, files in (
        (flash, 9.08, "https://huggingface.co/Cloudflare/clef-flash", FLASH_BYTES),
        (clef, 27.02, "https://huggingface.co/Cloudflare/clef", CLEF_BYTES),
    ):
        assert model.architecture == "clef"
        assert model.license == "Apache-2.0"
        assert model.license_url == origin
        assert "ggml-org" not in model.license_url
        assert model.params_b == params
        assert model.native_context == 65536
        assert model.context_tokens == 4096
        assert model.reasoning is False
        assert catalog.thinking_mode(model) == "none"
        assert catalog.is_decision_model(model)
        options = dict(model.quant_options)
        for quant, size in files.items():
            assert options[quant] == round(size / 1e9, 2)
            bits = catalog.quant_bits(quant)
            estimate = params * bits / 8
            assert 0.75 * estimate <= options[quant] <= 1.3 * estimate
        assert model.gguf_files == (clef_gguf_name(model, "Q4_K_M"),)
        assert "UNVERIFIED" not in model.blurb  # unverified gaps live in comments and the README, not as fake specs


def test_huggingface_search_refuses_clef_as_a_storyteller():
    reason = hf_discovery.rejection_reason({"id": "ggml-org/Clef-GGUF"})
    assert reason and "decision model" in reason
    reason = hf_discovery.rejection_reason({"id": "ggml-org/Clef-Flash-GGUF"})
    assert reason and "Clef" in reason


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("specs", [RTX_5090, LAPTOP_8, MAC_16, CPU_16, AMD_16, AMD_8], ids=
                         ["rtx5090", "laptop8", "mac16", "cpu16", "amd16", "amd8"])
def test_system_one_recommendation_fits_and_prefers_the_larger_comfortable_model(specs):
    reserved = specs
    fits = {model.key: catalog.evaluate_fit(reserved, model) for model in catalog.SYSTEM_ONE_CATALOG}
    rec = catalog.recommend_system_one(specs)
    comfortable = [
        key for key, fit in fits.items()
        if fit.verdict in ("great", "ok")
        and not (fit.placement == "partial" and (fit.gpu_share or 0) < catalog.SPLIT_RECOMMENDED_MIN_GPU_SHARE)
    ]
    if comfortable:
        assert rec is not None

        def home(fit):
            if fit.placement in ("gpu", "unified"):
                return 0
            if fit.placement == "partial":
                return 1
            if fit.placement == "cpu":
                return 2
            return 3

        best = min(home(fits[key]) for key in comfortable)
        housed = [key for key in comfortable if home(fits[key]) == best]
        assert rec.model.key == max(housed, key=lambda key: fits[key].model.params_b)
        _fits_somewhere(specs, rec)
        assert "recommended" in rec.badges
    else:
        snug = [key for key, fit in fits.items() if fit.verdict == "tight"]
        if snug:
            assert rec is not None and rec.model.key == max(snug, key=lambda key: fits[key].model.params_b)
            _fits_somewhere(specs, rec)
        else:
            assert rec is None
    for fit in fits.values():
        if fit.verdict == "no":
            assert fit.reason
            assert rec is None or rec.model.key != fit.model.key


def test_rtx_5090_recommends_clef_not_an_overflowing_quant():
    fit = catalog.recommend_system_one(RTX_5090)
    assert fit is not None and fit.model.key == "clef"
    assert fit.verdict in ("great", "ok")
    assert fit.quant == "Q4_K_M"  # Q8 is a snug/overflowing step up; BF16 does not fit
    assert fit.est_memory_gb <= 32.0 - catalog.GPU_VRAM_RESERVE_GB
    huge = catalog.evaluate_fit(RTX_5090, catalog.get_system_one("clef"))
    assert huge.quant != "BF16"


def test_eight_gig_laptop_does_not_recommend_the_27b():
    big = catalog.evaluate_fit(LAPTOP_8, catalog.get_system_one("clef"))
    # Q4 can be a snug split; Q8 and BF16 are not what we offer. It is not the pick.
    assert big.verdict in ("tight", "no")
    assert big.quant == "Q4_K_M"
    fit = catalog.recommend_system_one(LAPTOP_8)
    assert fit is not None and fit.model.key == "clef-flash"
    assert fit.quant != "BF16"
    _fits_somewhere(LAPTOP_8, fit)


def test_mac_16_and_cpu_only_get_the_smaller_model():
    for specs in (MAC_16, CPU_16):
        assert catalog.evaluate_fit(specs, catalog.get_system_one("clef")).verdict == "no"
        fit = catalog.recommend_system_one(specs)
        assert fit is not None and fit.model.key == "clef-flash"
        _fits_somewhere(specs, fit)
        assert "seconds per decision" in fit.reason
        assert "tokens/s" not in fit.reason


def test_amd_cards_get_a_fit_that_does_not_overflow():
    for specs in (AMD_16, AMD_8):
        fit = catalog.recommend_system_one(specs)
        assert fit is not None
        _fits_somewhere(specs, fit)
        assert fit.model.architecture == "clef"


def test_loaded_story_model_is_reserved_before_clef_is_picked():
    story = catalog.evaluate_fit(RTX_5090, catalog.get_model("qwen3-32b"))
    assert story.placement == "gpu" and story.est_memory_gb > 10
    bare = catalog.recommend_system_one(RTX_5090)
    reserved = catalog.recommend_system_one(RTX_5090, story_fit=story)
    assert bare is not None and bare.model.key == "clef"
    # The 27B model no longer fits on the card. It can sit in system RAM, but
    # Clef-flash still fits on the leftover GPU and the chat-model estimate
    # says that call returns much sooner, so the pick is the card, with the
    # reason saying why the larger CPU fit was passed over.
    assert reserved is not None and reserved.model.key == "clef-flash"
    assert reserved.placement in ("gpu", "partial", "unified")
    assert "CPU" in reserved.reason and "sooner" in reserved.reason
    assert "seconds per decision" in reserved.reason
    assert "tokens/s" not in reserved.reason
    left = catalog.reserve_for_loaded_model(RTX_5090, story)
    _fits_somewhere(left, reserved)
    on_card = catalog.evaluate_fit(left, catalog.get_system_one("clef"))
    assert on_card.placement != "gpu"
    # Disk: the story download is subtracted, and an unknown free-disk reading is left alone.
    tight_disk = catalog.reserve_for_loaded_model(
        machine(64, 50, RTX_5090.gpus, disk=story.download_gb), story,
    )
    assert tight_disk.disk_free_gb < story.download_gb
    unknown = catalog.reserve_for_loaded_model(
        machine(64, 50, RTX_5090.gpus, disk=-1), story,
    )
    assert unknown.disk_free_gb < 0


def test_gpu_offload_disabled_does_not_plan_on_the_card():
    blocked = machine(16, 40, (gpu("NVIDIA GeForce RTX 5090", "nvidia", 32.0),), gpu_offload=False)
    fit = catalog.recommend_system_one(blocked)
    assert catalog.evaluate_fit(blocked, catalog.get_system_one("clef")).verdict == "no"
    assert fit is None or fit.placement == "cpu"


def test_two_graphics_cards_are_named_as_a_pair():
    fit = catalog.evaluate_fit(TWO_3060, catalog.get_model("qwen3-14b"))
    assert fit.placement == "gpu"
    assert "2 graphics cards" in fit.reason
    assert "combined" in fit.reason
    assert fit.reason.count("RTX 3060") >= 2
    assert "Fits on your NVIDIA GeForce RTX 3060 (" not in fit.reason
    one = catalog.evaluate_fit(
        machine(16, 40, (gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0),)),
        catalog.get_model("qwen3-8b"),
    )
    assert "RTX 3060" in one.reason
    assert "2 graphics cards" not in one.reason


@pytest.mark.parametrize("specs", [RTX_5090, LAPTOP_8, MAC_16, CPU_16, AMD_16], ids=
                         ["rtx5090", "laptop8", "mac16", "cpu16", "amd16"])
def test_story_recommendation_still_fits(specs):
    fit = catalog.recommend(specs)
    assert fit is not None and fit.verdict != "no"
    assert not catalog.is_decision_model(fit.model)
    _fits_somewhere(specs, fit)


def test_why_line_says_clef_is_not_the_story():
    from gettowork.setup_flow import why_line

    fit = catalog.evaluate_fit(RTX_5090, catalog.get_system_one("clef"))
    assert why_line(fit).startswith("decision model (referee only, not the story)")


def test_empty_and_broken_hardware_does_not_crash():
    empty = machine(0, 0, disk=0)
    assert catalog.recommend_system_one(empty) is None
    unknown_gpu = machine(32, 40, (GPUInfo("Some GPU", "unknown", 0.0),))
    fit = catalog.recommend_system_one(unknown_gpu)
    assert fit is None or fit.verdict != "no"


# ---------------------------------------------------------------------------
# Engine gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tag, status", [
    ("b11100", "too_old"),
    ("b11370", "too_old"),
    ("b11371", "ok"),
    ("b12000", "ok"),
    (None, "unknown"),
    ("", "unknown"),
    ("not-a-release", "unknown"),
])
def test_clef_engine_status(tag, status):
    assert clef_engine_status(tag) == status
    if status != "ok":
        assert "b11371" in engine_block_message(tag, status)


def test_pinned_engine_can_load_clef_text():
    pin = pinned_engine_tag()
    assert pin == "b11485"
    assert clef_engine_status(pin) == "ok"
    assert CLEF_TEXT_MIN_BUILD == 11371


def test_clef_launch_turns_repack_off_and_fits_the_prompt_in_one_batch():
    # b11485's default repack buffer cannot GET_ROWS the Q6_K output tensor
    # Clef's head reads, and graph reserve aborts before the server listens.
    # Embedding mode also shrinks the batch to the default physical size (512),
    # which is smaller than a real referee prompt (~1150 tokens).
    cpu = clef_server_args("llama-server", "Clef-Flash-Q4_K_M.gguf", port=9, n_ctx=1024, cpu_only=True)
    base = build_server_args(
        "llama-server", "Clef-Flash-Q4_K_M.gguf", port=9, n_ctx=1024, cpu_only=True,
    )
    assert cpu[:len(base)] == base
    assert cpu[len(base):] == ["--no-repack", "-b", "1024", "-ub", "1024"]
    gpu = clef_server_args("llama-server", "m.gguf", port=9, n_ctx=4096, cpu_only=False)
    assert gpu[-5:] == ["--no-repack", "-b", "4096", "-ub", "4096"]
    assert "--fit" in gpu and gpu[gpu.index("--fit") + 1] == "on"
    wide = clef_server_args("llama-server", "m.gguf", port=9, n_ctx=65536, cpu_only=False)
    assert wide[wide.index("-c") + 1] == "65536"
    assert wide[-5:] == ["--no-repack", "-b", "4096", "-ub", "4096"]


def test_old_engine_does_not_call_the_downloader(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    called = []

    def launcher(entry, fit):
        called.append(entry.key)
        return object()

    fit = catalog.recommend_system_one(RTX_5090)
    ui = UI(console=Console(file=io.StringIO(), width=120, color_system=None))
    with pytest.raises(LocalClefUnavailable):
        launch_local_clef(ui, fit.model, fit, engine_tag="b11100", launcher=launcher)
    assert called == []
    with pytest.raises(LocalClefUnavailable):
        launch_local_clef(ui, fit.model, fit, engine_tag=None, launcher=launcher)
    assert called == []
    got = launch_local_clef(ui, fit.model, fit, engine_tag="b11371", launcher=launcher)
    assert called == ["clef"]
    assert got is not None


# ---------------------------------------------------------------------------
# Onboarding menu
# ---------------------------------------------------------------------------


class _Script:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts = []

    def input(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"unexpected prompt: {prompt!r}")
        return self.answers.pop(0)


def _ui(answers):
    script = _Script(answers)
    console = Console(file=io.StringIO(), width=200, color_system=None)
    ui = UI(console=console, input_fn=script.input, secret_fn=script.input)
    return ui, script, console


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    return tmp_path


def test_menu_shows_clef_and_enter_does_not_download_on_the_old_engine(home):
    ui, script, console = _ui([""])
    called = []
    client = run_jev_onboarding(
        ui, Settings(), specs=RTX_5090, engine_tag="b11100",
        clef_launcher=lambda entry, fit: called.append(entry.key),
    )
    text = console.file.getvalue()
    assert client is None and called == []
    assert "Clef" in text and "Clef-flash" in text
    assert "Apache-2.0" in text or "apache" in text.lower()
    assert "b11371" in text
    data = Settings.load()
    assert data.jev_enabled is False
    assert data.extra.get("system_one") == "local"


def test_choosing_clef_on_an_old_engine_never_downloads(home):
    ui, script, console = _ui(["clef", "no", "no"])
    called = []
    client = run_jev_onboarding(
        ui, Settings(), specs=LAPTOP_8, engine_tag="b11100",
        clef_launcher=lambda entry, fit: called.append(entry.key),
    )
    text = " ".join(console.file.getvalue().split())
    assert client is None and called == []
    assert "Apache-2.0" in text
    assert notices.HARDWARE_STRAIN_NOTICE.split(".")[0] in text
    assert "AS IS" in text
    assert "will not download" in text


def test_yes_still_means_jev(home):
    ui, script, console = _ui(["yes", "back"])
    called = []
    client = run_jev_onboarding(
        ui, Settings(), specs=RTX_5090, engine_tag="b11371",
        clef_launcher=lambda entry, fit: called.append("clef"),
    )
    assert client is None and called == []
    assert any("API key" in prompt for prompt in script.prompts)


def test_new_engine_confirms_before_launch_and_a_miss_does_not(home):
    ui, _script, console = _ui(["clef", "y"])
    launched = []

    def launcher(entry, fit):
        launched.append((entry.key, fit.quant))
        sentinel = type("Sentinel", (), {"close": lambda self: None, "referee_name": "Clef (Clef)"})()
        return sentinel

    client = run_jev_onboarding(
        ui, Settings(), specs=RTX_5090, engine_tag="b11371", clef_launcher=launcher,
    )
    text = console.file.getvalue()
    assert launched and launched[0][0] == "clef"
    assert launched[0][1] == "Q4_K_M"
    assert getattr(client, "referee_name", "").startswith("Clef")
    assert "AS IS" in text and "Apache-2.0" in text
    assert Settings.load().extra.get("system_one") == "clef"
    assert Settings.load().jev_enabled is False


def test_wont_fit_override_defaults_to_no(home):
    ui, script, console = _ui(["clef", "", "no"])
    launched = []
    client = run_jev_onboarding(
        ui, Settings(), specs=CPU_16, engine_tag="b11371",
        clef_launcher=lambda entry, fit: launched.append(entry.key) or object(),
    )
    assert client is None and launched == []
    assert any("Download anyway" in prompt for prompt in script.prompts)
    assert "doesn't look like it will fit" in console.file.getvalue() or "Download anyway" in " ".join(script.prompts)


def test_without_specs_the_old_jev_question_is_unchanged(home):
    ui, script, console = _ui(["no"])
    client = run_jev_onboarding(ui, Settings(), engine_tag="b11371",
                                clef_launcher=lambda entry, fit: (_ for _ in ()).throw(AssertionError("launched")))
    assert client is None
    assert any("Enable Jev" in prompt for prompt in script.prompts)
    assert "Who should referee?" not in " ".join(script.prompts)


def test_a_remembered_local_choice_is_not_asked_again(home):
    settings = Settings(jev_enabled=False, extra={"system_one": "local"})
    ui, script, _console = _ui([])
    client = run_jev_onboarding(ui, settings, specs=RTX_5090, engine_tag="b11371")
    assert client is None and script.prompts == []


def test_a_first_local_decline_without_a_system_one_memory_is_asked(home):
    """Players who said no to Jev before Clef existed see the new menu once."""
    settings = Settings(jev_enabled=False)
    ui, script, _console = _ui(["no"])
    client = run_jev_onboarding(ui, settings, specs=RTX_5090, engine_tag="b11100")
    assert client is None
    assert any("Who should referee?" in prompt for prompt in script.prompts)


def test_notices_cover_hardware_warranty_and_output():
    for text in notices.LOCAL_RUN_NOTICES:
        assert text
    joined = " ".join(notices.LOCAL_RUN_NOTICES)
    assert "AS IS" in joined and "without warranty" in joined
    assert "hot" in joined
    assert "responsible" in joined
    assert "wrong" in notices.AI_OUTPUT_RESPONSIBILITY
    assert "not advice" in notices.AI_OUTPUT_RESPONSIBILITY
    for text in (joined, notices.STEAM_AI_DISCLOSURE, notices.AI_OUTPUT_RESPONSIBILITY):
        lowered = text.lower()
        assert "offensive" not in lowered
        assert "odd" not in lowered
    assert "Clef" in notices.STEAM_AI_DISCLOSURE
    assert "b11371" in notices.STEAM_AI_DISCLOSURE
    assert "b11485" in notices.STEAM_AI_DISCLOSURE
    assert "does not send plans to Cloudflare" in notices.STEAM_AI_DISCLOSURE
    assert "no AI images" in notices.STEAM_AI_DISCLOSURE


def test_old_engine_upgrade_yes_then_confirms_the_model(home):
    ui, script, console = _ui(["clef", "yes", "y"])
    launched = []

    def launcher(entry, fit):
        launched.append((entry.key, fit.quant))
        return type("Sentinel", (), {"close": lambda self: None, "referee_name": "Clef"})()

    client = run_jev_onboarding(
        ui, Settings(), specs=RTX_5090, engine_tag="b11100",
        clef_launcher=launcher, engine_upgrader=lambda: "b11485",
    )
    text = console.file.getvalue()
    assert launched == [("clef", "Q4_K_M")]
    assert getattr(client, "referee_name", "") == "Clef"
    assert "b11485" in text
    assert "AI output can be wrong" in text
    assert any("Shall I go ahead?" in prompt for prompt in script.prompts)


def test_each_card_keeps_its_own_vram_reserve():
    usable = catalog._usable_vram_gb(TWO_3060)
    assert usable == pytest.approx(2 * (12.0 - catalog.GPU_VRAM_RESERVE_GB))
    fit = catalog.evaluate_fit(TWO_3060, catalog.get_model("qwen3-14b"))
    assert fit.placement == "gpu"
    assert "on each of 2 cards" in catalog.explain_fit(TWO_3060, fit)


def test_layer_split_does_not_add_graphics_card_bandwidth():
    from gettowork import perf

    one_card = perf.layer_split_bandwidth(TWO_3060, 5.0)
    assert one_card is not None
    assert one_card[1] == "published spec, a rough guess"
    assert one_card[0] == pytest.approx(TWO_3060.gpus[0].bandwidth_gbs)
    spanning = perf.layer_split_bandwidth(TWO_3060, 20.0)
    assert spanning is not None
    assert "not added together" in spanning[1]
    assert spanning[0] == pytest.approx(one_card[0])
    assert spanning[0] < one_card[0] * 2

    slow = gpu("NVIDIA GeForce RTX 3060", "nvidia", 12.0)
    fast = gpu("NVIDIA GeForce RTX 4090", "nvidia", 24.0)
    mixed = machine(64, 50, (slow, fast))
    both = perf.layer_split_bandwidth(mixed, 30.0)
    assert both is not None
    low, high = sorted((slow.bandwidth_gbs, fast.bandwidth_gbs))
    assert low < both[0] < high


def test_more_memory_pressure_never_picks_a_bigger_referee():
    previous = None
    previous_mem = -1.0
    for key in ("qwen3-4b", "qwen3-8b", "qwen3-14b", "qwen3-32b"):
        story = catalog.evaluate_fit(RTX_5090, catalog.get_model(key))
        rec = catalog.recommend_system_one(RTX_5090, story)
        assert rec is not None
        assert "seconds per decision" in rec.reason
        assert "tokens/s" not in rec.reason
        if previous is not None and story.est_memory_gb > previous_mem + 0.1:
            assert rec.model.params_b <= previous
        previous = rec.model.params_b
        previous_mem = story.est_memory_gb
    eight = catalog.recommend_system_one(
        RTX_5090, catalog.evaluate_fit(RTX_5090, catalog.get_model("qwen3-8b")),
    )
    fourteen = catalog.recommend_system_one(
        RTX_5090, catalog.evaluate_fit(RTX_5090, catalog.get_model("qwen3-14b")),
    )
    assert eight is not None and fourteen is not None
    assert eight.model.key == "clef-flash"
    assert fourteen.model.key == "clef-flash"


def test_in_use_memory_and_the_windows_margin_change_the_budget():
    card = gpu("NVIDIA GeForce RTX 5090", "nvidia", 32.0)
    card.vram_used_gb = 6.0
    linux = machine(64, 50, (card,))
    windows = dataclasses.replace(linux, os_name="Windows")
    linux_free = catalog._usable_vram_gb(linux)
    windows_free = catalog._usable_vram_gb(windows)
    assert linux_free == pytest.approx(32.0 - 6.0 - catalog.GPU_VRAM_RESERVE_GB)
    assert windows_free == pytest.approx(linux_free - catalog.WINDOWS_VRAM_MARGIN_GB)
    fit = catalog.evaluate_fit(windows, catalog.get_system_one("clef-flash"))
    assert "needs ~" in fit.reason and "of 32 GB" not in fit.reason
    assert catalog.fit_target_mib(None) == round(catalog.GPU_VRAM_RESERVE_GB * 1024)
    assert catalog.fit_target_mib(windows) == round(
        (catalog.GPU_VRAM_RESERVE_GB + catalog.WINDOWS_VRAM_MARGIN_GB + 6.0) * 1024
    )
    assert catalog.fit_target_mib(linux) == round((catalog.GPU_VRAM_RESERVE_GB + 6.0) * 1024)


def test_local_clef_timeout_and_request_cannot_exceed_the_batch():
    from gettowork.jev import (
        JevError,
        bound_system_one_request,
        build_round_questions,
        build_round_state,
        estimate_prompt_tokens,
    )

    assert clef_batch_tokens(1024) == 1024
    assert clef_batch_tokens(65536) == 4096
    assert clef_client_timeout("gpu") == 30
    assert clef_client_timeout("partial") == CLEF_TIMEOUT_PARTIAL_S
    assert clef_client_timeout("cpu") == CLEF_TIMEOUT_CPU_S == 180
    state = build_round_state(
        intro="x" * 800, challenge="c" * 600, plan="p" * 1000,
        progress=1, target=5, history=["h" * 300] * 4,
    )
    questions = build_round_questions()
    # A maxed game request fits the 4096 batch. It does not fit a 1024 batch,
    # which is what -ub is when the context itself is 1024, so the builder cuts it.
    assert estimate_prompt_tokens(state, questions, "clef") <= 4096
    assert estimate_prompt_tokens(state, questions, "clef") > 1024
    small_state, small_questions = bound_system_one_request(state, questions, "clef", 1024)
    assert estimate_prompt_tokens(small_state, small_questions, "clef") <= 1024
    huge = {"story_so_far": "x" * 50000, "player_plan": "y" * 50000}
    assert estimate_prompt_tokens(huge, questions, "clef") > 4096
    bound_state, bound_questions = bound_system_one_request(huge, questions, "clef", 4096)
    assert estimate_prompt_tokens(bound_state, bound_questions, "clef") <= 4096

    def transport(method, url, headers, body, timeout):
        raise TimeoutError("slow hardware")

    from gettowork.jev import JevClient

    client = JevClient(
        "clef_local_abc", base_url="http://127.0.0.1:9", model="clef",
        timeout=180, max_retries=0, local_referee=True, max_prompt_tokens=4096,
        transport=transport,
    )
    with pytest.raises(JevError) as info:
        client.system_one(state, questions)
    message = info.value.message
    assert info.value.kind == "timeout"
    assert "Clef didn't finish" in message
    assert "hardware" in message
    assert "Jev didn't answer" not in message
    assert "connection may be slow" not in message
