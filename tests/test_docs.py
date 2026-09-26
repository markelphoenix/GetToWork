"""Keep the documentation honest: it must match the code it describes.

README.md, docs/LEARN.md, CONTRIBUTING.md and NOTICE.md quote formulas,
constants, worked examples, command-line flags, file locations and the Jev
wire format. These tests recompute those numbers with the real code and fail
when the two drift apart. If one fails, update whichever side is wrong - the
docs are part of what this project teaches.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import re
import sys
from pathlib import Path

import pytest

from gettowork import catalog, config, download, game, hf_discovery, jev, perf, prompts, review, runtime_install
from gettowork.types import GPUInfo, JevExchange, ModelEntry, SystemSpecs
from gettowork.ui import UI

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
LEARN = ROOT / "docs" / "LEARN.md"
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
NOTICE = ROOT / "NOTICE.md"
ARCHITECTURE = ROOT / "docs" / "ARCHITECTURE.md"
DISTRIBUTION = ROOT / "docs" / "DISTRIBUTION.md"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
BUILD_WORKFLOW = ROOT / ".github" / "workflows" / "build.yml"
PYPROJECT = ROOT / "pyproject.toml"
SRC = ROOT / "src" / "gettowork"
STEAM_README = ROOT / "packaging" / "steam" / "README.md"

MARKDOWN_DOCS = [README, LEARN, CONTRIBUTING, NOTICE]
# The contracts: held to the same basic checks (fences, links, no emoji).
DESIGN_DOCS = [ARCHITECTURE, DISTRIBUTION]
ALL_DOCS = MARKDOWN_DOCS + DESIGN_DOCS


# ---------------------------------------------------------------------------
# Small Markdown helpers
# ---------------------------------------------------------------------------


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flat(text: str) -> str:
    """Collapse all whitespace, so line wrapping in the docs doesn't matter."""
    return re.sub(r"\s+", " ", text)


_FENCE_RE = re.compile(r"^(\s*)(```+)(\S*)\s*$")


def code_blocks(text: str) -> list[tuple[str, str]]:
    """Every fenced code block as (language, content), including indented ones."""
    blocks: list[tuple[str, str]] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _FENCE_RE.match(lines[i])
        if not match:
            i += 1
            continue
        indent, fence, lang = match.groups()
        body: list[str] = []
        i += 1
        while i < len(lines) and lines[i].strip() != fence:
            body.append(lines[i][len(indent):] if lines[i].startswith(indent) else lines[i])
            i += 1
        blocks.append((lang.lower(), "\n".join(body)))
        i += 1  # skip the closing fence
    return blocks


def prose(text: str) -> str:
    """The text with fenced code blocks and inline code removed (for link and heading scans)."""
    out: list[str] = []
    in_code = False
    fence = ""
    for line in text.splitlines():
        match = _FENCE_RE.match(line)
        if match and (not in_code or line.strip() == fence):
            in_code = not in_code
            fence = match.group(2)
            continue
        if not in_code:
            out.append(re.sub(r"`[^`]*`", "", line))
    return "\n".join(out)


def github_slug(heading: str) -> str:
    """The anchor GitHub generates for a heading (lower-case, punctuation dropped, spaces -> hyphens)."""
    text = re.sub(r"[`*_]|\[([^\]]*)\]\([^)]*\)", lambda m: m.group(1) or "", heading.strip().lower())
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def anchors(text: str) -> set[str]:
    seen: dict[str, int] = {}
    result: set[str] = set()
    for line in prose(text).splitlines():
        match = re.match(r"^#{1,6}\s+(.+?)\s*#*\s*$", line)
        if not match:
            continue
        slug = github_slug(match.group(1))
        count = seen.get(slug, 0)
        result.add(slug if count == 0 else f"{slug}-{count}")
        seen[slug] = count + 1
    return result


_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")


def links(text: str) -> list[str]:
    return _LINK_RE.findall(prose(text))


def json_blocks(text: str) -> list:
    return [json.loads(body) for lang, body in code_blocks(text) if lang == "json"]


def find_json(text: str, predicate) -> dict:
    found = [block for block in json_blocks(text) if isinstance(block, dict) and predicate(block)]
    assert found, "expected a JSON example in the docs that isn't there any more"
    return found[0]


def machine(**overrides) -> SystemSpecs:
    """The example computers used in LEARN.md's worked examples."""
    values = dict(
        os_name="Linux", os_version="6", arch="x86_64", cpu_name="Example CPU",
        cpu_cores_physical=8, cpu_cores_logical=16, ram_total_gb=16.0, ram_available_gb=12.0,
        disk_free_gb=200.0, ram_bandwidth_gbs=40.0, cpu_flags=["avx2"],
    )
    values.update(overrides)
    return SystemSpecs(**values)


RTX_3060 = GPUInfo("NVIDIA GeForce RTX 3060", "nvidia", 12.0)
APPLE_M2 = GPUInfo("Apple M2", "apple", 11.2)


# ---------------------------------------------------------------------------
# The files themselves
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ALL_DOCS + [CI_WORKFLOW, BUILD_WORKFLOW], ids=lambda p: p.name)
def test_doc_exists_and_is_not_empty(path: Path) -> None:
    assert path.is_file(), f"{path.relative_to(ROOT)} is missing"
    assert len(read(path).strip()) > 500


@pytest.mark.parametrize("path", ALL_DOCS, ids=lambda p: p.name)
def test_code_fences_are_balanced(path: Path) -> None:
    fences = [line for line in read(path).splitlines() if _FENCE_RE.match(line)]
    assert len(fences) % 2 == 0, f"{path.name} has an unclosed ``` code block"


@pytest.mark.parametrize("path", ALL_DOCS, ids=lambda p: p.name)
def test_relative_links_and_anchors_resolve(path: Path) -> None:
    text = read(path)
    problems = []
    for target in links(text):
        if re.match(r"^(https?:|mailto:)", target):
            continue
        file_part, _, anchor = target.partition("#")
        linked = (path.parent / file_part).resolve() if file_part else path
        if not linked.exists():
            problems.append(f"{target}: {linked} does not exist")
            continue
        if anchor and linked.suffix == ".md" and anchor not in anchors(read(linked)):
            problems.append(f"{target}: no heading with anchor #{anchor} in {linked.name}")
    assert not problems, "\n".join(problems)


def test_the_helpers_slug_headings_like_github() -> None:
    assert github_slug("Part 7: Inside the game's fit engine") == "part-7-inside-the-games-fit-engine"
    assert github_slug("Where files are stored (and how to delete them)") == "where-files-are-stored-and-how-to-delete-them"
    assert github_slug("The engine: llama.cpp, CUDA, Vulkan and Metal") == "the-engine-llamacpp-cuda-vulkan-and-metal"
    assert github_slug('Chain-of-thought and "thinking" models') == "chain-of-thought-and-thinking-models"


def test_docs_use_no_emoji() -> None:
    emoji = re.compile("[\U0001F300-\U0001FAFF☀-⛿✀-➿]")
    # (ARCHITECTURE.md is left out: it quotes the "✓" that ui.make_stream_safe replaces.)
    for path in MARKDOWN_DOCS + [DISTRIBUTION]:
        assert not emoji.search(read(path)), f"{path.name} contains an emoji"


# ---------------------------------------------------------------------------
# LEARN.md: JSON examples match the real wire formats
# ---------------------------------------------------------------------------


def test_every_json_example_in_learn_is_valid_json() -> None:
    blocks = [body for lang, body in code_blocks(read(LEARN)) if lang == "json"]
    assert len(blocks) >= 5
    for body in blocks:
        json.loads(body)  # raises with a helpful position if a doc example is broken


def _fragments_in_order(excerpt: str, real: str) -> bool:
    """True if every '...'-separated fragment of a trimmed doc excerpt appears in the real text, in order."""
    position = 0
    for fragment in re.split(r"\.\.\.|…", excerpt):
        fragment = fragment.strip()
        if not fragment:
            continue
        found = real.find(fragment, position)
        if found < 0:
            return False
        position = found + len(fragment)
    return True


def test_learn_jev_request_matches_the_questions_the_game_really_asks() -> None:
    body = find_json(read(LEARN), lambda b: "questions" in b)
    assert set(body) == {"state", "model", "questions"}
    assert body["model"] == jev.JEV_DEFAULT_MODEL

    real = jev.build_round_questions()
    shown = body["questions"]
    assert list(shown) == list(real), "the question names in LEARN.md differ from jev.build_round_questions()"
    for name, question in shown.items():
        assert question["type"] == real[name]["type"]
        assert _fragments_in_order(question["instructions"], real[name]["instructions"]), name

    noul = shown["made_progress"]["criteria"]
    assert set(noul) == {"true", "false"}
    for key in ("true", "false"):
        assert _fragments_in_order(noul[key], real["made_progress"]["criteria"][key])

    choice = shown["outcome"]["criteria"]
    assert list(choice) == list(jev.ROUND_OUTCOME_LABELS)
    for label, text in choice.items():
        assert _fragments_in_order(text, real["outcome"]["criteria"][label]), label

    levels = shown["creativity"]["criteria"]
    assert len(levels) == len(jev.CREATIVITY_LEVELS)
    for excerpt, level in zip(levels, jev.CREATIVITY_LEVELS):
        assert _fragments_in_order(excerpt, level)


def test_learn_jev_state_matches_build_round_state() -> None:
    body = find_json(read(LEARN), lambda b: "questions" in b)
    real = jev.build_round_state(intro="x", challenge="y", plan="z", progress=1, target=5, history=["a"])
    assert set(body["state"]) == set(real)
    assert set(body["state"]["progress"]) == set(real["progress"])
    assert body["state"]["game"] == real["game"]
    assert body["state"]["note"] == real["note"]


def test_learn_jev_response_parses_with_the_games_own_parser() -> None:
    body = find_json(read(LEARN), lambda b: "answers" in b)
    assert set(body) == {"model", "answers", "usage"}
    assert set(body["usage"]) == {"input_tokens", "output_tokens"}

    exchange = JevExchange(url="https://api.typesafe.ai/v1/systemone", request_headers={}, request_body={},
                           status=200, response_body=body, error=None, elapsed_s=0.1)
    verdict = jev.parse_verdict(body, exchange)
    answers = body["answers"]
    assert verdict.made_progress == (answers["made_progress"]["noul"] >= 0.5)
    assert verdict.outcome in jev.ROUND_OUTCOME_LABELS
    assert math.isclose(sum(answers["outcome"]["probabilities"].values()), 1.0, abs_tol=0.01)

    score = answers["creativity"]
    expected = sum(int(level) * p for level, p in score["probabilities"].items())
    assert math.isclose(score["score"], expected, abs_tol=0.005), "the Score example should be the expected level"
    assert math.isclose(sum(score["probabilities"].values()), 1.0, abs_tol=0.01)
    assert set(score["legend"]) == {str(i) for i in range(len(jev.CREATIVITY_LEVELS))}


def test_readme_sample_game_quotes_the_learn_response() -> None:
    answers = find_json(read(LEARN), lambda b: "answers" in b)["answers"]
    text = read(README)
    # The sample mirrors the real "Jev's verdict" panel (game._jev_verdict_panel).
    assert f"{round(answers['made_progress']['noul'] * 100)}% chance of yes" in text
    choice = answers["outcome"]
    assert f'{choice["choice"]}  ({round(choice["confidence"] * 100)}% confident)' in text
    creativity = answers["creativity"]
    assert (f"{creativity['score']:.1f} out of {len(jev.CREATIVITY_LEVELS) - 1}  "
            f"({round(creativity['confidence'] * 100)}% confident)") in text


def test_learn_local_judge_example_is_what_the_game_can_read() -> None:
    body = find_json(read(LEARN), lambda b: set(b) == {"made_progress", "explanation"})
    assert prompts.parse_judge_json(json.dumps(body)) == (body["made_progress"], body["explanation"])


def test_learn_llama_server_examples_match_the_backend() -> None:
    from gettowork.backends.llamaserver import build_server_args

    text = read(LEARN)
    request = find_json(text, lambda b: "messages" in b)
    assert request["stream"] is False
    assert request["temperature"] == game.NARRATION_TEMPERATURE
    assert request["max_tokens"] == game.NARRATION_MAX_TOKENS
    assert request["messages"][0]["content"].startswith("TASK: ")

    reply = find_json(text, lambda b: "choices" in b)
    assert "reasoning_content" in reply["choices"][0]["message"]
    assert "predicted_per_second" in reply["timings"]

    # The launch command shown in Part 8 uses exactly the backend's flags.
    command = next(body for lang, body in code_blocks(text) if body.lstrip().startswith("llama-server -m model.gguf"))
    shown = command.replace("\\\n", " ").split()
    real = build_server_args("llama-server", "model.gguf", port=54321, n_ctx=4096)
    assert shown == real
    assert build_server_args("x", "m", port=1, n_ctx=1, cpu_only=True)[-2:] == ["-ngl", "0"]
    assert "`-ngl 0`" in text


def test_learn_temperatures_match_the_game() -> None:
    text = flat(read(LEARN))
    assert f"temperature of {game.NARRATION_TEMPERATURE:g} for storytelling" in text
    assert f"{game.JUDGE_TEMPERATURE:g} when the local model referees" in text


# ---------------------------------------------------------------------------
# LEARN.md: numbers and worked examples recomputed with the real code
# ---------------------------------------------------------------------------


def test_learn_quant_table_matches_catalog() -> None:
    rows = 0
    for line in read(LEARN).splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4 or cells[0] not in catalog.QUANT_BITS:
            continue
        quant, bits, quality, size = cells[:4]
        assert float(bits) == catalog.QUANT_BITS[quant], quant
        assert float(quality) == pytest.approx(catalog.quant_quality(quant)), quant
        estimate = catalog.estimate_quant_size_gb(8.0, quant)
        assert size == f"{estimate:.1f} GB", quant
        rows += 1
    assert rows >= 10
    # The smallest quant the ladder normally picks is the one LEARN.md names.
    assert catalog.QUANT_PREFERENCE[-1] == "IQ3_M"
    assert catalog.QUANT_PREFERENCE[3] == "Q4_K_M"


def test_learn_size_formula_example() -> None:
    text = flat(read(LEARN))
    assert "parameters (billions) × bits per weight ÷ 8 × 1.05" in text
    assert catalog.estimate_quant_size_gb(8.19, "Q4_K_M") == pytest.approx(5.16, abs=0.01)
    assert "8.19 × 4.8 ÷ 8 × 1.05 ≈ **5.2 GB**" in text
    assert catalog.get_model("qwen3-8b").file_size_gb == 5.03
    assert "The real file on Hugging Face is 5.03 GB" in text


def test_learn_kv_cache_example_matches_catalog() -> None:
    text = flat(read(LEARN))
    model = catalog.get_model("qwen3-8b")
    exact = 2 * 36 * 8 * 128 * 2 * 4096
    assert catalog.kv_cache_gb(model, 4096) == pytest.approx(exact / 1e9)
    assert "2 × 36 × 8 × 128 × 2 × 4,096 = 603,979,776 bytes ≈ 0.60 GB" in text
    assert f"{exact:,}" == "603,979,776"

    # The rule of thumb for unknown families.
    unknown = ModelEntry(
        key="x/unknown-10b", display_name="Unknown 10B", family="Unknown", params_b=10.0, active_params_b=None,
        license="MIT", license_url="", hf_repo="x/unknown-10b", quant="Q4_K_M", file_size_gb=6.0, ollama_ref="",
        reasoning=False, blurb="",
    )
    assert catalog.kv_cache_gb(unknown, 1024) == pytest.approx(0.1 + 0.006 * 10)
    assert "(0.1 + 0.006 × parameters in billions) GB per 1,024 tokens" in text

    # Total memory for Qwen3 8B Q4_K_M on a CPU.
    need = model.file_size_gb + catalog.kv_cache_gb(model) + catalog.OVERHEAD_GB
    assert round(need, 1) == 6.2
    assert "5.03 + 0.60 + 0.6 ≈ **6.2 GB**" in text


def test_learn_and_readme_quote_the_memory_constants() -> None:
    learn, readme = flat(read(LEARN)), flat(read(README))
    for text in (learn, readme):
        assert f"{catalog.OVERHEAD_GB:g} GB" in text
        assert f"{catalog.GPU_COMPUTE_BUFFER_GB:g} GB" in text
        assert f"minus {catalog.GPU_VRAM_RESERVE_GB:g} GB" in text
        assert f"minus {catalog.OS_RAM_HEADROOM_GB:g} GB" in text
        for _verdict, limit in catalog.VERDICT_THRESHOLDS:
            assert f"up to {round(limit * 100)}%" in text
    assert f"the download plus {catalog.DISK_SPARE_GB:g} GB" in learn
    assert f"at least {round(catalog.PARTIAL_MIN_GPU_SHARE * 100)}% of the model" in learn
    assert f"stays at {catalog.UPGRADE_MIN_TOKENS_PER_S:g}+ tokens/sec" in learn


def test_learn_unified_memory_share_matches_specs() -> None:
    from gettowork import specs

    apple_gpu = getattr(specs, "_apple_gpu", None)
    if apple_gpu is None:  # pragma: no cover - the helper was renamed
        pytest.skip("specs._apple_gpu no longer exists; check LEARN.md's unified-memory share by hand")
    assert apple_gpu("Apple M2", 16.0).vram_gb == pytest.approx(16.0 * 0.70) == APPLE_M2.vram_gb
    assert apple_gpu("Apple M2 Max", 64.0).vram_gb == pytest.approx(64.0 * 0.75)
    assert "about 70% (75% on Macs with 64 GB or more)" in flat(read(LEARN))


def test_learn_speed_constants_match_perf() -> None:
    text = flat(read(LEARN))
    for placement in ("gpu", "unified", "cpu"):
        assert f"**{perf.EFFICIENCY[placement]:.2f}**" in text, placement
    ms = {k: f"{v * 1000:g} ms" for k, v in perf.OVERHEAD_S_PER_TOKEN.items()}
    assert f"({ms['gpu']} on a graphics card, {ms['unified']} on Apple Silicon, {ms['cpu']} on a CPU)" in text
    assert catalog.KV_READ_SHARE == 0.25 and "0.25 × 0.60" in text
    assert perf.speed_label(20) == "fast" and perf.speed_label(19.9) == "usable"
    assert perf.speed_label(8) == "usable" and perf.speed_label(3) == "slow" and perf.speed_label(2.9) == "very slow"
    assert "**20+ tokens/sec** *fast*, **8+** *usable*, **3+** *slow*" in text


def test_learn_ram_benchmark_description_matches_perf() -> None:
    assert perf._TARGET_TOTAL_BYTES == 512 * 1024 * 1024 and perf._MIN_THREAD_BYTES == 32 * 1024 * 1024
    assert perf._BENCH_ROUNDS == 3
    assert inspect.signature(perf.measure_ram_bandwidth).parameters["budget_s"].default == 0.3
    assert ("reading about 512 MB of memory (at least 32 MB per core, on all cores at once, like the engine) "
            "for 0.3 seconds, and takes the middle of three short runs") in flat(read(LEARN))


def test_learn_speed_worked_examples() -> None:
    text = flat(read(LEARN))
    model = catalog.get_model("qwen3-8b")
    per_token = model.file_size_gb + catalog.KV_READ_SHARE * catalog.kv_cache_gb(model)
    assert round(per_token, 2) == 5.18
    assert "5.03 + 0.25 × 0.60 = 5.18 GB" in text

    assert perf.estimate_gpu_bandwidth(RTX_3060) == 360
    assert perf.estimate_gpu_bandwidth(APPLE_M2) == 100
    gpu = perf.estimate_tokens_per_s(machine(gpus=[RTX_3060]), active_gb=5.18, placement="gpu")
    mac = perf.estimate_tokens_per_s(
        machine(os_name="Darwin", arch="arm64", unified_memory=True, gpus=[APPLE_M2]), active_gb=5.18, placement="unified"
    )
    cpu = perf.estimate_tokens_per_s(machine(), active_gb=5.18, placement="cpu")
    assert round(gpu) == 48 and "about 48 tokens/s" in text
    assert round(mac) == 15 and "about 15 tokens/s" in text
    assert round(cpu, 1) == 5.3 and "about 5.3 tokens/s" in text
    assert round(gpu / cpu) == 9 and "nine times faster" in text


def test_learn_moe_worked_example() -> None:
    text = flat(read(LEARN))
    model = catalog.get_model("gpt-oss-20b")
    assert (model.file_size_gb, model.params_b, model.active_params_b) == (12.1, 20.9, 3.6)
    assert catalog.MOE_ROUTING_OVERHEAD == 1.15
    share = model.active_params_b / model.params_b * catalog.MOE_ROUTING_OVERHEAD
    assert round(share, 3) == 0.198 and "3.6 ÷ 20.9 × 1.15 ≈ 0.198" in text
    per_token = model.file_size_gb * share + catalog.KV_READ_SHARE * catalog.kv_cache_gb(model)
    assert round(per_token, 2) == 2.45 and "≈ 2.45 GB" in text
    speed = perf.estimate_tokens_per_s(machine(), active_gb=per_token, placement="cpu")
    assert round(speed) == 11 and "≈ 11 tokens/s" in text
    dense = perf.estimate_tokens_per_s(machine(), active_gb=model.file_size_gb, placement="cpu")
    assert round(dense) == 2 and "about 2 tokens/sec" in text
    mistral = catalog.evaluate_fit(machine(), catalog.get_model("mistral-7b"))
    assert round(mistral.est_tokens_per_s) == 6 and "only manages about 6" in text
    assert round(math.sqrt(20.9 * 3.6), 1) == 8.7 and "√(20.9 × 3.6) ≈ 8.7 billion" in text


def test_learn_score_worked_example() -> None:
    text = flat(read(LEARN))
    fit = catalog.evaluate_fit(machine(), catalog.get_model("qwen3-4b"))
    assert (fit.quant, fit.download_gb, fit.est_memory_gb, fit.est_tokens_per_s) == ("Q4_K_M", 2.5, 3.5, 10.3)
    assert fit.score == pytest.approx(42.47, abs=0.01)
    assert "picks Q4_K_M (2.5 GB), needs 3.5 GB of the 13.5 GB budget, and estimates 10.3 tokens/sec" in text
    assert "The game gets 42.47" in text
    plan = catalog._plan(machine(), fit.model, fit.quant, fit.download_gb)
    turn = catalog.turn_tokens_per_s(plan.tokens_per_s, "cpu")
    assert round(turn, 2) == 6.46 and "a turn speed of about 6.5" in text
    parts = [
        catalog.W_QUALITY * math.log2(1 + 4.02) * catalog.quant_quality("Q4_K_M"),
        catalog.W_SPEED * math.log(6.46 / 3) / math.log(15 / 3) - catalog.SLOW_PENALTY_PER_TOKEN * (8 - 6.46),
        catalog.W_HEADROOM * (1 - 3.5 / 13.5),
        catalog.BONUS_CURATED + catalog.BONUS_TRUSTED + catalog.BONUS_INSTRUCT,
    ]
    assert [round(p, 1) for p in parts] == [26.8, 5.7, 4.4, 5.5]
    assert round(sum(round(p, 1) for p in parts), 1) == 42.4 and "≈ 42.4" in text
    assert fit.est_tokens_per_s < catalog.THINKING_MIN_TOKENS_PER_S and "no thinking bonus under 20 tok/s" in text
    assert round(plan.tokens_per_s, 2) == 10.34 and "a speed of 10.34 tokens/sec" in text
    assert catalog.TURN_PROMPT_TOKENS == 1200 and catalog.TURN_ANSWER_TOKENS == 250
    assert catalog.PREFILL_SPEEDUP["cpu"] == 8 and catalog.PREFILL_SPEEDUP["gpu"] == 60
    assert "turn speed = 250 ÷ (250 ÷ speed + 1200 ÷ (speed × prefill speed-up))" in text


def test_learn_scoring_table_matches_catalog_weights() -> None:
    text = flat(read(LEARN))
    c = catalog
    expected = [
        f"{c.W_QUALITY:g} × log2(1 + effective billions of parameters)",
        f"up to {c.W_SPEED:g} points: 0 at {c.SPEED_ZERO_AT:g} tokens/s, full marks at {c.SPEED_FULL_AT:g}",
        f"minus {c.SLOW_PENALTY_PER_TOKEN:g} per token/s below 8; minus {c.VERY_SLOW_PENALTY:g} below 3",
        f"{c.W_HEADROOM:g} × (1 − needed ÷ budget); or −{c.TIGHT_PENALTY:g} if \"tight\"; −{c.PARTIAL_PENALTY:g} more",
        f"up to {c.W_POPULARITY:g} points: {c.W_POPULARITY:g} × log10(1 + downloads) ÷ 6",
        f"+{c.BONUS_CURATED:g} in the built-in list, +{c.BONUS_TRUSTED:g} trusted publisher, "
        f"+{c.BONUS_REASONING:g} shows its reasoning, +{c.BONUS_INSTRUCT:g} instruction-tuned",
        f"−{c.PENALTY_GATED:g} gated, −{c.PENALTY_NOT_PERMISSIVE:g} not Apache-2.0/MIT, "
        f"−{c.BIG_DOWNLOAD_PENALTY_PER_GB:g} per GB of download above {c.BIG_DOWNLOAD_GB:g} GB",
    ]
    for snippet in expected:
        assert snippet in text, snippet


def test_learn_discovery_description_matches_hf_discovery() -> None:
    text = flat(read(LEARN))
    for publisher in catalog.TRUSTED_PUBLISHERS:
        assert publisher in text, publisher
    assert f"Between {hf_discovery.MIN_PARAMS_B:g} and {hf_discovery.MAX_PARAMS_B:g} billion parameters" in text
    assert f"at least {hf_discovery.MIN_DOWNLOADS_UNTRUSTED:,} downloads" in text
    ttl = inspect.signature(hf_discovery.discover_models).parameters["ttl_hours"].default
    assert ttl == 72 and "saves the result for 3 days" in text
    assert "cached for 3 days" in flat(read(README))


def test_learn_build_table_matches_plan_variants() -> None:
    def plan(**kw):
        return [v.name for v in runtime_install.plan_variants(machine(**kw))]

    nvidia = GPUInfo("NVIDIA GeForce RTX 4070", "nvidia", 12.0, driver_version="581.15")
    amd = GPUInfo("AMD Radeon RX 7800 XT", "amd", 16.0)
    assert plan(os_name="Darwin", arch="arm64", unified_memory=True, gpus=[APPLE_M2]) == ["metal", "cpu"]
    assert plan(os_name="Darwin", arch="x86_64") == ["cpu"]
    assert plan(os_name="Windows", gpus=[nvidia]) == ["cuda-13", "cuda-12", "vulkan", "cpu"]
    assert plan(os_name="Windows", gpus=[amd]) == ["vulkan", "cpu"]
    assert plan(os_name="Windows", arch="arm64") == ["cpu"]
    assert plan(os_name="Windows", arch="arm64", gpus=[nvidia]) == ["cuda-13", "cpu"]
    assert plan(gpus=[nvidia], cpu_flags=["avx2", "vulkan"]) == ["cuda-13", "cuda-12", "vulkan", "cpu"]
    assert plan(gpus=[amd], cpu_flags=["avx2", "vulkan"]) == ["vulkan", "cpu"]
    assert plan(gpus=[amd], cpu_flags=["avx2", "no-vulkan"]) == ["cpu"]
    text = flat(read(LEARN))
    assert "CUDA 13 (driver 580+) and/or CUDA 12 (driver 525+), then Vulkan, then CPU" in text
    assert "Windows on ARM | CUDA 13 (with an NVIDIA GPU and driver 580+), then CPU" in text
    readme = flat(read(README))
    assert "about version 525 or newer for CUDA 12, 580 or newer for CUDA 13" in readme


def test_learn_and_readme_describe_the_speed_calibration() -> None:
    from gettowork import setup_flow

    learn, readme = flat(read(LEARN)), flat(read(README))
    low, high = setup_flow.CALIBRATION_RANGE
    assert f"between {low:g} and {high:g}" in learn
    assert "measured speed ÷ estimated speed" in learn
    # The worked example: a 1.25 correction on the RTX 3060 turns 360 GB/s into 450 GB/s.
    gpu_specs = machine(gpus=[RTX_3060])
    tuned = setup_flow.calibrate_specs(gpu_specs, "gpu", 45 / 36)
    assert perf.bandwidth_for(tuned, "gpu")[0] == 450
    before = perf.estimate_tokens_per_s(gpu_specs, active_gb=6.9, placement="gpu")
    after = perf.estimate_tokens_per_s(tuned, active_gb=6.9, placement="gpu")
    assert 1.2 < after / before < 1.25 and "a little under 25%" in learn
    assert "from 360 to 450 GB/s" in learn
    assert setup_flow.calibrate_specs(gpu_specs, "partial", 1.5) is gpu_specs
    assert "GPU+RAM splits aren't calibrated" in learn
    # The fingerprint is made of the parts LEARN.md lists.
    fingerprint = setup_flow.hardware_fingerprint(gpu_specs)
    for part in (gpu_specs.os_name, gpu_specs.cpu_name, RTX_3060.name):
        assert part in fingerprint
    assert "operating system, CPU, RAM size and graphics cards" in learn
    assert "remembers the difference for this computer" in readme
    assert f"under {setup_flow.SLOW_TOKENS_PER_S:g} tokens/sec" in readme


def test_readme_model_menu_words_exist_in_setup_flow() -> None:
    source = read(SRC / "setup_flow.py")
    readme = read(README)
    for word in ("more", "refresh", "custom", "mock", "learn"):
        assert f'"{word}"' in source, word
        assert f"`{word}`" in readme, word
    assert "_WHY_RE" in source and "`why 2`" in readme and "`why 2`" in read(LEARN)


# ---------------------------------------------------------------------------
# README.md: commands, flags, locations, URLs
# ---------------------------------------------------------------------------


def _contract_cli_section() -> str:
    text = read(ARCHITECTURE)
    start = text.index("### cli.py")
    end = text.index("\n## ", start)
    return text[start:end]


def test_readme_flag_table_matches_the_cli_contract() -> None:
    readme_flags = set(re.findall(r"^\| `(--[a-z][a-z-]*)", read(README), flags=re.MULTILINE))
    contract_flags = set(re.findall(r"`(--[a-z][a-z-]*)", _contract_cli_section()))
    assert readme_flags == contract_flags
    choices = re.search(r"--backend \{([a-z,]+)\}", _contract_cli_section()).group(1)
    assert f"`--backend {{{choices}}}`" in read(README)


def test_readme_flags_exist_in_cli_once_it_is_written() -> None:
    cli = SRC / "cli.py"
    if not cli.exists():
        pytest.skip("cli.py hasn't been written yet")
    source = read(cli)
    for flag in re.findall(r"^\| `(--[a-z][a-z-]*)", read(README), flags=re.MULTILINE):
        assert f'"{flag}"' in source or f"'{flag}'" in source, f"{flag} is documented but not in cli.py"


def test_readme_play_from_source_commands() -> None:
    text = read(README)
    assert "pip install -e ." in text
    assert "gettowork --mock" in text
    assert "python -m gettowork" in text and "python -m gettowork.launcher" in text
    requires = re.search(r'requires-python\s*=\s*">=\s*([\d.]+)"', read(PYPROJECT)).group(1)
    assert f"Python {requires} or newer" in text
    # The two commands pip installs are the ones the README tells people to type.
    pyproject = read(PYPROJECT)
    assert re.search(r'^gettowork = "gettowork\.cli:main"', pyproject, re.MULTILINE)
    assert re.search(r'^gettowork-gui = "gettowork\.launcher:gui_main"', pyproject, re.MULTILINE)
    assert "gettowork-gui" in text


def test_readme_windows_commands_work_in_powershell_and_command_prompt() -> None:
    """PowerShell (Windows Terminal's default) runs nothing from the current folder without .\\, and Command
    Prompt reads "./" as a switch: the Windows forms must be .\\GetToWork.exe / .\\gettowork-cli.exe."""
    text = read(README)
    assert "`.\\GetToWork.exe --jev`" in text and "`.\\gettowork-cli.exe --mock`" in text
    assert "`GetToWork.exe --jev`" not in text
    windows_lines = [line for line in text.splitlines() if "**Windows:**" in line and "`" in line]
    assert windows_lines and not any("`./" in line for line in windows_lines)


def test_readme_names_smart_app_control_for_unsigned_windows_builds() -> None:
    """Smart App Control blocks unsigned programs even from Steam (0x11C7): the README may not promise that
    Steam builds avoid the unsigned-build problem, and says what works."""
    text = flat(read(README))
    assert "Steam builds don't have this problem" not in text
    for words in ("Smart App Control", "0x11C7", "Ollama", "llama-server.exe"):
        assert words in text, words
    steam = flat(read(STEAM_README))
    assert "Smart App Control" in steam and "packaging/sign_windows.py" in read(STEAM_README).replace("../", "packaging/")


def test_readme_test_build_instructions_match_the_build() -> None:
    """The README's "Playing the test builds" names what the build workflow and assemble.py really make."""
    readme = flat(read(README))
    section = readme[readme.index("## Playing the test builds"):readme.index("## Playing from source")]
    workflow = read(ROOT / ".github" / "workflows" / "build.yml")
    for target in ("windows-x64", "macos-arm64", "linux-x64"):
        assert f"target: {target}" in workflow, target
    # The game archives are uploaded as they are (the artifact takes the archive's name)...
    assert "archive: false" in workflow and "one layer" in section and "two layers" not in section
    assert "name: gettowork-python-package" in workflow and "`gettowork-python-package`" in section
    retention = re.findall(r"retention-days:\s*(\d+)", workflow)
    assert sorted(retention) == ["3", "7"] and "kept for **3 days**" in section and "the Python package for 7" in section
    assert "Extract All" in section  # Windows: the game can't start from inside the zip

    assemble = _packaging_module("assemble")
    for os_key, arch in (("windows", "x64"), ("macos", "arm64"), ("linux", "x64")):
        name = assemble.archive_name(os_key, arch, "<version>")
        assert f"`{name}`" in section, name
    assert "`GetToWork.exe`" in section and f"`{assemble.MAC_APP}`" in section and "`GetToWork`" in section
    assert f"`{assemble.CLI_NAME}`" in section and f"`{assemble.CLI_NAME}.exe`" in section
    assert f"Contents/MacOS/{assemble.CLI_NAME}" in section
    assert "More info" in section and "Run anyway" in section  # Windows SmartScreen
    assert "Open Anyway" in section and "xattr -dr com.apple.quarantine" in section  # macOS Gatekeeper
    assert "./GetToWork" in section  # Linux


def _packaging_module(name: str):
    """Import one of the build scripts in packaging/ (they aren't part of the installed game)."""
    import importlib.util

    module_name = f"_docs_packaging_{name}"
    if module_name in sys.modules:
        return sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(module_name, ROOT / "packaging" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module  # dataclasses look their module up while the class is made
    spec.loader.exec_module(module)
    return module


def test_readme_window_controls_match_the_window() -> None:
    from gettowork.gui import app

    text = flat(read(README))
    section = text[text.index("### In the game's window"):text.index("## How does it know which models fit?")]
    assert app.EXIT_HINT.lower() in section.lower()
    source = read(SRC / "gui" / "app.py")
    for key in ("<F11>", "<Escape>", "<Up>", "<Down>", "<Prior>", "<Next>"):
        assert key in source, key
    assert '"equal": +1' in source and '"minus": -1' in source and '"0": 0' in source
    assert "Ctrl + = / Ctrl + - / Ctrl + 0" in section and "F11" in section and "Escape" in section
    assert "Keyboard" in section and 'text="Keyboard"' in source
    assert app.default_export_dir().name == "Get To Work" or not (Path.home() / "Documents").is_dir()
    assert "`Documents/Get To Work`" in text


def test_readme_troubleshooting_covers_the_window_and_steam_deck() -> None:
    from gettowork.gui import app

    readme = flat(read(README))
    assert f"`logs/{app.CRASH_FILE}`" in readme
    assert "STEAM + X" in readme and "X11" in readme and "XWayland" in readme
    assert "python3-tk" in readme
    # The "engine is missing" advice is the game's own message.
    missing = runtime_install.ENGINE_MISSING_MESSAGE
    assert "Verify integrity" in missing and "Verify integrity" in readme
    assert "packaging/steam/README.md" in readme


def _source_text() -> str:
    return "\n".join(read(p) for p in sorted(SRC.rglob("*.py")))


def test_readme_env_vars_are_really_read_by_the_code() -> None:
    readme, source = read(README), _source_text()
    assert jev.JEV_API_KEY_ENV == "TYPESAFE_API_KEY"
    for var in ("TYPESAFE_API_KEY", "GETTOWORK_HOME", "GETTOWORK_MODELS_DIR", "GITHUB_TOKEN", "OLLAMA_HOST"):
        assert var in readme, var
        assert var in source, f"README documents {var}, but no code reads it"


def test_readme_urls_match_the_code() -> None:
    text = read(README)
    for url in (jev.JEV_HOME_URL, jev.JEV_DOCS_URL, runtime_install.OLLAMA_DOWNLOAD_URL, runtime_install.LLAMA_CPP_URL):
        assert url in text, url


def test_readme_storage_locations_match_config(monkeypatch, tmp_path) -> None:
    text = read(README)
    monkeypatch.delenv("GETTOWORK_HOME", raising=False)
    monkeypatch.delenv("GETTOWORK_MODELS_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)

    monkeypatch.setattr(config.platform, "system", lambda: "Linux")
    assert config.config_dir() == Path.home() / ".config" / "gettowork"
    assert "`~/.config/gettowork`" in text and "`$XDG_CONFIG_HOME/gettowork`" in text

    monkeypatch.setattr(config.platform, "system", lambda: "Darwin")
    assert config.config_dir() == Path.home() / "Library" / "Application Support" / "GetToWork"
    assert "`~/Library/Application Support/GetToWork`" in text

    monkeypatch.setattr(config.platform, "system", lambda: "Windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setenv("APPDATA", str(tmp_path / "Roaming"))
    assert config.config_dir() == tmp_path / "Local" / "GetToWork"  # big files stay out of the roaming profile
    assert "`%LOCALAPPDATA%\\GetToWork`" in text
    (tmp_path / "Roaming" / "GetToWork").mkdir(parents=True)  # made by an older version...
    assert config.config_dir() == tmp_path / "Roaming" / "GetToWork"  # ...keeps being used
    assert "`%APPDATA%\\GetToWork` keeps being used" in text

    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path / "home"))
    assert config.models_dir() == tmp_path / "home" / "models"
    assert config.runtime_dir() == tmp_path / "home" / "runtime"
    assert config.cache_dir() == tmp_path / "home" / "cache"
    assert config.Settings().path.name == "settings.json"
    assert hf_discovery.default_cache_path() == tmp_path / "home" / "cache" / hf_discovery.CACHE_FILENAME
    assert download.model_folder("unsloth/Qwen3-4B-GGUF") == tmp_path / "home" / "models" / "unsloth--Qwen3-4B-GGUF"
    for snippet in ("`settings.json`", "`models/`", "models/unsloth--Qwen3-4B-GGUF/", "`runtime/llama.cpp/`",
                    "`runtime/logs/`", f"`cache/{hf_discovery.CACHE_FILENAME}`", "llama-server.log"):
        assert snippet in text, snippet
    assert "llama-server.log" in read(SRC / "backends" / "llamaserver.py")
    assert "runtime_dir()/llama.cpp/" in inspect.getsource(runtime_install)
    assert f"{review.EXPORT_PREFIX}1.json" in text


def test_readme_project_layout_lists_every_module() -> None:
    layout = next(body for lang, body in code_blocks(read(README)) if body.startswith("src/gettowork/"))
    for path in sorted(SRC.glob("*.py")):
        if path.name != "__init__.py":
            assert path.name in layout, f"{path.name} is missing from the README's project layout"
    for path in sorted((SRC / "backends").glob("*.py")):
        if path.name not in ("__init__.py", "base.py"):
            assert path.name in layout, f"backends/{path.name} is missing from the README's project layout"


def test_readme_how_to_play_matches_the_game() -> None:
    text = read(README)
    for word in game.QUIT_WORDS:
        assert f"`{word}`" in text, word
    assert "`help`" in text and "help" in game.HELP_WORDS
    default_target = inspect.signature(game.Game.__init__).parameters["target"].default
    assert f"Reach {default_target} steps" in text and f"(default {default_target})" in text
    default_chars = inspect.signature(game.Game.__init__).parameters["max_input_chars"].default
    assert f"trimmed to {default_chars} characters" in flat(read(LEARN))


def test_readme_disclaimer_covers_the_essentials() -> None:
    text = flat(read(README))
    disclaimer = text[text.index("DISCLAIMER"):]
    for phrase in ('"AS IS"', "without warranty", "Estimates can be wrong", "not affiliated with, endorsed by",
                   "TypeSafe AI", "Hugging Face", "llama.cpp", "Ollama", "model author", "trademarks",
                   "responsible for complying", "Jev may cost money", "unpredictable"):
        assert phrase in disclaimer, phrase


def test_readme_privacy_section_names_every_outside_service() -> None:
    text = flat(read(README))
    section = text[text.index("## Privacy"):text.index("## Troubleshooting")]
    for phrase in ("huggingface.co", "api.github.com", "api.typesafe.ai", "GET /v1/models", "no telemetry"):
        assert phrase in section, phrase


# ---------------------------------------------------------------------------
# NOTICE.md and CONTRIBUTING.md
# ---------------------------------------------------------------------------


def _pyproject_dependencies() -> dict[str, list[str]]:
    """{"": runtime deps, "<extra>": optional deps} - parsed without tomllib (Python 3.10)."""
    text = read(PYPROJECT)
    if sys.version_info >= (3, 11):
        import tomllib

        project = tomllib.loads(text)["project"]
        groups = {"": project["dependencies"], **project.get("optional-dependencies", {})}
    else:  # pragma: no cover - exercised on Python 3.10 in CI
        groups = {"": re.findall(r'"([^"]+)"', re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M).group(1))}
        extras = re.search(r"\[project\.optional-dependencies\](.*?)(?:\n\[|\Z)", text, re.S).group(1)
        for name, body in re.findall(r"^(\w+)\s*=\s*\[(.*?)\]", extras, re.M | re.S):
            groups[name] = re.findall(r'"([^"]+)"', body)
    return {group: [re.split(r"[<>=!~\[; ]", dep, maxsplit=1)[0] for dep in deps] for group, deps in groups.items()}


def test_notice_lists_every_python_dependency_with_its_license() -> None:
    lines = read(NOTICE).splitlines()
    licenses = {"rich": "MIT", "psutil": "BSD-3-Clause", "huggingface_hub": "Apache-2.0",
                "llama-cpp-python": "MIT", "pytest": "MIT"}
    for group, names in _pyproject_dependencies().items():
        for name in names:
            assert name in licenses, f"new dependency {name!r} ({group or 'runtime'}): add it to NOTICE.md and this test"
            assert any(name in line and licenses[name] in line for line in lines), name


def test_notice_covers_runtime_downloads_services_and_references() -> None:
    lines = read(NOTICE).splitlines()
    text = flat(read(NOTICE))
    for name, license_id in (("llama.cpp", "MIT"), ("Ollama", "MIT"), ("TypeSafe SDK", "MIT")):
        assert any(name in line and license_id in line for line in lines), name
    for phrase in ("not bundled", "NVIDIA's license terms", "Hugging Face Hub", "GitHub", "TypeSafe AI",
                   "not affiliated with, endorsed by or sponsored by", "\"AS IS\""):
        assert phrase in text, phrase


def test_contributing_explains_setup_and_names_real_knobs() -> None:
    text = read(CONTRIBUTING)
    for phrase in ('pip install -e ".[dev]"', "python -m pytest", "--mock", "Apache-2.0", "MIT",
                   "Contributor Covenant", "tests/test_docs.py"):
        assert phrase in text, phrase
    for name in ("_UNSAFE_RE", "_NOT_CHAT_WORDS", "_NOT_CHAT_PIPELINES", "_BASE_WORDS", "_INSTRUCT_RE",
                 "_CHAT_FAMILIES", "MIN_PARAMS_B", "MAX_PARAMS_B", "MIN_DOWNLOADS_UNTRUSTED", "_ALWAYS_THINKS_RE",
                 "_SWITCHABLE_THINKS_RE", "_THINK_SWITCH_RE", "_FORCED_THINK_RE", "_FAMILIES",
                 "CACHE_SCHEMA_VERSION", "RULES_VERSION"):
        assert hasattr(hf_discovery, name) and f"`{name}`" in text, name
    for name in ("MODEL_CATALOG", "TRUSTED_PUBLISHERS", "PERMISSIVE_LICENSES", "_KV_SHAPES", "OVERHEAD_GB", "W_SPEED"):
        assert hasattr(catalog, name) and name in text, name
    assert hasattr(perf, "EFFICIENCY") and "`EFFICIENCY`" in text
    assert "EXPECTED_REPOS" in read(ROOT / "tests" / "test_catalog.py") and "`EXPECTED_REPOS`" in text


def test_contributing_injection_table_matches_real_signatures() -> None:
    from gettowork import cli, setup_flow
    from gettowork.backends.llamacpp import LlamaCppBackend
    from gettowork.backends.llamaserver import LlamaServerBackend
    from gettowork.backends.ollama import OllamaBackend

    def params(func) -> set[str]:
        return set(inspect.signature(func).parameters)

    from gettowork.gui import app

    text = read(CONTRIBUTING)
    expectations = {
        UI.__init__: {"console", "input_fn", "secret_fn", "open_url_fn", "choices_fn"},
        app.run_gui: {"game_main"},
        app.GameWindow.__init__: {"game_main", "opener", "env"},
        hf_discovery.discover_models: {"api", "cache_path", "clock"},
        download.download_gguf: {"hf_api", "hf_download"},
        runtime_install.ensure_llama_server: {"http", "runtime_root"},
        LlamaServerBackend.__init__: {"http", "popen", "installer", "downloader", "sleep", "clock", "log_dir"},
        OllamaBackend.__init__: {"http"},
        LlamaCppBackend.__init__: {"llama_factory", "downloader"},
        jev.JevClient.__init__: {"transport"},
        cli.main: {"ui", "services"},
    }
    services = {f.name for f in dataclasses.fields(setup_flow.SetupServices)}
    for name in ("detect_specs", "discover_models", "make_backend", "installed_runtimes", "custom_entry"):
        assert name in services and f"{name}=" in text, name
    for func, names in expectations.items():
        missing = names - params(func)
        assert not missing, f"{func.__qualname__} has no parameter(s) {missing}, but CONTRIBUTING.md says so"
        for name in names:
            assert f"{name}=" in text, name
    assert "hides_input" in params(UI.__init__) and "`hides_input=True`" in text
    # The environment variables CONTRIBUTING tells test writers about are the ones distribution.py reads.
    from gettowork import distribution

    for var in (distribution.ENV_DISTRIBUTION, distribution.ENV_ENGINE_DIR, distribution.ENV_ALLOW_DOWNLOAD):
        assert f"`{var}`" in text, var
    assert "refresh" in params(distribution.load) and "distribution.load(refresh=True)" in text


# ---------------------------------------------------------------------------
# The CI workflow
# ---------------------------------------------------------------------------


def test_ci_workflow_runs_pytest_on_every_os_and_python() -> None:
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load(read(CI_WORKFLOW))
    triggers = data.get("on", data.get(True))  # PyYAML reads the bare key `on` as True
    assert "push" in triggers and "pull_request" in triggers

    (job,) = data["jobs"].values()
    matrix = job["strategy"]["matrix"]
    assert set(matrix["os"]) == {"ubuntu-latest", "windows-latest", "macos-latest"}
    assert matrix["python-version"] == ["3.10", "3.12"], "versions must be quoted strings (3.10 != 3.1)"
    assert job["strategy"]["fail-fast"] is False

    uses = [step.get("uses", "") for step in job["steps"]]
    assert "actions/checkout@v4" in uses and "actions/setup-python@v5" in uses
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert 'pip install -e ".[dev]"' in runs
    assert "llamacpp" not in runs, "CI must not need the optional llama-cpp-python backend"
    assert "python -m pytest" in runs


def test_contributing_describes_the_safety_workflow_and_the_build() -> None:
    from gettowork import prompts, safety, safety_terms

    text = read(CONTRIBUTING)
    for name in ("check_text", "soften", "check_player_input"):
        assert hasattr(safety, name) and (f"safety.{name}" in text or f"`{name}`" in text), name
    assert hasattr(prompts, "safety_retry_messages") and "prompts.safety_retry_messages" in text
    assert "ROT13" in text and "rot13" in text and "tests/test_safety.py" in text
    assert "ROT13" in (safety_terms.__doc__ or "")
    assert "notices.STEAM_AI_DISCLOSURE" in text
    for script in ("fetch_engine.py", "gettowork.spec", "collect_licenses.py", "assemble.py", "smoke_test.sh",
                   "engine_isolation_check.sh", "sign_windows.py"):
        assert (ROOT / "packaging" / script).is_file(), script
        assert f"packaging/{script}" in text, script
    assert "--gui-selftest" in text and "gettowork_entry.py" not in text
    # A maintainer note about the license of future versions, and LICENSE itself still MIT.
    assert "MIT License" in read(ROOT / "LICENSE")
    assert "license of future versions" in text.lower() and "stay MIT" in text


def test_notice_covers_what_ships_inside_the_game_builds() -> None:
    lines = read(NOTICE).splitlines()
    text = flat(read(NOTICE))
    collect = _packaging_module("collect_licenses")
    assemble = _packaging_module("assemble")
    assert assemble.LICENSES_FILE == "THIRD_PARTY_LICENSES.txt" and "`THIRD_PARTY_LICENSES.txt`" in text
    assert "packaging/collect_licenses.py" in text
    # Each bundled component is named with its license on one table row.
    for name, license_words in (("Tcl/Tk", "BSD"), ("PyInstaller", "GPL"), ("Python", "PSF-2.0"),
                                ("llama.cpp", "MIT")):
        assert any(name in line and license_words in line for line in lines), name
    assert "bootloader exception" in text and "BSD" in collect.TCL_TK_NOTICE
    assert "**Bundled** in the game builds" in text
    # The engine builds NOTICE names are the ones the build workflow bundles.
    workflow = read(BUILD_WORKFLOW)
    assert workflow.count("engines: vulkan,cpu") == 2 and "engines: metal" in workflow
    assert "Vulkan and CPU builds on Windows and Linux, the Metal build on macOS" in text


def test_readme_ai_content_section_matches_the_filter() -> None:
    from gettowork import game, notices, safety

    readme = flat(read(README))
    section = readme[readme.index("## AI content and the family-friendly filter"):readme.index("## Privacy")]
    assert game.FAMILY_FRIENDLY_REFUSAL in section
    for label in ("sexual content", "self-harm", "graphic gore"):
        assert label in safety.CATEGORY_LABELS.values() and label in section, label
    assert "slurs and hate speech" in section and "hard drugs" in section
    assert safety.soften("damn") == "d***" and '"d***"' in section
    assert "hidden by the family-friendly filter" in safety.hidden_note(safety.SafetyVerdict(ok=False, category="gore"))
    assert '"hidden by the family-friendly filter"' in section
    for word in ("Scunthorpe", "assassin", "classic"):
        assert safety.check_text(f"a {word} morning").ok and word in section, word
    # Steam's overlay can't open over the game's Tk window (outside a Deck's Game Mode):
    # the window's own button is the reporting path the notice and the README promise.
    assert "Shift+Tab" not in notices.REPORT_HOW and notices.REPORT_BUTTON in notices.REPORT_HOW
    assert notices.REPORT_BUTTON in section and "Discussions" in section
    assert "`STEAM_AI_DISCLOSURE`" in section and hasattr(notices, "STEAM_AI_DISCLOSURE")
    assert "abliterated" in section


def test_readme_privacy_section_names_the_steam_keyboard_request() -> None:
    from gettowork.gui import app

    text = flat(read(README))
    section = text[text.index("## Privacy"):text.index("## Troubleshooting")]
    assert f"`{app.STEAM_KEYBOARD_URL}`" in section
    assert "their engine is built in" in section  # built games never ask GitHub for the engine


def test_distribution_contract_matches_the_build() -> None:
    from gettowork import distribution, launcher, setup_flow
    from gettowork.gui import app

    doc = read(DISTRIBUTION)
    flat_doc = flat(doc)
    assemble = _packaging_module("assemble")
    # The programs and archives.
    assert f"`{assemble.CLI_NAME}(.exe)`" in doc and f"`{assemble.GUI_NAME}(.exe)`" in doc
    assert "Contents/MacOS/gettowork-cli" in doc and f"`{assemble.MAC_APP}`" in doc
    spec = read(ROOT / "packaging" / "gettowork.spec")
    assert f'CLI_NAME = "{assemble.CLI_NAME}"' in spec and f'GUI_NAME = "{assemble.GUI_NAME}"' in spec
    for os_key, arch in (("windows", "x64"), ("macos", "arm64"), ("linux", "x64")):
        assert f"`{assemble.archive_name(os_key, arch, '<version>')}`" in doc
    # distribution.json: the example has exactly the keys assemble.py writes.
    example = find_json(doc, lambda b: "engine_downloads" in b)
    assert set(example) == set(assemble.distribution_info("b1", "0.2.0", "sha"))
    assert example["engine_downloads"] is False and example["engine_dir"] == assemble.ENGINE_DIR
    # The Distribution dataclass fields and the environment variables.
    fields = {f.name for f in dataclasses.fields(distribution.Distribution)}
    for name in fields:
        assert f"    {name}:" in doc, name
    for var in (distribution.ENV_DISTRIBUTION, distribution.ENV_ENGINE_DIR, distribution.ENV_ALLOW_DOWNLOAD,
                app.FULLSCREEN_ENV, app.SELFTEST_OUT_ENV, app.SELFTEST_TIMEOUT_ENV):
        assert var in doc, var
    # Messages and names quoted from the code.
    assert flat(runtime_install.ENGINE_MISSING_MESSAGE) in flat_doc
    assert runtime_install.BUNDLED_UNUSABLE_FILE in doc
    assert "Built into the game (" in read(SRC / "setup_flow.py") and "Built into the game (llama.cpp" in doc
    for quoted in (app.EXIT_HINT, app.SELFTEST_SUCCESS_TEXT, launcher.SELFTEST_FLAG, launcher.CLOSE_PROMPT,
                   app.CRASH_FILE, f"{app.SELFTEST_TIMEOUT_S:g} s", f"{app.CLOSE_WAIT_S:g} s",
                   f"{app.FULLSCREEN_FONT_SIZE}-point", setup_flow.__name__.rsplit(".", 1)[-1]):
        assert quoted in flat_doc, quoted
    assert app.SelftestPlayer().max_answers == 120 and "after 120 answers" in flat_doc
    # Launch options agree with the Steam checklist.
    steam = read(STEAM_README)
    for launch in ("GetToWork\\GetToWork.exe", "Get To Work.app", "GetToWork/GetToWork"):
        assert f"`{launch}`" in steam and f"`{launch}`" in doc, launch


def test_distribution_contract_matches_the_workflows() -> None:
    doc = flat(read(DISTRIBUTION))
    build = read(BUILD_WORKFLOW)
    for path in ("packaging/**", "src/gettowork/gui/**", "src/gettowork/launcher.py",
                 "src/gettowork/distribution.py", ".github/workflows/build.yml"):
        assert f'- "{path}"' in build, path
    for runner in ("windows-latest", "ubuntu-22.04", "macos-latest"):
        assert f"os: {runner}" in build and runner in doc, runner
    assert "if: github.event_name != 'pull_request'" in build and "not on pull requests" in doc
    assert "archive: false" in build and "`archive: false`" in doc
    assert "--tag pinned" in build and "`packaging/llama_cpp_tag.txt`" in doc
    assert "retention-days: 3" in build and "retention 3 days" in doc
    ci = read(CI_WORKFLOW)
    assert "xvfb-run -a python -m pytest" in ci and "`xvfb-run`" in doc
    assert "github.event_name != 'workflow_dispatch' && 'macos-latest'" in ci and "macOS counts 10" in doc
    live = read(ROOT / ".github" / "workflows" / "live-check.yml")
    assert "GETTOWORK_ALLOW_ENGINE_DOWNLOAD" in live and "GETTOWORK_ENGINE_DIR" in live
    assert "gettowork --list-models --refresh-models" in live and "`gettowork --list-models --refresh-models`" in doc


def test_architecture_documents_the_distribution_modules() -> None:
    text = read(ARCHITECTURE)
    start = text.index("## Distribution: the game window, the built game and safety (as built)")
    section = text[start:text.index("\n## ", start + 1)]
    for heading in ("### distribution.py", "### gui/", "### ui.py (additive)", "### launcher.py",
                    "### safety.py"):
        assert heading in section, heading
    layout = next(body for _lang, body in code_blocks(text) if body.startswith("src/gettowork/"))
    for path in sorted(SRC.glob("*.py")):
        if path.name != "__init__.py":
            assert path.name in layout, f"{path.name} is missing from ARCHITECTURE.md's package layout"
    for path in sorted((SRC / "gui").glob("*.py")):
        assert path.name in layout, f"gui/{path.name} is missing from ARCHITECTURE.md's package layout"
    assert "(DISTRIBUTION.md)" in text


def test_readme_project_layout_lists_the_window_package() -> None:
    layout = next(body for lang, body in code_blocks(read(README)) if body.startswith("src/gettowork/"))
    assert "gui/" in layout
    for path in sorted((SRC / "gui").glob("*.py")):
        if path.name != "__init__.py":
            assert path.name in layout, f"gui/{path.name} is missing from the README's project layout"
    for folder in ("packaging/", "steam/", "docs/DISTRIBUTION.md", "assets/icon.png"):
        assert folder in layout, folder
