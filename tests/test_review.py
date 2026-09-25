"""Tests for the end-of-game review and transcript export (scripted input, temp folders only)."""

from __future__ import annotations

import errno
import io
import json
import math
from pathlib import Path

import pytest
from rich.console import Console

from gettowork.review import (
    export_transcript,
    next_export_number,
    run_review,
    summary_to_dict,
    summary_to_markdown,
)
from gettowork.types import GameSummary, JevExchange, JevVerdict, LLMResult, RoundRecord
from gettowork.ui import UI

try:
    from gettowork.backends.mock import MockBackend
except ImportError:  # pragma: no cover - written by another builder
    MockBackend = None

SECRET = "tsk_live_SUPERSECRETKEY1234567890abcd"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class Script:
    def __init__(self, answers):
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        if not self.answers:
            raise AssertionError(f"Unexpected prompt: {prompt!r}")
        item = self.answers.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        return item


def make_ui(answers):
    console = Console(file=io.StringIO(), width=110)
    script = Script(answers)
    return UI(console=console, input_fn=script, secret_fn=script, open_url_fn=lambda url: True), script, console


def output(console):
    return console.file.getvalue()


def llm_result(text, reasoning=None):
    return LLMResult(
        text=text,
        reasoning=reasoning,
        model="test-model",
        backend="scripted",
        elapsed_s=0.5,
        messages=[{"role": "system", "content": "TASK: test"}, {"role": "user", "content": "hi"}],
        raw={"content": text},
    )


def exchange(*, status=200, auth="Bearer ****abcd", response=None, error=None):
    return JevExchange(
        url="https://api.typesafe.ai/v1/systemone",
        request_headers={"Authorization": auth, "Content-Type": "application/json", "Accept": "application/json"},
        request_body={
            "state": {"current_challenge": "A goose", "player_plan": "bribe it"},
            "model": "jev-latest",
            "questions": {"made_progress": {"type": "noul", "instructions": "Did they make progress?"}},
        },
        status=status,
        response_body=response
        if response is not None
        else {"model": "jev-latest", "answers": {"made_progress": {"type": "noul", "noul": 0.91}}, "usage": {"input_tokens": 1, "output_tokens": 1}},
        error=error,
        elapsed_s=0.42,
    )


def verdict(ex):
    return JevVerdict(
        made_progress=True,
        progress_probability=0.91,
        outcome="progress",
        outcome_confidence=0.8,
        outcome_probabilities={"triumph": 0.1, "progress": 0.8, "stalled": 0.05, "setback": 0.05},
        creativity=2.5,
        creativity_confidence=0.7,
        creativity_legend={"0": "None", "1": "Ordinary", "2": "Some flair", "3": "Very inventive", "4": "Genius"},
        exchange=ex,
    )


def make_summary(*, jev=True, reasoning=True, failed=False, won=False, quit_early=True, jev_exchange=None):
    think = (lambda words: words) if reasoning else (lambda words: None)
    round1 = RoundRecord(
        number=1,
        challenge="A goose guards the door.",
        player_plan="I bribe it with a bagel",
        judge="jev" if jev else "local",
        made_progress=True,
        judge_explanation="Jev puts the chance at 91%.",
        progress_after=1,
        jev=verdict(jev_exchange or exchange()) if jev else None,
        llm_calls=[] if jev else [("judge", llm_result('{"made_progress": true}', think("round one judge thoughts")))],
    )
    round1.llm_calls.append(("outcome", llm_result("The goose accepts.", think("outcome thoughts"))))
    round2 = RoundRecord(
        number=2,
        challenge="A moose wants a word.",
        player_plan="I hug the moose",
        judge="local",
        made_progress=False,
        judge_explanation="Hugging isn't a plan.",
        progress_after=1,
        llm_calls=[
            ("judge", llm_result('{"made_progress": false, "explanation": "Hugging isn\'t a plan."}', think("judge thoughts"))),
            ("outcome", llm_result("The moose sighs.", None)),
        ],
        failed_jev_exchange=exchange(status=500, response={"error": "hamsters resting"}, error="Jev is having trouble on its side")
        if failed
        else None,
    )
    return GameSummary(
        won=won,
        quit_early=quit_early,
        progress=1,
        target=5,
        intro="You overslept.",
        ending="You go home for tea.",
        rounds=[round1, round2],
        intro_calls=[("intro", llm_result("You overslept.", think("intro thoughts")))],
        ending_calls=[("ending_quit", llm_result("You go home for tea.", think("ending thoughts")))],
    )


# ---------------------------------------------------------------------------
# The two independent questions
# ---------------------------------------------------------------------------


def test_neither_detail_selected(tmp_path):
    ui, script, console = make_ui(["n", "n", "n"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    text = output(console)
    assert "Behind the scenes" in text
    assert "Your morning at a glance" in text and "A goose guards the door." in text
    assert "You called it a day after 1 of 5 steps (2 rounds)." in text
    assert "request to Jev" not in text
    assert "reasoning while" not in text
    assert "Jev request & response" in script.prompts[0]
    assert "reasoning (chain-of-thought)" in script.prompts[1]
    assert "Save a transcript" in script.prompts[2]
    assert list(tmp_path.iterdir()) == []


def test_jev_only(tmp_path):
    ui, _, console = make_ui(["y", "n", "n"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    text = output(console)
    assert "Round 1: request to Jev" in text and "Round 1: Jev's response" in text
    assert "https://api.typesafe.ai/v1/systemone" in text
    assert '"Authorization": "Bearer ****abcd"' in text
    assert '"noul": 0.91' in text
    assert "answers.made_progress.noul" in text  # the reading tip
    assert "Refereed by your local model - no Jev call this round." in text  # round 2
    assert "reasoning while" not in text and "intro thoughts" not in text


def test_reasoning_only(tmp_path):
    ui, _, console = make_ui(["n", "y", "n"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    text = output(console)
    for thought in ("intro thoughts", "outcome thoughts", "judge thoughts", "ending thoughts"):
        assert thought in text
    assert "The opening" in text and "The ending" in text
    assert "reasoning while writing the opening story (2 words)" in text
    assert "didn't show its reasoning while narrating what happened next" in text  # round 2 outcome
    assert "Its final answer:" in text  # judge answers are shown with their reasoning
    assert "request to Jev" not in text


def test_both_details(tmp_path):
    ui, _, console = make_ui(["y", "y", "n"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    text = output(console)
    assert "Round 1: request to Jev" in text
    assert "judge thoughts" in text and "ending thoughts" in text
    assert text.index("Round 1: request to Jev") < text.index("Round 2")


def test_failed_jev_call_is_shown_in_the_jev_view(tmp_path):
    ui, _, console = make_ui(["y", "n", "n"])
    run_review(ui, make_summary(failed=True), export_dir=tmp_path)
    text = output(console)
    assert "This Jev call failed" in text
    assert "Round 2: request to Jev" in text and '"status": 500' in text
    assert "hamsters resting" in text and "Jev is having trouble on its side" in text


def test_jev_question_skipped_when_no_round_used_jev(tmp_path):
    ui, script, _ = make_ui(["y", "n"])  # only: reasoning? export?
    run_review(ui, make_summary(jev=False), export_dir=tmp_path)
    assert not any("Jev" in p for p in script.prompts)
    assert len(script.prompts) == 2


def test_jev_question_asked_when_only_a_failed_call_exists(tmp_path):
    ui, script, console = make_ui(["y", "n", "n"])
    run_review(ui, make_summary(jev=False, failed=True), export_dir=tmp_path)
    assert "Jev request & response" in script.prompts[0]
    assert "Round 2: request to Jev" in output(console)


def test_reasoning_question_skipped_with_a_note_when_none_exposed(tmp_path):
    ui, script, console = make_ui(["n", "n", "n"])  # only: jev? the local verdicts? export?
    run_review(ui, make_summary(reasoning=False), export_dir=tmp_path)
    assert not any("chain-of-thought" in p for p in script.prompts)
    assert "didn't expose any reasoning" in output(console)


def test_only_export_offered_when_nothing_to_peek_at(tmp_path):
    summary = make_summary(jev=False, reasoning=False)
    for record in summary.rounds:
        record.llm_calls = [(p, r) for p, r in record.llm_calls if not p.startswith("judge")]
    ui, script, console = make_ui(["n"])
    run_review(ui, summary, export_dir=tmp_path)
    assert len(script.prompts) == 1 and "Save a transcript" in script.prompts[0]
    assert "Pick either, both or neither" not in output(console)


def test_a_local_only_game_without_reasoning_can_still_show_each_verdict(tmp_path):
    """The Jev lesson points players at the review to compare: a local-only game on
    a slow computer (thinking off) must still be able to show the local verdicts."""
    ui, script, console = make_ui(["y", "n"])  # the local verdicts? export?
    run_review(ui, make_summary(jev=False, reasoning=False), export_dir=tmp_path)
    assert "verdict (its JSON answer)" in script.prompts[0]
    text = output(console)
    assert "Pick either, both or neither" not in text  # only one question follows
    assert "Want to see how the game really worked?" in text
    assert "Your local model's answer:" in text and '"made_progress": false' in text


def test_either_both_or_neither_only_when_two_questions_follow(tmp_path):
    ui, script, console = make_ui(["n", "n", "n"])
    run_review(ui, make_summary(jev=True, reasoning=True), export_dir=tmp_path)
    assert "Pick either, both or neither" in output(console)
    ui, script, console = make_ui(["n", "n"])
    run_review(ui, make_summary(jev=False, reasoning=True), export_dir=tmp_path)
    assert "Pick either, both or neither" not in output(console)


def test_markup_in_recorded_text_is_shown_literally(tmp_path):
    summary = make_summary()
    summary.rounds[0].challenge = "A [red]goose[/red] [/] appears."
    summary.rounds[0].player_plan = "I type [bold]markup[/bold]"
    summary.intro_calls[0] = ("intro", llm_result("x", "Thinking [blink]loudly[/blink] [/]"))
    ui, _, console = make_ui(["y", "y", "n"])
    run_review(ui, summary, export_dir=tmp_path)
    text = output(console)
    assert "A [red]goose[/red] [/] appears." in text
    assert "I type [bold]markup[/bold]" in text
    assert "Thinking [blink]loudly[/blink] [/]" in text


def test_ctrl_c_during_the_review_ends_it_politely(tmp_path):
    ui, _, console = make_ui([KeyboardInterrupt])
    run_review(ui, make_summary(), export_dir=tmp_path)  # must not raise
    assert "Skipping the rest of the review" in output(console)


@pytest.mark.parametrize(
    "won, quit_early, expected",
    [
        (True, False, "You made it to work!"),
        (False, True, "You called it a day"),
        (False, False, "The clock struck nine"),
    ],
)
def test_result_headline(won, quit_early, expected, tmp_path):
    ui, _, console = make_ui(["n", "n", "n"])
    run_review(ui, make_summary(won=won, quit_early=quit_early), export_dir=tmp_path)
    assert expected in output(console)


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def test_export_from_the_review_writes_both_files(tmp_path):
    ui, _, console = make_ui(["n", "n", "y"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gettowork-transcript-1.json", "gettowork-transcript-1.md"]
    text = " ".join(output(console).split())
    # A long folder path (e.g. Windows' temp folder) wraps mid-name, so look for the
    # file names with all line breaks and spaces squeezed out.
    squeezed = "".join(output(console).split())
    assert "gettowork-transcript-1.json" in squeezed and "gettowork-transcript-1.md" in squeezed
    assert "API key is never included" in text


def test_export_naming_uses_the_first_unused_number(tmp_path):
    assert next_export_number(tmp_path) == 1
    (tmp_path / "gettowork-transcript-1.json").write_text("{}")
    (tmp_path / "gettowork-transcript-2.md").write_text("old")  # either extension blocks a number
    (tmp_path / "gettowork-transcript-4.json").write_text("{}")
    assert next_export_number(tmp_path) == 3
    json_path, md_path = export_transcript(make_summary(), tmp_path)
    assert (json_path.name, md_path.name) == ("gettowork-transcript-3.json", "gettowork-transcript-3.md")
    json_path2, _ = export_transcript(make_summary(), tmp_path)
    assert json_path2.name == "gettowork-transcript-5.json"
    assert (tmp_path / "gettowork-transcript-2.md").read_text() == "old"  # never overwritten


def test_export_creates_the_folder(tmp_path):
    target = tmp_path / "new" / "folder"
    json_path, md_path = export_transcript(make_summary(), target)
    assert json_path.parent == target and md_path.exists()


def test_export_defaults_to_the_current_folder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    json_path, _ = export_transcript(make_summary())
    assert json_path.resolve() == (tmp_path / "gettowork-transcript-1.json").resolve()


def test_export_failure_is_explained(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("I am a file")
    ui, _, console = make_ui(["n", "n", "y"])
    run_review(ui, make_summary(), export_dir=blocker)
    assert "Couldn't save the transcript" in output(console)


class _FullDiskFile:
    """A file on a full disk: it was created, but its contents can't be stored."""

    def __init__(self, real):
        self._real = real

    def write(self, text):
        raise OSError(errno.ENOSPC, "No space left on device")

    def close(self):
        self._real.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


@pytest.mark.parametrize("full_at", [".json", ".md"])
def test_a_full_disk_leaves_no_empty_or_half_written_transcript(tmp_path, monkeypatch, full_at):
    from gettowork import review

    def full_disk_open(path, mode="r", *args, **kwargs):
        real = open(path, mode, *args, **kwargs)
        return _FullDiskFile(real) if str(path).endswith(full_at) else real

    monkeypatch.setattr(review, "open", full_disk_open, raising=False)
    ui, _, console = make_ui(["n", "n", "y"])
    run_review(ui, make_summary(), export_dir=tmp_path)
    text = " ".join(output(console).split())
    assert "Couldn't save the transcript" in text and "No space left on device" in text
    assert list(tmp_path.iterdir()) == []  # not an empty .json, nor a .json without its .md
    monkeypatch.delattr(review, "open")
    json_path, _ = export_transcript(make_summary(), tmp_path)  # room again: number 1 is still free
    assert json_path.name == "gettowork-transcript-1.json"


def test_exported_json_and_markdown_contents(tmp_path):
    json_path, md_path = export_transcript(make_summary(failed=True), tmp_path)
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["format"] == "gettowork-transcript" and data["version"] == 1
    assert data["result"] == "quit" and data["progress"] == 1 and data["target"] == 5
    assert [r["number"] for r in data["rounds"]] == [1, 2]
    assert data["rounds"][0]["jev"]["progress_probability"] == 0.91
    assert data["rounds"][0]["jev"]["exchange"]["request_headers"]["Authorization"] == "Bearer ****abcd"
    assert data["rounds"][1]["failed_jev_exchange"]["status"] == 500
    assert data["intro_calls"][0]["reasoning"] == "intro thoughts"
    assert data["rounds"][1]["llm_calls"][0]["purpose"] == "judge"
    assert data["rounds"][1]["llm_calls"][0]["messages"][0]["content"] == "TASK: test"

    md = md_path.read_text(encoding="utf-8")
    assert md.startswith("# Get To Work - game transcript")
    for heading in ("## Good morning!", "## Round 1", "## Round 2", "## The ending", "### Jev request", "### Jev response"):
        assert heading in md
    assert "intro thoughts" in md and "judge thoughts" in md
    assert "this call failed" in md


def test_exports_never_contain_the_api_key():
    leaky = exchange(
        auth=f"Bearer {SECRET}",
        response={"error": f"bad key {SECRET}"},
        error=f"Jev said the key {SECRET} is wrong",
    )
    summary = make_summary(jev_exchange=leaky)
    summary.rounds[1].failed_jev_exchange = JevExchange(
        url="https://api.typesafe.ai/v1/systemone",
        request_headers={"authorization": SECRET, "X-Api-Key": SECRET},
        request_body={},
        status=401,
        response_body=None,
        error=None,
        elapsed_s=0.1,
    )
    as_json = json.dumps(summary_to_dict(summary))
    as_md = summary_to_markdown(summary)
    for exported in (as_json, as_md):
        assert SECRET not in exported
        assert SECRET[9:-4] not in exported  # not even most of it
        assert "Bearer ****abcd" in exported
    data = summary_to_dict(summary)
    assert data["rounds"][0]["jev"]["exchange"]["request_headers"]["Authorization"] == "Bearer ****abcd"
    assert data["rounds"][1]["failed_jev_exchange"]["request_headers"] == {"authorization": "****abcd", "X-Api-Key": "****abcd"}


def test_review_screen_never_shows_the_api_key(tmp_path):
    summary = make_summary(jev_exchange=exchange(auth=f"Bearer {SECRET}", response={"echo": SECRET}))
    ui, _, console = make_ui(["y", "y", "y"])
    run_review(ui, summary, export_dir=tmp_path)
    assert SECRET not in output(console)
    for path in tmp_path.iterdir():
        assert SECRET not in path.read_text(encoding="utf-8")


def test_markdown_sections_can_be_left_out():
    summary = make_summary()
    no_jev = summary_to_markdown(summary, include_jev=False)
    assert "### Jev request" not in no_jev and "intro thoughts" in no_jev
    no_reasoning = summary_to_markdown(summary, include_reasoning=False)
    assert "### Jev request" in no_reasoning and "intro thoughts" not in no_reasoning
    neither = summary_to_markdown(summary, include_jev=False, include_reasoning=False)
    assert "## Round 2" in neither and "Jev request" not in neither and "thoughts" not in neither


def test_markdown_code_fences_survive_backticks_in_reasoning():
    summary = make_summary()
    summary.intro_calls[0] = ("intro", llm_result("x", "I will write ```json\n{}\n``` here"))
    md = summary_to_markdown(summary)
    assert "````text\nI will write ```json" in md
    assert "```\n``` here\n````" in md or "``` here\n````" in md


def test_summary_to_dict_is_strict_json_safe():
    summary = make_summary()
    summary.intro_calls[0][1].raw = {"timings": float("nan"), "path": Path("/tmp/x"), "tuple": (1, 2), 3: "int key"}
    data = summary_to_dict(summary)
    text = json.dumps(data, allow_nan=False)
    raw = json.loads(text)["intro_calls"][0]["raw"]
    assert raw["timings"] is None and raw["tuple"] == [1, 2] and raw["3"] == "int key"
    assert raw["path"].endswith("x")


def test_summary_without_rounds():
    summary = GameSummary(won=False, quit_early=True, progress=0, target=5, intro="", ending="Bye.")
    data = summary_to_dict(summary)
    assert data["rounds"] == [] and data["intro_calls"] == []
    md = summary_to_markdown(summary)
    assert "_(no opening story)_" in md and "Bye." in md
    ui, script, console = make_ui(["n"])
    run_review(ui, summary)
    assert len(script.prompts) == 1


# ---------------------------------------------------------------------------
# End to end: a real (mock-model) game with a fake Jev, then the review
# ---------------------------------------------------------------------------


class FakeJevTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, method, url, headers, body, timeout):
        self.calls += 1
        answers = {
            "made_progress": {"type": "noul", "noul": 0.9},
            "outcome": {"type": "choice", "choice": "triumph", "confidence": 0.7,
                        "probabilities": {"triumph": 0.7, "progress": 0.2, "stalled": 0.05, "setback": 0.05}},
            "creativity": {"type": "score", "score": 3.4, "confidence": 0.6,
                           "legend": {str(i): f"level {i}" for i in range(5)},
                           "probabilities": {"0": 0, "1": 0, "2": 0.1, "3": 0.4, "4": 0.5}},
        }
        return 200, {}, json.dumps({"model": "jev-latest", "answers": answers, "usage": {}}).encode()


@pytest.mark.skipif(MockBackend is None, reason="backends/mock.py not available")
def test_real_game_then_review_and_export(tmp_path):
    from gettowork.game import Game
    from gettowork.jev import JevClient

    client = JevClient(SECRET, transport=FakeJevTransport(), max_retries=0)
    plans = ["I bribe the obstacle with a very large sandwich"] * 5
    ui, _, console = make_ui(plans + ["y", "y", "y"])
    summary = Game(MockBackend(seed=2), ui, jev=client).run()
    assert summary.won and all(r.judge == "jev" for r in summary.rounds)
    run_review(ui, summary, export_dir=tmp_path)

    text = output(console)
    assert "Round 5: request to Jev" in text
    # The pretend model's "thinking" is labelled as scripted example text (#82); the story itself
    # is written without thinking, and the review says why.
    assert "The pretend model's scripted example reasoning while writing the opening story" in text
    assert "The pretend model didn't show its reasoning while writing the victory story (the game asks" in text
    assert "Your model's reasoning" not in text
    assert SECRET not in text
    assert "Couldn't save the transcript" not in text  # (a full disk, say): report that, not a JSON error
    data = json.loads((tmp_path / "gettowork-transcript-1.json").read_text(encoding="utf-8"))
    assert len(data["rounds"]) == 5 and data["result"] == "won"
    assert data["ending_calls"][0]["purpose"] == "victory"
    assert all(SECRET not in p.read_text(encoding="utf-8") for p in tmp_path.iterdir())
    assert not math.isnan(data["rounds"][0]["jev"]["creativity"])
    markdown = (tmp_path / "gettowork-transcript-1.md").read_text(encoding="utf-8")
    assert "### Scripted example reasoning (pretend model) while writing the opening story" in markdown


# ---------------------------------------------------------------------------
# Review fixes: no personal paths, no crash on broken characters
# ---------------------------------------------------------------------------


def test_exported_transcripts_hide_the_home_folder(tmp_path, monkeypatch):
    fake_home = tmp_path / "home" / "alice"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    model_path = str(fake_home / "GetToWork" / "models" / "qwen3.gguf")
    result = llm_result("You overslept.")
    result.raw = {"model": model_path, "usage": {"prompt_tokens": 12}}
    summary = GameSummary(won=False, quit_early=True, progress=0, target=5, intro="You overslept.", ending="Bye.",
                          intro_calls=[("intro", result)])
    json_path, md_path = export_transcript(summary, tmp_path / "out")
    exported = json_path.read_text(encoding="utf-8") + md_path.read_text(encoding="utf-8")
    assert "alice" not in exported
    assert json.loads(json_path.read_text(encoding="utf-8"))["intro_calls"][0]["raw"]["model"].startswith("~")


def test_broken_characters_from_an_odd_terminal_never_crash_the_export(tmp_path):
    plan = "I shout \udc9dgoose\udc81 at the door"  # what surrogateescape makes of undecodable bytes
    record = RoundRecord(number=1, challenge="A goose.", player_plan=plan, judge="local", made_progress=True,
                         judge_explanation="ok", progress_after=1)
    summary = GameSummary(won=False, quit_early=True, progress=1, target=5, intro="Hi.", ending="Bye.", rounds=[record])
    json_path, md_path = export_transcript(summary, tmp_path)
    assert "goose" in json_path.read_text(encoding="utf-8") and "goose" in md_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Round 3: live keys are scrubbed everywhere; model text can't drive the terminal;
# the Jev review doesn't repeat identical question definitions
# ---------------------------------------------------------------------------


def _summary_with_pasted_key():
    summary = make_summary(jev=False, reasoning=True)
    summary.rounds[0].player_plan = SECRET  # pasted by accident as a plan
    summary.rounds[0].llm_calls[0] = (
        "judge",
        LLMResult(text='{"made_progress": false}', reasoning=f"The plan is just {SECRET}.", model="m",
                  backend="scripted", elapsed_s=0.1, messages=[{"role": "user", "content": SECRET}]),
    )
    return summary


def test_a_key_pasted_as_a_plan_is_masked_in_the_transcript(tmp_path):
    summary = _summary_with_pasted_key()
    assert SECRET not in json.dumps(summary_to_dict(summary, secrets={SECRET}))
    assert SECRET not in summary_to_markdown(summary, secrets={SECRET})
    json_path, md_path = export_transcript(summary, tmp_path, secrets={SECRET})
    assert SECRET not in json_path.read_text(encoding="utf-8")
    assert SECRET not in md_path.read_text(encoding="utf-8")
    assert "abcd" in md_path.read_text(encoding="utf-8")  # the masked form ("****abcd") is kept


def test_a_key_pasted_as_a_plan_is_masked_in_the_review_on_screen(tmp_path):
    ui, _, console = make_ui(["y", "y"])
    run_review(ui, _summary_with_pasted_key(), export_dir=tmp_path, secrets={SECRET})
    text = output(console)
    assert SECRET not in text and "Your plan:" in text


def test_review_strips_terminal_control_codes_from_model_text(tmp_path):
    evil = "\x1b]0;HACKED TITLE\x1b\\ \x1b[2J\x1b[H\x1b[1A"
    summary = make_summary(jev=False)
    summary.rounds[1].challenge = "A moose" + evil
    summary.rounds[1].llm_calls[0] = ("judge", llm_result('{"made_progress": false}' + evil, "thinking" + evil))
    buf = io.StringIO()
    console = Console(file=buf, width=110, force_terminal=True)
    script = Script(["y", "n"])
    ui = UI(console=console, input_fn=script, secret_fn=script, open_url_fn=lambda url: True)
    run_review(ui, summary, export_dir=tmp_path)
    out = buf.getvalue()
    assert "HACKED" not in out and "\x1b]0" not in out and "\x1b[2J" not in out and "\x1b[1A" not in out


def test_jev_review_shows_the_questions_once_then_only_the_state(tmp_path):
    summary = make_summary(jev=True, failed=True)
    ui, _, console = make_ui(["y", "n", "n"])
    run_review(ui, summary, export_dir=tmp_path)
    text = output(console)
    assert text.count("Did they make progress?") == 1  # the question text is shown once
    assert "the same three questions as in the first request above" in text
    assert "Round 2: request to Jev" in text and '"status": 500' in text


def test_the_review_says_when_the_game_switched_thinking_off(tmp_path):
    ui, _, console = make_ui(["n", "n", "n"])
    note = "Your model can think out loud, but I asked it to skip that to keep turns quick."
    run_review(ui, make_summary(jev=True, reasoning=False), export_dir=tmp_path, thinking_skipped_note=note)
    text = " ".join(output(console).split())
    assert note in text
    assert "not every model 'thinks out loud'" not in text


def test_interactive_review_pauses_between_rounds(tmp_path):
    console = Console(file=io.StringIO(), width=110)
    script = Script(["y", "n", "", "n"])  # Jev? yes; reasoning? no; Enter between rounds 1 and 2; save? no
    ui = UI(console=console, input_fn=script, secret_fn=script, open_url_fn=lambda url: True, pauses=True)
    run_review(ui, make_summary(jev=True), export_dir=tmp_path)
    assert any("next round" in p for p in script.prompts)


def test_exported_transcripts_carry_no_terminal_control_codes(tmp_path):
    summary = make_summary(jev=False)
    summary.rounds[1].llm_calls[0] = ("judge", llm_result('{"made_progress": false}', "think \x1b]0;T\x07 \x9b2J"))
    json_path, md_path = export_transcript(summary, tmp_path)
    md = md_path.read_text(encoding="utf-8")
    raw_json = json_path.read_text(encoding="utf-8")
    assert "\x1b" not in md and "\x9b" not in md and "\x9b" not in raw_json
    assert json.loads(raw_json)["rounds"][1]["llm_calls"][0]["reasoning"].endswith("\x9b2J")  # kept, just escaped


def test_an_escape_code_followed_by_a_backslash_can_never_crash_the_review(tmp_path):
    """ESC + backslash + "[/]" once slipped past safe_text: escape() doubled the
    backslash and safe_text took one of the pair, exposing a live "[/]"."""
    summary = make_summary(jev=False)
    summary.rounds[0].challenge = "A goose \x1b\\[/] waves a sign."
    summary.rounds[1].judge_explanation = "Waving a sign that says \x1b\\[/] and \x9b\\[/] and \x1b]0;t\x1b\\[/]"
    summary.rounds[1].player_plan = "I type \x1b\\[red] loudly"
    ui, _, console = make_ui(["y", "n"])  # reasoning? export?
    run_review(ui, summary, export_dir=tmp_path)  # must not raise MarkupError
    text = output(console)
    assert "[/]" in text and "\x1b" not in text and "\x9b" not in text
