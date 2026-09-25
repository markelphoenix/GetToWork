"""The end-of-game review: a peek behind the curtain.

After the story ends, the player can look at how the game really worked -
two *independent* choices (either, both or neither):

1. **Jev's request and response** for every round it refereed: the exact
   JSON sent to ``POST /v1/systemone`` and the typed answers that came back.
2. **The local model's reasoning** (its "chain-of-thought") for every call:
   the opening, each verdict and outcome, and the ending.

Then the whole game can be saved as a transcript (JSON for programs,
Markdown for people). API keys never appear: Jev exchanges are already
masked (``Bearer ****abcd``), and this module masks any authorization-style
header again before showing or saving anything - belt and braces.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from .jev import redact_key
from .types import GameSummary, JevExchange, JevVerdict, LLMResult, RoundRecord
from .ui import UI, UserQuit, plain, safe_text

__all__ = [
    "run_review",
    "summary_to_dict",
    "summary_to_markdown",
    "export_transcript",
    "next_export_number",
]

TRANSCRIPT_FORMAT = "gettowork-transcript"
TRANSCRIPT_VERSION = 1
EXPORT_PREFIX = "gettowork-transcript-"

# Friendly names for the purpose of each local-model call.
PURPOSE_LABELS = {
    "intro": "writing the opening story",
    "judge": "refereeing your plan",
    "judge_retry": "refereeing your plan (second try)",
    "outcome": "narrating what happened next",
    "victory": "writing the victory story",
    "ending_quit": "writing the ending",
}

# Headers that can carry credentials. Their values are always masked.
_SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"})
_ALREADY_MASKED_RE = re.compile(r"^(?:bearer\s+)?\*{4}\S{0,4}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# The interactive review
# ---------------------------------------------------------------------------


def run_review(ui: UI, summary: GameSummary, *, export_dir: Optional[Path] = None,
               secrets: Iterable[str] = (), thinking_skipped_note: Optional[str] = None) -> None:
    """Show the recap, offer the independent peeks (Jev, reasoning, local verdicts), then offer an export.

    `secrets` are the player's live credentials (the Jev API key, in every
    form it could appear): they are masked everywhere - on screen and in the
    saved transcript - even if the key was pasted by accident as a plan.
    `thinking_skipped_note` explains, when the game itself switched a
    thinking model's reasoning off (see ``Game.thinking_note``), why there's
    no reasoning to show. Pressing Ctrl+C here just ends the review politely.
    """
    try:
        _run_review(ui, summary, export_dir, _secret_set(secrets), thinking_skipped_note)
    except UserQuit:
        ui.say()
        ui.info("Skipping the rest of the review.")


def _secret_set(secrets: Iterable[str]) -> set[str]:
    """Credentials worth scrubbing (very short strings would mangle ordinary text)."""
    return {s for s in (str(x).strip() for x in (secrets or ())) if len(s) >= 8}


def _run_review(ui: UI, summary: GameSummary, export_dir: Optional[Path], secrets: set[str],
                thinking_skipped_note: Optional[str] = None) -> None:
    summary = _scrub_summary(summary, secrets | _raw_secrets_in(_all_exchanges(summary)))
    ui.heading("Behind the scenes")
    _show_recap(ui, summary)

    has_jev = any(_round_exchange(r) is not None for r in summary.rounds)
    thinking = [result for _purpose, result in _all_calls(summary) if result.reasoning]
    has_reasoning = bool(thinking)
    scripted = has_reasoning and all(_is_scripted(result) for result in thinking)
    # The local referee's own answers (its JSON verdicts): always worth a look - the
    # reasoning view shows them too, so they're offered separately only without it.
    has_answers = not has_reasoning and any(p.startswith("judge") for r in summary.rounds for p, _ in r.llm_calls)
    offers = [has_jev, has_reasoning, has_answers].count(True)
    if offers >= 2:
        ui.say("Want to see how the game really worked? Pick either, both or neither.")
    elif offers == 1:
        ui.say("Want to see how the game really worked?")

    show_jev = False
    if has_jev:
        show_jev = ui.confirm("See the Jev request & response for each round?", default=False)
    show_reasoning = False
    if has_reasoning:
        whose = "the pretend model's scripted example" if scripted else "your local model's"
        show_reasoning = ui.confirm(f"See {whose} reasoning (chain-of-thought) for each round?", default=False)
    elif thinking_skipped_note:
        ui.info(escape(thinking_skipped_note))
    elif not _is_pretend(summary):
        ui.info(
            "Your local model didn't expose any reasoning this game - not every model 'thinks out loud' "
            "(\"thinking\" models such as Qwen3 or DeepSeek-R1 do)."
        )
    show_answers = False
    if has_answers:
        show_answers = ui.confirm(f"See {_local_name(summary)}'s verdict (its JSON answer) for each round?",
                                  default=False)

    if show_jev or show_reasoning or show_answers:
        _show_details(ui, summary, show_jev=show_jev, show_reasoning=show_reasoning, show_answers=show_answers)
    _offer_export(ui, summary, export_dir, secrets)


def _is_pretend(summary: GameSummary) -> bool:
    """Was this a --mock game (the pretend model's scripted rules, not a real model)?"""
    calls = [result for _p, result in _all_calls(summary)]
    return bool(calls) and all(_is_scripted(result) for result in calls)


def _local_name(summary: GameSummary) -> str:
    return "the pretend model" if _is_pretend(summary) else "your local model"


def _result_line(summary: GameSummary) -> str:
    rounds = len(summary.rounds)
    played = f"{rounds} round{'' if rounds == 1 else 's'}"
    if summary.won:
        return f"You made it to work! {summary.progress} of {summary.target} steps in {played}."
    if summary.quit_early:
        return f"You called it a day after {summary.progress} of {summary.target} steps ({played})."
    return f"The clock struck nine after {summary.progress} of {summary.target} steps ({played})."


def _show_recap(ui: UI, summary: GameSummary) -> None:
    ui.say(f"[bold]{escape(_result_line(summary))}[/bold]")
    if not summary.rounds:
        return
    rows = [
        (
            str(r.number),
            plain(_clip(r.challenge, 60)),
            plain(_clip(r.player_plan, 45)),
            "Jev" if r.judge == "jev" else ("pretend model" if _is_pretend(summary) else "local model"),
            "progress" if r.made_progress else "not yet",
        )
        for r in summary.rounds
    ]
    ui.table("Your morning at a glance", ["#", "Challenge", "Your plan", "Referee", "Result"], rows)


def _show_details(ui: UI, summary: GameSummary, *, show_jev: bool, show_reasoning: bool,
                  show_answers: bool = False) -> None:
    if show_reasoning and summary.intro_calls:
        ui.heading("The opening")
        _show_reasoning(ui, summary.intro_calls)

    tip_shown = False
    shown = {"questions": False}
    for index, record in enumerate(summary.rounds):
        if index:
            ui.pause("Press Enter for the next round")  # interactive terminals only: one round at a time
        ui.heading(f"Round {record.number}")
        referee = "Jev" if record.judge == "jev" else _local_name(summary)
        verdict = "made progress" if record.made_progress else "no progress"
        ui.say(f"[bold]Challenge:[/bold] {plain(record.challenge)}")
        ui.say(f"[bold]Your plan:[/bold] {plain(record.player_plan)}")
        ui.say(f"[bold]Verdict:[/bold] {verdict}, according to {referee}. {plain(record.judge_explanation)}")
        if show_jev:
            tip_shown = _show_jev_round(ui, record, tip_shown, shown)
        if show_reasoning:
            _show_reasoning(ui, record.llm_calls)
        elif show_answers:
            for purpose, result in record.llm_calls:
                if purpose.startswith("judge"):
                    ui.console.print(Text.assemble((f"{_local_name(summary).capitalize()}'s answer: ", "bold"),
                                                   safe_text(_clip(result.text, 300))))

    if show_reasoning and summary.ending_calls:
        ui.heading("The ending")
        _show_reasoning(ui, summary.ending_calls)


def _show_jev_round(ui: UI, record: RoundRecord, tip_shown: bool, shown: Optional[dict] = None) -> bool:
    """Show one round's Jev exchange. Returns whether the reading tip has been shown.

    The three questions Jev is asked (with their long instructions) are the
    same every round, so they're shown in full once; after that each round's
    request shows only what changes - the state - and the answers.
    """
    shown = shown if shown is not None else {"questions": False}
    exchange = _round_exchange(record)
    if exchange is None:
        ui.say("[dim]Refereed by your local model - no Jev call this round.[/dim]")
        return tip_shown
    view = _exchange_to_dict(exchange, _raw_secrets_in([exchange]))
    if record.jev is None:
        ui.warn("This Jev call failed, so your local model refereed instead. Here's what was sent and what came back:")
    method = "POST" if view["request_body"] else "GET"
    body = view["request_body"]
    title = f"Round {record.number}: request to Jev"
    if isinstance(body, dict) and isinstance(body.get("questions"), dict):
        if shown.get("questions"):
            body = {**body, "questions": "(the same three questions as in the first request above)"}
            title += " - what changed: the state"
        else:
            shown["questions"] = True
            title += " (the questions are the same every round, so later rounds only show the state)"
    ui.json(
        {"method": method, "url": view["url"], "headers": view["request_headers"], "body": body},
        title=title,
    )
    response = {"status": view["status"], "elapsed_s": view["elapsed_s"], "body": view["response_body"]}
    if view["error"]:
        response["error"] = view["error"]
    ui.json(response, title=f"Round {record.number}: Jev's response")
    if record.jev is not None and not tip_shown:
        ui.info(
            "Tip: answers.made_progress.noul is the single number that decided this round - "
            "the game counts progress when it's 0.5 or more."
        )
        return True
    return tip_shown


# Story calls where the game asks a thinking model to answer straight away (see game.Game._story_call).
_NO_THINKING_PURPOSES = frozenset({"outcome", "victory", "ending_quit"})


def _is_scripted(result: LLMResult) -> bool:
    """The pretend model's "thinking" is scripted example text, and is labelled that way."""
    return result.backend == "mock"


def _show_reasoning(ui: UI, calls: list[tuple[str, LLMResult]]) -> None:
    for purpose, result in calls:
        label = PURPOSE_LABELS.get(purpose, purpose)
        who = "The pretend model" if _is_scripted(result) else "Your model"
        if not result.reasoning:
            why = " (the game asks for the story straight away, to keep turns quick)" if purpose in _NO_THINKING_PURPOSES else ""
            ui.say(f"[dim]{who} didn't show its reasoning while {escape(label)}{why}.[/dim]")
            continue
        words = len(result.reasoning.split())
        if _is_scripted(result):
            title = f"The pretend model's scripted example reasoning while {label} ({words} words)"
        else:
            title = f"Your model's reasoning while {label} ({words} words)"
        ui.console.print(
            Panel(
                Text(safe_text(result.reasoning).strip()),
                title=escape(safe_text(title)),
                border_style="magenta",
                padding=(0, 1),
            )
        )
        if purpose.startswith("judge"):
            ui.console.print(Text.assemble(("Its final answer: ", "bold"), safe_text(_clip(result.text, 300))))


def _offer_export(ui: UI, summary: GameSummary, export_dir: Optional[Path], secrets: Iterable[str] = ()) -> None:
    directory = Path(export_dir) if export_dir is not None else Path.cwd()
    if not ui.confirm(f"Save a transcript of this game (JSON + Markdown) in {escape(str(directory))}?", default=False):
        return
    try:
        json_path, md_path = export_transcript(summary, directory, secrets=secrets)
    except (OSError, UnicodeError) as exc:
        ui.error(f"Couldn't save the transcript: {escape(str(exc))}")
        return
    ui.success(f"Saved {escape(str(json_path))} and {escape(md_path.name)}.")
    if any(_round_exchange(r) is not None for r in summary.rounds) or secrets:
        ui.info("Your API key is never included - it's masked wherever it appears in transcripts.")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


def next_export_number(directory: Path) -> int:
    """The first ``n`` for which neither ``gettowork-transcript-<n>.json`` nor ``.md`` exists."""
    n = 1
    while (directory / f"{EXPORT_PREFIX}{n}.json").exists() or (directory / f"{EXPORT_PREFIX}{n}.md").exists():
        n += 1
    return n


def export_transcript(summary: GameSummary, export_dir: Optional[Path] = None, *,
                      secrets: Iterable[str] = ()) -> tuple[Path, Path]:
    """Write ``gettowork-transcript-<n>.json`` and ``.md``; returns both paths.

    Never overwrites an existing file (files are opened in exclusive-create
    mode). `secrets` (live credentials) are masked wherever they appear.
    """
    directory = Path(export_dir) if export_dir is not None else Path.cwd()
    directory.mkdir(parents=True, exist_ok=True)
    json_text = json.dumps(summary_to_dict(summary, secrets=secrets), indent=2, ensure_ascii=False) + "\n"
    # json.dumps escapes C0 control codes but not the C1 range: escape those too.
    json_text = re.sub(r"[\x7f-\x9f]", lambda m: f"\\u{ord(m.group()):04x}", json_text)
    md_text = summary_to_markdown(summary, secrets=secrets)
    n = next_export_number(directory)
    while True:
        json_path = directory / f"{EXPORT_PREFIX}{n}.json"
        md_path = directory / f"{EXPORT_PREFIX}{n}.md"
        try:
            # errors="replace": a character that can't be stored (e.g. a broken
            # one typed in an old terminal) becomes "?" instead of a crash.
            with open(json_path, "x", encoding="utf-8", errors="replace") as f:
                f.write(json_text)
        except FileExistsError:  # someone created it a moment ago: try the next number
            n += 1
            continue
        try:
            with open(md_path, "x", encoding="utf-8", errors="replace") as f:
                f.write(md_text)
        except FileExistsError:
            json_path.unlink()
            n += 1
            continue
        return json_path, md_path


# ---------------------------------------------------------------------------
# Converting a GameSummary to plain data / Markdown
# ---------------------------------------------------------------------------


def summary_to_dict(summary: GameSummary, *, secrets: Iterable[str] = ()) -> dict:
    """The whole game as JSON-safe data, with every credential masked (`secrets` too)."""
    secrets = _raw_secrets_in(_all_exchanges(summary)) | _secret_set(secrets)
    data = {
        "format": TRANSCRIPT_FORMAT,
        "version": TRANSCRIPT_VERSION,
        "result": "won" if summary.won else ("quit" if summary.quit_early else "out_of_rounds"),
        "won": summary.won,
        "quit_early": summary.quit_early,
        "progress": summary.progress,
        "target": summary.target,
        "intro": summary.intro,
        "ending": summary.ending,
        "intro_calls": [_call_to_dict(p, r) for p, r in summary.intro_calls],
        "rounds": [_round_to_dict(r, secrets) for r in summary.rounds],
        "ending_calls": [_call_to_dict(p, r) for p, r in summary.ending_calls],
    }
    return _hide_home(_scrub(_jsonable(data), secrets))


def summary_to_markdown(summary: GameSummary, *, include_jev: bool = True, include_reasoning: bool = True,
                        secrets: Iterable[str] = ()) -> str:
    """A readable transcript. Jev exchanges and model reasoning can each be left out.
    Credentials (`secrets` too) are masked wherever they appear."""
    secrets = _raw_secrets_in(_all_exchanges(summary)) | _secret_set(secrets)
    out: list[str] = ["# Get To Work - game transcript", "", f"**Result:** {_result_line(summary)}", ""]

    out += ["## Good morning!", "", summary.intro.strip() or "_(no opening story)_", ""]
    if include_reasoning:
        out += _markdown_reasoning(summary.intro_calls)

    for record in summary.rounds:
        referee = "Jev" if record.judge == "jev" else _local_name(summary)
        verdict = "made progress" if record.made_progress else "no progress"
        out += [
            f"## Round {record.number}",
            "",
            f"**Challenge:** {record.challenge}",
            "",
            f"**Your plan:** {record.player_plan}",
            "",
            f"**Verdict:** {verdict}, according to {referee} "
            f"(progress: {record.progress_after} of {summary.target}).",
            "",
            f"**Why:** {record.judge_explanation}",
            "",
        ]
        if include_jev:
            out += _markdown_jev(record, secrets)
        if include_reasoning:
            out += _markdown_reasoning(record.llm_calls)

    out += ["## The ending", "", summary.ending.strip() or "_(no ending)_", ""]
    if include_reasoning:
        out += _markdown_reasoning(summary.ending_calls)
    out += [
        "---",
        "",
        "_Made with Get To Work (MIT licensed, no warranty). API keys are never included in transcripts._",
        "",
    ]
    # Terminal control codes from model text are dropped, so `cat`-ing a shared
    # transcript can't retitle a window or clear a screen either.
    return _scrub(safe_text("\n".join(out)), secrets)


def _markdown_jev(record: RoundRecord, secrets: set[str]) -> list[str]:
    exchange = _round_exchange(record)
    if exchange is None:
        return ["_Refereed by your local model - no Jev call this round._", ""]
    view = _exchange_to_dict(exchange, secrets)
    note = "" if record.jev is not None else " (this call failed, so the local model refereed instead)"
    request = {"url": view["url"], "headers": view["request_headers"], "body": view["request_body"]}
    response = {"status": view["status"], "elapsed_s": view["elapsed_s"], "body": view["response_body"]}
    if view["error"]:
        response["error"] = view["error"]
    return [
        f"### Jev request{note}",
        "",
        _fenced(json.dumps(request, indent=2, ensure_ascii=False, default=str), "json"),
        "",
        "### Jev response",
        "",
        _fenced(json.dumps(response, indent=2, ensure_ascii=False, default=str), "json"),
        "",
    ]


def _markdown_reasoning(calls: list[tuple[str, LLMResult]]) -> list[str]:
    lines: list[str] = []
    for purpose, result in calls:
        if not result.reasoning:
            continue
        label = PURPOSE_LABELS.get(purpose, purpose)
        words = len(result.reasoning.split())
        heading = "Scripted example reasoning (pretend model)" if _is_scripted(result) else "Model reasoning"
        lines += [f"### {heading} while {label} ({words} words)", "", _fenced(result.reasoning.strip(), "text"), ""]
        if purpose.startswith("judge"):
            answer = _clip(result.text, 300).replace("`", "'")
            lines += [f"Final answer: `{answer}`", ""]
    return lines


def _fenced(text: str, lang: str = "") -> str:
    """A Markdown code block whose fence is longer than any backtick run inside."""
    longest = max((len(m) for m in re.findall(r"`+", text)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}{lang}\n{text}\n{fence}"


# -- plain-data helpers -------------------------------------------------------------


def _call_to_dict(purpose: str, result: LLMResult) -> dict:
    return {
        "purpose": purpose,
        "model": result.model,
        "backend": result.backend,
        "elapsed_s": round(float(result.elapsed_s or 0.0), 3),
        "text": result.text,
        "reasoning": result.reasoning,
        "messages": [dict(m) for m in (result.messages or [])],
        "raw": result.raw,
    }


def _round_to_dict(record: RoundRecord, secrets: set[str]) -> dict:
    failed = getattr(record, "failed_jev_exchange", None)
    return {
        "number": record.number,
        "challenge": record.challenge,
        "player_plan": record.player_plan,
        "judge": record.judge,
        "made_progress": record.made_progress,
        "judge_explanation": record.judge_explanation,
        "progress_after": record.progress_after,
        "jev": _verdict_to_dict(record.jev, secrets) if record.jev is not None else None,
        "failed_jev_exchange": _exchange_to_dict(failed, secrets) if failed is not None else None,
        "llm_calls": [_call_to_dict(p, r) for p, r in record.llm_calls],
    }


def _verdict_to_dict(verdict: JevVerdict, secrets: set[str]) -> dict:
    return {
        "made_progress": verdict.made_progress,
        "progress_probability": verdict.progress_probability,
        "outcome": verdict.outcome,
        "outcome_confidence": verdict.outcome_confidence,
        "outcome_probabilities": dict(verdict.outcome_probabilities),
        "creativity": verdict.creativity,
        "creativity_confidence": verdict.creativity_confidence,
        "creativity_legend": dict(verdict.creativity_legend),
        "exchange": _exchange_to_dict(verdict.exchange, secrets),
    }


def _exchange_to_dict(exchange: JevExchange, secrets: set[str]) -> dict:
    data = {
        "url": exchange.url,
        "request_headers": _safe_headers(exchange.request_headers),
        "request_body": exchange.request_body,
        "status": exchange.status,
        "response_body": exchange.response_body,
        "error": exchange.error,
        "elapsed_s": exchange.elapsed_s,
    }
    return _scrub(_jsonable(data), secrets)


def _safe_headers(headers: Optional[dict]) -> dict:
    return {str(k): (_mask_credential(str(v)) if str(k).lower() in _SECRET_HEADERS else v) for k, v in (headers or {}).items()}


def _mask_credential(value: str) -> str:
    """``Bearer tsk_live_...wxyz`` -> ``Bearer ****wxyz``; already-masked values are kept."""
    value = value.strip()
    if _ALREADY_MASKED_RE.match(value):
        return value
    if value.lower().startswith("bearer "):
        return "Bearer " + redact_key(value[7:].strip())
    return redact_key(value)


def _raw_secrets_in(exchanges: Iterable[JevExchange]) -> set[str]:
    """Credentials that were NOT masked in a header, so they can be scrubbed everywhere else too."""
    secrets: set[str] = set()
    for exchange in exchanges:
        for name, value in (exchange.request_headers or {}).items():
            value = str(value).strip()
            if str(name).lower() not in _SECRET_HEADERS or _ALREADY_MASKED_RE.match(value):
                continue
            token = value[7:].strip() if value.lower().startswith("bearer ") else value
            if len(token) >= 8:
                secrets.update({value, token})
    return secrets


def _scrub(value: Any, secrets: set[str]) -> Any:
    if not secrets:
        return value
    if isinstance(value, str):
        for secret in sorted(secrets, key=len, reverse=True):
            value = value.replace(secret, redact_key(secret))
        return value
    if isinstance(value, dict):
        return {_scrub(k, secrets): _scrub(v, secrets) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub(v, secrets) for v in value]
    return value


def _scrub_summary(summary: GameSummary, secrets: set[str]) -> GameSummary:
    """A copy of the whole game (plans, prompts, answers, reasoning...) with `secrets` masked."""
    if not secrets:
        return summary
    return _scrub_object(summary, secrets)


def _scrub_object(value: Any, secrets: set[str]) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {f.name: _scrub_object(getattr(value, f.name), secrets) for f in dataclasses.fields(value)}
        return dataclasses.replace(value, **changes)
    if isinstance(value, tuple):
        return tuple(_scrub_object(v, secrets) for v in value)
    if isinstance(value, (str, dict, list)):
        if isinstance(value, list):
            return [_scrub_object(v, secrets) for v in value]
        if isinstance(value, dict):
            return {_scrub_object(k, secrets): _scrub_object(v, secrets) for k, v in value.items()}
        return _scrub(value, secrets)
    return value


def _home_prefixes() -> list[str]:
    try:
        home = str(Path.home())
    except (RuntimeError, OSError):
        return []
    if len(home) < 3:  # "/" or similar: nothing personal to hide
        return []
    variants = {home, home.replace("\\", "/"), home.replace("/", "\\")}
    return sorted(variants, key=len, reverse=True)


def _hide_home(value: Any, prefixes: Optional[list[str]] = None) -> Any:
    """Replace the home folder (which contains your user name) with "~" everywhere.

    Some engines report the model's full file path in their raw answers
    (e.g. ``C:\\Users\\alice\\AppData\\...\\model.gguf``); a transcript you share
    shouldn't reveal your account name.
    """
    prefixes = _home_prefixes() if prefixes is None else prefixes
    if not prefixes:
        return value
    if isinstance(value, str):
        for prefix in prefixes:
            value = value.replace(prefix, "~")
        return value
    if isinstance(value, dict):
        return {_hide_home(k, prefixes): _hide_home(v, prefixes) for k, v in value.items()}
    if isinstance(value, list):
        return [_hide_home(v, prefixes) for v in value]
    return value


def _jsonable(value: Any) -> Any:
    """Anything -> plain JSON types (odd objects become strings, NaN/inf become null)."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    return str(value)


def _round_exchange(record: RoundRecord) -> Optional[JevExchange]:
    """The Jev exchange for a round: the verdict's, or the failed call's."""
    if record.jev is not None:
        return record.jev.exchange
    return getattr(record, "failed_jev_exchange", None)


def _all_exchanges(summary: GameSummary) -> list[JevExchange]:
    return [ex for ex in (_round_exchange(r) for r in summary.rounds) if ex is not None]


def _all_calls(summary: GameSummary) -> list[tuple[str, LLMResult]]:
    calls = list(summary.intro_calls)
    for record in summary.rounds:
        calls += record.llm_calls
    return calls + list(summary.ending_calls)


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
