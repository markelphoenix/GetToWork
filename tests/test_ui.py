"""Tests for gettowork.ui: prompts, progress bars and output on old consoles."""

from __future__ import annotations

import io

from rich.console import Console
from rich.text import Text

from gettowork.ui import UI, make_stream_safe


def make_ui(answers):
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        return answers.pop(0)

    buf = io.StringIO()
    return UI(console=Console(file=buf, width=80), input_fn=input_fn), prompts, buf


def test_confirm_hint_survives_rich_markup():
    # "[y/N]" looks like a style tag to rich, which would silently hide it.
    ui, prompts, _ = make_ui(["", ""])
    assert ui.confirm("Save it?", default=False) is False
    assert ui.confirm("Go?", default=True) is True
    assert Text.from_markup(prompts[0]).plain.startswith("Save it? [y/N]")
    assert Text.from_markup(prompts[1]).plain.startswith("Go? [Y/n]")


def test_ask_default_is_shown_literally():
    ui, prompts, _ = make_ui([""])
    assert ui.ask("Pick", default="[odd]") == "[odd]"
    assert "([odd])" in Text.from_markup(prompts[0]).plain


def test_download_bar_stays_on_one_line_at_80_columns():
    ui, _, buf = make_ui([])
    with ui.download_progress("Downloading part 1/2: Qwen3-30B-A3B-UD-Q4_K_XL-00001-of-00002.gguf", 10**9) as advance:
        advance(10**9)
    lines = [line for line in buf.getvalue().splitlines() if line.strip()]
    assert len(lines) == 1 and "part 1/2" in lines[0] and "GB" in lines[0]


def test_legacy_code_pages_get_plain_look_alikes_instead_of_crashing():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp437", errors="strict")
    make_stream_safe(stream)
    stream.write("✓ ok — 3 → 4 ≈ 5 · ★ ⚡ 🧠 é\n")
    stream.flush()
    # cp437 has its own "≈", "·" and "é", so only the characters it lacks are swapped.
    assert raw.getvalue().decode("cp437") == "OK ok - 3 -> 4 ≈ 5 · * ! + é\n"


def test_utf8_streams_are_left_alone():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="utf-8", errors="strict")
    make_stream_safe(stream)
    assert stream.errors == "strict"


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------

import pytest  # noqa: E402

from gettowork import ui as ui_module  # noqa: E402
from gettowork.ui import UserQuit, make_input_safe, safe_text  # noqa: E402


def test_safe_text_strips_terminal_control_sequences_but_keeps_text():
    nasty = "Hi\x1b]0;pwned\x07 there\x1b[2J\x1b[1;1H!\x9b31m\tok\nnext\x07\x00"
    assert safe_text(nasty) == "Hi there!\tok\nnext"
    assert safe_text(None) == "" and safe_text(42) == "42"
    assert safe_text("[bold]kept for escape()[/bold]") == "[bold]kept for escape()[/bold]"


def test_every_output_method_strips_control_codes():
    ui, _, buf = make_ui([])
    for method in (ui.say, ui.info, ui.warn, ui.error, ui.success, ui.heading):
        method("x\x1b]0;title\x07y")
    ui.narrate("story\x1b[2Jtext")
    ui.teach("t", "body\x1b[31m")
    assert "\x1b]0;" not in buf.getvalue() and "\x1b[2J" not in buf.getvalue()


def test_status_label_can_change_mid_wait_and_counts_seconds():
    ticks = iter([0.0, 1.0, 10.0, 12.0])
    label = ui_module._Elapsed("Thinking", lambda: next(ticks), 3.0)
    assert label.__rich__().plain == "Thinking"  # 1s: no counter yet
    label.update("Asking again\x1b[2J")
    assert label.__rich__().plain == "Asking again  10s"  # the timer keeps running, the text is cleaned
    ui, _, _ = make_ui([])
    with ui.status("Waiting") as update:
        update("Still waiting")  # callers may change the label


def test_pause_only_waits_when_pauses_are_on():
    prompts = []

    def input_fn(prompt):
        prompts.append(prompt)
        return ""

    quiet = UI(console=Console(file=io.StringIO()), input_fn=input_fn)  # not a terminal: never pauses
    quiet.pause()
    assert prompts == []
    UI(console=Console(file=io.StringIO()), input_fn=input_fn, pauses=True).pause("Press Enter please")
    assert len(prompts) == 1 and "Press Enter please" in prompts[0]


@pytest.mark.parametrize("answer, expected", [("y", "yes"), ("YES", "yes"), ("n", "no"), ("nope", "no"),
                                              ("2", "no"), ("b", "back"), ("  Back ", "back")])
def test_choose_accepts_y_n_numbers_and_unique_prefixes(answer, expected):
    ui, _, _ = make_ui([answer])
    assert ui.choose("Keep going?", [("yes", "Yes please"), ("no", "No thanks"), ("back", "Go back")]) == expected


def test_choose_rejects_ambiguous_prefixes_and_asks_again():
    ui, prompts, buf = make_ui(["s", "save"])
    assert ui.choose("What now?", [("save", "Save"), ("skip", "Skip")]) == "save"
    assert len(prompts) == 2 and "Pick one of the options" in buf.getvalue()


def test_ask_survives_undecodable_input_and_strips_control_codes():
    answers = [UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte"), "caf\x1b[31mé"]

    def input_fn(prompt):
        item = answers.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item

    buf = io.StringIO()
    ui = UI(console=Console(file=buf, width=80), input_fn=input_fn)
    assert ui.ask("Name") == "café"
    assert "couldn't read that text" in buf.getvalue()


def test_ctrl_c_and_ctrl_d_at_a_prompt_raise_user_quit():
    for exc in (KeyboardInterrupt, EOFError):
        def input_fn(prompt, exc=exc):
            raise exc()
        with pytest.raises(UserQuit):
            UI(console=Console(file=io.StringIO()), input_fn=input_fn).ask("?")


def test_make_input_safe_replaces_undecodable_bytes():
    stream = io.TextIOWrapper(io.BytesIO(b"caf\xe9 au lait\n"), encoding="utf-8")
    make_input_safe(stream)
    assert stream.readline() == "caf� au lait\n"
    make_input_safe(object())  # streams that can't be reconfigured are left alone


def test_can_hide_input_is_honest_without_a_terminal(monkeypatch):
    monkeypatch.setattr(ui_module.sys, "stdin", io.StringIO("piped"))
    assert UI(console=Console(file=io.StringIO())).can_hide_input() is False
    assert UI(console=Console(file=io.StringIO()), secret_fn=lambda p: "k").can_hide_input() is True


# ---------------------------------------------------------------------------
# Round 3: robust menus and control-code handling
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from rich.markup import escape  # noqa: E402

from gettowork.ui import _match_option, plain  # noqa: E402


@pytest.mark.parametrize("typed", ["²", "³", "¹", "①", "1²"])
def test_non_decimal_digits_at_a_menu_are_rejected_not_a_crash(typed):
    assert _match_option(typed, ["yes", "no"]) is None
    ui, _, buf = make_ui([typed, "no"])
    assert ui.choose("Enable?", [("yes", "y"), ("no", "n")]) == "no"
    assert "Pick one of the options" in buf.getvalue()


def test_other_scripts_decimal_digits_still_pick_by_number():
    assert _match_option("５", ["a", "b", "c", "d", "e"]) == "e"
    assert _match_option("٢", ["a", "b"]) == "b"


@pytest.mark.parametrize(
    "text",
    [
        "hello \x1b[/] world",
        "a \x9b[/b] c",
        "a \x1b[/bold] b",
        "x \x1b[mfoo] y",
        "\x1b[1mError\x1b[m: could not load model (see log]",
        "\x1bPxx\x1b[/]yy",
        "\x1b]0;title\x1b[/] after",
    ],
)
def test_escaped_text_with_control_codes_never_breaks_the_markup(text):
    ui, _, buf = make_ui([])
    for method in (ui.say, ui.narrate, ui.info, ui.warn, ui.error):
        method(escape(text))  # callers escape first; safe_text must not undo that
    out = buf.getvalue()
    assert "\x1b" not in out and "\x9b" not in out


def test_escaped_text_keeps_the_hidden_part_visible():
    ui, _, buf = make_ui([])
    ui.error(escape("\x1b[1mError\x1b[m: could not load model (see log]"))
    assert "could not load model (see log]" in buf.getvalue()


def test_plain_strips_control_codes_before_escaping():
    assert plain("x \x1b[mfoo] y") == "x foo] y"  # a real SGR reset is removed, the text kept
    assert plain("[bold]x \x1b]0;T\x07") == "\\[bold]x "
    assert safe_text("\x1b[31mred\x1b[0m \x1b]0;T\x07ok") == "red ok"


def test_table_cells_and_json_strip_control_codes():
    ui, _, _ = make_ui([])
    buf = io.StringIO()
    ui.console = Console(file=buf, width=80, force_terminal=True)
    ui.table("t", ["a"], [["\x1b]0;PWNED\x07cell \x1b[2J"]])
    ui.json({"k": "c1 \x9b31m here"})
    out = buf.getvalue()
    assert "PWNED" not in out and "\x1b]0" not in out and "\x1b[2J" not in out and "\x9b" not in out


def test_choose_accepts_aliases_for_back_out_words():
    ui, _, _ = make_ui(["back"])
    assert ui.choose("Enable?", [("yes", "y"), ("no", "n")], aliases={"back": "no"}) == "no"


def test_confirm_understands_back_out_and_quit_words():
    ui, _, _ = make_ui(["back", "cancel"])
    assert ui.confirm("Go ahead?", default=True) is False
    assert ui.confirm("Go ahead?", default=True) is False
    ui, _, _ = make_ui(["quit"])
    with pytest.raises(UserQuit):
        ui.confirm("Go ahead?", default=True)


def test_a_pasted_wall_of_digits_never_crashes_a_menu():
    from gettowork.ui import _match_option

    assert _match_option("9" * 5000, ["yes", "no"]) is None
    answers = iter(["9" * 5000, "2"])
    ui = UI(console=Console(file=io.StringIO(), width=80), input_fn=lambda prompt: next(answers))
    assert ui.choose("Pick", [("a", "A"), ("b", "B")]) == "b"
