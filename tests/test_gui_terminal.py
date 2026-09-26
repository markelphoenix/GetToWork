"""Tests for gettowork.gui.terminal: the pure-Python terminal model behind the game window.

The most important property: whatever rich prints to a real terminal - panels,
tables, Markdown, progress bars and spinners that redraw themselves - ends up
looking exactly the same here. Those tests pipe *real* rich output in and
compare the visible text with what rich itself says it drew.
"""

from __future__ import annotations

import io
import random
import re
import time

import pytest
from rich.cells import cell_len, get_character_cell_size
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TaskProgressColumn, TextColumn
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.tree import Tree

from gettowork.gui import terminal
from gettowork.gui.terminal import DEFAULT_STYLE, Style, TerminalBuffer, xterm_color
from gettowork.ui import UI

ESC = "\x1b"
CSI = ESC + "["


def fed(*chunks: str, **kwargs) -> TerminalBuffer:
    buf = TerminalBuffer(**kwargs)
    for chunk in chunks:
        buf.feed(chunk)
    return buf


def visible(buf: TerminalBuffer) -> list[str]:
    """The buffer's lines as plain text, without trailing blank lines."""
    lines = buf.plain_lines()
    while lines and not lines[-1]:
        lines.pop()
    return lines


def styles_of(buf: TerminalBuffer, index: int = 0) -> dict[str, Style]:
    return {text: style for text, style in buf.line(index)}


def terminal_console(width: int = 80, **kwargs) -> tuple[Console, io.StringIO]:
    """A console that writes to a StringIO exactly as it would to a truecolor terminal."""
    out = io.StringIO()
    console = Console(file=out, force_terminal=True, force_interactive=True, color_system="truecolor",
                      width=width, height=40, legacy_windows=False, soft_wrap=False, record=True, **kwargs)
    return console, out


def recorded_lines(console: Console) -> list[str]:
    lines = [line.rstrip() for line in console.export_text(clear=False).split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


# ---------------------------------------------------------------------------
# Plain text and control characters
# ---------------------------------------------------------------------------


def test_starts_with_one_empty_line_and_the_cursor_home():
    buf = TerminalBuffer()
    assert len(buf) == 1 and buf.cursor == (0, 0) and buf.lines == [[]]
    assert buf.text() == ""


def test_text_and_newlines():
    buf = fed("hello\nworld\n")
    assert visible(buf) == ["hello", "world"]
    assert buf.cursor == (2, 0)  # "\n" is a new line *and* a carriage return
    assert len(buf) == 3


def test_carriage_return_overwrites_the_start_of_the_line():
    assert visible(fed("hello world\rHELLO")) == ["HELLO world"]


def test_backspace_moves_left_and_stops_at_the_edge():
    assert visible(fed("abc\b\bX")) == ["aXc"]
    assert visible(fed("\b\b\bok")) == ["ok"]


def test_tab_moves_to_the_next_multiple_of_eight():
    buf = fed("a\tb\tc")
    assert buf.line_text(0) == "a       b       c"
    assert fed("12345678\tx").line_text(0) == "12345678        x"


def test_bell_nul_and_c1_controls_are_ignored():
    assert visible(fed("a\x07b\x00c\x85d\x9ce")) == ["abcde"]


def test_vertical_tab_and_form_feed_act_as_newlines():
    assert visible(fed("a\x0bb\x0cc")) == ["a", "b", "c"]


def test_writing_past_the_end_of_a_line_pads_with_spaces():
    buf = fed(f"ab{CSI}5Gx")
    assert buf.line_text(0) == "ab  x"
    assert buf.line(0)[-1] == ("ab  x", DEFAULT_STYLE)


# ---------------------------------------------------------------------------
# Wide and zero-width characters
# ---------------------------------------------------------------------------


def test_wide_characters_take_two_cells():
    buf = fed("中文x")
    assert buf.cursor == (0, 5)
    assert buf.line_text(0) == "中文x"


def test_overwriting_half_of_a_wide_character_blanks_the_other_half():
    assert fed("中文\rX").line_text(0) == "X 文"  # first half overwritten
    assert fed(f"中文{CSI}2GX").line_text(0) == " X文"  # second half overwritten
    buf = fed(f"ab中{CSI}2G世")  # a wide write over "b" and the first half of 中
    assert buf.line_text(0) == "a世 "


def test_zero_width_characters_join_the_previous_character():
    buf = fed("éx")  # e + combining acute accent
    assert buf.line_text(0) == "éx"
    assert buf.cursor == (0, 2)
    assert fed("́abc").line_text(0) == "abc"  # nothing to attach to: dropped


@pytest.mark.parametrize("text", ["🧠", "⚠️", "👨‍👩‍👧", "a‍b", "🇺🇸", "中文", "é"])
def test_cursor_advances_exactly_as_rich_measures(text):
    buf = fed(text)
    assert buf.cursor == (0, cell_len(text))
    assert buf.line_text(0) == text


def test_the_fast_path_only_covers_single_width_characters():
    # Every character the fast path writes one cell at a time must be one cell wide to rich too.
    char_class = terminal._NARROW_RUN_RE.pattern[1:-2]
    single = re.compile(f"[{char_class}]")
    for code in range(0x20, 0x2600):
        ch = chr(code)
        if single.fullmatch(ch):
            assert get_character_cell_size(ch) == 1, hex(code)


# ---------------------------------------------------------------------------
# Styles (SGR)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code, field", [
    ("1", "bold"), ("2", "dim"), ("3", "italic"), ("4", "underline"), ("7", "reverse"), ("9", "strike"),
    ("21", "underline"),
])
def test_sgr_attributes(code, field):
    style = styles_of(fed(f"a{CSI}{code}mb"))["b"]
    assert getattr(style, field) is True
    assert styles_of(fed(f"a{CSI}{code}mb"))["a"] == DEFAULT_STYLE


@pytest.mark.parametrize("on, off, field", [
    ("1", "22", "bold"), ("2", "22", "dim"), ("3", "23", "italic"), ("4", "24", "underline"),
    ("7", "27", "reverse"), ("9", "29", "strike"), ("4", "4:0", "underline"),
])
def test_sgr_attributes_turn_off(on, off, field):
    buf = fed(f"{CSI}{on}ma{CSI}{off}mb")
    assert getattr(styles_of(buf)["a"], field) is True
    assert getattr(styles_of(buf)["b"], field) is False


def test_standard_and_bright_colours_are_palette_indices():
    buf = fed(f"{CSI}31;42ma{CSI}91;102mb{CSI}39;49mc")
    s = styles_of(buf)
    assert (s["a"].fg, s["a"].bg) == (1, 2)
    assert (s["b"].fg, s["b"].bg) == (9, 10)
    assert s["c"] == DEFAULT_STYLE


def test_256_colours_and_truecolor_semicolon_form():
    buf = fed(f"{CSI}38;5;208;48;5;237ma{CSI}38;2;255;136;0;48;2;1;2;3mb")
    s = styles_of(buf)
    assert (s["a"].fg, s["a"].bg) == (208, 237)
    assert (s["b"].fg, s["b"].bg) == ("#ff8800", "#010203")


def test_truecolor_colon_forms():
    s = styles_of(fed(f"{CSI}38:2::10:20:30ma{CSI}38:2:40:50:60mb{CSI}48:5:99mc"))
    assert s["a"].fg == "#0a141e" and s["b"].fg == "#28323c" and s["c"].bg == 99


def test_invalid_colours_are_ignored_but_later_codes_still_apply():
    s = styles_of(fed(f"{CSI}38;5;999;1ma{CSI}0;38;2;1;2mb{CSI}0;38;7;1mc"))
    assert s["a"].fg is None and s["a"].bold is True
    assert s["b"].fg is None
    assert s["c"].fg is None and s["c"].bold is True  # 38;7 is unknown: skip "7", then "1" is bold


def test_reset_clears_every_attribute():
    buf = fed(f"{CSI}1;3;4;7;9;31;44ma{CSI}0mb{CSI}1mc{CSI}md")
    s = styles_of(buf)
    assert s["b"] == DEFAULT_STYLE and s["d"] == DEFAULT_STYLE and s["c"].bold


def test_unknown_sgr_numbers_are_ignored():
    buf = fed(f"{CSI}1;12345678901234567890;53;3ma")
    assert styles_of(buf)["a"].bold and styles_of(buf)["a"].italic


def test_a_letter_ends_a_sequence_like_in_a_terminal():
    buf = fed(f"{CSI}1;xa")  # "x" is a final byte: an unknown sequence, dropped whole
    assert visible(buf) == ["a"] and styles_of(buf)["a"] == DEFAULT_STYLE


def test_consecutive_cells_in_one_style_form_one_run():
    buf = fed(f"ab{CSI}1mcd{CSI}0mef")
    assert [text for text, _ in buf.line(0)] == ["ab", "cd", "ef"]


def test_xterm_palette():
    assert xterm_color(16) == "#000000"
    assert xterm_color(196) == "#ff0000"
    assert xterm_color(231) == "#ffffff"
    assert xterm_color(232) == "#080808"
    assert xterm_color(255) == "#eeeeee"
    with pytest.raises(ValueError):
        xterm_color(3)


# ---------------------------------------------------------------------------
# OSC: hyperlinks and the rest
# ---------------------------------------------------------------------------


def test_osc8_links_mark_their_text():
    buf = fed(f"see {ESC}]8;id=1;https://example.com{ESC}\\here{ESC}]8;;{ESC}\\ now")
    s = styles_of(buf)
    assert s["here"].link == "https://example.com"
    assert s["see "].link is None and s[" now"].link is None


def test_osc8_with_bel_terminators():
    buf = fed(f"{ESC}]8;;https://a.example\x07x{ESC}]8;;\x07y")
    assert styles_of(buf)["x"].link == "https://a.example" and styles_of(buf)["y"].link is None


def test_a_style_reset_inside_a_link_keeps_the_link():
    buf = fed(f"{ESC}]8;;https://x.example{ESC}\\{CSI}1ma{CSI}0mb{ESC}]8;;{ESC}\\c")
    s = styles_of(buf)
    assert s["a"].link == s["b"].link == "https://x.example" and s["c"].link is None


def test_window_titles_and_other_osc_are_ignored():
    buf = fed(f"a{ESC}]0;Evil title\x07b{ESC}]52;c;aGk={ESC}\\c")
    assert visible(buf) == ["abc"]
    assert all(style.link is None for _, style in buf.line(0))


def test_overlong_osc_is_dropped_and_parsing_recovers():
    buf = fed(f"{ESC}]8;;https://x/" + "a" * (terminal.MAX_OSC_LENGTH + 10) + f"{ESC}\\ok")
    assert visible(buf) == ["ok"] and styles_of(buf)["ok"].link is None


def test_a_link_with_control_characters_is_not_kept():
    buf = fed(f"{ESC}]8;;https://x\x01y{ESC}\\z")
    assert styles_of(buf)["z"].link is None


def test_an_esc_inside_an_osc_that_is_not_st_ends_it_and_starts_a_new_sequence():
    buf = fed(f"{ESC}]8;;https://a.example{ESC}[1mb")
    assert styles_of(buf)["b"].link == "https://a.example" and styles_of(buf)["b"].bold


# ---------------------------------------------------------------------------
# Sequences that are ignored
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sequence", [
    f"{CSI}?25l", f"{CSI}?25h", f"{CSI}?1049h", f"{CSI}?2004h", f"{CSI}>c", f"{CSI}5n", f"{CSI}1 q",
    f"{CSI}2;20r", f"{CSI}3S", f"{ESC}(B", f"{ESC})0", f"{ESC}=", f"{ESC}>", f"{ESC}Pq#0;2;0;0;0{ESC}\\",
    f"{ESC}_hidden{ESC}\\", f"{ESC}^pm{ESC}\\", f"{ESC}Xsos\x9c", f"{ESC}\\", f"{CSI}3J",
])
def test_unknown_and_private_sequences_are_dropped(sequence):
    assert visible(fed(f"ab{sequence}cd")) == ["abcd"]


def test_can_and_sub_cancel_a_sequence():
    assert visible(fed(f"a{CSI}12\x18b{ESC}]8;;x\x1ac")) == ["abc"]


def test_an_overlong_csi_is_skipped():
    assert visible(fed(f"a{CSI}" + "1;" * 100 + "mb")) == ["ab"]
    assert styles_of(fed(f"a{CSI}" + "1;" * 100 + "mb"))["ab"] == DEFAULT_STYLE


def test_controls_inside_a_csi_are_obeyed():
    buf = fed(f"ab{CSI}1\nmc")
    assert visible(buf) == ["ab", "c"] and styles_of(buf, 1)["c"].bold


def test_a_new_escape_interrupts_an_unfinished_one():
    buf = fed(f"a{CSI}1;{CSI}3mb")
    assert styles_of(buf)["b"].italic and not styles_of(buf)["b"].bold


# ---------------------------------------------------------------------------
# Cursor movement and erasing
# ---------------------------------------------------------------------------


def test_cursor_up_down_forward_back():
    buf = fed(f"line0\nline1\nline2{CSI}2AX{CSI}BY{CSI}3DZ{CSI}2CW")
    # X at the end of line0; down to line1 (padding to the column), Y; back 3, Z; forward 2, W.
    assert visible(buf) == ["line0X", "lineZ YW", "line2"]


def test_cursor_up_and_down_with_zero_or_no_argument_move_one_line():
    buf = fed(f"a\nb\nc{CSI}Ax{CSI}0Ay")
    assert visible(buf) == ["a y", "bx", "c"]


def test_cursor_up_stops_at_the_first_line():
    buf = fed(f"a\nb{CSI}99AX")
    assert visible(buf) == ["aX", "b"]


def test_cursor_down_creates_lines_only_within_the_screen():
    buf = fed(f"a{CSI}3Bb", rows=24)
    assert visible(buf) == ["a", "", "", " b"]
    full = fed("\n".join(str(i) for i in range(30)) + f"{CSI}5Bx", rows=10)
    assert len(full) == 30 and full.cursor == (29, 3)  # already at the bottom of the screen


def test_next_and_previous_line_and_column_absolute():
    buf = fed(f"aaaa\nbbbb{CSI}F1{CSI}E2{CSI}3G3")
    assert visible(buf) == ["1aaa", "2b3b"]


def test_cursor_position_is_relative_to_the_screen():
    buf = fed("\n".join(f"line{i}" for i in range(10)) + f"{CSI}1;1HX", rows=4)
    assert buf.plain_lines()[6] == "Xine6"  # the screen is the last 4 lines: 6-9
    buf.feed(f"{CSI}2;3HY{CSI}HZ")
    assert buf.plain_lines()[6:8] == ["Zine6", "liYe7"]
    buf.feed(f"{CSI}3dQ")  # line 3 of the screen, same column (after the Z)
    assert buf.plain_lines()[8] == "lQne8"


def test_column_moves_are_bounded():
    buf = fed(f"{CSI}999999Cx")
    assert buf.cursor == (0, terminal.MAX_COLUMN + 1)


def test_erase_line_modes():
    assert visible(fed(f"hello world{CSI}6D{CSI}K")) == ["hello"]
    assert visible(fed(f"hello world{CSI}6D{CSI}0K")) == ["hello"]
    assert fed(f"hello world{CSI}7G{CSI}1K").line_text(0) == "       orld"
    buf = fed(f"hello{CSI}2K")
    assert buf.line_text(0) == "" and buf.cursor == (0, 5)


def test_erase_characters():
    assert fed(f"abcdef{CSI}5D{CSI}2X").line_text(0) == "a  def"
    assert fed(f"abc{CSI}10X").line_text(0) == "abc"  # nothing under the cursor


def test_erase_display_below_and_above():
    below = fed(f"one\ntwo\nthree{CSI}2A{CSI}2G{CSI}J")
    assert visible(below) == ["o"]
    above = fed(f"one\ntwo\nthree{CSI}A{CSI}2G{CSI}1J", rows=10)
    assert visible(above) == ["", "  o", "three"]


def test_clear_screen_keeps_the_transcript_and_starts_a_blank_screen():
    buf = fed("\n".join(f"old{i}" for i in range(5)), rows=4)
    buf.feed(f"{CSI}2J{CSI}Hnew")
    lines = buf.plain_lines()
    assert lines[:5] == [f"old{i}" for i in range(5)]  # pushed up, not destroyed
    assert lines[5] == "new" and len(buf) == 5 + 4
    assert fed(f"{CSI}2Jx").plain_lines() == ["x"]  # an empty screen: nothing to push


def test_save_and_restore_cursor():
    buf = fed(f"ab{ESC}7cd\nef{ESC}8X")
    assert visible(buf) == ["abXd", "ef"]
    buf = fed(f"ab{CSI}s{CSI}1mcd{CSI}uX")
    assert visible(buf) == ["abXd"] and styles_of(buf)["Xd"].bold  # CSI u restores the position only
    buf = fed(f"ab{ESC}7{CSI}1mcd{ESC}8X")
    assert styles_of(buf)["abX"] == DEFAULT_STYLE  # ESC 8 restores the style too
    assert visible(fed(f"abc{CSI}uX")) == ["Xbc"]  # nothing saved: start of the line


def test_escape_index_and_reverse_index():
    buf = fed(f"ab{ESC}Dc{ESC}Md{ESC}Ee")  # down (same column), up, then down to the next line's start
    assert visible(buf) == ["ab d", "e c"]


def test_full_reset_resets_the_style_but_keeps_the_transcript():
    buf = fed(f"{CSI}1mbold{ESC}cplain")
    assert styles_of(buf)["bold"].bold and styles_of(buf)["plain"] == DEFAULT_STYLE


# ---------------------------------------------------------------------------
# Split sequences, dirty tracking, scrollback
# ---------------------------------------------------------------------------

SAMPLE = (
    f"{CSI}1;38;2;255;0;128mTitle{CSI}0m 中文 🧠\n"
    f"{ESC}]8;id=7;https://example.com/path{ESC}\\link{ESC}]8;;{ESC}\\ {ESC}]0;title\x07after\n"
    f"{CSI}?25lprogress 10%\r{CSI}2Kprogress 50%\r{CSI}2Kdone{CSI}?25h\n"
    f"{CSI}38:5:33mblue{CSI}m {ESC}Pdcs{ESC}\\ é ok\n"
)


def test_feeding_in_two_parts_at_every_split_point_gives_the_same_result():
    whole = fed(SAMPLE)
    for cut in range(1, len(SAMPLE)):
        parts = fed(SAMPLE[:cut], SAMPLE[cut:])
        assert parts.lines == whole.lines, cut
        assert parts.cursor == whole.cursor


def test_feeding_one_character_at_a_time_gives_the_same_result():
    whole = fed(SAMPLE)
    single = TerminalBuffer()
    for ch in SAMPLE:
        single.feed(ch)
    assert single.lines == whole.lines and single.cursor == whole.cursor


def test_empty_feed_is_harmless():
    buf = fed("a")
    buf.take_dirty()
    buf.feed("")
    assert buf.take_dirty() == [] and visible(buf) == ["a"]


def test_dirty_lines_are_reported_once():
    buf = TerminalBuffer()
    buf.feed("a\nb\nc")
    assert buf.take_dirty() == [0, 1, 2]
    assert buf.take_dirty() == []
    buf.feed(f"{CSI}Ax")  # only line 1 changes
    assert buf.dirty == {1}
    assert buf.take_dirty() == [1]
    buf.feed(f"{CSI}B{CSI}3D")  # moving without writing changes nothing visible
    assert buf.take_dirty() == []
    buf.feed(f"{CSI}2B")  # ...but moving onto a line of the screen that wasn't there yet adds it
    assert buf.take_dirty() == [3, 4] and len(buf) == 5


def test_a_spinner_redraw_dirties_one_line_only():
    buf = fed("header\n")
    buf.take_dirty()
    for frame in "⠋⠙⠹⠸":
        buf.feed(f"\r{CSI}2K{frame} waiting")
        assert buf.take_dirty() == [1]
    assert visible(buf) == ["header", "⠸ waiting"]


def test_scrollback_is_bounded():
    buf = TerminalBuffer(max_lines=100)
    for i in range(1000):
        buf.feed(f"line {i}\n")
    assert len(buf) <= 100
    assert buf.trimmed_total == 1001 - len(buf)
    assert buf.plain_lines()[-2] == "line 999"
    assert buf.cursor == (len(buf) - 1, 0)
    assert all(0 <= i < len(buf) for i in buf.take_dirty())


def test_trimming_drops_a_chunk_at_a_time_and_shifts_dirty_indices():
    buf = TerminalBuffer(max_lines=1000)
    buf.feed("\n".join(str(i) for i in range(1000)))
    buf.take_dirty()
    assert buf.trimmed_total == 0 and len(buf) == 1000
    buf.feed("\nnew")
    assert buf.trimmed_total == 20 and len(buf) == 981  # 2% of the limit at once
    assert buf.take_dirty() == [980]
    assert buf.plain_lines()[0] == "20" and buf.plain_lines()[-1] == "new"


def test_a_tiny_scrollback_still_works():
    buf = TerminalBuffer(max_lines=2)
    buf.feed("a\nb\nc\nd")
    assert visible(buf) == ["c", "d"]


def test_reset_counts_as_trimming_everything():
    buf = fed("a\nb\nc")
    buf.reset()
    assert buf.trimmed_total == 3 and len(buf) == 1 and buf.cursor == (0, 0) and buf.text() == ""
    buf.feed("x")
    assert visible(buf) == ["x"]


def test_random_garbage_never_breaks_the_buffer():
    rng = random.Random(1234)
    alphabet = ["\x1b", "[", "]", ";", ":", "?", "\\", "\x07", "\n", "\r", "\b", "\t", "m", "H", "A", "B", "J",
                "K", "8", "2", "38", "5", "0", "a", "中", "🧠", "́", "\x9c", "\x18", "P", "(", " ", "~"]
    for _ in range(200):
        buf = TerminalBuffer(max_lines=50, rows=rng.randint(1, 30))
        for _ in range(rng.randint(1, 20)):
            buf.feed("".join(rng.choice(alphabet) for _ in range(rng.randint(0, 60))))
        row, col = buf.cursor
        assert 0 <= row < len(buf) <= 50 and 0 <= col <= terminal.MAX_COLUMN
        for i in range(len(buf)):
            assert len(buf._chars[i]) == len(buf._styles[i])
            buf.line(i)
        buf.feed("\x18\r\nstill fine")  # CAN ends any unfinished sequence
        assert buf.line_text(buf.cursor[0]).startswith("still fine")


# ---------------------------------------------------------------------------
# Real rich output
# ---------------------------------------------------------------------------


def _static_renderables():
    table = Table(title="Best picks for this computer")
    table.add_column("#", justify="right")
    table.add_column("Model")
    table.add_column("Size", justify="right")
    table.add_row("1", "Qwen3 4B ⚡", "2.5 GB")
    table.add_row("2", "Gemma 3 1B 🧠", "0.8 GB")
    table.add_row("3", "中文 model with a very long name that has to wrap somewhere", "12 GB")
    tree = Tree("root")
    tree.add("branch").add("leaf")
    return [
        Panel("[bold magenta]GET TO WORK[/]\nA farcical race - [link=https://example.com]link[/link]",
              title="Hi", border_style="magenta", padding=(1, 2)),
        table,
        Markdown("# Heading\n\nSome *emphasis* and `code`.\n\n* one\n* two\n\n> a quote\n\n```\nprint('x')\n```"),
        Rule("[bold]A rule[/bold]"),
        Syntax('{"key": [1, 2, 3]}', "json", word_wrap=True),
        tree,
        "[on grey23]background[/] [reverse]reverse[/] [dim]dim[/] [strike]strike[/] ✓ → …",
    ]


@pytest.mark.parametrize("width", [40, 80, 100])
def test_static_rich_output_looks_exactly_as_rich_drew_it(width):
    console, out = terminal_console(width)
    for renderable in _static_renderables():
        console.print(renderable)
    buf = fed(out.getvalue())
    assert visible(buf) == recorded_lines(console)


def test_rich_colours_and_links_arrive_as_styles():
    console, out = terminal_console(60)
    console.print("[bold magenta]GET[/] [#ff8800 on grey23]x[/] [link=https://example.com]here[/link]")
    s = styles_of(fed(out.getvalue()))
    assert s["GET"].bold and s["GET"].fg == 5
    assert s["x"].fg == "#ff8800" and s["x"].bg == 237
    assert s["here"].link == "https://example.com"


def test_a_rich_progress_run_leaves_exactly_its_final_frame():
    console, out = terminal_console(70)
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(), TaskProgressColumn())
    console.print("before")
    with Progress(*columns, console=console, auto_refresh=False) as progress:
        task = progress.add_task("download", total=20)
        for step in range(20):
            progress.advance(task)
            progress.refresh()
            if step == 7:
                console.print("a message printed mid-way")
    console.print("after")
    buf = fed(out.getvalue())

    reference, _ = terminal_console(70)
    reference.print(progress.make_tasks_table(progress.tasks))
    final_frame = recorded_lines(reference)
    assert visible(buf) == ["before", "a message printed mid-way", *final_frame, "after"]
    assert "20/20" in final_frame[0]


def test_a_progress_with_several_tasks_redraws_in_place():
    console, out = terminal_console(60)
    with Progress(TextColumn("{task.description}"), MofNCompleteColumn(), console=console,
                  auto_refresh=False) as progress:
        tasks = [progress.add_task(f"part {i}", total=3) for i in range(3)]
        for _ in range(3):
            for task in tasks:
                progress.advance(task)
                progress.refresh()
    assert visible(fed(out.getvalue())) == ["part 0 3/3", "part 1 3/3", "part 2 3/3"]


def test_a_status_spinner_leaves_no_trace():
    console, out = terminal_console(60)
    console.print("one")
    with console.status("working hard...", refresh_per_second=50):
        time.sleep(0.15)
        console.print("printed while spinning")
        time.sleep(0.15)
    console.print("two")
    raw = out.getvalue()
    assert "working hard" in raw  # it really was drawn...
    assert visible(fed(raw)) == ["one", "printed while spinning", "two"]  # ...and really erased


def test_a_live_table_ends_as_its_last_update():
    console, out = terminal_console(50)

    def table(n: int) -> Table:
        t = Table("step", "value")
        for i in range(n):
            t.add_row(str(i), "x" * i)
        return t

    with Live(table(1), console=console, auto_refresh=False) as live:
        for n in range(2, 6):
            live.update(table(n), refresh=True)
    reference, _ = terminal_console(50)
    reference.print(table(5))
    assert visible(fed(out.getvalue())) == recorded_lines(reference)


def test_the_games_own_ui_output_renders_like_rich_says():
    console, out = terminal_console(80)
    ui = UI(console=console, input_fn=lambda prompt: "")
    ui.heading("Your computer")
    ui.narrate("You wake up on the kitchen floor, cuddling a baguette. " * 3, title="Good morning!")
    ui.teach("memory", "* one\n* **two**")
    ui.table("Models", ["#", "Name"], [["1", "Qwen3 4B"]])
    ui.info("info")
    ui.warn("warn")
    static = recorded_lines(console)
    with ui.download_progress("Downloading model.gguf", 1000) as advance:
        for _ in range(10):
            advance(100)
    with ui.status("Thinking..."):
        time.sleep(0.05)
    ui.success("done")
    lines = visible(fed(out.getvalue()))
    assert lines[:len(static)] == static
    assert len(lines) == len(static) + 2  # one final progress line, then "OK done"
    assert lines[len(static)].startswith("Downloading model.gguf") and "1.0/1.0 kB" in lines[len(static)]
    assert lines[-1] == "OK done"


def test_ten_redraws_a_second_is_cheap():
    # A spinner or progress bar redraws ~10 times a second; each redraw must
    # touch one line and cost next to nothing, even with a long transcript.
    console, out = terminal_console(100)
    for i in range(2000):
        console.print(f"[bold]line {i}[/bold] " + "text " * 15)
    buf = fed(out.getvalue())
    buf.take_dirty()
    progress_out = io.StringIO()
    progress_console = Console(file=progress_out, force_terminal=True, force_interactive=True,
                               color_system="truecolor", width=100, height=40, legacy_windows=False)
    with Progress(TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn(),
                  console=progress_console, auto_refresh=False) as progress:
        task = progress.add_task("downloading", total=600)
        frames = []
        for _ in range(600):
            progress.advance(task)
            progress.refresh()
            frames.append(progress_out.getvalue())
            progress_out.seek(0)
            progress_out.truncate()
    start = time.perf_counter()
    max_dirty = 0
    for frame in frames:
        buf.feed(frame)
        max_dirty = max(max_dirty, len(buf.take_dirty()))
    elapsed = time.perf_counter() - start
    assert max_dirty <= 2
    assert elapsed < 3.0  # 600 frames = a minute of redraws at 10/s; typically ~0.1 s
    assert visible(buf)[-1].startswith("downloading") and "600/600" in visible(buf)[-1]


# ---------------------------------------------------------------------------
# Swapping characters the window can't draw
# ---------------------------------------------------------------------------


def test_line_can_swap_whole_wide_characters_keeping_the_columns():
    buf = fed("a\U0001f9e0b\u2502 \u00e9\n")
    assert buf.line_text(0) == "a\U0001f9e0b\u2502 \u00e9"
    swapped = buf.line(0, substitute=lambda cluster, width: "#" * width if cluster == "\U0001f9e0" else None)
    assert "".join(text for text, _style in swapped) == "a##b\u2502 \u00e9"
    asked = []
    buf.line(0, substitute=lambda cluster, width: asked.append((cluster, width)))
    assert asked == [("\U0001f9e0", 2), ("\u2502", 1), ("\u00e9", 1)]  # only non-ASCII characters are asked about
    assert buf.line(0, substitute=lambda cluster, width: "") == [("ab ", DEFAULT_STYLE)]


def test_a_failing_substitute_keeps_the_character():
    buf = fed("x\U0001f600y")
    assert buf.line(0, substitute=lambda cluster, width: 1 / 0) == buf.line(0)


def test_swapped_characters_keep_their_style():
    buf = fed(f"{CSI}31m\U0001f9e0{CSI}0m ok")
    runs = buf.line(0, substitute=lambda cluster, width: "+" + " " * (width - 1))
    assert runs[0] == ("+ ", Style(fg=1)) and runs[1] == (" ok", DEFAULT_STYLE)
