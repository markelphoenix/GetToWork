"""Tests for gettowork.gui.bridge (and the UI hooks it relies on) - plain threads, no Tk.

The window's side of the bridge is played here by a small "fake window"
thread that polls the bridge and answers questions, exactly as the Tk code
does every 30 ms. That lets the whole real game run through the bridge
without a display.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Optional

import pytest
from rich.console import Console

from gettowork.gui.app import SelftestPlayer
from gettowork.gui.bridge import (
    CHOICES,
    FINISHED,
    OUTPUT,
    PROMPT,
    GuiBridge,
    Prompt,
    is_openable_url,
)
from gettowork.gui.terminal import TerminalBuffer
from gettowork.ui import UI, UserChoseQuit, UserQuit


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path / "home"))
    for var in ("GETTOWORK_MODELS_DIR", "TYPESAFE_API_KEY"):
        monkeypatch.delenv(var, raising=False)


def wait_for(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out waiting")
        time.sleep(0.002)


def next_prompt(bridge: GuiBridge, timeout: float = 5.0) -> Prompt:
    """Poll like the window does until a question arrives."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for kind, payload in bridge.poll():
            if kind == PROMPT:
                return payload
        time.sleep(0.002)
    raise AssertionError("no prompt arrived")


class GameThread:
    """Runs ``fn`` on a worker thread and keeps its result or exception."""

    def __init__(self, fn: Callable[[], object]) -> None:
        self.result: object = None
        self.error: Optional[BaseException] = None

        def run() -> None:
            try:
                self.result = fn()
            except BaseException as exc:  # noqa: BLE001 - reported by the test
                self.error = exc

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def join(self, timeout: float = 5.0) -> None:
        self.thread.join(timeout)
        assert not self.thread.is_alive(), "the game thread is stuck"


# ---------------------------------------------------------------------------
# The stream
# ---------------------------------------------------------------------------


def test_the_stream_looks_like_a_utf8_terminal():
    bridge = GuiBridge()
    stream = bridge.stream
    assert stream.isatty() is True
    assert stream.encoding == "utf-8"
    assert stream.writable() and not stream.readable() and not stream.seekable()
    assert stream.write("héllo ✓") == 7
    stream.flush()
    assert bridge.poll() == [(OUTPUT, "héllo ✓")]


def test_the_stream_refuses_bytes_and_ignores_empty_writes():
    bridge = GuiBridge()
    with pytest.raises(TypeError):
        bridge.stream.write(b"bytes")  # type: ignore[arg-type]
    assert bridge.stream.write("") == 0
    assert bridge.poll() == []


def test_print_and_close_work_like_on_a_file():
    bridge = GuiBridge()
    print("one", "two", file=bridge.stream)
    bridge.stream.close()  # a stray close() must not break later prints
    assert not bridge.stream.closed
    print("three", file=bridge.stream)
    assert bridge.poll() == [(OUTPUT, "one two\nthree\n")]


def test_rich_prints_colours_to_the_stream():
    bridge = GuiBridge()
    console = Console(file=bridge.stream, force_terminal=True, color_system="truecolor", width=40,
                      legacy_windows=False)
    console.print("[bold red]hi[/]")
    [(kind, text)] = bridge.poll()
    assert kind == OUTPUT and "\x1b[1;31mhi" in text


def test_consecutive_output_is_joined_but_order_is_kept():
    bridge = GuiBridge()
    bridge.stream.write("a")
    bridge.stream.write("b")
    bridge.show_choices([("y", "Yes"), ("n", "No")])
    bridge.stream.write("c")
    bridge.finish(3)
    assert bridge.poll() == [(OUTPUT, "ab"), (CHOICES, (("y", "Yes"), ("n", "No"))), (OUTPUT, "c"), (FINISHED, 3)]
    assert bridge.poll() == []


def test_many_writer_threads_lose_nothing():
    bridge = GuiBridge()

    def writer(tag: str) -> None:
        for i in range(500):
            bridge.stream.write(f"{tag}{i};")

    threads = [threading.Thread(target=writer, args=(chr(65 + n),)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    text = "".join(payload for _kind, payload in bridge.poll())
    parts = [p for p in text.split(";") if p]
    assert len(parts) == 8 * 500 and len(set(parts)) == 8 * 500


# ---------------------------------------------------------------------------
# Questions and answers
# ---------------------------------------------------------------------------


def test_request_line_blocks_until_the_window_answers():
    bridge = GuiBridge()
    game = GameThread(lambda: bridge.request_line("Your plan?"))
    prompt = next_prompt(bridge)
    assert prompt.text == "Your plan?" and prompt.secret is False
    time.sleep(0.05)
    assert game.thread.is_alive()  # still waiting
    assert bridge.pending == prompt
    assert bridge.submit("I cycle", prompt.id) is True
    game.join()
    assert game.result == "I cycle" and bridge.pending is None


def test_secret_questions_are_marked():
    bridge = GuiBridge()
    game = GameThread(lambda: bridge.request_line("Key", secret=True))
    prompt = next_prompt(bridge)
    assert prompt.secret is True
    bridge.submit("tsk-123", prompt.id)
    game.join()
    assert game.result == "tsk-123"


def test_an_answer_with_no_question_waiting_is_not_kept():
    bridge = GuiBridge()
    assert bridge.submit("too early") is False
    game = GameThread(lambda: bridge.request_line("Q?"))
    prompt = next_prompt(bridge)
    assert bridge.submit("real", prompt.id)
    game.join()
    assert game.result == "real"  # the early Enter didn't answer it


def test_a_stale_answer_is_refused():
    bridge = GuiBridge()
    game = GameThread(lambda: (bridge.request_line("first"), bridge.request_line("second")))
    first = next_prompt(bridge)
    assert bridge.submit("1", first.id)
    second = next_prompt(bridge)
    assert bridge.submit("late click for the first question", first.id) is False
    assert bridge.submit("2", second.id) is True
    game.join()
    assert game.result == ("1", "2")


def test_only_the_first_of_two_quick_answers_counts():
    bridge = GuiBridge()
    game = GameThread(lambda: bridge.request_line("Q?"))
    prompt = next_prompt(bridge)
    assert bridge.submit("first", prompt.id) is True
    assert bridge.submit("second", prompt.id) is False
    game.join()
    assert game.result == "first"


def test_an_empty_answer_is_an_answer():
    bridge = GuiBridge()
    game = GameThread(lambda: bridge.request_line("Press Enter"))
    bridge.submit("", next_prompt(bridge).id)
    game.join()
    assert game.result == ""


def test_questions_from_two_threads_take_turns():
    bridge = GuiBridge()
    games = [GameThread(lambda n=n: bridge.request_line(f"q{n}")) for n in range(2)]
    answered = []
    for _ in range(2):
        prompt = next_prompt(bridge)
        answered.append(prompt.text)
        bridge.submit(prompt.text.upper(), prompt.id)
    for game in games:
        game.join()
    assert sorted(answered) == ["q0", "q1"]
    assert sorted(g.result for g in games) == ["Q0", "Q1"]


def test_closing_the_window_wakes_a_waiting_question_with_eof():
    bridge = GuiBridge()
    game = GameThread(lambda: bridge.request_line("Q?"))
    next_prompt(bridge)
    bridge.close()
    game.join()
    assert isinstance(game.error, EOFError)
    assert bridge.closed


def test_after_closing_every_question_raises_eof_at_once():
    bridge = GuiBridge()
    bridge.close()
    started = time.monotonic()
    with pytest.raises(EOFError):
        bridge.request_line("Q?")
    assert time.monotonic() - started < 0.2
    bridge.stream.write("goodbye!")  # the game's goodbye still reaches the transcript
    bridge.show_choices([("y", "Yes")])  # (no more buttons, though)
    bridge.finish(0)
    assert bridge.poll() == [(OUTPUT, "goodbye!"), (FINISHED, 0)]
    assert bridge.submit("x") is False


def test_after_detaching_output_is_dropped():
    bridge = GuiBridge()
    bridge.detach()
    assert bridge.closed
    bridge.stream.write("nobody is watching")
    with pytest.raises(EOFError):
        bridge.request_line("Q?")
    bridge.finish(0)
    assert bridge.poll() == [(FINISHED, 0)]


def test_set_size_records_the_view_size():
    bridge = GuiBridge(columns=100, rows=32)
    assert (bridge.columns, bridge.rows) == (100, 32)
    bridge.set_size(120, 40)
    assert (bridge.columns, bridge.rows) == (120, 40)
    bridge.set_size(0, -3)
    assert (bridge.columns, bridge.rows) == (1, 1)


# ---------------------------------------------------------------------------
# Links
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url, ok", [
    ("https://typesafe.ai/docs", True), ("http://localhost:8080/x", True), ("HTTPS://Example.com", True),
    ("file:///etc/passwd", False), ("javascript:alert(1)", False), ("https://", False), ("", False),
    ("ftp://example.com", False), ("https://exa\nmple.com", False), (None, False), ("https://" + "a" * 5000, False),
])
def test_only_web_links_are_openable(url, ok):
    assert is_openable_url(url) is ok


def test_open_url_uses_the_opener_for_web_links_only():
    opened = []
    bridge = GuiBridge(opener=lambda url: opened.append(url) or True)
    assert bridge.open_url("https://typesafe.ai") is True
    assert bridge.open_url("file:///secret") is False
    assert opened == ["https://typesafe.ai"]


def test_open_url_reports_failures():
    def broken(url):
        raise OSError("no browser")

    assert GuiBridge(opener=broken).open_url("https://x.example") is False
    assert GuiBridge(opener=lambda url: None).open_url("https://x.example") is False


# ---------------------------------------------------------------------------
# The UI hooks the window uses (ui.py: choices_fn, hides_input)
# ---------------------------------------------------------------------------


def bridged_ui(bridge: GuiBridge, **kwargs) -> UI:
    console = Console(file=bridge.stream, force_terminal=True, color_system="truecolor", width=80,
                      legacy_windows=False)
    return UI(console=console, input_fn=bridge.request_line,
              secret_fn=lambda p: bridge.request_line(p, secret=True),
              choices_fn=bridge.show_choices, hides_input=True, **kwargs)


def answer_with_events(bridge: GuiBridge, answer: str) -> list:
    """Poll until a question arrives, answer it, and return every event seen up to it."""
    seen = []
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        for kind, payload in bridge.poll():
            seen.append((kind, payload))
            if kind == PROMPT:
                bridge.submit(answer, payload.id)
                return seen
        time.sleep(0.002)
    raise AssertionError("no prompt arrived")


def choices_events(events: list) -> list:
    return [payload for kind, payload in events if kind == CHOICES]


def test_choose_shows_buttons_before_asking_and_hides_them_after():
    bridge = GuiBridge()
    ui = bridged_ui(bridge)
    options = [("retry", "Try again"), ("quit", "Stop for now")]
    game = GameThread(lambda: ui.choose("What now?", options, default="retry"))
    before = answer_with_events(bridge, "quit")
    game.join()
    assert game.result == "quit"
    assert choices_events(before) == [tuple(options)]
    assert choices_events(bridge.poll()) == [()]


def test_a_wrong_menu_answer_asks_again_with_the_buttons_still_up():
    bridge = GuiBridge()
    ui = bridged_ui(bridge)
    game = GameThread(lambda: ui.choose("Pick", [("a", "A"), ("b", "B")]))
    first = answer_with_events(bridge, "nonsense")
    second = answer_with_events(bridge, "2")
    game.join()
    assert game.result == "b"
    assert choices_events(first) == [(("a", "A"), ("b", "B"))]
    assert () not in choices_events(second)  # not hidden in between
    assert choices_events(bridge.poll()) == [()]


def test_confirm_shows_yes_and_no():
    bridge = GuiBridge()
    ui = bridged_ui(bridge)
    game = GameThread(lambda: ui.confirm("Play again?", default=True))
    events = answer_with_events(bridge, "n")
    game.join()
    assert game.result is False
    assert choices_events(events) == [(("y", "Yes"), ("n", "No"))]
    assert choices_events(bridge.poll()) == [()]


def test_a_pause_shows_a_continue_button():
    bridge = GuiBridge()
    ui = bridged_ui(bridge, pauses=True)
    game = GameThread(lambda: ui.pause())
    events = answer_with_events(bridge, "")
    game.join()
    assert choices_events(events) == [(("", "Continue"),)]
    assert choices_events(bridge.poll()) == [()]


def test_closing_the_window_at_a_menu_is_a_polite_quit_and_hides_the_buttons():
    bridge = GuiBridge()
    ui = bridged_ui(bridge)
    game = GameThread(lambda: ui.confirm("Save it?"))
    next_prompt(bridge)
    bridge.close()
    game.join()
    assert isinstance(game.error, UserQuit) and not isinstance(game.error, UserChoseQuit)


def test_a_broken_choices_fn_never_breaks_the_question():
    def broken(options):
        raise RuntimeError("window trouble")

    answers = iter(["y", "a"])
    ui = UI(console=Console(file=None, width=80, quiet=True), input_fn=lambda p: next(answers), choices_fn=broken)
    assert ui.confirm("OK?") is True
    assert ui.choose("Pick", [("a", "A")]) == "a"


def test_without_choices_fn_nothing_changes_for_the_terminal():
    answers = iter(["", "2"])
    ui = UI(console=Console(file=None, width=80, quiet=True), input_fn=lambda p: next(answers))
    assert ui.confirm("OK?", default=True) is True
    assert ui.choose("Pick", [("a", "A"), ("b", "B")]) == "b"


def test_choices_calls_come_in_pairs_even_when_the_player_quits():
    calls = []

    def quitting(prompt):
        raise EOFError

    ui = UI(console=Console(file=None, width=80, quiet=True), input_fn=quitting, choices_fn=calls.append)
    with pytest.raises(UserQuit):
        ui.choose("Pick", [("a", "A")])
    assert calls == [[("a", "A")], []]


def test_hides_input_says_secrets_are_hidden(monkeypatch):
    monkeypatch.setattr("sys.stdin", None)
    console = Console(file=None, width=80, quiet=True)
    assert UI(console=console).can_hide_input() is False
    assert UI(console=console, hides_input=True).can_hide_input() is True


def test_secret_input_through_the_bridge():
    bridge = GuiBridge()
    ui = bridged_ui(bridge)
    game = GameThread(lambda: ui.secret("Paste your key"))
    prompt = next_prompt(bridge)
    assert prompt.secret and prompt.text == "Paste your key: "
    bridge.submit("  tsk-abc  ", prompt.id)
    game.join()
    assert game.result == "tsk-abc"


# ---------------------------------------------------------------------------
# The whole real game through the bridge, with a fake window thread
# ---------------------------------------------------------------------------


def play_through_bridge(argv: list[str], *, answer: Callable[[Prompt, tuple], Optional[str]],
                        timeout: float = 60.0) -> tuple[object, TerminalBuffer, list[str]]:
    """Run cli.main on a worker thread; this thread plays the window. Returns (exit code, screen, prompts)."""
    from gettowork import cli

    bridge = GuiBridge()
    console = Console(file=bridge.stream, force_terminal=True, force_interactive=True,
                      color_system="truecolor", width=100, height=32, legacy_windows=False, soft_wrap=False)
    ui = UI(console=console, input_fn=bridge.request_line,
            secret_fn=lambda p: bridge.request_line(p, secret=True),
            choices_fn=bridge.show_choices, hides_input=True, pauses=True)
    game = GameThread(lambda: cli.main(argv, ui=ui))
    screen, prompts, choices = TerminalBuffer(), [], ()
    deadline = time.monotonic() + timeout
    finished = False
    while not finished and time.monotonic() < deadline:
        for kind, payload in bridge.poll():
            if kind == OUTPUT:
                screen.feed(payload)
            elif kind == CHOICES:
                choices = payload
            elif kind == PROMPT:
                prompts.append(payload.text)
                reply = answer(payload, choices)
                if reply is None:
                    bridge.close()
                else:
                    bridge.submit(reply, payload.id)
            elif kind == FINISHED:
                finished = True
        if game.error is not None or not game.thread.is_alive():
            for kind, payload in bridge.poll():
                if kind == OUTPUT:
                    screen.feed(payload)
            break
        time.sleep(0.002)
    bridge.close()
    game.join(10)
    if game.error is not None:
        raise game.error
    return game.result, screen, prompts


def test_the_real_game_plays_start_to_finish_through_the_bridge():
    player = SelftestPlayer()
    code, screen, prompts = play_through_bridge(
        ["--mock", "--no-jev"],
        answer=lambda p, choices: player.answer(p.text, secret=p.secret, choices=choices),
    )
    assert code == 0
    assert "YOU GOT TO WORK" in screen.text()
    assert player.plans_given >= 5
    # The pauses a terminal player sees are there too (a person reads this window).
    assert any("Press Enter" in p for p in prompts)
    assert any("How do you plan to get to work?" in p for p in prompts)


def test_closing_mid_game_ends_the_game_politely():
    from gettowork import cli

    seen = []

    def answer(prompt, choices):
        seen.append(prompt.text)
        return None if "plan" in prompt.text.lower() else ""  # close the window at the first plan

    code, screen, _prompts = play_through_bridge(["--mock", "--no-jev"], answer=answer)
    assert code in (cli.EXIT_OK, cli.EXIT_INTERRUPTED)
    assert any("plan" in p.lower() for p in seen)
