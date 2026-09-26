"""The ``gettowork`` command: parse options, then banner -> setup -> Jev -> game -> review.

Run ``gettowork --help`` for every option. The ones worth knowing:

* ``gettowork``               - the full, friendly experience (recommended)
* ``gettowork --mock``        - play instantly with a pretend model (no downloads)
* ``gettowork --specs``       - just show what the game thinks of your computer
* ``gettowork --list-models`` - just show which models would fit, then exit

After each game (and its review) a player is asked "Play again?": the model
keeps running, so another morning starts straight away. The very first
launch also shows a short note that the story is written by an AI (see
``notices.py``). The game's own window (``gui/app.py``, what Steam
launches) runs this same ``main()`` in a worker thread.

Whatever happens, the local model's engine is shut down on exit, and Ctrl+C
ends the game politely (exit code 130, the Unix convention for "interrupted
by the user"). Even when no cleanup code gets to run at all (the console
window closed on Windows, the game killed outright), the engine is tied to
the game's process by the operating system - see ``backends/llamaserver.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

from rich import box
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from . import __version__, catalog, config, crashlog, launcher, onboarding, perf, runtime_install
from .config import Settings, command_name
from .setup_flow import (
    planned_engine_key,
    SetupServices,
    apply_engine_limits,
    apply_saved_calibration,
    disk_space_warning,
    find_models,
    run_setup,
    show_hardware,
    show_model_table,
    symbols_for,
)
from .ui import UI, UserChoseQuit, UserQuit, make_input_safe, make_stream_safe, safe_text

__all__ = ["main", "build_parser", "show_ai_notice_once", "ask_to_play_again", "after_pretend_game",
           "player_is_present",
           "EXIT_OK", "EXIT_ERROR", "EXIT_USAGE", "EXIT_INTERRUPTED"]

EXIT_OK = 0
EXIT_ERROR = 1  # something unexpected went wrong
EXIT_USAGE = 2  # a bad option (argparse uses 2 as well)
EXIT_INTERRUPTED = 130  # Ctrl+C: 128 + SIGINT(2), the shell convention

MAX_TARGET = 50
GOODBYE = "No problem - see you next time you're running late! (Anything already downloaded is kept.)"
THANKS = "Thanks for playing Get To Work! Your model is being tucked back into bed."
THANKS_PRETEND = "Thanks for playing Get To Work!"
REAL_MODEL_NEXT_TIME = ("Next time, pick a real AI model from the menu - it's free, runs privately on your computer, "
                        "and makes up a new story every time.")
# After a game with the pretend model picked from the menu: the way on to a real one.
AFTER_PRETEND_OPTIONS = (
    ("again", "Play again with the pretend model"),
    ("real", "Pick a real AI model"),
    ("quit", "Quit"),
)
_AFTER_PRETEND_ALIASES = {"y": "again", "yes": "again", "n": "quit", "no": "quit", "model": "real", "ai": "real"}

AI_NOTICE_SETTING = "ai_notice_seen"  # in Settings.extra: the first-launch AI note was shown
# Used only if notices.py can't be loaded: the same message, shorter.
FALLBACK_AI_NOTICE = (
    "**A quick note before you play:** the story in Get To Work is written live by an AI that runs "
    "on your own computer, so every morning turns out differently. A family-friendly filter checks "
    "what it writes, but AI can still surprise us now and then - if you ever see something that "
    "shouldn't be there, please tell us in the game's Steam Discussions."
)


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def _target(text: str) -> int:
    """argparse type for --target: a whole number from 1 to MAX_TARGET."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"'{text}' isn't a whole number") from None
    if not 1 <= value <= MAX_TARGET:
        raise argparse.ArgumentTypeError(f"pick a number from 1 to {MAX_TARGET}")
    return value


def build_parser() -> argparse.ArgumentParser:
    """All of the game's command-line options (every one of them is optional)."""
    parser = argparse.ArgumentParser(
        prog="gettowork",
        description=(  # line breaks by hand: RawDescriptionHelpFormatter doesn't re-wrap text
            "Get To Work: a farcical race to the office, and a friendly hands-on tour of\n"
            "local AI models (and, optionally, the Jev typed-judgment API).\n\n"
            "Just run it with no options: the game checks your computer, suggests models\n"
            "that fit, and sets everything up for you."
        ),
        epilog=(
            "examples:\n"
            "  gettowork                    the full experience (recommended)\n"
            "  gettowork --mock             play right now with a pretend model, no downloads\n"
            "  gettowork --list-models      see which models fit this computer\n"
            "  gettowork --model unsloth/Qwen3-4B-GGUF --quant Q4_K_M\n"
            "\nMIT licensed, provided with no warranty. Models are downloaded from Hugging Face\n"
            "under their own licenses."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    play = parser.add_argument_group("playing")
    play.add_argument("--mock", action="store_true", help="play with a built-in pretend model: offline, instant, no downloads")
    play.add_argument("--no-jev", action="store_true", help="skip the optional Jev question and play local-only")
    play.add_argument("--jev", action="store_true",
                      help="ask about the optional Jev referee again (after you chose the local model only)")
    play.add_argument("--target", type=_target, default=5, metavar="N", help="steps needed to reach work (default: 5)")
    play.add_argument("--think", action="store_true",
                      help="let a thinking model think out loud even on a slow computer (turns take longer)")
    play.add_argument("--export-dir", type=Path, default=None, metavar="DIR",
                      help="folder for the optional transcript export (default: the current folder)")

    model = parser.add_argument_group("choosing a model (normally automatic)")
    model.add_argument("--model", metavar="REPO_OR_KEY",
                       help="skip the menu: a Hugging Face GGUF repo (owner/name) or a built-in key like qwen3-4b")
    model.add_argument("--quant", metavar="TAG", help="use this quantization, e.g. Q4_K_M or Q8_0")
    model.add_argument("--gguf", metavar="PATH", help="use a GGUF file you already have (with the built-in engine)")
    model.add_argument("--ollama-model", metavar="TAG", help="use this Ollama model tag (implies --backend ollama)")
    model.add_argument("--backend", choices=("auto", "managed", "ollama", "llamacpp"), default="auto",
                       help="engine: auto (default: built-in llama.cpp, falling back to Ollama), managed, "
                            "ollama or llamacpp (needs llama-cpp-python)")
    model.add_argument("--refresh-models", action="store_true", help="search Hugging Face again instead of using the saved list")
    model.add_argument("--offline", action="store_true", help="don't go online for the model list (saved list or built-in picks)")
    model.add_argument("--models-dir", type=Path, default=None, metavar="DIR",
                       help="keep downloaded models in this folder from now on (e.g. on a bigger drive; remembered)")
    model.add_argument("--all-licenses", action="store_true",
                       help="also show models that aren't Apache-2.0/MIT licensed (each license is shown)")

    info = parser.add_argument_group("information")
    info.add_argument("--specs", action="store_true", help="show what the game detects about this computer, then exit")
    info.add_argument("--list-models", action="store_true", help="show the best model picks for this computer, then exit")
    info.add_argument("--reset", action="store_true", help="forget saved settings (your choices and any saved key)")
    info.add_argument("--debug", action="store_true", help="show full technical details if something goes wrong")
    info.add_argument("--version", action="version", version=f"gettowork {__version__}")
    return parser


def _parse(argv: Optional[list[str]]) -> tuple[Optional[argparse.Namespace], int]:
    """(args, 0), or (None, exit code) for --help / --version / a bad option."""
    try:
        args = build_parser().parse_args(argv)
    except SystemExit as exc:  # argparse already printed help, the version or the error
        code = exc.code if isinstance(exc.code, int) else (EXIT_OK if exc.code is None else EXIT_USAGE)
        return None, code
    if args.quant:
        args.quant = args.quant.strip().upper()
    return args, EXIT_OK


# ---------------------------------------------------------------------------
# Screens
# ---------------------------------------------------------------------------


def show_banner(ui: UI) -> None:
    ui.console.print(
        Panel(
            "[bold magenta]GET TO WORK[/bold magenta]\n"
            "A farcical race to the office - and a friendly, hands-on tour of local AI models.\n"
            f"[dim]v{__version__} · MIT licensed · no warranty · not affiliated with any model or API provider[/dim]",
            border_style="magenta",
            padding=(1, 2),
        )
    )


def ai_notice_text() -> str:
    """The first-launch note about AI-written content (Markdown)."""
    try:
        from .notices import AI_CONTENT_NOTICE

        return str(AI_CONTENT_NOTICE)
    except Exception:  # never let a missing or broken notices module stop the game
        return FALLBACK_AI_NOTICE


def show_ai_notice_once(ui: UI, settings: Settings) -> bool:
    """First launch only: say that the story is written by an AI, and how to report a problem.

    Steam asks games with live-generated AI content to tell players so, and
    it's only fair. Remembered in the settings, so it's shown once; a player
    who resets their settings sees it again. Only when a person is playing
    (a real terminal or the game's window): piped input and scripted tests
    skip it and never write a settings file for it. Returns True if shown.
    """
    if settings.extra.get(AI_NOTICE_SETTING) or not player_is_present(ui):
        return False
    ui.console.print(Panel(Markdown(safe_text(ai_notice_text())), title="Before you play",
                           border_style="cyan", padding=(1, 2)))
    settings.extra[AI_NOTICE_SETTING] = True
    try:
        settings.save()
    except OSError:
        pass  # a read-only settings folder: no harm done, the note just shows again next time
    ui.pause("Press Enter to start")
    return True


def player_is_present(ui: UI) -> bool:
    """Is a person reading and typing - a real terminal, or the game's own window?

    The same test the UI uses for its "Press Enter" pauses. Piped input and
    scripted tests are never asked extra questions such as "Play again?".
    """
    return bool(getattr(ui, "_interactive", False))


def ask_to_play_again(ui: UI, summary: Any) -> bool:
    """After a game and its review: "Play again? [Y/n]" (the model stays loaded).

    Only asked when a person is playing, and not after they typed ``quit``
    in the game - that already said what they wanted.
    """
    if getattr(summary, "quit_early", False) or not player_is_present(ui):
        return False
    ui.say()
    return ui.confirm("Play again?", default=True)


def after_pretend_game(ui: UI, summary: Any) -> str:
    """After a game with the pretend model (picked from the model menu): "again", "real" or "quit".

    "Play again?" alone could only replay the scripted pretend model; this
    also offers the way on to a real AI model, back through the model menu.
    """
    if getattr(summary, "quit_early", False) or not player_is_present(ui):
        return "quit"
    ui.say()
    return ui.choose("Play again?", list(AFTER_PRETEND_OPTIONS), default="again", aliases=_AFTER_PRETEND_ALIASES)


def show_specs(ui: UI, services: SetupServices) -> None:
    """``--specs``: hardware, memory speeds and the engine build we'd use."""
    with ui.status("Let me take a look at your computer..."):
        specs = apply_engine_limits(services.detect_specs())
    show_hardware(ui, specs, teach=False)
    ui.heading("What decides speed")
    cpu_bw, cpu_source = perf.bandwidth_for(specs, "cpu")
    rows = [("RAM bandwidth", f"~{cpu_bw:.0f} GB/s ({cpu_source})")]
    gpu = perf.primary_gpu(specs)
    if gpu is not None:
        gpu_bw, gpu_source = perf.bandwidth_for(specs, "unified" if gpu.vendor == "apple" else "gpu")
        rows.append(("Graphics memory bandwidth", f"~{gpu_bw:.0f} GB/s ({gpu_source})"))
    try:
        # Only builds the game would really try: a built game counts just the ones it ships.
        plan = runtime_install.available_plan(specs)
        arrow = f" {symbols_for(ui).arrow} "
        if plan:
            rows.append(("llama.cpp build I'd use", arrow.join(v.display for v in plan) + " (first one that works)"))
    except Exception:  # purely informative: never fail --specs over it
        pass
    try:
        # Where the engine comes from - in a built game, the release and builds it ships
        # (the build check in packaging/smoke_test.sh reads this line).
        rows.append(("llama.cpp engine", runtime_install.engine_summary()))
    except Exception:
        pass
    table = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    for label, value in rows:
        table.add_row(escape(label), escape(value))
    ui.console.print(table)
    ui.teach("why speed is all about memory bandwidth", perf.SPEED_EXPLAINER)
    name = command_name()
    ui.info(f"Run [bold]{name} --list-models[/bold] to see which models fit, or just [bold]{name}[/bold] to play.")


def list_models(ui: UI, args: argparse.Namespace, settings: Settings, services: SetupServices) -> None:
    """``--list-models``: the ranked shortlist for this computer."""
    with ui.status("Let me take a look at your computer..."):
        specs = apply_engine_limits(services.detect_specs(), args.backend)
    engine = args.backend if args.backend not in ("auto", "managed") else planned_engine_key(specs, settings)
    specs, _applied = apply_saved_calibration(specs, settings, engine=engine)
    show_hardware(ui, specs, teach=False)
    search = find_models(
        ui,
        specs,
        services,
        refresh=args.refresh_models,
        offline=args.offline,
        allow_all_licenses=args.all_licenses,
        engine_limits=args.backend in ("auto", "managed"),
    )
    ui.heading("Best picks for this computer")
    if not search.shortlist:
        advice = disk_space_warning(ui, specs, list(search.discovery.models))
        if advice is not None:
            ui.warn(escape(advice) + " You can still play with --mock!")
        else:
            ui.warn("None of the models I found run comfortably on this computer. You can still play with --mock!")
        return
    show_model_table(ui, search.shortlist, show_repo=True, interactive=False)
    first = search.recommended or search.shortlist[0]
    name = command_name()
    ui.info(f"To play with one directly: [bold]{name} --model {escape(first.model.hf_repo)}[/bold] "
            f"- or just run [bold]{name}[/bold] and pick from the menu.")


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


def _remote_ollama(backend: Any) -> Optional[str]:
    """"gpu-box.lan:11434" when the model runs in an Ollama on another computer (OLLAMA_HOST), else None."""
    host = getattr(backend, "host", None) if getattr(backend, "name", None) == "ollama" else None
    if not isinstance(host, str) or not host:
        return None
    from .backends import ollama as ollama_backend

    return None if ollama_backend.is_local_host(host) else ollama_backend.display_host(host)


def _stop_politely_on_termination() -> Callable[[], None]:
    """Treat "terminate" / "terminal closed" (SIGTERM / SIGHUP / SIGBREAK) like Ctrl+C.

    By default Python dies on those signals without running ``finally`` blocks.
    Turning them into KeyboardInterrupt lets main() stop the llama.cpp engine
    and say goodbye. (SIGBREAK is Ctrl+Break on Windows.) This is the polite
    path; if the game is killed without warning - or the Windows console
    window is closed, which ends the process before Python code can run - the
    operating system stops the engine instead (a kill-on-close Job Object on
    Windows, a parent-death signal on Linux, and a clean-up of leftovers on
    the next launch everywhere).
    """
    import signal
    import threading

    previous: list[tuple[int, Any]] = []

    def restore() -> None:
        for sig, handler in previous:
            try:
                signal.signal(sig, handler)
            except (OSError, ValueError, TypeError):
                pass

    if threading.current_thread() is not threading.main_thread():
        return restore  # signal handlers can only be installed from the main thread

    def interrupt(signum: int, frame: object) -> None:
        raise KeyboardInterrupt

    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            previous.append((sig, signal.signal(sig, interrupt)))
        except (OSError, ValueError):
            pass
    return restore


def _use_models_folder(ui: UI, settings: Settings, chosen: Optional[Path]) -> bool:
    """Apply ``--models-dir`` (and remember it), or the folder remembered last time.

    False if the chosen folder can't be used (the game then stops with a usage error).
    """
    if chosen is not None:
        folder = Path(chosen).expanduser()
        try:
            folder = folder.resolve()
            folder.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            ui.error(f"I can't use {escape(str(chosen))} for models ({escape(str(exc))}). Pick another folder.")
            return False
        settings.models_dir = str(folder)
        try:
            settings.save()
        except OSError:
            pass  # still used this time
        ui.info(f"Models are kept in {escape(str(folder))} from now on.")
    config.use_models_dir(settings.models_dir)
    return True


def _offer_fresh_start(ui: UI) -> None:
    """After an unexpected error: how to start again with default settings.

    A terminal gets the command; the game's window (no command line there)
    asks right away instead. Never raises.
    """
    if not ui.in_window:
        ui.info(f"If it keeps happening, [bold]{command_name()} --reset[/bold] starts fresh with default settings.")
        return
    try:
        if ui.confirm("If it keeps happening, starting fresh can help. Forget your saved settings now? "
                      "(Downloaded models are kept.)", default=False):
            Settings.reset()
            ui.info("Done - the next start is a fresh one.")
    except Exception:  # (includes the window closing)
        pass


def main(argv: Optional[list[str]] = None, *, ui: Optional[UI] = None,
         services: Optional[SetupServices] = None) -> int:
    """Run the game; returns the process exit code.

    `ui` and `services` are for tests (scripted input, fake hardware and
    network) and for the game's own window, which passes a UI wired to it
    and calls this from a worker thread; normally both are created here.
    """
    # The built game's console program double-clicked on Windows: its window
    # closes the moment we return, so the last words (a goodbye, an error
    # message) would vanish unread. Wait for Enter first, whatever happened.
    hold_window = ui is None and launcher.console_closes_on_exit()
    try:
        return _run(argv, ui=ui, services=services)
    finally:
        if hold_window:
            launcher.wait_before_closing()


def _run(argv: Optional[list[str]], *, ui: Optional[UI], services: Optional[SetupServices]) -> int:
    """main() without the Windows console courtesy: options -> setup -> Jev -> games -> goodbye."""
    if ui is None:
        # Old consoles / legacy code pages: print plain look-alikes, never crash.
        for stream in (sys.stdout, sys.stderr):
            make_stream_safe(stream)
        # ...and never trip over typed or piped text in an unexpected encoding.
        make_input_safe(sys.stdin)
    args, code = _parse(argv)
    if args is None:
        return code
    ui = ui or UI()
    services = services or SetupServices()
    backend = None
    restore_signals = _stop_politely_on_termination()
    try:
        if args.reset:
            Settings.reset()
            ui.info("Forgot your saved settings - fresh start!")
        if args.gguf and not Path(args.gguf).expanduser().is_file():
            ui.error(f"I can't find the model file {escape(str(args.gguf))}. Check the path and try again.")
            return EXIT_USAGE
        settings = Settings.load()
        if not _use_models_folder(ui, settings, args.models_dir):
            return EXIT_USAGE

        if args.specs:
            show_specs(ui, services)
            return EXIT_OK
        if args.list_models:
            list_models(ui, args, settings, services)
            return EXIT_OK

        show_banner(ui)
        show_ai_notice_once(ui, settings)

        # Imported here: they're only needed once a model is up and running.
        from .game import Game
        from .review import run_review

        taught: set[str] = set()  # "Learn" panels already shown this session: never twice
        while True:  # once - or again, when a pretend-model game leads on to a real model
            result = run_setup(ui, settings, args=args, services=services)
            if result is None:
                ui.say()
                ui.info(GOODBYE)
                return EXIT_OK
            backend = result.backend

            # A pretend-model game is a trial: a "no" to Jev there isn't remembered, so the first
            # game with a real model still offers it.
            trial = getattr(backend, "name", "") == "mock"
            jev = None if args.no_jev else onboarding.run_jev_onboarding(
                ui, settings, local_model_elsewhere=_remote_ollama(backend),
                ask_again=args.jev or getattr(result, "ask_jev", False), remember_no=not trial)
            # The player's API key in every form it could appear: masked in the
            # review and transcripts, and refused if it's pasted as a plan.
            secret_values = getattr(jev, "secret_values", None)
            secrets = set(secret_values()) if callable(secret_values) else set()
            secrets |= {k.strip() for k in (settings.jev_api_key, os.environ.get(onboarding.JEV_API_KEY_ENV))
                        if k and k.strip()}
            # One game per loop: the model (and Jev) stay ready, so "Play again?"
            # starts a brand-new morning straight away. Each game gets its own review.
            # (A pretend model picked from the menu also offers the way on to a real one;
            # with --mock the player asked for it, and hears how to play for real.)
            menu_trial = trial and not args.mock
            while True:
                game = Game(
                    backend,
                    ui,
                    jev=jev,
                    target=args.target,
                    # How fast the model really is (measured at warm-up) and how much room it has
                    # decide whether a thinking model may think out loud during the game.
                    tokens_per_s=result.tokens_per_s,
                    context_tokens=result.entry.context_tokens if result.entry is not None else None,
                    secrets=secrets,
                    thinking=catalog.thinking_mode(result.entry) if result.entry is not None else None,
                    force_think=bool(getattr(args, "think", False)),
                    taught=taught,
                )
                summary = game.run()
                run_review(ui, summary, export_dir=args.export_dir, secrets=secrets,
                           thinking_skipped_note=getattr(game, "thinking_note", None))
                if menu_trial:
                    step = after_pretend_game(ui, summary)
                else:
                    step = "again" if ask_to_play_again(ui, summary) else "quit"
                if step != "again":
                    break
                ui.say()
                ui.info("Here comes a brand-new morning!")
            if step == "real":
                backend.close()  # the pretend model: nothing to keep running
                backend = None
                ui.say()
                ui.info("Let's find you a real AI model!")
                continue
            break
        ui.say()
        ui.info(THANKS_PRETEND if trial else THANKS)
        if menu_trial:
            ui.info(REAL_MODEL_NEXT_TIME)
        return EXIT_OK
    except UserChoseQuit:  # "quit" typed at a yes/no question: a choice, like "quit" at a menu
        ui.say()
        ui.info(GOODBYE)
        return EXIT_OK
    except (KeyboardInterrupt, UserQuit):
        ui.say()
        ui.info(GOODBYE)
        return EXIT_INTERRUPTED
    except Exception as exc:  # a bug or something truly unexpected: be kind, and helpful
        # The details always go into a file too: in the window, closing it would
        # otherwise lose them, and a player can attach the file to a bug report.
        report = crashlog.write_crash_report("the game stopped with an unexpected error", exc)
        if args.debug:
            ui.console.print_exception()
        else:
            detail = f"{type(exc).__name__}: {exc}"
            if len(detail) > 200:
                detail = detail[:197] + "..."
            ui.error(f"Oops - something unexpected went wrong ({escape(detail)}).")
            if report is not None:
                ui.info(f"The technical details are saved in {escape(str(report))} - handy for a bug report.")
            elif not ui.in_window:
                ui.info("Rerun with [bold]--debug[/bold] to see the technical details (handy for a bug report).")
            if Settings().path.exists():
                _offer_fresh_start(ui)
        return EXIT_ERROR
    finally:
        if backend is not None:
            try:
                backend.close()  # always stop the local model's engine
            except Exception:
                pass
        restore_signals()
