"""The optional System One referee step: Jev or Laya, for someone new to API keys.

Principles:

* **Local-only is always one step away.** Every menu has a "play with the local
  model only" option, and Ctrl+C at any prompt here simply skips the referee.
* **The player picks the model.** System One Model Options are Jev (TypeSafe AI)
  and Laya (Laya Studio). They speak the same request; each has its own key.
* **Hand-holding, not walls of text.** Short explanations, numbered steps, and
  the browser is opened for you.
* **The key stays secret.** It's typed with hidden input (and if this window
  can't hide input, the player is told before pasting), only ever shown as
  ``****abcd``, and saved to disk only if the player says so (default: no).
  A saved key the player replaces, or that the service rejects, is forgotten.
* **Informed consent.** Before a referee is switched on, the player is told what
  it sends over the internet (their plan, the challenge, a story summary) and to whom.

Jev and Laya are paid third-party services with their own pricing and terms.
Get To Work isn't affiliated with TypeSafe AI or Laya Studio.
"""

from __future__ import annotations

import inspect
import os
import platform
import urllib.parse
from typing import Callable, Mapping, Optional

from rich.markup import escape

from .config import Settings, command_name
from .jev import (
    JEV_OPTION,
    LAYA_OPTION,
    SYSTEM_ONE_OPTIONS,
    SystemOneOption,
    JevClient,
    JevError,
    other_system_one,
    redact_key,
    system_one_lesson,
    system_one_option,
    validate_api_key_format,
)
from .ui import UI, UserQuit, WindowClosed, looks_like_secret

ClientFactory = Callable[..., JevClient]

LOCAL_ONLY_LABEL = "Never mind, play with the local model only (free, nothing leaves this computer)"
SYSTEM_ONE_PROMPT = "System One Model Options"

# The welcome-back button (setup_flow.WELCOME_BACK_JEV_OPTIONS). The window shows
# the part before " (", so this sentence has to quote that shorter label.
_WELCOME_BACK_BUTTON = "Play with Jev or Laya this time"

# Typed at the hidden key prompt, these mean "take me back", not "here's my key".
_NAVIGATION_WORDS = frozenset({"back", "b", "skip", "cancel", "no", "n", "quit", "q", "exit", "help", "h", "menu", "?"})

# What the key-entry steps can ask the menu to do next.
_PASTE, _HELP, _MENU, _LOCAL = "paste", "help", "menu", "local"


def run_jev_onboarding(
    ui: UI,
    settings: Settings,
    *,
    env: Optional[Mapping[str, str]] = None,
    client_factory: Optional[ClientFactory] = None,
    local_model_elsewhere: Optional[str] = None,
    ask_again: bool = False,
    remember_no: bool = True,
) -> Optional[JevClient]:
    """Ask whether to enable Jev and, if so, get a working API key.

    Returns a ready :class:`JevClient`, or ``None`` to play with the local
    model only. Never raises for anything the player does: Ctrl+C anywhere in
    here just means "skip Jev". (Closing the game's window is the exception:
    :class:`~gettowork.ui.WindowClosed` goes through, and the game ends.)

    Args:
        ui: Where all questions and messages go.
        settings: Loaded settings. ``jev_enabled`` is remembered, and the key
            is saved only if the player explicitly opts in.
        env: Environment to read ``TYPESAFE_API_KEY`` from (default ``os.environ``).
        client_factory: ``callable(api_key) -> JevClient``; tests inject one
            with a fake transport.
        local_model_elsewhere: where the "local" model really runs when it
            isn't this computer (an Ollama at another address, from
            OLLAMA_HOST), so the menus don't promise "nothing leaves this computer".
        ask_again: ask about Jev even though the player said no last time
            (``--jev``). Otherwise a player who chose the local model only is
            not asked again - one line says how to turn Jev on - so a
            returning player goes straight into the game.
        remember_no: save a "no" (``jev_enabled = False``). False for a
            pretend-model trial game: declining Jev there must not stop the
            first real game from offering it.
    """
    env = os.environ if env is None else env
    factory = client_factory or _default_factory(env)
    flow = _JevOnboarding(ui, settings, env, factory, local_model_elsewhere, ask_again=ask_again,
                          remember_no=remember_no)
    try:
        return flow.run()
    except WindowClosed:
        raise  # the window closed: the game ends here (no opening story behind a hidden window)
    except (UserQuit, KeyboardInterrupt):
        # Ctrl+C at a prompt (UserQuit) or while a key is being checked (KeyboardInterrupt).
        ui.say()
        ui.info("No problem - skipping Jev. Your local model will referee the game.")
        return None
    except Exception as exc:  # Jev is optional: no surprise here may end the game
        ui.say()
        ui.warn(escape(f"Something went wrong while setting up Jev ({type(exc).__name__}: {exc})."))
        ui.info("No harm done - your local model will referee the game.")
        return None


def _default_factory(env: Mapping[str, str]) -> ClientFactory:
    def make(key: str, option: SystemOneOption = JEV_OPTION) -> JevClient:
        return JevClient(
            key,
            option=option,
            base_url=(env.get(option.base_url_env) or "").strip() or None,
            model=(env.get(option.model_env) or "").strip() or None,
        )

    return make


def _call_factory(factory: ClientFactory, key: str, option: SystemOneOption) -> JevClient:
    """``factory(key, option)`` when it accepts the option, otherwise ``factory(key)``."""
    try:
        parameters = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return factory(key)
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in parameters.values()):
        return factory(key, option)
    positional = [
        p for p in parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 2:
        return factory(key, option)
    return factory(key)


def clean_pasted_key(text: str) -> str:
    """Tidy common copy-paste accidents around a key.

    Strips surrounding whitespace and quotes, and a leading ``Bearer `` or
    ``TYPESAFE_API_KEY=`` / ``LAYA_API_KEY=`` in case a whole line was copied
    from docs or a shell.
    """
    key = (text or "").strip()
    prefixes = ["Bearer ", "bearer "]
    for option in SYSTEM_ONE_OPTIONS:
        prefixes.extend((f"export {option.api_key_env}=", f"{option.api_key_env}="))
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
            break
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    return key


def _host_for(env: Mapping[str, str], option: SystemOneOption) -> str:
    """The host a referee's requests go to, even when the address is mistyped."""
    raw = (env.get(option.base_url_env) or "").strip() or option.default_base_url
    try:
        return urllib.parse.urlsplit(raw if "://" in raw else "https://" + raw).hostname or raw
    except ValueError:
        return f"{raw} - an address that doesn't look valid; check {option.base_url_env}"


def privacy_notice(env: Mapping[str, str], option: SystemOneOption = JEV_OPTION) -> str:
    """What one System One referee sends, and to whom.

    Never raises: a mistyped base URL (say, ``http://[::1``, which the URL
    parser rejects) is shown as it is, flagged as odd.
    """
    host = _host_for(env, option)
    return (
        f"Privacy: with {option.name} on, each round sends the plan you type, the current challenge, a short summary "
        f"of the story and your progress over the internet to {option.vendor} ({host}), under their terms and "
        "privacy policy. With the local model only, everything stays on this computer."
    )


def system_one_privacy(env: Mapping[str, str]) -> str:
    """Both referees, before the player has picked one."""
    return (
        "Privacy: a System One referee sends the plan you type, the current challenge, a short summary "
        "of the story and your progress over the internet. "
        f"Jev goes to TypeSafe AI ({_host_for(env, JEV_OPTION)}). "
        f"Laya goes to Laya Studio ({_host_for(env, LAYA_OPTION)}). "
        "Each service has its own terms and privacy policy. "
        "With the local model only, everything stays on this computer."
    )


# The "back out" words the game teaches ("choose 'no' or 'back'"), accepted at
# the yes/no/learn menus that don't have a 'back' option of their own.
_BACK_OUT_ALIASES = {w: "no" for w in ("back", "b", "skip", "cancel", "quit", "exit", "local", "nope", "nah")}
# A confused player's words, at menus with a "learn" option: the same words that
# open the tips in the game and the lessons at the model menu.
_HELP_ALIASES = {w: "learn" for w in ("help", "h", "?", "info", "what", "explain", "more", "tell me more")}
_ENABLE_ALIASES = {**_BACK_OUT_ALIASES, **_HELP_ALIASES}
# "yes" used to mean "turn Jev on". It still picks Jev, so old answers keep working.
_YES_TO_JEV = {w: "jev" for w in ("y", "yes", "yeah", "yep", "sure")}
_SYSTEM_ONE_ALIASES = {**_ENABLE_ALIASES, **_YES_TO_JEV}
# "Use it?" is asked as a yes/no question: a yes means "use" (a no already means "no").
_USE_IT_ALIASES = {**_ENABLE_ALIASES, **{w: "use" for w in ("y", "yes", "yeah", "yep", "sure", "ok", "okay")}}
# ...and at menus whose way out is called 'back'.
_TO_BACK_ALIASES = {w: "back" for w in ("no", "n", "skip", "cancel", "quit", "exit", "local", "nope", "nah")}
# "Paste it here anyway?" is a yes/no question too.
_PASTE_ANYWAY_ALIASES = {**_TO_BACK_ALIASES, **{w: "paste" for w in ("y", "yes", "yeah", "yep", "sure", "ok")}}


def _pasted_key_shape(answer: str) -> bool:
    """A menu answer that is really an API key pasted into the wrong box."""
    key = clean_pasted_key(answer)
    return looks_like_secret(key) and validate_api_key_format(key) is None


class _JevOnboarding:
    """The onboarding conversation, one small method per step."""

    def __init__(self, ui: UI, settings: Settings, env: Mapping[str, str], factory: ClientFactory,
                 local_model_elsewhere: Optional[str] = None, *, ask_again: bool = False,
                 remember_no: bool = True) -> None:
        self.ui = ui
        self.ask_again = ask_again
        self.remember_no = remember_no
        self._pasted: Optional[str] = None  # a key pasted straight into a menu, waiting to be checked
        self.settings = settings
        self.env = env
        self.factory = factory
        self.option = self._remembered_option() or JEV_OPTION
        # "nothing leaves this computer" is only true when the local model runs here:
        # an Ollama on another machine (OLLAMA_HOST) gets the plans too.
        if local_model_elsewhere:
            self._local_note = f"free; your plans go only to your own Ollama at {escape(local_model_elsewhere)}"
            self._local_only_label = f"Never mind, play with the local model only ({self._local_note})"
        else:
            self._local_note = "free, nothing leaves this computer"
            self._local_only_label = LOCAL_ONLY_LABEL

    # -- the overall flow ---------------------------------------------------------

    def run(self) -> Optional[JevClient]:
        if self.settings.jev_enabled is False and not self.ask_again and not self._key_in_env():
            # A returning player who chose the local referee: straight into the game, with
            # one line on how to change their mind (asked again only with --jev).
            self.ui.info("Jev (the optional, paid AI referee) is off - you chose your local model last time. "
                         "Laya stays off too. " + self._how_to_turn_jev_on())
            return None
        self.ui.heading("Optional extra: a System One referee")
        known = self._known_credentials()
        if len(known) == 1:
            self.option, key, source = known[0]
            return self._after_existing_offer(key, source)
        if len(known) > 1:
            picked = self._ask_model(short=self.settings.jev_enabled is False and not self._remembered_option())
            if picked is None:
                return self._go_local()
            self.option = picked
            match = next((item for item in known if item[0].id == picked.id), None)
            if match is not None:
                return self._after_existing_offer(match[1], match[2])
            return self._key_menu()

        picked = self._ask_model(short=self.settings.jev_enabled is False)
        if picked is None:
            return self._go_local()
        self.option = picked
        return self._key_menu()

    def _after_existing_offer(self, key: str, source: str) -> Optional[JevClient]:
        choice = self._offer_existing_key(source, key)
        if choice == "use":
            result = self._try_key(key, source)
            return self._continue_from(result)
        if choice == "no":
            return self._go_local()
        return self._key_menu()

    def _remembered_option(self) -> Optional[SystemOneOption]:
        if self.settings.jev_enabled is not True:
            return None
        return system_one_option(self.settings.system_one)

    def _continue_from(self, result: "JevClient | str") -> Optional[JevClient]:
        """After a key check: a client means done; anything else is a next step."""
        if isinstance(result, JevClient):
            return result
        if result == _LOCAL:
            return self._go_local()
        return self._key_menu(start=result)

    # -- step: an API key we already know about -------------------------------------------

    def _key_in_env(self) -> bool:
        """A key set on purpose for this run (Jev or Laya), so it's always offered."""
        return any((self.env.get(option.api_key_env) or "").strip() for option in SYSTEM_ONE_OPTIONS)

    def _how_to_turn_jev_on(self) -> str:
        if getattr(self.ui, "in_window", False):
            return (f"To turn one on, choose '{_WELCOME_BACK_BUTTON}' when the game welcomes you back - or start "
                    "it once with --jev (on Steam: right-click Get To Work > Properties > General > Launch Options).")
        return (f"To turn one on, choose jev ('{_WELCOME_BACK_BUTTON}') when the game welcomes you back, or start "
                f"the game with [bold]{command_name()} --jev[/bold].")

    def _saved_key(self, option: SystemOneOption) -> str:
        raw = self.settings.laya_api_key if option.id == "laya" else self.settings.jev_api_key
        return (raw or "").strip()

    def _store_key(self, key: Optional[str], option: Optional[SystemOneOption] = None) -> None:
        option = option or self.option
        if option.id == "laya":
            self.settings.laya_api_key = key
        else:
            self.settings.jev_api_key = key

    def _credential(self, option: SystemOneOption) -> Optional[tuple[str, str]]:
        """(key, source) for one model: an env key beats a saved one. None if neither is usable."""
        env_key = (self.env.get(option.api_key_env) or "").strip()
        if env_key:
            if validate_api_key_format(env_key) is None:
                return env_key, "env"
            self.ui.warn(
                f"Your {option.api_key_env} environment variable is set, but the value doesn't look like an "
                "API key, so I'll ignore it."
            )
        saved = self._saved_key(option)
        if saved and validate_api_key_format(saved) is None:
            return saved, "saved"
        return None

    def _known_credentials(self) -> list[tuple[SystemOneOption, str, str]]:
        found: list[tuple[SystemOneOption, str, str]] = []
        for option in SYSTEM_ONE_OPTIONS:
            credential = self._credential(option)
            if credential is not None:
                found.append((option, credential[0], credential[1]))
        return found

    def _offer_existing_key(self, source: str, key: str) -> str:
        option = self.option
        other = other_system_one(option)
        where = (
            f"in your {option.api_key_env} environment variable"
            if source == "env"
            else "that you saved last time"
        )
        self.ui.say(
            f"{option.name} is {option.vendor}'s System One model. It can referee your plans with numbers instead "
            "of words (an optional, paid, third-party service)."
        )
        self.ui.say(f"[dim]{escape(privacy_notice(self.env, option))}[/dim]")
        self.ui.info(f"I found a {option.name} API key {where} (ending {escape(redact_key(key))}).")
        options = [
            ("use", f"Check the key and turn on {option.name}"),
            ("new", "Use a different key"),
            ("switch", f"Use {other.name} instead"),
            ("no", f"No thanks, play with the local model only ({self._local_note})"),
            ("learn", "What's a System One model? Tell me more first"),
        ]
        if source == "saved":
            options.append(("forget", "Forget the saved key and play with the local model only"))
        while True:
            choice = self.ui.choose(
                "Use it?", options, default="no" if self.settings.jev_enabled is False else "use",
                aliases=_USE_IT_ALIASES,
            )
            if choice == "forget":
                self._forget_saved_key("Done - the saved key is gone from this computer.")
                return "no"
            if choice == "switch":
                self.option = other
                self.ui.say(f"[dim]{escape(privacy_notice(self.env, other))}[/dim]")
                return "new"
            if choice != "learn":
                return choice
            self.ui.teach("System One models", system_one_lesson())

    # -- step: which System One model, if any? -----------------------------------------------

    def _model_options(self) -> list[tuple[str, str]]:
        return [
            (option.id, option.summary)
            for option in SYSTEM_ONE_OPTIONS
        ] + [
            ("no", f"No thanks, play with the local model only ({self._local_note})"),
            ("learn", "Tell me more about System One models first"),
        ]

    def _ask_model(self, *, short: bool) -> Optional[SystemOneOption]:
        """The System One Model Options menu. None means play local-only."""
        ui = self.ui
        if short:
            # A returning player who asked to be asked again (--jev): one short
            # question, not the whole introduction again (it's behind "learn").
            ui.say("System One referees are off - last time you chose to play with your local model only.")
            options = [
                ("no", f"No thanks, play with the local model only ({self._local_note})"),
                ("jev", "Jev - referee my plans (needs a TypeSafe API key)"),
                ("laya", "Laya - referee my plans (needs a Laya Studio API key)"),
                ("learn", "What's a System One model? Tell me more first"),
            ]
            while True:
                choice = ui.choose(
                    SYSTEM_ONE_PROMPT, options, default="no", aliases=_SYSTEM_ONE_ALIASES,
                )
                if choice == "learn":
                    ui.teach("System One models", system_one_lesson())
                    continue
                if choice == "no":
                    return None
                picked = system_one_option(choice)
                ui.say(f"[dim]{escape(privacy_notice(self.env, picked))}[/dim]")
                return picked
        ui.say(
            "A [bold]System One[/bold] model doesn't write the story. It gives [bold]typed judgments[/bold]: "
            "instead of chatting, it answers questions with numbers."
        )
        ui.say(
            "Turn one on and it referees each of your plans three ways - a [bold]Noul[/bold] (yes/no: did "
            "you make progress?), a [bold]Choice[/bold] (what kind of outcome?) and a [bold]Score[/bold] "
            "(how creative, 0-4) - so you can see how each one works."
        )
        ui.say(
            "[bold]Jev[/bold] (TypeSafe AI) reads a long story. [bold]Laya[/bold] (Laya Studio) is an "
            "open-weight model on the same kind of API, and it reads a shorter one. Each is a paid, "
            "third-party service with its own pricing and terms. This game isn't affiliated with TypeSafe AI "
            "or Laya Studio."
        )
        ui.say(escape(system_one_privacy(self.env)))
        back_out = ("choose 'no' or 'back'" if getattr(ui, "in_window", False)  # (Ctrl+C copies text there)
                    else "choose 'no' or 'back', or press Ctrl+C")
        ui.say(
            "The game works fully without either - your local model can referee for free. "
            f"[dim](You can back out at any step: {back_out}.)[/dim]"
        )
        remembered = self._remembered_option()
        while True:
            choice = ui.choose(
                SYSTEM_ONE_PROMPT,
                self._model_options(),
                default=remembered.id if remembered is not None else "no",
                aliases=_SYSTEM_ONE_ALIASES,
            )
            if choice == "learn":
                ui.teach("System One models", system_one_lesson())
                continue
            if choice == "no":
                return None
            return system_one_option(choice)

    # -- step: get a key -----------------------------------------------------------------------

    def _key_menu(self, start: Optional[str] = None) -> Optional[JevClient]:
        """Loop until we have a working client or the player backs out.

        ``start`` lets a previous step jump straight to "paste" or "help".
        """
        step = start if start in (_PASTE, _HELP) else None
        while True:
            if step is None:
                step = self._menu_or_key(self.ui.choose(
                    f"How would you like to add your {self.option.name} API key?",
                    [
                        ("paste", "I have a key, let me paste it"),
                        ("help", "I don't have one yet, walk me through getting one"),
                        ("back", self._local_only_label),
                    ],
                    default="paste",
                    aliases=_TO_BACK_ALIASES,
                    accept=_pasted_key_shape,
                ))
            if step == _HELP:
                step = self._show_help()
                continue
            if step == _PASTE:
                pasted, self._pasted = self._pasted, None
                key = self._key_from(pasted) if pasted is not None else self._ask_for_key()
                if key is None:
                    step = None
                    continue
                result = self._try_key(key, "pasted")
                if isinstance(result, JevClient):
                    return result
                step = None if result == _MENU else result
                continue
            # "back" or _LOCAL
            return self._go_local()

    def _ask_for_key(self) -> Optional[str]:
        ui = self.ui
        if ui.can_hide_input():
            if getattr(ui, "in_window", False):
                paste = "Cmd+V" if platform.system() == "Darwin" else "Ctrl+V"
                ui.say(
                    f"[dim]Paste it into the box below with {paste} (or right-click it and choose Paste): "
                    "it shows as dots, so your key stays hidden.[/dim]"
                )
            else:
                ui.say(
                    "[dim]Your key stays hidden: nothing shows on screen while you paste. "
                    "(Tip: some terminals paste with right-click or Ctrl+Shift+V.)[/dim]"
                )
            ui.say("[dim](Press Enter with nothing typed to go back.)[/dim]")
            raw = ui.secret(f"Paste your {self.option.name} API key and press Enter")
        else:
            # getpass would quietly *show* the key here (an IDE console, piped input...): say so first.
            ui.warn(
                "This window can't hide what you type, so your key would be visible on screen (and in "
                "screen shares or recordings)."
            )
            ui.say(
                f"[dim]Safer: set the {self.option.api_key_env} environment variable and restart the game - "
                "it's picked up automatically.[/dim]"
            )
            choice = ui.choose(
                "Paste it here anyway?",
                [("back", "No, go back"), ("paste", "Yes, paste it (it will be visible)")],
                default="back",
                aliases=_PASTE_ANYWAY_ALIASES,
            )
            if choice != "paste":
                return None
            raw = ui.ask(f"Paste your {self.option.name} API key and press Enter (leave it empty to go back)")
        return self._key_from(raw)

    def _menu_or_key(self, answer: str) -> str:
        """A menu's answer: its option - or, for a key pasted straight into the menu, "paste" with that key.

        Pasting there is the natural thing to do after copying a key in the
        browser; the window already showed it as "(hidden)".
        """
        if answer in (_PASTE, _HELP, "back", "open", "docs"):
            return answer
        self._pasted = answer
        if not getattr(self.ui, "in_window", False):
            self.ui.say("[dim](That looks like your key, so I'll use it. Next time, choose 'paste' first: "
                        "then it stays hidden as you paste it.)[/dim]")
        return _PASTE

    def _key_from(self, raw: Optional[str]) -> Optional[str]:
        """A pasted key, cleaned up and checked for the right shape (None = go back a step)."""
        ui = self.ui
        key = clean_pasted_key(raw or "")
        if not key:
            ui.info("Nothing was entered - that's fine, let's go back a step.")
            return None
        if key.lower() in _NAVIGATION_WORDS:
            # Someone typing "back" or "help" here means the menu, not a key to send to Jev.
            ui.info("Going back a step.")
            return None
        problem = validate_api_key_format(key)
        if problem:
            self.ui.warn(problem)
            return None
        self.ui.info(f"Got a key ending {escape(redact_key(key))}.")
        return key

    def _show_help(self) -> str:
        """Step-by-step directions for getting a key; returns the next step."""
        ui = self.ui
        option = self.option
        host = urllib.parse.urlsplit(option.home_url).hostname or option.home_url
        ui.heading(f"Getting a {option.name} API key - step by step")
        ui.say(f"  [bold]1.[/bold] Open the {option.vendor} website: {option.home_url}  (choose 'open' below and I'll do it for you)")
        ui.say("  [bold]2.[/bold] Sign up for an account, or log in if you already have one.")
        ui.say("  [bold]3.[/bold] In your dashboard, open the API keys section and create a new key.")
        ui.say("  [bold]4.[/bold] Copy the key. Keep it private, like a password - many sites only show it once.")
        ui.say("  [bold]5.[/bold] Come back to this window and choose 'paste'.")
        ui.say(f"Want to read more first? {option.name}'s documentation is at {option.docs_url}")
        ui.say(
            f"[dim]Remember: {option.name} is a paid service with its own pricing and terms - check them on the site "
            "before signing up. You never have to: 'back' plays the game free with your local model.[/dim]"
        )
        ui.say(f"[dim]{escape(privacy_notice(self.env, option))}[/dim]")
        opened = False
        while True:
            choice = self._menu_or_key(ui.choose(
                "What next?",
                [
                    ("open", f"Open {host} in my browser"),
                    ("docs", f"Open the {option.name} documentation in my browser"),
                    ("paste", "I've got my key, paste it now"),
                    ("back", self._local_only_label),
                ],
                default="paste" if opened else "open",
                aliases=_TO_BACK_ALIASES,
                accept=_pasted_key_shape,
            ))
            if choice == "open":
                ui.open_url(option.home_url)
                ui.info("Take your time. When you've copied your key, come back and choose 'paste'.")
                opened = True
            elif choice == "docs":
                ui.open_url(option.docs_url)
                opened = True
            elif choice == "paste":
                return _PASTE
            else:
                return _LOCAL

    # -- step: check the key with Jev -------------------------------------------------------------------

    def _try_key(self, key: str, source: str) -> "JevClient | str":
        """Validate the key with ``GET /v1/models``.

        Returns a ready client, or the next step for the menu: ``"paste"``,
        ``"help"``, ``"menu"`` or ``"local"``.
        """
        ui = self.ui
        try:
            client = _call_factory(self.factory, key, self.option)
        except JevError as exc:
            ui.warn(escape(exc.message))
            return _MENU
        except Exception as exc:  # a surprise here must never end the whole game
            ui.warn(escape(_unexpected(exc, key)))
            return _MENU
        while True:
            try:
                try:
                    with ui.status(f"Checking your key with {self.option.name}..."):
                        models = client.check_key() if hasattr(client, "check_key") else client.list_models()
                except (JevError, KeyboardInterrupt):
                    raise
                except Exception as exc:  # treat anything unexpected like network trouble: retry or back out
                    raise JevError(_unexpected(exc, key), kind="error") from None
            except JevError as exc:
                if exc.is_auth_error:
                    ui.error(escape(exc.message))
                    if source == "saved":
                        self._forget_saved_key("Jev no longer accepts the key saved last time, so I've removed it.")
                    ui.say(
                        "Double-check that you copied the whole key (no missing characters) and that it "
                        "hasn't been deleted in your dashboard. Creating a fresh key often fixes this."
                    )
                    choice = ui.choose(
                        "What would you like to do?",
                        [
                            ("paste", "Paste a different key"),
                            ("help", "Show me how to get a key"),
                            ("back", self._local_only_label),
                        ],
                        default="paste",
                        aliases=_TO_BACK_ALIASES,
                    )
                    return {"paste": _PASTE, "help": _HELP}.get(choice, _LOCAL)
                if exc.kind == "billing":
                    ui.error(escape(exc.message))
                    ui.say("You can sort this out in your TypeSafe dashboard, then check again here.")
                    choice = ui.choose(
                        "What would you like to do?",
                        [
                            ("retry", "Check this key again"),
                            ("paste", "Paste a different key"),
                            ("back", self._local_only_label),
                        ],
                        default="back",
                        aliases=_TO_BACK_ALIASES,
                    )
                    if choice == "retry":
                        continue
                    return _PASTE if choice == "paste" else _LOCAL
                # Network trouble, a busy service, or something unexpected.
                ui.warn(escape(exc.message))
                choice = ui.choose(
                    "What would you like to do?",
                    [
                        ("retry", "Try again"),
                        (
                            "continue",
                            "Use this key without checking (if Jev can't be reached during the game, "
                            "your local model referees instead)",
                        ),
                        ("back", self._local_only_label),
                    ],
                    default="retry",
                    aliases=_TO_BACK_ALIASES,
                )
                if choice == "retry":
                    continue
                if choice == "continue":
                    ui.info("OK - we'll use the key without checking it first.")
                    self._enable(client, key, source, verified=False)
                    return client
                return _LOCAL
            else:
                names = [str(m.get("name")) for m in models if isinstance(m, dict) and m.get("name")]
                ui.success(f"Your key works - {self.option.name} is ready!")
                if names:
                    ui.info(escape(f"Models available to your key: {', '.join(names[:5])}"))
                self._enable(client, key, source, verified=True)
                return client

    # -- finishing up -----------------------------------------------------------------------------------

    def _enable(self, client: JevClient, key: str, source: str, *, verified: bool) -> None:
        self.settings.jev_enabled = True
        self.settings.system_one = self.option.id
        if source == "env":
            self.ui.info(f"Using the key from your {self.option.api_key_env} environment variable (nothing saved to disk).")
        elif source == "pasted" and verified:
            self._offer_to_save(key)
        elif source == "pasted" and self._saved_key(self.option) and self._saved_key(self.option) != key:
            # A new, unchecked key replaces the saved one for this game: don't keep the old one on disk.
            self._forget_saved_key("The key saved last time has been removed - you're using a new one now.",
                                   save=False)
        self._save_settings()
        self.ui.say(
            f"{self.option.name} is on! Each round it will referee your plan with a [bold]Noul[/bold], a "
            "[bold]Choice[/bold] and a [bold]Score[/bold]."
        )

    def _offer_to_save(self, key: str) -> None:
        ui = self.ui
        ui.say("I can remember this key so you don't have to paste it next time.")
        if platform.system() == "Windows":
            protection = ("and I'll set that file's Windows permissions so only your user account can read it "
                          "(I'll tell you if Windows won't let me)")
        else:
            protection = "with file permissions set so only your user account can read it"
        ui.say(
            f"[dim]If you say yes, it is stored as plain text in {escape(str(self.settings.path))}, "
            f"{protection}. Prefer not to? Set the {self.option.api_key_env} environment variable instead and the "
            "game will find it automatically.[/dim]"
        )
        choice = ui.choose(
            "Save the key on this computer?",
            [
                ("no", "Don't save it, I'll paste it again next time"),
                ("yes", "Save it in my settings file"),
            ],
            default="no",
        )
        if choice == "yes":
            self._store_key(key)
            where = escape(str(self.settings.path))
            if getattr(ui, "in_window", False):
                ui.info(f"Saved. (It's in {where} - delete it from there any time.)")
            else:
                ui.info(f"Saved. (Delete it any time by running [bold]{command_name()} --reset[/bold], "
                        f"or by editing {where}.)")
        elif self._saved_key(self.option):
            self._store_key(None)  # the old key was replaced: what's on disk must match what we say
            ui.info("Not saved - the key only lives in memory while the game runs, and the key saved last "
                    "time has been removed from this computer.")
        else:
            ui.info("Not saved - the key only lives in memory while the game runs.")

    def _forget_saved_key(self, message: str, *, save: bool = True) -> None:
        if not self._saved_key(self.option):
            return
        self._store_key(None)
        if save:
            self._save_settings()
        self.ui.info(message)

    def _go_local(self) -> None:
        if self.remember_no:
            self.settings.jev_enabled = False
            self._save_settings()
            self.ui.info("Playing with your local model only - it will referee your plans. Have fun!")
        else:  # a pretend-model trial: nothing remembered, so a real game asks again
            self.ui.info("No Jev this time - the pretend model will referee. I'll ask again when you play "
                         "with a real model.")
        return None

    def _save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as exc:
            self.ui.warn(
                escape(f"Couldn't save your settings ({exc}). No harm done - the game will just ask again next time.")
            )
            return
        if (self.settings.jev_api_key or self.settings.laya_api_key) and getattr(self.settings, "key_file_protected", None) is False:
            # Windows wouldn't give the file an owner-only access list: say so plainly.
            where = escape(str(self.settings.path))
            if not getattr(self.ui, "in_window", False):
                self.ui.warn(
                    f"Windows wouldn't let me restrict who can read {where}, so other accounts on this PC may be "
                    "able to read your saved key. To be safe, run the game once with --reset and set the "
                    f"{self.option.api_key_env} environment variable instead."
                )
                return
            # The window has no command line for --reset: offer the safe choice right here.
            self.ui.warn(f"Windows wouldn't let me restrict who can read {where}, so other accounts on this PC "
                         "may be able to read your saved key.")
            if self.ui.confirm(f"Forget the saved key? ({self.option.name} still works for this session - you'd paste the key "
                               "again next time.)", default=True):
                self._store_key(None)
                try:
                    self.settings.save()
                    self.ui.info("Done - your key isn't saved on this PC any more.")
                except OSError as exc:
                    self.ui.warn(escape(f"Couldn't update your settings ({exc}). You can delete {self.settings.path} "
                                        "yourself to remove the key."))


def _unexpected(exc: BaseException, key: str) -> str:
    """A friendly, key-free description of a surprise error."""
    detail = f"{type(exc).__name__}: {exc}".replace(key, redact_key(key)) if key else f"{type(exc).__name__}: {exc}"
    return f"Something unexpected went wrong while checking the key ({detail[:200]})."
