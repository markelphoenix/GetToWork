"""The optional "turn on Jev?" step, designed for someone who has never seen an API key.

Principles:

* **Local-only is always one step away.** Every menu has a "play with the local
  model only" option, and Ctrl+C at any prompt here simply skips Jev.
* **Hand-holding, not walls of text.** Short explanations, numbered steps, and
  the browser is opened for you.
* **The key stays secret.** It's typed with hidden input (and if this window
  can't hide input, the player is told before pasting), only ever shown as
  ``****abcd``, and saved to disk only if the player says so (default: no).
  A saved key the player replaces, or that Jev rejects, is forgotten.
* **Informed consent.** Before Jev is switched on, the player is told what it
  sends over the internet (their plan, the challenge, a story summary) and to whom.

Jev is a paid third-party API from TypeSafe AI with its own pricing and terms;
Get To Work isn't affiliated with TypeSafe AI.
"""

from __future__ import annotations

import os
import platform
import urllib.parse
from typing import Callable, Mapping, Optional

from rich.markup import escape

from .config import Settings
from .jev import (
    JEV_API_KEY_ENV,
    JEV_BASE_URL_ENV,
    JEV_DEFAULT_BASE_URL,
    JEV_DOCS_URL,
    JEV_HOME_URL,
    JEV_MODEL_ENV,
    TEACH_JEV,
    JevClient,
    JevError,
    redact_key,
    validate_api_key_format,
)
from .ui import UI, UserQuit

ClientFactory = Callable[[str], JevClient]

LOCAL_ONLY_LABEL = "Never mind, play with the local model only (free, nothing leaves this computer)"

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
) -> Optional[JevClient]:
    """Ask whether to enable Jev and, if so, get a working API key.

    Returns a ready :class:`JevClient`, or ``None`` to play with the local
    model only. Never raises for anything the player does: Ctrl+C anywhere in
    here just means "skip Jev".

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
    """
    env = os.environ if env is None else env
    factory = client_factory or _default_factory(env)
    flow = _JevOnboarding(ui, settings, env, factory, local_model_elsewhere)
    try:
        return flow.run()
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
    def make(key: str) -> JevClient:
        return JevClient(
            key,
            base_url=(env.get(JEV_BASE_URL_ENV) or "").strip() or None,
            model=(env.get(JEV_MODEL_ENV) or "").strip() or None,
        )

    return make


def clean_pasted_key(text: str) -> str:
    """Tidy common copy-paste accidents around a key.

    Strips surrounding whitespace and quotes, and a leading ``Bearer `` or
    ``TYPESAFE_API_KEY=`` in case a whole line was copied from docs or a shell.
    """
    key = (text or "").strip()
    for prefix in (f"export {JEV_API_KEY_ENV}=", f"{JEV_API_KEY_ENV}=", "Bearer ", "bearer "):
        if key.startswith(prefix):
            key = key[len(prefix):].strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    return key


def privacy_notice(env: Mapping[str, str]) -> str:
    """What Jev sends over the internet, and to whom - shown before the player turns it on.

    Never raises: a mistyped ``TYPESAFE_BASE_URL`` (say, ``http://[::1``,
    which the URL parser rejects) is shown as it is, flagged as odd.
    """
    raw = (env.get(JEV_BASE_URL_ENV) or "").strip() or JEV_DEFAULT_BASE_URL
    try:
        host = urllib.parse.urlsplit(raw if "://" in raw else "https://" + raw).hostname or raw
    except ValueError:
        host = f"{raw} - an address that doesn't look valid; check {JEV_BASE_URL_ENV}"
    return (
        f"Privacy: with Jev on, each round sends the plan you type, the current challenge, a short summary "
        f"of the story and your progress over the internet to TypeSafe AI ({host}), under their terms and "
        "privacy policy. With the local model only, everything stays on this computer."
    )


# The "back out" words the game teaches ("choose 'no' or 'back'"), accepted at
# the yes/no/learn menus that don't have a 'back' option of their own.
_BACK_OUT_ALIASES = {w: "no" for w in ("back", "b", "skip", "cancel", "quit", "exit", "local", "nope", "nah")}
# A confused player's words, at menus with a "learn" option: the same words that
# open the tips in the game and the lessons at the model menu.
_HELP_ALIASES = {w: "learn" for w in ("help", "h", "?", "info", "what", "explain", "more", "tell me more")}
_ENABLE_ALIASES = {**_BACK_OUT_ALIASES, **_HELP_ALIASES}
# "Use it?" is asked as a yes/no question: a yes means "use" (a no already means "no").
_USE_IT_ALIASES = {**_ENABLE_ALIASES, **{w: "use" for w in ("y", "yes", "yeah", "yep", "sure", "ok", "okay")}}
# ...and at menus whose way out is called 'back'.
_TO_BACK_ALIASES = {w: "back" for w in ("no", "n", "skip", "cancel", "quit", "exit", "local", "nope", "nah")}
# "Paste it here anyway?" is a yes/no question too.
_PASTE_ANYWAY_ALIASES = {**_TO_BACK_ALIASES, **{w: "paste" for w in ("y", "yes", "yeah", "yep", "sure", "ok")}}


class _JevOnboarding:
    """The onboarding conversation, one small method per step."""

    def __init__(self, ui: UI, settings: Settings, env: Mapping[str, str], factory: ClientFactory,
                 local_model_elsewhere: Optional[str] = None) -> None:
        self.ui = ui
        self.settings = settings
        self.env = env
        self.factory = factory
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
        self.ui.heading("Optional extra: Jev, the AI referee")
        existing = self._existing_key()
        if existing:
            key, source = existing
            choice = self._offer_existing_key(source, key)
            if choice == "use":
                result = self._try_key(key, source)
                return self._continue_from(result)
            if choice == "no":
                return self._go_local()
            # "new": fall through to the normal key menu
            return self._key_menu()

        if not self._ask_enable():
            return self._go_local()
        return self._key_menu()

    def _continue_from(self, result: "JevClient | str") -> Optional[JevClient]:
        """After a key check: a client means done; anything else is a next step."""
        if isinstance(result, JevClient):
            return result
        if result == _LOCAL:
            return self._go_local()
        return self._key_menu(start=result)

    # -- step: an API key we already know about -------------------------------------------

    def _existing_key(self) -> Optional[tuple[str, str]]:
        env_key = (self.env.get(JEV_API_KEY_ENV) or "").strip()
        if env_key:
            if validate_api_key_format(env_key) is None:
                return env_key, "env"
            self.ui.warn(
                f"Your {JEV_API_KEY_ENV} environment variable is set, but the value doesn't look like an "
                "API key, so I'll ignore it."
            )
        saved = (self.settings.jev_api_key or "").strip()
        if saved and validate_api_key_format(saved) is None:
            return saved, "saved"
        return None

    def _offer_existing_key(self, source: str, key: str) -> str:
        where = (
            f"in your {JEV_API_KEY_ENV} environment variable"
            if source == "env"
            else "that you saved last time"
        )
        self.ui.say(
            "Jev is TypeSafe AI's typed-judgment API. It can referee your plans with numbers instead "
            "of words (an optional, paid, third-party service)."
        )
        self.ui.say(f"[dim]{escape(privacy_notice(self.env))}[/dim]")
        self.ui.info(f"I found a Jev API key {where} (ending {escape(redact_key(key))}).")
        options = [
            ("use", "Check the key and turn on Jev"),
            ("new", "Use a different key"),
            ("no", f"No thanks, play with the local model only ({self._local_note})"),
            ("learn", "What's Jev? Tell me more first"),
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
            if choice != "learn":
                return choice
            self.ui.teach("What is Jev?", TEACH_JEV)

    # -- step: do you want Jev at all? ---------------------------------------------------------

    def _ask_enable(self) -> bool:
        ui = self.ui
        if self.settings.jev_enabled is False:
            # A returning player who chose the local referee last time: one short
            # question, not the whole introduction again (it's behind "learn").
            ui.say("Jev (the optional, paid AI referee) is off - last time you chose to play with your local model only.")
            choice = ui.choose(
                "Turn Jev on this time?",
                [
                    ("no", f"No thanks, play with the local model only ({self._local_note})"),
                    ("yes", "Yes, let Jev referee my plans (needs an API key)"),
                    ("learn", "What's Jev? Tell me more first"),
                ],
                default="no",
                aliases=_ENABLE_ALIASES,
            )
            if choice == "no":
                return False
            if choice == "yes":
                ui.say(f"[dim]{escape(privacy_notice(self.env))}[/dim]")
                return True
            ui.teach("What is Jev?", TEACH_JEV)
        ui.say(
            "[bold]Jev[/bold] is an AI from TypeSafe AI that gives [bold]typed judgments[/bold]: instead "
            "of chatting, it answers questions with numbers."
        )
        ui.say(
            "Turn it on and Jev referees each of your plans three ways - a [bold]Noul[/bold] (yes/no: did "
            "you make progress?), a [bold]Choice[/bold] (what kind of outcome?) and a [bold]Score[/bold] "
            "(how creative, 0-4) - so you can see how each one works."
        )
        ui.say(
            "Heads-up: Jev is a paid, third-party service with its own pricing and terms. "
            "This game isn't affiliated with TypeSafe AI."
        )
        ui.say(escape(privacy_notice(self.env)))
        ui.say(
            "The game works fully without it - your local model can referee for free. "
            "[dim](You can back out at any step: choose 'no' or 'back', or press Ctrl+C.)[/dim]"
        )
        remembered_yes = self.settings.jev_enabled is True
        while True:
            choice = self.ui.choose(
                "Enable Jev for this game?",
                [
                    ("yes", "Let Jev referee my plans (needs an API key)"),
                    ("no", f"No thanks, play with the local model only ({self._local_note})"),
                    ("learn", "Tell me more about Jev first"),
                ],
                default="yes" if remembered_yes else "no",
                aliases=_ENABLE_ALIASES,
            )
            if choice == "learn":
                ui.teach("What is Jev?", TEACH_JEV)
                continue
            return choice == "yes"

    # -- step: get a key -----------------------------------------------------------------------

    def _key_menu(self, start: Optional[str] = None) -> Optional[JevClient]:
        """Loop until we have a working client or the player backs out.

        ``start`` lets a previous step jump straight to "paste" or "help".
        """
        step = start if start in (_PASTE, _HELP) else None
        while True:
            if step is None:
                step = self.ui.choose(
                    "How would you like to add your Jev API key?",
                    [
                        ("paste", "I have a key, let me paste it"),
                        ("help", "I don't have one yet, walk me through getting one"),
                        ("back", self._local_only_label),
                    ],
                    default="paste",
                    aliases=_TO_BACK_ALIASES,
                )
            if step == _HELP:
                step = self._show_help()
                continue
            if step == _PASTE:
                key = self._ask_for_key()
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
            ui.say(
                "[dim]Your key stays hidden: nothing shows on screen while you paste. "
                "(Tip: some terminals paste with right-click or Ctrl+Shift+V.)[/dim]"
            )
            ui.say("[dim](Press Enter with nothing typed to go back.)[/dim]")
            raw = ui.secret("Paste your Jev API key and press Enter")
        else:
            # getpass would quietly *show* the key here (an IDE console, piped input...): say so first.
            ui.warn(
                "This window can't hide what you type, so your key would be visible on screen (and in "
                "screen shares or recordings)."
            )
            ui.say(
                f"[dim]Safer: set the {JEV_API_KEY_ENV} environment variable and restart the game - "
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
            raw = ui.ask("Paste your Jev API key and press Enter (leave it empty to go back)")
        key = clean_pasted_key(raw)
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
        ui.heading("Getting a Jev API key - step by step")
        ui.say(f"  [bold]1.[/bold] Open the TypeSafe AI website: {JEV_HOME_URL}  (choose 'open' below and I'll do it for you)")
        ui.say("  [bold]2.[/bold] Sign up for an account, or log in if you already have one.")
        ui.say("  [bold]3.[/bold] In your dashboard, open the API keys section and create a new key.")
        ui.say("  [bold]4.[/bold] Copy the key. Keep it private, like a password - many sites only show it once.")
        ui.say("  [bold]5.[/bold] Come back to this window and choose 'paste'.")
        ui.say(f"Want to read more first? Jev's documentation is at {JEV_DOCS_URL}")
        ui.say(
            "[dim]Remember: Jev is a paid service with its own pricing and terms - check them on the site "
            "before signing up. You never have to: 'back' plays the game free with your local model.[/dim]"
        )
        ui.say(f"[dim]{escape(privacy_notice(self.env))}[/dim]")
        opened = False
        while True:
            choice = ui.choose(
                "What next?",
                [
                    ("open", "Open typesafe.ai in my browser"),
                    ("docs", "Open the Jev documentation in my browser"),
                    ("paste", "I've got my key, paste it now"),
                    ("back", self._local_only_label),
                ],
                default="paste" if opened else "open",
                aliases=_TO_BACK_ALIASES,
            )
            if choice == "open":
                ui.open_url(JEV_HOME_URL)
                ui.info("Take your time. When you've copied your key, come back and choose 'paste'.")
                opened = True
            elif choice == "docs":
                ui.open_url(JEV_DOCS_URL)
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
            client = self.factory(key)
        except JevError as exc:
            ui.warn(escape(exc.message))
            return _MENU
        except Exception as exc:  # a surprise here must never end the whole game
            ui.warn(escape(_unexpected(exc, key)))
            return _MENU
        while True:
            try:
                try:
                    with ui.status("Checking your key with Jev..."):
                        models = client.list_models()
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
                names = [str(m.get("name")) for m in models if m.get("name")]
                ui.success("Your key works - Jev is ready!")
                if names:
                    ui.info(escape(f"Models available to your key: {', '.join(names[:5])}"))
                self._enable(client, key, source, verified=True)
                return client

    # -- finishing up -----------------------------------------------------------------------------------

    def _enable(self, client: JevClient, key: str, source: str, *, verified: bool) -> None:
        self.settings.jev_enabled = True
        if source == "env":
            self.ui.info(f"Using the key from your {JEV_API_KEY_ENV} environment variable (nothing saved to disk).")
        elif source == "pasted" and verified:
            self._offer_to_save(key)
        elif source == "pasted" and self.settings.jev_api_key and self.settings.jev_api_key != key:
            # A new, unchecked key replaces the saved one for this game: don't keep the old one on disk.
            self._forget_saved_key("The key saved last time has been removed - you're using a new one now.",
                                   save=False)
        self._save_settings()
        self.ui.say(
            "Jev is on! Each round it will referee your plan with a [bold]Noul[/bold], a "
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
            f"{protection}. Prefer not to? Set the {JEV_API_KEY_ENV} environment variable instead and the "
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
            self.settings.jev_api_key = key
            ui.info(f"Saved. (Delete it any time by running the game with --reset, or by editing {escape(str(self.settings.path))}.)")
        elif self.settings.jev_api_key:
            self.settings.jev_api_key = None  # the old key was replaced: what's on disk must match what we say
            ui.info("Not saved - the key only lives in memory while the game runs, and the key saved last "
                    "time has been removed from this computer.")
        else:
            ui.info("Not saved - the key only lives in memory while the game runs.")

    def _forget_saved_key(self, message: str, *, save: bool = True) -> None:
        if not self.settings.jev_api_key:
            return
        self.settings.jev_api_key = None
        if save:
            self._save_settings()
        self.ui.info(message)

    def _go_local(self) -> None:
        self.settings.jev_enabled = False
        self._save_settings()
        self.ui.info("Playing with your local model only - it will referee your plans. Have fun!")
        return None

    def _save_settings(self) -> None:
        try:
            self.settings.save()
        except OSError as exc:
            self.ui.warn(
                escape(f"Couldn't save your settings ({exc}). No harm done - the game will just ask again next time.")
            )
            return
        if self.settings.jev_api_key and getattr(self.settings, "key_file_protected", None) is False:
            # Windows wouldn't give the file an owner-only access list: say so plainly.
            self.ui.warn(
                f"Windows wouldn't let me restrict who can read {escape(str(self.settings.path))}, so other "
                "accounts on this PC may be able to read your saved key. To be safe, run the game once with "
                f"--reset and set the {JEV_API_KEY_ENV} environment variable instead."
            )


def _unexpected(exc: BaseException, key: str) -> str:
    """A friendly, key-free description of a surprise error."""
    detail = f"{type(exc).__name__}: {exc}".replace(key, redact_key(key)) if key else f"{type(exc).__name__}: {exc}"
    return f"Something unexpected went wrong while checking the key ({detail[:200]})."
