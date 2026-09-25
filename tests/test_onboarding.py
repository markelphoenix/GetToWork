"""Tests for the Jev opt-in / API-key flow, driven by scripted input (no network, no browser)."""

from __future__ import annotations

import io
import json
import os
import stat
import sys
import urllib.error

import pytest
from rich.console import Console

from gettowork import onboarding
from gettowork.config import Settings
from gettowork.jev import JEV_DOCS_URL, JEV_HOME_URL, JevClient
from gettowork.onboarding import clean_pasted_key, run_jev_onboarding
from gettowork.ui import UI

KEY = "tsk_live_ABCDEFGHIJKLMNOPQRSTUVWXYZ123456wxyz"
OTHER_KEY = "tsk_live_ZYXWVUTSRQPONMLKJIHGFEDCBA654321abcd"
MODELS_OK = (200, {}, json.dumps({"models": [{"name": "jev-latest", "description": "d", "release_date": "2026-09-15"}]}).encode())


def status(code, body=None):
    return (code, {}, json.dumps(body).encode() if body is not None else b"")


class FakeTransport:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append({"method": method, "url": url, "headers": dict(headers)})
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class Script:
    """Scripted answers for UI prompts. KeyboardInterrupt/EOFError entries are raised."""

    def __init__(self, answers=(), secrets=()):
        self.answers = list(answers)
        self.secrets = list(secrets)
        self.prompts: list[str] = []
        self.opened: list[str] = []

    def _next(self, queue, prompt):
        self.prompts.append(prompt)
        if not queue:
            raise AssertionError(f"Unexpected prompt: {prompt!r}")
        item = queue.pop(0)
        if isinstance(item, type) and issubclass(item, BaseException):
            raise item()
        return item

    def input(self, prompt):
        return self._next(self.answers, prompt)

    def secret(self, prompt):
        return self._next(self.secrets, prompt)

    def open_url(self, url):
        self.opened.append(url)
        return True


class RecordingUI(UI):
    """A UI that remembers every menu it showed, to check there is always a way back."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.menus: list[tuple[str, list[tuple[str, str]]]] = []

    def choose(self, prompt, options, default=None, **kwargs):
        self.menus.append((prompt, list(options)))
        return super().choose(prompt, options, default, **kwargs)


class Harness:
    def __init__(self, answers=(), secrets=(), responses=(), env=None, settings=None):
        self.script = Script(answers, secrets)
        self.console = Console(file=io.StringIO(), width=200, color_system=None)
        self.ui = RecordingUI(
            console=self.console,
            input_fn=self.script.input,
            secret_fn=self.script.secret,
            open_url_fn=self.script.open_url,
        )
        self.transport = FakeTransport(*responses)
        self.settings = settings if settings is not None else Settings()
        self.env = env if env is not None else {}
        self.clients: list[JevClient] = []

    def factory(self, key):
        client = JevClient(key, transport=self.transport, max_retries=0, sleep=lambda s: None)
        self.clients.append(client)
        return client

    def run(self):
        return run_jev_onboarding(self.ui, self.settings, env=self.env, client_factory=self.factory)

    @property
    def output(self):
        return self.console.file.getvalue()

    def assert_no_leak(self, *keys):
        everything = self.output + "\n".join(self.script.prompts)
        for key in keys:
            assert key not in everything
            for client in self.clients:
                assert key not in repr(client)
            for call in self.transport.calls:
                # the real key is only ever in the outgoing Authorization header
                assert set(k for k, v in call["headers"].items() if key in v) <= {"Authorization"}

    def assert_menus_offer_local_only(self):
        for prompt, options in self.ui.menus:
            if prompt.startswith("Save the key"):
                continue  # Jev is already working by then; this only asks about saving
            labels = " ".join(label for _, label in options).lower()
            assert "local model only" in labels, prompt


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("GETTOWORK_HOME", str(tmp_path))
    for name in ("TYPESAFE_API_KEY", "TYPESAFE_BASE_URL", "TYPESAFE_DEFAULT_MODEL"):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def saved(home):
    return json.loads((home / "settings.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_env_key_valid_is_used_without_saving(home):
    h = Harness(answers=[""], responses=[MODELS_OK], env={"TYPESAFE_API_KEY": KEY})  # Enter = "use"
    client = h.run()
    assert isinstance(client, JevClient)
    assert client.key_hint == "****wxyz"
    assert h.transport.calls[0]["url"].endswith("/v1/models")
    assert h.transport.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"
    assert "TYPESAFE_API_KEY environment variable" in h.output
    assert "****wxyz" in h.output
    assert "Your key works" in h.output
    data = saved(home)
    assert data["jev_enabled"] is True and data["jev_api_key"] is None
    assert not any(p.startswith("Save the key") for p, _ in h.ui.menus)
    h.assert_no_leak(KEY)
    h.assert_menus_offer_local_only()


def test_paste_valid_key_and_decline_saving(home):
    h = Harness(answers=["yes", "paste", "no"], secrets=[KEY], responses=[MODELS_OK])
    client = h.run()
    assert client is not None and client.key_hint == "****wxyz"
    assert "Your key works" in h.output and "Jev is on!" in h.output
    assert "Not saved" in h.output
    assert "paid, third-party service" in h.output and "isn't affiliated with TypeSafe AI" in h.output
    data = saved(home)
    assert data["jev_enabled"] is True and data["jev_api_key"] is None
    assert KEY not in (home / "settings.json").read_text()
    h.assert_no_leak(KEY)
    h.assert_menus_offer_local_only()


def test_paste_valid_key_and_save_it(home):
    h = Harness(answers=["yes", "paste", "yes"], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None
    data = saved(home)
    assert data["jev_api_key"] == KEY and data["jev_enabled"] is True
    assert "plain text" in h.output and "TYPESAFE_API_KEY" in h.output
    if os.name == "posix":
        mode = stat.S_IMODE((home / "settings.json").stat().st_mode)
        assert mode == 0o600
    h.assert_no_leak(KEY)  # saved to the file, never printed


def test_save_prompt_defaults_to_no(home):
    h = Harness(answers=["yes", "paste", ""], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None
    assert saved(home)["jev_api_key"] is None


def test_key_menu_enter_means_paste(home):
    h = Harness(answers=["yes", "", "no"], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None


def test_saved_key_is_offered_and_kept(home):
    settings = Settings(jev_api_key=KEY, jev_enabled=True)
    h = Harness(answers=["use"], responses=[MODELS_OK], settings=settings)
    client = h.run()
    assert client is not None
    assert "saved last time" in h.output
    assert saved(home)["jev_api_key"] == KEY
    assert not any(p.startswith("Save the key") for p, _ in h.ui.menus)
    h.assert_no_leak(KEY)


def test_existing_key_but_use_a_different_one(home):
    h = Harness(answers=["new", "paste", "no"], secrets=[OTHER_KEY], responses=[MODELS_OK], env={"TYPESAFE_API_KEY": KEY})
    client = h.run()
    assert client.key_hint == "****abcd"
    assert h.transport.calls[0]["headers"]["Authorization"] == f"Bearer {OTHER_KEY}"
    h.assert_no_leak(KEY, OTHER_KEY)


def test_existing_key_declined_goes_local(home):
    h = Harness(answers=["no"], env={"TYPESAFE_API_KEY": KEY})
    assert h.run() is None
    assert h.transport.calls == []
    assert saved(home)["jev_enabled"] is False
    assert "local model only" in h.output


def test_existing_key_learn_then_use(home):
    h = Harness(answers=["learn", "use"], responses=[MODELS_OK], env={"TYPESAFE_API_KEY": KEY})
    assert h.run() is not None
    assert "Learn: What is Jev?" in h.output


# ---------------------------------------------------------------------------
# Saying no / learning more
# ---------------------------------------------------------------------------


def test_no_means_local_only_and_is_remembered(home):
    h = Harness(answers=["no"])
    assert h.run() is None
    assert saved(home)["jev_enabled"] is False
    assert "local model" in h.output
    h.assert_menus_offer_local_only()


def test_learn_then_no(home):
    h = Harness(answers=["learn", "no"])
    assert h.run() is None
    assert "Learn: What is Jev?" in h.output
    assert "respond with JSON" in h.output


def test_enter_defaults_to_no_for_new_players(home):
    h = Harness(answers=[""])
    assert h.run() is None


def test_enter_defaults_to_yes_if_jev_was_enabled_before(home):
    h = Harness(answers=["", "back"], settings=Settings(jev_enabled=True))
    assert h.run() is None
    assert any(p.startswith("How would you like to add") for p, _ in h.ui.menus)


def test_options_can_be_picked_by_number(home):
    h = Harness(answers=["2"])  # 2 = "no"
    assert h.run() is None


# ---------------------------------------------------------------------------
# Help path
# ---------------------------------------------------------------------------


def test_help_opens_browser_then_paste(home):
    h = Harness(answers=["yes", "help", "open", "docs", "paste", "no"], secrets=[KEY], responses=[MODELS_OK])
    client = h.run()
    assert client is not None
    assert h.script.opened == [JEV_HOME_URL, JEV_DOCS_URL]
    out = h.output
    for step in ("1.", "2.", "3.", "4.", "5."):
        assert step in out
    assert "Sign up" in out and "API keys section" in out and "create a new key" in out
    assert "https://typesafe.ai" in out and "https://docs.typesafe.ai/" in out
    assert "pricing and terms" in out
    h.assert_menus_offer_local_only()


def test_help_then_back_is_local_only(home):
    h = Harness(answers=["yes", "help", "back"])
    assert h.run() is None
    assert h.script.opened == []
    assert saved(home)["jev_enabled"] is False


def test_help_default_is_open_then_paste(home):
    h = Harness(answers=["yes", "help", "", "", "no"], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None
    assert h.script.opened == [JEV_HOME_URL]


def test_browser_failure_still_shows_link(home):
    h = Harness(answers=["yes", "help", "open", "back"])
    h.ui._open_url = lambda url: False
    assert h.run() is None
    assert "Copy this link instead: https://typesafe.ai" in h.output


# ---------------------------------------------------------------------------
# Problems with the key
# ---------------------------------------------------------------------------


def test_paste_invalid_key_then_back(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[status(401, {"error": "Invalid API key"})])
    assert h.run() is None
    assert "didn't accept the API key" in h.output
    assert "Invalid API key" in h.output
    assert "copied the whole key" in h.output
    assert saved(home)["jev_enabled"] is False
    h.assert_no_leak(KEY)
    h.assert_menus_offer_local_only()


def test_invalid_key_then_help_then_valid_key(home):
    h = Harness(
        answers=["yes", "paste", "help", "paste", "no"],
        secrets=[OTHER_KEY, KEY],
        responses=[status(401, {"error": "Invalid API key"}), MODELS_OK],
    )
    client = h.run()
    assert client.key_hint == "****wxyz"
    h.assert_no_leak(KEY, OTHER_KEY)


def test_invalid_key_then_paste_again(home):
    h = Harness(
        answers=["yes", "paste", "paste", "no"],
        secrets=[OTHER_KEY, KEY],
        responses=[status(403, {"detail": "Forbidden"}), MODELS_OK],
    )
    assert h.run().key_hint == "****wxyz"


def test_rejected_env_key_offers_paste(home):
    h = Harness(
        answers=["use", "paste", "no"],
        secrets=[OTHER_KEY],
        responses=[status(401, {"error": "Invalid API key"}), MODELS_OK],
        env={"TYPESAFE_API_KEY": KEY},
    )
    assert h.run().key_hint == "****abcd"
    h.assert_no_leak(KEY, OTHER_KEY)


def test_server_echoing_the_key_is_redacted_on_screen(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[status(401, {"error": f"bad key {KEY}"})])
    assert h.run() is None
    assert "bad key ****wxyz" in h.output
    h.assert_no_leak(KEY)


def test_badly_formatted_paste_returns_to_menu(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=["abc def"])
    assert h.run() is None
    assert "space or line break" in h.output
    assert h.transport.calls == []
    h.assert_no_leak("abc def")


def test_empty_paste_returns_to_menu(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[""])
    assert h.run() is None
    assert "Nothing was entered" in h.output


def test_pasted_key_is_cleaned_up(home):
    h = Harness(answers=["yes", "paste", "no"], secrets=[f'  Bearer "{KEY}"\n'], responses=[MODELS_OK])
    assert h.run() is not None
    assert h.transport.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize(
    "raw",
    [KEY, f"  {KEY}  ", f"Bearer {KEY}", f"'{KEY}'", f'"{KEY}"', f"TYPESAFE_API_KEY={KEY}", f"export TYPESAFE_API_KEY='{KEY}'"],
)
def test_clean_pasted_key(raw):
    assert clean_pasted_key(raw) == KEY


def test_billing_problem_then_retry_succeeds(home):
    h = Harness(
        answers=["yes", "paste", "retry", "no"],
        secrets=[KEY],
        responses=[status(402, {"message": "Payment required"}), MODELS_OK],
    )
    assert h.run() is not None
    assert "billing" in h.output and "Payment required" in h.output


def test_billing_problem_default_is_back(home):
    h = Harness(answers=["yes", "paste", ""], secrets=[KEY], responses=[status(402, {"message": "Payment required"})])
    assert h.run() is None


def test_invalid_env_key_is_ignored_with_warning(home):
    h = Harness(answers=["no"], env={"TYPESAFE_API_KEY": "has a space"})
    assert h.run() is None
    assert "doesn't look like an" in h.output
    h.assert_no_leak("has a space")


# ---------------------------------------------------------------------------
# Network trouble
# ---------------------------------------------------------------------------


def test_network_error_then_continue_without_validating(home):
    h = Harness(answers=["yes", "paste", "continue"], secrets=[KEY], responses=[urllib.error.URLError("offline")])
    client = h.run()
    assert isinstance(client, JevClient)
    assert "Couldn't reach Jev" in h.output
    assert "without checking" in h.output
    assert not any(p.startswith("Save the key") for p, _ in h.ui.menus)  # unverified keys aren't offered for saving
    data = saved(home)
    assert data["jev_enabled"] is True and data["jev_api_key"] is None
    h.assert_no_leak(KEY)
    h.assert_menus_offer_local_only()


def test_network_error_then_retry_succeeds(home):
    h = Harness(
        answers=["yes", "paste", "retry", "no"], secrets=[KEY], responses=[urllib.error.URLError("offline"), MODELS_OK]
    )
    assert h.run() is not None
    assert len(h.transport.calls) == 2


def test_network_error_then_back(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[TimeoutError("slow")])
    assert h.run() is None
    assert "didn't answer" in h.output


def test_server_error_offers_retry_continue_back(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[status(503, {"error": "maintenance"})])
    assert h.run() is None
    assert "trouble on its side" in h.output
    options = [key for key, _ in h.ui.menus[-1][1]]
    assert options == ["retry", "continue", "back"]


# ---------------------------------------------------------------------------
# Ctrl+C / Ctrl+D always backs out to local-only
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "answers, secrets, responses, env",
    [
        ([KeyboardInterrupt], [], [], {}),  # at "enable Jev?"
        ([EOFError], [], [], {}),
        (["yes", KeyboardInterrupt], [], [], {}),  # at the key menu
        (["yes", "paste"], [KeyboardInterrupt], [], {}),  # while pasting (hidden input)
        (["yes", "help", KeyboardInterrupt], [], [], {}),  # in the help menu
        (["yes", "paste", KeyboardInterrupt], [KEY], [status(401, {"error": "no"})], {}),  # after a rejected key
        (["yes", "paste", KeyboardInterrupt], [KEY], [urllib.error.URLError("offline")], {}),  # network menu
        (["yes", "paste", KeyboardInterrupt], [KEY], [MODELS_OK], {}),  # at "save the key?"
        ([KeyboardInterrupt], [], [], {"TYPESAFE_API_KEY": KEY}),  # at "use the found key?"
    ],
)
def test_ctrl_c_anywhere_means_local_only(home, answers, secrets, responses, env):
    h = Harness(answers=answers, secrets=secrets, responses=responses, env=env)
    assert h.run() is None
    assert "skipping Jev" in h.output
    h.assert_no_leak(KEY)


# ---------------------------------------------------------------------------
# Odds and ends
# ---------------------------------------------------------------------------


def test_settings_save_failure_is_friendly(home, monkeypatch):
    def broken_save(self):
        raise OSError("disk full")

    monkeypatch.setattr(Settings, "save", broken_save)
    h = Harness(answers=["yes", "paste", "no"], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None
    assert "Couldn't save your settings" in h.output and "disk full" in h.output


def test_default_factory_reads_base_url_and_model_from_env():
    make = onboarding._default_factory({"TYPESAFE_BASE_URL": "https://example.test/", "TYPESAFE_DEFAULT_MODEL": "jev-x"})
    client = make(KEY)
    assert client.base_url == "https://example.test" and client.model == "jev-x"
    plain = onboarding._default_factory({})(KEY)
    assert plain.base_url == "https://api.typesafe.ai" and plain.model == "jev-latest"


def test_env_defaults_to_os_environ(home, monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", KEY)
    h = Harness(answers=["no"])
    result = run_jev_onboarding(h.ui, h.settings, client_factory=h.factory)  # env=None -> os.environ
    assert result is None
    assert "environment variable" in h.output


def test_markup_in_server_messages_does_not_break_output(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[status(401, {"error": "bad [/bold] key [red]"})])
    assert h.run() is None
    assert "bad [/bold] key [red]" in h.output


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permissions")
def test_saved_file_is_owner_only(home):
    h = Harness(answers=["yes", "paste", "yes"], secrets=[KEY], responses=[MODELS_OK])
    h.run()
    assert stat.S_IMODE((home / "settings.json").stat().st_mode) & 0o077 == 0


# ---------------------------------------------------------------------------
# Review fixes
# ---------------------------------------------------------------------------


def test_ctrl_c_while_the_key_is_being_checked_skips_jev(home):
    """The spinner isn't a prompt: Ctrl+C there used to end the whole game."""
    h = Harness(answers=["yes", "paste"], secrets=[KEY], responses=[KeyboardInterrupt()])
    assert h.run() is None
    assert "skipping Jev" in h.output
    h.assert_no_leak(KEY)


def test_a_base_url_without_https_scheme_is_fixed_not_fatal(home):
    make = onboarding._default_factory({"TYPESAFE_BASE_URL": "api.typesafe.ai"})
    assert make(KEY).base_url == "https://api.typesafe.ai"


def test_an_unusable_base_url_explains_itself_and_offers_the_way_back(home):
    factory = onboarding._default_factory({"TYPESAFE_BASE_URL": "http://jev.example.com"})
    h = Harness(answers=["use", "back"], env={"TYPESAFE_API_KEY": KEY})
    result = run_jev_onboarding(h.ui, h.settings, env=h.env, client_factory=factory)
    assert result is None
    assert "doesn't use https" in h.output and "local model" in h.output


def test_an_unexpected_error_during_the_key_check_is_a_retry_menu_not_a_crash(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[KEY], responses=[RuntimeError("surprise " + KEY)])
    assert h.run() is None
    assert "Something unexpected went wrong" in h.output
    h.assert_no_leak(KEY)


@pytest.mark.parametrize("word", ["back", "skip", "no", "help", "Quit", "cancel"])
def test_navigation_words_at_the_key_prompt_are_never_sent_to_jev(home, word):
    h = Harness(answers=["yes", "paste", "back"], secrets=[word])
    assert h.run() is None
    assert h.transport.calls == []  # nothing went to the network
    assert "Got a key" not in h.output and "Going back a step" in h.output


def test_the_key_prompt_says_how_to_go_back(home):
    h = Harness(answers=["yes", "paste", "back"], secrets=[""])
    h.run()
    assert "Press Enter with nothing typed to go back" in h.output


def test_no_hidden_input_promise_when_the_window_cannot_hide_it(home, monkeypatch):
    h = Harness(answers=["yes", "paste", "back", "back"])
    monkeypatch.setattr(h.ui, "can_hide_input", lambda: False)
    assert h.run() is None
    assert "can't hide what you type" in h.output and "TYPESAFE_API_KEY" in h.output
    assert "nothing shows on screen" not in h.output
    assert h.script.secrets == [] and h.transport.calls == []


def test_visible_paste_is_allowed_after_the_warning(home, monkeypatch):
    h = Harness(answers=["yes", "paste", "paste", KEY, "no"], responses=[MODELS_OK])
    monkeypatch.setattr(h.ui, "can_hide_input", lambda: False)
    assert h.run() is not None


def test_opt_in_screens_say_what_jev_sends_and_to_whom(home):
    h = Harness(answers=["no"])
    h.run()
    text = " ".join(h.output.split())
    assert "sends the plan you type, the current challenge" in text and "api.typesafe.ai" in text
    assert "everything stays on this computer" in text
    h2 = Harness(answers=["no"], env={"TYPESAFE_API_KEY": KEY, "TYPESAFE_BASE_URL": "https://jev.example.com"})
    h2.run()
    assert "jev.example.com" in h2.output  # the real destination, even when overridden
    h3 = Harness(answers=["yes", "help", "back"])
    h3.run()
    assert h3.output.count("Privacy:") == 2  # the opt-in screen and the how-to-get-a-key walkthrough


def test_replacing_a_saved_key_without_saving_forgets_the_old_one(home):
    Settings(jev_api_key=KEY, jev_enabled=True).save()
    h = Harness(answers=["new", "paste", "no"], secrets=[OTHER_KEY], responses=[MODELS_OK], settings=Settings.load())
    assert h.run() is not None
    assert saved(home)["jev_api_key"] is None  # what's on disk matches "not saved"
    assert "removed from this computer" in h.output


def test_a_rejected_saved_key_is_forgotten(home):
    Settings(jev_api_key=KEY, jev_enabled=True).save()
    h = Harness(answers=["use", "back"], responses=[status(401, {"error": "revoked"})], settings=Settings.load())
    assert h.run() is None
    assert saved(home)["jev_api_key"] is None
    assert "removed it" in h.output


def test_a_saved_key_can_be_forgotten_from_the_menu(home):
    Settings(jev_api_key=KEY, jev_enabled=True).save()
    h = Harness(answers=["forget"], settings=Settings.load())
    assert h.run() is None
    assert saved(home)["jev_api_key"] is None and saved(home)["jev_enabled"] is False
    assert h.transport.calls == []


def test_yes_no_menus_accept_y_and_n(home):
    h = Harness(answers=["y", "p", "n"], secrets=[KEY], responses=[MODELS_OK])
    assert h.run() is not None  # "y" = yes, "p" = paste, "n" = don't save
    assert saved(home)["jev_api_key"] is None


# ---------------------------------------------------------------------------
# Round 3
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("url", ["http://[::1", "[fe80::1", "https://[", "https://api.x.ai]"])
def test_a_malformed_base_url_never_crashes_onboarding(home, url):
    assert url in onboarding.privacy_notice({"TYPESAFE_BASE_URL": url})
    h = Harness(answers=[""], env={"TYPESAFE_BASE_URL": url})
    assert h.run() is None  # Enter = no, local model only; no ValueError


def test_a_key_ending_in_markup_is_shown_literally(home):
    odd = "tsk_live_abcdefghijkl[/]"
    h = Harness(answers=["yes", "paste", "no"], secrets=[odd], responses=[MODELS_OK])
    assert h.run() is not None
    assert "ending ****l[/]" in h.output


def test_an_unexpected_error_in_onboarding_falls_back_to_local(home):
    h = Harness(answers=["yes", "paste"], secrets=[KEY])

    def broken_factory(key):
        raise RuntimeError("surprise")

    assert run_jev_onboarding(h.ui, h.settings, env={}, client_factory=broken_factory) is None
    assert "local model will referee" in h.output


@pytest.mark.parametrize("word", ["back", "quit", "skip", "cancel"])
def test_back_out_words_work_at_the_enable_menu(home, word):
    h = Harness(answers=[word])
    assert h.run() is None
    assert "Pick one of the options" not in h.output


def test_back_out_words_work_at_the_key_menu(home):
    h = Harness(answers=["yes", "quit"])
    assert h.run() is None
    assert "Pick one of the options" not in h.output


def test_returning_local_only_players_get_one_short_question(home):
    h = Harness(answers=[""], settings=Settings(jev_enabled=False))
    assert h.run() is None
    out = h.output
    assert "last time you chose to play with your local model only" in out
    assert "Noul" not in out and "Privacy:" not in out  # no full pitch again
    assert len(h.ui.menus) == 1


def test_returning_local_only_players_can_still_turn_jev_on(home):
    h = Harness(answers=["yes", "paste", "no"], secrets=[KEY], responses=[MODELS_OK],
                settings=Settings(jev_enabled=False))
    assert h.run() is not None
    assert "Privacy:" in h.output  # shown before anything is sent


def test_returning_local_only_players_can_ask_to_learn_first(home):
    h = Harness(answers=["learn", "no"], settings=Settings(jev_enabled=False))
    assert h.run() is None
    assert "Learn: What is Jev?" in h.output and "Noul" in h.output


# ---------------------------------------------------------------------------
# Round 4: yes-words at yes/no-shaped menus; help words open the lesson
# ---------------------------------------------------------------------------


def _menu_ui(answers):
    script = Script(answers)
    return UI(console=Console(file=io.StringIO(), width=120), input_fn=script.input), script


@pytest.mark.parametrize("word", ["y", "yes", "yeah", "sure", "ok"])
def test_a_yes_at_use_it_means_use_the_key(word):
    ui, _ = _menu_ui([word])
    options = [("use", "Check the key"), ("new", "Different key"), ("no", "Local only"), ("learn", "More")]
    assert ui.choose("Use it?", options, default="use", aliases=onboarding._USE_IT_ALIASES) == "use"


@pytest.mark.parametrize("word", ["y", "yes"])
def test_a_yes_at_paste_it_anyway_means_paste(word):
    ui, _ = _menu_ui([word])
    options = [("back", "No, go back"), ("paste", "Yes, paste it (it will be visible)")]
    assert ui.choose("Paste it here anyway?", options, default="back",
                     aliases=onboarding._PASTE_ANYWAY_ALIASES) == "paste"


@pytest.mark.parametrize("word", ["help", "h", "?", "what", "explain"])
def test_help_words_open_the_jev_lesson(word):
    h = Harness(answers=[word, "no"])
    assert h.run() is None
    assert "calibrated" in h.output  # the lesson (TEACH_JEV) was shown
    assert h.script.prompts[0].startswith("[bold]Enable Jev") and len(h.script.prompts) == 2
