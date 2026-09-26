"""What the game tells players (and Steam) about its AI-written content.

* :data:`AI_CONTENT_NOTICE` - shown once, the first time the game starts
  (``cli.py`` remembers it in the settings). Markdown, short and friendly.
* :data:`STEAM_AI_DISCLOSURE` - the text for the "Live-Generated" answer of
  Steam's content survey (and the store page's AI disclosure). It has to
  describe the guardrails *accurately*, so if the filter in ``safety.py`` or
  the retry rules in ``game.py`` change, update it too.
"""

from __future__ import annotations

from typing import Optional

__all__ = ["AI_CONTENT_NOTICE", "STEAM_AI_DISCLOSURE", "REPORT_HOW", "REPORT_BUTTON", "STEAM_APP_ID",
           "report_url"]

# The game's Steam app ID - set it once Steam has given the game one (see the release
# checklist in packaging/steam/README.md). Until then "Report a problem" opens a Steam
# store search for the game, which leads to the same Discussions.
STEAM_APP_ID: Optional[int] = None
STORE_SEARCH_URL = "https://store.steampowered.com/search/?term=Get+To+Work"

REPORT_BUTTON = "Report a problem"  # the button in the game's window

# How players can report something the AI shouldn't have written. (Steam's overlay
# can't open over the game's window on Windows, macOS or a Linux desktop - the window
# isn't drawn with a 3D graphics API for it to hook into - so the report path is the
# game's own button, which opens the Discussions in the browser.)
REPORT_HOW = (
    f"click **{REPORT_BUTTON}** in the game's window to open the game's Steam Discussions, or open "
    "the Discussions from the game's Steam store page (on a Steam Deck in Game Mode, the Steam button's "
    "menu works too)"
)


def report_url(app_id: Optional[int] = None) -> str:
    """Where "Report a problem" leads: the game's Steam Discussions (or a store search before the app ID is known)."""
    app_id = STEAM_APP_ID if app_id is None else app_id
    if app_id:
        return f"https://steamcommunity.com/app/{int(app_id)}/discussions/"
    return STORE_SEARCH_URL

AI_CONTENT_NOTICE = f"""\
**A quick note before you play:** the story in Get To Work is written live by an AI \
that runs on your own computer, so every morning turns out differently.

The game asks the AI to keep things silly and kind, and a family-friendly filter checks \
everything it writes (and everything you type) before it appears.

AI can still surprise us now and then. If you ever see something that shouldn't be there, \
please tell us: {REPORT_HOW}. Thank you!
"""

STEAM_AI_DISCLOSURE = """\
Get To Work uses generative AI to write story text live while you play. The text is written by an \
open-weight language model that runs locally on the player's own computer, through the llama.cpp engine \
that ships with the game, so the story is written on the player's machine rather than on a server. The \
in-game model menu only offers popular, instruction-tuned chat models, and it leaves out models that \
are marketed as uncensored, have had their safety training removed, or are made for adult content; \
such a model is refused even when a player names it themselves. \
Optionally, players can connect their own account for TypeSafe AI's Jev referee service: it then \
receives the player's plan and a short summary of the story, and returns only numbers and labels that \
decide each round - it writes no text that appears in the story. The game creates no AI images, audio \
or voices.

Guardrails for the live-generated text:
- Every request tells the model to write farcical, family-friendly slapstick in which nobody gets hurt, \
and the player's typed plan is always quoted as an in-story action, never followed as an instruction.
- Everything the model writes - story narration, challenges, endings, the referee's explanations, and any \
"thinking" shown in the end-of-game review - is checked by a built-in filter before it is shown. The \
filter blocks sexual content, slurs and hate speech, self-harm, graphic gore and hard drugs, and it sees \
through common disguises (capital letters, accents, look-alike characters, leetspeak, and letters spaced \
out or broken up with symbols). Milder swearing is masked (for example "d***").
- A blocked reply is never shown: the game asks the model once more with a stricter family-friendly \
instruction, and if the new reply is also blocked, it shows a pre-written line instead.
- What the player types is checked with the same filter; a blocked plan is refused with a friendly message \
before it reaches any AI model or service.
- On first launch the game explains that its story is AI-generated and how to report anything \
inappropriate: a "Report a problem" button in the game's window opens the game's Steam Discussions, which \
are also reachable from the store page.
"""
