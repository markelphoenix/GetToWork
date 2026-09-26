"""The core game loop: story -> your plan -> the referee -> what happens next.

How a game goes: the opening scene says why you're late; round 1 asks how
you plan to get to work (bike, bus, broomstick...); after that, every round
is an obstacle that fits your way of travelling, each sillier than the last,
until you reach your desk.

Who does what:

* **Your local model** is the storyteller. It writes the opening, narrates
  what each plan does, invents the next (ever sillier) challenge, and writes
  the ending. When Jev is off, it is also the referee, answering with a tiny
  JSON verdict.
* **Jev** (optional) is the referee when enabled: one typed request per round
  answers a Noul (did you make progress?), a Choice (what kind of outcome?)
  and a Score (how creative?). The Noul decides; the other two are there to
  show off the question types.

Nothing the player does - and nothing a model or Jev gets wrong - should
cost the player their round: a failed Jev call falls back to the local
referee, an unreadable verdict is retried and then decided by a simple
backup rule, and a model error offers retry / skip / quit.

Everything is recorded in a :class:`~gettowork.types.GameSummary` for the
end-of-game review (``review.py``).

Safety note for learners: every piece of text that came from a model, from
Jev or from the player is printed with ``rich.markup.escape`` (or as a
``rich.text.Text``), so a model writing ``[bold]`` - or ``[/]`` - can't
restyle or crash the terminal UI; and it goes through ``ui.safe_text``, which
removes terminal control codes (which could otherwise retitle the window or
draw over earlier lines).

Family-friendly filter (``safety.py``): every piece of model text the player
will see - the opening, each story and challenge, the endings and the
referee's explanation - is checked first. A reply that doesn't pass is asked
for once more with a firmer reminder (``prompts.safety_retry_messages``);
if that fails too, a built-in line takes its place. Swearing is masked
("d***") before anything is shown. A plan that doesn't pass is refused
before it reaches the model or Jev, and the player simply tries another one.
What the filter did is noted in the round record (``RoundRecord.safety_notes``)
and in ``Game.safety_notes`` - categories only, never the words - and the
transcript keeps a "hidden" note in place of any blocked reply.

Speed: thinking models only think out loud when they're quick enough (see
``catalog.THINKING_MIN_TOKENS_PER_S``), and never for the story narration -
a 120-word joke doesn't need it - so a turn never hides minutes of silent
pondering behind a spinner.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Iterable, Iterator, Optional

from rich.console import Group
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from . import catalog
from . import jev as jevlib
from . import prompts
from . import safety
from .backends.base import BackendError, LLMBackend, supported_chat_options
from .jev import JevClient, JevError
from .prompts import COMMUTE_CHALLENGE
from .types import GameSummary, JevVerdict, LLMResult, RoundRecord
from .ui import UI, UserChoseQuit, option_hint, plain, safe_text

__all__ = ["Game", "backup_verdict", "progress_meter", "probability_bar"]

QUIT_WORDS = frozenset({"quit", "q", "exit"})
HELP_WORDS = frozenset({"help", "h", "?"})

# Token budgets. "Thinking" models spend tokens on reasoning before the answer,
# so calls that may think get generous budgets; calls that don't think need
# only room for the answer. Generation stops early at the end of the answer anyway.
NARRATION_MAX_TOKENS = 1200  # a story call that may think first
JUDGE_MAX_TOKENS = 768  # a referee call that may think first
STORY_MAX_TOKENS = 700  # a story call without thinking (~150 words plus plenty of slack)
JUDGE_NO_THINK_MAX_TOKENS = 300  # a referee call without thinking
SMALL_CONTEXT_TOKENS = 2048  # models the fit engine squeezed into a short context...
SMALL_CONTEXT_STORY_MAX_TOKENS = 500  # ...get shorter answers, so prompt + answer still fit
SMALL_CONTEXT_JUDGE_MAX_TOKENS = 256
# A model that *always* thinks first (its chat template forces it; asking it to
# skip isn't guaranteed to work with every engine) gets room for that thinking
# on top of the answer, so the story isn't cut off before it starts.
ALWAYS_THINKING_EXTRA_TOKENS = 1500
NARRATION_TEMPERATURE = 0.9  # creative
JUDGE_TEMPERATURE = 0.2  # consistent
# Where a rambling small model starts writing the player's lines or the next round.
STORY_STOPS = ["\nPlayer:", "\nPLAYER:", "\nYou:", "\nWhat do you do"]

SPINNERS = {
    "intro": "Your alarm clock is clearing its throat…",
    "judge": "The universe is consulting its rulebook…",
    "judge_retry": "The universe is squinting at its rulebook again…",
    "jev": "Asking Jev…",
    "outcome": "The plot thickens…",
    "victory": "Rolling out the red carpet…",
    "ending_quit": "Writing your excuse note…",
    "safety_retry": "Asking for a more family-friendly version…",
}

# What the player is told when the family-friendly filter steps in (see safety.py).
FAMILY_FRIENDLY_REFUSAL = "Let's keep it family-friendly - try another plan!"
SAFETY_RETRY_INFO = "That bit didn't pass the family-friendly filter, so I'm asking for a cleaner version…"
SAFETY_BUILT_IN_INFO = "Still not quite family-friendly, so here's a built-in version instead."

# Used when the model can't (or the player chooses to skip it).
FALLBACK_INTRO = (
    "BRRRING! You wake up with your face in a bowl of cereal and your alarm clock humming a sad little tune. "
    "It is 8:45. Work starts at 9:00. Today is the day your boss is handing out the Golden Stapler for "
    "punctuality, and you have promised - in writing - to be on time. You grab one shoe, a slice of toast and "
    "your dignity (mostly), and fling open the front door."
)
# Built-in challenges, by how absurd they are (the same five levels as
# prompts.ABSURDITY_LEVELS, ending with the finale at the office), used when
# the model's story has no usable challenge or the player skips it.
FALLBACK_CHALLENGE_TIERS: tuple[tuple[str, ...], ...] = (
    (  # a silly mishap close to home
        "A committee of squirrels has declared your front path a nut-storage zone and demands a toll of three acorns.",
        "Your bicycle has joined a jazz band and refuses to move unless someone plays the tambourine.",
    ),
    (  # a strange problem out on the street
        "A cloud shaped exactly like your boss is hovering over the street, raining tiny memos on everyone.",
        "The bus stop has wandered off to see the sea, and the bus is circling, looking very confused.",
    ),
    (  # a surreal obstacle on the journey
        "The pavement has turned into a very slow escalator going the wrong way, and it is enjoying itself.",
        "A polite dragon is using the zebra crossing as a sunbed and has asked not to be disturbed until noon.",
    ),
    (  # something magical near the edge of town
        "A troll under the bridge is charging a toll of one good joke, and it has heard all of the good ones.",
        "Every door in town now opens only for people who can whistle the national anthem backwards.",
    ),
    (  # the finale, right at the office
        "The office's revolving door has become a philosopher and will only let you in if you can say what a "
        "Monday really is.",
        "The lift to your floor has declared itself an independent nation and wants to see your passport.",
    ),
)
FALLBACK_CHALLENGES = tuple(c for tier in FALLBACK_CHALLENGE_TIERS for c in tier)
# Built-in obstacles that only make sense for one way of travelling.
_FALLBACK_NEEDS = {
    FALLBACK_CHALLENGE_TIERS[0][1]: "bike",  # "Your bicycle has joined a jazz band..."
    FALLBACK_CHALLENGE_TIERS[1][1]: "bus",  # "...the bus is circling..."
}


def _fits_commute(challenge: str, commute: Optional[str]) -> bool:
    need = _FALLBACK_NEEDS.get(challenge)
    if need is None:
        return True
    from .backends.mock import travel_mode  # the same simple word rules as the pretend model

    return travel_mode(commute or "") == need
FALLBACK_VICTORY = (
    "The office doors swing open with a heroic whoosh. A brass band that nobody booked plays a triumphant "
    "fanfare as you stride across the lobby - slightly singed, faintly smelling of adventure - and drop into "
    "your chair at 8:59 and 59 seconds exactly. Your boss looks up, looks at the clock, and gives you the "
    "smallest nod in recorded history. YOU GOT TO WORK!"
)
FALLBACK_QUIT = (
    "You decide that today is simply not a day for heroics. You wander home, make a cup of tea, and watch "
    "the ridiculous morning carry on without you through the window. The office will still be there "
    "tomorrow. Probably. Come back and try again whenever you're ready!"
)
FALLBACK_OUT_OF_ROUNDS = (
    "The town clock clears its throat and strikes nine, very pointedly. You're not at your desk yet, but "
    "what a morning it has been! Somewhere, your boss sighs and makes a note to buy you a louder alarm "
    "clock. Better luck next time!"
)

HOW_TO_PLAY = """\
- **First, say how you'll get to work** - on foot, by bike, bus, broomstick... The
  obstacles that follow will fit your choice.
- **Then type what you do** to get past each challenge, in your own words - for example
  *"I bribe the geese with a bagel and tiptoe past."*
- **Cartoon logic rules.** Silly, magical or impossible ideas are welcome, as long as they
  deal with *this* challenge.
- Doing nothing, giving up, or *"I teleport to work and win"* won't count - the referee
  wants to see you tackle the obstacle.
- Reach **{target}** steps to get to work. Type **quit** to stop early - you'll still get
  the behind-the-scenes review.
"""

LOCAL_JUDGE_EXPLAINER = """\
**Your local model is the referee.** Each round the game sends it your plan and asks for
a tiny JSON verdict:

```json
{{"made_progress": true, "explanation": "one short sentence"}}
```

A chat model writes that JSON *one token at a time*, like any other text. Small models
sometimes add chatty words, wrap it in a code block or forget a quote - so the game reads
it leniently, asks once more if it can't, and uses a simple backup rule as a last resort.

Notice the answer is a plain **yes or no**: nothing tells you how *sure* the model is.
{tail}
"""
_LOCAL_TAIL_NO_JEV = (
    "Jev, the optional typed-judgment API, answers with a calibrated *probability* instead - "
    "you can switch it on next time you play."
)
_LOCAL_TAIL_FALLBACK = (
    "Jev normally referees with a calibrated *probability*; your local model is just filling in this time."
)

# With --mock there is no model: the pretend model referees with a simple scripted rule.
LOCAL_JUDGE_EXPLAINER_MOCK = """\
**The pretend model is the referee** - and it isn't a real AI: it judges with a simple
scripted rule (a real plan of four or more words that doesn't give up, just wait, claim
victory or give the referee orders). It still answers the way the game asks a real model to,
with a tiny JSON verdict:

```json
{{"made_progress": true, "explanation": "one short sentence"}}
```

A real local model writes that JSON *one token at a time*, like any other text - small models
sometimes add chatty words or forget a quote, so the game reads it leniently and has a backup
rule. Play with a real model to see the difference.

Notice the answer is a plain **yes or no**: nothing tells you how *sure* the referee is.
{tail}
"""

COT_EXPLAINER_MOCK = """\
The pretend model shows **{words} words of scripted example "thinking"** - you can read it at
the end. It isn't a real AI, but real thinking models look just like this.

"Thinking" models (such as Qwen3, DeepSeek-R1 or gpt-oss) write out their reasoning - their
*chain-of-thought* - before the final answer. The game keeps it out of the story so things
flow, and saves every word for the behind-the-scenes review when the game ends.
"""

COT_EXPLAINER = """\
Your model **thought for {words} words** before answering - you can read it at the end.

"Thinking" models (such as Qwen3, DeepSeek-R1 or gpt-oss) write out their reasoning - their
*chain-of-thought* - before the final answer. The game keeps it out of the story so things
flow, and saves every word for the behind-the-scenes review when the game ends.

Thinking often makes answers better, but it costs time: every thought is generated token by
token, just like the story itself. So the game only lets your model think while setting the
scene and refereeing - and only if it's quick enough - never while narrating the story.
"""

# A little stage direction for the narrator, taken from Jev's Choice answer
# (only used when it agrees with the Noul verdict, which always decides).
_OUTCOME_FLAVOUR = {
    "triumph": (True, "Make the success spectacular - a marching band would not be out of place."),
    "progress": (True, "It works, even if things get a little messy."),
    "stalled": (False, "Nothing much changes; the obstacle is still there."),
    "setback": (False, "It backfires in a harmless, comic way."),
}

# The backup rule's explanation for each kind of plan that doesn't count (see prompts.screen_plan).
_BACKUP_REASONS = {
    "empty": "doing nothing doesn't get you to work.",
    "gave_up": "doing nothing doesn't get you to work.",
    "waits": "waiting doesn't get you past it.",
    "claims_victory": "just saying you're at work doesn't get you past the obstacle.",
    "orders_referee": "that's an instruction to the referee, not something you do in the story.",
    "too_short": "that's a bit short to count as a plan - try describing what you do.",
}


# A Noul between these counts as a close call when the Choice disagrees with it.
CLOSE_CALL_LOW, CLOSE_CALL_HIGH = 0.35, 0.65


def _screened_raw(value: Any) -> Any:
    """A backend's raw answer with every piece of text through the family-friendly filter.

    Blocked text becomes the short "hidden" note, milder swearing is masked;
    numbers, labels and the structure are kept as they were.
    """
    if isinstance(value, str):
        verdict = safety.check_text(value)
        return safety.soften(value) if verdict.ok else safety.hidden_note(verdict)
    if isinstance(value, dict):
        return {key: _screened_raw(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_screened_raw(item) for item in value]
    return value


class _QuitGame(Exception):
    """The player chose "quit" from an error menu."""


# ---------------------------------------------------------------------------
# Small helpers (public ones are handy for tests and learners)
# ---------------------------------------------------------------------------


def probability_bar(fraction: float, width: int = 20) -> str:
    """A plain-text bar such as ``#######-----`` for a value from 0 to 1."""
    fraction = min(1.0, max(0.0, float(fraction)))
    filled = int(round(fraction * width))
    return "#" * filled + "-" * (width - filled)


def progress_meter(progress: int, target: int, width: int = 10) -> str:
    """The progress meter, e.g. ``[####------] 2/5``."""
    target = max(1, int(target))
    shown = min(max(0, int(progress)), target)
    return f"[{probability_bar(shown / target, width)}] {shown}/{target}"


def backup_verdict(plan: str, *, commute: bool = False) -> tuple[bool, str]:
    """The last-resort referee, used when the model's verdict can't be read.

    A plan of at least 4 words counts as progress, unless it gives up, just
    waits, claims victory ("I teleport to work and win") or gives the referee
    orders - the same simple screen the pretend model uses
    (:func:`gettowork.prompts.screen_plan`). Deliberately simple and generous.
    In round 1 (``commute=True``) any way of travelling counts, however short ("by bike").
    """
    problem = prompts.screen_plan(plan, commute=commute)
    lead = "The referee couldn't give a clear verdict, so the backup rule decided: "
    if problem is not None:
        return False, lead + _BACKUP_REASONS.get(problem, _BACKUP_REASONS["too_short"])
    return True, lead + "that's a real attempt, so it counts!"


def _clip(text: Any, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _pct(fraction: float) -> str:
    return f"{round(min(1.0, max(0.0, fraction)) * 100)}%"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _first_flag(parts: Iterable[Optional[str]]) -> safety.SafetyVerdict:
    """The family-friendly filter's verdict on several texts: the first one that doesn't pass, else OK."""
    for part in parts:
        verdict = safety.check_text(part)
        if not verdict.ok:
            return verdict
    return safety.SafetyVerdict(ok=True)


def _family_label(text: Any) -> str:
    """A label that came back in Jev's reply: softened, or hidden if it doesn't pass the filter."""
    cleaned = safe_text(text)
    return safety.soften(cleaned) if safety.check_text(cleaned).ok else "(hidden)"


# ---------------------------------------------------------------------------
# The game
# ---------------------------------------------------------------------------



# The Jev answer types, taught one per round in this order (see Game._show_verdict).
_JEV_LESSONS = (
    ("noul", "Noul - a yes/no question", jevlib.TEACH_NOUL),
    ("choice", "Choice - pick one label", jevlib.TEACH_CHOICE),
    ("score", "Score - rate on a rubric", jevlib.TEACH_SCORE),
)

class Game:
    """One play-through of Get To Work.

    Args:
        llm: The local model backend (storyteller, and referee when Jev is off).
        ui: Where all input and output goes.
        jev: A ready :class:`~gettowork.jev.JevClient`, or ``None`` for local-only.
        target: Steps needed to reach your desk (default 5).
        max_rounds: Optional cap on rounds (``None`` = play until you win or quit).
        max_input_chars: Longer plans are trimmed to this many characters.
        tokens_per_s: The model's measured speed, if known. Below
            ``catalog.THINKING_MIN_TOKENS_PER_S`` thinking models are asked to
            answer straight away (no visible thinking), so turns stay quick.
        context_tokens: The model's context window, if the fit engine had to
            shorten it; small windows get shorter answers too.
        taught: The "Learn" panels already shown this session (updated in
            place), so a second game after "Play again" skips them.

    After a game, ``safety_notes`` lists what the family-friendly filter did
    (each round's share is also in its ``RoundRecord.safety_notes``).
    """

    def __init__(
        self,
        llm: LLMBackend,
        ui: UI,
        *,
        jev: Optional[JevClient] = None,
        target: int = 5,
        max_rounds: Optional[int] = None,
        max_input_chars: int = 500,
        tokens_per_s: Optional[float] = None,
        context_tokens: Optional[int] = None,
        secrets: Iterable[str] = (),
        thinking: Optional[str] = None,
        force_think: bool = False,
        taught: Optional[set[str]] = None,
    ) -> None:
        if int(target) < 1:
            raise ValueError("target must be at least 1")
        if max_rounds is not None and int(max_rounds) < 1:
            raise ValueError("max_rounds must be at least 1 (or None for no limit)")
        if int(max_input_chars) < 1:
            raise ValueError("max_input_chars must be at least 1")
        self.llm = llm
        self.ui = ui
        self.jev = jev
        self.target = int(target)
        self.max_rounds = None if max_rounds is None else int(max_rounds)
        self.max_input_chars = int(max_input_chars)

        self._use_jev = jev is not None  # turned off if the player gives up on Jev after an error
        self._summary = GameSummary(won=False, quit_early=False, progress=0, target=self.target, intro="", ending="")
        self._history: list[str] = []  # one short line per finished round, for prompts and Jev
        # Which "Learn" panels have been shown. cli passes one set for the whole
        # session, so "Play again" doesn't teach the same lessons a second time.
        self._taught: set[str] = taught if taught is not None else set()
        self._pending_thought_words: Optional[int] = None  # first exposed reasoning, taught after display
        self._pending_thought_mock = False  # ...and whether it came from the scripted pretend model
        self._fallback_used: set[str] = set()
        self._commute: Optional[str] = None  # how the player said they'd travel (round 1)
        self._challenges_seen = 0  # the number of the obstacle on screen (= steps completed; the commute doesn't count)
        self._last_challenge: Optional[str] = None  # a failed round may show the same obstacle again
        # Everything the family-friendly filter did this game (categories only, never the words).
        self.safety_notes: list[str] = []
        self._refusals: list[str] = []  # refused plans, noted in the next round's record
        small = context_tokens is not None and int(context_tokens) <= SMALL_CONTEXT_TOKENS
        fast = tokens_per_s is None or float(tokens_per_s) >= catalog.THINKING_MIN_TOKENS_PER_S
        self._may_think = (fast or bool(force_think)) and not small
        self._small_context = small
        # "none" / "switchable" / "always" (see catalog.thinking_mode); None = unknown.
        self.thinking = thinking
        # Why the game itself kept a thinking model from thinking out loud (shown in the review).
        self.thinking_note: Optional[str] = None
        if thinking == "switchable" and not self._may_think:
            if small:
                why = "its conversation memory had to be kept short to fit this computer"
            else:
                why = f"it runs at about {float(tokens_per_s or 0):.0f} tokens/s here, and thinking makes each turn much longer"
            hint = None if small else option_hint(ui, "--think", markup=False)  # (None: no command line to use)
            self.thinking_note = (
                "Your model can think out loud, but I asked it to answer straight away this game because "
                f"{why} - so there's no reasoning to show."
                + (f" To see its thinking anyway (turns take longer), {hint}." if hint else "")
            )
        self._chat_options = supported_chat_options(llm)
        # Credentials the player might paste by accident as a plan (e.g. the Jev
        # key still on the clipboard): such a plan is refused, never sent anywhere.
        known = set(secrets or ()) | (set(jev.secret_values()) if jev is not None and hasattr(jev, "secret_values") else set())
        self._secrets = {str(x).strip() for x in known if len(str(x).strip()) >= 8}

    # -- the main loop ---------------------------------------------------------

    def run(self) -> GameSummary:
        """Play until the player wins, quits or runs out of rounds.

        Ctrl+C / Ctrl+D at a prompt raises :class:`~gettowork.ui.UserQuit`,
        which the command line catches to exit cleanly.
        """
        summary = self._summary
        self._welcome()
        try:
            self._opening()
        except _QuitGame:
            return self._finish_quit(narrate=False)
        if self.jev is not None:
            self._teach_once("jev", "Jev, your referee today", jevlib.TEACH_JEV)
        self.ui.pause("Press Enter when you're ready to set off")
        challenge = COMMUTE_CHALLENGE  # round 1: how will you get to work?

        while True:
            if self.max_rounds is not None and len(summary.rounds) >= self.max_rounds:
                return self._finish_out_of_rounds()
            number = len(summary.rounds) + 1
            commute_round = challenge == COMMUTE_CHALLENGE
            self._show_challenge(number, challenge)
            plan = self._ask_plan(commute=commute_round)
            if plan is None:
                return self._finish_quit()

            record = RoundRecord(
                number=number,
                challenge=challenge,
                player_plan=plan,
                judge="local",
                made_progress=False,
                judge_explanation="",
                progress_after=summary.progress,
            )
            record.safety_notes.extend(self._refusals)  # plans refused before this one was accepted
            self._refusals.clear()
            try:
                backup_used = self._referee(record)
            except _QuitGame:
                return self._finish_quit(narrate=False)

            summary.rounds.append(record)
            if record.made_progress:
                summary.progress += 1
                if commute_round:
                    self._commute = plan
            record.progress_after = summary.progress
            self._show_verdict(record, backup_used)

            earlier_rounds = list(self._history)
            self._history.append(self._history_entry(record))
            if summary.progress >= self.target:
                return self._finish_victory(record)
            try:
                challenge = self._outcome(record, earlier_rounds)
            except _QuitGame:
                return self._finish_quit(narrate=False)

    # -- talking to the local model --------------------------------------------

    def _call_llm(
        self,
        purpose: str,
        messages: list[dict[str, str]],
        *,
        calls: list[tuple[str, LLMResult]],
        max_tokens: int = NARRATION_MAX_TOKENS,
        temperature: float = NARRATION_TEMPERATURE,
        json_mode: bool = False,
        skip_label: Optional[str] = None,
        interactive: bool = True,
        think: Optional[bool] = None,
        stop: Optional[list[str]] = None,
        spinner: Optional[str] = None,
    ) -> Optional[LLMResult]:
        """One local-model call, recorded in ``calls`` as ``(purpose, result)``.

        On a :class:`BackendError` the player picks: try again, skip (when
        ``skip_label`` is given; returns ``None``) or quit (raises
        ``_QuitGame``). With ``interactive=False`` an error just returns ``None``.
        ``think``/``stop`` are passed on to backends that support them.
        Any exposed reasoning goes through the family-friendly filter straight
        away (it's only ever shown in the review); the answer itself is checked
        by the caller, once it has picked out the parts the player will see.
        """
        spinner = spinner or SPINNERS.get(purpose, "Thinking…")
        extra: dict[str, Any] = {}
        if think is not None and "think" in self._chat_options:
            extra["think"] = think
        if stop and "stop" in self._chat_options:
            extra["stop"] = list(stop)
        while True:
            try:
                with self.ui.status(spinner) as update_status:
                    with self._notices_to(update_status):
                        result = self.llm.chat(messages, temperature=temperature, max_tokens=max_tokens,
                                               json_mode=json_mode, **extra)
            except BackendError as exc:
                if not interactive:
                    self.ui.warn(f"Your local model couldn't write this part ({plain(str(exc))}), so here's a built-in one.")
                    return None
                self.ui.error(f"Your local model hit a snag: {plain(str(exc))}")
                options = [("retry", "Try again")]
                if skip_label:
                    options.append(("skip", skip_label))
                options.append(("quit", "End the game here"))
                choice = self.ui.choose("What would you like to do?", options, default="retry")
                if choice == "retry":
                    continue
                if choice == "skip":
                    return None
                raise _QuitGame() from exc
            calls.append((purpose, result))
            # The engine's raw answer (kept for the saved transcript) repeats the text and the
            # thinking word for word: it gets the same filter, so nothing blocked or unmasked
            # survives there either.
            result.raw = _screened_raw(result.raw)
            if result.reasoning and not self._screen_reasoning(purpose, result):
                return result  # its reasoning was hidden: nothing to teach about
            if result.reasoning and self._pending_thought_words is None and "reasoning" not in self._taught:
                self._pending_thought_words = len(result.reasoning.split())
                self._pending_thought_mock = result.backend == "mock"
            return result

    @contextmanager
    def _notices_to(self, update_status: Any) -> Iterator[None]:
        """While a call runs, let the backend put short notes in the spinner ("asking again...")."""
        if not callable(update_status) or not isinstance(self.llm, LLMBackend):
            yield
            return
        self.llm.on_notice = update_status
        try:
            yield
        finally:
            self.llm.on_notice = None

    def _thinking_call(self, story: bool) -> dict[str, Any]:
        """``max_tokens`` and ``think`` for the intro (story=True) or the referee (story=False).

        These are the calls where a thinking model may think out loud - if
        it's fast enough and has room. Otherwise it's asked to answer straight away.
        """
        if self._small_context:
            return {"think": False,
                    "max_tokens": SMALL_CONTEXT_STORY_MAX_TOKENS if story else SMALL_CONTEXT_JUDGE_MAX_TOKENS}
        if self.thinking == "always":
            # It will think whatever we ask: leave room for the thinking and the answer.
            limit = (STORY_MAX_TOKENS if story else JUDGE_NO_THINK_MAX_TOKENS) + ALWAYS_THINKING_EXTRA_TOKENS
            return {"think": None if self._may_think else False, "max_tokens": limit}
        if self._may_think:
            return {"think": None, "max_tokens": NARRATION_MAX_TOKENS if story else JUDGE_MAX_TOKENS}
        return {"think": False, "max_tokens": STORY_MAX_TOKENS if story else JUDGE_NO_THINK_MAX_TOKENS}

    def _story_call(self) -> dict[str, Any]:
        """Options for narration (outcome / victory / quit): never any thinking, stop if it rambles."""
        limit = SMALL_CONTEXT_STORY_MAX_TOKENS if self._small_context else STORY_MAX_TOKENS
        if self.thinking == "always" and not self._small_context:
            limit += ALWAYS_THINKING_EXTRA_TOKENS  # asking it to skip the thinking may not work
        return {"think": False, "max_tokens": limit, "stop": STORY_STOPS}

    # -- the family-friendly filter (safety.py) ------------------------------------

    def _family_friendly(
        self,
        purpose: str,
        messages: list[dict[str, str]],
        result: Optional[LLMResult],
        read: Callable[[LLMResult], tuple[str, ...]],
        *,
        calls: list[tuple[str, LLMResult]],
        options: dict[str, Any],
        what: str,
        record: Optional[RoundRecord] = None,
    ) -> Optional[tuple[Optional[str], ...]]:
        """Check the parts of a story reply the player will see, before they're shown.

        ``read(result)`` picks those parts out of a reply - ``(story,)``, or
        ``(narration, challenge)``. If any part doesn't pass the filter, the
        model is asked once more (same request, firmer reminder). Parts that
        still don't pass come back as ``None``, so the caller can use a
        built-in line; the rest come back softened. ``None`` overall means
        there was no reply at all (the player skipped it, or the model failed).
        """
        if result is None:
            return None
        parts = tuple(read(result))
        verdict = _first_flag(parts)
        self._screen_record(result, parts)
        if verdict.ok:
            return tuple(safety.soften(p) for p in parts)

        self._safety_note(f"{what}: the model's reply didn't pass the family-friendly filter "
                          f"({verdict.label}), so it was asked again.", record)
        self.ui.info(SAFETY_RETRY_INFO)
        retry = self._call_llm(purpose, prompts.safety_retry_messages(messages), calls=calls, interactive=False,
                               spinner=SPINNERS["safety_retry"], **options)
        if retry is not None:
            parts = tuple(read(retry))
            verdict = _first_flag(parts)
            self._screen_record(retry, parts)
            if verdict.ok:
                self._safety_note(f"{what}: the second try passed the filter.", record)
                return tuple(safety.soften(p) for p in parts)
            self.ui.info(SAFETY_BUILT_IN_INFO)
        self._safety_note(f"{what}: no family-friendly reply, so a built-in line was used instead.", record)
        return tuple(safety.soften(p) if safety.check_text(p).ok else None for p in parts)

    def _screen_record(self, result: Optional[LLMResult], parts: Iterable[str] = ()) -> None:
        """Make the kept copy of a reply safe to show in the review and in saved transcripts.

        A reply that didn't pass the filter (as a whole, or any of the ``parts``
        picked out of it) is replaced by a short "hidden" note; otherwise its
        swearing is masked. Call this only *after* reading the reply.
        """
        if result is None:
            return
        verdict = _first_flag((result.text, *parts))
        if verdict.ok:
            result.text = safety.soften(result.text)
        else:
            result.text = safety.hidden_note(verdict)
            result.raw = None  # the engine's raw answer repeats the same text

    def _screen_reasoning(self, purpose: str, result: LLMResult) -> bool:
        """Filter a reply's exposed reasoning (shown in the review). False if it had to be hidden."""
        verdict = safety.check_text(result.reasoning)
        if verdict.ok:
            result.reasoning = safety.soften(result.reasoning)
            return True
        result.reasoning = safety.hidden_note(verdict)
        result.raw = None  # the engine's raw answer repeats the same thinking
        self._safety_note(f"The model's reasoning ({purpose}) didn't pass the family-friendly filter "
                          f"({verdict.label}) and was hidden.")
        return False

    def _safety_note(self, note: str, record: Optional[RoundRecord] = None) -> None:
        """Remember what the filter did: for the whole game, and for the round when there is one."""
        self.safety_notes.append(note)
        if record is not None:
            record.safety_notes.append(note)

    def _built_in_story(self, record: RoundRecord) -> str:
        """A safe, ready-made story line for this round, from the pretend model's script."""
        from .backends import mock  # the same built-in lines the offline pretend model uses

        plan, n = _clip(record.player_plan, 90), record.number
        if record.challenge == COMMUTE_CHALLENGE:
            lines = mock.COMMUTE_SUCCESS if record.made_progress else mock.COMMUTE_FAILURE
            return lines[n % len(lines)].format(plan=plan)
        openers = mock.SUCCESS_OPENERS if record.made_progress else mock.FAILURE_OPENERS
        follow_ups = mock.SUCCESS_TRANSITIONS if record.made_progress else mock.FAILURE_TRANSITIONS
        return f"{openers[n % len(openers)].format(plan=plan)} {follow_ups[n % len(follow_ups)]}"

    @staticmethod
    def _built_in_explanation(made_progress: bool, number: int) -> str:
        """A safe referee explanation, used when the real one didn't pass the filter."""
        from .backends import mock

        if made_progress:
            return mock.JUDGE_YES[number % len(mock.JUDGE_YES)]
        return "That doesn't quite get you past it."

    # -- phases -------------------------------------------------------------------

    def _welcome(self) -> None:
        ui = self.ui
        ui.heading("Get To Work!")
        ui.say(
            f"You're running late, and the universe is in a playful mood. Reach your desk in "
            f"[bold]{self.target}[/bold] steps by telling me what you do - silly ideas welcome!"
        )
        ui.say("[dim]Type [bold]help[/bold] for tips, or [bold]quit[/bold] to stop.[/dim]")
        try:
            storyteller = self.llm.model_label
        except Exception:  # a label is nice to have, never worth crashing over
            storyteller = getattr(self.llm, "name", "your local model")
        referee = f"Jev ({self.jev.model})" if self.jev is not None else self._local_referee_name
        ui.info(f"Storyteller: {escape(str(storyteller))}  |  Referee: {escape(referee)}")

    @property
    def _pretend(self) -> bool:
        """--mock: the "local model" is the pretend model's scripted rules, not an AI."""
        return getattr(self.llm, "name", None) == "mock"

    @property
    def _local_referee_name(self) -> str:
        return "the pretend model (a simple scripted rule)" if self._pretend else "your local model"

    def _opening(self) -> None:
        """The intro story. (No obstacle yet: round 1 asks how the player will get to work.)"""
        messages = prompts.intro_messages()
        options: dict[str, Any] = dict(stop=STORY_STOPS, **self._thinking_call(story=True))
        calls = self._summary.intro_calls
        result = self._call_llm("intro", messages, calls=calls, skip_label="Skip it and use a built-in opening",
                                **options)
        parts = self._family_friendly("intro", messages, result, lambda r: (prompts.clean_story(r.text),),
                                      calls=calls, options=options, what="The opening story")
        self._summary.intro = (parts[0] if parts else "") or FALLBACK_INTRO
        self.ui.narrate(plain(self._summary.intro), title="Good morning!")
        self._flush_thought_teach()

    def _ask_plan(self, *, commute: bool = False) -> Optional[str]:
        """Ask what the player does (or, in round 1, how they'll travel). ``None`` means quit."""
        question = "How do you plan to get to work?" if commute else "What do you do?"
        while True:
            text = " ".join(self.ui.ask(question).split())
            command = text.lower().strip(" .!")
            if not text:
                if commute:
                    self.ui.info("Tell me how you'll travel - on foot, by bike, bus, dragon... (help for tips, quit to stop)")
                else:
                    self.ui.info("Type what you do to get past it - anything goes! (help for tips, quit to stop)")
                continue
            if command in QUIT_WORDS:
                return None
            if command in HELP_WORDS:
                self._show_help()
                continue
            if any(secret in text for secret in self._secrets):
                self.ui.warn("That looks like your API key - I haven't used it as a plan or sent it anywhere. "
                             "Type what you do instead.")
                continue
            plan = text[: self.max_input_chars].rstrip() if len(text) > self.max_input_chars else text
            # The family-friendly filter: a plan that doesn't pass never reaches the model or Jev,
            # and doesn't use up a round - the player just tries another one.
            verdict = safety.check_player_input(text)
            if verdict.ok and plan != text:
                verdict = safety.check_player_input(plan)  # trimming could leave a different word at the end
            if not verdict.ok:
                self.ui.warn(FAMILY_FRIENDLY_REFUSAL)
                note = f"A plan was refused by the family-friendly filter ({verdict.label}); the player tried again."
                self._safety_note(note)
                self._refusals.append(note)
                continue
            if plan != text:
                self.ui.info(f"That's an epic plan! I kept the first {self.max_input_chars} characters.")
            return safety.soften(plan)

    def _referee(self, record: RoundRecord) -> bool:
        """Decide whether the plan made progress. Returns True if the backup rule was used."""
        if self._use_jev and self.jev is not None:
            verdict = self._ask_jev(record)
            if verdict is not None:
                record.judge = "jev"
                record.jev = verdict
                record.made_progress = verdict.made_progress
                record.judge_explanation = self._jev_explanation(record, verdict)
                return False
        return self._judge_locally(record)

    def _jev_explanation(self, record: RoundRecord, verdict: JevVerdict) -> str:
        """Jev's verdict in words. The game writes the sentence, but the labels in it come from Jev's
        reply, so it goes through the family-friendly filter too (a built-in line if it doesn't pass)."""
        text = jevlib.explain_verdict(verdict)
        checked = safety.check_text(text)
        if checked.ok:
            return safety.soften(text)
        self._safety_note(f"Round {record.number} referee: Jev's answer didn't pass the family-friendly filter "
                          f"({checked.label}), so a built-in explanation was used.", record)
        return self._built_in_explanation(verdict.made_progress, record.number)

    def _ask_jev(self, record: RoundRecord) -> Optional[JevVerdict]:
        """Ask Jev; on any failure explain, keep the exchange, and return None (local fallback)."""
        assert self.jev is not None
        try:
            with self.ui.status(SPINNERS["jev"]):
                return jevlib.judge_round(
                    self.jev,
                    intro=self._summary.intro,
                    # Round 1's "challenge" is choosing how to travel: any real way of setting off counts.
                    challenge=(prompts.COMMUTE_JUDGE_CHALLENGE if record.challenge == COMMUTE_CHALLENGE
                               else record.challenge),
                    plan=record.player_plan,
                    progress=self._summary.progress,
                    target=self.target,
                    history=list(self._history),
                )
        except JevError as exc:
            record.failed_jev_exchange = exc.exchange
            message, suggest_keep = exc.message, not (exc.is_auth_error or exc.kind in ("billing", "config"))
        except Exception as exc:  # Jev is optional: never let a surprise end the player's round
            message, suggest_keep = f"Something unexpected went wrong ({type(exc).__name__}: {exc}).", True
        self.ui.error(f"Jev couldn't referee this round: {plain(message)}")
        self.ui.info(f"No problem - {self._local_referee_name} will referee this round instead, so you don't "
                     "lose your turn.")
        try:
            keep = self.ui.confirm("Keep asking Jev in the next rounds?", default=suggest_keep)
        except UserChoseQuit:
            # A typed "quit" here ends the game the way it does at the plan prompt:
            # the quit ending and the behind-the-scenes review, as the help promises.
            raise _QuitGame() from None
        if not keep:
            self._use_jev = False
            self.ui.info(f"Okay - {self._local_referee_name} is the referee from now on.")
        return None

    def _judge_locally(self, record: RoundRecord) -> bool:
        """Ask the local model for a JSON verdict (one retry), else the backup rule."""
        messages = prompts.judge_messages(
            intro=self._summary.intro,
            challenge=record.challenge,
            plan=record.player_plan,
            progress=self._summary.progress,
            target=self.target,
            history=list(self._history),
        )
        backup_label = "Let the simple backup rule decide this round"
        options: dict[str, Any] = dict(
            calls=record.llm_calls,
            temperature=JUDGE_TEMPERATURE,
            json_mode=True,
            skip_label=backup_label,
            **self._thinking_call(story=False),
        )
        record.judge = "local"
        plan = record.player_plan

        def read(reply: Optional[LLMResult]) -> Optional[tuple[bool, str]]:
            return prompts.parse_judge_json(reply.text, plan=plan) if reply is not None else None

        result = self._call_llm("judge", messages, **options)
        replies: list[tuple[Optional[LLMResult], Optional[tuple[bool, str]]]] = []  # (reply, its verdict)
        parsed = read(result)
        replies.append((result, parsed))
        # A garbled reply is shown back to the model once - unless it wasn't family-friendly
        # either: then it isn't repeated, and the filter's retry below asks afresh.
        if result is not None and parsed is None and safety.check_text(result.text).ok:
            self.ui.info("The referee's answer came out garbled - asking it once more, very clearly…")
            retry = self._call_llm("judge_retry", prompts.judge_retry_messages(messages, result.text), **options)
            parsed = read(retry)
            replies.append((retry, parsed))
        parsed = self._family_friendly_verdict(record, messages, options, result, parsed, replies)
        for reply, verdict in replies:  # the kept copies, made safe for the review and transcripts
            self._screen_record(reply, (verdict[1],) if verdict is not None else ())

        if parsed is None:
            record.made_progress, record.judge_explanation = backup_verdict(
                record.player_plan, commute=record.challenge == COMMUTE_CHALLENGE
            )
            if result is not None:
                self.ui.warn(
                    "Your model's verdict still couldn't be read, so the game used its simple backup rule: "
                    "a real plan of 4+ words counts as progress, unless it gives up, just waits or claims victory."
                )
            return True
        made_progress, explanation = parsed
        record.made_progress = made_progress
        record.judge_explanation = safety.soften(explanation) or (
            "That deals with it - progress!" if made_progress else "That doesn't quite get you past it."
        )
        return False

    def _family_friendly_verdict(
        self,
        record: RoundRecord,
        messages: list[dict[str, str]],
        options: dict[str, Any],
        first: Optional[LLMResult],
        parsed: Optional[tuple[bool, str]],
        replies: list[tuple[Optional[LLMResult], Optional[tuple[bool, str]]]],
    ) -> Optional[tuple[bool, str]]:
        """The referee's explanation through the family-friendly filter.

        When the explanation (or an unreadable reply) doesn't pass, the model
        is asked once more with a firmer reminder. A readable, clean answer
        replaces the old one; otherwise the verdict stands (a yes/no can't be
        rude) with a built-in explanation - or, with no verdict at all, the
        backup rule decides as usual. Every reply is added to ``replies``.
        """
        if parsed is not None:
            flagged = safety.check_text(parsed[1])
        else:
            flagged = safety.check_text(first.text) if first is not None else safety.SafetyVerdict(ok=True)
        if flagged.ok:
            return parsed
        what = f"Round {record.number} referee"
        self._safety_note(f"{what}: the model's answer didn't pass the family-friendly filter "
                          f"({flagged.label}), so it was asked again.", record)
        self.ui.info(SAFETY_RETRY_INFO)
        again_options = {k: v for k, v in options.items() if k not in ("calls", "skip_label")}
        again = self._call_llm("judge", prompts.safety_retry_messages(messages), calls=record.llm_calls,
                               interactive=False, spinner=SPINNERS["safety_retry"], **again_options)
        fresh = prompts.parse_judge_json(again.text, plan=record.player_plan) if again is not None else None
        replies.append((again, fresh))
        if fresh is not None and safety.check_text(fresh[1]).ok:
            self._safety_note(f"{what}: the second try passed the filter.", record)
            return fresh
        if parsed is None:
            self._safety_note(f"{what}: no family-friendly verdict, so the backup rule decided.", record)
            return None
        self._safety_note(f"{what}: no family-friendly explanation, so a built-in one was used.", record)
        return parsed[0], self._built_in_explanation(parsed[0], record.number)

    def _outcome(self, record: RoundRecord, earlier_rounds: list[str]) -> str:
        """Narrate the consequences of the plan; returns the next challenge."""
        commute_round = record.challenge == COMMUTE_CHALLENGE
        messages = prompts.outcome_messages(
            intro=self._summary.intro,
            challenge=record.challenge,
            plan=record.player_plan,
            made_progress=record.made_progress,
            judge_note=self._judge_note(record),
            progress=self._summary.progress,
            target=self.target,
            history=earlier_rounds,
            commute=self._commute,
        )
        options = self._story_call()
        result = self._call_llm("outcome", messages, calls=record.llm_calls,
                                skip_label="Skip the story and carry on with a surprise challenge", **options)
        checked = dict(calls=record.llm_calls, options=options, what=f"Round {record.number} story", record=record)
        if commute_round and not record.made_progress:
            # They haven't set off yet: tell the comic story, then ask again how they'll travel.
            parts = self._family_friendly("outcome", messages, result, lambda r: (prompts.clean_story(r.text),),
                                          **checked)
            story = parts[0] if parts else ""
            if story is None:
                story = self._built_in_story(record)
            if story:
                self.ui.narrate(plain(story), title="What happens next")
            self._flush_thought_teach()
            return COMMUTE_CHALLENGE
        # After a win the obstacle must be new; after a failure the same one may stay.
        current = record.challenge if record.made_progress and not commute_round else None
        parts = self._family_friendly(
            "outcome", messages, result,
            lambda r: prompts.parse_challenge(r.text, current=current, truncated=r.truncated), **checked,
        )
        narration, challenge = parts if parts else ("", "")
        if narration is None:
            narration = self._built_in_story(record)
        if narration:
            self.ui.narrate(plain(narration), title="What happens next")
        self._flush_thought_teach()
        return challenge or self._fallback_challenge()

    def _finish_victory(self, record: RoundRecord) -> GameSummary:
        summary = self._summary
        summary.won = True
        messages = prompts.victory_messages(intro=summary.intro, history=list(self._history),
                                            final_plan=record.player_plan)
        options = self._story_call()
        try:
            result = self._call_llm("victory", messages, calls=summary.ending_calls,
                                    skip_label="Skip straight to the celebration", **options)
        except _QuitGame:
            result = None  # you won anyway!
        parts = self._family_friendly("victory", messages, result, lambda r: (prompts.clean_story(r.text),),
                                      calls=summary.ending_calls, options=options, what="The victory story")
        summary.ending = (parts[0] if parts else "") or FALLBACK_VICTORY
        self.ui.console.print()
        self.ui.console.print(self._meter_line())
        self.ui.console.print(
            Panel(Text(safe_text(summary.ending)), title="YOU GOT TO WORK!", border_style="bold green", padding=(1, 2))
        )
        self.ui.success(f"You reached your desk in {_plural(len(summary.rounds), 'round')}. Magnificent!")
        self._flush_thought_teach()
        return summary

    def _finish_quit(self, *, narrate: bool = True) -> GameSummary:
        """End early. ``narrate=False`` skips the model (e.g. it just failed)."""
        summary = self._summary
        summary.quit_early = True
        story: Optional[str] = ""
        if narrate:
            messages = prompts.quit_messages(
                intro=summary.intro, history=list(self._history), progress=summary.progress, target=self.target
            )
            options = self._story_call()
            result = self._call_llm("ending_quit", messages, calls=summary.ending_calls, interactive=False, **options)
            parts = self._family_friendly("ending_quit", messages, result, lambda r: (prompts.clean_story(r.text),),
                                          calls=summary.ending_calls, options=options, what="The goodbye story")
            story = parts[0] if parts else ""
        summary.ending = story or FALLBACK_QUIT
        self.ui.narrate(plain(summary.ending), title="See you tomorrow?")
        self.ui.info(
            f"You got {summary.progress} of {_plural(self.target, 'step')} closer to your desk. Tomorrow's another day!"
        )
        self._flush_thought_teach()
        return summary

    def _finish_out_of_rounds(self) -> GameSummary:
        summary = self._summary
        summary.ending = FALLBACK_OUT_OF_ROUNDS
        self.ui.narrate(plain(summary.ending), title="Out of time!")
        self.ui.info(f"That was the last of {_plural(self.max_rounds or 0, 'round')}. You got {summary.progress} of {self.target} steps.")
        return summary

    # -- display --------------------------------------------------------------------

    def _meter_line(self) -> Text:
        return Text.assemble(
            ("Progress to your desk  ", "bold"),
            (progress_meter(self._summary.progress, self.target), "bold green"),
        )

    def _show_challenge(self, number: int, challenge: str) -> None:
        console = self.ui.console
        console.print()
        console.print(self._meter_line())
        if challenge == COMMUTE_CHALLENGE:
            title = f"Round {number}: the journey"
        else:
            # Numbered by progress, not by distinct text: after a failed round a real
            # model writes the same obstacle "with a sillier twist", and that's still
            # the same step - so it keeps its number (setting off was step 1).
            self._challenges_seen = max(1, self._summary.progress)
            title = f"Challenge {self._challenges_seen}"
        self._last_challenge = challenge
        console.print(Panel(Text(safe_text(challenge)), title=title, border_style="yellow", padding=(1, 2)))

    def _show_help(self) -> None:
        self.ui.console.print(
            Panel(Markdown(HOW_TO_PLAY.format(target=self.target)), title="How to play", border_style="green")
        )

    def _show_verdict(self, record: RoundRecord, backup_used: bool) -> None:
        if record.jev is not None:
            self.ui.console.print(_jev_verdict_panel(record.jev))
            # One new answer type per round (Noul, then Choice, then Score), so the
            # first verdict isn't followed by three long lessons in a row.
            for key, title, body in _JEV_LESSONS:
                if key not in self._taught:
                    self._teach_once(key, title, body)
                    # A verdict plus a lesson is a lot of text: let it be read before it scrolls away.
                    self.ui.pause()
                    break
            return
        made = record.made_progress
        headline = Text(
            "PROGRESS! You're a step closer to work." if made else "NOT YET! That didn't get you past it.",
            style="bold green" if made else "bold yellow",
        )
        who = "the pretend model" if self._pretend else "your local model"
        title = "Referee's verdict (backup rule)" if backup_used else f"Referee's verdict ({who})"
        self.ui.console.print(
            Panel(Group(headline, Text(safe_text(record.judge_explanation))), title=title,
                  border_style="green" if made else "yellow", padding=(0, 2))
        )
        if "local_judge" not in self._taught:
            tail = _LOCAL_TAIL_NO_JEV if self.jev is None else _LOCAL_TAIL_FALLBACK
            if self._pretend:
                self._teach_once("local_judge", "How the pretend model referees",
                                 LOCAL_JUDGE_EXPLAINER_MOCK.format(tail=tail))
            else:
                self._teach_once("local_judge", "How your local model referees",
                                 LOCAL_JUDGE_EXPLAINER.format(tail=tail))
            self.ui.pause()

    # -- teaching -----------------------------------------------------------------

    def _teach_once(self, key: str, title: str, body: str) -> None:
        if key in self._taught:
            return
        self._taught.add(key)
        self.ui.teach(title, body)

    def _flush_thought_teach(self) -> None:
        """Explain chain-of-thought the first time the model shows some (after the story is on screen)."""
        if self._pending_thought_words is None:
            return
        words, self._pending_thought_words = self._pending_thought_words, None
        template = COT_EXPLAINER_MOCK if self._pending_thought_mock else COT_EXPLAINER
        self._teach_once("reasoning", "Chain-of-thought", template.format(words=words))

    # -- bookkeeping ----------------------------------------------------------------

    def _fallback_challenge(self) -> str:
        """A built-in challenge as absurd as the player's progress calls for (never repeated if avoidable)."""
        tier = prompts.absurdity_index(self._summary.progress, self.target)
        options = FALLBACK_CHALLENGE_TIERS[min(tier, len(FALLBACK_CHALLENGE_TIERS) - 1)]
        # Only obstacles that fit how the player is travelling (a walker has no bicycle to join a band).
        fitting = [c for c in options if _fits_commute(c, self._commute)] or list(options)
        fresh = [c for c in fitting if c not in self._fallback_used] or fitting
        challenge = fresh[0]
        self._fallback_used.add(challenge)
        return challenge

    @staticmethod
    def _history_entry(record: RoundRecord) -> str:
        """One line per finished round, for later prompts and Jev.

        The player's own words are quoted *inside* the <player_plan> delimiters
        (defanged like the current plan), so nothing typed in an earlier round
        can pose as instructions in a later prompt.
        """
        result = "it worked" if record.made_progress else "it didn't work"
        facing = "choosing how to get to work" if record.challenge == COMMUTE_CHALLENGE else f'facing "{_clip(record.challenge, 60)}"'
        return f"Round {record.number}: {facing}, the player tried {prompts.quote_plan(record.player_plan, 90)} - {result}."

    @staticmethod
    def _judge_note(record: RoundRecord) -> str:
        """What the narrator is told about the verdict, beyond yes/no.

        Jev's Choice label is only mentioned when it agrees with the Noul (which
        decides): a small model shown "made progress: true" next to "stalled"
        may narrate the wrong outcome.
        """
        verdict = record.jev
        if verdict is None:
            return record.judge_explanation
        agrees, direction = _OUTCOME_FLAVOUR.get(verdict.outcome, (None, ""))
        creativity = f"rated its creativity {verdict.creativity:.1f} out of 4"
        if agrees == record.made_progress:
            note = f'Jev filed the outcome under "{_family_label(verdict.outcome)}" and {creativity}.'
            return note + (" " + direction if direction else "")
        return f"Jev {creativity}."


# ---------------------------------------------------------------------------
# The Jev verdict panel
# ---------------------------------------------------------------------------


def _ordered_probabilities(probabilities: dict[str, float]) -> list[tuple[str, float]]:
    """The game's labels in their natural order first, then any others by probability."""
    known = [(label, probabilities[label]) for label in jevlib.ROUND_OUTCOME_LABELS if label in probabilities]
    others = sorted(
        ((k, v) for k, v in probabilities.items() if k not in jevlib.ROUND_OUTCOME_LABELS),
        key=lambda item: -item[1],
    )
    return known + others


def _score_top(verdict: JevVerdict) -> int:
    levels = [int(k) for k in verdict.creativity_legend if str(k).isdecimal() and len(str(k)) <= 6]
    return max(1, max(levels) if levels else len(jevlib.CREATIVITY_LEVELS) - 1)


def _nearest_level(verdict: JevVerdict) -> str:
    nearest = verdict.creativity_legend.get(str(int(round(verdict.creativity))))
    return nearest.split(":")[0].strip() if isinstance(nearest, str) else ""


def _jev_verdict_panel(verdict: JevVerdict) -> Panel:
    """Jev's three typed answers, drawn as bars. All Jev text goes in as plain ``Text``."""
    made = verdict.made_progress
    tone = "green" if made else "yellow"
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold cyan", no_wrap=True)  # question type
    grid.add_column(style="bold", no_wrap=True)  # question name (as sent to Jev)
    grid.add_column()  # the answer

    p_yes = verdict.progress_probability
    grid.add_row("Noul", "made_progress", Text.assemble((f"[{probability_bar(p_yes, 20)}]", tone), f" {_pct(p_yes)} chance of yes"))
    grid.add_row("", "", Text("=> that counts as progress!" if made else "=> not enough to count this time", style=f"bold {tone}"))
    grid.add_row("", "", "")

    grid.add_row("Choice", "outcome", Text.assemble((_family_label(verdict.outcome), "bold"), f"  ({_pct(verdict.outcome_confidence)} confident)"))
    for label, probability in _ordered_probabilities(verdict.outcome_probabilities):
        marker = ">" if label == verdict.outcome else " "
        grid.add_row("", "", Text.assemble(f"{marker} {_family_label(label):<9} ", (f"[{probability_bar(probability, 12)}]", "cyan"), f" {_pct(probability):>4}"))
    grid.add_row("", "", "")

    top = _score_top(verdict)
    grid.add_row(
        "Score",
        "creativity",
        Text.assemble(
            (f"{verdict.creativity:.1f} out of {top}", "bold"),
            "  ",
            (f"[{probability_bar(verdict.creativity / top, 12)}]", "magenta"),
            f"  ({_pct(verdict.creativity_confidence)} confident)",
        ),
    )
    nearest = _nearest_level(verdict)
    if nearest:
        grid.add_row("", "", Text(f'nearest level: "{_family_label(nearest)}"', style="dim"))

    parts: list[Any] = [grid]
    agrees = _OUTCOME_FLAVOUR.get(verdict.outcome, (None, ""))[0]
    if agrees is not None and agrees != made:
        # The two answers point different ways: worth explaining, not a bug. Only a
        # Noul near 50% is a close call; otherwise the questions simply disagree.
        outcome = _family_label(verdict.outcome)
        if CLOSE_CALL_LOW <= p_yes <= CLOSE_CALL_HIGH:
            note = (f"Only the Noul decides progress. Here the Choice leans the other way (\"{outcome}\"), "
                    f"which is a sign of a close call ({_pct(p_yes)} chance of yes).")
        else:
            note = (f"Only the Noul decides progress ({_pct(p_yes)} chance of yes - a clear answer). The Choice "
                    f"(\"{outcome}\") is a separate question, answered on its own, so now and then the two "
                    "disagree - that's why the game decides with one number and only shows the others.")
        parts += [Text(), Text(note, style="italic")]
    return Panel(Group(*parts), title="Jev's verdict", border_style=tone, padding=(1, 2))
