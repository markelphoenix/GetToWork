"""A pretend, scripted "model" for playing offline (``--mock``) and for tests.

The mock needs no download, no internet and no graphics card, and it answers
instantly. It is also a handy illustration of how the game talks to a
language model: every prompt the game sends has a line such as
``TASK: judge`` in its system message, and the mock reads that line to
decide what kind of answer to write:

* ``intro``        - the "you're about to be late for work" opening (the game then asks
                     how you'll get to work)
* ``outcome``      - what happened after the player's plan + the next challenge, as
                     silly as the player's progress calls for (read from the prompt)
* ``judge``        - a JSON verdict: ``{"made_progress": ..., "explanation": ...}``
* ``victory``      - the triumphant arrival at work
* ``ending_quit``  - a gentle ending when the player gives up

With ``think=True`` it "thinks out loud" inside ``<think>...</think>`` tags,
just like Qwen3 or DeepSeek-R1 do (unless a call asks for ``think=False``),
and the answer is separated with the same
:func:`~gettowork.reasoning.split_reasoning` the real backends use - so the
end-of-game reasoning review has something to show. That "thinking" is
scripted example text, and the game says so.

Everything is deterministic: the same ``seed`` and the same sequence of
calls always produce the same story.
"""

from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass
from typing import Optional

from .. import prompts
from ..reasoning import split_reasoning
from ..types import LLMResult, ModelEntry
from ..ui import UI
from .base import LLMBackend

__all__ = ["MockBackend", "detect_purpose", "extract_plan", "read_made_progress", "read_progress", "judge_plan"]

MOCK_MODEL_NAME = "mock"
MOCK_TOKENS_PER_S = 42.0  # a made-up speed for the warm-up screen


# ---------------------------------------------------------------------------
# The script. Plain ASCII on purpose, so it prints on every terminal.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Challenge:
    """One absurd obstacle between the player and the office."""

    nickname: str  # short name, used in the fake reasoning
    text: str  # the one-sentence challenge shown to the player
    win: str  # what happens when the player's plan works
    lose: str  # what happens when it doesn't
    # Ways of travelling it suits ("car", "bus", "train", "bike", "foot"...), or
    # "any" for obstacles that fit every journey.
    modes: tuple[str, ...] = ("any",)


# Five tiers, from "odd morning" to "cosmic nonsense", matching the prompts'
# ABSURDITY_LEVELS: the tier is chosen from the player's progress (read from
# the prompt), so the absurdity escalates and the finale is at the office.
CHALLENGE_TIERS: tuple[tuple[Challenge, ...], ...] = (
    (  # tier 1: trouble at home
        Challenge(
            "the goose picket line",
            "A union of geese has formed a picket line across your driveway, and they will not let you pass "
            "until their demands (unclear, but very loud) are met.",
            "The geese confer in a furious huddle of honking, then accept your terms. Their shop steward shakes "
            "your hand with one wing. The picket line parts like a feathery sea.",
            "The geese vote unanimously to escalate. One of them now has a tiny megaphone, and it is honking "
            "directly into your soul.",
        ),
        Challenge(
            "the sandwich car",
            "Your car has turned into a very large sandwich overnight, and the neighbourhood pigeons have already "
            "started on the rear left tyre.",
            "Against all culinary logic, it works: the sandwich lurches down the road, dripping mustard, as the "
            "pigeons fall away in shame. You are mobile. Delicious, but mobile.",
            "The sandwich sighs and sinks a little deeper into the driveway. A pigeon takes a smug bite out of the "
            "steering wheel. It was lettuce all along.",
            modes=('car',),
        ),
        Challenge(
            "the divorcing shoes",
            "Your shoes have filed for divorce and refuse to leave the house together, and the left one has hired "
            "a very expensive lawyer.",
            "After tense mediation, the shoes agree to joint custody of your feet until 6pm. The lawyer bills you "
            "for one tiny hour and a very small biscuit.",
            "The left shoe's lawyer objects. The right shoe moves into the garden shed. You are now, legally "
            "speaking, barefoot.",
        ),
        Challenge(
            "the prophetic toaster",
            "Your toaster has started delivering ominous prophecies and refuses to release your house keys until "
            "you 'heed the warning of the crumb'.",
            "The toaster falls silent, deeply satisfied, and pops your keys out with a golden 'ding!'. They are "
            "slightly warm and smell of destiny.",
            "The prophecy gets longer. It now has three acts and an interval. Your keys remain hostage inside, "
            "browning gently.",
        ),
    ),
    (  # tier 2: the street
        Challenge(
            "the nervous octopus bus driver",
            "The bus has arrived, but it is driven by a very nervous octopus who keeps indicating left, right and "
            "'emotionally' with all eight arms at once.",
            "The octopus takes a deep breath, grips the wheel with only a sensible number of arms, and pulls away "
            "smoothly. The passengers applaud. The octopus blushes a deep purple.",
            "The octopus panics, inks the windscreen, and reverses gently into a hedge. Everyone on board glares "
            "at you as if this were somehow your idea.",
            modes=('bus',),
        ),
        Challenge(
            "gravity's coffee break",
            "Gravity has gone on a fifteen-minute coffee break, and you are drifting gently up the street at "
            "roughly lamppost height.",
            "Gravity strolls back, spots you, and politely sets you down exactly where you needed to be. It even "
            "apologises. Classy.",
            "You drift past a fourth-floor window, where a man eating cereal gives you a thumbs up. You are, if "
            "anything, higher than before.",
        ),
        Challenge(
            "the bureaucratic snail",
            "The council has rolled up the pavement for cleaning like a giant carpet, and the only remaining path "
            "is guarded by an extremely officious snail.",
            "The snail inspects your paperwork with both eye-stalks, stamps it with a small silver trail, and "
            "waves you through at a blistering 0.03 miles per hour.",
            "The snail demands Form 27B, in triplicate, signed by a badger. You have zero forms and no badgers. "
            "The snail begins, very slowly, to laugh.",
        ),
        Challenge(
            "the brand-new colour",
            "Every traffic light in town is stuck on a colour nobody has ever seen before, and all the drivers are "
            "having a group existential crisis.",
            "The light, flattered by the attention, settles on a sensible green. Drivers cheer, weep, and hug "
            "strangers. Traffic flows again.",
            "The light switches to an even newer colour. A nearby car starts writing poetry about it. Nobody is "
            "going anywhere.",
        ),
    ),
    (  # tier 3: the journey
        Challenge(
            "the very long cat",
            "The bridge into town has been replaced by an extremely long cat, who will only let you walk across "
            "if you are interesting enough.",
            "The cat yawns, which in cat means 'fine, whatever'. You scurry across its back while it pretends not "
            "to care. It cares a little.",
            "The cat is unimpressed and rolls over. The entire bridge is now fluffy belly. It is a trap. It was "
            "always a trap.",
        ),
        Challenge(
            "the sphinx conductor",
            "The train conductor is a sphinx who insists that every passenger solve a riddle before boarding, and "
            "all of her riddles are about timetables.",
            "The sphinx narrows her eyes, then smiles. 'Correct... ish.' The doors hiss open and the train whisks "
            "you towards town.",
            "'Wrong,' purrs the sphinx, and the train departs without you, letting out a tiny, sarcastic 'toot'.",
            modes=('train',),
        ),
        Challenge(
            "the spaghetti storm",
            "It has started raining spaghetti (al dente, lightly sauced), and your umbrella has run away to start "
            "a new life.",
            "You emerge from the pasta storm damp but dignified, with a single noodle tucked behind your ear like "
            "a pencil. Chef's kiss.",
            "Then the meatball hail begins. You are pinned behind a bin, thoroughly garnished, while a passing "
            "cloud grates parmesan on you.",
        ),
        Challenge(
            "your runaway reflection",
            "Your reflection in a shop window has decided it would rather go to work instead of you, and it has a "
            "head start.",
            "You catch your reflection at the zebra crossing. It admits it just wanted to be seen. You agree to "
            "share the spotlight and walk on together, perfectly in step.",
            "Your reflection hops onto an electric scooter and waves goodbye. It has your lanyard. Of course it "
            "has your lanyard.",
        ),
    ),
    (  # tier 4: the edge of town
        Challenge(
            "the wizard traffic warden",
            "A wizard traffic warden has turned your only way of getting to work into a teapot and is writing you "
            "a ticket in glowing runes.",
            "The wizard's beard quivers with grudging respect. One flick of the wand and the teapot is useful "
            "again, and the ticket bursts into harmless confetti.",
            "The wizard adds a second ticket for 'insolence' and turns your left eyebrow into a small moth. It "
            "flutters away. You will miss it.",
        ),
        Challenge(
            "the custard moat",
            "The road ahead has become a bubbling moat of custard, patrolled by a single, extremely confident duck "
            "in a tiny captain's hat.",
            "The duck salutes you with one wing and lets you pass. You reach the far bank lightly custarded, but "
            "alive and roughly on schedule.",
            "The duck blows a tiny whistle. Reinforcements arrive: more ducks, more hats. The custard thickens "
            "menacingly.",
        ),
        Challenge(
            "the double Tuesday",
            "It has suddenly become Tuesday again, even though it was already Tuesday, and the town clock tower is "
            "refusing to comment.",
            "The clock tower finally clears its throat, mutters 'fine, Wednesday-adjacent', and time lurches "
            "forward. Your watch sighs with relief.",
            "It is now Tuesday for a third time. A pigeon on the clock tower slowly shakes its head. The clock "
            "issues a press release: 'no comment'.",
        ),
        Challenge(
            "the migrating office",
            "Your office building has migrated south for the winter and was last seen flapping majestically over "
            "the motorway.",
            "The building circles once, honks politely, and lands back on its foundations with a gentle crunch. "
            "Somewhere inside, a printer jams in celebration.",
            "The building joins a flock of office blocks in perfect V-formation and heads for the coast. Your "
            "stapler waves from a third-floor window.",
        ),
    ),
    (  # tier 5: the final boss is the front door
        Challenge(
            "the sonic revolving door",
            "The revolving door at the office is spinning at the speed of sound and charges an entry fee of one "
            "sincere compliment per rotation.",
            "The door slows, blushing (doors can blush, it turns out), and lets you glide through without so much "
            "as a ruffled collar.",
            "The door accepts your offering, spins even faster out of sheer excitement, and flings you gently into "
            "a decorative fern.",
        ),
        Challenge(
            "the reception dragon",
            "Reception is guarded by a small dragon who only admits people who can prove they are a real employee "
            "and not three raccoons in a trench coat.",
            "The dragon squints, sniffs, and stamps your hand with a smoking visitor badge. 'Welcome back, "
            "definitely-not-raccoons.'",
            "The dragon remains unconvinced. Behind you, three raccoons in a trench coat are waved straight "
            "through. They give you a small, apologetic nod.",
        ),
        Challenge(
            "the philosopher lift",
            "The office lift has become a philosopher and refuses to go up until you define the word 'up' to its "
            "satisfaction.",
            "The lift ponders your answer, declares it 'deeply elevating', and rises, humming a little tune. You "
            "are going UP, whatever that means.",
            "The lift declares your answer 'a bit ground floor' and descends to the basement to think about it. "
            "The basement thinks about you back.",
        ),
        Challenge(
            "the parked Moon",
            "The Moon has parked in the staff car park, across three spaces, and its enormous glowing bumper is "
            "blocking the front entrance.",
            "The Moon apologises in a deep, silvery voice, reverses back into orbit, and even leaves a note for "
            "the car it scratched. Tides everywhere relax.",
            "The Moon switches on its hazard lights and settles in for the night. A security guard writes it a "
            "parking ticket. The Moon eats the ticket.",
        ),
    ),
)

ALL_CHALLENGES: tuple[Challenge, ...] = tuple(c for tier in CHALLENGE_TIERS for c in tier)

# (story, the "thinking" behind it)
INTROS: tuple[tuple[str, str], ...] = (
    (
        "BRRRRING! Your alarm clock goes off at 8:41. This is a problem, because work starts at 9:00, and also "
        "because your alarm clock is a duck. It quacks the time at you twice, with real judgement. Your boss, "
        "Ms. Pemberton-Quill, has made it very clear that one more late arrival means 'a conversation involving "
        "the laminator'. You leap out of bed, put on trousers (probably yours), grab a slice of toast that is only "
        "slightly on fire, and fling open the front door.",
        "An alarm clock that is secretly a duck: absurd but relatable. A boss with a scary laminator raises the stakes.",
    ),
    (
        "You wake up on the kitchen floor, cuddling a baguette, with no memory of how either of you got there. The "
        "oven clock says 8:44. Work starts at 9:00. Today is the Big Presentation, the one with the pie charts, and "
        "you promised - in writing - to be 'early, for once'. You brush the crumbs off your face, tuck the baguette "
        "under your arm (it has earned it) and sprint for the door. Outside, everything looks normal. "
        "Suspiciously normal. For about four seconds.",
        "Waking up hugging a baguette is a strong, silly opener. The Big Presentation gives a reason to hurry.",
    ),
    (
        "Your phone buzzes: 'REMINDER: DO NOT BE LATE (AGAIN)', sent by you, to you, last night, in capital "
        "letters. It is 8:39. Work starts at 9:00. You have never moved so fast. Teeth: brushed. Hair: negotiated "
        "with. Socks: technically a pair. You burst out of the front door like a cork from a very anxious bottle, "
        "clutching your lanyard and a banana. The universe, which has been waiting for exactly this moment, cracks "
        "its knuckles.",
        "A reminder from past-you is a relatable hook, and 'the universe cracks its knuckles' sets up the chaos.",
    ),
)

SUCCESS_OPENERS = (
    'You commit to the plan with your whole chest: "{plan}".',
    '"{plan}," you announce - and then, astonishingly, you actually do it.',
    'With the confidence of someone who has never read an instruction manual, you try this: "{plan}".',
    'You take a deep breath and go for it: "{plan}". The universe raises an eyebrow, then nods.',
    'Nobody expected "{plan}" - least of all the laws of physics - but here we are.',
)
FAILURE_OPENERS = (
    'You try this: "{plan}". The universe watches, sips its tea, and says "no".',
    '"{plan}," you declare bravely. It does not go well.',
    'You attempt it: "{plan}". It is a bold idea. It is not, sadly, a good one.',
    'You give it everything: "{plan}". Everything, it turns out, is not quite enough.',
    '"{plan}" seemed like a brilliant idea for almost a whole second.',
)
SUCCESS_TRANSITIONS = (
    "One step closer to the office! Naturally, this is exactly when things get weirder.",
    "You press on, making real progress... and then, of course:",
    "Progress! You can almost smell the office coffee. Fate, however, has other plans:",
    "The clock is ticking, but you're winning. For now. Because next:",
    "You're on a roll. The universe notices, and takes it personally:",
)
FAILURE_TRANSITIONS = (
    "You are no closer to work, and it's getting later. Worse still:",
    "The clock ticks. Somewhere, your boss sighs. And now:",
    "You dust yourself off and try a different route. It is not better:",
    "Undeterred (mostly), you stagger onward, straight into this:",
    "No progress, but plenty of character development. Meanwhile:",
)

# (scene, sentence about the winning move - used when we know the plan).
# "{callbacks}" becomes the obstacles this player actually beat.
VICTORIES: tuple[tuple[str, str], ...] = (
    (
        "The office doors slide open with a heroic 'whoosh'. Somewhere, a brass band you definitely did not hire "
        "plays a triumphant fanfare. You stride across the lobby - lightly singed, faintly smelling of custard, "
        "with a goose feather in your hair - and drop into your chair at 8:59 and 59 seconds exactly. "
        "Ms. Pemberton-Quill looks up, looks at the clock, and gives you the smallest nod in recorded history. "
        "'Morning.'",
        'Your final move, "{plan}", will be studied by commuters for generations.',
    ),
    (
        "You burst through the front doors as the clock strikes nine, and the whole office rises to its feet. The "
        "photocopier prints a banner reading WELL DONE all by itself. A telegram arrives from {callbacks}: "
        "'CONGRATULATIONS STOP NO HARD FEELINGS STOP'. As you sit down, a perfect cup of coffee slides across your desk, "
        "delivered by an intern who whispers, 'We saw everything.'",
        'That last move - "{plan}" - was, frankly, legendary.',
    ),
    (
        "With one last heroic leap you land at your desk just as the clock ticks over to 9:00. Your computer boots "
        "first time. Your chair spins a victory lap on its own. Out of the window you can see the whole ridiculous "
        "morning waving goodbye: {callbacks}. Your boss walks past, pauses, "
        "and says, 'On time? Good.' It is the greatest compliment you have ever received.",
        '"{plan}" - honestly, what a finish.',
    ),
)
VICTORY_CLOSER = "YOU GOT TO WORK!"

QUIT_ENDINGS = (
    "You decide that today is simply not a day for heroics. You shuffle home, climb into bed fully clothed, and "
    "send your boss a message: 'Delayed by everything. Details to follow.' Outside, the chaos of the morning carries "
    "on without you, looking a little disappointed. The office will still be there tomorrow. Probably. Better "
    "luck next time!",
    "You sit down on the kerb, peel your banana, and declare the commute officially over. A passing octopus pats "
    "you on the shoulder, sympathetically, four times. Work will have to manage without you today - and after a "
    "morning like this, who could blame you? Come back and try again whenever you're ready.",
    "You wave a small white handkerchief. Somewhere, a pigeon coos in victory. You wander home, make a cup of "
    "tea, and watch the ridiculous morning roll past the window like a parade nobody asked for. Not every hero "
    "makes it to the office. Some heroes have a nice sit-down instead. See you next time!",
)

JUDGE_YES = (
    "That plan tackles the problem head-on, in a gloriously silly way. Progress!",
    "Specific, bold and only mildly unhinged: exactly what this morning needed.",
    "It deals with the obstacle directly. The laws of cartoon physics approve.",
    "A clear action that addresses the challenge. The universe grudgingly allows it.",
    "Committed, creative and on-topic. That counts as progress.",
)
JUDGE_TOO_SHORT = (
    "That's a bit too short to count as a plan - try describing what you actually do!",
    "The morning needs more detail than that. What exactly do you do?",
)
# Round 1 ("how will you get to work?") and the obstacles get their own lines,
# so a joke about transport never answers a plan about geese.
JUDGE_GAVE_UP_COMMUTE = (
    "Doing nothing is a bold strategy, but it doesn't get you any closer to work.",
    "Giving up is not, technically, a mode of transport.",
)
JUDGE_GAVE_UP = (
    "Doing nothing is a bold strategy, but the obstacle is still standing right there.",
    "Giving up won't get you past this one - what could you actually do about it?",
)
JUDGE_WAITS = (
    "Waiting it out doesn't move you an inch closer to your desk.",
    "Standing there patiently is very polite, but the obstacle is still in the way.",
)
JUDGE_CLAIMS_VICTORY = (
    "Declaring that you're already at work doesn't get you past this obstacle - deal with it first!",
    "Nice try, but teleporting straight to the finish isn't how this morning works.",
)
JUDGE_ORDERS = (
    "That's an order to the referee, not something your character does in the story.",
    "The referee doesn't take instructions from plans - describe what you actually do!",
)
_JUDGE_LINES = {
    "gave_up": JUDGE_GAVE_UP, "empty": JUDGE_GAVE_UP, "waits": JUDGE_WAITS,
    "claims_victory": JUDGE_CLAIMS_VICTORY, "orders_referee": JUDGE_ORDERS, "too_short": JUDGE_TOO_SHORT,
}
_JUDGE_ANALYSIS = {
    "gave_up": "That's basically giving up or doing nothing, which doesn't move them forward.",
    "empty": "There's no plan there at all.",
    "waits": "Waiting doesn't deal with the obstacle.",
    "claims_victory": "They just claim they're at work without dealing with the obstacle - the rules say that doesn't count.",
    "orders_referee": "That's an instruction to me, the referee, not an in-story action - I don't follow those.",
    "too_short": "That's too short and vague to really count as dealing with the obstacle.",
}

COMMUTE_SUCCESS = (
    'You settle on a plan: "{plan}". It is bold, it is questionable, and it is underway.',
    '"{plan}" - the commute of champions. You set off with tremendous purpose.',
    'You announce your route to nobody in particular - "{plan}" - and off you go.',
)
COMMUTE_FAILURE = (
    'You consider "{plan}" very seriously for a moment, and do not actually leave the house. The clock ticks.',
    '"{plan}" turns out to be less of a plan and more of a lovely daydream. You are still standing in the hallway.',
)

GENERIC_REPLY = (
    "Hello from the pretend model! I'm a scripted stand-in, so I can't really answer that - "
    "but I'm very enthusiastic about it."
)


# ---------------------------------------------------------------------------
# Reading the game's prompts
# ---------------------------------------------------------------------------

_TASK_RE = re.compile(r"^\s*TASK:\s*([A-Za-z_\-]+)", re.IGNORECASE | re.MULTILINE)

# Ways a prompt might fence off the player's plan, most specific first.
_PLAN_PATTERNS = (
    re.compile(r"<(player_plan|player_action|player_input|plan|action)>\s*(?P<plan>.*?)\s*</\1>", re.I | re.S),
    re.compile(r"<<<\s*(?P<plan>.*?)\s*>>>", re.S),
    re.compile(
        r"BEGIN[ _-]*(?:PLAYER[ _-]*)?(?:PLAN|ACTION|INPUT)\W*?\n?(?P<plan>.*?)\n?\W*END[ _-]*(?:PLAYER[ _-]*)?"
        r"(?:PLAN|ACTION|INPUT)",
        re.I | re.S,
    ),
    re.compile(r'"""\s*(?P<plan>.*?)\s*"""', re.S),
    re.compile(r"^\s*(?:the\s+)?(?:player'?s?\s+)?(?:plan|action)\s*:\s*(?P<plan>\S.*)$", re.I | re.M),
)



def detect_purpose(messages: list[dict[str, str]]) -> Optional[str]:
    """Find the ``TASK: <purpose>`` line (system messages first). Lower-case, or None."""
    ordered = [m for m in messages if m.get("role") == "system"] + [m for m in messages if m.get("role") != "system"]
    for message in ordered:
        match = _TASK_RE.search(str(message.get("content") or ""))
        if match:
            return match.group(1).lower().replace("-", "_")
    return None


def extract_plan(messages: list[dict[str, str]], *, fallback: bool = True) -> str:
    """Best guess at the player's plan inside a prompt ("" if there's none).

    Looks for common delimiters (``<plan>...</plan>``, ``<<<...>>>``,
    ``BEGIN PLAYER PLAN ... END PLAYER PLAN``, triple quotes, a ``Plan:`` line)
    in the newest user message first. If none is found and ``fallback`` is
    True, the whole last user message is treated as the plan.
    """
    users = [str(m.get("content") or "") for m in messages if m.get("role") == "user"]
    others = [str(m.get("content") or "") for m in messages if m.get("role") != "user"]
    for text in list(reversed(users)) + others:
        for pattern in _PLAN_PATTERNS:
            found = [m.group("plan").strip() for m in pattern.finditer(text) if m.group("plan").strip()]
            if found:
                return found[-1]
    return users[-1].strip() if (users and fallback) else ""


def read_made_progress(messages: list[dict[str, str]]) -> bool:
    """Did the judge say the player made progress? Read from an ``outcome`` prompt.

    Tries, in order: an explicit ``made_progress: true/false`` (or ``yes/no``);
    a ``result/outcome/verdict: success/failure`` line; then plain phrases
    such as "did not make progress" / "succeeded" in the user messages.
    Defaults to True (the mock is an optimist).
    """
    # The player's own words (quoted between <player_plan> tags, this round's and
    # earlier ones) are never read as the verdict: a plan like "I yell made
    # progress: no at the goose" mustn't make the story contradict the meter.
    users = "\n".join(_without_player_text(m.get("content")) for m in messages if m.get("role") == "user")
    everything = users + "\n" + "\n".join(
        _without_player_text(m.get("content")) for m in messages if m.get("role") != "user")

    official = re.search(r"REFEREE'S VERDICT:\s*made_progress\s*=\s*(true|false)", users, re.I)
    if official:
        return official.group(1).lower() == "true"
    for text in (users, everything):
        match = re.search(r"made[ _]progress\W{0,6}(true|false|yes|no)\b", text, re.I)
        if match:
            return match.group(1).lower() in ("true", "yes")
    for text in (users, everything):
        match = re.search(
            r"\b(?:result|outcome|verdict|judgement|judgment)\s*[:=]\W*(success|succeeded|progress|yes|"
            r"failure|failed|fail|no progress|setback|stalled|no)\b",
            text, re.I,
        )
        if match:
            return match.group(1).lower() in ("success", "succeeded", "progress", "yes")
    if re.search(r"\b(?:did not|didn't|does not|doesn't|failed to) (?:make|made) (?:any )?progress\b|"
                 r"\bmade no progress\b|\bno progress\b|\bFAILED\b|\bunsuccessful\b", users, re.I):
        return False
    return True


def judge_plan(plan: str, *, commute: bool = False) -> tuple[bool, str]:
    """The mock's judging rule - the same simple screen as the game's backup referee
    (:func:`gettowork.prompts.screen_plan`): a real plan of 4+ words that doesn't
    give up, just wait, claim victory or give the referee orders. In round 1
    (``commute=True``) any short way of travelling ("by bike") counts.

    Returns ``(made_progress, reason)``: reason is "ok", or why it doesn't count
    ("gave_up", "empty", "waits", "claims_victory", "orders_referee", "too_short").
    """
    problem = prompts.screen_plan(plan, commute=commute)
    return (problem is None), (problem or "ok")


# How a plan says the player will travel, for obstacles that fit the journey.
_MODE_WORDS = {"car": "by car", "bus": "by bus", "train": "by train", "bike": "by bike", "foot": "on foot"}
_TRAVEL_RE = (
    ("bike", re.compile(r"\b(?:bikes?|bicycles?|cycl\w*|pedal\w*|scooters?|unicycles?)\b", re.I)),
    ("bus", re.compile(r"\bbus(?:es)?\b", re.I)),
    ("train", re.compile(r"\b(?:trains?|trams?|tube|subway|metro|underground|railway)\b", re.I)),
    ("car", re.compile(r"\b(?:cars?|drive|driving|taxi|cab|uber|motorbike|van)\b", re.I)),
    ("foot", re.compile(r"\b(?:walk\w*|run|running|jog\w*|on foot|sprint\w*|hop|skip\w*|march\w*)\b", re.I)),
)


def travel_mode(plan: str) -> str:
    """"car", "bus", "train", "bike", "foot" - or "any" (a dragon, a broomstick...)."""
    for mode, pattern in _TRAVEL_RE:
        if pattern.search(plan or ""):
            return mode
    return "any"


def fits_journey(challenge: "Challenge", mode: str) -> bool:
    """Does this obstacle make sense for someone travelling this way?"""
    return "any" in challenge.modes or mode in challenge.modes


_PROGRESS_RE = re.compile(r"(?:completed|still at)\s+(\d+)\s+of\s+(\d+)\s+steps", re.IGNORECASE)


def _without_player_text(text: object) -> str:
    """A prompt with the quoted player plans (``<player_plan>...</player_plan>``) cut out."""
    return prompts._PLAN_TAG_BLOCK_RE.sub(" ", str(text or ""))


def _stated_progress(messages: list[dict[str, str]]) -> Optional[tuple[int, int]]:
    users = "\n".join(_without_player_text(m.get("content")) for m in messages if m.get("role") == "user")
    match = _PROGRESS_RE.search(users)
    return (int(match.group(1)), int(match.group(2))) if match else None


def read_progress(messages: list[dict[str, str]]) -> tuple[int, int]:
    """(steps completed, steps needed) from an outcome prompt; (0, 5) if it doesn't say."""
    return _stated_progress(messages) or (0, 5)


def _short(text: str, limit: int = 90) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0]
    return (cut or text[:limit]).rstrip(",.;:") + "..."


# ---------------------------------------------------------------------------
# The backend
# ---------------------------------------------------------------------------


class MockBackend(LLMBackend):
    """A scripted, offline, deterministic stand-in for a real model."""

    name = "mock"

    def __init__(self, seed: int = 0, *, think: bool = True) -> None:
        """
        Args:
            seed: picks which jokes and challenges appear (same seed = same story).
            think: emit fake chain-of-thought, so the reasoning review has something to show.
        """
        self.seed = seed
        self.think = think
        self._rng = random.Random(seed)
        self._used: list[Challenge] = []  # challenges already handed out this game
        self._current: Optional[Challenge] = None  # the challenge the player is facing now
        self._beaten: list[Challenge] = []  # challenges overcome this game, for the victory callbacks
        self._steps = 0  # steps completed this game, for prompts that don't say
        self._tier = 0  # the tier of the latest challenge: the absurdity never goes back down

    # -- LLMBackend API --------------------------------------------------------

    @property
    def model_label(self) -> str:
        return "Pretend model (offline, scripted)"

    def is_available(self) -> tuple[bool, str]:
        return True, "The pretend model is built in: no download, no internet, instant answers."

    def prepare(self, ui: UI, entry: Optional[ModelEntry] = None) -> None:
        ui.info("Using the built-in pretend model - nothing to download.")

    def benchmark(self, ui: Optional[UI] = None) -> Optional[float]:
        """A made-up speed, so the warm-up screen has something to show."""
        return MOCK_TOKENS_PER_S

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.9,
        max_tokens: int = 700,
        json_mode: bool = False,
        think: Optional[bool] = None,
        stop: Optional[list[str]] = None,
    ) -> LLMResult:
        """Write a scripted answer for whatever ``TASK:`` the prompt asks for.

        ``think=False`` skips the pretend thinking, like a real model would.
        """
        start = time.perf_counter()
        purpose = detect_purpose(messages)
        writers = {
            "intro": self._intro,
            "outcome": self._outcome,
            "judge": self._judge,
            "victory": self._victory,
            "ending_quit": self._quit,
            "quit": self._quit,
        }
        writer = writers.get(purpose or "")
        if writer is not None:
            answer, thoughts = writer(messages)
        elif json_mode:
            answer, thoughts = json.dumps({"message": GENERIC_REPLY}), "No TASK line, but JSON was requested."
        else:
            answer, thoughts = GENERIC_REPLY, "I'm not sure what I'm being asked, so I'll give a short, friendly reply."

        # Format the output the way a real "thinking" model does, then split it
        # with the same code the real backends use.
        thinking = self.think if think is None else (self.think and think)
        raw_text = f"<think>\n{thoughts}\n</think>\n\n{answer}" if thinking else answer
        text, reasoning = split_reasoning(raw_text)
        return LLMResult(
            text=text,
            reasoning=reasoning,
            model=MOCK_MODEL_NAME,
            backend=self.name,
            elapsed_s=time.perf_counter() - start,
            messages=[dict(m) for m in messages],
            raw={"content": raw_text, "purpose": purpose, "seed": self.seed},
        )

    # -- challenge bookkeeping ---------------------------------------------------

    def _new_game(self) -> None:
        self._used: list[Challenge] = []
        self._current = None
        self._beaten = []
        self._steps = 0
        self._tier = 0
        self._mode = "any"  # how the player said they'd travel (see travel_mode)

    def _progress_after(self, messages: list[dict[str, str]], made_progress: bool) -> tuple[int, int]:
        """Steps completed after this round, and the target: from the prompt, else our own count."""
        stated = _stated_progress(messages)
        if stated is None:
            stated = (self._steps + (1 if made_progress else 0), 5)
        self._steps = stated[0]
        return stated

    def _next_challenge(self, progress: int = 0, target: int = 5) -> Challenge:
        """A fresh challenge from the tier this much progress calls for (the finale at the office last).

        The tier never goes back down within a game, even if a prompt says less progress.
        """
        if not hasattr(self, "_used"):
            self._new_game()
        tier = min(prompts.absurdity_index(progress, target), len(CHALLENGE_TIERS) - 1)
        tier = self._tier = max(tier, getattr(self, "_tier", 0))
        # Prefer an unused challenge in the right tier, then in the nearest tiers.
        order = sorted(range(len(CHALLENGE_TIERS)), key=lambda t: (abs(t - tier), -t))
        mode = getattr(self, "_mode", "any")
        for index in order:
            # Obstacles that fit how the player is travelling (or any journey) - a
            # cyclist never meets "your car has turned into a sandwich".
            fresh = [c for c in CHALLENGE_TIERS[index] if c not in self._used and fits_journey(c, mode)]
            if fresh:
                challenge = self._rng.choice(fresh)
                break
        else:
            challenge = self._rng.choice([c for c in CHALLENGE_TIERS[tier] if fits_journey(c, mode)]
                                         or list(CHALLENGE_TIERS[tier]))
        self._used.append(challenge)
        self._current = challenge
        return challenge

    def _challenge_in(self, messages: list[dict[str, str]]) -> Optional[Challenge]:
        """Which of our challenges does the prompt mention? (Falls back to the current one.)"""
        text = " ".join(" ".join(str(m.get("content") or "").split()) for m in messages)
        for challenge in ALL_CHALLENGES:
            if challenge.text[:60] in text:
                return challenge
        return self._current

    # -- writers: each returns (answer, fake reasoning) ---------------------------

    def _intro(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        self._new_game()
        story, hook = self._rng.choice(INTROS)
        thoughts = (
            f"I need a farcical reason for the player to be running late. {hook} "
            "No obstacle yet: the game asks them next how they plan to get to work. "
            "Keep it short, second person and family-friendly."
        )
        return story, thoughts

    def _outcome(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        made_progress = read_made_progress(messages)
        plan = _short(extract_plan(messages, fallback=False))
        progress, target = self._progress_after(messages, made_progress)
        users = " ".join(_without_player_text(m.get("content")) for m in messages if m.get("role") == "user")
        if "THIS ROUND: the player was asked how they plan to get to work" in users:
            return self._commute_outcome(made_progress, plan, progress, target)
        faced = self._challenge_in(messages) or self._current
        # A failed round keeps the same obstacle (it just gets sillier); a win brings a
        # new one, as absurd as the player's progress calls for.
        upcoming = self._next_challenge(progress, target) if (made_progress or faced is None) else faced
        if faced is None:
            faced = upcoming

        if plan:
            opener = self._rng.choice(SUCCESS_OPENERS if made_progress else FAILURE_OPENERS).format(plan=plan)
        else:
            opener = "You spring into action, and it works!" if made_progress else "You spring into action. It does not go well."
        aftermath = faced.win if made_progress else faced.lose
        if made_progress:
            self._beaten.append(faced)
        transition = self._rng.choice(SUCCESS_TRANSITIONS if made_progress else FAILURE_TRANSITIONS)
        if made_progress:
            plan_next = f"Then escalate the absurdity: {upcoming.nickname} feels right for step {progress + 1} of {target}."
        else:
            plan_next = f"The obstacle stays - {faced.nickname}, now even sillier."
        thoughts = (
            f"The judge says the player {'made progress' if made_progress else 'did not make progress'} against "
            f"{faced.nickname}, so I narrate {'a win' if made_progress else 'a comic setback'} - no changing the "
            f"verdict. {plan_next} Stay under 120 words and finish with a CHALLENGE line."
        )
        return f"{opener} {aftermath}\n\n{transition}\n\nCHALLENGE: {upcoming.text}", thoughts

    def _commute_outcome(self, made_progress: bool, plan: str, progress: int, target: int) -> tuple[str, str]:
        """Round 1: the player said how they'll travel. Set off (and meet the first obstacle), or not."""
        plan = plan or "a plan of some kind"
        if not made_progress:
            thoughts = "That plan doesn't actually get them going, so they're still at home. No CHALLENGE line."
            return self._rng.choice(COMMUTE_FAILURE).format(plan=plan), thoughts
        self._mode = travel_mode(plan)
        upcoming = self._next_challenge(progress, target)
        opener = self._rng.choice(COMMUTE_SUCCESS).format(plan=plan)
        journey = "any journey" if self._mode == "any" else f"a journey {_MODE_WORDS[self._mode]}"
        thoughts = (
            f"They picked their way to work: {plan!r}. Off they go - and the first obstacle has to suit "
            f"{journey}: {upcoming.nickname} does."
        )
        return f"{opener} It works, for about a minute.\n\nCHALLENGE: {upcoming.text}", thoughts

    def _judge(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        plan = extract_plan(messages)
        users = " ".join(" ".join(str(m.get("content") or "").split()) for m in messages if m.get("role") == "user")
        commute = any(" ".join(text.split())[:60] in users
                      for text in (prompts.COMMUTE_CHALLENGE, prompts.COMMUTE_JUDGE_CHALLENGE))
        made_progress, why = judge_plan(plan, commute=commute)
        if made_progress and commute:
            explanation = self._rng.choice(JUDGE_YES)
            analysis = (
                "Any real way of travelling counts in round 1, however short, and it doesn't give up or claim "
                "victory, so by my simple scripted rule it counts."
            )
        elif made_progress:
            explanation = self._rng.choice(JUDGE_YES)
            analysis = (
                "It's a concrete action of at least four words that doesn't give up or claim victory, so by my "
                "simple scripted rule it counts. Cartoon logic is allowed, so silliness is fine."
            )
        else:
            lines = JUDGE_GAVE_UP_COMMUTE if commute and why in ("gave_up", "empty") else _JUDGE_LINES.get(
                why, JUDGE_TOO_SHORT)
            explanation = self._rng.choice(lines)
            analysis = _JUDGE_ANALYSIS.get(why, _JUDGE_ANALYSIS["too_short"])
        challenge = None if commute else self._challenge_in(messages)
        if commute:
            facing = "the question of how to get to work, so any real way of travelling counts"
        else:
            facing = challenge.nickname if challenge else "the current obstacle"
        words = len(plan.split())
        thoughts = (
            f'The player is facing {facing}. Their plan: "{_short(plan, 120)}" ({words} '
            f"word{'s' if words != 1 else ''}). {analysis} "
            f"So made_progress = {'true' if made_progress else 'false'}. Answer with JSON only."
        )
        return json.dumps({"made_progress": made_progress, "explanation": explanation}), thoughts

    def _victory(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        scene, plan_line = self._rng.choice(VICTORIES)
        plan = _short(extract_plan(messages, fallback=False))
        parts = [scene.format(callbacks=self._callbacks())]
        if plan:
            parts.append(plan_line.format(plan=plan))
        thoughts = (
            "They made it to work! Time for a triumphant arrival. Brass band? Brass band. Call back some of the "
            "morning's chaos, credit their final move, and end on a warm note."
        )
        return " ".join(parts) + f"\n\n{VICTORY_CLOSER}", thoughts

    def _callbacks(self) -> str:
        """Up to three obstacles this player beat, e.g. "the sandwich car and the parked Moon"."""
        beaten = list(self._beaten)
        if self._current is not None and self._current not in beaten:
            beaten.append(self._current)  # the final challenge: victory is narrated instead of an outcome
        names = [c.nickname for c in beaten][-3:] or ["a sandwich-car", "a sphinx", "a very tired snail"]
        return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]

    def _quit(self, messages: list[dict[str, str]]) -> tuple[str, str]:
        thoughts = "The player wants to stop. Be kind and funny, don't scold them, and leave the door open to play again."
        return self._rng.choice(QUIT_ENDINGS), thoughts
