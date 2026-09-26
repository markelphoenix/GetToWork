"""The family-friendly filter (safety.py) and the AI-content notices (notices.py).

What it must block, what it only softens - and, just as important, the
innocent words a silly story about getting to work uses all the time, which it
must leave alone. The blocked words themselves are never written out in this
file: they're read from safety_terms.py (ROT13) or written in ROT13 here.
"""

from __future__ import annotations

import codecs
import dataclasses
import re
import time
from pathlib import Path

import pytest

from gettowork import game, notices, safety, safety_terms
from gettowork.backends import mock
from gettowork.safety import SafetyVerdict, check_player_input, check_text, normalize, soften


def r13(text: str) -> str:
    return codecs.decode(text, "rot13")


BLOCKED = [(category, r13(term)) for category, terms in safety_terms.BLOCKED.items() for term in terms]
SINGLE_WORDS = [(category, term) for category, term in BLOCKED if " " not in term]
MILD = [r13(term) for term in safety_terms.MILD_PROFANITY]
# Test ids never spell the words out (pytest prints them).
BLOCKED_IDS = [f"{category}-{i}" for i, (category, _term) in enumerate(BLOCKED)]
SINGLE_IDS = [f"{category}-{i}" for i, (category, _term) in enumerate(SINGLE_WORDS)]
MILD_IDS = [f"mild-{i}" for i in range(len(MILD))]


# ---------------------------------------------------------------------------
# The word lists
# ---------------------------------------------------------------------------


def test_every_category_has_a_label_and_terms():
    assert set(safety_terms.BLOCKED) == set(safety.CATEGORY_LABELS) == {"sexual", "hate", "self_harm", "gore", "drugs"}
    for category, terms in safety_terms.BLOCKED.items():
        assert len(terms) >= 10, category
    assert len(safety_terms.MILD_PROFANITY) >= 30
    assert safety_terms.EXEMPT_PHRASES


def test_self_harm_phrases_cover_every_person_and_tense():
    """Generated from building blocks, so narration about a side character is covered too
    (every listed term is then checked by test_every_listed_term_is_blocked_in_its_category)."""
    terms = set(safety_terms.BLOCKED["self_harm"])
    assert len(terms) > 200
    assert set(safety_terms._phrases(safety_terms._KILL, safety_terms._SELF)) <= terms
    assert set(safety_terms._phrases(safety_terms._TAKE, safety_terms._WHOSE, ("bja yvsr",))) <= terms
    # ...and the figures of speech built from the same words are exempt.
    assert set(safety_terms._phrases(safety_terms._KILL, safety_terms._SELF, ("ynhtuvat",))) <= set(
        safety_terms.EXEMPT_PHRASES)


def test_every_entry_is_rot13_of_lowercase_letters_and_single_spaces():
    lists = [*safety_terms.BLOCKED.values(), safety_terms.MILD_PROFANITY, safety_terms.EXEMPT_PHRASES]
    for terms in lists:
        assert len(set(terms)) == len(terms), "duplicate entry"
        for term in terms:
            assert re.fullmatch(r"[a-z]+(?: [a-z]+)*", term), term


def test_no_term_is_listed_twice_across_lists():
    everything = [t for terms in safety_terms.BLOCKED.values() for t in terms] + list(safety_terms.MILD_PROFANITY)
    assert len(set(everything)) == len(everything)


def test_the_terms_file_never_spells_a_listed_word_out():
    source = Path(safety_terms.__file__).read_text(encoding="utf-8").lower()
    words = set(re.findall(r"[a-z]+", source))
    category_names = set(" ".join(safety.CATEGORY_LABELS.values()).split())  # "sexual content", "graphic gore"...
    for _category, term in BLOCKED:
        if " " not in term and len(term) >= 4 and term not in category_names:
            assert term not in words, "a blocked word appears in plain text in safety_terms.py"
    for term in MILD:
        if len(term) >= 4:
            assert term not in words, "a mild word appears in plain text in safety_terms.py"


def test_the_terms_file_explains_why_it_is_scrambled():
    doc = safety_terms.__doc__ or ""
    assert "ROT13" in doc and "Why is everything in this file scrambled?" in doc
    assert "not* security" in doc


# ---------------------------------------------------------------------------
# Hard blocks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("category", "term"), BLOCKED, ids=BLOCKED_IDS)
def test_every_listed_term_is_blocked_in_its_category(category, term):
    for text in (term, f"The goose says {term} very loudly.", f"{term.upper()}!", f'"{term.capitalize()}"'):
        verdict = check_text(text)
        assert not verdict.ok
        assert verdict.category == category
        assert verdict.matched
    assert check_player_input(f"I shout {term} at the bus").category == category


def _leet(word: str) -> str:
    return word.translate(str.maketrans({"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}))


def _full_width(word: str) -> str:
    return "".join(chr(ord(c) + 0xFEE0) if c.isascii() and c.isalpha() else c for c in word)


def _accented(word: str) -> str:
    return word.translate(str.maketrans({"a": "á", "e": "ë", "i": "î", "o": "ö", "u": "ü"}))


def _cyrillic(word: str) -> str:
    look_alikes = {"a": "\u0430", "e": "\u0435", "o": "\u043e", "p": "\u0440", "c": "\u0441", "x": "\u0445",
                   "y": "\u0443"}  # Cyrillic letters that look Latin
    return word.translate(str.maketrans(look_alikes))


def _disguises(word: str) -> list[str]:
    stretched = word[:-1] + word[-1] * 4
    return [
        word.upper(),
        word.title(),
        stretched,
        *([_leet(word)] if any(c.isalpha() for c in _leet(word)) else []),  # all-digit "words" are numbers
        " ".join(word),  # s e x
        ".".join(word),  # s.e.x
        "-".join(word),
        "_".join(word),
        " * ".join(word),
        word[:1] + "*" + word[1:],  # a symbol stuck inside
        word[:1] + "\u200b" + word[1:],  # an invisible character inside
        word[:1] + "\u00ad" + word[1:],  # a soft hyphen inside
        _full_width(word),
        _accented(word),
        _cyrillic(word),
    ]


@pytest.mark.parametrize(("category", "term"), SINGLE_WORDS, ids=SINGLE_IDS)
def test_disguised_spellings_are_still_blocked(category, term):
    for disguised in _disguises(term):
        verdict = check_text(f"Then it says {disguised} and runs off.")
        assert not verdict.ok, f"disguise #{_disguises(term).index(disguised)} slipped through"
        assert verdict.category == category


def test_dollar_at_and_one_as_l_are_read_as_letters():
    assert not check_text(r13("$rk")).ok  # $ for s
    assert not check_text(r13("encr").replace("a", "@").replace("e", "3")).ok  # @ for a, 3 for e
    assert not check_text(r13("p1vg")).ok  # 1 read as "l"
    assert not check_text(r13("u3eb!a")).ok  # "!" between letters read as "i"


def test_phrases_match_across_hyphens_but_not_across_sentences():
    phrase = r13("xvyy lbhefrys")
    assert check_text(phrase).category == "self_harm"
    assert check_text(phrase.replace(" ", "-")).category == "self_harm"
    assert check_text(phrase.replace(" ", "   ")).category == "self_harm"
    first, second = phrase.split()
    assert check_text(f"I {first}. {second.capitalize()} is lovely").ok  # two sentences
    assert check_text(f"I {first}, {second} included").ok  # a clause break


def test_squeezed_compounds_are_blocked():
    assert check_text(r13("frys-unez")).category == "self_harm"
    assert check_text(r13("oybj-wbo")).category == "sexual"


@pytest.mark.parametrize("text", ["", "   ", None, "\n\n"])
def test_empty_text_is_fine(text):
    assert check_text(text) == SafetyVerdict(ok=True)
    assert check_player_input(text).ok


def test_non_strings_are_read_as_text():
    assert check_text(12345).ok
    assert soften(12345) == "12345"


# ---------------------------------------------------------------------------
# Innocent text: never blocked, never changed
# ---------------------------------------------------------------------------

INNOCENT = [
    # The classic "Scunthorpe problem": towns and words with a short word hidden inside.
    "I drive through Scunthorpe, Penistone and Clitheroe on my way to work.",
    "Middlesex, Sussex, Essex and Wessex are all very sleepy this morning.",
    "The assassin bug assesses the class of bass players with a classic compass.",
    "A grape therapist drapes a scrap of trapeze cloth over the parapet of rapeseed.",
    "I order a cocktail for the peacock, the cockatoo and the shuttlecock in the cockpit.",
    "I cook shitake and shiitake mushrooms for my boss.",
    "The analyst's analysis is banal, and the canal is analogue.",
    "The goose sniggers, and a bigger digger pulls the trigger on the confetti cannon.",
    "The heroine and the heroines of the story are heroic.",
    "The methane-powered method of the Methodist minister works.",
    "Japan, the Japanese japonica and a few harmless japes in Pakistan.",
    "Niger, Nigeria and a niggling doubt about the bus timetable.",
    "The pedestrian checks his pedometer, then pedals past the torpedo.",
    "A nudge, a thorny hornet, a trombone and a swanky titter.",
    "Cumin, cucumber, the cumulus clouds of Cumbria and a sextant for the sexton.",
    "Summa cum laude, and my bedroom-cum-office is spic and span.",
    "I pull on my Gore-Tex jacket and wave at Al Gore.",
    "The heebie-jeebies make the squawking parrot jump.",
    "The flame retardant condominium has a condor on the roof.",
    "My assistant hassles the passenger about the cassette, the lasso and the molasses.",
    "Hello! The shell of Othello's hellebore is a hellfire orange.",
    "Classy glasses, a harassed bassoon and a casserole.",
    # Words with an innocent everyday meaning that the lists deliberately leave out.
    "A chink of light shines on the blue tits while the cock crows.",
    "The little Dutch boy plugs the dyke with his finger.",
    "I grope in the dark for the light switch and spill my guts to the cat.",
    "A hot dog with kraut, and a honky goose.",
    "Moby Dick and Dick Whittington ride a pussycat.",
    "Paint stripper, a rugby hooker and a naked flame.",
    "Chicken breasts, a Coke machine, a crack in the pavement and a pot of tea.",
    "Acid rain, a speed bump and the ecstasy of flight.",
    "An overdose of coffee and a line of Coke cans.",
    "Spare me the gory details, but waiting for the bus is torture.",
    "Get Jack off the roof before the master race-car driver arrives!",
    "Does my brain matter? The grey matter of the argument is clear.",
    "A smoking crack in the volcano and the smoke cracks the window.",
    "Ana is a pro at tennis.",
    "Who're you calling a goose? She'll assess the Shiite scholar's notes while rapping on the door.",
    "He rapped twice, and the rapper kept rapping about the busy bees.",
    "Lifting the sofa is a two-hand job, and the bus gives a jerk off the kerb.",
    # Cartoon slapstick: exactly what the game is about.
    "The goose bonks you with a baguette.",
    "Kill the lights and sneak past the sleeping dragon.",
    "The bus died halfway up the hill.",
    "A custard pie hits you square in the face, and you fall flat on your back.",
    "The troll whacks the lamppost with a rubber chicken until it apologises.",
    "The car explodes into a cloud of confetti and harmless glitter.",
    "The ninja squirrels karate-kick the hedge and you get squashed flat like a pancake.",
    "Your boss will kill you if you're late, and I could murder a cup of tea.",
    "You're dead meat, says the goose, dying of laughter.",
    "I cut myself a slice of cake and hang up my coat.",
    "I take my own life jacket onto the ferry, and my own life-sized cardboard cutout to work.",
    "I cycle through the oilseed rape field, then past a vintage car with suicide doors.",
    "A blood orange, a bloodhound, bloodshot eyes and high blood pressure.",
    "He beheld the forehead of the decapod ahead.",
    # Other languages, numbers, abbreviations.
    "Voy en coche al trabajo. Je prends le vélo. Ich fahre mit dem Bus.",
    "我骑自行车去上班。自転車で行きます。Я еду на автобусе.",
    "It's 8:45, work starts at 9:00, room 101, a 4x4, 7am, 1,500 steps, a $5 bill and 555-0123.",
    "Email me at bob@example.com about the S.W.A.T. team from the U.S.A. at 9 a.m.",
    "I sing A B C and hum l33t h4x0r tunes b4 9am.",
    "3 geese and 5 swans.",
]
MILD_BUT_FINE = [
    "What a hell of a day!",
    "A bloody nose, a damn fine plan and a crappy umbrella.",
    "Hell's bells, the bus is late again.",
]


@pytest.mark.parametrize("text", INNOCENT + MILD_BUT_FINE)
def test_innocent_text_is_never_blocked(text):
    assert check_text(text).ok
    assert check_player_input(text).ok


@pytest.mark.parametrize("text", INNOCENT)
def test_innocent_text_is_never_changed(text):
    assert soften(text) == text


def test_the_innocent_list_is_big():
    assert len(INNOCENT) >= 50


# ---------------------------------------------------------------------------
# Softening swearing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("word", MILD, ids=MILD_IDS)
def test_every_mild_word_is_masked_but_not_blocked(word):
    assert check_text(word).ok
    masked = word[0] + "*" * (len(word) - 1)
    assert soften(word) == masked
    assert soften(f"Oh {word}, the bus!") == f"Oh {masked}, the bus!"
    assert soften(word.capitalize()) == word[0].upper() + "*" * (len(word) - 1)


def test_soften_examples():
    assert soften("What a hell of a day!") == "What a h*** of a day!"
    assert soften("Hell's bells") == "H***'s bells"
    assert soften("damn") == "d***"
    assert soften("DAMN it") == "D*** it"
    assert soften("d4mn") == "d***"
    assert soften("daaamn") == "d*****"
    assert soften("he11 no") == "h*** no"
    assert soften("A bloody nose") == "A b***** nose"


def test_soften_masks_spaced_out_words_and_keeps_the_gaps():
    assert soften("d a m n") == "d * * *"
    assert soften("D.A.M.N!") == "D.*.*.*!"
    assert soften("a b c") == "a b c"


def test_soften_keeps_everything_else_exactly():
    text = "  Line one: hell.\n\n\tLine two, (damn) - done!  "
    assert soften(text) == "  Line one: h***.\n\n\tLine two, (d***) - done!  "


def test_soften_is_idempotent_and_drops_invisible_characters():
    once = soften("What the hell, damn it, d a m n")
    assert soften(once) == once
    assert soften("be\u200bst") == "best"


def test_soften_of_nothing():
    assert soften(None) == ""
    assert soften("") == ""
    assert soften("   ") == "   "


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def test_verdict_fields_and_labels():
    verdict = check_text(r13("pbpnvar"))
    assert (verdict.ok, verdict.category) == (False, "drugs")
    assert verdict.label == "drugs"
    assert SafetyVerdict(ok=True).label == ""
    for category, label in safety.CATEGORY_LABELS.items():
        assert SafetyVerdict(ok=False, category=category).label == label


def test_hidden_note_names_the_category_never_the_words():
    verdict = check_text(f"a {r13('qrpncvgngrq')} gingerbread man")
    note = safety.hidden_note(verdict)
    assert note == "(hidden by the family-friendly filter: graphic gore)"
    assert verdict.matched not in note


def test_normalize():
    assert normalize("\uff33\uff43únthorpe  $h1t! Bye") == "scunthorpe shit | bye"
    assert normalize("It's 8:45") == "it s 8 | 45"
    assert normalize(None) == ""


def test_long_text_is_fast():
    text = "The goose bonks you with a baguette and you tumble into the custard moat. " * 1500
    start = time.perf_counter()
    assert check_text(text).ok
    assert soften(text) == text
    assert time.perf_counter() - start < 3.0


def test_verdict_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SafetyVerdict(ok=True).ok = False  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The game's own built-in text passes untouched
# ---------------------------------------------------------------------------


def _strings(value):
    if isinstance(value, str):
        yield value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for f in dataclasses.fields(value):
            yield from _strings(getattr(value, f.name))
    elif isinstance(value, (tuple, list, set, frozenset)):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _strings(key)
            yield from _strings(item)


def test_every_built_in_line_passes_the_filter_untouched():
    """The canned lines the filter falls back to (and the pretend model's whole script) must be clean."""
    texts = []
    for module in (mock, game):
        for name in dir(module):
            if not name.startswith("__"):
                texts += list(_strings(getattr(module, name)))
    assert len(texts) > 200
    for text in texts:
        assert check_text(text).ok, text[:80]
        assert soften(text) == text, text[:80]


# ---------------------------------------------------------------------------
# notices.py
# ---------------------------------------------------------------------------


def test_ai_content_notice_is_short_friendly_and_complete():
    text = notices.AI_CONTENT_NOTICE
    assert len(text.split()) < 120
    for phrase in ("written live by an AI", "your own computer", "family-friendly filter", "surprise",
                   "Report a problem", "Discussions", "store page"):
        assert phrase in text, phrase
    # Steam's overlay can't open over the game's window on Windows, macOS or a Linux desktop:
    # it's never the reporting path the notice promises.
    assert "Shift+Tab" not in text and "overlay" not in text
    assert check_text(text).ok and soften(text) == text
    assert "<" not in text  # plain Markdown, no HTML


def test_steam_disclosure_describes_the_real_guardrails():
    text = " ".join(notices.STEAM_AI_DISCLOSURE.split())
    for phrase in ("generative AI", "live", "locally", "llama.cpp", "sexual content", "hate speech", "self-harm",
                   "graphic gore", "hard drugs", "leetspeak", "masked", "once more", "stricter",
                   "pre-written line", "What the player types", "refused", "Report a problem", "store page",
                   "uncensored", "Jev"):
        assert phrase in text, phrase
    # It matches what the code really does.
    assert "safety_retry" in game.SPINNERS and "SAFETY_REMINDER" in dir(game.prompts)
    assert notices.REPORT_HOW in notices.AI_CONTENT_NOTICE


# ---------------------------------------------------------------------------
# prompts.safety_retry_messages: the firmer second request
# ---------------------------------------------------------------------------


def _all_requests():
    from gettowork import prompts

    story = dict(intro="You woke up late.", history=["Round 1: it worked."])
    return {
        "intro": prompts.intro_messages(),
        "outcome": prompts.outcome_messages(challenge="A goose.", plan="I bow to the goose", made_progress=True,
                                            judge_note="", progress=2, target=5, **story),
        "judge": prompts.judge_messages(challenge="A goose.", plan="I bow to the goose", progress=1, target=5,
                                        **story),
        "victory": prompts.victory_messages(final_plan="I bow", **story),
        "ending_quit": prompts.quit_messages(progress=1, target=5, **story),
    }


@pytest.mark.parametrize("purpose", ["intro", "outcome", "judge", "victory", "ending_quit"])
def test_the_safety_retry_is_the_same_request_with_a_firmer_reminder(purpose):
    from gettowork import prompts

    original = _all_requests()[purpose]
    before = [dict(m) for m in original]
    retry = prompts.safety_retry_messages(original)
    assert original == before  # the first request is left untouched
    assert len(retry) == len(original)
    assert retry[0]["content"].startswith(f"TASK: {purpose}\n")
    assert retry[0]["content"].endswith(prompts.SAFETY_REMINDER)
    assert retry[-1]["content"].startswith(original[-1]["content"])
    assert retry[-1]["content"].endswith("Remember: keep it completely family-friendly - clean, gentle and kind.")
    assert mock.detect_purpose(retry) == purpose  # the pretend model still knows what it's asked for


def test_the_pretend_model_answers_a_safety_retry_in_the_right_shape():
    from gettowork import prompts

    requests = _all_requests()
    backend = mock.MockBackend(seed=1, think=False)
    reply = backend.chat(prompts.safety_retry_messages(requests["judge"]), json_mode=True)
    assert prompts.parse_judge_json(reply.text) is not None
    reply = backend.chat(prompts.safety_retry_messages(requests["outcome"]))
    assert all(prompts.parse_challenge(reply.text))


def test_the_safety_retry_handles_odd_message_lists():
    from gettowork import prompts

    assert prompts.safety_retry_messages([]) == [{"role": "user", "content": prompts.SAFETY_REMINDER}]
    only_system = prompts.safety_retry_messages([{"role": "system", "content": "TASK: intro\nHi"}])
    assert only_system == [{"role": "system", "content": "TASK: intro\nHi\n\n" + prompts.SAFETY_REMINDER}]


def test_an_echoed_reminder_never_ends_up_in_the_story():
    from gettowork import prompts

    reply = ("You leap over the goose.\nIMPORTANT: this game is for all ages, including children.\n"
             "Remember: keep it completely family-friendly - clean, gentle and kind.")
    assert prompts.clean_story(reply) == "You leap over the goose."


def test_the_reminder_itself_passes_the_filter():
    from gettowork import prompts

    assert check_text(prompts.SAFETY_REMINDER).ok
    assert "family-friendly" in prompts._NARRATOR_RULES and "no swearing" in prompts._NARRATOR_RULES


def test_markdown_clean_up_keeps_masked_words_but_still_removes_bold():
    from gettowork import prompts

    story = prompts.clean_story('You shout "d*** it" and **run** past a** h*** and a **big** goose.')
    assert story == 'You shout "d*** it" and run past a** h*** and a big goose.'
    narration, challenge = prompts.parse_challenge('You flee.\n**CHALLENGE:** A goose yells "d***", then c***.')
    assert (narration, challenge) == ("You flee.", 'A goose yells "d***", then c***.')
    assert prompts.clean_story("**Bold** start, x**2 maths and **A** letter.") == "Bold start, x2 maths and A letter."


def test_report_a_problem_leads_to_the_games_steam_discussions():
    assert notices.report_url(1234560) == "https://steamcommunity.com/app/1234560/discussions/"
    assert notices.report_url() == (notices.report_url(notices.STEAM_APP_ID) if notices.STEAM_APP_ID
                                    else notices.STORE_SEARCH_URL)
    assert notices.report_url().startswith("https://")  # the window only opens http(s) links
    assert "overlay" not in notices.STEAM_AI_DISCLOSURE.replace("\n", " ")
