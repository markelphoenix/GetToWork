"""The family-friendly filter: guardrails for the text the local AI writes live.

Get To Work's story is written on the fly by a language model running on the
player's computer, and a model can say anything. Steam asks games with
live-generated AI content to have guardrails against illegal or offensive
output, and this module is the heart of them. The defences, in order:

1. The prompts ask for farcical, family-friendly slapstick (``prompts.py``).
2. Every piece of model text is checked here before it is shown:

   * :func:`check_text` - hard blocks: sexual content, slurs and hate speech,
     self-harm, graphic gore and hard drugs. The game asks the model once more
     with a stricter reminder, and if that fails too it uses a built-in line
     instead (``game.py``).
   * :func:`soften` - masks swearing ("d***") so the story can carry on.
   * :func:`check_player_input` - the same lists for what the player types: a
     flagged plan is refused before it ever reaches the model or Jev.

**How matching works** (and why innocent words don't trip it):

* The text is *normalised* first (:func:`normalize`): Unicode look-alikes are
  folded (full-width letters, "é" -> "e", a few Cyrillic and Greek letters that
  look Latin), everything is lowercased, invisible characters are dropped, and
  leetspeak is read as letters - 0->o, 1->i (or l), 3->e, 4->a, 5->s, 7->t,
  @->a, $->s - but only inside words that also contain a letter, so "8:45" or
  "1,500" stay numbers.
* Spacing and punctuation tricks are undone as *extra* readings of the text:
  runs of single letters are joined ("s e x", "s.e.x"), and symbols inside a
  word are dropped ("se*x", "se-x").
* Terms match **whole words only**, and a word stretched with a letter
  repeated three or more times still matches. So "Scunthorpe", "assassin",
  "classic", "grape" and "therapist" never match the shorter words hidden
  inside them, and "assess" or "rapping" never match a listed word that is
  just one doubled letter away.

The word lists live, scrambled with ROT13, in ``safety_terms.py`` (its
docstring explains why). Cartoon slapstick - a goose bonking you with a
baguette, a pie in the face, a bus that "dies" - is exactly what the game is
about, so none of that is listed.

No filter is perfect: the game also tells players that AI can surprise them
and how to report it (``notices.py``).
"""

from __future__ import annotations

import codecs
import functools
import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Optional

from . import safety_terms

__all__ = [
    "CATEGORY_LABELS",
    "SafetyVerdict",
    "check_player_input",
    "check_text",
    "hidden_note",
    "normalize",
    "soften",
]

# What each category is called when the game tells the player (or the review) about it.
CATEGORY_LABELS: dict[str, str] = {
    "sexual": "sexual content",
    "hate": "hateful language",
    "self_harm": "self-harm",
    "gore": "graphic gore",
    "drugs": "drugs",
}


@dataclass(frozen=True)
class SafetyVerdict:
    """The filter's answer for one piece of text.

    ``ok`` is True when the text can be shown. Otherwise ``category`` is one
    of :data:`CATEGORY_LABELS`' keys and ``matched`` the (normalised) words
    that tripped it - handy for tests and debugging; the game never shows or
    stores the matched words, only the category.
    """

    ok: bool
    category: Optional[str] = None
    matched: Optional[str] = None

    @property
    def label(self) -> str:
        """The category in plain words ("graphic gore"), or "" when the text is fine."""
        return CATEGORY_LABELS.get(self.category or "", self.category or "")


_OK = SafetyVerdict(ok=True)


def hidden_note(verdict: SafetyVerdict) -> str:
    """What the game keeps in place of text the filter blocked (for the review and transcripts)."""
    return f"(hidden by the family-friendly filter: {verdict.label or 'not family-friendly'})"


# ---------------------------------------------------------------------------
# Normalising text
# ---------------------------------------------------------------------------

# Invisible characters someone could hide inside a word to split it up:
# zero-width spaces and joiners, soft hyphens, bidi controls, variation selectors...
_INVISIBLE_RE = re.compile(
    "[\u00ad\u034f\u061c\u115f\u1160\u17b4\u17b5\u180b-\u180e\u200b-\u200f\u202a-\u202e"
    "\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0]"
)
# Cyrillic and Greek letters that look exactly like Latin ones (after lowercasing):
# Cyrillic a e o p c x y s i j k d l, Greek alpha omicron rho iota kappa nu.
_HOMOGLYPHS = str.maketrans({
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0445": "x", "\u0443": "y",
    "\u0455": "s", "\u0456": "i", "\u0458": "j", "\u043a": "k", "\u0501": "d", "\u04cf": "l",
    "\u03b1": "a", "\u03bf": "o", "\u03c1": "p", "\u03b9": "i", "\u03ba": "k", "\u03bd": "v",
})
# Leetspeak. "1" and "!" can stand for "i" or "l", so both readings are tried.
_LEET_I = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_LEET_L = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "l"})

# One character of a word: a letter or digit in any script, or a leetspeak
# symbol ("$h1t", "@ss"). A "!" only counts between two letters ("sh!t"), so
# the "!" at the end of "What a day!" stays punctuation.
_CHAR = r"(?:[^\W_]|[@$])"
_WORD_RE = re.compile(r"(?:[^\W_]|[@$]|(?<=[^\W_])!+(?=[^\W_]))+")
# A run of three or more single characters with a little punctuation between
# them: "s e x", "s.e.x", "s-e-x", "d * a * m * n".
_SPACED_RE = re.compile(
    rf"(?<![^\W_])(?<![@$]){_CHAR}(?![^\W_])(?![@$])"
    rf"(?:[\s.*_\-·•,/|+~]{{1,3}}{_CHAR}(?![^\W_])(?![@$])){{2,}}"
)
# Symbols inside a word: "se*x", "se-x", "s_e_x" (for the "squeezed" reading).
# Apostrophes are left alone: squeezing them would glue contractions into
# different words ("who're" is not a rude word).
_INNER_SYMBOLS_RE = re.compile(r"(?<=[^\W_])[*_.\-^~]+(?=[^\W_])")
# Punctuation that ends a sentence or clause. A listed *phrase* never matches
# across one, so "...at the end. My life..." is not read as one phrase.
_CLAUSE_BREAK_RE = re.compile(r"[.!?;:,()\[\]{}\"“”«»]")
_BREAK = "|"  # how a clause break shows in the normalised reading


def _fold(text: str) -> str:
    """Unicode folding: look-alikes, invisible characters, case, accents."""
    text = unicodedata.normalize("NFKC", text)  # full-width letters -> plain ones, ligatures -> letters
    text = _INVISIBLE_RE.sub("", text).casefold()
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return text.translate(_HOMOGLYPHS)


def _read_word(word: str, leet: dict) -> str:
    """One word with leetspeak read as letters - only if it has a letter at all ("$5" stays "$5")."""
    return word.translate(leet) if any(c.isalpha() for c in word) else word


def _words_to_text(text: str, leet: dict) -> str:
    """The words of ``text`` one space apart, with a "|" wherever a clause ends between two words."""
    out: list[str] = []
    end = 0
    for match in _WORD_RE.finditer(text):
        if out and _CLAUSE_BREAK_RE.search(text, end, match.start()):
            out.append(_BREAK)
        out.append(_read_word(match.group(0), leet))
        end = match.end()
    return " ".join(out)


def normalize(text: Any) -> str:
    """The plain reading the filter matches against: folded, lowercased words, one space apart
    (with a "|" where a sentence or clause ends).

    >>> normalize("SCÚNTHORPE  $h1t! Bye")
    'scunthorpe shit | bye'
    """
    return _words_to_text(_fold(str(text if text is not None else "")), _LEET_I)


def _readings(text: str) -> list[str]:
    """Every way the filter reads a text: plain, symbols squeezed out, spaced letters joined,
    each with "1"/"!" read as "i" (and, when there are any, as "l")."""
    folded = _fold(text)
    joined = _SPACED_RE.sub(lambda m: "".join(re.findall(_CHAR, m.group(0))), folded)
    variants = dict.fromkeys([folded, _INNER_SYMBOLS_RE.sub("", folded), joined])
    tables = [_LEET_I, _LEET_L] if ("1" in folded or "!" in folded) else [_LEET_I]
    readings = dict.fromkeys(_words_to_text(v, t) for v in variants for t in tables)
    return [r for r in readings if r]


# ---------------------------------------------------------------------------
# The word lists, compiled
# ---------------------------------------------------------------------------


def _term_pattern(term: str) -> str:
    """A regex for one listed term, words one space apart.

    A letter may be stretched to three or more copies ("heeeelp" still reads as
    "help"), but not merely doubled: plenty of real words differ from a
    listed one only by a doubled letter ("shiite", "assess", "rapping"), and
    those must never match.
    """
    words = []
    for word in term.split():
        pattern = ""
        for run in re.finditer(r"(.)\1*", word):  # runs of the same letter: "ass" -> "a", "ss"
            letter, count = re.escape(run.group(1)), len(run.group(0))
            pattern += f"{letter}(?:{letter}{{2,}})?" if count == 1 else f"{letter}{{{count},}}"
        words.append(pattern)
    return " ".join(words)


def _decode(terms: tuple[str, ...]) -> list[str]:
    return [" ".join(codecs.decode(t, "rot13").split()) for t in terms]


def _alternation(terms: list[str]) -> str:
    # Longest first, so "white supremacists" wins over "white supremacist".
    return "|".join(_term_pattern(t) for t in sorted(set(terms), key=len, reverse=True))


@functools.lru_cache(maxsize=1)
def _compiled() -> tuple[list[tuple[str, re.Pattern[str]]], re.Pattern[str], re.Pattern[str]]:
    """(category -> whole-word pattern, the mild-swearing pattern, the exempt phrases). Built once."""
    blocked = [
        (category, re.compile(rf"(?<![^ ])(?:{_alternation(_decode(terms))})(?![^ ])"))
        for category, terms in safety_terms.BLOCKED.items()
    ]
    mild = re.compile(rf"(?:{_alternation(_decode(safety_terms.MILD_PROFANITY))})")
    exempt = re.compile(rf"(?<![^ ])(?:{_alternation(_decode(safety_terms.EXEMPT_PHRASES))})(?![^ ])")
    return blocked, mild, exempt


# ---------------------------------------------------------------------------
# The public checks
# ---------------------------------------------------------------------------


def check_text(text: Any) -> SafetyVerdict:
    """Is this model-written text fine to show? Hard blocks only (swearing is :func:`soften`'s job).

    Blocks sexual content, slurs and hate speech, self-harm, graphic gore and
    hard drugs, however they're spelled (case, accents, leetspeak, spaced-out
    letters). Never raises; ``None`` or empty text is fine.
    """
    raw = str(text if text is not None else "")
    if not raw.strip():
        return _OK
    blocked, _mild, exempt = _compiled()
    readings = [exempt.sub(" ", reading) for reading in _readings(raw)]
    for category, pattern in blocked:
        for reading in readings:
            match = pattern.search(reading)
            if match:
                return SafetyVerdict(ok=False, category=category, matched=match.group(0))
    return _OK


def check_player_input(text: Any) -> SafetyVerdict:
    """Is this plan fine to use? The same lists as :func:`check_text`.

    A separate function, because what the *player* types is handled
    differently: a flagged plan is refused before it reaches the model or Jev,
    and the player simply tries another one.
    """
    return check_text(text)


def _is_mild(word: str) -> bool:
    _blocked, mild, _exempt = _compiled()
    folded = _fold(word)
    return any(mild.fullmatch(_read_word(folded, table)) for table in (_LEET_I, _LEET_L))


def _mask_word(match: re.Match[str]) -> str:
    word = match.group(0)
    return word[0] + "*" * (len(word) - 1) if _is_mild(word) else word


def _mask_spaced(match: re.Match[str]) -> str:
    """"d a m n" -> "d * * *": keep the first letter and the gaps, star the rest."""
    run = match.group(0)
    letters = list(re.finditer(_CHAR, run))
    if not _is_mild("".join(m.group(0) for m in letters)):
        return run
    out = list(run)
    for m in letters[1:]:
        out[m.start()] = "*"
    return "".join(out)


def soften(text: Any) -> str:
    """Mask swearing, keeping the first letter: "damn" -> "d***", "What the hell" -> "What the h***".

    Everything else is left exactly as it was, so the story still reads
    naturally. Safe to run twice (a masked word is never masked again).
    Invisible characters are removed along the way.
    """
    cleaned = _INVISIBLE_RE.sub("", str(text if text is not None else ""))
    if not cleaned.strip():
        return cleaned
    cleaned = _WORD_RE.sub(_mask_word, cleaned)
    return _SPACED_RE.sub(_mask_spaced, cleaned)
