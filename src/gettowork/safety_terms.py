"""The word lists behind the family-friendly filter (``safety.py`` reads them).

**Why is everything in this file scrambled?** The lists are, by design, full
of words nobody wants to read: slurs, sexual terms, drug names. They are
stored in ROT13 - every letter moved 13 places along the alphabet, so
``"uryyb"`` is ``"hello"`` - which means:

* anyone browsing the code (learners, reviewers, kids looking over a
  shoulder) doesn't get a wall of offensive words;
* code-search tools, chat bots and automatic scanners don't flag the whole
  project because of this one file.

ROT13 is *not* security - it's politeness. ``codecs.decode(term, "rot13")``
turns a term back into plain text, and that is all ``safety.py`` does.

**How the lists are used.** Each term is one word or a short phrase. Terms are
matched as *whole words* on normalised text (see ``safety.normalize``), so
"Scunthorpe", "assassin" or "grape" never trip over the shorter words hidden
inside them. Inflections are listed one by one rather than guessed, because a
guessed ending can turn an innocent word into a match (a word plus "es" can be
an ordinary, harmless word). Stretched-out spellings are matched
automatically: a listed word with a letter repeated three or more times still
counts (a merely doubled letter doesn't, as that is often a different word).
A phrase only matches within one sentence or clause (never across a comma or
a full stop).

**What's deliberately left out.** Words with a common innocent meaning in a
silly story about getting to work are *not* listed, even when they can be
used as insults: a chink of light, blue tits in the garden, a cock crowing,
the Dutch boy's dyke, "grope in the dark", "spill your guts", kraut on a hot
dog, a honky goose. The prompts already tell the model to stay
family-friendly; this list is the safety net for clear failures, so it
prefers never to spoil an innocent sentence.

**Deliberate choices the other way.** A couple of slurs stay blocked even
though they also have an innocent British meaning (a slang word for a
cigarette, an old name for a dish of meatballs): the slur is the far more
likely reading, and a wrongly blocked line only costs the model a second
try. The gore words stay blocked even when the victim is a snowman or a
garden gnome: telling a snowman from a person needs more than a word list,
and the game's prompts never ask for either. The British name of a yellow
field crop, and figures of speech like "killing myself laughing", are let
through (``EXEMPT_PHRASES``).

To add a term: ``codecs.encode("new term", "rot13")`` and paste the result
into the right list. ``tests/test_safety.py`` checks that every entry decodes
to lowercase letters and spaces, and that none appears here in plain text.
"""

from __future__ import annotations

import itertools

__all__ = ["BLOCKED", "MILD_PROFANITY", "EXEMPT_PHRASES"]


def _phrases(*choices: tuple[str, ...]) -> tuple[str, ...]:
    """Every combination of the word choices, as phrases: ("n", "o"), ("p",) -> ("n p", "o p")."""
    return tuple(" ".join(words) for words in itertools.product(*choices))


# Building blocks for the self-harm phrases, so every person and tense is covered.
_SELF = ("zlfrys", "lbhefrys", "uvzfrys", "urefrys", "gurzfryirf", "gurzfrys", "bhefryirf", "lbhefryirf",
         "lbhe frys")  # myself ... yourselves, and "your self" written as two words
_WHOSE = ("zl", "lbhe", "uvf", "ure", "gurve", "bhe")  # my, your, his, her, their, our
_KILL = ("xvyy", "xvyyf", "xvyyrq", "xvyyvat")
_END = ("raq", "raqf", "raqrq", "raqvat")
_TAKE = ("gnxr", "gnxrf", "gbbx", "gnxvat", "gnxra")
_CUT = ("phg", "phgf", "phggvat", "fyvg", "fyvgf", "fyvggvat", "fynfu", "fynfurf", "fynfurq", "fynfuvat")

# Hard blocks, by category: text containing any of these is not shown (the
# game asks the model again, then uses a built-in line) and a plan containing
# one is refused.
BLOCKED: dict[str, tuple[str, ...]] = {
    "sexual": (  # sexual content
        "frk", "frkhny", "frkhnyyl", "frkl", "cbea", "cbeab", "cbeabf", "cbeaf", "cbeabtencul",
        "cbeabtencuvp", "ahqr", "ahqrf", "ahqvgl", "obbof", "gvggvrf", "cravf", "cravfrf", "intvan",
        "intvanf", "ihyin", "pyvgbevf", "pyvg", "qvyqb", "qvyqbf", "betnfz", "betnfzf", "betl", "betvrf",
        "ubeal", "obare", "obaref", "oybjwbo", "oybjwbof", "unaqwbo", "unaqwbof", "evzwbo", "phzfubg",
        "phzfubgf", "znfgheongr", "znfgheongrf", "znfgheongrq", "znfgheongvat", "znfgheongvba", "jnax",
        "jnaxvat", "encr", "encrq", "encrf", "encvat", "encvfg", "encvfgf", "zbyrfg", "zbyrfgf", "zbyrfgrq",
        "zbyrfgvat", "zbyrfgre", "zbyrfgref", "crqbcuvyr", "crqbcuvyrf", "cnrqbcuvyr", "cnrqbcuvyrf",
        "crqbcuvyvn", "cnrqbcuvyvn", "crqb", "crqbf", "cnrqb", "cnrqbf", "vaprfg", "orfgvnyvgl", "sryyngvb",
        "phaavyvathf", "ohggcyht", "fgevcgrnfr", "cebfgvghgr", "cebfgvghgrf", "cebfgvghgvba", "juber",
        "juberf", "fyhg", "fyhgf", "fyhggl", "oebgury", "oebguryf", "uragnv", "afsj", "zvys", "zvysf", "oqfz",
        "obaqntr", "phag", "phagf", "pbpxfhpxre", "pbpxfhpxref", "travgnyf", "travgnyvn", "grfgvpyrf",
        "fpebghz", "pbaqbz", "pbaqbzf", "frkgvat", "baylsnaf", "rebgvp", "rebgvpn", "yrjq", "guerrfbzr",
        "sbercynl", "frzra", "wvmm", "rwnphyngr", "rwnphyngrq", "rwnphyngvba", "oybj wbo", "oybj wbof",
        "nany frk", "ohgg cyht", "fgevc anxrq", "fgevccrq anxrq", "trg anxrq", "trgf anxrq", "trggvat anxrq",
    ),
    "hate": (  # slurs and hate speech
        "avttre", "avttref", "avttn", "avttnf", "avttnm", "snttbg", "snttbgf", "snt", "sntf", "genaal",
        "genaavrf", "furznyr", "furznyrf", "ergneq", "ergneqf", "ergneqrq", "fcnm", "fcnmm", "fcnfgvp",
        "fcnfgvpf", "xvxr", "xvxrf", "fcvp", "fcvpf", "tbbx", "tbbxf", "jrgonpx", "jrgonpxf", "enturnq",
        "enturnqf", "gbjryurnq", "gbjryurnqf", "mvccreurnq", "mvccreurnqf", "cnxv", "cnxvf", "wnc", "wncf",
        "ornare", "ornaref", "tbyyvjbt", "tbyyvjbtf", "tbyyljbt", "qntb", "qntbf", "erqfxva", "erqfxvaf",
        "fdhnj", "fdhnjf", "urro", "urrof", "ulzvr", "xxx", "hagrezrafpu", "urvy uvgyre", "fvrt urvy",
        "juvgr fhcerznpl", "juvgr fhcerznpvfg", "juvgr fhcerznpvfgf", "xh xyhk xyna", "tnf gur wrjf",
        "rguavp pyrnafvat",
    ),
    "self_harm": (  # self-harm: every person and tense ("kill himself", "took her own life"...)
        "fhvpvqr", "fhvpvqrf", "fhvpvqny", "xlf", "frysunez", "hanyvir", "hanyvirq", "hanyvivat", "guvafcb",
        "guvafcvengvba", "frys unez", "frys unezvat", "frys unezrq", "ceb nan",
        *_phrases(_KILL, _SELF),
        *_phrases(_END, _WHOSE, ("yvsr", "yvirf")),
        *_phrases(_TAKE, _WHOSE, ("bja yvsr", "bja yvirf")),
        *_phrases(_CUT, _WHOSE, ("jevfg", "jevfgf")),
    ),
    "gore": (  # graphic gore (cartoon slapstick is fine and not listed)
        "tber", "qvfrzobjry", "qvfrzobjryf", "qvfrzobjryrq", "qvfrzobjryyrq", "qvfrzobjryzrag", "rivfprengr",
        "rivfprengrq", "rivfprengvat", "rivfprengvba", "qrpncvgngr", "qrpncvgngrf", "qrpncvgngrq",
        "qrpncvgngvat", "qrpncvgngvba", "orurnq", "orurnqf", "orurnqrq", "orurnqvat", "qvfzrzore",
        "qvfzrzoref", "qvfzrzorerq", "qvfzrzorevat", "qvfzrzorezrag", "zhgvyngr", "zhgvyngrf", "zhgvyngrq",
        "zhgvyngvat", "zhgvyngvba", "ragenvyf", "oybbqongu", "oybbqfbnxrq", "oybbq ongu", "oybbq fbnxrq",
        "frirerq urnq", "frirerq urnqf", "frirerq yvzo", "frirerq yvzof", "frirerq nez", "frirerq nezf",
        "frirerq yrt", "frirerq yrtf", "frirerq unaq", "frirerq unaqf", "fcynggrerq oenvaf",
        "oenvaf fcynggrerq", "rlrf tbhtrq", "rlronyyf tbhtrq", "oyrq gb qrngu", "oyrrqvat gb qrngu",
        "oyrrq gb qrngu", "oyrrqf gb qrngu", "cbby bs oybbq", "vagrfgvarf fcvyyrq", "fcvyyrq vagrfgvarf",
        "punvafnj znffnper", "thfuvat oybbq", "oybbq thfuvat", "oybbq thfurq", "fchegvat oybbq",
        "oybbq fchegvat", "oybbq fchegrq",
        "oybbq rireljurer", "oyrq bhg", "oyrrq bhg", "oyrrqf bhg", "oyrrqvat bhg", "frirerq svatre",
        "frirerq svatref", "frirerq gbr", "frirerq gbrf", "frirerq sbbg", "frirerq srrg", "frirerq rne",
        "frirerq rnef", "frirerq gbathr", "oybbq fcenlrq", "oybbq fcynggrerq", "oybbq fcnggrerq", "fcynggrerq oybbq",
        "fcenlrq oybbq", "thgf fcvyyrq bhg", "thgf fcvyyvat bhg", "thgf fcvyy bhg",
    ),
    "drugs": (  # hard drugs
        "pbpnvar", "urebva", "penpxurnq", "penpxurnqf", "zrgu", "zrgunzcurgnzvar", "zrgunzcurgnzvarf",
        "nzcurgnzvar", "nzcurgnzvarf", "zqzn", "yfq", "xrgnzvar", "sragnaly", "bcvhz", "cfvybplova", "cpc",
        "serronfr", "serronfvat", "bklpbagva", "bklpbqbar", "penpx pbpnvar", "penpx cvcr", "pelfgny zrgu",
        "qeht qrnyre", "qeht qrnyref", "qeht qrnyvat", "fabeg pbxr", "fabegvat pbxr",
    ),
}

# Mild (and not so mild) swearing: not blocked, just masked by safety.soften()
# ("d***"), so the story keeps going.
MILD_PROFANITY: tuple[str, ...] = (
    "qnza", "qnzarq", "qnzavg", "qnzzvg", "tbqqnza", "tbqqnzarq", "tbqqnzzvg", "tbqnzzvg", "uryy", "uryyf",
    "penc", "penccl", "penccrq", "penccvat", "ohyypenc", "oybbql", "ohttre", "ohttref", "ohttrerq", "nefr",
    "nefrf", "nefrq", "nefrubyr", "nefrubyrf", "nff", "nffrf", "nffubyr", "nffubyrf", "wnpxnff", "wnpxnffrf",
    "qhzonff", "onqnff", "fznegnff", "onfgneq", "onfgneqf", "obyybpxf", "ohyyfuvg", "fuvg", "fuvgf", "fuvggl",
    "fuvgr", "fuvggvat", "fuvgurnq", "fuvgurnqf", "cvff", "cvffrq", "cvffrf", "cvffvat", "fbqqvat", "ovgpu",
    "ovgpurf", "ovgpul", "ovgpuvat", "shpx", "shpxf", "shpxrq", "shpxre", "shpxref", "shpxvat", "shpxva",
    "shpxjvg", "zbgureshpxre", "zbgureshpxref", "zbgureshpxvat", "jnaxre", "jnaxref", "gjng", "gjngf",
    "gbffre", "gbffref", "qvpxurnq", "qvpxurnqf", "xabournq", "qbhpuront", "qbhpurontf", "jgs", "fgsh",
)

# Innocent phrases that contain a listed word (a waterproof jacket, a tidy
# office...): removed before matching, so they never trip the filter.
EXEMPT_PHRASES: tuple[str, ...] = (
    "tber grk", "ny tber", "fcvp naq fcna", "fcvp a fcna", "fcvpx naq fcna",
    # "take my own life jacket", "my own life-sized cardboard cutout": a compound noun, not self-harm.
    "bja yvsr wnpxrg", "bja yvsr wnpxrgf", "bja yvsr ensg", "bja yvsr ensgf", "bja yvsr fvmr", "bja yvsr fvmrq", "bja yvsr fgbel",
    # The British name of a yellow field crop (not the crime); a car's rear-hinged doors.
    "bvyfrrq encr", "fhvpvqr qbbe", "fhvpvqr qbbef",
    # "Nearly killed himself laughing", "don't kill yourself rushing": figures of speech.
    *_phrases(_KILL, _SELF, ("ynhtuvat", "gelvat", "ehfuvat")),
)
