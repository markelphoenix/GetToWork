"""Separate a model's exposed reasoning ("chain-of-thought") from its answer.

Many open-weight models "think out loud" before answering. How that thinking
shows up in the raw text depends on the model family:

* Qwen3, DeepSeek-R1, SmolLM3 and friends wrap it in ``<think> ... </think>``
  (some fine-tunes use ``<thinking>`` or ``<reasoning>`` instead).
* Chat templates sometimes put the opening ``<think>`` in the *prompt*, so the
  model's output starts mid-thought and only contains the closing ``</think>``.
* If the token budget runs out while the model is still thinking, the closing
  tag never arrives: everything after ``<think>`` is reasoning, and there is no
  answer at all.
* OpenAI's gpt-oss uses the "harmony" format, with named channels:
  ``<|channel|>analysis<|message|>...<|end|>`` for thinking and
  ``<|channel|>final<|message|>...`` for the answer.
  If a runtime strips those special tokens, what's left looks like
  ``analysis...assistantfinal...``.
* Mistral's Magistral uses ``[THINK] ... [/THINK]``.
* ByteDance's Seed-OSS uses ``<seed:think> ... </seed:think>``.

`split_reasoning` handles all of these so the game can show the player a clean
answer, and keep the reasoning for the end-of-game review.
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["split_reasoning"]

# The tag names we treat as "thinking". Matching is case-insensitive.
_TAG_NAMES = r"(?:seed:think|think|thinking|reasoning)"
_OPEN_TAG = rf"<{_TAG_NAMES}(?:\s[^>]*)?>"
_CLOSE_TAG = rf"</{_TAG_NAMES}\s*>"

# A complete block: opener, lazily-matched body, closer. We accept a closer of a
# different name from the same family (e.g. <thinking>...</think>) because small
# models are not always tidy.
_BLOCK_RE = re.compile(rf"{_OPEN_TAG}(.*?){_CLOSE_TAG}", re.IGNORECASE | re.DOTALL)
_BRACKET_BLOCK_RE = re.compile(r"\[THINK\](.*?)\[/THINK\]", re.IGNORECASE | re.DOTALL)
_OPEN_RE = re.compile(rf"{_OPEN_TAG}|\[THINK\]", re.IGNORECASE)
_CLOSE_RE = re.compile(rf"{_CLOSE_TAG}|\[/THINK\]", re.IGNORECASE)

# Harmony (gpt-oss): "<|channel|>NAME[ optional header]<|message|>BODY" where BODY
# runs until the next special token or the end of the text.
_HARMONY_SEGMENT_RE = re.compile(
    r"<\|channel\|>\s*(?P<channel>[A-Za-z_]+)[^<]*?(?:<\|constrain\|>[^<]*)?<\|message\|>"
    r"(?P<body>.*?)"
    r"(?=<\|(?:end|return|call|start|channel)\|>|\Z)",
    re.DOTALL,
)
# Any leftover harmony control token, plus the role name that follows <|start|>.
_HARMONY_TOKEN_RE = re.compile(r"<\|start\|>\s*\w*|<\|[a-z_]+\|>")
_HARMONY_REASONING_CHANNELS = {"analysis", "commentary"}
# When a runtime drops harmony's special tokens while detokenizing, gpt-oss
# output degrades to "analysis<thinking>assistantfinal<answer>".
_HARMONY_FLAT_RE = re.compile(r"^\s*analysis(?P<reasoning>.*?)assistantfinal(?P<answer>.*)$", re.DOTALL)


def split_reasoning(text: Optional[str]) -> tuple[str, Optional[str]]:
    """Split raw model output into ``(answer, reasoning)``.

    ``answer`` is always a stripped string (possibly empty, e.g. when the model
    spent its whole token budget thinking). ``reasoning`` is the exposed
    chain-of-thought, or ``None`` if the model didn't show any.

    >>> split_reasoning("<think>The goose looks angry.</think>Offer it bread.")
    ('Offer it bread.', 'The goose looks angry.')
    >>> split_reasoning("Just an answer.")
    ('Just an answer.', None)
    """
    if text is None:
        return "", None
    if not isinstance(text, str):
        text = str(text)

    if "<|channel|>" in text or "<|message|>" in text:
        return _split_harmony(text)
    flat = _HARMONY_FLAT_RE.match(text)
    if flat:
        answer, extra = _split_tags(flat.group("answer"))
        return answer, _join_reasoning([flat.group("reasoning"), extra or ""])
    return _split_tags(text)


# ---------------------------------------------------------------------------
# Tag-style reasoning: <think>, <thinking>, <reasoning>, [THINK]
# ---------------------------------------------------------------------------


def _split_tags(text: str) -> tuple[str, Optional[str]]:
    reasoning_parts: list[str] = []
    changed = False

    # 1. Remove every complete block, remembering its contents in order.
    def _take(match: re.Match[str]) -> str:
        reasoning_parts.append(match.group(1))
        return _joiner(match)

    for pattern in (_BLOCK_RE, _BRACKET_BLOCK_RE):
        text, n = pattern.subn(_take, text)
        changed = changed or n > 0

    # 2. A closing tag with no opener: the chat template opened the block in the
    #    prompt, so everything before the (last) stray closer is reasoning.
    closers = list(_CLOSE_RE.finditer(text))
    if closers:
        last = closers[-1]
        reasoning_parts.insert(0, text[: last.start()])
        text = text[last.end():]
        changed = True

    # 3. An opener with no closer: the model ran out of tokens mid-thought, so
    #    everything after the opener is reasoning (and the answer may be empty).
    opener = _OPEN_RE.search(text)
    if opener:
        reasoning_parts.append(text[opener.end():])
        text = text[: opener.start()]
        changed = True

    answer = _tidy(text) if changed else text.strip()
    return answer, _join_reasoning(reasoning_parts)


def _joiner(match: re.Match[str]) -> str:
    """What to leave behind when a block is cut out of the middle of a sentence.

    "Take the<think>...</think>bus" should become "Take the bus", while a block
    that already sits between whitespace (or at either end) leaves nothing.
    """
    s = match.string
    before = s[match.start() - 1] if match.start() > 0 else " "
    after = s[match.end()] if match.end() < len(s) else " "
    return " " if not before.isspace() and not after.isspace() else ""


# ---------------------------------------------------------------------------
# Harmony (gpt-oss) channels
# ---------------------------------------------------------------------------


def _split_harmony(text: str) -> tuple[str, Optional[str]]:
    reasoning_parts: list[str] = []
    final_parts: list[str] = []
    leftovers: list[str] = []
    pos = 0
    for match in _HARMONY_SEGMENT_RE.finditer(text):
        leftovers.append(text[pos: match.start()])
        pos = match.end()
        channel = match.group("channel").lower()
        body = match.group("body")
        if channel in _HARMONY_REASONING_CHANNELS:
            reasoning_parts.append(body)
        else:  # "final" (or an unknown channel: better shown than hidden)
            final_parts.append(body)
    leftovers.append(text[pos:])

    if final_parts:
        answer_text = "\n\n".join(p.strip() for p in final_parts if p.strip())
    else:
        # No final channel: whatever sits outside the segments is the answer
        # (often nothing, when the model ran out of tokens while analysing).
        answer_text = "".join(leftovers)
    answer_text = _HARMONY_TOKEN_RE.sub("", answer_text)

    # The final text might still contain tag-style thinking; handle that too.
    answer, tag_reasoning = _split_tags(answer_text)
    if tag_reasoning:
        reasoning_parts.append(tag_reasoning)
    return _tidy(answer), _join_reasoning(reasoning_parts)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _join_reasoning(parts: list[str]) -> Optional[str]:
    cleaned = [p.strip() for p in parts if p and p.strip()]
    return "\n\n".join(cleaned) if cleaned else None


def _tidy(text: str) -> str:
    """Strip, and collapse the blank-line gaps left where blocks were removed."""
    text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
