"""Tests for gettowork.reasoning.split_reasoning."""

from __future__ import annotations

import pytest

from gettowork.reasoning import split_reasoning


# ---------------------------------------------------------------------------
# No reasoning at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text, answer",
    [
        ("Just an answer.", "Just an answer."),
        ("  padded answer \n\n", "padded answer"),
        ("", ""),
        ("   ", ""),
        ("Line one\n\n\n\nLine four keeps its spacing", "Line one\n\n\n\nLine four keeps its spacing"),
        ("I think therefore I am", "I think therefore I am"),
        ("Use <b>bold</b> tags freely", "Use <b>bold</b> tags freely"),
        ("a < b and c > d", "a < b and c > d"),
    ],
)
def test_plain_text_has_no_reasoning(text, answer):
    assert split_reasoning(text) == (answer, None)


def test_none_input_is_treated_as_empty():
    assert split_reasoning(None) == ("", None)


def test_non_string_input_is_coerced():
    assert split_reasoning(42) == ("42", None)


# ---------------------------------------------------------------------------
# Closed tag blocks
# ---------------------------------------------------------------------------


def test_basic_think_block():
    assert split_reasoning("<think>The goose looks angry.</think>Offer it bread.") == (
        "Offer it bread.",
        "The goose looks angry.",
    )


def test_think_block_with_surrounding_newlines():
    text = "<think>\nLet me consider the octopus.\nIt is nervous.\n</think>\n\nCALM THE OCTOPUS."
    assert split_reasoning(text) == (
        "CALM THE OCTOPUS.",
        "Let me consider the octopus.\nIt is nervous.",
    )


@pytest.mark.parametrize("tag", ["think", "thinking", "reasoning", "THINK", "Thinking", "ReAsOnInG"])
def test_all_supported_tag_names_case_insensitive(tag):
    text = f"<{tag}>hidden thoughts</{tag}>visible answer"
    assert split_reasoning(text) == ("visible answer", "hidden thoughts")


def test_mismatched_but_related_tag_names_still_pair_up():
    assert split_reasoning("<thinking>hmm</think>answer") == ("answer", "hmm")


def test_tag_with_attributes():
    assert split_reasoning('<think type="deep">ponder</think>done') == ("done", "ponder")


def test_closing_tag_with_trailing_space():
    assert split_reasoning("<think>ponder</think >done") == ("done", "ponder")


def test_multiple_blocks_joined_in_order():
    text = "<think>first</think>Part A.\n<think>second</think>\nPart B."
    answer, reasoning = split_reasoning(text)
    assert answer == "Part A.\n\nPart B."
    assert reasoning == "first\n\nsecond"


def test_mixed_tag_families_in_one_text():
    text = "<reasoning>one</reasoning>A <thinking>two</thinking>B"
    assert split_reasoning(text) == ("A B", "one\n\ntwo")


def test_block_in_the_middle_of_a_sentence_keeps_words_apart():
    assert split_reasoning("Take the<think>bus?</think>bus.") == ("Take the bus.", "bus?")


def test_block_between_spaces_leaves_single_gap():
    assert split_reasoning("Take the <think>x</think> bus.") == ("Take the bus.", "x")


def test_empty_think_block_gives_none_reasoning():
    # Qwen3 with /no_think emits an empty block.
    assert split_reasoning("<think>\n\n</think>\n\nHello there.") == ("Hello there.", None)


def test_reasoning_only_closed_block_gives_empty_answer():
    assert split_reasoning("<think>I ran out of words.</think>") == ("", "I ran out of words.")


def test_blank_lines_left_behind_are_collapsed():
    text = "Intro line.\n\n<think>x</think>\n\n\nNext line."
    answer, reasoning = split_reasoning(text)
    assert answer == "Intro line.\n\nNext line."
    assert reasoning == "x"


def test_answer_json_survives():
    text = '<think>Is this progress? Yes.</think>{"made_progress": true, "explanation": "Geese love bread."}'
    answer, reasoning = split_reasoning(text)
    assert answer == '{"made_progress": true, "explanation": "Geese love bread."}'
    assert reasoning == "Is this progress? Yes."


# ---------------------------------------------------------------------------
# Unclosed opener (budget ran out mid-thought)
# ---------------------------------------------------------------------------


def test_unclosed_leading_think_is_all_reasoning():
    assert split_reasoning("<think>I was thinking and then the tokens ran o") == (
        "",
        "I was thinking and then the tokens ran o",
    )


def test_unclosed_think_after_text_keeps_the_prefix_as_answer():
    assert split_reasoning("Hello!\n<think>and then") == ("Hello!", "and then")


def test_unclosed_opener_after_a_closed_block():
    answer, reasoning = split_reasoning("<think>one</think>Answer. <think>two, never finish")
    assert answer == "Answer."
    assert reasoning == "one\n\ntwo, never finish"


def test_bare_unclosed_opener_with_nothing_after():
    assert split_reasoning("<think>") == ("", None)


@pytest.mark.parametrize("opener", ["<thinking>", "<reasoning>", "<THINK>"])
def test_unclosed_other_openers(opener):
    assert split_reasoning(f"{opener}still going") == ("", "still going")


# ---------------------------------------------------------------------------
# Stray closer (template opened the block inside the prompt)
# ---------------------------------------------------------------------------


def test_stray_closer_text_before_is_reasoning():
    assert split_reasoning("The goose union is strong.\n</think>\n\nBring a tiny picket sign.") == (
        "Bring a tiny picket sign.",
        "The goose union is strong.",
    )


def test_stray_closer_at_very_start():
    assert split_reasoning("</think>Answer only") == ("Answer only", None)


def test_stray_closer_with_later_closed_block():
    text = "early thoughts</think>Answer part.<think>more thoughts</think> Done."
    answer, reasoning = split_reasoning(text)
    assert answer == "Answer part. Done."
    assert reasoning == "early thoughts\n\nmore thoughts"


def test_multiple_stray_closers_use_the_last_one():
    answer, reasoning = split_reasoning("a</think>b</think>c")
    assert answer == "c"
    assert "a" in reasoning and "b" in reasoning


def test_stray_closer_then_unclosed_opener():
    answer, reasoning = split_reasoning("thought one</think>Visible<think>thought two")
    assert answer == "Visible"
    assert reasoning == "thought one\n\nthought two"


@pytest.mark.parametrize("closer", ["</thinking>", "</reasoning>", "</THINK>"])
def test_other_stray_closers(closer):
    assert split_reasoning(f"pondering{closer}answer") == ("answer", "pondering")


# ---------------------------------------------------------------------------
# Magistral-style [THINK]
# ---------------------------------------------------------------------------


def test_bracket_think_block():
    assert split_reasoning("[THINK]Octopi have eight arms.[/THINK]Hold on tight.") == (
        "Hold on tight.",
        "Octopi have eight arms.",
    )


def test_bracket_think_unclosed():
    assert split_reasoning("[THINK]so many arms") == ("", "so many arms")


def test_bracket_think_stray_closer():
    assert split_reasoning("so many arms[/THINK]Hold on.") == ("Hold on.", "so many arms")


# ---------------------------------------------------------------------------
# Harmony (gpt-oss)
# ---------------------------------------------------------------------------


def test_harmony_analysis_then_final():
    text = (
        "<|channel|>analysis<|message|>User wants to cross a moat of pudding.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>You row across in a teacup.<|return|>"
    )
    assert split_reasoning(text) == ("You row across in a teacup.", "User wants to cross a moat of pudding.")


def test_harmony_with_leading_start_token():
    text = (
        "<|start|>assistant<|channel|>analysis<|message|>Think.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Answer.<|end|>"
    )
    assert split_reasoning(text) == ("Answer.", "Think.")


def test_harmony_final_without_terminator():
    text = "<|channel|>analysis<|message|>Hmm<|end|><|start|>assistant<|channel|>final<|message|>Go by kite."
    assert split_reasoning(text) == ("Go by kite.", "Hmm")


def test_harmony_analysis_only_gives_empty_answer():
    text = "<|channel|>analysis<|message|>Still analysing when the budget ran ou"
    assert split_reasoning(text) == ("", "Still analysing when the budget ran ou")


def test_harmony_analysis_closed_without_final():
    text = "<|channel|>analysis<|message|>Thinking done.<|end|><|start|>assistant"
    assert split_reasoning(text) == ("", "Thinking done.")


def test_harmony_multiple_analysis_segments_and_commentary():
    text = (
        "<|channel|>analysis<|message|>A<|end|>"
        "<|start|>assistant<|channel|>commentary to=functions.lookup <|constrain|>json<|message|>{\"q\":1}<|call|>"
        "<|start|>assistant<|channel|>analysis<|message|>B<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Final answer.<|return|>"
    )
    answer, reasoning = split_reasoning(text)
    assert answer == "Final answer."
    assert reasoning == 'A\n\n{"q":1}\n\nB'


def test_harmony_final_only():
    assert split_reasoning("<|channel|>final<|message|>Only the answer.<|return|>") == ("Only the answer.", None)


def test_harmony_message_without_channel_strips_tokens():
    assert split_reasoning("<|start|>assistant<|message|>Hello<|end|>") == ("Hello", None)


def test_harmony_final_containing_think_tags():
    text = "<|channel|>analysis<|message|>outer<|end|><|channel|>final<|message|><think>inner</think>Answer"
    assert split_reasoning(text) == ("Answer", "outer\n\ninner")


def test_harmony_json_answer():
    text = '<|channel|>analysis<|message|>judge it<|end|><|channel|>final<|message|>{"made_progress": false}'
    assert split_reasoning(text) == ('{"made_progress": false}', "judge it")


def test_harmony_flattened_when_special_tokens_are_dropped():
    text = "analysisThe user needs a plan.assistantfinalRide the escalator backwards."
    assert split_reasoning(text) == ("Ride the escalator backwards.", "The user needs a plan.")


def test_word_analysis_alone_is_not_mistaken_for_harmony():
    text = "analysis of the situation: the bus is a sandwich."
    assert split_reasoning(text) == (text, None)


# ---------------------------------------------------------------------------
# Return-type guarantees
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "plain",
        "<think>x</think>y",
        "<think>x",
        "x</think>y",
        "<|channel|>analysis<|message|>x<|end|><|channel|>final<|message|>y",
        "",
    ],
)
def test_answer_is_always_stripped_str_and_reasoning_is_none_or_nonempty(text):
    answer, reasoning = split_reasoning(text)
    assert isinstance(answer, str)
    assert answer == answer.strip()
    assert reasoning is None or (isinstance(reasoning, str) and reasoning.strip() == reasoning and reasoning)
