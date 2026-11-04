import pytest

from textsummarizer.metrics import ROUGE_KEYS, compute_rouge, prepare_for_rouge, split_sentences


def test_split_sentences_handles_newlines_and_terminal_punctuation():
    assert split_sentences("A b. C d!  E?\nF g") == ["A b.", "C d!", "E?", "F g"]


def test_prepare_for_rouge_turns_pegasus_newline_markers_into_sentence_breaks():
    assert prepare_for_rouge("One two.<n>Three four.") == "One two.\nThree four."


def test_identical_texts_score_100():
    text = ["Hannah needs Betty's number."]
    assert compute_rouge(text, text) == {key: 100.0 for key in ROUGE_KEYS}


def test_disjoint_texts_score_0():
    assert compute_rouge(["alpha beta"], ["gamma delta"]) == {key: 0.0 for key in ROUGE_KEYS}


def test_known_example_matches_hand_checked_scores():
    scores = compute_rouge(
        ["Amanda can't find the number.<n>Larry called Betty."],
        ["Amanda can't find Betty's number. Larry called her."],
    )
    expected = {"rouge1": 84.2, "rouge2": 58.8, "rougeL": 73.7, "rougeLsum": 84.2}
    assert scores == pytest.approx(expected, abs=0.1)


def test_mismatched_or_empty_inputs_are_rejected():
    with pytest.raises(ValueError):
        compute_rouge(["a"], [])
    with pytest.raises(ValueError):
        compute_rouge([], [])
