from lebse.text import split_sentences


def test_splits_and_filters_by_length():
    text = (
        "The district court held that the waiver was invalid. "
        "It vacated the judgment below and remanded the case for further proceedings."
    )
    sents = split_sentences(text)
    assert len(sents) == 2
    assert all(40 <= len(s) <= 350 for s in sents)
    assert sents[0].startswith("The district court held")


def test_drops_too_short_and_nonalpha():
    assert split_sentences("Hi. Ok.") == []  # too short
    assert split_sentences("123456789012345678901234567890123456789012345 6789") == []  # no letters


def test_splits_across_newlines():
    text = (
        "First substantive sentence about jurisdiction and standing here.\n"
        "Second substantive sentence about due process and fair notice here."
    )
    assert len(split_sentences(text)) == 2


def test_deterministic():
    t = "A sufficiently long first legal sentence for the test. And a second equally long one here."
    assert split_sentences(t) == split_sentences(t)
