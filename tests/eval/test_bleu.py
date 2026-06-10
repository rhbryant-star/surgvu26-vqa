from surgvu_vqa.eval.bleu import question_bleu, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("A Large  Needle") == ["a", "large", "needle"]


def test_exact_match_scores_one():
    score = question_bleu(
        "A large needle driver was not used.",
        ["A large needle driver was not used."],
    )
    assert score == 1.0


def test_max_is_taken_over_references():
    refs = [
        "No",
        "No, a large needle driver was not used",
        "A large needle driver was not used.",
    ]
    score = question_bleu("No", refs)
    assert 0.0 < score <= 1.0
    # Choosing the best reference must beat scoring against only the long one.
    long_only = question_bleu("No", ["A large needle driver was not used."])
    assert score >= long_only


def test_empty_prediction_scores_zero():
    assert question_bleu("", ["No"]) == 0.0


def test_no_references_scores_zero():
    assert question_bleu("No", []) == 0.0
