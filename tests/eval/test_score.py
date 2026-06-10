from surgvu_vqa.eval.score import mean_bleu, score_run

_A = "A stapler was not used."
_B = "The suturing step is shown."


def test_mean_bleu_averages_questions():
    items = [
        {"prediction": _A, "references": [_A]},
        {"prediction": _B, "references": [_B]},
    ]
    assert mean_bleu(items) == 1.0


def test_mean_bleu_empty_is_zero():
    assert mean_bleu([]) == 0.0


def test_score_run_handles_missing_prediction():
    truth = {
        "clip_0": {"question": "Q?", "references": [_A]},
        "clip_1": {"question": "Q?", "references": [_B]},
    }
    predictions = {"clip_0": _A}  # clip_1 deliberately missing
    result = score_run(truth, predictions)
    assert result["per_question"]["clip_0"] == 1.0
    assert result["per_question"]["clip_1"] == 0.0
    assert result["mean_bleu"] == 0.5
