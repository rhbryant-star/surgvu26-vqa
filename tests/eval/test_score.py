import json

from surgvu_vqa.eval.score import main, mean_bleu, score_run

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


def test_cli_prints_mean(tmp_path, capsys):
    # 4+-token answer: 1-token fixtures cannot reach BLEU 1.0 (see fixture note in the plan).
    answer = "A stapler was not used."
    truth = {"clip_0": {"question": "Q?", "references": [answer]}}
    predictions = {"clip_0": answer}
    t = tmp_path / "truth.json"
    p = tmp_path / "pred.json"
    t.write_text(json.dumps(truth))
    p.write_text(json.dumps(predictions))

    rc = main(["--truth", str(t), "--predictions", str(p)])

    out = capsys.readouterr().out
    assert rc == 0
    assert "mean_bleu: 1.0000" in out
    assert "clip_0: 1.0000" in out
