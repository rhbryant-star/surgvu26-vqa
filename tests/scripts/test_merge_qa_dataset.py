from __future__ import annotations

import json
from pathlib import Path

from scripts.merge_qa_dataset import merge
from surgvu_vqa.predict.answer import SYSTEM_PROMPT, build_user_text


def _shard(d: Path, case: str, n: int):
    recs = [
        {
            "case": case,
            "kind": "tool_yes",
            "question": f"Q{i}?",
            "answer": f"Yes, answer {i} is here.",
            "images": [f"frames/{case}/clip{i:04d}_f{j}.jpg" for j in range(8)],
        }
        for i in range(n)
    ]
    (d / f"{case}_qa.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_merge_emits_sharegpt_multiimage(tmp_path):
    _shard(tmp_path, "case_001", 3)
    _shard(tmp_path, "case_000", 2)
    out = tmp_path / "lf"
    n_train, n_val = merge(tmp_path, out, train_cases=["case_001"], val_cases=["case_000"])
    assert (n_train, n_val) == (3, 2)
    rec = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert rec["messages"][0]["role"] == "system"
    user = rec["messages"][1]
    assert user["role"] == "user" and user["content"].count("<image>") == 8
    assert rec["messages"][2]["role"] == "assistant"
    assert len(rec["images"]) == 8
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["surgvu_vqa_train"]["columns"]["images"] == "images"


def test_system_prompt_matches_inference(tmp_path):
    """PROMPT-PARITY (revision item 10): the system message content in the
    sharegpt record must be the same string as SYSTEM_PROMPT used by model.py's
    apply_chat_template path — training and inference must share the same prompt.
    """
    _shard(tmp_path, "case_001", 1)
    out = tmp_path / "lf"
    merge(tmp_path, out, train_cases=["case_001"], val_cases=[])
    rec = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert rec["messages"][0]["content"] == SYSTEM_PROMPT


def test_user_turn_eight_leading_images_then_text(tmp_path):
    """PROMPT-PARITY (revision item 10): the user turn must start with exactly
    8 <image> placeholders (one per line) before the question text.  This
    mirrors model.py's inference content list which puts 8 {"type":"image"}
    dicts BEFORE the {"type":"text"} dict — LLaMA-Factory's qwen2_vl template
    and apply_chat_template both render leading images in the same order.
    """
    question = "Was a needle driver used during this clip?"
    shard_rec = {
        "case": "case_001",
        "kind": "tool_yes",
        "question": question,
        "answer": "Yes, a needle driver was used.",
        "images": [f"frames/case_001/clip0000_f{j}.jpg" for j in range(8)],
    }
    (tmp_path / "case_001_qa.jsonl").write_text(json.dumps(shard_rec) + "\n")
    out = tmp_path / "lf"
    merge(tmp_path, out, train_cases=["case_001"], val_cases=[])
    rec = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    user_content = rec["messages"][1]["content"]
    lines = user_content.split("\n")
    # First 8 lines must each be exactly "<image>"
    assert lines[:8] == ["<image>"] * 8, (
        f"Expected 8 leading '<image>' lines but got: {lines[:8]}"
    )
    # The question text (via build_user_text) must follow after the images
    expected_text = build_user_text(question)
    remainder = "\n".join(lines[8:]).lstrip("\n")
    assert remainder == expected_text, (
        f"User text after images does not match build_user_text.\n"
        f"Expected:\n{expected_text!r}\nGot:\n{remainder!r}"
    )


def test_assistant_content_is_answer(tmp_path):
    _shard(tmp_path, "case_001", 2)
    out = tmp_path / "lf"
    merge(tmp_path, out, train_cases=["case_001"], val_cases=[])
    for line in (out / "train.jsonl").read_text().splitlines():
        rec = json.loads(line)
        asst = rec["messages"][2]
        assert asst["role"] == "assistant"
        assert asst["content"].startswith("Yes, answer")


def test_missing_shard_is_skipped(tmp_path, capsys):
    """A case whose shard file doesn't exist should emit a warning and be skipped."""
    _shard(tmp_path, "case_001", 2)
    out = tmp_path / "lf"
    # case_999 has no shard file
    n_train, n_val = merge(
        tmp_path, out, train_cases=["case_001", "case_999"], val_cases=[]
    )
    assert n_train == 2
    captured = capsys.readouterr()
    assert "case_999" in captured.out


def test_val_split_written_separately(tmp_path):
    _shard(tmp_path, "case_001", 3)
    _shard(tmp_path, "case_000", 2)
    out = tmp_path / "lf"
    merge(tmp_path, out, train_cases=["case_001"], val_cases=["case_000"])
    val_lines = (out / "val.jsonl").read_text().splitlines()
    assert len(val_lines) == 2
    for line in val_lines:
        rec = json.loads(line)
        assert rec["messages"][1]["content"].count("<image>") == 8
        assert len(rec["images"]) == 8


def test_dataset_info_has_both_splits(tmp_path):
    _shard(tmp_path, "case_001", 1)
    _shard(tmp_path, "case_000", 1)
    out = tmp_path / "lf"
    merge(tmp_path, out, train_cases=["case_001"], val_cases=["case_000"])
    info = json.loads((out / "dataset_info.json").read_text())
    assert "surgvu_vqa_train" in info
    assert "surgvu_vqa_val" in info
    for split in ("surgvu_vqa_train", "surgvu_vqa_val"):
        assert info[split]["formatting"] == "sharegpt"
        cols = info[split]["columns"]
        assert cols["messages"] == "messages"
        assert cols["images"] == "images"
        tags = info[split]["tags"]
        assert tags["role_tag"] == "role"
        assert tags["content_tag"] == "content"
        assert tags["user_tag"] == "user"
        assert tags["assistant_tag"] == "assistant"
        assert tags["system_tag"] == "system"
