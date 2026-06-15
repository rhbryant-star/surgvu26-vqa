#!/usr/bin/env python3
# scripts/merge_qa_dataset.py
"""Merge per-case qa.jsonl shards into LLaMA-Factory sharegpt train/val files.

Output format matches the proven phase-detector recipe: messages with string
contents, 8 <image> placeholders in the user turn, and a parallel "images"
list of 8 RELATIVE frame paths (the training wrapper rewrites them to the
scratch dir after untarring the frame tars).

PROMPT-PARITY NOTE (revision item 10):
The user turn is built as::

    "<image>\\n" * len(rec["images"]) + build_user_text(question)

which puts 8 leading ``<image>`` lines before the question text.  This mirrors
model.py's inference path::

    content = [{"type": "image"} for _ in frames]        # 8 image dicts first
    content.append({"type": "text", "text": build_user_text(question)})

When ``apply_chat_template`` processes that list it renders images before the
text token — exactly the order LLaMA-Factory's ``qwen2_vl`` template expects
from the ``<image>`` placeholders.  ``tests/scripts/test_merge_qa_dataset.py``
(``test_user_turn_eight_leading_images_then_text``) enforces this contract.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from surgvu_vqa.predict.answer import SYSTEM_PROMPT, build_user_text


def _to_sharegpt(rec: dict) -> dict:
    """Convert a qa.jsonl shard record to a LLaMA-Factory sharegpt sample.

    The user content starts with one ``<image>`` placeholder per frame (8),
    each on its own line, followed by the question text produced by
    ``build_user_text``.  This layout is required by the LLaMA-Factory
    ``qwen2_vl`` template and must match the ``apply_chat_template`` rendering
    used in model.py (see module docstring for the parity contract).
    """
    placeholders = "\n".join(["<image>"] * len(rec["images"]))
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"{placeholders}\n{build_user_text(rec['question'])}",
            },
            {"role": "assistant", "content": rec["answer"]},
        ],
        "images": rec["images"],
    }


def merge(
    shards_dir: Path,
    out_dir: Path,
    train_cases: list[str],
    val_cases: list[str],
) -> tuple[int, int]:
    """Merge per-case qa.jsonl shards into LF train/val splits.

    Parameters
    ----------
    shards_dir:
        Directory containing ``{case}_qa.jsonl`` files produced by
        ``build_case_dataset.py``.
    out_dir:
        Output directory.  Receives ``train.jsonl``, ``val.jsonl``, and
        ``dataset_info.json``.
    train_cases:
        Case IDs whose shards go into the training split.
    val_cases:
        Case IDs whose shards go into the validation split.

    Returns
    -------
    (n_train, n_val)
        Number of QA records written to each split.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for split, cases in (("train", train_cases), ("val", val_cases)):
        n = 0
        with open(out_dir / f"{split}.jsonl", "w") as out_f:
            for case in cases:
                shard = Path(shards_dir) / f"{case}_qa.jsonl"
                if not shard.exists():
                    print(f"WARNING: missing shard {shard}")
                    continue
                for line in shard.read_text().splitlines():
                    if not line.strip():
                        continue
                    out_f.write(json.dumps(_to_sharegpt(json.loads(line))) + "\n")
                    n += 1
        counts[split] = n

    info = {
        f"surgvu_vqa_{split}": {
            "file_name": f"{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
        for split in ("train", "val")
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(f"train={counts['train']} val={counts['val']}")
    return counts["train"], counts["val"]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Merge per-case qa.jsonl shards into LLaMA-Factory sharegpt format."
    )
    p.add_argument("--shards-dir", type=Path, required=True,
                   help="Directory containing {case}_qa.jsonl shards.")
    p.add_argument("--out-dir", type=Path, required=True,
                   help="Output directory for train.jsonl, val.jsonl, dataset_info.json.")
    p.add_argument("--train-cases", type=Path, required=True,
                   help="File with one train case ID per line.")
    p.add_argument("--val-cases", type=Path, required=True,
                   help="File with one val case ID per line.")
    a = p.parse_args()
    merge(
        a.shards_dir,
        a.out_dir,
        a.train_cases.read_text().split(),
        a.val_cases.read_text().split(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
