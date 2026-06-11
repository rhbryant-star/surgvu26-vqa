from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "container" / "test" / "input" / "interf0"


def _load_inference():
    spec = importlib.util.spec_from_file_location("inference", REPO / "container" / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_input_dir(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in ("inputs.json", "visual-context-question.json", "endoscopic-robotic-surgery-video.mp4"):
        shutil.copy(FIXTURE / name, input_dir / name)
    return input_dir


def test_fake_model_end_to_end(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    monkeypatch.setenv("SURGVU_INPUT_PATH", str(input_dir))
    monkeypatch.setenv("SURGVU_OUTPUT_PATH", str(output_dir))
    monkeypatch.setenv("SURGVU_FAKE_MODEL", "1")

    inference = _load_inference()
    rc = inference.run()

    assert rc == 0
    out = json.loads((output_dir / "visual-context-response.json").read_text())
    assert isinstance(out, str)
    assert out == "A fake answer for container testing."


def test_unknown_interface_raises(tmp_path, monkeypatch):
    input_dir = _make_input_dir(tmp_path)
    bogus = [{"interface": {"slug": "some-unknown-socket"}}]
    (input_dir / "inputs.json").write_text(json.dumps(bogus))
    monkeypatch.setenv("SURGVU_INPUT_PATH", str(input_dir))
    monkeypatch.setenv("SURGVU_OUTPUT_PATH", str(tmp_path / "out2"))
    monkeypatch.setenv("SURGVU_FAKE_MODEL", "1")

    inference = _load_inference()
    try:
        inference.run()
        raised = False
    except KeyError:
        raised = True
    assert raised
