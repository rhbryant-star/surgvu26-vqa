from surgvu_vqa.predict import model as model_mod
from surgvu_vqa.predict.model import HF_MODEL_ID, MODEL_DIR_ENV, resolve_model_path


def test_env_override_wins(monkeypatch, tmp_path):
    monkeypatch.setenv(MODEL_DIR_ENV, str(tmp_path))
    assert resolve_model_path() == str(tmp_path)


def test_tarball_dir_when_present(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    fake_tarball_dir = tmp_path / "qwen2.5-vl-7b-awq"
    fake_tarball_dir.mkdir()
    monkeypatch.setattr(model_mod, "TARBALL_MODEL_DIR", fake_tarball_dir)
    assert resolve_model_path() == str(fake_tarball_dir)


def test_falls_back_to_hub_id(monkeypatch, tmp_path):
    monkeypatch.delenv(MODEL_DIR_ENV, raising=False)
    monkeypatch.setattr(model_mod, "TARBALL_MODEL_DIR", tmp_path / "absent")
    assert resolve_model_path() == HF_MODEL_ID


def test_module_imports_without_torch():
    # Heavy deps must stay out of module import time: the dev box has no torch,
    # and importing surgvu_vqa.predict.model above must not have pulled it in.
    import sys
    assert "torch" not in sys.modules
