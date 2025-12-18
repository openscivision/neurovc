import sys
import types
from pathlib import Path

from neurovc.thermal_landmarks import tfw_util


def test_model_downloader_uses_gdown(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, output, quiet=False):
        calls.append({"url": url, "output": Path(output), "quiet": quiet})
        Path(output).write_bytes(b"weights")
        return output

    fake_gdown = types.ModuleType("gdown")
    fake_gdown.download = fake_download
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    downloader = tfw_util._ModelDownloader("YOLOv5n-Face.modern", save_dir=tmp_path)
    model_path = downloader.download_model()

    expected_model_path = tmp_path / tfw_util._file_targets["YOLOv5n-Face.modern"]
    expected_license_path = tmp_path / tfw_util._file_targets["license"]

    assert model_path == expected_model_path
    assert model_path.exists()
    assert expected_license_path.exists()

    assert len(calls) == 2
    model_call = next(call for call in calls if call["output"] == expected_model_path)
    license_call = next(
        call for call in calls if call["output"] == expected_license_path
    )

    assert tfw_util._file_id_map["YOLOv5n-Face.modern"] in model_call["url"]
    assert tfw_util._file_id_map["license"] in license_call["url"]


def test_model_downloader_skips_existing_file(monkeypatch, tmp_path):
    existing = tmp_path / tfw_util._file_targets["YOLOv5n-Face.modern"]
    existing.write_bytes(b"preexisting")
    existing_license = tmp_path / tfw_util._file_targets["license"]
    existing_license.write_text("preexisting license", encoding="utf-8")

    def fail_download(*args, **kwargs):  # pragma: no cover - would signal a bug
        raise AssertionError("gdown should not be called when file exists")

    fake_gdown = types.ModuleType("gdown")
    fake_gdown.download = fail_download
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    downloader = tfw_util._ModelDownloader("YOLOv5n-Face.modern", save_dir=tmp_path)
    model_path = downloader.download_model()

    assert model_path == existing
    assert model_path.read_bytes() == b"preexisting"
    assert existing_license.read_text(encoding="utf-8") == "preexisting license"


def test_tfw_landmarker_init_loads_model(monkeypatch, tmp_path):
    model_path = tmp_path / "dummy.pt"
    model_path.write_bytes(b"stub")

    calls = {}

    def fake_prepare_model(model_name="YOLOv5n-Face.modern"):
        calls["prepare_model"] = model_name
        return model_path

    monkeypatch.setattr(tfw_util, "_prepare_model", fake_prepare_model)

    class DummyCuda:
        @staticmethod
        def is_available():
            return False

    dummy_torch = types.SimpleNamespace(device=lambda value: value, cuda=DummyCuda)
    monkeypatch.setattr(tfw_util, "torch", dummy_torch, raising=False)
    monkeypatch.setattr(tfw_util, "HAS_TORCH", True, raising=False)

    def fake_load_model(path, device):
        calls["load_model"] = {"path": Path(path), "device": device}
        return "MODEL"

    dummy_yf = types.SimpleNamespace(load_model=fake_load_model)
    monkeypatch.setattr(tfw_util, "yf", dummy_yf, raising=False)
    monkeypatch.setattr(tfw_util, "HAS_YOLOV5_FACE", True, raising=False)

    landmarker = tfw_util.TFWLandmarker(model_name="YOLOv5n-Face.modern")

    assert calls["prepare_model"] == "YOLOv5n-Face.modern"
    assert calls["load_model"]["path"] == model_path
    assert calls["load_model"]["device"] == "cpu"
    assert landmarker.device == "cpu"
    assert landmarker.model == "MODEL"
