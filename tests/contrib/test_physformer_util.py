import sys
import types
from pathlib import Path

from neurovc.contrib import physformer_util


def test_physformer_downloader_uses_gdown(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, output, quiet=False):
        calls.append({"url": url, "output": Path(output), "quiet": quiet})
        Path(output).write_bytes(b"weights")
        return output

    fake_gdown = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    downloader = physformer_util.physformer_model_downloader(save_dir=tmp_path)
    model_path = downloader.download_model()

    expected_model_path = tmp_path / physformer_util.PHYSFORMER_FILE_NAME

    assert model_path == expected_model_path
    assert expected_model_path.exists()

    assert len(calls) == 1
    call = calls[0]
    assert physformer_util.PHYSFORMER_FILE_ID in call["url"]
    assert call["output"] == expected_model_path
    assert call["quiet"] is False


def test_physformer_downloader_skips_existing_file(tmp_path):
    existing = tmp_path / physformer_util.PHYSFORMER_FILE_NAME
    existing.write_bytes(b"preexisting")

    downloader = physformer_util.physformer_model_downloader(save_dir=tmp_path)
    model_path = downloader.download_model()

    assert model_path == existing
    assert model_path.read_bytes() == b"preexisting"
