import sys
import types
from pathlib import Path

from neurovc.contrib.flowmag_util import models as flow_models


def test_flowmag_downloader_uses_gdown(monkeypatch, tmp_path):
    calls = []

    def fake_download(url, output, quiet=False):
        calls.append({"url": url, "output": Path(output), "quiet": quiet})
        Path(output).write_bytes(b"weights")
        return output

    fake_gdown = types.SimpleNamespace(download=fake_download)
    monkeypatch.setitem(sys.modules, "gdown", fake_gdown)

    downloader = flow_models._ModelDownloader("raft", save_dir=tmp_path)
    model_path = downloader.download_model()

    expected_model_path = tmp_path / "raft_chkpt_00140.pth"

    assert model_path == expected_model_path
    assert expected_model_path.exists()

    assert len(calls) == 1
    call = calls[0]
    assert flow_models.FLOWMAG_MODEL_FILE_IDS["raft"] in call["url"]
    assert call["output"] == expected_model_path
    assert call["quiet"] is False


def test_flowmag_downloader_skips_existing_file(tmp_path):
    existing = tmp_path / "raft_chkpt_00140.pth"
    existing.write_bytes(b"preexisting")

    downloader = flow_models._ModelDownloader("raft", save_dir=tmp_path)
    model_path = downloader.download_model()

    assert model_path == existing
    assert model_path.read_bytes() == b"preexisting"
