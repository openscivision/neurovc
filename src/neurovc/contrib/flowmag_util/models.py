"""FlowMag model download utilities."""

from pathlib import Path
from typing import Dict

from neurovc.contrib.flowmag import FLOWMAG_MODEL_FILE_IDS

_FILE_TARGETS: Dict[str, str] = {
    "raft": "raft_chkpt_00140.pth",
    "arflow": "arflow_chkpt_00140.pth",
}


class _ModelDownloader:
    def __init__(self, model_name: str, save_dir: Path | str | None = None):
        self.model_name = model_name
        self.file_id = FLOWMAG_MODEL_FILE_IDS.get(model_name)
        if self.file_id is None:
            raise ValueError(
                f"Model name '{model_name}' is not valid. "
                f"Available: {sorted(FLOWMAG_MODEL_FILE_IDS)}"
            )

        base_dir = (
            Path(save_dir)
            if save_dir is not None
            else Path("~/.neurovc/models/flowmag")
        )
        self.save_dir = base_dir.expanduser()
        target = _FILE_TARGETS.get(model_name, f"{model_name}.pth")
        self.model_path = self.save_dir / target

    def download_model(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return self.model_path

        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "gdown is required to download FlowMag checkpoints. "
                "Install it via 'pip install gdown'."
            ) from exc

        url = f"https://drive.google.com/uc?export=download&id={self.file_id}"
        gdown.download(url, str(self.model_path), quiet=False)
        return self.model_path


def download_flowmag_model(model_name: str, save_dir: Path | str | None = None) -> Path:
    """Download a single FlowMag checkpoint by model name and return its path."""
    return _ModelDownloader(model_name=model_name, save_dir=save_dir).download_model()


def download_all_flowmag_models(save_dir: Path | str | None = None) -> Dict[str, Path]:
    """Download all known FlowMag checkpoints and return their paths keyed by name."""
    return {
        name: download_flowmag_model(name, save_dir=save_dir)
        for name in FLOWMAG_MODEL_FILE_IDS
    }


# Pre-configured convenience instance mirroring tfake usage style
flowmag_model_downloader = _ModelDownloader

__all__ = [
    "_ModelDownloader",
    "download_flowmag_model",
    "download_all_flowmag_models",
    "flowmag_model_downloader",
]
