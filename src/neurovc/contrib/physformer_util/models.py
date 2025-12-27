"""PhysFormer model download utilities."""

from pathlib import Path

PHYSFORMER_FILE_ID = "1jBSbM88fA-beaoVi8ILFyL0SvVVMA9c9"
PHYSFORMER_FILE_NAME = "Physformer_VIPL_fold1.pkl"


class _ModelDownloader:
    def __init__(
        self,
        file_id: str = PHYSFORMER_FILE_ID,
        file_name: str = PHYSFORMER_FILE_NAME,
        save_dir: Path | str | None = None,
    ):
        self.file_id = file_id
        self.file_name = file_name
        base_dir = Path(save_dir) if save_dir is not None else Path("~/.neurovc/rppg")
        self.save_dir = base_dir.expanduser()
        self.model_path = self.save_dir / self.file_name

    def download_model(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if self.model_path.exists():
            return self.model_path

        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "gdown is required to download PhysFormer checkpoints. "
                "Install it via 'pip install gdown'."
            ) from exc

        url = f"https://drive.google.com/uc?export=download&id={self.file_id}"
        gdown.download(url, str(self.model_path), quiet=False)
        return self.model_path


def download_physformer_model(save_dir: Path | str | None = None) -> Path:
    """Download the default PhysFormer checkpoint and return its path."""
    return _ModelDownloader(save_dir=save_dir).download_model()


# Pre-configured convenience instance mirroring FlowMag util style
physformer_model_downloader = _ModelDownloader

__all__ = [
    "_ModelDownloader",
    "download_physformer_model",
    "physformer_model_downloader",
    "PHYSFORMER_FILE_ID",
    "PHYSFORMER_FILE_NAME",
]
