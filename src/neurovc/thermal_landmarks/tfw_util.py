from pathlib import Path

import cv2
import numpy as np

from neurovc.util import normalize_color

try:
    import torch

    HAS_TORCH = True
except ModuleNotFoundError:
    torch = None
    HAS_TORCH = False

try:
    import yolov5_face.detect_face as yf

    HAS_YOLOV5_FACE = True
except ModuleNotFoundError:
    yf = None
    HAS_YOLOV5_FACE = False


class _ModelDownloader:
    def __init__(
        self,
        model_name: str = "YOLOv5n-Face.modern",
        save_dir: Path | str | None = None,
    ):
        self.model_name = model_name
        if self.model_name not in _file_id_map:
            available = sorted(k for k in _file_id_map if k != "license")
            raise ValueError(
                f"Model name '{model_name}' is not valid. Available: {available}"
            )

        base_dir = (
            Path(save_dir) if save_dir is not None else Path("~/.neurovc/models/tfw")
        )
        self.save_dir = base_dir.expanduser()
        self.model_path = self.save_dir / _file_targets.get(
            model_name, f"{model_name}.pt"
        )
        try:
            self.license_id = _file_id_map["license"]
        except KeyError as exc:
            raise ValueError("License file id is missing from _file_id_map.") from exc

    @staticmethod
    def _download_file(file_id: str, output_path: Path) -> Path:
        if output_path.exists():
            return output_path

        try:
            import gdown
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "gdown is required to download TFW checkpoints. "
                "Install it via 'pip install gdown'."
            ) from exc

        url = f"https://drive.google.com/uc?export=download&id={file_id}"
        gdown.download(url, str(output_path), quiet=False)
        return output_path

    def download_model(self) -> Path:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        model_id = _file_id_map[self.model_name]
        model_path = self._download_file(model_id, self.model_path)
        self._download_file(self.license_id, self.save_dir / _file_targets["license"])
        return model_path


def _prepare_model(model_name: str = "YOLOv5n-Face.modern") -> Path:
    downloader = _ModelDownloader(model_name)
    return downloader.download_model()


class LandmarkWrapper:
    def process(self, img):
        return self.get_landmarks(img), None


class TFWLandmarker(LandmarkWrapper):
    def __init__(self, model_name="YOLOv5n-Face.modern"):
        if not HAS_TORCH:
            raise ImportError(
                "torch is required for TFWLandmarker. Install the torch extra: 'pip install neurovc[torch]'"
            )
        if not HAS_YOLOV5_FACE:
            raise ImportError(
                "yolov5-face is required for TFWLandmarker. Install the landmark extra: 'pip install neurovc[landmark]'"
            )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = _prepare_model(model_name)
        self.model = yf.load_model(model_path, self.device)

    def detect(self, img):
        img = normalize_color(img, color_map=cv2.COLORMAP_BONE)
        results = yf.detect_landmarks(self.model, img, self.device)
        return results

    def get_landmarks(self, img):
        results = self.detect(img)
        if len(results) == 0:
            return np.full((5, 2), -1)
        lm = results[0]["landmarks"]
        lm = np.array(lm).reshape((-1, 2))
        return lm


_file_id_map = {
    # "YOLOv5n": "1PLUq7WbOWS7Ve2VKW7_WBkC3Uksje8Fx",
    # "YOLOv5n6": "1wV9t5uH_eiy7WaHdQdWnbeEIijuDAdKI",
    # "YOLOv5s": "1IdsdR1-qUeRo5EKQJzGQmRDi2SrMXJG5",
    # "YOLOv5s6": "1YZX3t7cSPnWWoic7oJo86ljBQgE5PPb2",
    # "YOLOv5n-Face": "1vXk9P3CfhUtRBGI44SqWbuiTJ7rAI4hP",
    "YOLOv5n-Face.modern": "14MpWOt-LEWM1w1XxMCngXKAYbt1toqrA",
    "license": "13jydQUIgVjK4XDdPXhVweDv5t1_n6iJy",
}

_file_targets = {
    "YOLOv5n-Face.modern": "YOLOv5n-Face.modern.pt",
    "license": "license.txt",
}


__all__ = [
    "HAS_TORCH",
    "HAS_YOLOV5_FACE",
    "TFWLandmarker",
    "LandmarkWrapper",
]
