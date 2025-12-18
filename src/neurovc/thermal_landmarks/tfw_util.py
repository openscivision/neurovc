import importlib
import importlib.util
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np

from neurovc.util import normalize_color

HAS_YOLOV5_FACE = importlib.util.find_spec("yolov5_face") is not None
HAS_GDOWN = importlib.util.find_spec("gdown") is not None


class OptionalDependencyError(ImportError):
    pass


def require(module: str, *, extra: str, purpose: str) -> ModuleType:
    root = module.split(".", 1)[0]
    root_spec = importlib.util.find_spec(root)
    if root_spec is None:
        raise OptionalDependencyError(
            f"Missing optional dependency '{root}' required for {purpose}. "
            f"Install via `pip install neurovc[{extra}]`."
        )

    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as exc:
        raise OptionalDependencyError(
            f"Dependency '{root}' is installed, but importing '{module}' failed "
            f"(likely incompatible/partial install). "
            f"Try `pip install -U --force-reinstall {root}`. "
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc
    except Exception as exc:
        raise OptionalDependencyError(
            f"Dependency '{module}' was found but failed to import (broken binary/env). "
            f"Original error: {exc.__class__.__name__}: {exc}"
        ) from exc


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

        gdown = require("gdown", extra="landmark", purpose="TFW model download")

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
    def __init__(self, model_name="YOLOv5n-Face.modern", device=None):
        yf = require(
            "yolov5_face.detect_face", extra="landmark", purpose="TFWLandmarker"
        )

        self._yf = yf

        self.device = device
        model_path = _prepare_model(model_name)
        self.model = (
            yf.load_model(model_path, self.device)
            if self.device
            else yf.load_model(model_path)
        )

    def detect(self, img):
        img = normalize_color(img, color_map=cv2.COLORMAP_BONE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return (
            self._yf.detect_landmarks(self.model, img, self.device)
            if self.device
            else self._yf.detect_landmarks(self.model, img)
        )

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
    "HAS_YOLOV5_FACE",
    "HAS_GDOWN",
    "OptionalDependencyError",
    "require",
    "TFWLandmarker",
    "LandmarkWrapper",
]
