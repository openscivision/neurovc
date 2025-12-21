__author__ = "Philipp Flotho"
"""
multimodal_cam_calib
Copyright 2021 by Philipp Flotho, All rights reserved.
"""
import numpy as np  # noqa: E402
import cv2  # noqa: E402

from neurovc.util.IO_util import imagesc  # noqa: E402
from neurovc.util.IO_util import normalize_color  # noqa: E402


def _require_h5py(purpose: str):
    try:
        import h5py  # type: ignore
    except ImportError as exc:
        raise ImportError(
            f"h5py is required for {purpose}. "
            "Install via `pip install h5py` or `pip install neurovc[momag]`."
        ) from exc
    return h5py


class SingleRecording:
    # class works for hfr and thermal hdf recordings
    def __init__(self, hdf_file, n_frames=None, shape=None):
        self.hdf_file = hdf_file
        self._h5py = _require_h5py("SingleRecording")

        with self._h5py.File(self.hdf_file, "r") as f:
            frame_key = "Frames" if "Frames" in f.keys() else "FRAMES"
            self.width = f[frame_key].shape[1]  # .attrs['FrameWidth']
            self.height = f[frame_key].shape[2]  # .attrs['FrameHeight']
            if n_frames is None:
                self.n_frames = f[frame_key].shape[0]  # .attrs['FrameCount']
            else:
                self.n_frames = n_frames

        self.frames = self._h5py.File(self.hdf_file, "r")[frame_key]
        self.timestamps = self._h5py.File(self.hdf_file, "r")["Timestamps_ms"][:]
        self.shape = shape

    def load_data(self):
        pass

    def get_frames(self, idx):
        if self.shape is None:
            frames = self.frames[idx]
        else:
            frames = self.frames[idx, : self.shape[0], : self.shape[1]]
        timestamps = self.timestamps[idx]
        return frames, timestamps

    def plot_thermal_frame(self, frame_num):
        # resize the thermal image or representation
        test_frame = np.resize(self.frames[frame_num], (480, 640))
        # norm the thermal image
        norm_img = normalize_color(test_frame)
        imagesc(norm_img)

    def get_thermal_frame(self, frame_num):
        thermal_frame = np.resize(self.frames[frame_num], (480, 640))

        return thermal_frame


def __test_thermal2():
    # thermal_file = "/home/philipp/covid/Thermal/2020_05_07/ThermalData_07_05_2020_10_38_43.h5"
    thermal_file = (
        "G:/covid-19/data/pilot3/elena/thermal/ThermalData_18_06_2020_13_24_58.h5"
    )
    tmp = SingleRecording(thermal_file, n_frames=10)
    tmp.load_data()
    imagesc(tmp.frames[0])
    cv2.waitKey()


def synchronize_timestamps(left_ts, right_ts, fps=30):
    timestamp_baseline = np.min(np.array([left_ts[0], right_ts[0]]))

    left_ts = np.array(left_ts[:] - timestamp_baseline).astype(float)
    right_ts = np.array(right_ts[:] - timestamp_baseline).astype(float)

    start_time = 0
    end_time = np.min(np.array([left_ts[-1], right_ts[-1]]))

    left_time_resampled = np.arange(start_time, end_time, 1000 / fps, float)
    idx_left = np.round(
        np.interp(left_time_resampled, left_ts, np.arange(0, left_ts.shape[0]))
    ).astype(int)
    right_time_resampled = np.arange(start_time, end_time, 1000 / fps, float)
    idx_right = np.round(
        np.interp(right_time_resampled, right_ts, np.arange(0, right_ts.shape[0]))
    ).astype(int)

    return idx_left, left_time_resampled, idx_right, right_time_resampled


if __name__ == "__main__":
    __test_thermal2()
