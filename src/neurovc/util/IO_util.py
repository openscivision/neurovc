__author__ = "Philipp Flotho"
"""
multimodal_cam_calib
Copyright 2021 by Philipp Flotho, All rights reserved.
"""
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import os  # noqa: E402


class VideoLooper:
    def __init__(self):
        self.stop = False

    def __call__(self, keystroke):
        keystroke &= 0xFF
        if keystroke == ord(" "):
            while True:
                keystroke = cv2.waitKey() & 0xFF
                if keystroke == 27:
                    self.stop = True
                    break
                if keystroke == ord(" "):
                    break
        return keystroke == 27 or self.stop


class CircularFrameBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.counter = 0

    def append(self, frame):
        if len(self.buffer) < self.size:
            self.buffer.append(frame)
        else:
            self.buffer[self.counter] = frame
            self.counter = (self.counter + 1) % self.size

    def get_oldest(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer[self.counter]

    def get_newest(self):
        if len(self.buffer) == 0:
            return None
        return self.buffer[(self.counter - 1) % self.size]

    def get(self):
        return self.buffer


class VideoReader:
    def __init__(self, filepath):
        self.cap = cv2.VideoCapture(filepath)
        self.framerate = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

    def read_frame(self, i=None):
        ret = False
        frame = None
        if self.cap.isOpened():
            if i is not None and i < self.n_frames:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = self.cap.read()
        return ret, frame

    def has_frames(self):
        return (
            self.cap.isOpened()
            and self.cap.get(cv2.CAP_PROP_POS_FRAMES) < self.n_frames
        )

    def read_frames(self, start_idx=0, end_idx=-1, offset=1):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_idx)
        counter = start_idx
        frames = []
        while counter < end_idx:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(frame)
            counter += 1
        return ret, np.array(frames)

    def __getitem__(self, item):
        if isinstance(item, slice):
            start, stop, step = item.indices(int(self.n_frames))
            success, frames = self.read_frames(
                start_idx=start, end_idx=stop, offset=step
            )
            return frames if success else np.array([])
        if isinstance(item, int):
            idx = item
            if idx < 0:
                idx = int(self.n_frames) + idx
            ret, frame = self.read_frame(idx)
            return frame if ret else None
        raise TypeError("VideoReader indices must be integers or slices.")


class HDFFileVideoWriter:
    def __init__(self, filepath="output.h5", ds_names=("Frames",)):
        try:
            from h5py import File as _File
        except ImportError as exc:
            raise ImportError(
                "h5py is required for HDFFileVideoWriter. "
                "Install via `pip install h5py` or `pip install neurovc[momag]`."
            ) from exc
        self._File = _File
        self.__init = False
        self.ds_names = ds_names
        try:
            os.remove(filepath)
        except OSError:
            pass
        self.file = self._File(filepath, "w")

    def __call__(self, frames, ts):
        assert len(frames) == len(self.ds_names)
        if not self.__init:
            for i, frame in enumerate(frames):
                self.file.create_dataset(
                    self.ds_names[i],
                    (0,) + frame.shape,
                    dtype=frame.dtype,
                    chunks=(1,) + frame.shape,
                    maxshape=(None,) + frame.shape,
                )
            self.file.create_dataset("Timestamps", (0,), chunks=True, maxshape=(None,))
            self.__init = True

        for i, frame in enumerate(frames):
            ds = self.file[self.ds_names[i]]
            ds.resize((ds.shape[0] + 1,) + ds.shape[1:])
            ds[-1] = frame
        ds = self.file["Timestamps"]
        ds.resize((ds.shape[0] + 1,))
        ds[-1] = ts

    def __del__(self):
        if self.file is not None:
            self.file.close()


class VideoWriter:
    def __init__(self, filepath, framerate=50):
        self.writer = None
        self.filepath = filepath
        self.width = None
        self.height = None
        self.n_channels = None
        self.framerate = framerate

    def _check_range(self, min_val, max_val, checkrange):
        return

    def __call__(self, frame):
        if self.writer is None:
            self.height, self.width = frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )  # cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            self.writer = cv2.VideoWriter(
                self.filepath, fourcc, self.framerate, (self.width, self.height)
            )
            # self.writer = cv2.VideoWriter(self.filepath, cv2.VideoWriter_fourcc('D', 'I', 'V', 'X'), self.framerate, (self.width, self.height))

        if len(frame.shape) < 3:
            self.n_channels = 1
        else:
            self.n_channels = frame.shape[2]

        if self.n_channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        if frame.dtype != np.uint8:
            min_val = np.min(frame)
            max_val = np.max(frame)
            if min_val > 0 and max_val < 1:
                frame = (255 * frame).astype(np.uint8)
            elif min_val > 0 and max_val < 255:
                frame = frame.astype(np.uint8)
            else:
                frame = cv2.normalize(
                    frame,
                    None,
                    alpha=0,
                    beta=255,
                    norm_type=cv2.NORM_MINMAX,
                    dtype=cv2.CV_8UC3,
                )
        self.writer.write(frame)

    def write_frames(self, frames):
        for frame in frames:
            self(frame)

    def __del__(self):
        if self.writer is not None:
            self.writer.release()


class Debayerer:
    def __init__(self, pattern=cv2.COLOR_BAYER_RG2BGR, nh=2):
        self.pattern = pattern
        colors = np.array([255, 255, 255])

        self.reference = np.expand_dims(
            np.mean(np.array(colors), axis=0).astype(float), (0, 1)
        )

    def set_white_balance(self, colors):
        if isinstance(colors, np.ndarray):
            assert np.sum(colors.shape) == 3
        else:
            assert len(colors) == 3
        self.reference = np.expand_dims(
            np.mean(np.squeeze(np.array(colors)), axis=0).astype(float), (0, 1)
        )

    def __call__(self, img):
        tmp = (cv2.cvtColor(img, self.pattern).astype(float) / self.reference) * 255
        tmp[tmp > 255] = 255
        return tmp.astype(np.uint8)


def remap_RGB(frame, scaling, f1, f2):
    frame = scaling * (frame.astype(float) - f2) - f1
    frame[frame < 0] = 0
    frame[frame > 255] = 255
    return frame.astype(np.uint8)


def imagesc(img, window_name="Imagesc", color_map=cv2.COLORMAP_HOT):
    cv2.imshow(window_name, normalize_color(img, color_map))


def normalize_color(img, color_map=cv2.COLORMAP_HOT, normalize=True):
    img_8b = np.empty(img.shape, np.uint8)
    if normalize:
        cv2.normalize(
            img, img_8b, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1
        )
    else:
        img_8b = (255 * img).astype(np.uint8)
    img_8b = cv2.applyColorMap(img_8b, color_map)
    return img_8b


def draw_landmarks(frame, landmarks, weights=None, color=[0, 0, 0]):
    if frame.shape[2] != 3:
        frame_rgb = normalize_color(frame)
    else:
        frame_rgb = frame.copy()
    if weights is not None:
        weights -= weights.min()
        weights /= weights.max()
    for i, (x, y) in enumerate(landmarks[..., :2]):
        if weights is not None:
            color = [0, 0, 0]
            color[2] = 1 - weights[i]
            color[1] = weights[i]
            color = tuple((255 * np.array(color)).astype(int).tolist())
        cv2.circle(frame_rgb, (int(np.round(x)), int(np.round(y))), 2, color, -1)

    return frame_rgb


def imgaussfilt(img, sigma):
    if np.isscalar(sigma):
        width = 2 * (np.ceil(6 * sigma) // 2) + 1
        height = width
    else:
        width = 2 * (np.ceil(6 * sigma[0]) // 2) + 1
        height = 2 * (np.ceil(6 * sigma[1]) // 2) + 1
    width = int(width)
    height = int(height)
    return cv2.GaussianBlur(img, (width, height), sigmaX=sigma, sigmaY=sigma)


def map_temp(data, cam="A655", resolution="high"):
    a = 1
    if resolution == "high":
        a = 0.1
    if cam == "A65":
        return (a * 0.4 * data) - 273.15
    elif cam == "A655":
        return (a * 0.1 * data) - 273.15
    else:
        print("camera not implemented!")
