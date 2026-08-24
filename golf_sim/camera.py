"""USB camera capture helper — requests the highest practical resolution/FPS."""

from __future__ import annotations

import cv2
import numpy as np


class Camera:
    CANDIDATE_SIZES = (
        (1920, 1080),
        (1280, 720),
        (960, 540),
        (640, 480),
    )

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index
        self.cap: cv2.VideoCapture | None = None
        self.width = 0
        self.height = 0
        self.fps = 0.0

    def open(self) -> bool:
        self.release()
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        cap = None
        for backend in backends:
            trial = cv2.VideoCapture(self.device_index, backend)
            if trial.isOpened():
                cap = trial
                break
            trial.release()

        if cap is None or not cap.isOpened():
            return False

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for w, h in self.CANDIDATE_SIZES:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, 60)
            ok, frame = cap.read()
            if ok and frame is not None and frame.shape[1] >= min(w, 640):
                break

        self.cap = cap
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        self.fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        return True

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            return None
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return None
        return frame

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
