from collections import deque

import cv2
import numpy as np


def _decode(jpeg: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)


class ClipBuffer:
    def __init__(self, seconds: float) -> None:
        self._seconds = seconds
        self._frames: deque[tuple[float, bytes]] = deque()

    def add(self, frame: np.ndarray, ts: float) -> None:
        ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        self._frames.append((ts, jpeg.tobytes()))
        while self._frames[0][0] < ts - self._seconds:
            self._frames.popleft()

    def write(self, path: str, start: float, end: float) -> bool:
        picked = [(ts, jpeg) for ts, jpeg in self._frames if start <= ts <= end]
        if not picked:
            return False
        span = picked[-1][0] - picked[0][0]
        fps = (len(picked) - 1) / span if span > 0 else 1.0
        h, w = _decode(picked[0][1]).shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"VP80"), fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"cannot open video writer for {path}")
        try:
            for _, jpeg in picked:
                writer.write(_decode(jpeg))
        finally:
            writer.release()
        return True
