import itertools
import math
import os
import threading
import time
from collections.abc import Callable, Iterator

import cv2
import numpy as np

from theftprevention.clips import ClipBuffer
from theftprevention.core import Item, Watch


def frames(source: int | str, fps: float = 5.0, width: int = 960, pace: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {source!r}")
    live = isinstance(source, int)
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(src_fps) and src_fps > 0):
        src_fps = 30.0
    step = max(1, round(src_fps / fps))
    period = 1.0 / fps
    origin = time.monotonic()
    last = -math.inf
    try:
        for index in itertools.count():
            if not cap.grab():
                return
            ts = time.time() if live else index / src_fps
            if live:
                if ts - last < period:
                    continue
                last = ts
            elif index % step:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError("failed to decode a video frame")
            if pace and not live:
                origin = max(origin, time.monotonic() - ts)
                time.sleep(max(0.0, ts - (time.monotonic() - origin)))
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)
            yield ts, frame
    finally:
        cap.release()


def _pixel_rect(rect: list[float], w: int, h: int) -> tuple[int, int, int, int]:
    x, y, rw, rh = rect
    pw = min(max(8, round(rw * w)), w)
    ph = min(max(8, round(rh * h)), h)
    px = min(max(0, round(x * w)), w - pw)
    py = min(max(0, round(y * h)), h - ph)
    return px, py, pw, ph


class Session:
    def __init__(self, source: int | str, media_dir: str, on_event: Callable[[dict], None], fps: float = 5.0,
                 pre_s: float = 5.0, post_s: float = 3.0, pace: bool = False, **watch_kw) -> None:
        os.makedirs(media_dir, exist_ok=True)
        self._source, self._media_dir, self._on_event = source, media_dir, on_event
        self._fps, self._pre_s, self._post_s, self._pace = fps, pre_s, post_s, pace
        self._watch_kw = watch_kw
        self._clips = ClipBuffer(pre_s + post_s + watch_kw.get("absent_s", 3.0) + 2)
        self._lock = threading.Lock()
        self._pending: list[dict] | None = None
        self._queued: list[dict] = []
        self._stop_requested = False
        self.latest, self.ts, self.watch = None, 0.0, None
        self.live, self.ended = isinstance(source, int), False

    def arm(self, items: list[dict]) -> None:
        with self._lock:
            self._pending = items

    def disarm(self) -> None:
        with self._lock:
            self._pending = []

    def stop(self) -> None:
        self._stop_requested = True

    def run(self) -> None:
        try:
            for ts, frame in frames(self._source, self._fps, pace=self._pace):
                self.latest, self.ts = frame, ts
                self._clips.add(frame, ts)
                while not self.live and self.watch is None and self._pending is None and not self._stop_requested:
                    time.sleep(0.05)
                if self._stop_requested:
                    break
                with self._lock:
                    pending, self._pending = self._pending, None
                if pending is not None:
                    items = [Item(d["name"], _pixel_rect(d["rect"], frame.shape[1], frame.shape[0])) for d in pending]
                    self.watch = Watch(items, frame, ts, **self._watch_kw) if items else None
                if self.watch is not None:
                    self._queued += self.watch.step(frame, ts)
                self._flush(final=False)
            self._flush(final=True)
        finally:
            self.ended = True

    def _flush(self, final: bool) -> None:
        ready = [ev for ev in self._queued if final or self.ts > ev["ts"] + self._post_s]
        self._queued = [ev for ev in self._queued if ev not in ready]
        for ev in ready:
            name = f"{ev['id']}.webm"
            saved = self._clips.write(os.path.join(self._media_dir, name), ev["start"] - self._pre_s, ev["ts"] + self._post_s)
            ev["clip"] = name if saved else None
            self._on_event(ev)
