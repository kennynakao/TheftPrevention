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


def _is_live(source: int | str) -> bool:
    return isinstance(source, int) or ("://" in source and not source.startswith("file://"))


def _shown(source: int | str) -> int | str:
    if isinstance(source, str) and "://" in source:
        scheme, _, rest = source.partition("://")
        return f"{scheme}://{rest[rest.rfind('@') + 1:]}"
    return source


def _open(source: int | str) -> cv2.VideoCapture:
    if isinstance(source, int):
        return cv2.VideoCapture(source)
    os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
    return cv2.VideoCapture(source, cv2.CAP_FFMPEG, [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000])


def _fit(frame: np.ndarray, width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= width:
        return frame
    return cv2.resize(frame, (width, round(h * width / w)), interpolation=cv2.INTER_AREA)


def _reader(source: int | str, cap: cv2.VideoCapture, cond: threading.Condition, slot: list, stop: threading.Event) -> None:
    delay = 1.0
    try:
        while not stop.is_set():
            ok, frame = cap.read()
            if ok:
                delay = 1.0
                with cond:
                    slot[0] = (time.time(), frame)
                    cond.notify_all()
                continue
            cap.release()
            if stop.wait(delay):
                break
            delay = min(2 * delay, 30.0)
            cap = _open(source)
    finally:
        cap.release()


def _live(source: int | str, fps: float, stop: threading.Event, width: int = 960) -> Iterator[tuple[float, np.ndarray]]:
    cap = _open(source)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot open video source {_shown(source)!r}")
    cond, slot = threading.Condition(), [None]
    reader = threading.Thread(target=_reader, args=(source, cap, cond, slot, stop), daemon=True)
    reader.start()
    taken, last = None, -math.inf

    try:
        while not stop.is_set():
            time.sleep(max(0.0, last + 1.0 / fps - time.monotonic()))
            with cond:
                if not cond.wait_for(lambda: slot[0] is not None and slot[0] is not taken, 0.2):
                    if not reader.is_alive() and not stop.is_set():
                        raise RuntimeError(f"live reader for {_shown(source)!r} stopped")
                    continue
                taken = slot[0]
            last = time.monotonic()
            ts, frame = taken
            yield ts, _fit(frame, width)
    finally:
        stop.set()
        reader.join(2.0)


def frames(source: int | str, fps: float = 5.0, width: int = 960, pace: bool = False) -> Iterator[tuple[float, np.ndarray]]:
    if _is_live(source):
        yield from _live(source, fps, threading.Event(), width)
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video source {_shown(source)!r}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not (math.isfinite(src_fps) and src_fps > 0):
        src_fps = 30.0
    step = max(1, round(src_fps / fps))
    origin = time.monotonic()
    try:
        for index in itertools.count():
            if not cap.grab():
                return
            if index % step:
                continue
            ts = index / src_fps
            ok, frame = cap.retrieve()
            if not ok:
                raise RuntimeError("failed to decode a video frame")
            if pace:
                origin = max(origin, time.monotonic() - ts)
                time.sleep(max(0.0, ts - (time.monotonic() - origin)))
            yield ts, _fit(frame, width)
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
        self._halt = threading.Event()
        self.latest, self.ts, self.watch = None, 0.0, None
        self.live, self.ended = _is_live(source), False

    def arm(self, items: list[dict]) -> None:
        with self._lock:
            self._pending = items

    def disarm(self) -> None:
        with self._lock:
            self._pending = []

    def stop(self) -> None:
        self._halt.set()

    def run(self) -> None:
        feed = _live(self._source, self._fps, self._halt) if self.live else frames(self._source, self._fps, pace=self._pace)
        try:
            for ts, frame in feed:
                self.latest, self.ts = frame, ts
                self._clips.add(frame, ts)
                while not self.live and self.watch is None and self._pending is None and not self._halt.is_set():
                    time.sleep(0.05)
                if self._halt.is_set():
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
            feed.close()
            self._halt.set()
            self.ended = True

    def _flush(self, final: bool) -> None:
        ready = [ev for ev in self._queued if final or self.ts > ev["ts"] + self._post_s]
        self._queued = [ev for ev in self._queued if ev not in ready]
        for ev in ready:
            name = f"{ev['id']}.webm"
            saved = self._clips.write(os.path.join(self._media_dir, name), ev["start"] - self._pre_s, ev["ts"] + self._post_s)
            ev["clip"] = name if saved else None
            self._on_event(ev)
