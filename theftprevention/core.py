import uuid
from dataclasses import dataclass, field

import cv2
import numpy as np

PRESENT, BLOCKED, ABSENT, UNUSABLE = "present", "blocked", "absent", "unusable"
# ITEM_NCC was raised from 0.6 after a sweep on labelled synthetic clips: 0.8 caught the most removals on the
# training split and accepted no moved or swapped item as returned. Re-check it on real footage.
DARK, BLUR_RATIO, VIEW_NCC, ITEM_NCC, MOTION = 20.0, 0.25, 0.5, 0.8, 4.0
JITTER = 2  # max whole-pixel camera shift compensated when measuring motion
VIEW_GRID = 4  # the view check compares a VIEW_GRID by VIEW_GRID grid of tiles
FLAT = 2.0  # a reference tile whose grey standard deviation is at or below this has no texture to compare


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-mean normalised cross-correlation; 0.0 when either input is flat."""
    a = a.astype(np.float32).ravel()
    b = b.astype(np.float32).ravel()
    a -= a.mean()
    b -= b.mean()
    denom = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denom < 1e-6:
        return 0.0
    return float(np.dot(a, b)) / denom


def _view_match(gray: np.ndarray, ref: np.ndarray, tiles: list) -> float:
    """Median of the per-tile ncc over the given tiles, skipping flat reference tiles; whole-frame ncc if fewer than VIEW_GRID remain."""
    if len(tiles) < VIEW_GRID:
        return ncc(gray, ref)
    return float(np.median([ncc(gray[t], ref[t]) for t in tiles]))


@dataclass
class Item:
    name: str
    rect: tuple[int, int, int, int]  # x, y, w, h in pixels of the analysed frame
    state: str = PRESENT
    since: float = 0.0  # ts the current state began
    last_clear: float = 0.0  # last ts the item was clearly present
    removal_id: str | None = None  # id of the open, unreturned removal event
    removed_at: float = 0.0  # `start` of that removal
    history: list[tuple[float, str]] = field(default_factory=list)  # accepted (ts, state) transitions


def _gray(frame: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)


def _crop(gray: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = rect
    return gray[y : y + h, x : x + w]


def _change(crop: np.ndarray, prev: np.ndarray) -> float:
    """Smallest mean absolute difference over integer shifts of up to JITTER pixels, on the overlap."""
    a = crop.astype(np.float32)
    b = prev.astype(np.float32)
    h, w = a.shape
    if min(h, w) < 2 * JITTER + 1:
        return float(np.abs(a - b).mean())
    best = float("inf")
    for dy in range(-JITTER, JITTER + 1):
        for dx in range(-JITTER, JITTER + 1):
            ar = slice(max(0, -dy), h - max(0, dy))
            br = slice(max(0, dy), h - max(0, -dy))
            ac = slice(max(0, -dx), w - max(0, dx))
            bc = slice(max(0, dx), w - max(0, -dx))
            best = min(best, float(np.abs(a[ar, ac] - b[br, bc]).mean()))
    return best


def _seen(run: list, value, ts: float, confirm: int) -> float | None:
    """Count consecutive steps with `value`; return the run's first ts once it reaches `confirm`."""
    if run[0] != value:
        run[0], run[1], run[2] = value, 0, ts
    run[1] += 1
    return run[2] if run[1] >= confirm else None


def _event(kind: str, item: str | None, start: float, ts: float, reason: str, detail: dict) -> dict:
    return {"id": str(uuid.uuid4()), "type": kind, "item": item, "ts": ts, "start": start, "reason": reason, "detail": detail}


class Watch:
    def __init__(self, items: list[Item], frame: np.ndarray, ts: float, absent_s: float = 3.0, confirm: int = 2) -> None:
        self.items = items
        self.absent_s = absent_s
        self.confirm = confirm
        self.view_ok = True
        self._ref = _gray(frame)
        self._ref_sharp = cv2.Laplacian(self._ref, cv2.CV_64F).var()
        h, w = self._ref.shape
        grid = [(slice(i * h // VIEW_GRID, (i + 1) * h // VIEW_GRID), slice(j * w // VIEW_GRID, (j + 1) * w // VIEW_GRID)) for i in range(VIEW_GRID) for j in range(VIEW_GRID)]
        self._tiles = [t for t in grid if self._ref[t].std() > FLAT]
        self._refs = [_crop(self._ref, it.rect) for it in items]
        if any(ref.size == 0 for ref in self._refs):
            raise ValueError("rect outside frame")
        self._prev = self._refs
        self._runs: list[list] = [[None, 0, 0.0] for _ in items]
        self._view_run: list = [None, 0, 0.0]
        self._snaps: list[np.ndarray | None] = [None] * len(items)
        self._changed: list[float | None] = [None] * len(items)
        for it in items:
            it.since = it.last_clear = ts

    def step(self, frame: np.ndarray, ts: float) -> list[dict]:
        gray = _gray(frame)
        crops = [_crop(gray, it.rect) for it in self.items]
        events: list[dict] = []
        good = bool(
            gray.mean() >= DARK
            and cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_RATIO * self._ref_sharp
            and _view_match(gray, self._ref, self._tiles) >= VIEW_NCC
        )
        start = _seen(self._view_run, good, ts, self.confirm)
        if start is not None and good != self.view_ok:
            self.view_ok = good
            if good:
                for it in self.items:
                    if it.state == ABSENT and it.removal_id is None:
                        it.since = ts
            else:
                events.append(_event("view_unusable", None, start, ts, "Camera view is dark, blurred or shifted. Monitoring is paused until it recovers.", {}))
        if self.view_ok:
            for i, (it, crop) in enumerate(zip(self.items, crops)):
                events += self._classify(i, it, crop, ts)
        else:
            self._changed = [None] * len(self.items)
            for run in self._runs:
                run[0] = None
        self._prev = crops
        return events

    def _classify(self, i: int, it: Item, crop: np.ndarray, ts: float) -> list[dict]:
        if ncc(crop, self._refs[i]) >= ITEM_NCC:
            raw = PRESENT
        elif _change(crop, self._prev[i]) > MOTION:
            raw = BLOCKED
        else:
            raw = ABSENT
        start = _seen(self._runs[i], raw, ts, self.confirm)
        if start is not None and raw != it.state:
            it.history.append((start, raw))
            it.state, it.since = raw, start
        events = []
        if it.state == PRESENT:
            it.last_clear = ts
            if it.removal_id is not None:
                events.append(_event("returned", it.name, ts, ts, "Item is visible in its watched area again.", {"removal_id": it.removal_id, "away_s": ts - it.removed_at}))
                it.removal_id = None
        still_absent = it.state == ABSENT and raw == ABSENT
        snap = self._snaps[i] if it.removal_id is not None else None
        changed = snap is not None and still_absent and ncc(crop, snap) < ITEM_NCC
        if not changed:
            self._changed[i] = None
        elif self._changed[i] is None:
            self._changed[i] = ts
        if changed and ts - self._changed[i] >= self.absent_s:
            self._snaps[i] = None
            reason = "Watched area changed again but does not match the reference. The item may be back in a different position, or something else is there."
            events.append(_event("possible_return", it.name, ts, ts, reason, {"removal_id": it.removal_id, "away_s": ts - it.removed_at}))
        if still_absent and it.removal_id is None and ts - it.since >= self.absent_s:
            detail = {"last_clear": it.last_clear, "history": [[t, s] for t, s in it.history[-10:]]}
            ev = _event("removal_candidate", it.name, it.since, ts, f"Watched area has shown no item for {ts - it.since:.0f} s in a clear view.", detail)
            it.removal_id, it.removed_at = ev["id"], it.since
            self._snaps[i] = crop
            events.append(ev)
        return events
