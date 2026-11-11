import cv2
import numpy as np
import pytest

from theftprevention import core
from theftprevention.core import BLOCKED, PRESENT, Item, Watch, ncc

H, W = 480, 640
RECT = (280, 200, 80, 80)
EVENT_KEYS = {"id", "type", "item", "ts", "start", "reason", "detail"}


def texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    smooth = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (7, 7), 0)
    lo, hi = smooth.min(), smooth.max()
    gray = ((smooth - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def frame_with(bg: np.ndarray, patch: np.ndarray | None = None, at: tuple[int, int] = RECT[:2]) -> np.ndarray:
    frame = bg.copy()
    if patch is not None:
        x, y = at
        h, w = patch.shape[:2]
        frame[y : y + h, x : x + w] = patch
    return frame


def drive(watch: Watch, seq: list[np.ndarray]) -> list[dict]:
    events = []
    for k, frame in enumerate(seq, start=1):
        events += watch.step(frame, round(0.2 * k, 3))
    return events


def setup(seed: int = 0):
    rng = np.random.default_rng(seed)
    bg = texture(rng, H, W)
    patch = texture(rng, RECT[3], RECT[2])
    watch = Watch([Item("bag", RECT)], frame_with(bg, patch), 0.0)
    return rng, bg, patch, watch


def test_present_frames_emit_nothing():
    _, bg, patch, watch = setup()
    assert drive(watch, [frame_with(bg, patch)] * 30) == []
    assert watch.items[0].state == PRESENT


def test_removal_emits_one_candidate_and_return_resolves_it():
    _, bg, patch, watch = setup()
    seq = [frame_with(bg, patch)] * 10 + [frame_with(bg)] * 30 + [frame_with(bg, patch)] * 10
    events = drive(watch, seq)
    assert [e["type"] for e in events] == ["removal_candidate", "returned"]
    cand, ret = events
    assert set(cand) == EVENT_KEYS
    assert cand["item"] == "bag"
    assert abs(cand["start"] - 2.2) < 0.5
    assert cand["start"] + 3.0 <= cand["ts"] < cand["start"] + 3.5
    assert ret["detail"]["removal_id"] == cand["id"]
    assert ret["detail"]["away_s"] > 0


def test_shifted_item_gives_possible_return_then_restore_returns():
    _, bg, patch, watch = setup()
    shifted = frame_with(bg, patch, at=(RECT[0] + RECT[2] // 4, RECT[1]))
    seq = [frame_with(bg, patch)] * 10 + [frame_with(bg)] * 30 + [shifted] * 25 + [frame_with(bg, patch)] * 10
    events = drive(watch, seq)
    assert [e["type"] for e in events] == ["removal_candidate", "possible_return", "returned"]
    cand, possible, ret = events
    assert possible["detail"]["removal_id"] == cand["id"]
    assert possible["detail"]["away_s"] > 0
    assert possible["ts"] >= 8.2 + 3.0 - 1e-6
    assert ret["detail"]["removal_id"] == cand["id"]


def test_hand_at_candidate_step_gives_no_possible_return():
    rng, bg, patch, watch = setup()
    hand = texture(rng, RECT[3], RECT[2])
    seq = [frame_with(bg, patch)] * 10 + [frame_with(bg)] * 16 + [frame_with(bg, hand)] + [frame_with(bg)] * 25
    assert [e["type"] for e in drive(watch, seq)] == ["removal_candidate"]


def test_blocked_area_emits_no_candidate():
    rng, bg, patch, watch = setup()
    seq = [frame_with(bg, texture(rng, RECT[3], RECT[2])) for _ in range(40)]
    assert drive(watch, seq) == []
    assert watch.items[0].state == BLOCKED


def test_black_frames_emit_one_view_unusable():
    _, bg, patch, watch = setup()
    black = np.zeros((H, W, 3), np.uint8)
    events = drive(watch, [frame_with(bg, patch)] * 5 + [black] * 20)
    assert [e["type"] for e in events] == ["view_unusable"]
    assert events[0]["item"] is None
    assert events[0]["detail"] == {}
    assert watch.view_ok is False


def test_unusable_time_does_not_count_toward_absent():
    _, bg, patch, watch = setup()
    black = np.zeros((H, W, 3), np.uint8)
    seq = (
        [frame_with(bg, patch)] * 5
        + [frame_with(bg)] * 5
        + [black] * 30
        + [frame_with(bg)] * 20
    )
    events = drive(watch, seq)
    assert [e["type"] for e in events] == ["view_unusable", "removal_candidate"]
    recovered_at = 8.2
    assert events[1]["ts"] >= recovered_at + 3.0 - 1e-6


def test_rect_outside_frame_is_rejected():
    with pytest.raises(ValueError):
        Watch([Item("x", (W + 10, 0, 20, 20))], np.zeros((H, W, 3), np.uint8), 0.0)


def test_ncc_of_flat_arrays_is_zero():
    flat = np.full((5, 5), 7, np.uint8)
    assert ncc(flat, flat) == 0.0


def crop_of(frame: np.ndarray) -> np.ndarray:
    return core._crop(core._gray(frame), RECT)


def jittered_empty(rng: np.random.Generator, bg: np.ndarray, n: int) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
    frames, offsets = [], []
    for _ in range(n):
        dy, dx = (int(v) for v in rng.integers(-1, 2, size=2))
        rolled = np.roll(bg, (dy, dx), axis=(0, 1)).astype(np.int16)
        noise = rng.integers(-3, 4, size=bg.shape)
        frames.append(np.clip(rolled + noise, 0, 255).astype(np.uint8))
        offsets.append((dy, dx))
    return frames, offsets


def test_camera_jitter_on_empty_scene_still_emits_removal_candidate():
    rng, bg, patch, watch = setup()
    jitter, _ = jittered_empty(rng, bg, 40)
    events = drive(watch, [frame_with(bg, patch)] * 10 + jitter)
    assert "removal_candidate" in [e["type"] for e in events]


def test_change_compensates_jitter_where_plain_difference_fails():
    rng, bg, _, _ = setup()
    jitter, offsets = jittered_empty(rng, bg, 40)
    crops = [crop_of(f) for f in jitter]
    bites = []
    for k in range(1, len(crops)):
        if offsets[k] == offsets[k - 1]:
            continue
        plain = float(np.abs(crops[k].astype(np.float32) - crops[k - 1].astype(np.float32)).mean())
        if plain > core.MOTION:
            bites.append(k)
    assert bites, "no jittered pair exceeded MOTION under the plain difference"
    k = bites[0]
    assert core._change(crops[k], crops[k - 1]) < core.MOTION


def test_occlusion_still_counts_as_movement():
    rng, bg, patch, watch = setup()
    patches = [texture(rng, RECT[3], RECT[2]) for _ in range(40)]
    frames = [frame_with(bg, patch)] * 10 + [frame_with(bg, p) for p in patches]
    crops = [crop_of(f) for f in frames[10:]]
    for prev, cur in zip(crops, crops[1:]):
        assert core._change(cur, prev) > core.MOTION
    events = drive(watch, frames)
    assert "removal_candidate" not in [e["type"] for e in events]


def test_change_of_identical_crop_is_zero_and_of_2px_shift_is_small():
    rng, bg, _, _ = setup()
    gray = core._gray(bg)
    crop = gray[200:280, 280:360]
    shifted = gray[202:282, 282:362]
    assert core._change(crop, crop) == 0.0
    assert core._change(crop, shifted) < 1.0
