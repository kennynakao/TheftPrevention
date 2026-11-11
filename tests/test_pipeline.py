import os
import threading
import time

import cv2
import numpy as np
import pytest

import theftprevention.pipeline as pipeline
from test_core import H, RECT, W, frame_with, texture
from theftprevention.clips import ClipBuffer
from theftprevention.pipeline import Session, frames

LIVE_FPS = 20.0


class FakeCamera:
    """Stands in for cv2.VideoCapture at 30 fps. Open n delivers plan[n] good reads, then fails; None never fails."""

    plan: list = []
    opens: list = []

    def __init__(self, *args) -> None:
        self.args = args
        self.index = len(FakeCamera.opens)
        self.good = FakeCamera.plan[self.index] if self.index < len(FakeCamera.plan) else None
        self.reads = 0
        self.released = False
        FakeCamera.opens.append(self)

    def isOpened(self) -> bool:
        return True

    def read(self):
        time.sleep(1 / 30)
        self.reads += 1
        if self.good is not None and self.reads > self.good:
            return False, None
        return True, np.full((48, 64, 3), self.index, np.uint8)

    def release(self) -> None:
        self.released = True


class RefusedCamera(FakeCamera):
    def isOpened(self) -> bool:
        return False


class CrashingCamera(FakeCamera):
    """Delivers three frames, then read() raises."""

    def read(self):
        time.sleep(1 / 30)
        self.reads += 1
        if self.reads > 3:
            raise RuntimeError("corrupt stream")
        return True, np.full((48, 64, 3), 7, np.uint8)


@pytest.fixture
def camera(monkeypatch):
    FakeCamera.plan, FakeCamera.opens = [], []
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", FakeCamera)
    return FakeCamera


def write_video(path: str) -> None:
    rng = np.random.default_rng(1)
    bg = texture(rng, H, W)
    patch = texture(rng, RECT[3], RECT[2])
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    assert writer.isOpened()
    for i in range(100):
        writer.write(frame_with(bg, patch) if i < 30 else frame_with(bg))
    writer.release()


def test_session_reports_candidate_with_clip(tmp_path):
    video = str(tmp_path / "room.mp4")
    write_video(video)
    media = tmp_path / "media"
    media.mkdir()
    events = []
    session = Session(video, str(media), events.append, absent_s=1.0)
    x, y, w, h = RECT
    session.arm([{"name": "bag", "rect": [x / W, y / H, w / W, h / H]}])
    session.run()

    candidates = [e for e in events if e["type"] == "removal_candidate"]
    assert len(candidates) == 1
    clip = candidates[0]["clip"]
    assert clip is not None
    path = media / clip
    assert path.stat().st_size > 0
    cap = cv2.VideoCapture(str(path))
    ok, _ = cap.read()
    cap.release()
    assert ok
    assert session.ended


def test_callback_error_releases_video_file(tmp_path):
    video = str(tmp_path / "room.mp4")
    write_video(video)
    media = tmp_path / "media"
    media.mkdir()

    def on_event(ev):
        raise RuntimeError("callback failed")

    session = Session(video, str(media), on_event, absent_s=1.0)
    x, y, w, h = RECT
    session.arm([{"name": "bag", "rect": [x / W, y / H, w / W, h / H]}])
    with pytest.raises(RuntimeError) as info:
        session.run()
    os.remove(video)
    assert not os.path.exists(video)
    assert str(info.value) == "callback failed"


def test_clip_fps_follows_frame_spacing(tmp_path):
    buf = ClipBuffer(10.0)
    frame = np.full((48, 64, 3), 100, np.uint8)
    for k in range(10):
        buf.add(frame, k * 0.5)
    path = str(tmp_path / "clip.webm")
    assert buf.write(path, 0.0, 4.5)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(fps - 2.0) < 0.1


def test_live_rule(tmp_path):
    media = str(tmp_path)
    assert Session(0, media, print).live
    assert Session("rtsp://cam.test/stream", media, print).live
    assert not Session("room.mp4", media, print).live
    assert not Session("file:///C:/clips/a.mp4", media, print).live
    assert not Session(r"C:\clips\a.mp4", media, print).live


def test_url_opens_with_ffmpeg_options_and_timeouts(camera, monkeypatch):
    monkeypatch.delenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", raising=False)
    gen = frames("rtsp://cam.test/stream", fps=LIVE_FPS)
    next(gen)
    gen.close()
    assert camera.opens[0].args == (
        "rtsp://cam.test/stream",
        cv2.CAP_FFMPEG,
        [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000, cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000],
    )
    assert os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] == "rtsp_transport;tcp"
    assert camera.opens[0].released


def test_first_open_failure_raises(monkeypatch):
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", RefusedCamera)
    with pytest.raises(RuntimeError):
        next(frames(0))


def test_slow_consumer_gets_fresh_frames(camera):
    gen = frames(0, fps=LIVE_FPS)
    ages = []
    for _ in range(8):
        ts, _ = next(gen)
        ages.append(time.time() - ts)
        time.sleep(0.12)
    gen.close()
    assert max(ages) < 2 / LIVE_FPS


def test_yields_no_faster_than_fps(camera):
    gen = frames(0, fps=LIVE_FPS)
    times = []
    start = time.monotonic()
    while time.monotonic() - start < 0.5:
        next(gen)
        times.append(time.monotonic())
    gen.close()
    assert np.diff(times).min() >= 0.9 / LIVE_FPS
    assert len(times) <= 0.5 * LIVE_FPS + 2


def test_reconnects_after_outage(camera):
    camera.plan = [5]
    gen = frames(0, fps=LIVE_FPS)
    seen = set()
    deadline = time.monotonic() + 5
    while 1 not in seen:
        assert time.monotonic() < deadline
        _, frame = next(gen)
        seen.add(int(frame[0, 0, 0]))
    gen.close()
    assert camera.opens[0].released
    assert len(camera.opens) == 2


def test_close_during_outage_returns_fast(camera):
    camera.plan = [3]
    gen = frames(0, fps=LIVE_FPS)
    for _ in range(3):
        next(gen)
    time.sleep(0.3)
    start = time.monotonic()
    gen.close()
    assert time.monotonic() - start < 2
    assert len(camera.opens) == 1
    assert camera.opens[0].released


def test_session_stop_ends_live_run_during_outage(camera, tmp_path):
    camera.plan = [0]
    session = Session(0, str(tmp_path), lambda ev: None, fps=LIVE_FPS)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    time.sleep(0.3)
    session.stop()
    thread.join(2)
    assert not thread.is_alive()
    assert session.ended
    assert camera.opens[0].released


def test_reader_crash_raises_in_consumer(camera, monkeypatch):
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", CrashingCamera)
    gen = frames(0, fps=LIVE_FPS)
    for _ in range(3):
        next(gen)
    start = time.monotonic()
    with pytest.raises(RuntimeError):
        next(gen)
    assert time.monotonic() - start < 1
    assert camera.opens[0].released


def test_no_frame_is_repeated_during_outage(camera):
    camera.plan = [5]
    gen = frames(0, fps=LIVE_FPS)
    got, stamps = [], []
    while True:
        _, frame = next(gen)
        got.append(frame)
        stamps.append(time.monotonic())
        if frame[0, 0, 0] == 1:
            break
    gen.close()
    values = [int(f[0, 0, 0]) for f in got]
    assert values.count(0) <= 5
    assert values == sorted(values)
    assert len({id(f) for f in got}) == len(got)
    old_frames = [s for s, v in zip(stamps, values) if v == 0]
    assert stamps[values.index(1)] - old_frames[-1] >= 0.9


def test_clock_step_back_does_not_stall(camera, monkeypatch):
    real_time = time.time
    offset = [0.0]
    monkeypatch.setattr(pipeline.time, "time", lambda: real_time() + offset[0])
    gen = frames(0, fps=LIVE_FPS)
    for _ in range(3):
        next(gen)
    offset[0] = -5.0
    for _ in range(5):
        start = time.monotonic()
        next(gen)
        assert time.monotonic() - start < 1
    gen.close()


def test_live_session_runs_without_arm(camera, tmp_path):
    session = Session(0, str(tmp_path), lambda ev: None, fps=LIVE_FPS)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 2
    while session.latest is None and time.monotonic() < deadline:
        time.sleep(0.02)
    assert session.latest is not None
    session.stop()
    thread.join(2)
    assert not thread.is_alive()


def test_open_error_hides_credentials(monkeypatch):
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", RefusedCamera)
    with pytest.raises(RuntimeError) as info:
        next(frames("rtsp://camuser:s3cret@10.0.0.5:554/x"))
    message = str(info.value)
    assert "s3cret" not in message
    assert "camuser" not in message
    assert "rtsp://10.0.0.5:554/x" in message
