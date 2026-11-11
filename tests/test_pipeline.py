import cv2
import numpy as np

from test_core import H, RECT, W, frame_with, texture
from theftprevention.clips import ClipBuffer
from theftprevention.pipeline import Session


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
