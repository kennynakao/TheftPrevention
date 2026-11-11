import atexit
import os
import shutil
import tempfile
import time

import cv2
import numpy as np
from fastapi.testclient import TestClient

MEDIA = tempfile.mkdtemp(prefix="tp-test-")
os.environ["TP_MEDIA"] = MEDIA
atexit.register(shutil.rmtree, MEDIA, True)

from theftprevention.server import app  # noqa: E402

client = TestClient(app, base_url="http://127.0.0.1:8000")

H, W = 480, 640
RECT = (280, 200, 80, 80)
ITEM_FRAMES, EMPTY_FRAMES = 20, 40


def texture(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    smooth = cv2.GaussianBlur(rng.standard_normal((h, w)).astype(np.float32), (7, 7), 0)
    lo, hi = smooth.min(), smooth.max()
    gray = ((smooth - lo) / (hi - lo) * 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def write_video(path: str) -> None:
    rng = np.random.default_rng(1)
    bg = texture(rng, H, W)
    patch = texture(rng, RECT[3], RECT[2])
    x, y, w, h = RECT
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 10, (W, H))
    assert writer.isOpened()
    for i in range(ITEM_FRAMES + EMPTY_FRAMES):
        frame = bg.copy()
        if i < ITEM_FRAMES:
            frame[y : y + h, x : x + w] = patch
        writer.write(frame)
    writer.release()


def test_upload_arm_candidate_clip_and_review(tmp_path):
    assert client.post("/api/arm", json={"items": [{"name": "bag", "rect": [0, 0, 0.1, 0.1]}]}).status_code == 409

    video = tmp_path / "room.mp4"
    write_video(str(video))
    with video.open("rb") as f:
        assert client.post("/api/upload", files={"file": ("room.mp4", f, "video/mp4")}).status_code == 200
    assert client.get("/api/status").json()["source"] is True

    x, y, w, h = RECT
    rect = [x / W, y / H, w / W, h / H]
    assert client.post("/api/arm", json={"items": [{"name": "bag", "rect": rect}]}).status_code == 200

    candidates = []
    deadline = time.monotonic() + 30
    while not candidates and time.monotonic() < deadline:
        time.sleep(0.5)
        candidates = [e for e in client.get("/api/events").json() if e["type"] == "removal_candidate"]
    assert len(candidates) == 1
    event = candidates[0]
    assert event["item"] == "bag"
    assert "last_clear" in event["detail"]

    clip = client.get(f"/media/{event['clip']}")
    assert clip.status_code == 200
    assert clip.headers["content-type"] == "video/webm"
    assert clip.content[:4] == b"\x1aE\xdf\xa3"
    assert client.get("/media/events.db").status_code == 404

    url = f"/api/events/{event['id']}/review"
    assert client.post(url, json={"status": "bogus"}).status_code == 422
    reviewed = client.post(url, json={"status": "expected_removal", "note": "checked"})
    assert reviewed.status_code == 200
    stored = next(e for e in client.get("/api/events").json() if e["id"] == event["id"])
    assert stored["review"] == "expected_removal"
    assert stored["note"] == "checked"


def test_local_only_guard():
    assert client.get("/api/status", headers={"Host": "evil.example"}).status_code == 403
    assert client.post("/api/disarm", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.get("/api/status").status_code == 200


def test_unreadable_upload_is_rejected():
    junk = os.urandom(4096)
    assert client.post("/api/upload", files={"file": ("junk.mp4", junk, "video/mp4")}).status_code == 400
    before = {f for f in os.listdir(MEDIA) if f.startswith("upload-")}
    assert client.post("/api/upload", files={"file": ("x.b:c", junk, "application/octet-stream")}).status_code == 400
    assert {f for f in os.listdir(MEDIA) if f.startswith("upload-")} == before
