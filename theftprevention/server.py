import json
import os
import re
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import closing
from typing import Annotated, Literal

import cv2
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from theftprevention.pipeline import Session

MEDIA = os.environ.get("TP_MEDIA", "./media")
DB = os.path.join(MEDIA, "events.db")
CLIP = re.compile(r"[0-9a-f-]{36}\.webm")

os.makedirs(MEDIA, exist_ok=True)
with closing(sqlite3.connect(DB)) as c, c:
    c.execute(
        "CREATE TABLE IF NOT EXISTS events (id TEXT PRIMARY KEY, type TEXT, item TEXT, ts REAL, start REAL, "
        "reason TEXT, detail TEXT, clip TEXT, review TEXT DEFAULT 'needs_review', note TEXT DEFAULT '')"
    )

app = FastAPI()
_lock = threading.Lock()
_current: Session | None = None
_thread: threading.Thread | None = None


class ArmItem(BaseModel):
    name: str
    rect: Annotated[list[Annotated[float, Field(ge=0, le=1)]], Field(min_length=4, max_length=4)]


class Arm(BaseModel):
    items: list[ArmItem] = Field(min_length=1, max_length=3)


class Review(BaseModel):
    status: Literal["needs_review", "expected_removal", "returned", "confirmed_stolen"]
    note: str = ""


@app.middleware("http")
async def local_only(request: Request, call_next):
    host = request.headers.get("host", "")
    origin = request.headers.get("origin")
    if host.split(":")[0] not in ("127.0.0.1", "localhost") or (origin is not None and origin != f"http://{host}"):
        return JSONResponse({"detail": "forbidden"}, status_code=403)
    return await call_next(request)


def _record(ev: dict) -> None:
    with sqlite3.connect(DB) as c:
        c.execute(
            "INSERT INTO events (id, type, item, ts, start, reason, detail, clip) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ev["id"], ev["type"], ev["item"], ev["ts"], ev["start"], ev["reason"], json.dumps(ev["detail"]), ev["clip"]),
        )


def _start(source: int | str) -> None:
    global _current, _thread
    with _lock:
        if _current is not None:
            _current.stop()
        if _thread is not None:
            _thread.join()
        _current = Session(source, MEDIA, _record, pace=True)
        _thread = threading.Thread(target=_current.run, daemon=True)
        _thread.start()


def _mjpeg(session: Session):
    while _current is session and not session.ended:
        frame = session.latest
        if frame is not None:
            img = frame.copy()
            watch = session.watch
            for it in watch.items if watch is not None else []:
                x, y, w, h = it.rect
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(img, f"{it.name}: {it.state}", (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            _, jpg = cv2.imencode(".jpg", img)
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg.tobytes() + b"\r\n"
        time.sleep(0.1)


def _event(row: sqlite3.Row) -> dict:
    return {**dict(row), "detail": json.loads(row["detail"])}


@app.post("/api/webcam")
def webcam(index: int = 0) -> dict:
    _start(index)
    return {"ok": True}


@app.post("/api/upload")
def upload(file: UploadFile = File(...)) -> dict:
    ext = os.path.splitext(file.filename or "")[1].lower()
    ext = ext if re.fullmatch(r"\.[a-z0-9]{1,5}", ext) else ""
    name = f"upload-{uuid.uuid4().hex}{ext}"
    path = os.path.join(MEDIA, name)
    with open(path, "wb") as out:
        shutil.copyfileobj(file.file, out)
    cap = cv2.VideoCapture(path)
    opened = cap.isOpened()
    cap.release()
    if not opened:
        os.remove(path)
        raise HTTPException(400, "OpenCV cannot open this recording")
    _start(path)
    return {"ok": True}


@app.post("/api/arm")
def arm(body: Arm) -> dict:
    if _current is None or _current.ended:
        raise HTTPException(409, "no session")
    _current.arm([{"name": i.name, "rect": i.rect} for i in body.items])
    return {"ok": True}


@app.post("/api/disarm")
def disarm() -> dict:
    if _current is not None:
        _current.disarm()
    return {"ok": True}


@app.get("/api/status")
def status() -> dict:
    s = _current
    w = s and s.watch
    items = []
    if w:
        h, width = s.latest.shape[:2]
        items = [
            {
                "name": it.name,
                "rect": [it.rect[0] / width, it.rect[1] / h, it.rect[2] / width, it.rect[3] / h],
                "state": it.state,
                "since": it.since,
                "last_clear": it.last_clear,
            }
            for it in w.items
        ]
    return {
        "source": s is not None,
        "live": bool(s and s.live),
        "ended": bool(s and s.ended),
        "armed": w is not None,
        "view_ok": w.view_ok if w else True,
        "items": items,
    }


@app.get("/api/stream")
def stream() -> StreamingResponse:
    if _current is None:
        raise HTTPException(404, "no session")
    return StreamingResponse(_mjpeg(_current), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/events")
def events() -> list[dict]:
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        rows = c.execute("SELECT * FROM events ORDER BY ts DESC").fetchall()
    return [_event(r) for r in rows]


@app.post("/api/events/{event_id}/review")
def review(event_id: str, body: Review) -> dict:
    with sqlite3.connect(DB) as c:
        c.row_factory = sqlite3.Row
        if c.execute("UPDATE events SET review = ?, note = ? WHERE id = ?", (body.status, body.note, event_id)).rowcount == 0:
            raise HTTPException(404, "unknown event")
        return _event(c.execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone())


@app.get("/media/{name}")
def clip(name: str) -> FileResponse:
    path = os.path.join(MEDIA, name)
    if not CLIP.fullmatch(name) or not os.path.isfile(path):
        raise HTTPException(404, "not found")
    return FileResponse(path, media_type="video/webm")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
