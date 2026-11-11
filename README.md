# Belongings watch

A local demo that watches the spot where a belonging sits in a room, from a webcam or a recording. When that spot stays empty in a clear view, it records a removal candidate with a short evidence clip for a person to review. The watch core is a small Python package that other code can use directly.

It is not a trained classifier. It is a reference-comparison baseline: it compares each watched area with the frame taken when you armed it. It reports removal candidates for a person to review and never concludes theft. It identifies nobody.

## Quick start

```
python -m venv .venv      # Windows
python3 -m venv .venv     # macOS and Linux
```

Activate it (`.venv\Scripts\activate` on Windows, `source .venv/bin/activate` elsewhere), then:

```
pip install -e ".[demo]"
python -m theftprevention
```

Open http://127.0.0.1:8000, use the webcam or upload a recording, draw up to three rectangles around the items, and arm. On Windows, keep the project in a short path, because virtualenvs fail in very deep folders.

## How it works

Each watched area is in one of four states: `present`, `blocked` (something is moving over it), `absent`, or `unusable` (the whole view is not usable).

- View check: a frame is unusable when it is too dark, too blurred, or shifted from the reference. Monitoring pauses until the view recovers.
- Persistence: a state change is accepted only after two consecutive frames agree. An area that stays `absent` for `absent_s` seconds in a clear view gives a removal candidate.
- Returns: the item visible again gives a `returned` event that resolves the candidate. An area that changed but does not match the reference gives a `possible_return` event, which stays open for a person to check.

## Network cameras

A source can be a USB camera index (for example `0`), a video file path, or a stream address. A stream address is RTSP, or an HTTP stream such as the one a phone camera app shows. Any address containing `://` is treated as live, except `file://`.

Live sources are read by a background reader that keeps only the newest frame. Slow processing never works through a backlog of old frames, so delay does not build up. If a stream that was open drops, the reader waits 1 second, then doubles the wait after each failed attempt, up to 30 seconds. A stream that cannot be opened at the start raises an error at once. RTSP streams are read over TCP. Error messages raised by this module never include the user name or password from an address.

```python
from theftprevention.pipeline import Session

session = Session("rtsp://user:password@192.168.1.64:554/Streaming/Channels/102", "media", print)
```

The address format depends on the camera; check the camera's manual or settings for its stream address. The live reader was tested with simulated capture objects and against a simulated network camera (a local HTTP stream), not with physical hardware.

## Tuning

`theftprevention/core.py` holds six constants: `DARK`, `BLUR_RATIO`, `VIEW_NCC`, `ITEM_NCC`, `MOTION` and `JITTER`. `absent_s` (default 3 seconds) is passed to `Watch` or `Session`.

Two of them were set from a test on 83 labelled synthetic clips that include a pixel of camera shake and sensor noise:

- `JITTER` (2 pixels): the stillness test compares each crop with the previous one after allowing a small shift. Without it, camera shake alone made an empty area look like movement, and only 7 of 48 scripted removals were reported.
- `ITEM_NCC` (0.8): how closely a crop must match the reference to count as present. At 0.8 the same test reported 41 of 48 removals, raised no removal alarm on 35 clips without one, and never accepted a moved or swapped item as returned.

Those clips are synthetic composites. Re-check both values on your own footage.

## Privacy

The demo runs locally and binds to 127.0.0.1 only. Footage and the event database stay in `./media`, which is git-ignored. `./media` is relative to the folder you launch from. Uploaded recordings are copied there and kept until you delete them. Record only people who have agreed to it.

## Using the core from other code

With the package installed (`pip install -e .`):

```python
from theftprevention.pipeline import Session

session = Session(0, "media", print)  # 0 is the webcam; a video file path also works
session.arm([{"name": "bag", "rect": [0.4, 0.3, 0.1, 0.2]}])  # x, y, w, h, normalised 0..1
session.run()
```
