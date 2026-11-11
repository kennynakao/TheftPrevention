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

## Tuning

`theftprevention/core.py` holds five constants: `DARK`, `BLUR_RATIO`, `VIEW_NCC`, `ITEM_NCC` and `MOTION`. `absent_s` (default 3 seconds) is passed to `Watch` or `Session`.

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
