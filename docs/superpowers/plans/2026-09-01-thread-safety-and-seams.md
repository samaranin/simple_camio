# Thread Safety and Testable Seams Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the SIFT tracking state a single owning thread, then leave behind typed seams and tests so the same defect class is caught by pytest instead of by a user in the field.

**Architecture:** `SIFTWorker` becomes the sole owner of the detector's mutable state and publishes an immutable `TrackingSnapshot` under the existing lock; every main-thread consumer reads one snapshot per loop iteration instead of reaching into the detector. With that seam in place, the remaining work is mechanical: tests for the deterministic units, typed containers replacing two untyped dicts, CLI/env configuration overrides, and turning three private attribute pokes into explicit parameters.

**Tech Stack:** Python 3.9+, pytest, numpy, OpenCV (`opencv-contrib-python`), MediaPipe, pyglet. Existing suite: 75 tests, all passing.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-09-01-thread-safety-and-seams-design.md`. Read it before starting.
- Branch: `auto-narration`. One commit per task. Do not merge to `master` — a single merge happens after the last task.
- Python floor is 3.9. No `match`, no `X | Y` type unions, no `list[int]` builtin generics in annotations at runtime — use `typing.Optional` and `typing.List`.
- The full suite must pass at the end of every task: `.venv/bin/python -m pytest -q`. It is 75 tests before this plan starts; it only grows.
- **`src/detection/pose_detector.py` tap logic is out of scope.** Do not merge, rename, or alter any method of `PoseDetectorMP` or `PoseDetectorMPEnhanced` except the three transport-only changes in Task 8. See the spec's Non-goal section for why.
- Tasks 1+2, 6, 7 and 8 each end on a device check. If no device is available, say so in the commit body rather than skipping silently.
- Do not add a blanket `try`/`except` around the main-loop body. The spec's Error handling section rejects this explicitly.
- Never use `git checkout --` or `git restore` on a file holding uncommitted work.

## Before Task 1

- [ ] Tag the current tip, so a bad device check later has a known-good point to
      compare against and to `git diff` against:

```bash
git tag pre-thread-safety
git tag --list 'pre-thread-safety'
```

Expected: the tag name prints. It stays local; there is nothing to push.

---

### Task 1: `TrackingSnapshot` and its publisher

**Files:**
- Create: `src/core/tracking.py`
- Modify: `src/core/workers.py:252-352` (`SIFTWorker.__init__`, `SIFTWorker.run`)
- Test: `tests/test_tracking_snapshot.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.core.tracking.TrackingSnapshot` — frozen dataclass, fields `H`, `rect_pts`, `tracking`, `age`, `status_text`, `detect_generation`, all with defaults, so `TrackingSnapshot()` is the valid "nothing detected yet" value.
  - `SIFTWorker.snapshot` — attribute holding the current `TrackingSnapshot`. Read it only while holding `SIFTWorker.lock`.
  - `SIFTWorker._publish()` — called on the worker thread only; rebuilds and stores the snapshot.

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracking_snapshot.py`:

```python
"""The SIFT worker owns the detector's mutable state and publishes snapshots of it."""

import dataclasses
import queue
import threading

import numpy as np
import pytest

from src.core.tracking import TrackingSnapshot
from src.core.workers import SIFTWorker


class FakeDetector:
    """Stands in for SIFTModelDetectorMP: same attributes, no OpenCV."""

    def __init__(self):
        self.H = None
        self.requires_homography = True
        self.last_rect_pts = None
        self.frames_since_last_detection = 0
        self.homography_updated = False

    def get_tracking_status(self):
        if self.requires_homography:
            return "SEARCHING FOR MAP"
        return f"TRACKING (Age:{self.frames_since_last_detection})"

    def find_map(self):
        """Simulate a successful detection, the way the real detector does."""
        self.H = np.eye(3)
        self.requires_homography = False
        self.last_rect_pts = np.zeros((4, 1, 2))
        self.frames_since_last_detection = 0
        self.homography_updated = True


@pytest.fixture
def worker():
    return SIFTWorker(FakeDetector(), queue.Queue(maxsize=1), threading.Lock())


def test_default_snapshot_reports_no_map(worker):
    """Before the worker has run, consumers must still get a usable snapshot."""
    assert worker.snapshot.H is None
    assert worker.snapshot.tracking is False
    assert worker.snapshot.detect_generation == 0


def test_publish_copies_detector_state(worker):
    worker.sift_detector.find_map()
    worker._publish()

    snap = worker.snapshot
    assert snap.tracking is True
    assert snap.H is not None
    assert snap.rect_pts is not None
    assert snap.status_text.startswith("TRACKING")


def test_snapshot_is_frozen(worker):
    """A consumer cannot corrupt a snapshot it is holding."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        worker.snapshot.tracking = True


def test_publish_replaces_rather_than_mutates(worker):
    """A consumer holding an old snapshot keeps a consistent view of the old state."""
    worker._publish()
    first = worker.snapshot

    worker.sift_detector.find_map()
    worker._publish()

    assert worker.snapshot is not first
    assert first.tracking is False, "the snapshot handed out earlier was mutated"
    assert worker.snapshot.tracking is True


def test_generation_increments_once_per_detection(worker):
    """The flash notification survives without a flag the consumer has to clear."""
    worker._publish()
    assert worker.snapshot.detect_generation == 0

    worker.sift_detector.find_map()
    worker._publish()
    assert worker.snapshot.detect_generation == 1

    # No new detection: publishing again must not re-trigger a flash.
    worker._publish()
    assert worker.snapshot.detect_generation == 1

    worker.sift_detector.find_map()
    worker._publish()
    assert worker.snapshot.detect_generation == 2


def test_generation_survives_two_detections_between_reads(worker):
    """The lost-notification race: two detections with no consumer read in between."""
    worker.sift_detector.find_map()
    worker._publish()
    worker.sift_detector.find_map()
    worker._publish()

    assert worker.snapshot.detect_generation == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tracking_snapshot.py -v`
Expected: FAIL at collection with `ModuleNotFoundError: No module named 'src.core.tracking'`.

- [ ] **Step 3: Create the snapshot module**

Create `src/core/tracking.py`:

```python
"""
Tracking state published across threads.

The SIFT detector's tracking state is a cluster of related attributes that only
the SIFTWorker thread may touch. Consumers on the main thread read an immutable
snapshot of them instead, so a guard and the read it guards can never disagree.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TrackingSnapshot:
    """
    One consistent view of the SIFT tracking state.

    Attributes:
        H: Homography matrix, or None when the map is not currently tracked.
        rect_pts: Projected template corners in camera coordinates, or None.
        tracking: True while the detector holds a valid homography.
        age: Frames since the last successful detection.
        status_text: Display string, rendered by the thread that owns the state.
        detect_generation: Monotonic counter, incremented once per successful
            homography. A consumer flashes the tracking rectangle when the value
            differs from the one it last saw; nothing has to be cleared, so a
            detection landing between two reads cannot be missed.
    """

    H: Optional[Any] = None
    rect_pts: Optional[Any] = None
    tracking: bool = False
    age: int = 0
    status_text: str = "SEARCHING FOR MAP"
    detect_generation: int = 0
```

`Any` rather than `np.ndarray`: the annotation is documentation here, and keeping
numpy out of this module leaves it importable by anything.

- [ ] **Step 4: Publish snapshots from the worker**

In `src/core/workers.py`, add the import next to the existing ones:

```python
from src.core.tracking import TrackingSnapshot
```

In `SIFTWorker.__init__`, after `self.stop_event = stop_event`:

```python
        # Snapshot of the detector's tracking state, for consumers on other
        # threads. Read it only while holding self.lock.
        self.snapshot = TrackingSnapshot()
        self._generation = 0
```

Add the publisher after `_prepare_detection_attempts`:

```python
    def _publish(self):
        """
        Copy the detector's tracking state into a fresh snapshot.

        Runs on the worker thread, which is the only thread allowed to read the
        detector's attributes. The homography_updated flag is consumed here and
        converted into a monotonic counter, so no consumer has to clear it.
        """
        d = self.sift_detector

        if getattr(d, 'homography_updated', False):
            self._generation += 1
            d.homography_updated = False

        snapshot = TrackingSnapshot(
            H=d.H,
            rect_pts=d.last_rect_pts,
            tracking=not d.requires_homography,
            age=d.frames_since_last_detection,
            status_text=d.get_tracking_status(),
            detect_generation=self._generation,
        )

        with self.lock:
            self.snapshot = snapshot
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tracking_snapshot.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 6: Publish on every worker iteration**

In `SIFTWorker.run`, the loop body ends with the broad `except` that logs
`"SIFT worker error"`. Publish after it, so a snapshot is issued whether the
iteration detected, validated, or failed. The `continue` in the validation
branch skips the rest of the body, so that branch needs its own publish.

Replace the `else: continue` inside the validation branch:

```python
                        else:
                            self._publish()
                            continue
```

And add a publish as the last statement of the loop body, after the
`except Exception as e: logger.error(f"SIFT worker error: {e}")` block:

```python
            self._publish()
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 81 tests (75 + 6).

- [ ] **Step 8: Commit**

```bash
git add src/core/tracking.py src/core/workers.py tests/test_tracking_snapshot.py
git commit -m "feat: SIFTWorker publishes an immutable tracking snapshot

The detector's eight tracking attributes had no owning thread. The worker
now owns them and hands consumers a frozen snapshot, so a guard and the
read it guards cannot disagree.

homography_updated is consumed here and reissued as a monotonic counter,
which removes the lost-notification race in the rectangle flash: nothing
downstream has to clear a shared flag."
```

---

### Task 2: Main loop and display consume the snapshot

This is the task that actually removes the two reachable defects. Task 1 only
built the mechanism.

**Files:**
- Modify: `simple_camio.py` — `feed_worker_queues:176-211`, `handle_keyboard_input:311-357`, `update_sleep_mode:431-470`, `process_map_detection:474-522`, `run_main_loop:622-747`
- Modify: `src/ui/display.py:19-49` (`draw_map_tracking`), `src/ui/display.py:52-72` (`draw_ui_overlay`)
- Test: `tests/test_display_snapshot.py`, `tests/test_keyboard_input.py`

**Interfaces:**
- Consumes: `TrackingSnapshot` and `SIFTWorker.snapshot` from Task 1.
- Produces:
  - `draw_map_tracking(display_img, snapshot, interact, rect_flash_remaining)` — second parameter is now a `TrackingSnapshot`, not a detector.
  - `draw_ui_overlay(display_img, snapshot, gesture_status, timer, fps_state, cap)` — same substitution.
  - `feed_worker_queues(frame, gray, workers, snapshot)` — fourth parameter is now a snapshot.
  - `update_sleep_mode(components, cap, gesture_loc, sleep_state, snapshot)` — new trailing parameter.
  - `process_map_detection(components, workers, display_img, rect_flash_remaining, gesture_loc, gesture_status, last_double_tap_ts, prof_times, hand_state, snapshot)` — new trailing parameter.
  - `get_tracking_snapshot(workers)` in `simple_camio.py` — returns the current snapshot under the lock.

- [ ] **Step 1: Write the failing display test**

Create `tests/test_display_snapshot.py`:

```python
"""The drawing functions read a snapshot, never the live detector."""

import numpy as np
import pytest

from src.core.tracking import TrackingSnapshot
from src.ui.display import draw_map_tracking


class FakeInteract:
    """Carries only the map shape draw_map_tracking needs."""

    class _Img:
        shape = (100, 100, 3)

    image_map_color = _Img()


@pytest.fixture
def blank():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_draws_from_rect_points_when_present(blank):
    pts = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.float32)
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=pts, tracking=True)

    img, flash = draw_map_tracking(blank, snap, FakeInteract(), 0)

    assert img is not None
    assert flash == 0


def test_flash_counter_decrements_while_flashing(blank):
    pts = np.array([[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.float32)
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=pts, tracking=True)

    _, flash = draw_map_tracking(blank, snap, FakeInteract(), 5)

    assert flash == 4


def test_no_rect_points_falls_back_to_homography(blank):
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=None, tracking=True)

    img, _ = draw_map_tracking(blank, snap, FakeInteract(), 0)

    assert img is not None


def test_empty_snapshot_does_not_raise(blank):
    """The old code passed None into the draw helpers and took down the loop."""
    img, _ = draw_map_tracking(blank, TrackingSnapshot(), FakeInteract(), 0)

    assert img is not None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_display_snapshot.py -v`
Expected: FAIL — `draw_map_tracking` still expects a detector, so
`TrackingSnapshot` has no `last_rect_pts` and the `getattr` default sends every
case down the homography branch; `test_draws_from_rect_points_when_present`
fails on the flash assertion and `test_empty_snapshot_does_not_raise` raises
inside `draw_rectangle_on_image` with `H=None`.

- [ ] **Step 3: Rewrite the two drawing functions**

In `src/ui/display.py`, replace `draw_map_tracking` (lines 19-49) with:

```python
def draw_map_tracking(display_img, snapshot, interact, rect_flash_remaining):
    """
    Draw the map tracking rectangle on the display image.

    Args:
        display_img: Image to draw on
        snapshot (TrackingSnapshot): Consistent view of the tracking state
        interact: Interaction policy with map shape
        rect_flash_remaining (int): Frames remaining for flash effect

    Returns:
        tuple: (updated_image, updated_flash_remaining)
    """
    rect_pts = snapshot.rect_pts

    if rect_pts is not None:
        if rect_flash_remaining > 0:
            display_img = draw_rectangle_from_points(
                display_img, rect_pts,
                color=UIConfig.COLOR_YELLOW, thickness=5
            )
            rect_flash_remaining -= 1
        else:
            display_img = draw_rectangle_from_points(
                display_img, rect_pts,
                color=UIConfig.COLOR_GREEN, thickness=3
            )
    elif snapshot.H is not None:
        display_img = draw_rectangle_on_image(
            display_img, interact.image_map_color.shape, snapshot.H
        )

    return display_img, rect_flash_remaining
```

Three changes, each load-bearing: `rect_pts` is read into a local once so the
guard and the use cannot disagree; the `else` became `elif snapshot.H is not
None`, so a snapshot with neither value draws nothing instead of passing `None`
into the helper; and the detector is gone from the signature.

In `draw_ui_overlay`, replace the parameter and line 69:

```python
def draw_ui_overlay(display_img, snapshot, gesture_status, timer, fps_state, cap):
```

```python
    # Tracking status, rendered by the thread that owns the tracking state
    status_text = snapshot.status_text
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_display_snapshot.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Write the failing keyboard test**

Create `tests/test_keyboard_input.py`:

```python
"""The 'h' key requests re-detection through the worker, not by poking state."""

import queue

import numpy as np
import pytest

import simple_camio


class FakeDetector:
    def __init__(self):
        self.requires_homography = False
        self.last_rect_pts = np.zeros((4, 1, 2))


class FakeSiftWorker:
    def __init__(self):
        self.redetect_calls = 0

    def trigger_redetect(self):
        self.redetect_calls += 1


class FakeAudioWorker:
    def __init__(self):
        self.commands = []

    def enqueue_command(self, command):
        self.commands.append(command)


@pytest.fixture
def rig():
    detector = FakeDetector()
    return {
        'components': {'model_detector': detector},
        'workers': {
            'sift_queue': queue.Queue(maxsize=1),
            'sift_worker': FakeSiftWorker(),
            'audio_worker': FakeAudioWorker(),
        },
        'frame': np.zeros((16, 16, 3), dtype=np.uint8),
    }


def test_h_asks_the_worker_to_redetect(rig):
    simple_camio.handle_keyboard_input(
        ord('h'), _StopEvent(), rig['frame'], rig['workers'], rig['components']
    )

    assert rig['workers']['sift_worker'].redetect_calls == 1


def test_h_does_not_touch_detector_state(rig):
    """Those writes raced the worker thread; the worker owns this state now."""
    detector = rig['components']['model_detector']

    simple_camio.handle_keyboard_input(
        ord('h'), _StopEvent(), rig['frame'], rig['workers'], rig['components']
    )

    assert detector.requires_homography is False
    assert detector.last_rect_pts is not None


def test_q_signals_shutdown(rig):
    stop = _StopEvent()

    keep_going = simple_camio.handle_keyboard_input(
        ord('q'), stop, rig['frame'], rig['workers'], rig['components']
    )

    assert keep_going is False
    assert stop.is_set()


class _StopEvent:
    def __init__(self):
        self._set = False

    def set(self):
        self._set = True

    def is_set(self):
        return self._set
```

- [ ] **Step 6: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_keyboard_input.py -v`
Expected: `test_h_does_not_touch_detector_state` FAILS — the handler still sets
`requires_homography = True` and `last_rect_pts = None`. The other two pass.

- [ ] **Step 7: Delete the two racing writes**

In `simple_camio.py`, in `handle_keyboard_input`, delete exactly these two lines
(currently 335-336):

```python
        components['model_detector'].requires_homography = True
        components['model_detector'].last_rect_pts = None
```

Nothing replaces them. The call to `workers['sift_worker'].trigger_redetect()`
below already carries the request, and the worker applies it on its own thread.

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_keyboard_input.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 9: Thread the snapshot through the main loop**

In `simple_camio.py`, add the import:

```python
from src.core.tracking import TrackingSnapshot
```

Add the reader next to `get_pose_results`:

```python
def get_tracking_snapshot(workers):
    """
    Read the current tracking snapshot under the shared lock.

    Returns:
        TrackingSnapshot: One consistent view of the SIFT tracking state.
    """
    sift_worker = workers['sift_worker']
    with workers['lock']:
        return sift_worker.snapshot
```

`feed_worker_queues` — replace the `model_detector` parameter and the double
read at line 200:

```python
def feed_worker_queues(frame, gray, workers, snapshot):
    """
    Feed frames to worker queues for background processing.

    Args:
        frame: Color camera frame
        gray: Grayscale camera frame
        workers (dict): Worker threads and queues
        snapshot (TrackingSnapshot): Current tracking state
    """
```

```python
    # Feed pose worker with frame and current homography. Read once: the old
    # code tested and re-read the attribute, so None could reach the queue.
    H = snapshot.H
    H_current = H if H is not None else IDENTITY_3
```

`update_sleep_mode` — add the parameter and replace the `map_tracked` computation:

```python
def update_sleep_mode(components, cap, gesture_loc, sleep_state, snapshot):
```

```python
    now = time.time()
    map_tracked = snapshot.tracking and snapshot.H is not None
```

Delete the now-unused `model_detector = components['model_detector']` line above it.

`process_map_detection` — add `snapshot` as the last parameter, replace the
`components['model_detector'].H is None` test with `snapshot.H is None`, and
**delete** the age bookkeeping (currently 505-508):

```python
        # Map detected - increment age counter
        try:
            components['model_detector'].frames_since_last_detection += 1
        except Exception:
            components['model_detector'].frames_since_last_detection = 1
```

The worker owns that counter now. Pass the snapshot into the draw call:

```python
        display_img, rect_flash_remaining = draw_map_tracking(
            display_img, snapshot, components['interact'], rect_flash_remaining
        )
```

`run_main_loop` — add the generation tracker beside the other loop state:

```python
    last_detect_generation = 0
```

In the loop body, take the snapshot once, right after the early-exit check, and
use it everywhere below:

```python
        # One consistent view of tracking state for this whole iteration
        snapshot = get_tracking_snapshot(workers)

        # Feed worker queues
        t = time.time()
        feed_worker_queues(frame, gray, workers, snapshot)
        prof_times['feed'] += time.time() - t
```

Replace the `homography_updated` block (currently 696-698) with the generation
comparison:

```python
        # A new homography means flash the rectangle. Comparing a monotonic
        # counter cannot lose a detection the way clearing a shared flag did.
        if snapshot.detect_generation != last_detect_generation:
            rect_flash_remaining = UIConfig.RECT_FLASH_FRAMES
            last_detect_generation = snapshot.detect_generation
```

Pass the snapshot into the three call sites:

```python
        display_img, rect_flash_remaining, last_double_tap_ts, hand_state = process_map_detection(
            components, workers, display_img, rect_flash_remaining,
            gesture_loc, gesture_status, last_double_tap_ts, prof_times, hand_state,
            snapshot
        )

        sleep_state = update_sleep_mode(components, cap, gesture_loc, sleep_state, snapshot)

        # Draw UI overlay
        t = time.time()
        timer, fps_state = draw_ui_overlay(display_img, snapshot,
                                           gesture_status, timer, fps_state, cap)
        prof_times['ui'] += time.time() - t
```

- [ ] **Step 10: Confirm no consumer reaches into the detector any more**

Run:

```bash
grep -n "model_detector\." simple_camio.py src/ui/display.py src/core/display_thread.py
```

Expected: no matches. Every line the analysis flagged (`simple_camio.py:200`,
`335`, `336`, `442`, `443`, `506`, `508`, `696`, `698`; `src/ui/display.py:32`,
`35`, `41`, `46`, `69`) is gone. If a match remains, it is a missed consumer —
convert it before committing.

- [ ] **Step 11: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 88 tests (81 + 7).

- [ ] **Step 12: Device check**

Run the app against a real map and camera:

```bash
.venv/bin/python simple_camio.py --camera 0 --input1 models/UkraineMap/UkraineMap.json
```

Confirm, in order: the status line reads `SEARCHING FOR MAP`, then flips to
`TRACKING` with a yellow flash on first detection; the rectangle tracks the map;
pointing at a zone speaks it; a double tap registers; `h` re-detects and flashes
again; `q` exits with the goodbye message. Then headless:

```bash
xvfb-run -a .venv/bin/python simple_camio.py --headless --camera 0 --input1 models/UkraineMap/UkraineMap.json
```

Confirm it starts, speaks a zone, and stops cleanly on Ctrl+C.

Expected age change: the `Age:` number in the status line now counts
SIFT-processed frames rather than main-loop iterations, so it climbs more slowly
than before. That is the accepted behaviour change from the spec, not a defect.

- [ ] **Step 13: Commit**

```bash
git add simple_camio.py src/ui/display.py tests/test_display_snapshot.py tests/test_keyboard_input.py
git commit -m "fix: read SIFT tracking state through a snapshot, not the detector

draw_map_tracking tested last_rect_pts and then re-read it, so the worker
could null it in between and send None into the draw helper. The main loop
body has no try/except, so that exception left run_main_loop and exited the
process through cleanup - a systemd restart on the Pi. The same shape of
bug sat in feed_worker_queues, where H was read twice.

Every consumer now takes one snapshot per iteration. The 'h' handler drops
its two direct writes, keeping only the trigger_redetect() call it already
made, and the age counter belongs to the worker alone."
```

---

### Task 3: Tests for `InteractionPolicy2D`

Production code is untouched from here until Task 6, so Tasks 3-5 cannot regress
the device.

**Files:**
- Test: `tests/test_interaction_policy.py`
- Modify: `tests/conftest.py` (add one fixture)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `zone_map_model` fixture in `tests/conftest.py` — returns a dict with
  a `filename` key pointing at a written PNG, suitable for
  `InteractionPolicy2D(model)`.

- [ ] **Step 1: Add the fixture**

Append to `tests/conftest.py`:

```python
@pytest.fixture
def zone_map_model(tmp_path):
    """
    A model whose zone map is three flat colour bands.

    InteractionPolicy2D reads model['filename'] with cv.imread, so the file has
    to exist on disk. Bands run top to bottom: red, green, blue.
    """
    import cv2 as cv
    import numpy as np

    img = np.zeros((90, 30, 3), dtype=np.uint8)
    img[0:30, :] = (0, 0, 255)    # BGR red
    img[30:60, :] = (0, 255, 0)   # BGR green
    img[60:90, :] = (255, 0, 0)   # BGR blue

    path = tmp_path / 'zones.png'
    cv.imwrite(str(path), img)

    return {'filename': str(path)}
```

- [ ] **Step 2: Write the failing test**

Create `tests/test_interaction_policy.py`:

```python
"""Zone mapping and the mode filter that smooths it."""

import numpy as np
import pytest

from src.config import InteractionConfig
from src.core.interaction_policy import InteractionPolicy2D


@pytest.fixture
def policy(zone_map_model):
    return InteractionPolicy2D(zone_map_model)


def _touch(x, y):
    """A position on the map, close enough in z to count as touching."""
    return np.array([x, y, 0.0])


def test_filter_starts_empty(policy):
    assert (policy.zone_filter == -1).all()


def test_repeated_touch_settles_on_one_zone(policy):
    """Fill the ring buffer with one zone; the mode has to be that zone."""
    zone = None
    for _ in range(policy.ZONE_FILTER_SIZE):
        zone = policy.push_gesture(_touch(15, 15))

    assert zone >= 0
    settled = zone

    # Another sample of the same zone cannot change the answer.
    assert policy.push_gesture(_touch(15, 15)) == settled


def test_distinct_colours_map_to_distinct_zones(policy):
    for _ in range(policy.ZONE_FILTER_SIZE):
        top = policy.push_gesture(_touch(15, 15))
    policy.reset_zone_filter()
    for _ in range(policy.ZONE_FILTER_SIZE):
        middle = policy.push_gesture(_touch(15, 45))

    assert top != middle


def test_single_outlier_does_not_flip_the_zone(policy):
    """The mode filter exists to absorb exactly this."""
    for _ in range(policy.ZONE_FILTER_SIZE):
        settled = policy.push_gesture(_touch(15, 15))

    assert policy.push_gesture(_touch(15, 45)) == settled


def test_hand_too_high_reports_no_zone(policy):
    """Beyond the z threshold the user is not touching the map."""
    above = np.array([15, 15, InteractionConfig.Z_THRESHOLD + 10.0])

    assert policy.push_gesture(above) == -1


def test_reset_clears_the_buffer(policy):
    for _ in range(policy.ZONE_FILTER_SIZE):
        policy.push_gesture(_touch(15, 15))

    policy.reset_zone_filter()

    assert (policy.zone_filter == -1).all()
    assert policy.zone_filter_cnt == 0
```

- [ ] **Step 3: Run the tests**

Run: `.venv/bin/python -m pytest tests/test_interaction_policy.py -v`
Expected: PASS, 6 tests. These characterize existing behaviour, so they should
pass on the first run.

If `test_single_outlier_does_not_flip_the_zone` fails, read
`InteractionConfig.ZONE_FILTER_SIZE`: with a size of 1 or 2 a single sample can
legitimately carry the mode. Do not change the config to suit the test — adjust
the test to the real buffer size and note it in the commit body.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 94 tests (88 + 6).

- [ ] **Step 5: Commit**

```bash
git add tests/conftest.py tests/test_interaction_policy.py
git commit -m "test: characterize zone mapping and the mode filter

Colour to zone, the ring buffer that absorbs a single outlier, the z
threshold, and reset. Nothing in src changes; this is the net that Task 6
edits land under."
```

---

### Task 4: Regression baseline for `TapClassifier`, and the collected-data findings

**Files:**
- Create: `tests/test_tap_classifier_regression.py`
- Create: `tests/data/tap_predictions_baseline.json`
- Create: `docs/tap-data-findings.md`
- Move: `src/tap_classifier/test_tap_classifier.py` → `tools/check_tap_classifier.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `tests/data/tap_predictions_baseline.json` — a list of objects
  `{"features": [...18 floats...], "probability": float}`, one per collected
  sample, in file-then-index order.

- [ ] **Step 1: Move the stray script out of `src/`**

`src/tap_classifier/test_tap_classifier.py` is a script with a `main()`, but its
`test_*` name means pytest collects it, loads MediaPipe, and runs it as part of
the suite. Move it where its name is not a collection trigger:

```bash
mkdir -p tools
git mv src/tap_classifier/test_tap_classifier.py tools/check_tap_classifier.py
```

Fix its imports if it used a relative one, and confirm it still runs standalone:

```bash
.venv/bin/python tools/check_tap_classifier.py
```

- [ ] **Step 2: Confirm the suite shrank by exactly those tests**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 90 tests — 94 from Task 3 minus the 4 `test_*` functions that
file contributed (`test_classifier_basic`, `test_classifier_training`,
`test_feature_importance`, `test_pose_detector_integration`). If the count is
different, reconcile before continuing.

- [ ] **Step 3: Generate the baseline**

The shipped model plus the collected vectors are both in the repo, so the
baseline is reproducible. Write it once:

```bash
.venv/bin/python - <<'PY'
import glob, json, os

from src.tap_classifier.tap_classifier import TapClassifier

clf = TapClassifier()
clf.load_model('models/tap_model.json')

rows = []
for path in sorted(glob.glob('data/tap_dataset/*.json')):
    with open(path, encoding='utf-8') as f:
        payload = json.load(f)
    for sample in payload['samples']:
        rows.append({
            'features': sample['features'],
            'probability': float(clf.predict(sample['features'])),
        })

os.makedirs('tests/data', exist_ok=True)
with open('tests/data/tap_predictions_baseline.json', 'w', encoding='utf-8') as f:
    json.dump(rows, f, indent=1)

print(f'{len(rows)} rows written')
PY
```

Expected: `241 rows written`.

- [ ] **Step 4: Write the test**

Create `tests/test_tap_classifier_regression.py`:

```python
"""
The shipped classifier must keep predicting what it predicts today.

The 241 vectors in data/tap_dataset are the only real recorded input this
project has. They pin the classifier's arithmetic so a refactor cannot drift
it silently. They do NOT pin the tap state machine: the collector records the
finished feature vector after the machine has already decided a press ended,
so this data is the machine's output and cannot be replayed as its input.
"""

import json
from pathlib import Path

import pytest

from src.tap_classifier.tap_classifier import TapClassifier

BASELINE = Path(__file__).parent / 'data' / 'tap_predictions_baseline.json'
MODEL = Path(__file__).parents[1] / 'models' / 'tap_model.json'


@pytest.fixture(scope='module')
def baseline():
    with BASELINE.open(encoding='utf-8') as f:
        return json.load(f)


@pytest.fixture(scope='module')
def classifier():
    clf = TapClassifier()
    clf.load_model(str(MODEL))
    return clf


def test_baseline_covers_every_collected_sample(baseline):
    assert len(baseline) == 241


def test_predictions_match_the_baseline(classifier, baseline):
    drifted = []
    for i, row in enumerate(baseline):
        got = float(classifier.predict(row['features']))
        if abs(got - row['probability']) > 1e-9:
            drifted.append((i, row['probability'], got))

    assert not drifted, f'{len(drifted)} prediction(s) drifted, first: {drifted[:3]}'


def test_probabilities_stay_in_range(classifier, baseline):
    for row in baseline:
        prob = float(classifier.predict(row['features']))
        assert 0.0 <= prob <= 1.0
```

- [ ] **Step 5: Run the test**

Run: `.venv/bin/python -m pytest tests/test_tap_classifier_regression.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 6: Record the two dataset findings**

Both belong to the out-of-scope detector file, so they are written down rather
than fixed. Create `docs/tap-data-findings.md`:

```markdown
# Findings on the collected tap data

Recorded 2026-09-01 while adding the classifier regression baseline. Neither
item is fixed here: both live in `src/detection/pose_detector.py`, which the
thread-safety plan deliberately leaves alone.

## The dataset is almost entirely positive

241 samples across three sessions on 2025-10-27: **237 positive, 4 negative.**

`models/tap_model.json` is therefore trained with almost no counter-examples
and should be expected to over-predict taps. Any accuracy figure quoted from
this data is close to meaningless — a classifier answering "tap" unconditionally
scores 98.3% on it.

Before retraining, collect negatives deliberately: hovering, pointing without
pressing, dragging a finger across zones, and withdrawing the hand mid-press.

## No sample came from the enhanced detector

Every one of the 241 samples carries `metadata.detector == "base"`, even though
`CombinedPoseDetector.__init__` hands the base and enhanced detectors the same
collector object, and `_collect_enhanced_tap_data_positive` /
`_collect_enhanced_tap_data_negative` exist in
`PoseDetectorMPEnhanced`.

**Open question:** can the enhanced collection path fire at all? If it cannot,
training data for the enhanced detector is uncollectable, and the enhanced
detector's classifier features are exercised only at runtime. Worth confirming
against a live session before any work on the classifier.
```

- [ ] **Step 7: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 93 tests (90 + 3).

- [ ] **Step 8: Commit**

```bash
git add tests/test_tap_classifier_regression.py tests/data/tap_predictions_baseline.json docs/tap-data-findings.md tools/check_tap_classifier.py
git commit -m "test: pin classifier predictions to the collected vectors

The 241 recorded samples now serve as a regression baseline, and the manual
check script leaves src/ so pytest stops loading MediaPipe to run it.

Two things the data revealed are written to docs/tap-data-findings.md rather
than fixed, since both sit in the detector file this work leaves alone: the
set is 237 positive against 4 negative, and not one sample came from the
enhanced detector despite it sharing the collector."
```

---

### Task 5: Tests for `src/core/utils.py`

**Files:**
- Test: `tests/test_core_utils.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: nothing later tasks depend on.

- [ ] **Step 1: Read the functions under test**

Run:

```bash
sed -n '116,283p' src/core/utils.py
```

Read `load_map_parameters`, `is_gesture_valid`, `normalize_gesture_location` and
`color_to_index` before writing assertions. The test below encodes the contracts
their docstrings state; if an implementation disagrees, characterize what the
code actually does and note the discrepancy in the commit body rather than
changing `src/`.

- [ ] **Step 2: Write the test**

Create `tests/test_core_utils.py`:

```python
"""Utility helpers used across the pipeline."""

import json

import numpy as np
import pytest

from src.core.utils import (
    color_to_index,
    is_gesture_valid,
    load_map_parameters,
    normalize_gesture_location,
)


class TestNormalizeGestureLocation:
    def test_none_stays_none(self):
        assert normalize_gesture_location(None) is None

    def test_flat_triple_is_returned_as_three_elements(self):
        out = normalize_gesture_location(np.array([1.0, 2.0, 3.0]))

        assert out is not None
        assert np.asarray(out).size == 3

    def test_nested_array_is_flattened_to_three(self):
        out = normalize_gesture_location(np.array([[1.0, 2.0, 3.0]]))

        assert np.asarray(out).size == 3


class TestIsGestureValid:
    def test_none_is_not_valid(self):
        assert is_gesture_valid(None) is False

    def test_a_real_position_is_valid(self):
        assert is_gesture_valid(np.array([1.0, 2.0, 3.0])) is True


class TestColorToIndex:
    def test_distinct_colours_give_distinct_indices(self):
        assert color_to_index((255, 0, 0)) != color_to_index((0, 255, 0))

    def test_same_colour_gives_the_same_index(self):
        assert color_to_index((12, 34, 56)) == color_to_index((12, 34, 56))


class TestLoadMapParameters:
    def test_reads_the_model_block(self, tmp_path):
        path = tmp_path / 'model.json'
        path.write_text(json.dumps({
            'model': {'modelType': 'sift_2d_mediapipe', 'name': 'Test'}
        }), encoding='utf-8')

        model = load_map_parameters(str(path))

        assert model['modelType'] == 'sift_2d_mediapipe'
        assert model['name'] == 'Test'

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises((FileNotFoundError, OSError)):
            load_map_parameters(str(tmp_path / 'nope.json'))
```

- [ ] **Step 3: Run the test**

Run: `.venv/bin/python -m pytest tests/test_core_utils.py -v`
Expected: PASS, 9 tests. Where a helper's real behaviour differs from the
docstring contract, fix the test to match the code and say so in the commit.

- [ ] **Step 4: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 102 tests (93 + 9).

- [ ] **Step 5: Commit**

```bash
git add tests/test_core_utils.py
git commit -m "test: cover the core utility helpers

Gesture normalization and validity, colour-to-zone-index, and map parameter
loading. These are the helpers Task 6's signature changes run through."
```

---

### Task 6: Typed component containers

**Files:**
- Create: `src/core/containers.py`
- Modify: `simple_camio.py` — `initialize_system:43-97`, `create_worker_threads:99-156`, and every function taking `components` or `workers`
- Test: `tests/test_containers.py`

**Interfaces:**
- Consumes: `TrackingSnapshot` (Task 1) is unaffected by this task.
- Produces:
  - `src.core.containers.Components` — frozen dataclass, fields in this order:
    `model`, `cam_port`, `model_detector`, `pose_detector`, `gesture_detector`,
    `motion_filter`, `interact`, `camio_player`, `crickets_player`,
    `heartbeat_player`.
  - `src.core.containers.Workers` — frozen dataclass, fields in this order:
    `audio_worker`, `pose_worker`, `sift_worker`, `pose_queue`, `sift_queue`,
    `lock`.
  - `initialize_system(model_path, cam_port=None) -> Components`
  - `create_worker_threads(components, stop_event) -> Workers`

- [ ] **Step 1: Write the failing test**

Create `tests/test_containers.py`:

```python
"""The two containers that replaced untyped dicts."""

import dataclasses

import pytest

from src.core.containers import Components, Workers


def _components(**overrides):
    fields = {name: object() for name in (
        'model', 'cam_port', 'model_detector', 'pose_detector',
        'gesture_detector', 'motion_filter', 'interact', 'camio_player',
        'crickets_player', 'heartbeat_player',
    )}
    fields.update(overrides)
    return Components(**fields)


def test_components_exposes_every_field():
    c = _components()

    for name in ('model', 'cam_port', 'model_detector', 'pose_detector',
                 'gesture_detector', 'motion_filter', 'interact',
                 'camio_player', 'crickets_player', 'heartbeat_player'):
        assert getattr(c, name) is not None


def test_components_rejects_an_unknown_field():
    """A mistyped name fails at construction, not at some far-away use."""
    with pytest.raises(TypeError):
        _components(camio_playr=object())


def test_components_rejects_a_missing_field():
    with pytest.raises(TypeError):
        Components(model=object())


def test_components_is_frozen():
    c = _components()

    with pytest.raises(dataclasses.FrozenInstanceError):
        c.model = object()


def test_workers_exposes_every_field():
    w = Workers(
        audio_worker=object(), pose_worker=object(), sift_worker=object(),
        pose_queue=object(), sift_queue=object(), lock=object(),
    )

    assert w.audio_worker is not None
    assert w.lock is not None


def test_workers_is_frozen():
    w = Workers(
        audio_worker=object(), pose_worker=object(), sift_worker=object(),
        pose_queue=object(), sift_queue=object(), lock=object(),
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        w.lock = object()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_containers.py -v`
Expected: FAIL at collection — `No module named 'src.core.containers'`.

- [ ] **Step 3: Create the containers**

Create `src/core/containers.py`:

```python
"""
Typed containers for the objects the main loop threads through its helpers.

These replaced two untyped dicts. A frozen dataclass does not make a mistyped
attribute a compile-time error - Python has no such thing - but a wrong field
name now fails where the container is built rather than at some far-away use,
an editor and ruff can see the misspelling, and each helper's signature states
what it actually requires.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Components:
    """Everything initialize_system() builds from a map model."""

    model: dict
    cam_port: Any
    model_detector: Any
    pose_detector: Any
    gesture_detector: Any
    motion_filter: Any
    interact: Any
    camio_player: Any
    crickets_player: Any
    heartbeat_player: Any


@dataclass(frozen=True)
class Workers:
    """The background threads and the queues and lock they share."""

    audio_worker: Any
    pose_worker: Any
    sift_worker: Any
    pose_queue: Any
    sift_queue: Any
    lock: Any
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_containers.py -v`
Expected: PASS, 6 tests.

- [ ] **Step 5: Return `Components` from `initialize_system`**

In `simple_camio.py`, add the import:

```python
from src.core.containers import Components, Workers
```

Replace the `return { ... }` at the end of `initialize_system` (currently
85-97) with:

```python
    return Components(
        model=model,
        cam_port=cam_port,
        model_detector=model_detector,
        pose_detector=pose_detector,
        gesture_detector=gesture_detector,
        motion_filter=motion_filter,
        interact=interact,
        camio_player=camio_player,
        crickets_player=crickets_player,
        heartbeat_player=heartbeat_player,
    )
```

Update its docstring `Returns:` line to `Components: Initialized system components`.

- [ ] **Step 6: Return `Workers` from `create_worker_threads`**

Replace that function's returned dict with a `Workers(...)` construction using
the same values, and update its docstring the same way. Field names map
one-to-one to the old keys.

- [ ] **Step 7: Convert every index site**

Rewrite `components['x']` as `components.x` and `workers['x']` as `workers.x`
throughout `simple_camio.py`. There are 33 and 26 of them respectively.

Find every remaining one:

```bash
grep -n "components\['\|workers\['" simple_camio.py
```

Expected after the rewrite: no matches. Note `workers['lock']` inside
`get_tracking_snapshot` and `get_pose_results` — both become `workers.lock`.

- [ ] **Step 8: Update the entry point**

At the bottom of `simple_camio.py`, `setup_camera(components['cam_port'])`
becomes `setup_camera(components.cam_port)`.

- [ ] **Step 9: Verify nothing else subscripts them**

```bash
grep -rn "components\[\|workers\[" simple_camio.py src/ tools/
```

Expected: no matches. Then confirm the module still imports:

```bash
.venv/bin/python -c "import simple_camio; print('ok')"
```

- [ ] **Step 10: Update the keyboard test to the new containers**

`tests/test_keyboard_input.py` from Task 2 builds dicts. Change its `rig`
fixture to build the real containers, so the test exercises the production
shape:

```python
@pytest.fixture
def rig():
    detector = FakeDetector()
    components = Components(
        model={}, cam_port=0, model_detector=detector, pose_detector=None,
        gesture_detector=None, motion_filter=None, interact=None,
        camio_player=None, crickets_player=None, heartbeat_player=None,
    )
    workers = Workers(
        audio_worker=FakeAudioWorker(), pose_worker=None,
        sift_worker=FakeSiftWorker(), pose_queue=queue.Queue(maxsize=1),
        sift_queue=queue.Queue(maxsize=1), lock=threading.Lock(),
    )
    return {
        'components': components,
        'workers': workers,
        'frame': np.zeros((16, 16, 3), dtype=np.uint8),
    }
```

Add `import threading` and
`from src.core.containers import Components, Workers` at the top of that file.

- [ ] **Step 11: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 108 tests (102 + 6).

- [ ] **Step 12: Device check**

This task has the widest diff of the five, and a missed conversion surfaces only
at runtime. Run the full device check from Task 2 Step 12 — windowed and
headless, including `h` and `q`.

- [ ] **Step 13: Commit**

```bash
git add src/core/containers.py simple_camio.py tests/test_containers.py tests/test_keyboard_input.py
git commit -m "refactor: typed containers instead of two untyped dicts

initialize_system and create_worker_threads return frozen dataclasses, and
the 59 string-key index sites become attribute access. A wrong field name
now fails where the container is built instead of raising a KeyError
somewhere downstream, and the twelve helper signatures say what they need."
```

---

### Task 7: Runtime configuration overrides

**Files:**
- Create: `src/core/config_overrides.py`
- Modify: `simple_camio.py:866-880` (argument parser and the headless special case)
- Test: `tests/test_config_overrides.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `src.core.config_overrides.apply_overrides(args, environ=None) -> List[str]` —
    mutates the config classes in place and returns one human-readable line per
    override applied, for logging. `args` is the `argparse.Namespace`; `environ`
    defaults to `os.environ`.
  - `src.core.config_overrides.add_arguments(parser) -> None` — registers the
    new flags on an existing parser.

- [ ] **Step 1: Write the failing test**

Create `tests/test_config_overrides.py`:

```python
"""CLI beats environment beats the class default."""

import argparse
import logging

import cv2 as cv
import pytest

from src.config import CameraConfig, TapDetectionConfig
from src.core.config_overrides import add_arguments, apply_overrides


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    add_arguments(p)
    return p


@pytest.fixture(autouse=True)
def restore_config():
    """
    Config classes are process-global, so a test that overrides one would leak
    into every test after it. Restore every attribute apply_overrides can write.
    """
    saved = {
        'width': CameraConfig.DEFAULT_WIDTH,
        'height': CameraConfig.DEFAULT_HEIGHT,
        'backend': CameraConfig.BACKEND,
        'headless': CameraConfig.HEADLESS,
        'collect': TapDetectionConfig.COLLECT_TAP_DATA,
        'log_level': logging.getLogger().level,
    }
    yield
    CameraConfig.DEFAULT_WIDTH = saved['width']
    CameraConfig.DEFAULT_HEIGHT = saved['height']
    CameraConfig.BACKEND = saved['backend']
    CameraConfig.HEADLESS = saved['headless']
    TapDetectionConfig.COLLECT_TAP_DATA = saved['collect']
    logging.getLogger().setLevel(saved['log_level'])


def test_no_flags_changes_nothing(parser):
    before = CameraConfig.DEFAULT_WIDTH

    applied = apply_overrides(parser.parse_args([]), environ={})

    assert applied == []
    assert CameraConfig.DEFAULT_WIDTH == before


def test_resolution_flag_sets_both_dimensions(parser):
    apply_overrides(parser.parse_args(['--resolution', '640x480']), environ={})

    assert CameraConfig.DEFAULT_WIDTH == 640
    assert CameraConfig.DEFAULT_HEIGHT == 480


def test_malformed_resolution_is_rejected(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(['--resolution', 'huge'])


def test_collect_tap_data_flag(parser):
    apply_overrides(parser.parse_args(['--collect-tap-data']), environ={})

    assert TapDetectionConfig.COLLECT_TAP_DATA is True


def test_headless_flag(parser):
    apply_overrides(parser.parse_args(['--headless']), environ={})

    assert CameraConfig.HEADLESS is True


def test_environment_is_read_when_the_flag_is_absent(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_RESOLUTION': '800x600'})

    assert CameraConfig.DEFAULT_WIDTH == 800
    assert CameraConfig.DEFAULT_HEIGHT == 600


def test_cli_wins_over_environment(parser):
    apply_overrides(
        parser.parse_args(['--resolution', '640x480']),
        environ={'CAMIO_RESOLUTION': '1920x1080'},
    )

    assert CameraConfig.DEFAULT_WIDTH == 640


def test_environment_headless_is_read(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_HEADLESS': '1'})

    assert CameraConfig.HEADLESS is True


def test_bad_environment_value_raises_with_the_variable_named(parser):
    with pytest.raises(ValueError, match='CAMIO_RESOLUTION'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_RESOLUTION': 'nope'})


def test_applied_overrides_are_reported_for_logging(parser):
    applied = apply_overrides(parser.parse_args(['--collect-tap-data']), environ={})

    assert len(applied) == 1
    assert 'COLLECT_TAP_DATA' in applied[0]


def test_camera_backend_flag_maps_to_an_opencv_constant(parser):
    apply_overrides(parser.parse_args(['--camera-backend', 'v4l2']), environ={})

    assert CameraConfig.BACKEND == cv.CAP_V4L2


def test_camera_backend_auto_means_no_backend(parser):
    apply_overrides(parser.parse_args(['--camera-backend', 'auto']), environ={})

    assert CameraConfig.BACKEND is None


def test_camera_backend_is_read_from_the_environment(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_CAMERA_BACKEND': 'v4l2'})

    assert CameraConfig.BACKEND == cv.CAP_V4L2


def test_bad_camera_backend_in_environment_is_rejected(parser):
    with pytest.raises(ValueError, match='CAMIO_CAMERA_BACKEND'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_CAMERA_BACKEND': 'nope'})


def test_log_level_flag_sets_the_root_logger(parser):
    apply_overrides(parser.parse_args(['--log-level', 'DEBUG']), environ={})

    assert logging.getLogger().level == logging.DEBUG


def test_bad_log_level_in_environment_is_rejected(parser):
    with pytest.raises(ValueError, match='CAMIO_LOG_LEVEL'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_LOG_LEVEL': 'CHATTY'})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_config_overrides.py -v`
Expected: FAIL at collection — `No module named 'src.core.config_overrides'`.

- [ ] **Step 3: Write the module**

Create `src/core/config_overrides.py`:

```python
"""
Runtime configuration overrides.

The config classes hold their values as class attributes, so changing one used
to mean editing source - README still tells the reader to set
TapDetectionConfig.COLLECT_TAP_DATA by hand. These helpers let a flag or an
environment variable do it instead, extending the pattern --headless already
established.

Precedence: command-line flag, then CAMIO_* environment variable, then the
value on the class.
"""

import logging
import os

import cv2 as cv

from src.config import CameraConfig, TapDetectionConfig

logger = logging.getLogger(__name__)

BACKENDS = {
    'auto': None,
    'v4l2': cv.CAP_V4L2,
    'dshow': cv.CAP_DSHOW,
    'msmf': cv.CAP_MSMF,
    'any': cv.CAP_ANY,
}

LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

TRUTHY = ('1', 'true', 'yes', 'on')


def _resolution(value, source):
    """Parse WxH into a pair of ints, naming its source if it is malformed."""
    try:
        width, height = value.lower().split('x')
        return int(width), int(height)
    except (ValueError, AttributeError):
        raise ValueError(
            f'{source}: expected a resolution like 1280x720, got {value!r}'
        )


def add_arguments(parser):
    """Register the override flags on an existing parser."""
    parser.add_argument(
        '--headless', action='store_true', default=False,
        help='Run without a display window - useful for Raspberry Pi daemon mode'
    )
    parser.add_argument(
        '--resolution', type=lambda v: _resolution(v, '--resolution'), default=None,
        metavar='WxH', help='Camera capture resolution, e.g. 640x480'
    )
    parser.add_argument(
        '--camera-backend', choices=sorted(BACKENDS), default=None,
        help='OpenCV capture backend. v4l2 on Linux, dshow or msmf on Windows'
    )
    parser.add_argument(
        '--collect-tap-data', action='store_true', default=False,
        help='Record tap detection samples to data/tap_dataset for training'
    )
    parser.add_argument(
        '--log-level', choices=LOG_LEVELS, default=None,
        help='Logging verbosity (default: INFO)'
    )


def apply_overrides(args, environ=None):
    """
    Apply command-line and environment overrides to the config classes.

    Args:
        args (argparse.Namespace): Parsed arguments from a parser that
            add_arguments() was called on.
        environ (dict, optional): Environment to read. Defaults to os.environ.

    Returns:
        list[str]: One line per override applied, for logging.

    Raises:
        ValueError: If a CAMIO_* variable holds a value that cannot be parsed.
    """
    env = os.environ if environ is None else environ
    applied = []

    resolution = getattr(args, 'resolution', None)
    if resolution is None and env.get('CAMIO_RESOLUTION'):
        resolution = _resolution(env['CAMIO_RESOLUTION'], 'CAMIO_RESOLUTION')
    if resolution is not None:
        CameraConfig.DEFAULT_WIDTH, CameraConfig.DEFAULT_HEIGHT = resolution
        applied.append(f'CameraConfig.DEFAULT_WIDTH/HEIGHT = {resolution[0]}x{resolution[1]}')

    backend = getattr(args, 'camera_backend', None)
    if backend is None and env.get('CAMIO_CAMERA_BACKEND'):
        backend = env['CAMIO_CAMERA_BACKEND'].lower()
        if backend not in BACKENDS:
            raise ValueError(
                f'CAMIO_CAMERA_BACKEND: expected one of {sorted(BACKENDS)}, '
                f'got {backend!r}'
            )
    if backend is not None:
        CameraConfig.BACKEND = BACKENDS[backend]
        applied.append(f'CameraConfig.BACKEND = {backend}')

    headless = getattr(args, 'headless', False)
    if not headless:
        headless = env.get('CAMIO_HEADLESS', '').lower() in TRUTHY
    if headless:
        CameraConfig.HEADLESS = True
        applied.append('CameraConfig.HEADLESS = True')

    collect = getattr(args, 'collect_tap_data', False)
    if not collect:
        collect = env.get('CAMIO_COLLECT_TAP_DATA', '').lower() in TRUTHY
    if collect:
        TapDetectionConfig.COLLECT_TAP_DATA = True
        applied.append('TapDetectionConfig.COLLECT_TAP_DATA = True')

    level = getattr(args, 'log_level', None)
    if level is None and env.get('CAMIO_LOG_LEVEL'):
        level = env['CAMIO_LOG_LEVEL'].upper()
        if level not in LOG_LEVELS:
            raise ValueError(
                f'CAMIO_LOG_LEVEL: expected one of {list(LOG_LEVELS)}, got {level!r}'
            )
    if level is not None:
        logging.getLogger().setLevel(getattr(logging, level))
        applied.append(f'log level = {level}')

    return applied
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_config_overrides.py -v`
Expected: PASS, 16 tests.

- [ ] **Step 5: Wire it into the entry point**

In `simple_camio.py`, add the import and replace the parser block. The old
`--headless` argument moves into `add_arguments`, so delete it here along with
the manual `if args.headless:` assignment:

```python
from src.core.config_overrides import add_arguments, apply_overrides
```

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CamIO - Interactive Map System')
    parser.add_argument('--input1', help='Path to map configuration JSON file',
                       default='models/UkraineMap/UkraineMap.json')
    parser.add_argument('--camera', type=int, default=None, metavar='PORT',
                       help='Camera port to use, skipping auto-detection. '
                            'Recommended for headless/daemon runs, where detecting '
                            'several cameras would otherwise need an interactive choice.')
    add_arguments(parser)
    args = parser.parse_args()

    for line in apply_overrides(args):
        logger.info(f"Config override: {line}")
```

Leave the rest of the block as it is; `CameraConfig.HEADLESS` is already what it
reads further down.

- [ ] **Step 6: Check the flags reach the config**

```bash
.venv/bin/python simple_camio.py --help
CAMIO_RESOLUTION=640x480 .venv/bin/python -c "
import argparse
from src.core.config_overrides import add_arguments, apply_overrides
from src.config import CameraConfig
p = argparse.ArgumentParser(); add_arguments(p)
print(apply_overrides(p.parse_args([])))
print(CameraConfig.DEFAULT_WIDTH, CameraConfig.DEFAULT_HEIGHT)
"
```

Expected: help lists `--resolution`, `--camera-backend`, `--collect-tap-data`,
`--log-level`, `--headless`; the second command prints the override line and
`640 480`.

- [ ] **Step 7: Document the flags**

In `README.md`, the Data Collection section says to enable collection by editing
`src/config.py`. Replace that instruction with the flag:

```markdown
1. Run with collection enabled: `python simple_camio.py --collect-tap-data`
```

In the Troubleshooting section, replace "Enable debug logging: change `level` in
the `logging.basicConfig(...)` call near the top of `simple_camio.py`" with:

```markdown
- Enable debug logging: `python simple_camio.py --log-level DEBUG`
```

Add a short subsection under Configuration listing the `CAMIO_*` variables and
noting the CLI-over-env-over-default precedence, and mention in
`RASPBERRY_PI_DAEMON.md` that the service can use them instead of editing its
`ExecStart` line.

- [ ] **Step 8: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 124 tests (108 + 16).

- [ ] **Step 9: Device check**

```bash
.venv/bin/python simple_camio.py --camera 0 --resolution 640x480 --log-level DEBUG
```

Confirm the resolution override takes effect (the capture is visibly smaller,
and the debug log is verbose), then repeat the headless check with
`CAMIO_HEADLESS=1` set instead of the flag, to exercise the env path.

- [ ] **Step 10: Commit**

```bash
git add src/core/config_overrides.py simple_camio.py tests/test_config_overrides.py README.md RASPBERRY_PI_DAEMON.md
git commit -m "feat: configure at runtime instead of editing source

Tap-data collection, resolution, camera backend and log level are flags now,
each falling back to a CAMIO_* variable so the systemd unit can set them
without an edited ExecStart. --headless moved in with them.

README no longer tells the reader to assign to TapDetectionConfig or to edit
the logging.basicConfig call."
```

---

### Task 8: Explicit detector seam

The last task, and the only one touching `src/detection/pose_detector.py`. It
changes how three values are passed and nothing about what they are.

**Files:**
- Modify: `src/detection/pose_detector.py` — `PoseDetectorMPEnhanced.__init__:1234`, `.detect:1455`, `._get_base_outputs:1494`, `._get_mediapipe_results:1507`, `CombinedPoseDetector.__init__:2316`, `CombinedPoseDetector.detect:2352`
- Test: `tests/test_combined_detector.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `PoseDetectorMPEnhanced.__init__(model, reuse_base_outputs=False)`
  - `PoseDetectorMPEnhanced.detect(image, H, _, processing_scale=0.5, draw=False, base_outputs=None, mp_results=None)`

- [ ] **Step 1: Read the three seams before touching them**

```bash
sed -n '1227,1260p' src/detection/pose_detector.py
sed -n '1455,1530p' src/detection/pose_detector.py
sed -n '2316,2388p' src/detection/pose_detector.py
```

Note every read of `self._skip_super`, `self._base_cache` and
`self._provided_results`. The rewrite has to preserve each one's meaning
exactly: `_skip_super` selects whether `detect` recomputes or reuses,
`_base_cache` is the base detector's `(index_pos, status, img)` triple, and
`_provided_results` is the `(mp_results, orig_w, orig_h)` triple.

- [ ] **Step 2: Write the failing test**

Create `tests/test_combined_detector.py`:

```python
"""
CombinedPoseDetector's wiring, without MediaPipe.

The two detectors' tap logic is out of scope for this work. These tests pin
only the seam between them: MediaPipe runs once, and the base pass's outputs
reach the enhanced pass.
"""

import numpy as np
import pytest

from src.detection.pose_detector import PoseDetectorMPEnhanced


def test_enhanced_accepts_base_outputs_as_a_parameter():
    """The wiring is a parameter now, not an attribute poked from outside."""
    import inspect

    sig = inspect.signature(PoseDetectorMPEnhanced.detect)

    assert 'base_outputs' in sig.parameters
    assert 'mp_results' in sig.parameters
    assert sig.parameters['base_outputs'].default is None
    assert sig.parameters['mp_results'].default is None


def test_reuse_flag_is_a_constructor_parameter():
    import inspect

    sig = inspect.signature(PoseDetectorMPEnhanced.__init__)

    assert 'reuse_base_outputs' in sig.parameters
    assert sig.parameters['reuse_base_outputs'].default is False


def test_combined_runs_mediapipe_once(monkeypatch):
    """
    The whole reason the seam exists: MediaPipe is the expensive call and it
    must not run twice per frame.
    """
    from src.detection import pose_detector as pd

    calls = []

    class FakeBase:
        image_map_color = None
        data_collector = None

        def detect(self, image, H, _, processing_scale=0.5, draw=False):
            calls.append('base-mediapipe')
            return None, None, None

        def get_cached_mp_results(self):
            return (None, 640, 480)

    class FakeEnhanced:
        data_collector = object()

        def __init__(self):
            self.received = None

        def detect(self, image, H, _, processing_scale=0.5, draw=False,
                   base_outputs=None, mp_results=None):
            self.received = (base_outputs, mp_results)
            return 'pos', 'status', 'img'

    fake_base, fake_enh = FakeBase(), FakeEnhanced()
    monkeypatch.setattr(pd, 'PoseDetectorMP', lambda model: fake_base)
    monkeypatch.setattr(pd, 'PoseDetectorMPEnhanced',
                        lambda model, reuse_base_outputs=False: fake_enh)

    combined = pd.CombinedPoseDetector({'filename': 'unused'})
    result = combined.detect(np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3), None)

    assert calls == ['base-mediapipe'], 'MediaPipe ran more than once per frame'
    assert result == ('pos', 'status', 'img')
    assert fake_enh.received[0] == (None, None, None), 'base outputs did not arrive'
    assert fake_enh.received[1] == (None, 640, 480), 'mp results did not arrive'
```

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_combined_detector.py -v`
Expected: the two signature tests FAIL (`base_outputs` and
`reuse_base_outputs` do not exist yet); `test_combined_runs_mediapipe_once`
fails because `CombinedPoseDetector` still assigns `_base_cache` and
`_provided_results` on the fake, which then never reach `detect`.

- [ ] **Step 4: Make the reuse flag a constructor parameter**

In `PoseDetectorMPEnhanced.__init__`, take the flag and store it under a
non-underscore name, then delete whatever sets `self._skip_super` there:

```python
    def __init__(self, model, reuse_base_outputs=False):
        """
        Args:
            model (dict): Map model configuration
            reuse_base_outputs (bool): When True, detect() expects its caller to
                supply the base pass's outputs and MediaPipe results rather than
                computing them. CombinedPoseDetector sets this; a standalone
                enhanced detector leaves it False.
        """
        ...
        self.reuse_base_outputs = reuse_base_outputs
```

- [ ] **Step 5: Make the two caches parameters**

Change `PoseDetectorMPEnhanced.detect` to accept them, and pass them into the
two helpers instead of reading attributes:

```python
    def detect(self, image, H, _, processing_scale=0.5, draw=False,
               base_outputs=None, mp_results=None):
```

Replace each read of `self._base_cache` with the `base_outputs` parameter, each
read of `self._provided_results` with `mp_results`, and each read of
`self._skip_super` with `self.reuse_base_outputs`. Where `_get_base_outputs`
and `_get_mediapipe_results` read those attributes, give them parameters:

```python
    def _get_base_outputs(self, base_outputs):
    def _get_mediapipe_results(self, image, processing_scale, mp_results):
```

and update their call sites inside `detect` to forward what it received. Do not
change any computation in either helper.

- [ ] **Step 6: Have the wrapper pass rather than poke**

In `CombinedPoseDetector.__init__`, construct the enhanced detector with the
flag and delete the `self.enh._skip_super = True` line:

```python
        self.base = PoseDetectorMP(model)
        self.enh = PoseDetectorMPEnhanced(model, reuse_base_outputs=True)
```

In `CombinedPoseDetector.detect`, delete the two assignments to
`self.enh._base_cache` and `self.enh._provided_results`, and pass the values
instead:

```python
        # Draw only once (in enhanced), keep base compute-only
        base_outputs = self.base.detect(image, H, _, processing_scale, draw=False)

        # Share MediaPipe results so the expensive call happens once per frame
        mp_results = self.base.get_cached_mp_results()

        return self.enh.detect(image, H, _, processing_scale, draw,
                               base_outputs=base_outputs, mp_results=mp_results)
```

- [ ] **Step 7: Confirm the private attributes are gone**

```bash
grep -n "_skip_super\|_base_cache\|_provided_results" src/detection/pose_detector.py
```

Expected: no matches.

- [ ] **Step 8: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_combined_detector.py -v`
Expected: PASS, 3 tests.

- [ ] **Step 9: Run the full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS, 127 tests (124 + 3).

- [ ] **Step 10: Device check**

The behaviour of tap detection must be indistinguishable from before this task.
Run the windowed check and pay attention to tap feel specifically: single taps
register at the same pressure and speed as they did, double taps register, and
no zone became harder or easier to trigger. If anything feels different, the
change was not transport-only — revert and re-read Step 1.

- [ ] **Step 11: Commit**

```bash
git add src/detection/pose_detector.py tests/test_combined_detector.py
git commit -m "refactor: pass the enhanced detector its inputs explicitly

CombinedPoseDetector drove its child through _skip_super, _base_cache and
_provided_results, so the enhanced detector's behaviour depended on flags
set from outside it. Those are a constructor parameter and two detect()
arguments now.

Transport only: no computation in either detector changed. A mock test pins
the invariant the seam exists for, that MediaPipe runs once per frame."
```

---

## After the last task

- [ ] Full suite green: `.venv/bin/python -m pytest -q` — expected 127 tests.
- [ ] `graphify update .` to refresh the knowledge graph, as `CLAUDE.md` requires.
- [ ] Confirm the eight originally-flagged consumer lines are all gone:
      `grep -n "model_detector\." simple_camio.py src/ui/display.py` returns nothing.
- [ ] Merge `auto-narration` into `master` — one merge, as agreed. The TTS work
      on this branch merges with it.
- [ ] The out-of-scope items stay open: the duplicated tap state machine, the
      237/4 dataset imbalance, and the enhanced-collector question in
      `docs/tap-data-findings.md`.
