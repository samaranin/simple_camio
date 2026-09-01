# Thread safety and testable seams

**Date:** 2026-09-01
**Status:** approved, ready for implementation planning
**Branch:** `auto-narration`

## Problem

A read of the codebase surfaced six issues. One is a live defect that can restart the
daemon; the rest are debt that makes the defect class hard to prevent.

The live defect: the SIFT tracking state is a cluster of six related attributes —
`H`, `requires_homography`, `last_rect_pts`, `frames_since_last_detection`,
`tracking_quality_history`, `last_validation_time` — written from roughly 25 sites in
`src/detection/sift_detector.py` on the `SIFTWorker` thread, and read (and partly
written) from the main thread. No thread owns them. `SIFTWorker` is handed a lock at
`src/core/workers.py:273` and never acquires it; `PoseWorker` does, at
`src/core/workers.py:241`.

Two consequences are reachable today:

1. **Daemon restart.** `draw_map_tracking` (`src/ui/display.py:32-47`) tests
   `last_rect_pts is not None`, then re-reads the attribute at lines 35 and 41. The
   worker may null it in between, passing `None` into `draw_rectangle_from_points()`.
   The `else` branch re-reads `model_detector.H` at line 46 with the same exposure.
   The `while` body of `run_main_loop` (`simple_camio.py:675-747`) has no
   `try`/`except`, so the exception leaves the loop, does not match
   `except KeyboardInterrupt` in `__main__`, and the process exits through
   `finally: cleanup(...)`. Under systemd on the Pi that is a restart.
2. **Dropped tracking frame.** `simple_camio.py:200` reads `model_detector.H` twice in
   one expression, so `None` can reach `pose_queue` despite the guard. `PoseWorker`'s
   broad `except` (`src/core/workers.py:233`) absorbs it: one frame of hand tracking is
   lost and an ERROR line is logged.

`frames_since_last_detection` is additionally incremented from both threads
(`simple_camio.py:506-508` and `src/detection/sift_detector.py:147`), so increments are
lost.

The supporting debt:

| # | Issue | Evidence |
| --- | --- | --- |
| 2 | `CombinedPoseDetector` drives its children through their private attributes: `_skip_super`, `_base_cache`, `_provided_results` | `src/detection/pose_detector.py:2334-2385` |
| 3 | Test coverage is confined to the TTS and audio layers; the detection, worker, and interaction modules have none | 9 files in `tests/`, all TTS/audio; 75 tests pass |
| 5 | Configuration is class attributes, so changing behaviour means editing source — README instructs the reader to set `TapDetectionConfig.COLLECT_TAP_DATA = True` and to edit the `logging.basicConfig` call | `src/config.py`, `README.md` |
| 6 | `simple_camio.py` threads two untyped dicts through twelve helper functions; a mistyped key is a runtime `KeyError` | 33 `components['...']` and 26 `workers['...']` index sites |

## Goal

Remove the tracking-state race, and leave behind enough structure that the same class of
defect is catchable by tests rather than by a user in the field.

## Non-goal

**The duplicated tap state machine stays as it is.** `PoseDetectorMPEnhanced`
(`src/detection/pose_detector.py:1227`) re-implements the whole press/release/validate
pipeline in parallel with its own base class — roughly 40 methods against 40 —
and merging them is the single largest simplification available in this codebase.

It is out of scope because there is no way to prove the merge preserved behaviour. The
collected dataset does not serve: `TapDataCollector.collect_positive()`
(`src/tap_classifier/tap_data_collector.py:123`) records only the finished 18-value
feature vector at the moment the state machine has already decided a press ended. The
machine consumes a per-frame landmark stream and internal history deques, so its
recorded output cannot be replayed as its input. Merging it would need landmark-level
fixtures that do not exist, and a silent sensitivity regression on a live map is the
expected failure mode. Revisit only if such fixtures get recorded.

## Decisions

| Question | Decision | Why |
| --- | --- | --- |
| How is the tracking-state race fixed? | The worker publishes an immutable snapshot under the lock; the detector's internal state becomes private to the worker thread | Locking each of ~25 write sites does not give a consumer a consistent *set* of the six attributes. Publishing one object does, and it matches the pattern `PoseWorker` already uses. |
| Who owns `frames_since_last_detection`? | `SIFTWorker` | One writer removes the lost update. The main-thread increment is deleted. |
| How does the `h` key request re-detection? | Through `SIFTWorker.trigger_redetect()` | The method already exists (`src/core/workers.py:381`) for exactly this and is currently unused, while the key handler pokes the detector directly. |
| Order of work | Defect, then tests, then the mechanical refactors, then the coupling fix | The race is the only actual bug and its diff is smallest against an unchanged tree. Tests for `workers.py` are written *with* step 1, not before it, so they encode the fixed behaviour rather than the broken one. |
| Configuration override mechanism | CLI flag, falling back to `CAMIO_*` env var, falling back to the class default | Extends the precedent already set by `--headless` at `simple_camio.py:880`. The env layer lets `simple_camio.service` change behaviour without editing its arguments. |
| Scope of the step-5 coupling fix | Transport only — the three private attributes become explicit parameters. No computation is touched. | The file is the untested one. A behaviour-preserving signature change is verifiable by inspection; a logic change in there is not. |
| Branch strategy | All five steps on `auto-narration`, one commit each, a single merge at the end | As requested. A tag on the pre-step-1 commit gives a comparison point if a device run goes bad. |

## Architecture

### Step 1 — Tracking-state ownership

A frozen snapshot carries everything a consumer needs:

```
TrackingSnapshot(
    H,            # homography, or None when the map is not currently tracked
    rect_pts,     # projected template corners in camera coords, or None
    tracking,     # bool: the inverse of the detector's requires_homography
    age,          # frames since the last successful detection
    status_text,  # the display string, rendered by the worker that owns the state
)
```

`status_text` is built inside the worker rather than by the consumer calling
`get_tracking_status()`, because that method reads three of the six guarded attributes
(`src/detection/sift_detector.py:459-465`) and calling it from the main thread would
reintroduce exactly the exposure this step removes.

```
SIFTWorker thread                        main thread
-----------------                        -----------
detect / validate
  |
  v
compute into locals
  |
  v
with lock: self.snapshot = Snapshot(...)  ──►  with lock: snap = worker.snapshot
                                                 |
   (detector attributes never read                v
    outside this thread)                   feed_worker_queues(snap)
                                           sleep-mode decision(snap)
                                           draw_map_tracking(snap)
```

One acquisition per main-loop iteration yields one consistent set, and every repeated
read disappears with it. `draw_map_tracking` takes the snapshot instead of the detector,
which removes both exposed re-reads in `src/ui/display.py`.

**Accepted behaviour change:** age counts SIFT-processed frames rather than main-loop
iterations. The two already diverged, because the queue drops frames under load. The
status line will show a smaller number.

**Tests** (`tests/test_workers.py`, against a fake detector — no MediaPipe, no camera):
a consumer never observes a half-updated snapshot; `h` routes through the worker; a full
queue drops the oldest frame.

### Step 2 — Tests for the deterministic units

New files under `tests/` only. Production code is untouched, so this step cannot regress
the device.

- `test_interaction_policy.py` — colour to zone index, the zone filter buffer, reset when
  a hand reappears.
- `test_tap_classifier.py` — the 241 recorded vectors as a regression set: `predict()`
  must reproduce a committed baseline. `src/tap_classifier/test_tap_classifier.py` moves
  out of `src/` in the same step; it is a script with a `main()` that pytest collects
  today.
- `test_utils.py` — `load_map_parameters`, `is_gesture_valid`,
  `normalize_gesture_location`, `color_to_index`.

Two findings get recorded rather than fixed, both belonging to the out-of-scope file:

- All 241 collected samples report `detector: "base"`. Confirm whether
  `_collect_enhanced_tap_data_positive/negative` can fire at all. If the path is dead,
  training data for the enhanced detector cannot be collected.
- The dataset is 237 positive against 4 negative, so `models/tap_model.json` is trained
  almost without counter-examples and is expected to over-predict taps.

### Step 3 — Typed component containers

Two frozen dataclasses replace the dicts: `Components` (10 fields — `model`, `cam_port`,
`model_detector`, `pose_detector`, `gesture_detector`, `motion_filter`, `interact`,
`camio_player`, `crickets_player`, `heartbeat_player`) and `Workers` (6 fields —
`audio_worker`, `pose_worker`, `sift_worker`, `pose_queue`, `sift_queue`, `lock`).

The 59 index sites become attribute access. A mistyped attribute is still a runtime
error, so the honest gains are narrower than "type safety": a wrong field name fails at
construction instead of at the first use far away, an editor and `ruff` can see the
misspelling, the twelve helper signatures state what they actually require, and step 2's
tests can assemble a `Components` from fakes.

### Step 4 — Runtime configuration

One `apply_overrides(args, environ)` resolving CLI over env over class default, tested
directly. New flags:

- `--collect-tap-data` — what README currently asks the reader to edit source for.
- `--resolution WxH` and `--camera-backend` — the latter is edited by hand per OS today.
- `--log-level` — README currently says to edit the `logging.basicConfig` call.

`CAMIO_*` env equivalents for the daemon. `os` is already imported in `src/config.py` and
currently unused.

### Step 5 — Explicit detector seam

- `PoseDetectorMPEnhanced.detect(..., base_outputs=None, mp_results=None)` — the two
  caches become parameters.
- `_skip_super` becomes a constructor flag, `reuse_base_outputs`, set once by
  `CombinedPoseDetector` rather than mutated from outside.
- The diff is confined to the three assignments at
  `src/detection/pose_detector.py:2334-2385` and their read sites in `_get_base_outputs`
  and `_get_mediapipe_results`.
- Mock test: MediaPipe is invoked exactly once per `CombinedPoseDetector.detect()`, and
  the base outputs reach the fusion step.

## Error handling

Step 1 removes the two reachable `None` dereferences by construction — a snapshot field
read under the lock cannot change under its own guard. It deliberately does **not** add a
blanket `try`/`except` around the main-loop body: that would convert this defect from a
restart into a silent per-frame failure, and the loop has no meaningful recovery for an
arbitrary exception. If a later step wants the daemon to survive unexpected draw errors,
that is a separate decision with its own logging requirements.

## Verification

Steps 1, 3, 4 and 5 each end on a working device: the map tracks, several zones speak, a
double tap registers, `h` re-detects, `q` exits cleanly, and the headless daemon starts
and stops. Step 2 is verified by pytest alone.

The suite must stay green at every step; it is 75 tests today.

## Out of scope

- Merging the duplicated tap state machine (see Non-goal).
- Retraining `models/tap_model.json` on a balanced dataset.
- Recording landmark-level fixtures.
- Any change to the TTS layer already on this branch.
