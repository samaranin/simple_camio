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
