"""feed_worker_queues reads the homography once, so None cannot reach the queue."""

import queue

import numpy as np
import pytest

import simple_camio
from src.core.containers import Workers
from src.core.tracking import TrackingSnapshot


@pytest.fixture
def rig():
    return {
        'workers': Workers(
            audio_worker=None, pose_worker=None, sift_worker=None,
            pose_queue=queue.Queue(maxsize=1), sift_queue=queue.Queue(maxsize=1),
            lock=None,
        ),
        'frame': np.zeros((8, 8, 3), dtype=np.uint8),
        'gray': np.zeros((8, 8), dtype=np.uint8),
    }


def _queued_homography(rig):
    """The homography from the single item on the pose queue."""
    _, H = rig['workers'].pose_queue.get_nowait()
    return H


def test_missing_homography_feeds_identity(rig):
    """A snapshot with no homography must still hand the pose worker a matrix."""
    simple_camio.feed_worker_queues(
        rig['frame'], rig['gray'], rig['workers'], TrackingSnapshot()
    )

    H = _queued_homography(rig)
    assert H is not None
    assert np.array_equal(H, simple_camio.IDENTITY_3)
    assert rig['workers'].sift_queue.get_nowait() is rig['gray']


def test_real_homography_reaches_the_pose_queue(rig):
    matrix = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 7.0], [0.0, 0.0, 1.0]])

    simple_camio.feed_worker_queues(
        rig['frame'], rig['gray'], rig['workers'],
        TrackingSnapshot(H=matrix, tracking=True)
    )

    assert _queued_homography(rig) is matrix


def test_homography_is_read_exactly_once(rig):
    """
    The old code tested H and then re-read it, so a null landing between the
    two reads reached the pose queue despite the guard. This stand-in returns
    a matrix on its first read and None afterwards: the single-read version
    queues the matrix, the double-read version queues None.
    """
    matrix = np.eye(3)

    class VanishingHomography:
        def __init__(self):
            self.reads = 0

        @property
        def H(self):
            self.reads += 1
            return matrix if self.reads == 1 else None

    snapshot = VanishingHomography()

    simple_camio.feed_worker_queues(
        rig['frame'], rig['gray'], rig['workers'], snapshot
    )

    assert snapshot.reads == 1
    assert _queued_homography(rig) is matrix


def test_a_full_sift_queue_drops_the_oldest_frame(rig):
    """
    The queues hold one frame so the worker always sees the newest. When the
    worker is still busy the queue is full, and this branch has to evict the
    stale frame - otherwise put_nowait fails and the new frame is lost, leaving
    the worker to detect on an image that no longer matches what is on screen.
    """
    stale = np.full((8, 8), 7, dtype=np.uint8)
    rig['workers'].sift_queue.put_nowait(stale)

    simple_camio.feed_worker_queues(
        rig['frame'], rig['gray'], rig['workers'], TrackingSnapshot()
    )

    assert rig['workers'].sift_queue.get_nowait() is rig['gray']
    assert rig['workers'].sift_queue.empty()


def test_a_full_pose_queue_drops_the_oldest_frame(rig):
    """Same eviction on the pose side: the newest frame and homography win."""
    matrix = np.array([[3.0, 0.0, 1.0], [0.0, 3.0, 2.0], [0.0, 0.0, 1.0]])
    rig['workers'].pose_queue.put_nowait((np.zeros((8, 8, 3), dtype=np.uint8), np.eye(3)))

    simple_camio.feed_worker_queues(
        rig['frame'], rig['gray'], rig['workers'],
        TrackingSnapshot(H=matrix, tracking=True)
    )

    frame, H = rig['workers'].pose_queue.get_nowait()
    assert frame is rig['frame']
    assert H is matrix
    assert rig['workers'].pose_queue.empty()
