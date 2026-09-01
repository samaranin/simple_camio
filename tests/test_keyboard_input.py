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
