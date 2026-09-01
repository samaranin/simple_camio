"""The 'h' key requests re-detection through the worker, not by poking state."""

import queue
import threading

import numpy as np
import pytest

import simple_camio
from src.core.containers import Workers


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
    workers = Workers(
        audio_worker=FakeAudioWorker(), pose_worker=None,
        sift_worker=FakeSiftWorker(), pose_queue=queue.Queue(maxsize=1),
        sift_queue=queue.Queue(maxsize=1), lock=threading.Lock(),
    )
    return {
        'workers': workers,
        'frame': np.zeros((16, 16, 3), dtype=np.uint8),
    }


def test_h_asks_the_worker_to_redetect(rig):
    simple_camio.handle_keyboard_input(
        ord('h'), _StopEvent(), rig['frame'], rig['workers']
    )

    assert rig['workers'].sift_worker.redetect_calls == 1


def test_q_signals_shutdown(rig):
    stop = _StopEvent()

    keep_going = simple_camio.handle_keyboard_input(
        ord('q'), stop, rig['frame'], rig['workers']
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
