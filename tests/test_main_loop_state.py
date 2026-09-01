"""The main loop's two snapshot readers: the flash trigger and the sleep gate."""

import time

import numpy as np
import pytest

import simple_camio
from src.config import CameraConfig, UIConfig
from src.core.tracking import TrackingSnapshot


class FakeCap:
    """Records the target FPS the loop asks for, the way ThreadedCamera would."""

    def __init__(self):
        self.targets = []

    def set_target_fps(self, fps):
        self.targets.append(fps)


def _sleep_state(idle_seconds):
    """State whose last activity is idle_seconds in the past."""
    return {
        'active': False,
        'target_fps': None,
        'last_activity_ts': time.time() - idle_seconds,
    }


# --- the consumer half of the rectangle flash -----------------------------


def test_a_new_generation_starts_the_flash():
    snapshot = TrackingSnapshot(detect_generation=1)

    remaining, seen = simple_camio.apply_detect_flash(snapshot, 0, 0)

    assert remaining == UIConfig.RECT_FLASH_FRAMES
    assert seen == 1


def test_the_same_generation_leaves_the_flash_alone():
    """Re-triggering every frame would hold the rectangle bright forever."""
    snapshot = TrackingSnapshot(detect_generation=4)

    remaining, seen = simple_camio.apply_detect_flash(snapshot, 4, 3)

    assert remaining == 3
    assert seen == 4


def test_each_new_generation_retriggers_the_flash():
    """Two detections in a row must both flash, even mid-flash."""
    remaining, seen = simple_camio.apply_detect_flash(
        TrackingSnapshot(detect_generation=7), 6, 2
    )

    assert remaining == UIConfig.RECT_FLASH_FRAMES
    assert seen == 7


def test_a_generation_that_never_advances_never_flashes():
    """The worker publishes generation 0 until the first successful detection."""
    remaining, seen = simple_camio.apply_detect_flash(TrackingSnapshot(), 0, 0)

    assert remaining == 0
    assert seen == 0


# --- update_sleep_mode's tracking gate ------------------------------------


@pytest.mark.parametrize('snapshot', [
    TrackingSnapshot(tracking=False, H=np.eye(3)),
    TrackingSnapshot(tracking=True, H=None),
    TrackingSnapshot(),
])
def test_an_untracked_map_never_sleeps(snapshot):
    """
    Sleep throttles the camera to SLEEP_FPS. While the map is not tracked the
    app must stay at full rate so SIFT keeps getting frames to detect on, so
    both halves of the gate have to hold: tracking, and a real homography.
    """
    cap = FakeCap()
    state = _sleep_state(CameraConfig.SLEEP_AFTER_SECONDS + 5)

    result = simple_camio.update_sleep_mode(cap, None, state, snapshot)

    assert result['active'] is False
    assert result['target_fps'] == CameraConfig.TARGET_FPS
    assert cap.targets == [CameraConfig.TARGET_FPS]


def test_an_untracked_map_keeps_resetting_the_idle_timer():
    """Otherwise the app would drop straight to sleep the moment it locks on."""
    before = time.time() - (CameraConfig.SLEEP_AFTER_SECONDS + 5)
    state = _sleep_state(CameraConfig.SLEEP_AFTER_SECONDS + 5)

    result = simple_camio.update_sleep_mode(
        FakeCap(), None, state, TrackingSnapshot(tracking=True, H=None)
    )

    assert result['last_activity_ts'] > before


def test_a_tracked_idle_map_does_sleep():
    """The other side of the gate, so the tests above cannot pass vacuously."""
    cap = FakeCap()
    state = _sleep_state(CameraConfig.SLEEP_AFTER_SECONDS + 5)

    result = simple_camio.update_sleep_mode(
        cap, None, state, TrackingSnapshot(tracking=True, H=np.eye(3))
    )

    assert result['active'] is True
    assert result['target_fps'] == CameraConfig.SLEEP_FPS
    assert cap.targets == [CameraConfig.SLEEP_FPS]
