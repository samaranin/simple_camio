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
