"""The drawing functions read a snapshot, never the live detector."""

import time

import numpy as np
import pytest

from src.config import UIConfig
from src.core.tracking import TrackingSnapshot
from src.ui import display
from src.ui.display import draw_map_tracking, draw_ui_overlay

RECT_PTS = np.array(
    [[[10, 10]], [[90, 10]], [[90, 90]], [[10, 90]]], dtype=np.float32
)


class FakeInteract:
    """Carries only the map shape draw_map_tracking needs."""

    class _Img:
        shape = (100, 100, 3)

    image_map_color = _Img()


class FakeCap:
    """A capture object with neither get_fps() nor get(), so no FPS is drawn."""


@pytest.fixture
def blank():
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def spy(monkeypatch):
    """Records which draw helper was called, with which arguments."""
    calls = []

    def from_points(img, pts, color=None, thickness=None):
        calls.append(('points', pts, color, thickness))
        return img

    def on_image(img, shape, homography):
        calls.append(('homography', shape, homography))
        return img

    monkeypatch.setattr(display, 'draw_rectangle_from_points', from_points)
    monkeypatch.setattr(display, 'draw_rectangle_on_image', on_image)
    return calls


def test_draws_from_rect_points_when_present(blank, spy):
    """The snapshot's own corners are drawn, not a homography projection."""
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=RECT_PTS, tracking=True)

    _, flash = draw_map_tracking(blank, snap, FakeInteract(), 0)

    assert [call[0] for call in spy] == ['points']
    assert spy[0][1] is RECT_PTS
    assert spy[0][2] == UIConfig.COLOR_GREEN
    assert spy[0][3] == 3
    assert flash == 0


def test_flash_counter_decrements_while_flashing(blank, spy):
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=RECT_PTS, tracking=True)

    _, flash = draw_map_tracking(blank, snap, FakeInteract(), 5)

    assert [call[0] for call in spy] == ['points']
    assert spy[0][2] == UIConfig.COLOR_YELLOW
    assert spy[0][3] == 5
    assert flash == 4


def test_homography_fallback_used_only_when_a_matrix_is_present(blank, spy):
    """
    With corners missing the projection is the fallback - but only while there
    is a matrix to project. The old `else` had no such condition and handed
    None to the helper.
    """
    matrix = np.eye(3)
    interact = FakeInteract()

    draw_map_tracking(blank, TrackingSnapshot(H=matrix, tracking=True), interact, 0)

    assert [call[0] for call in spy] == ['homography']
    assert spy[0][1] == interact.image_map_color.shape
    assert spy[0][2] is matrix

    del spy[:]
    draw_map_tracking(blank, TrackingSnapshot(H=None, rect_pts=None), interact, 0)

    assert spy == []


def test_empty_snapshot_does_not_raise(blank, spy):
    """The old code passed None into the draw helpers and took down the loop."""
    img, flash = draw_map_tracking(blank, TrackingSnapshot(), FakeInteract(), 0)

    assert spy == []
    assert img is blank
    assert flash == 0


def test_real_helpers_draw_the_flash_in_yellow(blank):
    """
    Unspied, so a signature drift in the real helpers cannot hide. The
    homography fallback only ever draws green, so a yellow pixel proves the
    snapshot's own corners were used.
    """
    snap = TrackingSnapshot(H=np.eye(3), rect_pts=RECT_PTS, tracking=True)

    img, _ = draw_map_tracking(blank, snap, FakeInteract(), 5)

    assert img.any(), "nothing was drawn"
    yellow = np.all(img == np.array(UIConfig.COLOR_YELLOW, dtype=np.uint8), axis=-1)
    assert yellow.any(), "the flash was not drawn from the snapshot's corners"


def test_ui_overlay_renders_the_snapshot_status_text(blank, monkeypatch):
    """The status string comes from the snapshot, not from the detector."""
    drawn = []
    monkeypatch.setattr(
        display.cv, 'putText',
        lambda img, text, *args, **kwargs: drawn.append(text)
    )
    snap = TrackingSnapshot(status_text="TRACKING (Q:42 Age:3)", tracking=True)
    fps_state = {
        'display_count': 0,
        'start_time': time.time(),
        'display_fps': 0.0,
    }

    draw_ui_overlay(blank, snap, None, 0.0, fps_state, FakeCap())

    assert drawn == ["TRACKING (Q:42 Age:3)"]
