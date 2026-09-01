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
