"""
CombinedPoseDetector's wiring, without MediaPipe.

The two detectors' tap logic is out of scope for this work. These tests pin
only the seam between them: MediaPipe runs once, and the base pass's outputs
reach the enhanced pass.
"""

import numpy as np

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


def _bare_enhanced(reuse_base_outputs):
    """
    An enhanced detector with no __init__ run: no model file, no MediaPipe.

    The two helpers under test read exactly one attribute between them, so a
    bare instance is enough to exercise their branches and nothing else.
    """
    detector = PoseDetectorMPEnhanced.__new__(PoseDetectorMPEnhanced)
    detector.reuse_base_outputs = reuse_base_outputs
    return detector


# ---------- _get_base_outputs: the conjunction ----------

def test_base_outputs_reused_when_flag_set_and_value_supplied():
    """The one cell of the conjunction that takes the reuse path."""
    detector = _bare_enhanced(reuse_base_outputs=True)

    assert detector._get_base_outputs((1, 2, 3)) == (1, 2, 3)


def test_base_outputs_absent_falls_through_even_with_flag_set():
    """
    The load-bearing cell. Dropping `and base_outputs is not None` would return
    a bare None here, which detect() unpacks three ways.
    """
    detector = _bare_enhanced(reuse_base_outputs=True)

    assert detector._get_base_outputs(None) == (None, None, None)


def test_base_outputs_ignored_when_flag_clear():
    """A standalone enhanced detector computes its own, whatever it was handed."""
    detector = _bare_enhanced(reuse_base_outputs=False)

    assert detector._get_base_outputs((1, 2, 3)) == (None, None, None)


# ---------- _get_mediapipe_results: the fallback ----------

def _stub_mediapipe(detector):
    """Record calls to _process_with_mediapipe instead of running MediaPipe."""
    calls = []

    def fake_process(image, processing_scale):
        calls.append((image, processing_scale))
        return 'fallback-results', 111, 222

    detector._process_with_mediapipe = fake_process
    return calls


def test_mediapipe_runs_when_nothing_supplied():
    detector = _bare_enhanced(reuse_base_outputs=False)
    calls = _stub_mediapipe(detector)

    assert detector._get_mediapipe_results('frame', 0.5, None) == (
        'fallback-results', 111, 222
    )
    assert calls == [('frame', 0.5)]


def test_mediapipe_runs_when_supplied_value_is_malformed():
    """What the try/except is for: a value that will not unpack into three."""
    detector = _bare_enhanced(reuse_base_outputs=True)
    calls = _stub_mediapipe(detector)

    assert detector._get_mediapipe_results('frame', 0.25, (1, 2)) == (
        'fallback-results', 111, 222
    )
    assert calls == [('frame', 0.25)]


def test_mediapipe_runs_when_supplied_results_element_is_none():
    """
    The live case. get_cached_mp_results() returns (None, w, h) on a frame where
    the base pass found no hand, and the enhanced pass must still run MediaPipe.
    """
    detector = _bare_enhanced(reuse_base_outputs=True)
    calls = _stub_mediapipe(detector)

    assert detector._get_mediapipe_results('frame', 0.5, (None, 640, 480)) == (
        'fallback-results', 111, 222
    )
    assert calls == [('frame', 0.5)]


def test_supplied_mediapipe_results_pass_through_untouched():
    """The whole point of the seam: a usable triple short-circuits MediaPipe."""
    detector = _bare_enhanced(reuse_base_outputs=True)
    calls = _stub_mediapipe(detector)

    sentinel = object()

    assert detector._get_mediapipe_results('frame', 0.5, (sentinel, 640, 480)) == (
        sentinel, 640, 480
    )
    assert calls == [], 'MediaPipe ran despite usable results being supplied'
