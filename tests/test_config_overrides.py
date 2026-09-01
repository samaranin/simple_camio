"""CLI beats environment beats the class default."""

import argparse
import logging

import cv2 as cv
import pytest

from src.config import CameraConfig, TapDetectionConfig
from src.core.config_overrides import BACKENDS, add_arguments, apply_overrides
from src.ui.display import select_camera_backend


@pytest.fixture
def parser():
    p = argparse.ArgumentParser()
    add_arguments(p)
    return p


@pytest.fixture(autouse=True)
def restore_config():
    """
    Config classes are process-global, so a test that overrides one would leak
    into every test after it. Restore every attribute apply_overrides can write.
    """
    saved = {
        'width': CameraConfig.DEFAULT_WIDTH,
        'height': CameraConfig.DEFAULT_HEIGHT,
        'backend': CameraConfig.BACKEND,
        'headless': CameraConfig.HEADLESS,
        'collect': TapDetectionConfig.COLLECT_TAP_DATA,
        'log_level': logging.getLogger().level,
    }
    yield
    CameraConfig.DEFAULT_WIDTH = saved['width']
    CameraConfig.DEFAULT_HEIGHT = saved['height']
    CameraConfig.BACKEND = saved['backend']
    CameraConfig.HEADLESS = saved['headless']
    TapDetectionConfig.COLLECT_TAP_DATA = saved['collect']
    logging.getLogger().setLevel(saved['log_level'])


def test_no_flags_changes_nothing(parser):
    before = CameraConfig.DEFAULT_WIDTH

    applied = apply_overrides(parser.parse_args([]), environ={})

    assert applied == []
    assert CameraConfig.DEFAULT_WIDTH == before


def test_resolution_flag_sets_both_dimensions(parser):
    apply_overrides(parser.parse_args(['--resolution', '640x480']), environ={})

    assert CameraConfig.DEFAULT_WIDTH == 640
    assert CameraConfig.DEFAULT_HEIGHT == 480


def test_malformed_resolution_is_rejected(parser):
    with pytest.raises(SystemExit):
        parser.parse_args(['--resolution', 'huge'])


def test_collect_tap_data_flag(parser):
    apply_overrides(parser.parse_args(['--collect-tap-data']), environ={})

    assert TapDetectionConfig.COLLECT_TAP_DATA is True


def test_headless_flag(parser):
    apply_overrides(parser.parse_args(['--headless']), environ={})

    assert CameraConfig.HEADLESS is True


def test_environment_is_read_when_the_flag_is_absent(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_RESOLUTION': '800x600'})

    assert CameraConfig.DEFAULT_WIDTH == 800
    assert CameraConfig.DEFAULT_HEIGHT == 600


def test_cli_wins_over_environment(parser):
    apply_overrides(
        parser.parse_args(['--resolution', '640x480']),
        environ={'CAMIO_RESOLUTION': '1920x1080'},
    )

    assert CameraConfig.DEFAULT_WIDTH == 640


def test_environment_collect_tap_data_is_read(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_COLLECT_TAP_DATA': 'yes'})

    assert TapDetectionConfig.COLLECT_TAP_DATA is True


def test_environment_headless_is_read(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_HEADLESS': '1'})

    assert CameraConfig.HEADLESS is True


def test_bad_environment_value_raises_with_the_variable_named(parser):
    with pytest.raises(ValueError, match='CAMIO_RESOLUTION'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_RESOLUTION': 'nope'})


def test_applied_overrides_are_reported_for_logging(parser):
    applied = apply_overrides(parser.parse_args(['--collect-tap-data']), environ={})

    assert len(applied) == 1
    assert 'COLLECT_TAP_DATA' in applied[0]


def test_camera_backend_flag_maps_to_an_opencv_constant(parser):
    apply_overrides(parser.parse_args(['--camera-backend', 'v4l2']), environ={})

    assert CameraConfig.BACKEND == cv.CAP_V4L2


def test_camera_backend_any_maps_to_cap_any(parser):
    """
    'any' is OpenCV's own auto-selection. It used to be spelled 'auto' and map
    to None, which the camera setup read as "no preference" and replaced with
    DirectShow - unavailable on Linux, so the camera never opened.
    """
    apply_overrides(parser.parse_args(['--camera-backend', 'any']), environ={})

    assert CameraConfig.BACKEND == cv.CAP_ANY


def test_every_backend_choice_maps_to_a_real_constant():
    """A name mapping to None would be silently rewritten to DirectShow."""
    assert all(value is not None for value in BACKENDS.values())


def test_camera_setup_passes_cap_any_through_unchanged():
    """
    cv.CAP_ANY is 0, so the truthiness test this replaced sent it to
    cv.CAP_DSHOW. No camera is opened here - only the selection is exercised.
    """
    CameraConfig.BACKEND = cv.CAP_ANY

    assert select_camera_backend() == cv.CAP_ANY


def test_camera_setup_falls_back_to_dshow_when_no_backend_is_set():
    """src/config.py sets BACKEND = None on Windows expecting DirectShow."""
    CameraConfig.BACKEND = None

    assert select_camera_backend() == cv.CAP_DSHOW


def test_camera_backend_is_read_from_the_environment(parser):
    apply_overrides(parser.parse_args([]), environ={'CAMIO_CAMERA_BACKEND': 'v4l2'})

    assert CameraConfig.BACKEND == cv.CAP_V4L2


def test_bad_camera_backend_in_environment_is_rejected(parser):
    with pytest.raises(ValueError, match='CAMIO_CAMERA_BACKEND'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_CAMERA_BACKEND': 'nope'})


def test_log_level_flag_sets_the_root_logger(parser):
    apply_overrides(parser.parse_args(['--log-level', 'DEBUG']), environ={})

    assert logging.getLogger().level == logging.DEBUG


def test_bad_log_level_in_environment_is_rejected(parser):
    with pytest.raises(ValueError, match='CAMIO_LOG_LEVEL'):
        apply_overrides(parser.parse_args([]), environ={'CAMIO_LOG_LEVEL': 'CHATTY'})
