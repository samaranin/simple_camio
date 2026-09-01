"""
Runtime configuration overrides.

The config classes hold their values as class attributes, so changing one used
to mean editing source - README still tells the reader to set
TapDetectionConfig.COLLECT_TAP_DATA by hand. These helpers let a flag or an
environment variable do it instead, extending the pattern --headless already
established.

Precedence: command-line flag, then CAMIO_* environment variable, then the
value on the class.
"""

import logging
import os

import cv2 as cv

from src.config import CameraConfig, TapDetectionConfig

logger = logging.getLogger(__name__)

BACKENDS = {
    'auto': None,
    'v4l2': cv.CAP_V4L2,
    'dshow': cv.CAP_DSHOW,
    'msmf': cv.CAP_MSMF,
    'any': cv.CAP_ANY,
}

LOG_LEVELS = ('DEBUG', 'INFO', 'WARNING', 'ERROR')

TRUTHY = ('1', 'true', 'yes', 'on')


def _resolution(value, source):
    """Parse WxH into a pair of ints, naming its source if it is malformed."""
    try:
        width, height = value.lower().split('x')
        return int(width), int(height)
    except (ValueError, AttributeError):
        raise ValueError(
            f'{source}: expected a resolution like 1280x720, got {value!r}'
        )


def add_arguments(parser):
    """Register the override flags on an existing parser."""
    parser.add_argument(
        '--headless', action='store_true', default=False,
        help='Run without a display window - useful for Raspberry Pi daemon mode'
    )
    parser.add_argument(
        '--resolution', type=lambda v: _resolution(v, '--resolution'), default=None,
        metavar='WxH', help='Camera capture resolution, e.g. 640x480'
    )
    parser.add_argument(
        '--camera-backend', choices=sorted(BACKENDS), default=None,
        help='OpenCV capture backend. v4l2 on Linux, dshow or msmf on Windows'
    )
    parser.add_argument(
        '--collect-tap-data', action='store_true', default=False,
        help='Record tap detection samples to data/tap_dataset for training'
    )
    parser.add_argument(
        '--log-level', choices=LOG_LEVELS, default=None,
        help='Logging verbosity (default: INFO)'
    )


def apply_overrides(args, environ=None):
    """
    Apply command-line and environment overrides to the config classes.

    Args:
        args (argparse.Namespace): Parsed arguments from a parser that
            add_arguments() was called on.
        environ (dict, optional): Environment to read. Defaults to os.environ.

    Returns:
        list[str]: One line per override applied, for logging.

    Raises:
        ValueError: If a CAMIO_* variable holds a value that cannot be parsed.
    """
    env = os.environ if environ is None else environ
    applied = []

    resolution = getattr(args, 'resolution', None)
    if resolution is None and env.get('CAMIO_RESOLUTION'):
        resolution = _resolution(env['CAMIO_RESOLUTION'], 'CAMIO_RESOLUTION')
    if resolution is not None:
        CameraConfig.DEFAULT_WIDTH, CameraConfig.DEFAULT_HEIGHT = resolution
        applied.append(f'CameraConfig.DEFAULT_WIDTH/HEIGHT = {resolution[0]}x{resolution[1]}')

    backend = getattr(args, 'camera_backend', None)
    if backend is None and env.get('CAMIO_CAMERA_BACKEND'):
        backend = env['CAMIO_CAMERA_BACKEND'].lower()
        if backend not in BACKENDS:
            raise ValueError(
                f'CAMIO_CAMERA_BACKEND: expected one of {sorted(BACKENDS)}, '
                f'got {backend!r}'
            )
    if backend is not None:
        CameraConfig.BACKEND = BACKENDS[backend]
        applied.append(f'CameraConfig.BACKEND = {backend}')

    headless = getattr(args, 'headless', False)
    if not headless:
        headless = env.get('CAMIO_HEADLESS', '').lower() in TRUTHY
    if headless:
        CameraConfig.HEADLESS = True
        applied.append('CameraConfig.HEADLESS = True')

    collect = getattr(args, 'collect_tap_data', False)
    if not collect:
        collect = env.get('CAMIO_COLLECT_TAP_DATA', '').lower() in TRUTHY
    if collect:
        TapDetectionConfig.COLLECT_TAP_DATA = True
        applied.append('TapDetectionConfig.COLLECT_TAP_DATA = True')

    level = getattr(args, 'log_level', None)
    if level is None and env.get('CAMIO_LOG_LEVEL'):
        level = env['CAMIO_LOG_LEVEL'].upper()
        if level not in LOG_LEVELS:
            raise ValueError(
                f'CAMIO_LOG_LEVEL: expected one of {list(LOG_LEVELS)}, got {level!r}'
            )
    if level is not None:
        logging.getLogger().setLevel(getattr(logging, level))
        applied.append(f'log level = {level}')

    return applied
