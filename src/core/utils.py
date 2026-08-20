"""
Utility functions for Simple CamIO.

This module contains helper functions for camera management, file loading,
drawing, and other common operations.
"""

import os
import sys
import json
import cv2 as cv
import numpy as np
import logging

logger = logging.getLogger(__name__)


# ==================== Camera Management ====================

def list_camera_ports():
    """
    Test camera ports and return available and working ports.

    Returns:
        tuple: (available_ports, working_ports, non_working_ports)
               working_ports contains tuples of (port, height, width)
    """
    non_working_ports = []
    dev_port = 0
    working_ports = []
    available_ports = []

    # Stop testing after 3 consecutive non-working ports
    while len(non_working_ports) < 3:
        camera = cv.VideoCapture(dev_port)
        if not camera.isOpened():
            non_working_ports.append(dev_port)
            logger.info(f"Port {dev_port} is not working.")
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                logger.info(f"Port {dev_port} is working and reads images ({h} x {w})")
                working_ports.append((dev_port, h, w))
            else:
                logger.info(f"Port {dev_port} for camera ({h} x {w}) is present but does not read.")
                available_ports.append(dev_port)
        camera.release()
        dev_port += 1

    return available_ports, working_ports, non_working_ports


def _stdin_is_interactive():
    """True only when stdin is a terminal that can actually answer a prompt."""
    try:
        return sys.stdin is not None and sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def select_camera_port(preferred_port=None):
    """
    Select a camera port without ever blocking on input.

    Args:
        preferred_port (int, optional): Port to use as-is, skipping detection.
            Pass this (via --camera) for unattended and daemon runs.

    Returns:
        int: Selected camera port number
    """
    if preferred_port is not None:
        logger.info(f"Using camera port {preferred_port} (explicitly requested)")
        return preferred_port

    available_ports, working_ports, non_working_ports = list_camera_ports()

    if not working_ports:
        logger.warning("No working cameras detected, using default port 0")
        return 0

    if len(working_ports) == 1:
        logger.info(f"Auto-selected camera port {working_ports[0][0]}")
        return working_ports[0][0]

    for i, (port, height, width) in enumerate(working_ports):
        logger.info(f"Camera {i}) Port {port}: {height} x {width}")

    # Only a real terminal may be asked to choose. Under a daemon stdin is
    # /dev/null, and one USB camera often registers as two /dev/video nodes,
    # so prompting here used to hang the service forever with nothing logged.
    if not _stdin_is_interactive():
        port = working_ports[0][0]
        logger.warning(
            f"{len(working_ports)} cameras detected but stdin is not interactive; "
            f"using port {port}. Pass --camera to choose explicitly."
        )
        return port

    print("The following cameras were detected:")
    for i, (port, height, width) in enumerate(working_ports):
        print(f'{i}) Port {port}: {height} x {width}')
    try:
        selection = int(input("Please select which camera you would like to use: "))
        return working_ports[selection][0]
    except (EOFError, KeyboardInterrupt, ValueError, IndexError) as e:
        port = working_ports[0][0]
        logger.warning(f"Invalid camera selection ({e!r}); falling back to port {port}")
        return port


# ==================== File Loading ====================

def load_map_parameters(filename):
    """
    Load map parameters from a JSON configuration file.

    Args:
        filename (str): Path to the JSON configuration file

    Returns:
        dict: Map model parameters

    Raises:
        SystemExit: If the file is missing, unreadable, or has no "model" section
    """
    # Every failure below exits instead of waiting on stdin: under systemd that
    # wait never returns and the service hangs without logging a reason.
    if not os.path.isfile(filename):
        logger.error(f"No map parameters file found at {filename}")
        logger.error("Usage: simple_camio.py --input1 <filename>")
        sys.exit(1)

    try:
        with open(filename, 'r') as f:
            map_params = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        logger.error(f"Could not read map parameters from {filename}: {e}")
        sys.exit(1)

    if 'model' not in map_params:
        logger.error(f"Map parameters in {filename} have no 'model' section")
        sys.exit(1)

    logger.info(f"Loaded map parameters from {filename}")
    return map_params['model']


# ==================== Drawing Functions ====================

def draw_rectangle_on_image(image, map_shape, homography):
    """
    Draw a rectangle showing the detected map region.

    Args:
        image (numpy.ndarray): Image to draw on
        map_shape (tuple): Shape of the map image (height, width)
        homography (numpy.ndarray): 3x3 homography matrix

    Returns:
        numpy.ndarray: Image with rectangle drawn
    """
    img_corners = np.array([
        [0, 0],
        [map_shape[1], 0],
        [map_shape[1], map_shape[0]],
        [0, map_shape[0]]
    ], dtype=np.float32).reshape(-1, 1, 2)

    H_inv = np.linalg.inv(homography)
    pts = cv.perspectiveTransform(img_corners, H_inv)

    # Draw rectangle with lines
    pts_int = np.int32(pts)
    cv.polylines(image, [pts_int], isClosed=True, color=(0, 255, 0), thickness=3)

    # Draw corner dots for emphasis
    for pt in pts:
        cv.circle(image, (int(pt[0][0]), int(pt[0][1])), 5, (0, 255, 0), -1)

    return image


def draw_rectangle_from_points(image, pts, color=(0, 255, 0), thickness=3):
    """
    Draw a polygon from pre-computed transformed points.

    Args:
        image (numpy.ndarray): Image to draw on
        pts (numpy.ndarray): Points in cv.perspectiveTransform output format
        color (tuple): BGR color for the rectangle
        thickness (int): Line thickness

    Returns:
        numpy.ndarray: Image with rectangle drawn
    """
    if pts is None:
        return image

    try:
        pts_int = np.int32(pts)
        cv.polylines(image, [pts_int], isClosed=True, color=color, thickness=thickness)

        # Draw corner dots as subtle markers
        for pt in pts.reshape(-1, 2):
            cv.circle(image, (int(pt[0]), int(pt[1])), 4, color, -1)
    except Exception as e:
        logger.debug(f"Error drawing rectangle: {e}")

    return image


# ==================== Validation Functions ====================

def is_gesture_valid(gesture):
    """
    Check if a gesture location is valid.

    Args:
        gesture: Gesture location (should be array-like with at least 3 elements)

    Returns:
        bool: True if gesture is valid, False otherwise
    """
    if gesture is None:
        return False
    if not hasattr(gesture, "__len__"):
        return False
    try:
        arr = np.asarray(gesture)
        return arr.size >= 3
    except Exception:
        return False


def normalize_gesture_location(gesture_loc):
    """
    Normalize gesture location to ensure it's a 1D array with 3 elements.

    Args:
        gesture_loc: Raw gesture location data

    Returns:
        numpy.ndarray or None: Normalized [x, y, z] array or None if invalid
    """
    if gesture_loc is None:
        return None

    try:
        arr = np.asarray(gesture_loc)

        if arr.size == 0:
            return None
        elif arr.size >= 3:
            # If multiple triplets, take the last (most recent)
            if arr.size % 3 == 0 and arr.size > 3:
                return arr.reshape(-1, 3)[-1].astype(float)
            else:
                # Take first 3 elements as fallback
                return arr.flatten()[:3].astype(float)
        else:
            return None
    except Exception as e:
        logger.debug(f"Error normalizing gesture: {e}")
        return None


# ==================== Color Conversion ====================

def color_to_index(color):
    """
    Convert BGR color tuple to a unique integer index.

    Args:
        color (tuple/list): BGR color values [B, G, R]

    Returns:
        int: Unique index for the color
    """
    return 256 * 256 * color[2] + 256 * color[1] + color[0]

