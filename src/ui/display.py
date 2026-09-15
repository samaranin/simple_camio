"""
UI Display Module - Drawing and visualization functions for CamIO.

This module contains all UI-related functions for drawing overlays,
tracking rectangles, status information, and other visual elements.
"""

import cv2 as cv
import time
import logging

from src.config import UIConfig, CameraConfig
from src.core.utils import draw_rectangle_from_points, draw_rectangle_on_image
from src.core.camera_thread import ThreadedCamera

logger = logging.getLogger(__name__)


def draw_map_tracking(display_img, snapshot, interact, rect_flash_remaining):
    """
    Draw the map tracking rectangle on the display image.

    Args:
        display_img: Image to draw on
        snapshot (TrackingSnapshot): Consistent view of the tracking state
        interact: Interaction policy with map shape
        rect_flash_remaining (int): Frames remaining for flash effect

    Returns:
        tuple: (updated_image, updated_flash_remaining)
    """
    rect_pts = snapshot.rect_pts

    if rect_pts is not None:
        if rect_flash_remaining > 0:
            display_img = draw_rectangle_from_points(
                display_img, rect_pts,
                color=UIConfig.COLOR_YELLOW, thickness=5
            )
            rect_flash_remaining -= 1
        else:
            display_img = draw_rectangle_from_points(
                display_img, rect_pts,
                color=UIConfig.COLOR_GREEN, thickness=3
            )
    elif snapshot.H is not None:
        display_img = draw_rectangle_on_image(
            display_img, interact.image_map_color.shape, snapshot.H
        )

    return display_img, rect_flash_remaining


def draw_ui_overlay(display_img, snapshot, gesture_status, timer, fps_state, cap):
    """
    Draw status information overlay on the display image.

    Args:
        display_img: Image to draw on
        snapshot (TrackingSnapshot): Consistent view of the tracking state
        gesture_status: Current gesture status
        timer (float): Previous frame timestamp for FPS calculation
        fps_state (dict): FPS tracking state with keys: 'display_count', 
                         'start_time', 'display_fps'
        cap: Camera capture object (to get camera FPS)

    Returns:
        tuple: (current_timestamp, updated_fps_state)
    """
    # Tracking status, rendered by the thread that owns the tracking state
    status_text = snapshot.status_text
    cv.putText(display_img, status_text, (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
              UIConfig.COLOR_GREEN, UIConfig.FONT_THICKNESS)

    # Gesture status
    if gesture_status:
        cv.putText(display_img, f"Gesture: {gesture_status}", (10, 120),
                  cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
                  UIConfig.COLOR_YELLOW, UIConfig.FONT_THICKNESS)

    # FPS counters - update display FPS every second
    current_time = time.time()
    fps_state['display_count'] += 1
    elapsed = current_time - fps_state['start_time']
    
    if elapsed >= 1.0:  # Update FPS every second
        fps_state['display_fps'] = fps_state['display_count'] / elapsed
        fps_state['display_count'] = 0
        fps_state['start_time'] = current_time
    
    # Get camera FPS from the capture device
    camera_fps = 0.0
    try:
        if hasattr(cap, 'get_fps'):
            # ThreadedCamera has get_fps() method
            camera_fps = cap.get_fps()
        elif hasattr(cap, 'get'):
            # Regular VideoCapture - use CAP_PROP_FPS (not always accurate for actual rate)
            camera_fps = cap.get(cv.CAP_PROP_FPS)
    except Exception:
        camera_fps = 0.0
    
    # Display camera FPS (actual capture rate from device)
    if camera_fps > 0:
        camera_fps_text = f"Camera: {camera_fps:.1f} FPS"
        cv.putText(display_img, camera_fps_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
                  UIConfig.COLOR_GREEN, UIConfig.FONT_THICKNESS)
    
    # Display actual processing/rendering FPS
    if fps_state['display_fps'] > 0:
        display_fps_text = f"Processing: {fps_state['display_fps']:.1f} FPS"
        cv.putText(display_img, display_fps_text, (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
                  UIConfig.COLOR_CYAN, UIConfig.FONT_THICKNESS)

    return current_time, fps_state


def select_camera_backend():
    """
    Choose the OpenCV capture backend from the configuration.

    CameraConfig.BACKEND can legitimately hold a falsy constant - cv.CAP_ANY is
    0 - so a truthiness test here quietly rewrote 'any' into DirectShow, which
    does not exist on Linux. Only an absent or None value means "no
    preference", and on Windows that means DirectShow: src/config.py sets
    BACKEND = None there expecting exactly this fallback.

    Returns:
        int: OpenCV capture API preference to pass to cv.VideoCapture.
    """
    backend = getattr(CameraConfig, 'BACKEND', None)
    return cv.CAP_DSHOW if backend is None else backend


def setup_camera(cam_port):
    """
    Initialize and configure the camera.

    Args:
        cam_port (int): Camera port number

    Returns:
        cv.VideoCapture: Configured camera capture object
    """
    logger.info(f"Setting up camera on port {cam_port}")

    # Use configured backend or default
    backend = select_camera_backend()
    cap = cv.VideoCapture(cam_port, backend)
    
    # Set buffer size BEFORE other properties to reduce latency
    cap.set(cv.CAP_PROP_BUFFERSIZE, CameraConfig.BUFFER_SIZE)
    
    # Set frame rate to target FPS if possible
    cap.set(cv.CAP_PROP_FPS, CameraConfig.TARGET_FPS)
    
    # Set resolution
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CameraConfig.DEFAULT_HEIGHT)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CameraConfig.DEFAULT_WIDTH)
    cap.set(cv.CAP_PROP_FOCUS, CameraConfig.FOCUS)
    
    # Log actual settings
    actual_fps = cap.get(cv.CAP_PROP_FPS)
    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    actual_buffer = cap.get(cv.CAP_PROP_BUFFERSIZE)
    logger.info(f"Camera configured: {actual_width:.0f}x{actual_height:.0f} @ {actual_fps:.1f}fps, buffer={actual_buffer:.0f}")

    # Enable OpenCV optimizations and set a reasonable thread count
    try:
        cv.setUseOptimized(True)
    except Exception:
        pass
    try:
        # Use about half the CPUs to reduce contention with Python threads
        num_threads = max(1, int(getattr(cv, "getNumberOfCPUs", lambda: 4)() // 2))
        cv.setNumThreads(num_threads)
    except Exception:
        pass

    # Wrap in threaded capture if enabled
    if CameraConfig.USE_THREADED_CAPTURE:
        logger.info("Using threaded camera capture for improved FPS")
        cap = ThreadedCamera(cap)

    return cap
