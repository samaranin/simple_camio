"""
UI Display Module - Drawing and visualization functions for CamIO.

This module contains all UI-related functions for drawing overlays,
tracking rectangles, status information, and other visual elements.
"""

import cv2 as cv
import time
import logging

from src.config import UIConfig

logger = logging.getLogger(__name__)


def draw_map_tracking(display_img, model_detector, interact, rect_flash_remaining):
    """
    Draw the map tracking rectangle on the display image.

    Args:
        display_img: Image to draw on
        model_detector: SIFT detector with tracking info
        interact: Interaction policy with map shape
        rect_flash_remaining (int): Frames remaining for flash effect

    Returns:
        tuple: (updated_image, updated_flash_remaining)
    """
    from src.core.utils import draw_rectangle_from_points, draw_rectangle_on_image
    
    if getattr(model_detector, 'last_rect_pts', None) is not None:
        if rect_flash_remaining > 0:
            display_img = draw_rectangle_from_points(
                display_img, model_detector.last_rect_pts,
                color=UIConfig.COLOR_YELLOW, thickness=5
            )
            rect_flash_remaining -= 1
        else:
            display_img = draw_rectangle_from_points(
                display_img, model_detector.last_rect_pts,
                color=UIConfig.COLOR_GREEN, thickness=3
            )
    else:
        display_img = draw_rectangle_on_image(
            display_img, interact.image_map_color.shape, model_detector.H
        )

    return display_img, rect_flash_remaining


def draw_ui_overlay(display_img, model_detector, gesture_status, timer):
    """
    Draw status information overlay on the display image.

    Args:
        display_img: Image to draw on
        model_detector: SIFT detector for status
        gesture_status: Current gesture status
        timer (float): Previous frame timestamp for FPS calculation

    Returns:
        float: Current timestamp (new timer value)
    """
    # Tracking status
    status_text = model_detector.get_tracking_status()
    cv.putText(display_img, status_text, (10, 30),
              cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
              UIConfig.COLOR_GREEN, UIConfig.FONT_THICKNESS)

    # Gesture status
    if gesture_status:
        cv.putText(display_img, f"Gesture: {gesture_status}", (10, 90),
                  cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
                  UIConfig.COLOR_YELLOW, UIConfig.FONT_THICKNESS)

    # FPS counter
    prev_time = timer
    current_time = time.time()
    elapsed_time = current_time - prev_time
    if elapsed_time > 0:
        fps_text = f"FPS: {1/elapsed_time:.1f}"
        cv.putText(display_img, fps_text, (10, 60),
                  cv.FONT_HERSHEY_SIMPLEX, UIConfig.FONT_SCALE,
                  UIConfig.COLOR_GREEN, UIConfig.FONT_THICKNESS)

    return current_time


def setup_camera(cam_port):
    """
    Initialize and configure the camera.

    Args:
        cam_port (int): Camera port number

    Returns:
        cv.VideoCapture: Configured camera capture object
    """
    from src.config import CameraConfig
    
    logger.info(f"Setting up camera on port {cam_port}")

    cap = cv.VideoCapture(cam_port, cv.CAP_DSHOW)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, CameraConfig.DEFAULT_HEIGHT)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, CameraConfig.DEFAULT_WIDTH)
    cap.set(cv.CAP_PROP_FOCUS, CameraConfig.FOCUS)

    # Reduce latency
    try:
        cap.set(cv.CAP_PROP_BUFFERSIZE, CameraConfig.BUFFER_SIZE)
    except Exception as e:
        logger.warning(f"Could not set camera buffer size: {e}")

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

    return cap
