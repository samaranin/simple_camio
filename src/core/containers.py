"""
Typed containers for the objects the main loop threads through its helpers.

These replaced two untyped dicts. A frozen dataclass does not make a mistyped
attribute a compile-time error - Python has no such thing - but a wrong field
name now fails where the container is built rather than at some far-away use,
an editor and ruff can see the misspelling, and each helper's signature states
what it actually requires.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Components:
    """Everything initialize_system() builds from a map model."""

    model: dict
    cam_port: Any
    model_detector: Any
    pose_detector: Any
    gesture_detector: Any
    motion_filter: Any
    interact: Any
    camio_player: Any
    crickets_player: Any
    heartbeat_player: Any


@dataclass(frozen=True)
class Workers:
    """The background threads and the queues and lock they share."""

    audio_worker: Any
    pose_worker: Any
    sift_worker: Any
    pose_queue: Any
    sift_queue: Any
    lock: Any
