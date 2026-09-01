"""
Tracking state published across threads.

The SIFT detector's tracking state is a cluster of related attributes that only
the SIFTWorker thread may touch. Consumers on the main thread read an immutable
snapshot of them instead, so a guard and the read it guards can never disagree.
"""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class TrackingSnapshot:
    """
    One consistent view of the SIFT tracking state.

    Attributes:
        H: Homography matrix, or None when the map is not currently tracked.
        rect_pts: Projected template corners in camera coordinates, or None.
        tracking: True while the detector holds a valid homography.
        age: Frames since the last successful detection.
        status_text: Display string, rendered by the thread that owns the state.
        detect_generation: Monotonic counter, incremented once per successful
            homography. A consumer flashes the tracking rectangle when the value
            differs from the one it last saw; nothing has to be cleared, so a
            detection landing between two reads cannot be missed.
    """

    H: Optional[Any] = None
    rect_pts: Optional[Any] = None
    tracking: bool = False
    age: int = 0
    status_text: str = "SEARCHING FOR MAP"
    detect_generation: int = 0
