"""
Configuration module for Simple CamIO.

This module contains all configuration parameters and constants used throughout the application.
Centralizing configuration makes it easier to tune parameters and understand system behavior.
"""

import logging


# ==================== Logging Configuration ====================
LOG_LEVEL = logging.INFO
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'


# ==================== Camera Configuration ====================
class CameraConfig:
    """Camera capture configuration parameters."""

    # Default camera resolution
    DEFAULT_WIDTH = 1920
    DEFAULT_HEIGHT = 1080

    # Camera buffer size (reduce latency)
    BUFFER_SIZE = 1

    # Auto-focus setting
    FOCUS = 0

    # Processing scale for pose detection (smaller = faster but less accurate)
    POSE_PROCESSING_SCALE = 0.35


# ==================== Movement Filter Configuration ====================
class MovementFilterConfig:
    """Configuration for movement filtering algorithms."""

    # Exponential smoothing factor for simple filter
    BETA = 0.5

    # Maximum queue length for median filter
    MAX_QUEUE_LENGTH = 30

    # Time window for averaging positions (seconds)
    AVERAGING_TIME = 0.7


# ==================== Gesture Detection Configuration ====================
class GestureDetectorConfig:
    """Configuration for gesture recognition."""

    # Maximum queue length for position history
    MAX_QUEUE_LENGTH = 30

    # Time threshold for dwell detection (seconds)
    DWELL_TIME_THRESH = 0.75

    # Movement thresholds for still/moving detection
    X_MVMNT_THRESH = 0.95
    Y_MVMNT_THRESH = 0.95
    Z_MVMNT_THRESH = 4.0


# ==================== Tap Detection Configuration ====================
class TapDetectionConfig:
    """Configuration for single and double-tap detection."""

    # Z-based tap detection parameters
    TAP_BASE_DELTA = 0.025          # Base z delta vs baseline to start a press
    TAP_NOISE_MULT = 3.0            # Multiplier on median |dz| to raise threshold in noise
    TAP_MIN_VEL = 0.2               # Min negative z velocity to start a press
    TAP_RELEASE_VEL = 0.15          # Min positive z velocity to consider release
    TAP_MIN_DURATION = 0.05         # Minimum tap duration (seconds)
    TAP_MAX_DURATION = 0.50         # Maximum tap duration (seconds)
    TAP_MIN_INTERVAL = 0.05         # Minimum interval between taps (seconds)
    TAP_MAX_INTERVAL = 1.00         # Maximum interval between taps (seconds)
    TAP_MIN_PRESS_DEPTH = 0.010     # Minimal press depth needed to consider a tap
    TAP_MAX_XY_DRIFT = 180.0        # Maximum XY drift during a tap
    TAP_MAX_RELEASE_BACK = 0.45     # Fraction of press depth required for release

    # History buffer sizes
    Z_HISTORY_LEN = 7
    XY_HISTORY_LEN = 7

    # Cooldown period after double-tap detection (seconds)
    TAP_COOLDOWN = 0.7
    DOUBLE_TAP_COOLDOWN_MAIN = 0.7

    # Angle-based distal flexion thresholds (degrees)
    ANG_HISTORY_LEN = 7
    ANG_BASE_DELTA = 12.0           # Min angle rise above baseline to start a press
    ANG_NOISE_MULT = 3.0            # Noise-adaptive margin
    ANG_MIN_VEL = 120.0             # Deg/s minimum rising angular velocity
    ANG_RELEASE_VEL = -120.0        # Deg/s negative velocity (falling) for release
    ANG_MIN_PRESS_DEPTH = 10.0      # Degrees (min peak flexion over baseline)
    ANG_RELEASE_BACK = 0.5          # Fraction of peak angle to return for release

    # Enhanced detection parameters (used by PoseDetectorMPEnhanced)
    # Palm plane penetration (signed distance) thresholds
    PLANE_BASE_DELTA = 0.010
    PLANE_NOISE_MULT = 4.0
    PLANE_MIN_PRESS_DEPTH = 0.008
    PLANE_RELEASE_BACK = 0.45

    # Relative depth (tip Z relative to palm) thresholds
    ZREL_BASE_DELTA = 0.010
    ZREL_NOISE_MULT = 4.0
    ZREL_MIN_PRESS_DEPTH = 0.010

    # Temporal smoothing (EMA)
    EWMA_ALPHA = 0.35

    # Motion stability gates
    STABLE_XY_VEL_MAX = 50.0        # px/s
    STABLE_ROT_MAX = 0.25           # rad/s

    # Landmark confidence/jitter gates
    MIN_HAND_SCORE = 0.65
    JITTER_MAX_PX = 3.0

    # Ray-projection velocity threshold
    RAY_MIN_IN_VEL = 0.10           # norm units/s

    # Stronger pointing gate thresholds
    INDEX_STRONG_MIN = 0.78
    OTHERS_STRONG_MAX = 0.92

    # Tiny classifier over engineered features
    # Note: list is fine; implementation will coerce to np.array
    CLS_WEIGHTS = [2.0, 1.2, 1.0, -0.8, -0.9, -0.4, 0.6]
    CLS_BIAS = -2.0
    CLS_MIN_PROB = 0.65


# ==================== Interaction Policy Configuration ====================
class InteractionConfig:
    """Configuration for 2D interaction policy."""

    # Size of the zone filter buffer
    ZONE_FILTER_SIZE = 10

    # Z-axis threshold for touch detection (cm)
    Z_THRESHOLD = 2.0


# ==================== SIFT Detection Configuration ====================
class SIFTConfig:
    """Configuration for SIFT-based model detection."""

    # SIFT feature extraction parameters
    SIFT_N_FEATURES = 2000
    SIFT_CONTRAST_THRESHOLD = 0.03
    SIFT_EDGE_THRESHOLD = 15

    # ORB feature extraction parameters (fallback)
    ORB_N_FEATURES = 2000
    ORB_SCALE_FACTOR = 1.2
    ORB_N_LEVELS = 12

    # Corner detection parameters
    CORNER_MAX_CORNERS = 500
    CORNER_QUALITY_LEVEL = 0.01
    CORNER_MIN_DISTANCE = 10

    # Matching parameters
    FLANN_TREES = 8
    FLANN_CHECKS = 100
    RATIO_THRESH = 0.8              # Lowe's ratio test threshold

    # Homography computation
    MIN_INLIER_COUNT = 10
    RANSAC_REPROJ_THRESHOLD = 5.0
    RANSAC_CONFIDENCE = 0.99
    RANSAC_MAX_ITERS = 5000

    # Tracking quality monitoring
    REDETECT_INTERVAL = 150         # Force validation every N frames
    MIN_TRACKING_QUALITY = 8        # Minimum inliers to maintain tracking

    # Quick validation parameters
    VALIDATION_INTERVAL = 2.0       # Seconds between validation checks
    VALIDATION_MIN_MATCHES = 6
    VALIDATION_POSITION_THRESHOLD = 40  # Pixels


# ==================== MediaPipe Hand Detection Configuration ====================
class MediaPipeConfig:
    """Configuration for MediaPipe hand tracking."""

    # Hand detection parameters
    MODEL_COMPLEXITY = 1
    MIN_DETECTION_CONFIDENCE = 0.75
    MIN_TRACKING_CONFIDENCE = 0.75
    MAX_NUM_HANDS = 2


# ==================== Audio Configuration ====================
class AudioConfig:
    """Configuration for audio playback."""

    # Ambient sound volume levels
    HEARTBEAT_VOLUME = 0.05


# ==================== UI Configuration ====================
class UIConfig:
    """Configuration for user interface elements."""

    # Rectangle flash when homography updates
    RECT_FLASH_FRAMES = 10

    # Colors (BGR format)
    COLOR_GREEN = (0, 255, 0)
    COLOR_YELLOW = (0, 255, 255)
    COLOR_BLUE = (255, 0, 0)

    # Text display
    FONT = 1  # cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2


# ==================== Worker Thread Configuration ====================
class WorkerConfig:
    """Configuration for background worker threads."""

    # Queue sizes
    POSE_QUEUE_MAXSIZE = 1
    SIFT_QUEUE_MAXSIZE = 1

    # Queue timeout (seconds)
    QUEUE_TIMEOUT = 0.1
    QUEUE_GET_TIMEOUT = 0.2

    # SIFT worker retry attempts
    SIFT_RETRY_ATTEMPTS = 3

    # Thread shutdown timeout (seconds)
    THREAD_SHUTDOWN_TIMEOUT = 2.0
