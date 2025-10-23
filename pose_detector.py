"""
MediaPipe-based hand pose detection for Simple CamIO.

This module handles hand tracking, pointing gesture recognition, and tap detection
using MediaPipe's hand landmark detection.
"""

import numpy as np
import cv2 as cv
import mediapipe as mp
from collections import deque
import time
import logging
from google.protobuf.json_format import MessageToDict
from config import MediaPipeConfig, TapDetectionConfig

logger = logging.getLogger(__name__)

# --- Inject defaults for enhanced config fields if missing ---
def _ensure_enhanced_config_defaults():
    defaults = {
        'PLANE_BASE_DELTA': 0.010,
        'PLANE_NOISE_MULT': 4.0,
        'PLANE_MIN_PRESS_DEPTH': 0.008,
        'PLANE_RELEASE_BACK': 0.45,
        'ZREL_BASE_DELTA': 0.010,
        'ZREL_NOISE_MULT': 4.0,
        'ZREL_MIN_PRESS_DEPTH': 0.010,
        'EWMA_ALPHA': 0.35,
        'STABLE_XY_VEL_MAX': 50.0,
        'STABLE_ROT_MAX': 0.25,
        'MIN_HAND_SCORE': 0.65,
        'JITTER_MAX_PX': 3.0,
        'RAY_MIN_IN_VEL': 0.10,
        'INDEX_STRONG_MIN': 0.78,
        'OTHERS_STRONG_MAX': 0.92,
        'CLS_WEIGHTS': np.array([2.0, 1.2, 1.0, -0.8, -0.9, -0.4, 0.6], dtype=float),
        'CLS_BIAS': -2.0,
        'CLS_MIN_PROB': 0.65,
    }
    for k, v in defaults.items():
        if not hasattr(TapDetectionConfig, k):
            setattr(TapDetectionConfig, k, v)
            logger.debug(f"TapDetectionConfig: defaulted {k}={v}")

_ensure_enhanced_config_defaults()

class PoseDetectorMP:
    """
    MediaPipe-based hand pose detector with tap recognition.

    This class detects hands in camera frames, recognizes pointing gestures,
    and detects single/double taps using both Z-depth and finger flexion angles.
    """

    def __init__(self, model):
        """
        Initialize the MediaPipe pose detector.

        Args:
            model (dict): Map model configuration
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=MediaPipeConfig.MODEL_COMPLEXITY,
            min_detection_confidence=MediaPipeConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MediaPipeConfig.MIN_TRACKING_CONFIDENCE,
            max_num_hands=MediaPipeConfig.MAX_NUM_HANDS
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)

        # Initialize tap detection state (keyed by hand label: 'Left'/'Right')
        self._tap_state = {}

        # Load tap detection configuration
        self._load_tap_config()

        logger.info("Initialized MediaPipe pose detector")

    def _load_tap_config(self):
        """Load tap detection configuration parameters."""
        cfg = TapDetectionConfig

        # Z-based tap detection parameters
        self.TAP_BASE_DELTA = cfg.TAP_BASE_DELTA
        self.TAP_NOISE_MULT = cfg.TAP_NOISE_MULT
        self.TAP_MIN_VEL = cfg.TAP_MIN_VEL
        self.TAP_RELEASE_VEL = cfg.TAP_RELEASE_VEL
        self.TAP_MIN_DURATION = cfg.TAP_MIN_DURATION
        self.TAP_MAX_DURATION = cfg.TAP_MAX_DURATION
        self.TAP_MIN_INTERVAL = cfg.TAP_MIN_INTERVAL
        self.TAP_MAX_INTERVAL = cfg.TAP_MAX_INTERVAL
        self.TAP_MIN_PRESS_DEPTH = cfg.TAP_MIN_PRESS_DEPTH
        self.TAP_MAX_XY_DRIFT = cfg.TAP_MAX_XY_DRIFT
        self.TAP_MAX_RELEASE_BACK = cfg.TAP_MAX_RELEASE_BACK
        self.Z_HISTORY_LEN = cfg.Z_HISTORY_LEN
        self.XY_HISTORY_LEN = cfg.XY_HISTORY_LEN
        self.TAP_COOLDOWN = cfg.TAP_COOLDOWN

        # Angle-based tap detection parameters
        self.ANG_HISTORY_LEN = cfg.ANG_HISTORY_LEN
        self.ANG_BASE_DELTA = cfg.ANG_BASE_DELTA
        self.ANG_NOISE_MULT = cfg.ANG_NOISE_MULT
        self.ANG_MIN_VEL = cfg.ANG_MIN_VEL
        self.ANG_RELEASE_VEL = cfg.ANG_RELEASE_VEL
        self.ANG_MIN_PRESS_DEPTH = cfg.ANG_MIN_PRESS_DEPTH
        self.ANG_RELEASE_BACK = cfg.ANG_RELEASE_BACK

    def detect(self, image, H, _, processing_scale=0.5, draw=False):
        """
        Detect hand poses and gestures in the image.

        Args:
            image (numpy.ndarray): Input camera frame
            H (numpy.ndarray): Homography matrix for coordinate transformation
            _ : Unused parameter (kept for compatibility)
            processing_scale (float): Scale factor for processing (smaller = faster)
            draw (bool): Whether to draw hand landmarks on the image

        Returns:
            tuple: (index_pos, movement_status, img_out)
                   index_pos: Position of index finger [x, y, z] or None
                   movement_status: 'pointing', 'moving', 'double_tap', etc.
                   img_out: Annotated image if draw=True, else None
        """
        # Downscale image for faster processing
        if processing_scale < 1.0:
            small = cv.resize(image, (0, 0), fx=processing_scale, fy=processing_scale,
                            interpolation=cv.INTER_LINEAR)
        else:
            small = image

        # Convert to RGB for MediaPipe
        small_rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        results = self.hands.process(small_rgb)

        img_out = image.copy() if draw else None
        index_pos = None
        movement_status = None
        double_tap_emitted = False

        if results.multi_hand_landmarks:
            orig_h, orig_w = image.shape[0], image.shape[1]

            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Get hand label ('Left' or 'Right') for stable tracking
                handedness = MessageToDict(results.multi_handedness[h])['classification'][0]['label']
                hand_key = handedness

                # Detect pointing gesture
                is_pointing = self._detect_pointing_gesture(hand_landmarks, orig_w, orig_h)

                # Draw hand landmarks if requested
                if draw:
                    self._draw_hand_landmarks(img_out, hand_landmarks)

                # Get index finger tip position
                pos_x, pos_y, position = self._get_finger_position(
                    hand_landmarks, 8, orig_w, orig_h, H
                )

                if index_pos is None:
                    index_pos = np.array([
                        position[0] / position[2],
                        position[1] / position[2],
                        0
                    ], dtype=float)

                # Update movement status based on pointing gesture
                if movement_status != 'double_tap':
                    movement_status = self._update_movement_status(
                        hand_landmarks, is_pointing, index_pos, position, movement_status
                    )

                # Detect taps (single and double)
                tap_detected = self._detect_taps(
                    hand_landmarks, hand_key, is_pointing, pos_x, pos_y,
                    orig_w, orig_h, draw, img_out
                )

                if tap_detected:
                    movement_status = 'double_tap'
                    double_tap_emitted = True
                    break

        # Normalize index position to always return proper format
        normalized = self._normalize_index_position(index_pos)

        return normalized, movement_status, img_out

    def _detect_pointing_gesture(self, hand_landmarks, width, height):
        """
        Detect if hand is in pointing gesture (index extended, others curled).

        Args:
            hand_landmarks: MediaPipe hand landmarks
            width (int): Image width
            height (int): Image height

        Returns:
            bool: True if pointing gesture detected
        """
        def L(i):
            """Get landmark position in pixels."""
            lm = hand_landmarks.landmark[i]
            return np.array([lm.x * width, lm.y * height, lm.z], dtype=float)

        # Calculate finger extension ratios
        ratios = {}
        finger_indices = {
            'thumb': [1, 2, 3, 4],
            'index': [5, 6, 7, 8],
            'middle': [9, 10, 11, 12],
            'ring': [13, 14, 15, 16],
            'little': [17, 18, 19, 20]
        }

        for finger, indices in finger_indices.items():
            coors = np.array([L(idx) for idx in indices])
            ratios[finger] = self.ratio(coors)

        # Check if other fingers are curled (blocking index extension line)
        a = L(5)  # Index MCP
        ab = L(8) - L(5)  # Index vector
        is_pointing = True

        for finger in ['middle', 'ring', 'little']:
            indices = finger_indices[finger]
            for idx in indices:
                ap = L(idx) - a
                if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                    is_pointing = False

        # Alternative check: index extended, others curled
        overall = ratios['index'] - (
            (ratios['middle'] + ratios['ring'] + ratios['little']) / 3
        )
        is_pointing = is_pointing or overall > 0.1

        return is_pointing

    def _get_finger_position(self, hand_landmarks, landmark_idx, width, height, H):
        """
        Get finger position transformed by homography.

        Args:
            hand_landmarks: MediaPipe hand landmarks
            landmark_idx (int): Landmark index to get position for
            width (int): Image width
            height (int): Image height
            H (numpy.ndarray): Homography matrix

        Returns:
            tuple: (pos_x, pos_y, position) in transformed coordinates
        """
        lm = hand_landmarks.landmark[landmark_idx]
        pos_x = lm.x * width
        pos_y = lm.y * height
        position = np.matmul(H, np.array([pos_x, pos_y, 1]))
        return pos_x, pos_y, position

    def _update_movement_status(self, hand_landmarks, is_pointing, index_pos,
                                position, current_status):
        """
        Update movement status based on hand configuration.

        Returns:
            str: Updated movement status
        """
        # Calculate finger ratios
        def L(i):
            lm = hand_landmarks.landmark[i]
            return np.array([lm.x, lm.y, lm.z], dtype=float)

        index_coors = np.array([L(i) for i in [5, 6, 7, 8]])
        ratio_index = self.ratio(index_coors)

        middle_coors = np.array([L(i) for i in [9, 10, 11, 12]])
        ratio_middle = self.ratio(middle_coors)

        ring_coors = np.array([L(i) for i in [13, 14, 15, 16]])
        ratio_ring = self.ratio(ring_coors)

        little_coors = np.array([L(i) for i in [17, 18, 19, 20]])
        ratio_little = self.ratio(little_coors)

        # Check for pointing configuration
        if (is_pointing or
            (ratio_index > 0.7 and ratio_middle < 0.95 and
             ratio_ring < 0.95 and ratio_little < 0.95)):
            if current_status != "pointing":
                return "pointing"
            else:
                # Multiple hands pointing
                return "too_many"
        elif current_status != "pointing":
            return "moving"

        return current_status

    def _detect_taps(self, hand_landmarks, hand_key, is_pointing, pos_x, pos_y,
                    width, height, draw, img_out):
        """
        Detect single and double taps using Z-depth and finger angle analysis.

        Returns:
            bool: True if double tap detected
        """
        try:
            now = time.time()
            lm8_z = float(hand_landmarks.landmark[8].z)

            # Compute distal flexion angle
            angle_deg = self._compute_finger_flexion_angle(hand_landmarks, width, height)

            # Get or create tap state for this hand
            state = self._get_tap_state(hand_key, lm8_z, angle_deg, pos_x, pos_y, now)

            # Update histories
            state['z_history'].append(lm8_z)
            state['xy_history'].append((pos_x, pos_y))
            state['ang_history'].append(angle_deg)

            # Calculate baselines and noise
            baseline_z, noise_z, dz_press = self._calculate_z_baseline(state)
            baseline_ang, noise_ang, dang_press = self._calculate_angle_baseline(state)

            # Calculate velocities
            dt = max(1e-3, now - state['prev_ts'])
            vz = (lm8_z - state['prev_z']) / dt
            vang = (angle_deg - state['prev_angle']) / dt

            # Try to start a press
            if self._try_start_press(state, is_pointing, now, baseline_z, lm8_z,
                                    dz_press, vz, baseline_ang, angle_deg,
                                    dang_press, vang, pos_x, pos_y):
                pass  # Press started

            # Check for tap completion
            double_tap = self._check_tap_release(state, now, lm8_z, vz, angle_deg,
                                                 vang, pos_x, pos_y, draw, img_out)

            # Update state
            state['prev_z'] = lm8_z
            state['prev_ts'] = now
            state['prev_angle'] = angle_deg

            return double_tap

        except Exception as e:
            logger.debug(f"Error in tap detection: {e}")
            return False

    def _compute_finger_flexion_angle(self, hand_landmarks, width, height):
        """Compute the flexion angle of the index finger distal joint."""
        def L(i):
            lm = hand_landmarks.landmark[i]
            return np.array([lm.x * width, lm.y * height], dtype=float)

        pip = L(6)  # Index PIP
        dip = L(7)  # Index DIP
        tip = L(8)  # Index tip

        v1 = dip - pip
        v2 = tip - dip
        n1 = np.linalg.norm(v1) + 1e-6
        n2 = np.linalg.norm(v2) + 1e-6
        cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(cosang)))

        return angle_deg

    def _get_tap_state(self, hand_key, lm8_z, angle_deg, pos_x, pos_y, now):
        """Get or initialize tap state for a hand."""
        if hand_key not in self._tap_state:
            self._tap_state[hand_key] = {
                'pressing': False,
                'prev_z': lm8_z,
                'prev_ts': now,
                'press_start': 0.0,
                'min_z': lm8_z,
                'last_tap': 0.0,
                'z_history': deque(maxlen=self.Z_HISTORY_LEN),
                'xy_history': deque(maxlen=self.XY_HISTORY_LEN),
                'press_start_xy': (pos_x, pos_y),
                'cooldown_until': 0.0,
                'start_baseline': lm8_z,
                'peak_depth': 0.0,
                'ang_history': deque(maxlen=self.ANG_HISTORY_LEN),
                'prev_angle': angle_deg,
                'start_baseline_angle': angle_deg,
                'max_angle': angle_deg,
                'peak_angle_depth': 0.0,
                'press_mode': None
            }
        return self._tap_state[hand_key]

    def _calculate_z_baseline(self, state):
        """Calculate Z-depth baseline and noise-adaptive threshold."""
        z_hist = list(state['z_history'])
        baseline_z = np.median(z_hist) if len(z_hist) >= 3 else state['prev_z']
        dz_abs = np.abs(np.diff(z_hist)) if len(z_hist) >= 2 else np.array([0.0])
        noise_z = float(np.median(dz_abs)) if dz_abs.size > 0 else 0.0
        dz_press = max(self.TAP_BASE_DELTA, self.TAP_NOISE_MULT * noise_z)
        return baseline_z, noise_z, dz_press

    def _calculate_angle_baseline(self, state):
        """Calculate angle baseline and noise-adaptive threshold."""
        ang_hist = list(state['ang_history'])
        baseline_ang = np.median(ang_hist) if len(ang_hist) >= 3 else state['prev_angle']
        dang_abs = np.abs(np.diff(ang_hist)) if len(ang_hist) >= 2 else np.array([0.0])
        noise_ang = float(np.median(dang_abs)) if dang_abs.size > 0 else 0.0
        dang_press = max(self.ANG_BASE_DELTA, self.ANG_NOISE_MULT * noise_ang)
        return baseline_ang, noise_ang, dang_press

    def _try_start_press(self, state, is_pointing, now, baseline_z, lm8_z,
                        dz_press, vz, baseline_ang, angle_deg, dang_press, vang,
                        pos_x, pos_y):
        """Try to start a tap press based on Z or angle triggers."""
        if (not state['pressing']) and is_pointing and (now >= state.get('cooldown_until', 0.0)):
            z_press = (baseline_z - lm8_z > dz_press) and (vz <= -self.TAP_MIN_VEL)
            ang_press = (angle_deg - baseline_ang > dang_press) and (vang >= self.ANG_MIN_VEL)

            if z_press or ang_press:
                state['pressing'] = True
                state['press_start'] = now
                state['press_start_xy'] = (pos_x, pos_y)
                state['min_z'] = lm8_z
                state['start_baseline'] = baseline_z
                state['peak_depth'] = max(0.0, baseline_z - lm8_z)
                state['start_baseline_angle'] = baseline_ang
                state['max_angle'] = angle_deg
                state['peak_angle_depth'] = max(0.0, angle_deg - baseline_ang)

                if ang_press and not z_press:
                    state['press_mode'] = 'angle'
                elif z_press and not ang_press:
                    state['press_mode'] = 'z'
                else:
                    state['press_mode'] = 'either'

                return True
        return False

    def _check_tap_release(self, state, now, lm8_z, vz, angle_deg, vang,
                          pos_x, pos_y, draw, img_out):
        """Check if a press should be released and if it forms a (double) tap."""
        if not state['pressing']:
            return False

        # Update peak values
        if lm8_z < state['min_z']:
            state['min_z'] = lm8_z
        state['peak_depth'] = max(state['peak_depth'],
                                 state['start_baseline'] - state['min_z'])

        if angle_deg > state['max_angle']:
            state['max_angle'] = angle_deg
        state['peak_angle_depth'] = max(state['peak_angle_depth'],
                                       state['max_angle'] - state['start_baseline_angle'])

        # Check release conditions
        depth_z = max(0.0, state['peak_depth'])
        back_z = lm8_z - state['min_z']
        enough_back_z = ((depth_z >= self.TAP_MIN_PRESS_DEPTH) and
                        (back_z >= self.TAP_MAX_RELEASE_BACK * depth_z))
        velocity_release_z = ((vz >= self.TAP_RELEASE_VEL) and
                             ((now - state['press_start']) >= self.TAP_MIN_DURATION))

        depth_ang = max(0.0, state['peak_angle_depth'])
        back_ang = state['max_angle'] - angle_deg
        enough_back_ang = ((depth_ang >= self.ANG_MIN_PRESS_DEPTH) and
                          (back_ang >= self.ANG_RELEASE_BACK * depth_ang))
        velocity_release_ang = ((vang <= self.ANG_RELEASE_VEL) and
                               ((now - state['press_start']) >= self.TAP_MIN_DURATION))

        too_long = (now - state['press_start'] > self.TAP_MAX_DURATION)

        if (enough_back_z or velocity_release_z or enough_back_ang or
            velocity_release_ang or too_long):

            press_duration = now - state['press_start']
            sx, sy = state['press_start_xy']
            xy_drift = float(np.hypot(pos_x - sx, pos_y - sy))

            # Validate tap
            valid_tap = ((press_duration >= self.TAP_MIN_DURATION) and
                        (xy_drift <= self.TAP_MAX_XY_DRIFT) and
                        ((depth_z >= self.TAP_MIN_PRESS_DEPTH) or
                         (depth_ang >= self.ANG_MIN_PRESS_DEPTH)))

            if valid_tap:
                logger.info(f"Tap detected: duration={press_duration:.3f}s, "
                          f"depth={depth_z:.4f}, angleDepth={depth_ang:.1f}, "
                          f"drift={xy_drift:.1f}")

                last_tap = state.get('last_tap', 0.0)
                gap = now - last_tap if last_tap > 0.0 else 1e9

                # Check for double tap
                if ((last_tap > 0.0) and
                    (self.TAP_MIN_INTERVAL <= gap <= self.TAP_MAX_INTERVAL) and
                    (now >= state.get('cooldown_until', 0.0))):

                    logger.info(f"Double tap detected! Interval={gap:.3f}s")

                    if draw and img_out is not None:
                        cv.putText(img_out, "DOUBLE TAP",
                                 (int(pos_x), int(pos_y) - 10),
                                 cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    state['pressing'] = False
                    state['last_tap'] = 0.0
                    state['cooldown_until'] = now + self.TAP_COOLDOWN
                    state['z_history'].clear()
                    state['xy_history'].clear()
                    state['ang_history'].clear()

                    return True
                else:
                    state['last_tap'] = now
                    state['pressing'] = False
            else:
                state['pressing'] = False

        return False

    def _draw_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on the image."""
        self.mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )

    def _normalize_index_position(self, index_pos):
        """Normalize index position to 1D array of length 3."""
        if index_pos is None:
            return None

        arr = np.asarray(index_pos)
        if arr.size == 0:
            return None
        elif arr.size >= 3:
            if arr.size % 3 == 0 and arr.size > 3:
                return arr.reshape(-1, 3)[-1].astype(float)
            else:
                return arr.flatten()[:3].astype(float)
        else:
            return None

    @staticmethod
    def ratio(coors):
        """
        Calculate finger extension ratio.

        Ratio is 1 if points are collinear (extended), lower otherwise (curled).

        Args:
            coors (numpy.ndarray): 4x3 array of finger joint coordinates

        Returns:
            float: Extension ratio (0-1)
        """
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])
        return d / (a + b + c + 1e-6)


class PoseDetectorMPEnhanced(PoseDetectorMP):
    """
    Enhanced MediaPipe-based hand pose detector that adds higher-precision
    tap/double-tap detection signals on top of the existing Z-depth and distal
    flexion-angle logic. Does not modify base class methods.
    """

    def __init__(self, model):
        super().__init__(model)
        # Extra per-hand state
        self._tap_state_plus = {}  # keyed by hand ('Left'/'Right')

    # ---------- Geometry helpers ----------

    @staticmethod
    def _Lm(hand_landmarks, i, w=1.0, h=1.0):
        lm = hand_landmarks.landmark[i]
        return np.array([lm.x * w, lm.y * h, float(lm.z)], dtype=float)

    def _palm_plane(self, hand_landmarks, w, h):
        """Palm plane (p0, n) from wrist(0), index MCP(5), pinky MCP(17)."""
        p0 = self._Lm(hand_landmarks, 0, w, h)
        p1 = self._Lm(hand_landmarks, 5, w, h)
        p2 = self._Lm(hand_landmarks, 17, w, h)
        v1 = p1 - p0
        v2 = p2 - p0
        n = np.cross(v1, v2)
        n_norm = np.linalg.norm(n) + 1e-9
        n /= n_norm
        return p0, n

    def _plane_signed_distance_tip(self, hand_landmarks, w, h):
        p0, n = self._palm_plane(hand_landmarks, w, h)
        tip = self._Lm(hand_landmarks, 8, w, h)
        return float(np.dot(n, tip - p0)), n

    def _relative_tip_depth(self, hand_landmarks):
        """Tip Z relative to palm center Z (wrist + MCPs)."""
        idxs = [0, 5, 9, 13, 17]
        zs = [float(hand_landmarks.landmark[i].z) for i in idxs]
        palm_z = float(np.median(zs))
        tip_z = float(hand_landmarks.landmark[8].z)
        return tip_z - palm_z  # relative (negative = closer to camera in MP coords)

    def _index_dir(self, hand_landmarks, w, h):
        """Unit direction from index MCP(5) to tip(8)."""
        mcp = self._Lm(hand_landmarks, 5, w, h)
        tip = self._Lm(hand_landmarks, 8, w, h)
        v = tip - mcp
        n = np.linalg.norm(v) + 1e-9
        return v / n

    # ---------- Smoothing / state ----------

    def _get_plus_state(self, hand_key, now, pos_xy, zrel, ang, plane_d, palm_n):
        st = self._tap_state_plus.get(hand_key)
        if st is None:
            st = {
                'prev_ts': now,
                'prev_xy': np.array(pos_xy, dtype=float),
                'prev_zrel': zrel,
                'prev_ang': ang,
                'prev_plane': plane_d,
                'ema_zrel': zrel,
                'ema_ang': ang,
                'ema_plane': plane_d,
                'xy_hist': deque(maxlen=10),
                'palm_norm_hist': deque(maxlen=10),
                'plane_hist': deque(maxlen=40),
                'zrel_hist': deque(maxlen=40),
                'score_hist': deque(maxlen=10),
                'pressing': False,
                'press_start': 0.0,
                'press_start_xy': np.array(pos_xy, dtype=float),
                'peak_zrel_depth': 0.0,
                'peak_plane_depth': 0.0,
                'peak_ang_depth': 0.0,
                'min_zrel': zrel,
                'min_plane': plane_d,
                'max_ang': ang,
                'last_tap': 0.0,
                'cooldown_until': 0.0
            }
            self._tap_state_plus[hand_key] = st
        return st

    def _ema(self, prev, x, alpha):
        return alpha * x + (1.0 - alpha) * prev

    # ---------- Gating ----------

    def _is_motion_stable(self, st, now, pos_xy, palm_n):
        # XY velocity
        dt = max(1e-3, now - st['prev_ts'])
        vxy = (np.array(pos_xy, dtype=float) - st['prev_xy']) / dt
        # Palm rotation rate
        if len(st['palm_norm_hist']) >= 1:
            prev_n = st['palm_norm_hist'][-1]
            dot = float(np.clip(np.dot(prev_n, palm_n), -1.0, 1.0))
            rot = float(np.arccos(dot)) / dt
        else:
            rot = 0.0
        # Keep histories
        st['xy_hist'].append(vxy)
        st['palm_norm_hist'].append(palm_n)
        # Stability check (median for robustness)
        if len(st['xy_hist']) >= 3:
            v_med = np.median(np.linalg.norm(np.vstack(st['xy_hist']), axis=1))
        else:
            v_med = np.linalg.norm(vxy)
        return (v_med <= self.STABLE_XY_VEL_MAX) and (rot <= self.STABLE_ROT_MAX)

    def _confidence_ok(self, score, st):
        # Landmark jitter from recent XY history
        if len(st['xy_hist']) >= 4:
            xs = [v[0] for v in st['xy_hist']]
            ys = [v[1] for v in st['xy_hist']]
            jitter = float(np.median(np.abs(xs))) + float(np.median(np.abs(ys)))
        else:
            jitter = 0.0
        return (score is None or score >= self.MIN_HAND_SCORE) and (jitter <= self.JITTER_MAX_PX)

    # ---------- Thresholding / baselines ----------

    def _adaptive_baseline(self, hist, fallback):
        if len(hist) >= 5:
            return float(np.median(hist))
        return fallback

    def _noise_level(self, hist):
        if len(hist) >= 2:
            diffs = np.abs(np.diff(np.array(hist)))
            return float(np.median(diffs))
        return 0.0

    # ---------- Tiny classifier ----------

    def _tiny_cls_prob(self, features):
        z = float(np.dot(self.CLS_WEIGHTS, features) + self.CLS_BIAS)
        return 1.0 / (1.0 + np.exp(-z))

    # ---------- Enhanced tap detection ----------

    def detect(self, image, H, _, processing_scale=0.5, draw=False):
        # Use cached base outputs when used in combination wrapper to avoid double-processing
        if getattr(self, "_skip_super", False) and hasattr(self, "_base_cache"):
            base_index_pos, base_status, base_img = self._base_cache
        else:
            base_index_pos, base_status, base_img = super().detect(image, H, _, processing_scale, draw)

        # Now run the enhanced pipeline and fuse it with base outputs.
        # Downscale image for faster processing
        if processing_scale < 1.0:
            small = cv.resize(image, (0, 0), fx=processing_scale, fy=processing_scale,
                              interpolation=cv.INTER_LINEAR)
        else:
            small = image
        small_rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        results = self.hands.process(small_rgb)

        img_out = image.copy() if draw else None
        index_pos = None
        movement_status = None

        if results.multi_hand_landmarks:
            orig_h, orig_w = image.shape[0], image.shape[1]
            for h_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                handedness_msg = MessageToDict(results.multi_handedness[h_idx])['classification'][0]
                handedness = handedness_msg.get('label', 'Unknown')
                score = handedness_msg.get('score', None)
                hand_key = handedness

                # Pointing gate (stronger)
                is_pointing_base = self._detect_pointing_gesture(hand_landmarks, orig_w, orig_h)
                is_pointing_strong = self._strong_pointing_gate(hand_landmarks, orig_w, orig_h)
                is_pointing = is_pointing_base and is_pointing_strong

                if draw:
                    self._draw_hand_landmarks(img_out, hand_landmarks)

                # Index position
                pos_x, pos_y, position = self._get_finger_position(hand_landmarks, 8, orig_w, orig_h, H)
                if index_pos is None:
                    index_pos = np.array([position[0] / position[2], position[1] / position[2], 0.0], dtype=float)

                # Extra signals
                angle_deg = self._compute_finger_flexion_angle(hand_landmarks, orig_w, orig_h)
                plane_d, palm_n = self._plane_signed_distance_tip(hand_landmarks, orig_w, orig_h)
                zrel = self._relative_tip_depth(hand_landmarks)
                idx_dir = self._index_dir(hand_landmarks, orig_w, orig_h)

                # Per-hand enhanced state
                now = time.time()
                st = self._get_plus_state(hand_key, now, (pos_x, pos_y), zrel, angle_deg, plane_d, palm_n)

                # Temporal smoothing (EMA)
                st['ema_zrel'] = self._ema(st['ema_zrel'], zrel, self.EWMA_ALPHA)
                st['ema_ang'] = self._ema(st['ema_ang'], angle_deg, self.EWMA_ALPHA)
                st['ema_plane'] = self._ema(st['ema_plane'], plane_d, self.EWMA_ALPHA)

                # Histories for baselines
                st['zrel_hist'].append(st['ema_zrel'])
                st['plane_hist'].append(st['ema_plane'])

                # Motion stability and confidence gates
                stable = self._is_motion_stable(st, now, (pos_x, pos_y), palm_n)
                conf_ok = self._confidence_ok(score, st)

                # Derivatives
                dt = max(1e-3, now - st['prev_ts'])
                vzrel = (st['ema_zrel'] - st['prev_zrel']) / dt
                vang = (st['ema_ang'] - st['prev_ang']) / dt
                vplane = (st['ema_plane'] - st['prev_plane']) / dt

                # Ray-projection velocity (inward pulse along -index-dir)
                tip_prev = np.array([st['prev_xy'][0], st['prev_xy'][1], st['prev_zrel']], dtype=float)
                tip_curr = np.array([pos_x, pos_y, st['ema_zrel']], dtype=float)
                v_tip = (tip_curr - tip_prev) / dt
                ray_in_v = float(np.dot(v_tip / (np.linalg.norm(v_tip) + 1e-9), -idx_dir))

                # Baselines and noise-adaptive thresholds
                base_zrel = self._adaptive_baseline(st['zrel_hist'], st['prev_zrel'])
                noise_zrel = self._noise_level(st['zrel_hist'])
                dzrel_press = max(self.ZREL_BASE_DELTA, self.ZREL_NOISE_MULT * noise_zrel)

                base_plane = self._adaptive_baseline(st['plane_hist'], st['prev_plane'])
                noise_plane = self._noise_level(st['plane_hist'])
                dplane_press = max(self.PLANE_BASE_DELTA, self.PLANE_NOISE_MULT * noise_plane)

                # Triggers (note: MP z becomes more negative toward camera)
                zrel_press = (base_zrel - st['ema_zrel'] > dzrel_press) and (vzrel <= -self.TAP_MIN_VEL)
                ang_press = (st['ema_ang'] - st['prev_ang'] > 0) and (vang >= self.ANG_MIN_VEL)
                plane_press = (base_plane - st['ema_plane'] > dplane_press) and (vplane <= -self.TAP_MIN_VEL)
                ray_press = (ray_in_v >= self.RAY_MIN_IN_VEL)

                # Late fusion: require >=2 triggers and gates
                start_ok = is_pointing and stable and conf_ok and (sum([zrel_press, ang_press, plane_press, ray_press]) >= 2)

                # Start press
                if (not st['pressing']) and start_ok and now >= st['cooldown_until']:
                    st['pressing'] = True
                    st['press_start'] = now
                    st['press_start_xy'] = np.array([pos_x, pos_y], dtype=float)
                    st['min_zrel'] = st['ema_zrel']
                    st['min_plane'] = st['ema_plane']
                    st['max_ang'] = st['ema_ang']
                    st['peak_zrel_depth'] = max(0.0, base_zrel - st['ema_zrel'])
                    st['peak_plane_depth'] = max(0.0, base_plane - st['ema_plane'])
                    st['peak_ang_depth'] = 0.0

                # Update peaks while pressing
                if st['pressing']:
                    st['min_zrel'] = min(st['min_zrel'], st['ema_zrel'])
                    st['min_plane'] = min(st['min_plane'], st['ema_plane'])
                    st['max_ang'] = max(st['max_ang'], st['ema_ang'])
                    st['peak_zrel_depth'] = max(st['peak_zrel_depth'], base_zrel - st['min_zrel'])
                    st['peak_plane_depth'] = max(st['peak_plane_depth'], base_plane - st['min_plane'])
                    st['peak_ang_depth'] = max(st['peak_ang_depth'], st['max_ang'] - st['prev_ang'])

                    # Release conditions (consensus)
                    back_zrel = st['ema_zrel'] - st['min_zrel']
                    enough_back_zrel = (st['peak_zrel_depth'] >= self.ZREL_MIN_PRESS_DEPTH) and (back_zrel >= self.TAP_MAX_RELEASE_BACK * st['peak_zrel_depth'])
                    vrel_release = (vzrel >= self.TAP_RELEASE_VEL) and ((now - st['press_start']) >= self.TAP_MIN_DURATION)

                    back_plane = st['ema_plane'] - st['min_plane']
                    enough_back_plane = (st['peak_plane_depth'] >= self.PLANE_MIN_PRESS_DEPTH) and (back_plane >= self.PLANE_RELEASE_BACK * st['peak_plane_depth'])
                    vplane_release = (vplane >= self.TAP_RELEASE_VEL) and ((now - st['press_start']) >= self.TAP_MIN_DURATION)

                    back_ang = st['max_ang'] - st['ema_ang']
                    enough_back_ang = (st['peak_ang_depth'] >= self.ANG_MIN_PRESS_DEPTH) and (back_ang >= self.ANG_RELEASE_BACK * st['peak_ang_depth'])
                    vang_release = (vang <= self.ANG_RELEASE_VEL) and ((now - st['press_start']) >= self.TAP_MIN_DURATION)

                    too_long = (now - st['press_start'] > self.TAP_MAX_DURATION)
                    release_votes = sum([enough_back_zrel or vrel_release,
                                         enough_back_plane or vplane_release,
                                         enough_back_ang or vang_release])

                    if release_votes >= 2 or too_long:
                        # Validate tap
                        duration = now - st['press_start']
                        drift = float(np.hypot(pos_x - st['press_start_xy'][0], pos_y - st['press_start_xy'][1]))
                        valid_rule = (duration >= self.TAP_MIN_DURATION) and (drift <= self.TAP_MAX_XY_DRIFT) and (
                            (st['peak_zrel_depth'] >= self.ZREL_MIN_PRESS_DEPTH) or
                            (st['peak_plane_depth'] >= self.PLANE_MIN_PRESS_DEPTH) or
                            (st['peak_ang_depth'] >= self.ANG_MIN_PRESS_DEPTH)
                        )

                        # Tiny classifier features
                        feats = np.array([st['peak_zrel_depth'], st['peak_plane_depth'], st['peak_ang_depth'],
                                          drift, abs(vzrel), abs(vplane), duration], dtype=float)
                        prob = self._tiny_cls_prob(feats)
                        valid = valid_rule and (prob >= self.CLS_MIN_PROB)

                        if valid:
                            last_tap = st['last_tap']
                            gap = now - last_tap if last_tap > 0.0 else 1e9
                            if (last_tap > 0.0) and (self.TAP_MIN_INTERVAL <= gap <= self.TAP_MAX_INTERVAL) and (now >= st['cooldown_until']):
                                if draw and img_out is not None:
                                    cv.putText(img_out, "DOUBLE TAP", (int(pos_x), int(pos_y) - 10),
                                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                st['last_tap'] = 0.0
                                st['cooldown_until'] = now + self.TAP_COOLDOWN
                                st['pressing'] = False
                                movement_status = 'double_tap'
                                break
                            else:
                                st['last_tap'] = now
                                st['pressing'] = False
                        else:
                            st['pressing'] = False

                # Update state for next frame
                st['prev_ts'] = now
                st['prev_xy'] = np.array([pos_x, pos_y], dtype=float)
                st['prev_zrel'] = st['ema_zrel']
                st['prev_ang'] = st['ema_ang']
                st['prev_plane'] = st['ema_plane']

                if movement_status != 'double_tap':
                    movement_status = 'pointing' if is_pointing else 'moving'

        # Normalize index position to 3-vector
        normalized = self._normalize_index_position(index_pos)

        # Fuse enhanced outputs with base outputs
        enhanced_status = movement_status
        final_pos = normalized if normalized is not None else base_index_pos
        if (base_status == 'double_tap') or (enhanced_status == 'double_tap'):
            final_status = 'double_tap'
        elif (base_status == 'pointing') or (enhanced_status == 'pointing'):
            final_status = 'pointing'
        else:
            final_status = base_status or enhanced_status
        final_img = img_out if (draw and img_out is not None) else base_img

        return final_pos, final_status, final_img

    # ---------- Stronger pointing gate ----------
    def _strong_pointing_gate(self, hand_landmarks, w, h):
        def L(i):
            lm = hand_landmarks.landmark[i]
            return np.array([lm.x, lm.y, lm.z], dtype=float)
        idx = np.array([L(i) for i in [5, 6, 7, 8]])
        mid = np.array([L(i) for i in [9, 10, 11, 12]])
        rng = np.array([L(i) for i in [13, 14, 15, 16]])
        ltl = np.array([L(i) for i in [17, 18, 19, 20]])
        r_idx = self.ratio(idx)
        r_mid = self.ratio(mid)
        r_rng = self.ratio(rng)
        r_ltl = self.ratio(ltl)
        others_max = max(r_mid, r_rng, r_ltl)
        return (r_idx >= self.INDEX_STRONG_MIN) and (others_max <= self.OTHERS_STRONG_MAX)

# --- Combination wrapper to run base + enhanced and fuse outputs ---
class CombinedPoseDetector:
    def __init__(self, model):
        self.base = PoseDetectorMP(model)
        self.enh = PoseDetectorMPEnhanced(model)
        # Tell enhanced to use cached base outputs instead of calling super()
        self.enh._skip_super = True
        # Propagate image (for downstream tools that may inspect it)
        self.image_map_color = self.base.image_map_color

    def detect(self, image, H, _, processing_scale=0.5, draw=False):
        base_index_pos, base_status, base_img = self.base.detect(image, H, _, processing_scale, draw)
        # Cache base outputs for the enhanced detector
        self.enh._base_cache = (base_index_pos, base_status, base_img)
        return self.enh.detect(image, H, _, processing_scale, draw)
