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

