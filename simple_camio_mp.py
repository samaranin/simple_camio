import numpy as np
import cv2 as cv
import mediapipe as mp
from scipy import stats
from google.protobuf.json_format import MessageToDict
import time
from collections import deque

class PoseDetectorMP:
    def __init__(self, model):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=2
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)

        # --- Double-tap detection params / state ---
        # keyed by hand index in current processing (0..n)
        # each entry contains: pressing, prev_z, prev_ts, press_start, min_z, last_tap, z_history, xy_history, press_start_xy, cooldown_until
        self._tap_state = {}
        # Baseline thresholds (will be adapted per-frame using noise)
        self.TAP_BASE_DELTA = 0.025     # base z delta vs baseline to start a press
        self.TAP_NOISE_MULT = 3.0       # multiplier on median |dz| to raise threshold in noise
        # Relaxed velocity and durations for more tolerant taps
        self.TAP_MIN_VEL = 0.2          # min negative z velocity to start a press
        self.TAP_RELEASE_VEL = 0.15     # min positive z velocity to consider release
        self.TAP_MIN_DURATION = 0.05    # seconds
        self.TAP_MAX_DURATION = 0.50    # seconds (was 0.45)
        self.TAP_MIN_INTERVAL = 0.05    # seconds (was 0.12)
        self.TAP_MAX_INTERVAL = 1.00    # seconds (was 0.60)
        self.TAP_MIN_PRESS_DEPTH = 0.015  # minimal press depth needed to consider a tap
        self.TAP_MAX_XY_DRIFT = 180.0   # allow more XY drift during a tap
        self.TAP_MAX_RELEASE_BACK = 0.45 # fraction of press depth required to consider release complete
        self.Z_HISTORY_LEN = 7
        self.XY_HISTORY_LEN = 7
        self.TAP_COOLDOWN = 0.7         # seconds to ignore further double-tap detections for this hand

        # NEW: angle-based distal flexion thresholds (degrees)
        self.ANG_HISTORY_LEN = 7
        self.ANG_BASE_DELTA = 12.0         # min angle rise above baseline to start a press
        self.ANG_NOISE_MULT = 3.0          # noise-adaptive margin
        self.ANG_MIN_VEL = 120.0           # deg/s minimum rising angular velocity
        self.ANG_RELEASE_VEL = -120.0      # deg/s negative velocity (falling) for release
        self.ANG_MIN_PRESS_DEPTH = 10.0    # degrees (min peak flexion over baseline)
        self.ANG_RELEASE_BACK = 0.5        # fraction of peak angle to return for release

    def detect(self, image, H, _, processing_scale=0.5, draw=False):
        """
        Process a downscaled copy of `image` with MediaPipe to speed up processing.
        If `draw` is False, no overlay is drawn and no full-size copy is made.
        """
        if processing_scale < 1.0:
            small = cv.resize(image, (0, 0), fx=processing_scale, fy=processing_scale, interpolation=cv.INTER_LINEAR)
        else:
            small = image

        small_rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        results = self.hands.process(small_rgb)

        img_out = image.copy() if draw else None
        index_pos = None
        movement_status = None

        double_tap_emitted = False  # NEW: latch for this frame

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                orig_h, orig_w = image.shape[0], image.shape[1]
                handedness = MessageToDict(results.multi_handedness[h])['classification'][0]['label']
                # Use stable key based on handedness ("Left"/"Right") instead of the unstable loop index h
                hand_key = handedness
                coors = np.zeros((4, 3), dtype=float)
                is_pointing = True

                # helper to get landmark by index (normalized)
                def L(i):
                    lm = hand_landmarks.landmark[i]
                    return np.array([lm.x * orig_w, lm.y * orig_h, lm.z], dtype=float)

                # thumb
                for k, idx in enumerate([1, 2, 3, 4]):
                    coors[k, :] = L(idx)
                ratio_thumb = self.ratio(coors)

                # index
                for k, idx in enumerate([5, 6, 7, 8]):
                    coors[k, :] = L(idx)
                ratio_index = self.ratio(coors)
                a = coors[0, :].copy()
                ab = coors[3, :] - coors[0, :]

                # middle
                for k, idx in enumerate([9, 10, 11, 12]):
                    coors[k, :] = L(idx)
                ratio_middle = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                # ring
                for k, idx in enumerate([13, 14, 15, 16]):
                    coors[k, :] = L(idx)
                ratio_ring = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                # little
                for k, idx in enumerate([17, 18, 19, 20]):
                    coors[k, :] = L(idx)
                ratio_little = self.ratio(coors)
                for i in range(4):
                    ap = coors[i, :] - a
                    if np.dot(ap, ab) / np.dot(ab, ab) > 0.5:
                        is_pointing = False

                overall = ratio_index - ((ratio_middle + ratio_ring + ratio_little) / 3)
                is_pointing = is_pointing or overall > 0.1

                if draw:
                    self.mp_drawing.draw_landmarks(
                        img_out,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

                # fingertip 8 in normalized coords -> convert to original pixels
                pos_x = hand_landmarks.landmark[8].x * orig_w
                pos_y = hand_landmarks.landmark[8].y * orig_h
                # store pixel position transformed by homography (same as before)
                position = np.matmul(H, np.array([pos_x, pos_y, 1]))
                if index_pos is None:
                    index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)

                # Guard: do not overwrite a detected double_tap with pointing/moving
                if movement_status != 'double_tap':
                    if (is_pointing or (ratio_index > 0.7) and (ratio_middle < 0.95) and (ratio_ring < 0.95) and (ratio_little < 0.95)):
                        if movement_status != "pointing":
                            index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)
                            movement_status = "pointing"
                        else:
                            index_pos = np.append(index_pos,
                                                  np.array([position[0] / position[2], position[1] / position[2], 0],
                                                           dtype=float))
                            movement_status = "too_many"
                    elif movement_status != "pointing":
                        movement_status = "moving"

                # --- Combined z- and angle-based tap detection using stable hand key ---
                try:
                    now = time.time()
                    lm8_z = float(hand_landmarks.landmark[8].z)

                    # Compute distal flexion angle (between 6->7 and 7->8)
                    pip = L(6)[:2]   # index PIP
                    dip = L(7)[:2]   # index DIP
                    tip = L(8)[:2]   # index tip
                    v1 = dip - pip
                    v2 = tip - dip
                    n1 = np.linalg.norm(v1) + 1e-6
                    n2 = np.linalg.norm(v2) + 1e-6
                    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
                    angle_deg = float(np.degrees(np.arccos(cosang)))

                    state = self._tap_state.get(hand_key)
                    if state is None:
                        state = {
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
                            # NEW angle fields
                            'ang_history': deque(maxlen=self.ANG_HISTORY_LEN),
                            'prev_angle': angle_deg,
                            'start_baseline_angle': angle_deg,
                            'max_angle': angle_deg,
                            'peak_angle_depth': 0.0,
                            'press_mode': None  # 'z' or 'angle'
                        }
                        self._tap_state[hand_key] = state

                    # Append histories
                    state['z_history'].append(lm8_z)
                    state['xy_history'].append((pos_x, pos_y))
                    state['ang_history'].append(angle_deg)

                    # Baselines and noise estimates
                    z_hist = list(state['z_history'])
                    baseline_z = np.median(z_hist) if len(z_hist) >= 3 else state['prev_z']
                    dz_abs = np.abs(np.diff(z_hist)) if len(z_hist) >= 2 else np.array([0.0])
                    noise_z = float(np.median(dz_abs)) if dz_abs.size > 0 else 0.0
                    dz_press = max(self.TAP_BASE_DELTA, self.TAP_NOISE_MULT * noise_z)

                    ang_hist = list(state['ang_history'])
                    baseline_ang = np.median(ang_hist) if len(ang_hist) >= 3 else state['prev_angle']
                    dang_abs = np.abs(np.diff(ang_hist)) if len(ang_hist) >= 2 else np.array([0.0])
                    noise_ang = float(np.median(dang_abs)) if dang_abs.size > 0 else 0.0
                    dang_press = max(self.ANG_BASE_DELTA, self.ANG_NOISE_MULT * noise_ang)

                    # Velocities
                    dt = max(1e-3, now - state['prev_ts'])
                    vz = (lm8_z - state['prev_z']) / dt
                    vang = (angle_deg - state['prev_angle']) / dt  # deg/s

                    # Start press if z OR angle triggers (while pointing and not in cooldown)
                    if (not state['pressing']) and is_pointing and (now >= state.get('cooldown_until', 0.0)):
                        z_press = (baseline_z - lm8_z > dz_press) and (vz <= -self.TAP_MIN_VEL)
                        ang_press = (angle_deg - baseline_ang > dang_press) and (vang >= self.ANG_MIN_VEL)
                        if z_press or ang_press:
                            state['pressing'] = True
                            state['press_start'] = now
                            state['press_start_xy'] = (pos_x, pos_y)
                            # Z mode init
                            state['min_z'] = lm8_z
                            state['start_baseline'] = baseline_z
                            state['peak_depth'] = max(0.0, baseline_z - lm8_z)
                            # ANG mode init
                            state['start_baseline_angle'] = baseline_ang
                            state['max_angle'] = angle_deg
                            state['peak_angle_depth'] = max(0.0, angle_deg - baseline_ang)
                            state['press_mode'] = 'angle' if ang_press and (not z_press) else ('z' if z_press and (not ang_press) else 'either')

                    # Track press and check release
                    if state['pressing']:
                        # Update z peak
                        if lm8_z < state['min_z']:
                            state['min_z'] = lm8_z
                        state['peak_depth'] = max(state['peak_depth'], state['start_baseline'] - state['min_z'])

                        # Update angle peak
                        if angle_deg > state['max_angle']:
                            state['max_angle'] = angle_deg
                        state['peak_angle_depth'] = max(state['peak_angle_depth'], state['max_angle'] - state['start_baseline_angle'])

                        # Release conditions
                        depth_z = max(0.0, state['peak_depth'])
                        back_z = lm8_z - state['min_z']
                        enough_back_z = (depth_z >= self.TAP_MIN_PRESS_DEPTH) and (back_z >= self.TAP_MAX_RELEASE_BACK * depth_z)
                        velocity_release_z = (vz >= self.TAP_RELEASE_VEL) and ((now - state['press_start']) >= self.TAP_MIN_DURATION)

                        depth_ang = max(0.0, state['peak_angle_depth'])
                        back_ang = state['max_angle'] - angle_deg
                        enough_back_ang = (depth_ang >= self.ANG_MIN_PRESS_DEPTH) and (back_ang >= self.ANG_RELEASE_BACK * depth_ang)
                        velocity_release_ang = (vang <= self.ANG_RELEASE_VEL) and ((now - state['press_start']) >= self.TAP_MIN_DURATION)

                        too_long = (now - state['press_start'] > self.TAP_MAX_DURATION)

                        if enough_back_z or velocity_release_z or enough_back_ang or velocity_release_ang or too_long:
                            press_duration = now - state['press_start']
                            sx, sy = state['press_start_xy']
                            xy_drift = float(np.hypot(pos_x - sx, pos_y - sy))

                            # Accept a tap if either depth passes threshold
                            valid_tap = (
                                (press_duration >= self.TAP_MIN_DURATION) and
                                (xy_drift <= self.TAP_MAX_XY_DRIFT) and
                                ((depth_z >= self.TAP_MIN_PRESS_DEPTH) or (depth_ang >= self.ANG_MIN_PRESS_DEPTH))
                            )

                            if valid_tap:
                                print(f"Tap detected: duration={press_duration:.3f}s, depth={depth_z:.4f}, angleDepth={depth_ang:.1f}, drift={xy_drift:.1f}")
                                last_tap = state.get('last_tap', 0.0)
                                gap = now - last_tap if last_tap > 0.0 else 1e9
                                if (last_tap > 0.0) and (self.TAP_MIN_INTERVAL <= gap <= self.TAP_MAX_INTERVAL) and (now >= state.get('cooldown_until', 0.0)):
                                    movement_status = 'double_tap'
                                    print(f"Double tap detected! Interval={gap:.3f}s")
                                    double_tap_emitted = True
                                    if draw and img_out is not None:
                                        cv.putText(img_out, "DOUBLE TAP", (int(pos_x), int(pos_y) - 10),
                                                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                                    state['pressing'] = False
                                    state['last_tap'] = 0.0
                                    state['cooldown_until'] = now + self.TAP_COOLDOWN
                                    state['z_history'].clear()
                                    state['xy_history'].clear()
                                    state['ang_history'].clear()
                                else:
                                    state['last_tap'] = now
                                    state['pressing'] = False
                            else:
                                state['pressing'] = False

                    # Book-keeping
                    state['prev_z'] = lm8_z
                    state['prev_ts'] = now
                    state['prev_angle'] = angle_deg

                except Exception:
                    pass

                if double_tap_emitted:
                    break

        # Normalize index_pos: always return either None or a 1D numpy array of length 3 [x,y,z].
        if index_pos is None:
            normalized = None
        else:
            arr = np.asarray(index_pos)
            if arr.size == 0:
                normalized = None
            elif arr.size >= 3:
                # If arr is flat concatenation of multiple [x,y,z] triplets, take the last triplet (most recent)
                if arr.size % 3 == 0 and arr.size > 3:
                    normalized = arr.reshape(-1, 3)[-1].astype(float)
                else:
                    # take first 3 elements as fallback
                    normalized = arr.flatten()[:3].astype(float)
            else:
                normalized = None

        return normalized, movement_status, img_out

    def ratio(self, coors):  # ratio is 1 if points are collinear, lower otherwise (minimum is 0)
        d = np.linalg.norm(coors[0, :] - coors[3, :])
        a = np.linalg.norm(coors[0, :] - coors[1, :])
        b = np.linalg.norm(coors[1, :] - coors[2, :])
        c = np.linalg.norm(coors[2, :] - coors[3, :])

        return d / (a + b + c)

class InteractionPolicyMP:
    def __init__(self, model):
        self.model = model
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)
        self.ZONE_FILTER_SIZE = 10
        self.Z_THRESHOLD = 2.0
        self.zone_filter = -1 * np.ones(self.ZONE_FILTER_SIZE, dtype=int)
        self.zone_filter_cnt = 0

    # Sergio: we are currently returning the zone id also when the ring buffer is not full. Is this the desired behavior?
    # the impact is clearly minor, but conceptually I am not convinced that this is the right behavior.
    # Sergio (2): I have a concern about this function, I will discuss it in an email.
    def push_gesture(self, position):
        zone_color = self.get_zone(position, self.image_map_color)
        self.zone_filter[self.zone_filter_cnt] = self.get_dict_idx_from_color(zone_color)
        self.zone_filter_cnt = (self.zone_filter_cnt + 1) % self.ZONE_FILTER_SIZE
        zone = stats.mode(self.zone_filter).mode
        if isinstance(zone, np.ndarray):
            zone = zone[0]
        if np.abs(position[2]) < self.Z_THRESHOLD:
            return zone
        else:
            return -1


class SIFTModelDetectorMP:
    def __init__(self, model):
        self.model = model
        img_object = cv.imread(model["template_image"], cv.IMREAD_GRAYSCALE)

        # store template image for quick validation
        self.img_object = img_object

        # Use more features and multiple detectors for robustness
        self.sift_detector = cv.SIFT_create(nfeatures=2000, contrastThreshold=0.03, edgeThreshold=15)
        self.orb_detector = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=12)

        # Extract SIFT features from template
        keypoints_sift, descriptors_sift = self.sift_detector.detectAndCompute(img_object, mask=None)
        keypoints_sift = list(keypoints_sift)

        # Extract ORB features as backup
        self.keypoints_orb, self.descriptors_orb = self.orb_detector.detectAndCompute(img_object, mask=None)

        # Add corner detection to boost keypoints on edges
        corners = cv.goodFeaturesToTrack(img_object, maxCorners=500, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corner_kps = [cv.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=20) for c in corners]
            keypoints_sift.extend(corner_kps)
            keypoints_sift, descriptors_sift = self.sift_detector.compute(img_object, keypoints_sift)

        self.keypoints_sift = keypoints_sift
        self.descriptors_sift = descriptors_sift

        # store template shape so we can compute object corners
        self.template_shape = img_object.shape[:2]  # (h, w)

        self.requires_homography = True
        self.H = None
        self.MIN_INLIER_COUNT = 10

        # Add tracking quality monitoring
        self.frames_since_last_detection = 0
        self.REDETECT_INTERVAL = 150  # Force validation every 150 frames
        self.last_inlier_count = 0
        self.tracking_quality_history = []
        self.MIN_TRACKING_QUALITY = 8  # Minimum inliers to maintain tracking

        # New: store whether H was updated and the projected rectangle pts in camera coords
        self.homography_updated = False
        self.last_rect_pts = None  # will hold Nx1x2 array same as cv.perspectiveTransform output
        self.debug = False  # gate verbose prints

        print(f"Template features: SIFT={len(self.keypoints_sift)}, ORB={len(self.keypoints_orb)}")

    def detect(self, frame, force_redetect=False):
        """Detect with automatic re-detection when tracking degrades"""
        # Manual re-detection trigger
        if force_redetect:
            print("Manual re-detection triggered")
            self.requires_homography = True
            self.H = None
            self.tracking_quality_history.clear()

        # Automatic triggers
        if not self.requires_homography:
            self.frames_since_last_detection += 1

            # Trigger 1: Periodic validation
            if self.frames_since_last_detection >= self.REDETECT_INTERVAL:
                print(f"Periodic validation after {self.frames_since_last_detection} frames")
                self._validate_tracking(frame)

            # Trigger 2: Quality degradation
            if len(self.tracking_quality_history) >= 3:
                avg_quality = np.mean(self.tracking_quality_history[-3:])
                if avg_quality < self.MIN_TRACKING_QUALITY:
                    print(f"Tracking degraded (avg: {avg_quality:.1f}), re-detecting")
                    self.requires_homography = True
                    self.H = None
                    self.tracking_quality_history.clear()

        if not self.requires_homography:
            return True, self.H, None

        # Try SIFT matching first
        success, H = self._match_sift(frame)
        if success:
            self.frames_since_last_detection = 0
            return True, H, None

        # Fallback to ORB
        success, H = self._match_orb(frame)
        if success:
            self.frames_since_last_detection = 0
            return True, H, None

        # Return last known H if available
        if self.H is not None:
            return True, self.H, None
        else:
            return False, None, None

    def _validate_tracking(self, frame):
        """Validate current homography quality"""
        keypoints_scene, descriptors_scene = self.sift_detector.detectAndCompute(frame, None)
        if descriptors_scene is None or len(keypoints_scene) < 4:
            self.requires_homography = True
            self.H = None
            return

        try:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
            search_params = dict(checks=100)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(self.descriptors_sift, descriptors_scene, k=2)

            good_matches = []
            for match_pair in knn_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.8 * n.distance:
                        good_matches.append(m)

            match_count = len(good_matches)
            self.tracking_quality_history.append(match_count)
            if len(self.tracking_quality_history) > 10:
                self.tracking_quality_history.pop(0)

            if self.debug:
                print(f"Validation: {match_count} matches")

            if match_count < self.MIN_TRACKING_QUALITY:
                print("Validation failed: triggering re-detection")
                self.requires_homography = True
                self.H = None
            else:
                self.frames_since_last_detection = 0
        except Exception as e:
            if self.debug:
                print(f"Validation error: {e}")
            self.requires_homography = True
            self.H = None

    def _match_sift(self, frame):
        """SIFT-based matching with improved parameters"""
        keypoints_scene, descriptors_scene = self.sift_detector.detectAndCompute(frame, None)
        if descriptors_scene is None or len(keypoints_scene) < 4:
            return False, None

        # FLANN matcher with better parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
        search_params = dict(checks=100)
        matcher = cv.FlannBasedMatcher(index_params, search_params)

        knn_matches = matcher.knnMatch(self.descriptors_sift, descriptors_scene, k=2)

        # Lowe's ratio test with relaxed threshold
        RATIO_THRESH = 0.8  # Increased from 0.75 for more matches
        good_matches = []
        for match_pair in knn_matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < RATIO_THRESH * n.distance:
                    good_matches.append(m)

        if self.debug:
            print(f"SIFT good matches: {len(good_matches)}")

        if len(good_matches) < 4:
            return False, None

        return self._compute_homography(good_matches, self.keypoints_sift, keypoints_scene, "SIFT")

    def _match_orb(self, frame):
        """ORB-based matching as fallback"""
        keypoints_scene, descriptors_scene = self.orb_detector.detectAndCompute(frame, None)
        if descriptors_scene is None or len(keypoints_scene) < 4:
            return False, None

        # BFMatcher for binary descriptors
        matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=False)
        matches = matcher.knnMatch(self.descriptors_orb, descriptors_scene, k=2)

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if self.debug:
            print(f"ORB good matches: {len(good_matches)}")

        if len(good_matches) < 4:
            return False, None

        return self._compute_homography(good_matches, self.keypoints_orb, keypoints_scene, "ORB")

    def _compute_homography(self, good_matches, keypoints_obj, keypoints_scene, method_name):
        """Compute homography from matched keypoints"""
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)

        for i in range(len(good_matches)):
            obj[i, 0] = keypoints_obj[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = keypoints_obj[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        # Use MAGSAC++ for more robust homography estimation
        H, mask_out = cv.findHomography(
            scene, obj, cv.USAC_MAGSAC,
            ransacReprojThreshold=5.0,  # Reduced from 8.0 for stricter inliers
            confidence=0.99,
            maxIters=5000
        )

        if H is None:
            return False, None
        total = int(np.sum(mask_out))
        if self.debug:
            print(f'{method_name} inlier count: {total}')

        if total >= self.MIN_INLIER_COUNT:
            # If we accept the homography, store it and compute the rectangle projected into camera frame
            self.H = H
            self.last_inlier_count = total
            self.requires_homography = False
            self.tracking_quality_history.append(total)
            if len(self.tracking_quality_history) > 10:
                self.tracking_quality_history.pop(0)

            # Compute corners of the template in object coords and project to camera frame
            h_t, w_t = self.template_shape  # template height, width
            obj_corners = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], dtype=np.float32).reshape(-1, 1, 2)
            try:
                H_inv = np.linalg.inv(self.H)
                pts = cv.perspectiveTransform(obj_corners, H_inv)  # camera-frame points
                self.last_rect_pts = pts  # store for drawing
            except Exception as e:
                print(f"Could not compute projected rectangle: {e}")
                self.last_rect_pts = None

            # flag update so main loop can highlight newly rebuilt rectangle
            self.homography_updated = True

            if self.debug:
                print(f"Homography locked using {method_name}")
            return True, H

        return False, None

    def quick_validate_position(self, scene_gray, min_matches=6, position_threshold=50):
        """
        Quick validation by attempting fresh feature matching on full frame
        and checking if the resulting rectangle position is similar to stored position.
        Returns True if homography still valid, False if template moved.
        """
        if self.last_rect_pts is None or self.H is None:
            return False

        # Run a lightweight SIFT match on the full scene
        keypoints_scene, descriptors_scene = self.sift_detector.detectAndCompute(scene_gray, None)

        if descriptors_scene is None or len(keypoints_scene) < 4:
            if self.debug:
                print("quick_validate_position: not enough scene keypoints")
            self.requires_homography = True
            self.last_rect_pts = None
            return False

        # Match template to scene
        try:
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)
            search_params = dict(checks=50)
            matcher = cv.FlannBasedMatcher(index_params, search_params)
            knn_matches = matcher.knnMatch(self.descriptors_sift, descriptors_scene, k=2)
        except Exception as e:
            print(f"quick_validate_position: matcher failed: {e}")
            return False

        # Ratio test
        RATIO_THRESH = 0.8
        good_matches = []
        for pair in knn_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < RATIO_THRESH * n.distance:
                    good_matches.append(m)

        match_count = len(good_matches)
        if self.debug:
            print(f"quick_validate_position: found {match_count} matches (threshold {min_matches})")

        if match_count < min_matches:
            if self.debug:
                print("quick_validate_position: insufficient matches -> triggering re-detect")
            self.requires_homography = True
            self.last_rect_pts = None
            return False

        # Compute fresh homography from these matches
        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)

        for i in range(len(good_matches)):
            obj[i, 0] = self.keypoints_sift[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.keypoints_sift[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints_scene[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints_scene[good_matches[i].trainIdx].pt[1]

        try:
            H_test, mask_out = cv.findHomography(scene, obj, cv.RANSAC, ransacReprojThreshold=8.0)
            if H_test is None:
                if self.debug:
                    print("quick_validate_position: could not compute test homography")
                self.requires_homography = True
                self.last_rect_pts = None
                return False
        except Exception as e:
            if self.debug:
                print(f"quick_validate_position: homography computation failed: {e}")
            self.requires_homography = True
            self.last_rect_pts = None
            return False

        # Project template corners using the fresh homography
        h_t, w_t = self.template_shape
        obj_corners = np.array([[0, 0], [w_t, 0], [w_t, h_t], [0, h_t]], dtype=np.float32).reshape(-1, 1, 2)
        try:
            H_test_inv = np.linalg.inv(H_test)
            new_pts = cv.perspectiveTransform(obj_corners, H_test_inv)
        except Exception as e:
            print(f"quick_validate_position: could not project corners: {e}")
            self.requires_homography = True
            self.last_rect_pts = None
            return False

        # Compare new projected corners to stored corners
        old_pts = self.last_rect_pts.reshape(-1, 2)
        new_pts_flat = new_pts.reshape(-1, 2)

        # Compute average distance between corresponding corners
        distances = np.linalg.norm(old_pts - new_pts_flat, axis=1)
        avg_distance = np.mean(distances)
        max_distance = np.max(distances)

        if self.debug:
            print(f"quick_validate_position: corner movement - avg: {avg_distance:.1f}px, max: {max_distance:.1f}px (threshold: {position_threshold}px)")

        if max_distance > position_threshold:
            if self.debug:
                print("quick_validate_position: template position changed -> triggering re-detect")
            self.requires_homography = True
            self.last_rect_pts = None
            return False
        if self.debug:
            print("quick_validate_position: validation PASSED")
        return True

    def get_tracking_status(self):
        """Return tracking status for display"""
        if self.requires_homography:
            return "SEARCHING FOR MAP"
        else:
            avg_quality = np.mean(self.tracking_quality_history) if self.tracking_quality_history else self.last_inlier_count
            return f"TRACKING (Q:{avg_quality:.0f} Age:{self.frames_since_last_detection})"

