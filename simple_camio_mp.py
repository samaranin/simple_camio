import numpy as np
import cv2 as cv
import mediapipe as mp
from scipy import stats
from google.protobuf.json_format import MessageToDict


class PoseDetectorMP:
    def __init__(self, model):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.image_map_color = cv.imread(model['filename'], cv.IMREAD_COLOR)

    def detect(self, image, H, _, processing_scale=0.5):
        """
        Process a downscaled copy of `image` with MediaPipe to speed up processing.
        `processing_scale` controls the resize factor for the detector (0 < scale <= 1).
        Landmark coordinates are converted back relative to the original image size.
        """
        # create a downscaled copy for MediaPipe (reduce CPU)
        if processing_scale < 1.0:
            small = cv.resize(image, (0, 0), fx=processing_scale, fy=processing_scale, interpolation=cv.INTER_LINEAR)
        else:
            small = image

        # MediaPipe expects RGB
        small_rgb = cv.cvtColor(small, cv.COLOR_BGR2RGB)
        results = self.hands.process(small_rgb)

        # prepare output image (draw on original size)
        img_out = image.copy()
        index_pos = None
        movement_status = None

        if results.multi_hand_landmarks:
            for h, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # handedness
                # use original normalized coordinates (they are scale-invariant) for math/drawing
                # compute coordinates using original image size
                orig_h, orig_w = image.shape[0], image.shape[1]
                handedness = MessageToDict(results.multi_handedness[h])['classification'][0]['label']
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

                # draw landmarks on the original image (MediaPipe drawing uses normalized coords)
                self.mp_drawing.draw_landmarks(
                    img_out,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style())

                # fingertip 8 in normalized coords -> convert to original pixels
                pos_x = hand_landmarks.landmark[8].x * orig_w
                pos_y = hand_landmarks.landmark[8].y * orig_h
                position = np.matmul(H, np.array([pos_x, pos_y, 1]))
                if index_pos is None:
                    index_pos = np.array([position[0] / position[2], position[1] / position[2], 0], dtype=float)
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

        return index_pos, movement_status, img_out


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

        # Use more features and multiple detectors for robustness
        self.sift_detector = cv.SIFT_create(nfeatures=2000, contrastThreshold=0.03, edgeThreshold=15)
        self.orb_detector = cv.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=12)

        # Extract SIFT features from template
        keypoints_sift, descriptors_sift = self.sift_detector.detectAndCompute(img_object, mask=None)

        # Convert tuple to list for manipulation
        keypoints_sift = list(keypoints_sift)

        # Extract ORB features as backup
        self.keypoints_orb, self.descriptors_orb = self.orb_detector.detectAndCompute(img_object, mask=None)

        # Add corner detection to boost keypoints on edges
        corners = cv.goodFeaturesToTrack(img_object, maxCorners=500, qualityLevel=0.01, minDistance=10)
        if corners is not None:
            corner_kps = [cv.KeyPoint(x=float(c[0][0]), y=float(c[0][1]), size=20) for c in corners]
            keypoints_sift.extend(corner_kps)
            # Recompute SIFT descriptors with added corners
            keypoints_sift, descriptors_sift = self.sift_detector.compute(img_object, keypoints_sift)

        # Store as instance variables
        self.keypoints_sift = keypoints_sift
        self.descriptors_sift = descriptors_sift

        self.requires_homography = True
        self.H = None
        self.MIN_INLIER_COUNT = 10  # Lowered from 15 for better recognition
        print(f"Template features: SIFT={len(self.keypoints_sift)}, ORB={len(self.keypoints_orb)}")

    def detect(self, frame):
        if not self.requires_homography:
            return True, self.H, None

        # Try SIFT matching first (more accurate)
        success, H = self._match_sift(frame)
        if success:
            return True, H, None

        # Fallback to ORB if SIFT fails (faster, works with less texture)
        success, H = self._match_orb(frame)
        if success:
            return True, H, None

        # Return last known good H if available
        if self.H is not None:
            return True, self.H, None
        else:
            return False, None, None

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
        print(f'{method_name} inlier count: {total}')

        if total >= self.MIN_INLIER_COUNT:
            self.H = H
            self.requires_homography = False
            print(f"Homography locked using {method_name}")
            return True, H

        return False, None
