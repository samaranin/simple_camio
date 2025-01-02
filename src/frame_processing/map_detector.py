# type: ignore
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from src.config import config
from src.modules_repository import Module


class MapDetector(Module):
    """
    Module used to detect the map in the camera feed and calculate the homography matrix
    This class exposes only two method, `detect` and `fix_model`:
    - `detect` takes an image as input and returns the homography matrix and the image with the corners drawn
    - `fix_model` stops the detection process and returns the last homography matrix calculated
    If the detection process is stopped, the same homography matrix will be returned for every call to `detect`.
    This is useful when the map is detected we don't want to waste resources on further detections
    """

    DETECTION_INTERVAL = 5  # seconds
    """ Minimum interval between two map detections """

    RATIO_THRESHOLD = 0.75
    """ Threshold used to filter the good matches """

    INLIERS_THRESH = 24
    """ Threshold used to filter good homographies"""
    def __init__(self) -> None:
        super().__init__()

        img_template = cv2.imread(config.template_path, cv2.IMREAD_GRAYSCALE)
        self.map_shape = img_template.shape

        self.detector = cv2.SIFT_create()
        self.template_keypoints, self.template_descriptors = (
            self.detector.detectAndCompute(img_template, mask=None)
        )

        self.last_detection = 0.0, None
        """ Tuple containing the time of the last detection and the homography matrix """

        self.__run_detection = True
        """ Flag used to stop the detection process """

    @property
    def homography(self) -> npt.NDArray[np.float32]:
        """
        Return the homography matrix of the last detection.
        """
        return self.last_detection[1]

    def fix_model(self) -> None:
        """
        Stop the detection process.
        This will cause the `detect` method to always return the last homography matrix calculated.
        """
        self.__run_detection = False

    def detect(
        self, img: npt.NDArray[np.uint8]
    ) -> Tuple[Optional[npt.NDArray[np.float32]], npt.NDArray[np.uint8]]:
        """
        Detect the map in the input image and return the homography matrix and the image with the corners drawn on it (if in debug mode).
        If the detection process is stopped by calling `fix_model`, the homography matrix will be returned without further detections.
        If this method is called before the `DETECTION_INTERVAL` has passed since the last detection, the homography matrix will be returned without further detections.
        """

        if (
            not self.__run_detection
            or time.time() - self.last_detection[0] < self.DETECTION_INTERVAL
        ):
            if self.homography is not None and config.debug:
                img = self.__draw_corners(img, self.last_detection[1])

            return self.homography, img

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        keypoints, descriptors = self.detector.detectAndCompute(img_gray, None)
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)

        try:
            knn_matches = matcher.knnMatch(self.template_descriptors, descriptors, 2)
        except:
            return None, img

        good_matches = list()
        for m, n in knn_matches:
            if m.distance < MapDetector.RATIO_THRESHOLD * n.distance:
                good_matches.append(m)

        if len(good_matches) < 4:
            return None, img

        obj = np.empty((len(good_matches), 2), dtype=np.float32)
        scene = np.empty((len(good_matches), 2), dtype=np.float32)
        for i in range(len(good_matches)):
            # -- Get the keypoints from the good matches
            obj[i, 0] = self.template_keypoints[good_matches[i].queryIdx].pt[0]
            obj[i, 1] = self.template_keypoints[good_matches[i].queryIdx].pt[1]
            scene[i, 0] = keypoints[good_matches[i].trainIdx].pt[0]
            scene[i, 1] = keypoints[good_matches[i].trainIdx].pt[1]

        H, inliers = cv2.findHomography(
            scene, obj, cv2.RANSAC, ransacReprojThreshold=8.0, confidence=0.995
        )

        total = np.sum([int(i) for i in inliers])
        # print(f"Total number of inliners: {total}")
        if total > self.INLIERS_THRESH:
            self.last_detection = time.time(), H
        else:
            print(f"Warning: not enough inliers found for confident estimate of homography ({total}/{self.INLIERS_THRESH})")
            H = self.last_detection[1]

        if H is not None and config.debug:
            img = self.__draw_corners(img, H)

        return H, img

    def __draw_corners(
        self, img: npt.NDArray[np.uint8], homography: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.uint8]:
        """
        Draw the corners of the map detected on the input image.
        This is used for debugging purposes to visualize the detection.
        """
        inverted_homography = np.linalg.inv(homography)

        h, w = self.map_shape[:2]
        corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
        projected_corners = cv2.perspectiveTransform(corners, inverted_homography)

        for i in range(len(projected_corners)):
            x, y = projected_corners[i][0]
            x, y = int(x), int(y)
            cv2.circle(img, (x, y), 8, (255, 255, 255), -1)
            cv2.circle(img, (x, y), 6, (0, 0, 0), -1)

        return img
