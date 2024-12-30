from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import numpy.typing as npt

from src.config import config
from src.modules_repository import Module
from src.utils import Buffer, Coords

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

landmarks = mp_hands.HandLandmark.__members__.values()

active_landmark_style = mp_styles.get_default_hand_landmarks_style()
inactive_landmark_style = mp_styles.get_default_hand_landmarks_style()
connection_style = mp_styles.get_default_hand_connections_style()

red_style = mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=5, circle_radius=1)
green_style = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=5, circle_radius=1)
blue_style = mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=5, circle_radius=1)
for landmark1 in landmarks:
    active_landmark_style[landmark1] = green_style
    inactive_landmark_style[landmark1] = red_style


def ratio(coors: npt.NDArray[np.float32]) -> np.float32:
    """
    This function calculates a value between 0 and 1 that represents how close the points are to be collinear.
    1 means that the points are collinear, 0 means that the points are as far as possible.
    """
    d = np.linalg.norm(coors[0, :] - coors[3, :])
    a = np.linalg.norm(coors[0, :] - coors[1, :])
    b = np.linalg.norm(coors[1, :] - coors[2, :])
    c = np.linalg.norm(coors[2, :] - coors[3, :])

    return d / (a + b + c)


class Hand:
    """
    This class represents a hand detected in an image.
    """

    POINTING_THRESHOLD = 0.08

    class Side(IntEnum):
        """
        Possible sides of a hand.
        """

        RIGHT = 0
        LEFT = 1

        def __str__(self) -> str:
            return self.name.lower()

    def __init__(self, side: Side, is_index_visible: bool, landmarks) -> None:
        self.side = side
        " The side of the hand. "

        self.is_index_visible = is_index_visible
        " A boolean that indicates if the index finger is visible. "

        self.landmarks = landmarks
        " The landmarks of the hand extracted by the MediaPipe library. "

        self.pointing_ratio = self.__get_pointing_ratio()
        " A floating point number between -1 and 1 that represents how much the hand is pointing. "

    @property
    def is_pointing(self) -> bool:
        """
        Whether the hand is pointing or not.
        """
        return self.pointing_ratio > self.POINTING_THRESHOLD

    @property
    def landmark(self) -> List[Any]:
        """
        The landmarks of the hand extracted by the MediaPipe library.
        """
        return self.landmarks

    def draw(self, img: npt.NDArray[np.uint8], active: bool = False) -> None:
        """
        Draw the hand in the image.

        :param img: The image where the hand will be drawn. It should be the same image that was used to detect the hand.
        :param active: A boolean that indicates if the hand should be drawn as active or not.
        """

        mp_drawing.draw_landmarks(
            img,
            self,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=(
                active_landmark_style if active else inactive_landmark_style
            ),
            connection_drawing_spec=connection_style,
        )

    def __get_pointing_ratio(self) -> float:
        """
        Calculate the pointing ratio of a hand.
        The pointing ratio is a floating point number between -1 and 1.
        If the pointing ratio is positive, the hand is pointing.
        """

        if not self.is_index_visible:
            return 0.0

        coors = np.zeros((4, 3), dtype=float)

        for k in [5, 6, 7, 8]:  # joints in index finger
            coors[k - 5, 0], coors[k - 5, 1], coors[k - 5, 2] = (
                self.landmarks[k].x,
                self.landmarks[k].y,
                self.landmarks[k].z,
            )
        ratio_index = ratio(coors)

        for k in [9, 10, 11, 12]:  # joints in middle finger
            coors[k - 9, 0], coors[k - 9, 1], coors[k - 9, 2] = (
                self.landmarks[k].x,
                self.landmarks[k].y,
                self.landmarks[k].z,
            )
        ratio_middle = ratio(coors)

        for k in [13, 14, 15, 16]:  # joints in ring finger
            coors[k - 13, 0], coors[k - 13, 1], coors[k - 13, 2] = (
                self.landmarks[k].x,
                self.landmarks[k].y,
                self.landmarks[k].z,
            )
        ratio_ring = ratio(coors)

        for k in [17, 18, 19, 20]:  # joints in little finger
            coors[k - 17, 0], coors[k - 17, 1], coors[k - 17, 2] = (
                self.landmarks[k].x,
                self.landmarks[k].y,
                self.landmarks[k].z,
            )
        ratio_little = ratio(coors)

        overall = ratio_index - ((ratio_middle + ratio_ring + ratio_little) / 3)
        # print("overall evidence for index pointing:", overall)

        return float(overall)


@dataclass(frozen=True)
class GestureResult:
    """
    This class represents the result of a gesture detection.
    """

    class Status(Enum):
        """
        This class represents the possible statuses of a gesture detection.
        """

        MORE_THAN_ONE_HAND = -1
        """This status represents the case where more than one hand of the same side is found in the image."""

        NOT_FOUND = 0
        """This status represents the case where no hand is found in the image."""

        POINTING = 1
        """This status represents the case where a hand is pointing."""

        EXPLORING = 2
        """This status represents the case where no hand is pointing."""

    status: Status
    """The status of the gesture detection."""

    position: Optional[Coords] = None
    """
    The position of the index finger in the image.
    This field is not None only when the status is `POINTING`.
    """

    side: Optional[Hand.Side] = None
    """
    The side of the hand that is pointing.
    This field is not None only when the status is `POINTING`.
    """


NOT_FOUND = GestureResult(GestureResult.Status.NOT_FOUND)
"""This constant represents the case where no hand is found in the image."""

MORE_THAN_ONE_HAND = GestureResult(GestureResult.Status.MORE_THAN_ONE_HAND)
"""This constant represents the case where more than one hand of the same side is found in the image."""

EXPLORING = GestureResult(GestureResult.Status.EXPLORING)
"""This constant represents the case where no hand is pointing."""


class GestureRecognizer(Module):
    """
    This class is responsible for detecting pointing gestures.
    It exposes only one method, `detect`, that receives an image and a homography matrix and returns a `GestureResult`.
    The homography matrix can be calculated using the `MapDetector` class.
    """

    MOVEMENT_THRESHOLD = 0.25  # inch

    def __init__(self) -> None:
        super().__init__()

        self.movement_threshold = self.MOVEMENT_THRESHOLD * config.feets_per_inch

        template = cv2.imread(config.template_path, cv2.IMREAD_GRAYSCALE)
        self.image_size = template.shape[:2]

        self.hands_detector = mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.75,
            min_tracking_confidence=0.75,
            max_num_hands=4,
        )

        self.buffers = [
            Buffer[Coords](15) for _ in range(2)
        ]  # Left and right hand buffers
        self.gesture_buffer = GestureBuffer()

        self.last_side_pointing: Optional[Hand.Side] = None

    def detect(
        self, img: npt.NDArray[np.uint8], H: npt.NDArray[np.float32]
    ) -> Tuple[GestureResult, npt.NDArray[np.uint8]]:
        """
        Detect pointing gestures in an image and return the result of the detection and the image with the hands drawn.

        :param img: The image where the pointing gestures will be detected.
        :param H: The homography matrix used to detect the hands.
        """

        def detection(hands: List[Hand]) -> GestureResult:
            if len(hands) == 0:
                return NOT_FOUND

            hands = list(filter(lambda h: h.is_index_visible, hands))
            if len(hands) == 0:
                return EXPLORING

            hands_per_side = {
                side: [h for h in hands if h.side == side] for side in Hand.Side
            }

            if (
                len(hands_per_side[Hand.Side.LEFT]) > 1
                or len(hands_per_side[Hand.Side.RIGHT]) > 1
            ):
                return MORE_THAN_ONE_HAND

            for hand in hands:
                self.buffers[hand.side].add(self.get_index_position(hand, img, H))

            pointing_hands = list(filter(lambda h: h.is_pointing, hands))

            if len(pointing_hands) == 0:
                return EXPLORING

            side_pointing = self.last_side_pointing
            if len(pointing_hands) == 1:
                side_pointing = pointing_hands[0].side

            elif self.is_moving(Hand.Side.RIGHT):
                side_pointing = Hand.Side.RIGHT

            elif self.is_moving(Hand.Side.LEFT):
                side_pointing = Hand.Side.LEFT

            if side_pointing is None:
                return EXPLORING

            result = GestureResult(
                GestureResult.Status.POINTING,
                self.buffers[side_pointing].last(),
                side_pointing,
            )

            return result

        hands = self.__get_hands(img, H)
        gesture = detection(hands)

        self.last_side_pointing = gesture.side

        for hand in hands:
            hand.draw(img, active=hand.side == self.last_side_pointing)

        return self.gesture_buffer.aggregate(gesture), img

    def is_moving(self, side: Hand.Side) -> bool:
        """
        Return whether the index finger of a hand is moving or not for a given side.
        This method uses a buffer to store the last positions of the index finger. The buffer is updated every time the `detect` method is called.

        :param side: The side of the hand.
        """

        first = self.buffers[side].first()
        last = self.buffers[side].last()

        if first is None or last is None:
            return False

        return last.distance_to(first) > self.movement_threshold

    def get_index_position(
        self, hand, img: npt.NDArray[np.uint8], H: npt.NDArray[np.float32]
    ) -> Coords:
        """
        Return the position of the index finger in the image for a given hand.

        :param hand: The landmarks of the hand extracted by the MediaPipe library.
        :param img: The image where the hand was detected.
        """

        position = np.array(
            [
                [
                    hand.landmark[8].x * img.shape[1],
                    hand.landmark[8].y * img.shape[0],
                ]
            ],
            dtype=np.float32,
        ).reshape(-1, 1, 2)

        position = cv2.perspectiveTransform(position, H)[0][0]
        return Coords(position[0], position[1])

    def is_index_visible(
        self, hand, img: npt.NDArray[np.uint8], H: npt.NDArray[np.float32]
    ) -> bool:
        """
        Return whether the index finger is visible in the image for a given hand.

        :param hand: The landmarks of the hand extracted by the MediaPipe library.
        :param img: The image where the hand was detected.
        :param H: The homography matrix used to detect the hand
        """
        index_position = self.get_index_position(hand, img, H)

        return (
            0 <= index_position[0] < self.image_size[0]
            and 0 <= index_position[1] < self.image_size[1]
        )

    def __get_hands(
        self, img: npt.NDArray[np.uint8], H: npt.NDArray[np.float32]
    ) -> List[Hand]:
        """
        Apply the hand detection model to an image and return a list of `Hand` objects.

        :param img: The image where the hands will be detected.
        :param H: The homography matrix used to detect the hands.
        """

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img.flags.writeable = False
        results = self.hands_detector.process(img)
        img.flags.writeable = True

        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        return self.__process_results(img, H, results)

    def __process_results(
        self, img: npt.NDArray[np.uint8], H: npt.NDArray[np.float32], results
    ) -> List[Hand]:
        """
        Process the results of the hand detection model and return a list of `Hand` objects.

        :param img: The image where the hands were detected.
        :param H: The homography matrix used to detect the hands.
        :param results: The results of the hand detection model.
        """

        if not results.multi_hand_landmarks:
            return list()

        return [
            Hand(
                side=Hand.Side(results.multi_handedness[i].classification[0].index),
                is_index_visible=self.is_index_visible(hand, img, H),
                landmarks=hand.landmark,
            )
            for i, hand in enumerate(results.multi_hand_landmarks)
        ]


class GestureBuffer:
    """
    This class is responsible for aggregating gesture results.
    It uses a buffer to store the last results and return the aggregated result.
    The buffer is updated every time the `aggregate` method is called.
    The buffer has a maximum size of 5 and a maximum life of 1 second.
    """

    MAX_SIZE = 5
    MAX_LIFE = 1

    def __init__(self) -> None:
        self.buffer = Buffer[GestureResult.Status](
            GestureBuffer.MAX_SIZE, GestureBuffer.MAX_LIFE
        )
        self.last_position: Optional[Coords] = None

    def aggregate(self, gesture: GestureResult) -> GestureResult:
        """
        Add a gesture result to the buffer and return the aggregated result.
        """

        self.buffer.add(gesture.status)

        res = NOT_FOUND
        status = self.buffer.mode()

        if status == GestureResult.Status.EXPLORING:
            res = EXPLORING

        elif status == GestureResult.Status.MORE_THAN_ONE_HAND:
            res = MORE_THAN_ONE_HAND

        elif status == GestureResult.Status.POINTING:
            position = gesture.position or self.last_position
            res = GestureResult(GestureResult.Status.POINTING, position, gesture.side)

        self.last_position = gesture.position or self.last_position
        return res
