import mediapipe as mp
import cv2
import numpy as np
from sld.configs import Config

options = ["face", "pose", "lh", "rh"]


class MediaPipe:
    def __init__(self, detection_option=None):
        if detection_option is None:
            detection_option = ["face", "pose", "lh", "rh"]

        if not (1 <= len(detection_option) <= 4):
            raise Exception("잘못된 옵션입니다.")

        for option in detection_option:
            if option not in options:
                raise Exception("잘못된 옵션입니다.")

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=Config.MIN_DECTION_CONFIDENCE,
                                                  min_tracking_confidence=Config.MIN_TRACKING_CONFIDENCE)
        self.detection_option = detection_option

    def mediapipe_detection(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
        image.flags.writeable = False  # Image is no longer writeable
        results = self.holistic.process(image)  # Make prediction
        image.flags.writeable = True  # Image is now writeable
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
        return image, results

    def extract_keypoints(self, results):
        keypoint = []
        if "pose" in self.detection_option:
            # pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
            #                  results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

            # key point에서 23 이상으로는 인식하지 않음
            # https://google.github.io/mediapipe/solutions/pose.html
            pose = np.array([[res.x, res.y, res.z, res.visibility if idx < 23 else 0]
                             for idx, res in enumerate(results.pose_landmarks.landmark)]).flatten() \
                if results.pose_landmarks else np.zeros(132)
            keypoint.append(pose)

        if "face" in self.detection_option:
            face = np.array([[res.x, res.y, res.z] for res in
                             results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(1404)
            keypoint.append(face)

        if "lh" in self.detection_option:
            lh = np.array([[res.x, res.y, res.z]
                           for res in
                           results.left_hand_landmarks.landmark]).flatten() \
                if results.left_hand_landmarks \
                else np.zeros(63)
            keypoint.append(lh)

        if "rh" in self.detection_option:
            rh = np.array([[res.x, res.y, res.z]
                           for res in
                           results.right_hand_landmarks.landmark]).flatten() \
                if results.right_hand_landmarks \
                else np.zeros(63)
            keypoint.append(rh)
        return np.concatenate(keypoint)

    # def draw_landmarks(self, image, results):
    #     if "face" in self.detection_option:
    #         # Draw face connections
    #         self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS)
    #     if "pose" in self.detection_option:
    #         # Draw pose connections
    #         self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
    #     if "lh" in self.detection_option:
    #         # Draw left hand connections
    #         self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
    #     if "rh" in self.detection_option:
    #         # Draw right hand connections
    #         self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)

    def draw_styled_landmarks(self, image, results):
        if "face" in self.detection_option:
            # Draw face connections
            self.mp_drawing.draw_landmarks(image, results.face_landmarks, self.mp_holistic.FACEMESH_CONTOURS,
                                           self.mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1,
                                                                       circle_radius=1),
                                           self.mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1,
                                                                       circle_radius=1))
        if "pose" in self.detection_option:
            # Draw pose connections
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2,
                                                                       circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2,
                                                                       circle_radius=2))
        if "lh" in self.detection_option:
            # Draw left hand connections
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2,
                                                                       circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2,
                                                                       circle_radius=2))
        if "rh" in self.detection_option:
            # Draw right hand connections
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS,
                                           self.mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                       circle_radius=4),
                                           self.mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                       circle_radius=2))
