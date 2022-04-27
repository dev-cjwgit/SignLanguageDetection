from sld.mediapipes import *
from sld.configs import Config
import cv2


if __name__ == "__main__":
    mp = MediaPipe(detection_option=["face", "pose", "lh", "rh"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)  # 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)  # 720

    while cap.isOpened():
        ret, frame = cap.read()

        image, result = mp.mediapipe_detection(frame)

        mp.draw_styled_landmarks(image, result)

        cv2.imshow("MediaPipe", image)

        cv2.waitKey(1)
