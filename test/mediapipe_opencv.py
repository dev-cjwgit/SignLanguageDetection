from sld.mediapipes import *
from sld.configs import Config
import cv2

if __name__ == "__main__":
    mp = MediaPipe(detection_option=["face", "pose", "lh", "rh"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)  # 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)  # 720

    width_center = Config.CAMERA_WIDTH // 2

    while cap.isOpened():
        ret, frame = cap.read()
        image, result = mp.mediapipe_detection(frame)
        mp.draw_styled_landmarks(image, result)
        image = cv2.circle(image,
                           (width_center, 170),
                           130,
                           (255, 0, 0),
                           thickness=2,
                           lineType=cv2.LINE_AA)

        # 원의 중심, 반지름, 선의 색, 굵기, 선 표현법

        # cv2.imshow("image circle", frame)

        cv2.rectangle(image,
                      (width_center - 400, 300),
                      (width_center + 400, Config.CAMERA_HEIGHT),
                      (255, 0, 0),
                      thickness=2,
                      lineType=cv2.LINE_AA)

        # 좌측 상단 꼭지점, 우측 하단 꼭지점 , 색, 두께, 선 표현법

        cv2.imshow("MediaPipe", image)

        cv2.waitKey(1)
