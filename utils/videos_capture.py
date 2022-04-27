import time

from sld.mediapipes import *
import cv2
from sld.configs import Config
import os

if __name__ == "__main__":
    DATA_PATH = os.path.join(Config.VIDEO_FOLDER)

    # 제스처 이름
    action = str(input("제스처 이름 : "))
    count = int(input("동영상 개수 : "))
    if not os.path.isdir(DATA_PATH + "/" + action):
        os.makedirs(DATA_PATH + "/" + action)

    pos = 0
    while os.path.isfile('./' + Config.VIDEO_FOLDER + '/' + str(action) + "/" + str(pos) + ".avi"):
        pos += 1

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)  # 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)  # 720
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    start = time.time()
    sequence = 0
    while sequence < count:
        ret, frame = cap.read()
        image, result = mp.mediapipe_detection(frame)
        mp.draw_styled_landmarks(image, result)
        remain = time.time() - start
        if remain > Config.WAIT_TIME:
            out = cv2.VideoWriter(DATA_PATH + "/" + action + "/" + str(sequence + pos) + ".avi", fourcc, Config.FPS,
                                  (Config.CAMERA_WIDTH, Config.CAMERA_HEIGHT))
            # st = time.time()
            # about 2.5sec
            for idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                out.write(frame)

                image, result = mp.mediapipe_detection(frame)
                mp.draw_styled_landmarks(image, result)

                cv2.putText(image, 'capture %d frame' % (idx), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

                cv2.imshow("utils", image)
                cv2.waitKey(1)
            sequence += 1
            start = time.time()
            # print(time.time() - st)
        else:
            cv2.putText(image, '%d wait %.2f sec ' % (sequence, Config.WAIT_TIME - remain), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)

        cv2.imshow("utils", image)
        cv2.waitKey(1)
