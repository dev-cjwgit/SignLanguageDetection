import time
import os
from sld.mediapipes import *
from sld.configs import Config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sld.mediapipes import *
import cv2
from sld.configs import Config

if __name__ == "__main__":
    # test_model = input("사용 모델 : ")
    # test_model += ".h5"
    test_model = "cjw_khy20_90.h5"

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    result_arr = Config.get_action_num()

    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(Config.SEQUENCE_LENGTH, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(result_arr.shape[0], activation='softmax'))
    model.load_weights(test_model)

    sequence = 0

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)  # 1280
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)  # 720
    start = time.time()
    while True:
        ret, frame = cap.read()
        image, result = mp.mediapipe_detection(frame)
        mp.draw_styled_landmarks(image, result)
        remain = time.time() - start
        if remain > Config.WAIT_TIME:
            print('start')
            sequences = []
            st = time.time()
            for idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                image, result = mp.mediapipe_detection(frame)
                mp.draw_styled_landmarks(image, result)
                keypoints = mp.extract_keypoints(result)
                sequences.append(keypoints)
                cv2.putText(image, 'capture %d frame' % (idx), (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

                cv2.imshow("utils", image)
                cv2.waitKey(2)
            print(time.time() - st)
            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            print("per : " + str(res[np.argmax(res)]) + "\nRes : " +
                  str(Config.get_action_name(result_arr[np.argmax(res)])))
            sequence += 1
            start = time.time()
        else:
            cv2.putText(image, '%d wait %.2f sec ' % (sequence, Config.WAIT_TIME - remain), (100, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
        cv2.imshow("utils", image)
        cv2.waitKey(1)
