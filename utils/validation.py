import os
from tqdm import tqdm
from sld.mediapipes import *
from sld.configs import Config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

if __name__ == "__main__":
    test_model = input("테스트 모델 : ")
    test_model += ".h5"

    action_list = os.listdir(Config.VALID_FOLDER)

    print(action_list)

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    model = Sequential()
    result_arr = Config.get_action_num()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(Config.SEQUENCE_LENGTH, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(result_arr.shape[0], activation='softmax'))

    model.load_weights(test_model)
    total = 0
    right = 0
    wrong = []
    print()
    for action in tqdm(action_list):
        movie_list = os.listdir(Config.VALID_FOLDER + "/" + action)
        action_name = Config.get_action_name(action)
        print("start : [" + action_name + "]", action)
        for idx, movie in enumerate(movie_list):
            cap = cv2.VideoCapture('./' + Config.VALID_FOLDER + "/" + action + "/" + movie)
            sequences = []
            for frame_idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                image, result = mp.mediapipe_detection(frame)
                keypoints = mp.extract_keypoints(result)
                sequences.append(keypoints)
            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            predict_action_name = str(Config.get_action_name(result_arr[np.argmax(res)]))
            if predict_action_name != action_name:
                wrong.append((action_name, movie, predict_action_name))
                print("faild", wrong[-1])
            else:
                right += 1
            cap.release()
            total += 1
    print("total : " + str(total) + "\n정답률 : " + str(right / total * 100))
    print('-' * 100)
    print(len(wrong), "개 실패")
    for i in wrong:
        print(i)
