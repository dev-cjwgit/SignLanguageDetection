import os
import time
from tqdm import tqdm
from sld.mediapipes import *
from sld.configs import Config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

if __name__ == "__main__":
    test_model = input("테스트 모델 : ")
    test_model += ".h5"

    action_list = os.listdir(Config.VALID_FOLDER_MP)

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
    st = time.time()
    for action in action_list:
        movie_list = os.listdir(Config.VALID_FOLDER_MP + "/" + action)
        action_name = Config.get_action_name(action)
        for movie in tqdm(movie_list, desc=action_name):
            sequences = []
            for frame_num in range(Config.SEQUENCE_LENGTH):
                res = np.load(os.path.join(Config.VALID_FOLDER_MP, action, str(movie), "{}.npy".format(frame_num)))
                sequences.append(res)

            res = model.predict(np.expand_dims(sequences, axis=0))[0]
            predict_action_name = str(Config.get_action_name(result_arr[np.argmax(res)]))
            if predict_action_name != action_name:
                wrong.append((action_name, action, movie, predict_action_name))
            else:
                right += 1
            total += 1
    print(time.time() - st)
    print("total : " + str(total) + "\n정답률 : " + str(right / total * 100))
    print('-' * 100)
    print(len(wrong), "개 실패")
    for i in wrong:
        print(i)
