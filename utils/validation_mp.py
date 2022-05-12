import os
from tqdm import tqdm
from sld.mediapipes import *
from sld.configs import Config
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

from utils.DatasetLoader import DatasetLoader

if __name__ == "__main__":
    test_model = input("테스트 모델 : ")
    test_model += ".h5"

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

    validset = DatasetLoader(Config.VALIDSET_DB_FILE, 'valid')

    total = 0
    right = 0
    wrong = []
    for action in tqdm(validset.data):
        for idx, frames in enumerate(validset.data[action]):
            res = model.predict(np.expand_dims(frames, axis=0))[0]
            predict_action_name = str(Config.get_action_name(result_arr[np.argmax(res)]))
            if predict_action_name != Config.get_action_name(str(action)):
                wrong.append((action, Config.get_action_name(str(action)), idx, predict_action_name))
            else:
                right += 1
            total += 1

    print("total : " + str(total) + "\n정답률 : " + str(right / total * 100))
    print('-' * 100)
    print(len(wrong), "개 실패")
    for i in wrong:
        print(i)
