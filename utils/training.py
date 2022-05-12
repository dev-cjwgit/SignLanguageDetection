import os
from sld.configs import Config
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

from utils.DatasetLoader import DatasetLoader

if __name__ == "__main__":
    model_name = input("모델 이름 : ")
    model_name += ".h5"
    if os.path.isfile(model_name):
        raise Exception("이미 존재하는 모델입니다.")

    loader = DatasetLoader(Config.DATASET_DB_FILE)

    X = np.array(loader.data)
    Y = to_categorical(loader.labels).astype(int)

    X_train, Y_train = X, Y

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)
    model = Sequential()

    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(Config.SEQUENCE_LENGTH, 258)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(Config.get_action_num().shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    try:
        model.fit(X_train, Y_train, epochs=150, callbacks=[tb_callback])
    except Exception as e:
        print(e)
    except KeyboardInterrupt as e:
        print(e)

    model.summary()

    model.save(model_name)
    print("finished")
