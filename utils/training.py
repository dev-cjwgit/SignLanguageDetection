import os
from sld.configs import Config
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical

if __name__ == "__main__":
    DATA_PATH = os.path.join(Config.FRAME_FOLDER)
    model_name = input("모델 이름 : ")
    model_name += ".h5"
    if os.path.isfile(model_name):
        raise Exception("이미 존재하는 모델입니다.")

    label_map = {label: num for num, label in enumerate(Config.get_action_num())}
    sequences, labels = [], []
    for action in tqdm(Config.get_action_num(), desc="ALL"):
        for sequence in tqdm(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)):
            window = []
            for frame_num in range(Config.SEQUENCE_LENGTH):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[action])

    X = np.array(sequences)
    Y = to_categorical(labels).astype(int)

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
        model.fit(X_train, Y_train, epochs=10000, callbacks=[tb_callback])
    except Exception as e:
        print(e)
    except KeyboardInterrupt as e:
        print(e)

    model.summary()

    model.save(model_name)
    print("finished")
