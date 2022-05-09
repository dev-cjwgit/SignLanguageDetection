import os
from tqdm import tqdm
import numpy as np
from sld.configs import Config
from utils.DatasetLoader import DatasetLoader
from utils.DatasetWriter import DatasetWriter
from tensorflow.keras.utils import to_categorical
import math

if __name__ == "__main__":
    DATA_PATH = os.path.join('utils/' + Config.FRAME_FOLDER)

    writer = DatasetWriter('utils/' + Config.DATASET_DB_FILE)

    label_map = {label: num for num, label in enumerate(Config.get_action_num())}
    sequences, labels = [], []
    for action in tqdm(Config.get_action_num(), desc="ALL"):
        # action_list = os.listdir(os.path.join(DATA_PATH, action))
        # action_list.sort(key=lambda x: int(x))
        for sequence in tqdm(np.array(os.listdir(os.path.join(DATA_PATH, action))).astype(int)):
            window = []
            for frame_num in range(Config.SEQUENCE_LENGTH):
                res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
                window.append(res)
            writer.append(action, sequence, window)
            sequences.append(window)
            labels.append(label_map[action])
    print()
    count = 0


