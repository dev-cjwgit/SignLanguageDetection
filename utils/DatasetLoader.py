import os
import numpy as np
import sqlite3

from sld.configs import Config
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical


class DatasetLoader:
    data = []
    labels = list()

    def load(self):
        self.data.clear()
        label_map = {label: num for num, label in enumerate(Config.get_action_num())}

        self.cur.execute('SELECT A.action_num, A.movie_num, F.frame_num, ' +
                         ', '.join(['p%d REAL' % (idx) for idx in range(258)]) +
                         ' FROM action A ' +
                         'LEFT OUTER JOIN frame F ' +
                         'ON A.uid = F.action_uid ' +
                         'ORDER BY A.action_num, A.movie_num, F.frame_num;')
        result = self.cur.fetchall()

        # movie_num과 frame_num은 빈 수 없이 순서적이여야 함.
        window = []
        for idx, item in tqdm(enumerate(result), desc="loader"):
            action, count, frame = item[0], item[1], item[2]
            window.append(np.array(item[3:]))

            if idx % Config.SEQUENCE_LENGTH == Config.SEQUENCE_LENGTH - 1:
                self.labels.append(label_map[str(action)])
                self.data.append(window[:])
                window.clear()

    def __init__(self, file_name):
        self.conn = sqlite3.connect(file_name)
        self.cur = self.conn.cursor()
        self.load()

    def __getitem__(self, item):
        return self.data[item]

    def __del__(self):
        self.cur.close()
        self.conn.close()
