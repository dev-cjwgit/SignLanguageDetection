import os

import cv2
from tqdm import tqdm
import numpy as np
from sld.configs import Config
from sld.mediapipes import MediaPipe
from utils.DatasetLoader import DatasetLoader
from utils.DatasetWriter import DatasetWriter
from tensorflow.keras.utils import to_categorical
import math

if __name__ == "__main__":
    writer = DatasetWriter('utils/' + Config.DATASET_DB_FILE)

    writer.delete(0, 0)
    writer.delete(0, 1)
    writer.delete(0, 2)
