from sld.mediapipes import *
import os
from tqdm import tqdm

from utils.DatasetWriter import DatasetWriter

mode = "valid"  # train, valid

if __name__ == "__main__":
    if mode == "train":
        action_list = os.listdir(Config.VIDEO_FOLDER)
        print(action_list)
        writer = DatasetWriter(Config.DATASET_DB_FILE)
    else:
        action_list = os.listdir(Config.VALID_FOLDER_VIDEO)
        print(action_list)
        writer = DatasetWriter(Config.VALIDSET_DB_FILE)

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    for action in tqdm(action_list, desc="all"):
        if mode == "train":
            movie_list = os.listdir(Config.VIDEO_FOLDER + "/" + action)
        else:
            movie_list = os.listdir(Config.VALID_FOLDER_VIDEO + "/" + action)

        for movie in tqdm(movie_list, desc=action):
            movie_name = ''.join(movie.split('.')[:-1])
            if mode == "train":
                cap = cv2.VideoCapture(Config.VIDEO_FOLDER + "/" + action + "/" + movie)
            else:
                cap = cv2.VideoCapture(Config.VALID_FOLDER_VIDEO + "/" + action + "/" + movie)
            window = []
            for frame_idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                image, result = mp.mediapipe_detection(frame)
                keypoints = mp.extract_keypoints(result)
                window.append(keypoints)
            writer.append(action, window)
