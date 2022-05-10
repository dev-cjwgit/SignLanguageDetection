from sld.mediapipes import *
import os
from tqdm import tqdm

from utils.DatasetWriter import DatasetWriter

if __name__ == "__main__":
    DATA_PATH = os.path.join(Config.FRAME_FOLDER)
    action_list = os.listdir(Config.VIDEO_FOLDER)
    print(action_list)
    writer = DatasetWriter(Config.DATASET_DB_FILE)

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    for action in tqdm(action_list, desc="all"):
        movie_list = os.listdir(Config.VIDEO_FOLDER + "/" + action)
        for movie in tqdm(movie_list, desc=action):
            movie_name = ''.join(movie.split('.')[:-1])
            cap = cv2.VideoCapture(Config.VIDEO_FOLDER + "/" + action + "/" + movie)
            window = []
            for frame_idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                image, result = mp.mediapipe_detection(frame)
                keypoints = mp.extract_keypoints(result)
                window.append(keypoints)
            writer.append(action, window)
