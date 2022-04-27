from sld.mediapipes import *
import os
from tqdm import tqdm

if __name__ == "__main__":
    DATA_PATH = os.path.join(Config.FRAME_FOLDER)

    if not os.path.isdir(DATA_PATH + "/"):
        os.makedirs(DATA_PATH + "/")
    else:
        raise Exception("MP_Data가 존재합니다.")

    action_list = os.listdir(Config.VIDEO_FOLDER)
    print(action_list)

    mp = MediaPipe(detection_option=["pose", "lh", "rh"])

    for action in tqdm(action_list):
        movie_list = os.listdir(Config.VIDEO_FOLDER + "/" + action)
        print("start : [" + action + "]")
        for idx, movie in tqdm(enumerate(movie_list)):
            cap = cv2.VideoCapture('./' + Config.VIDEO_FOLDER + "/" + action + "/" + movie)
            os.makedirs(os.path.join(DATA_PATH, str(action), str(idx)))
            for frame_idx in range(Config.SEQUENCE_LENGTH):
                ret, frame = cap.read()
                if not ret:
                    break

                image, result = mp.mediapipe_detection(frame)
                keypoints = mp.extract_keypoints(result)
                npy_path = os.path.join(DATA_PATH, action, str(idx), str(frame_idx))
                np.save(npy_path, keypoints)
