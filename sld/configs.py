import numpy as np

# Project Settings

ACTIONS = [
    {"0": "동작 없음"},
    {"6045": "신기록"},
    {"6112": "수제비"},
    {"2673": "뽀뽀"},
    {"8229": "월세"},
    {"15526": "사용자"}
]


class Config:
    # MediaPipe Settings
    MIN_DECTION_CONFIDENCE = 0.5
    MIN_TRACKING_CONFIDENCE = 0.5

    # Train & Test
    VALID_FOLDER = "VV_Data"
    RECOGNIZE_THRESHOLD = 0.5

    # Videos_capture
    WAIT_TIME = 3
    SEQUENCE_LENGTH = 40
    VIDEO_FOLDER = "GV_Data"
    FRAME_FOLDER = "MP_Data"
    FPS = 10

    # OpenCV Settings
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720

    @staticmethod
    def get_action_num():
        return np.array([list(i.keys()) for i in ACTIONS]).squeeze(axis=1)

    @staticmethod
    def get_action_dict():
        return ACTIONS
