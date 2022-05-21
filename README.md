#SignLanguageDection
https://www.youtube.com/watch?v=doDUihpj6ro

Python 3.8.13
opencv 4.5.5.64

#Dependency Library
<h2>terminal</h2>

    cd ../SignLanguageDetection/
    pip install tensorflow==2.4.1 tensorflow-gpu==2.4.1 opencv-python mediapipe sklearn matplotlib tqdm


<h2>m1 mac<h2>
   
    pip install mediapipe-silicon
    conda install -c apple tensorflow-deps
    python -m pip install tensorflow-macos
    python -m pip install tensorflow-metal
    pip install opencv-python sklearn matplotlib tqdm
    
비디오 저장 = utils/videos_capture.py

비디오를 프레임으로 변환 = utils/videos_to_frame.py

학습(데이터 학습) = utils/training.py

학습 모델 검증 = utils/validation.py

실시간 테스트 = test/gesutre_recognize.py

configs/Config의 ACTIONS 참고

GV_Data = 동영상 원본 데이터
MP_Data = 동영상 프레임 MP Point 데이터
VV_Data = Valid 동영상 원본 데이터
VMP_Data = Valid 동영상 프레임 MP Point 데이터
