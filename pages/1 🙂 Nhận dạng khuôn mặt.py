import streamlit as st
import numpy as np
import cv2 as cv
import joblib

# Function to convert string to boolean
def str2bool(v):
    if v.lower() in ['on', 'yes', 'true', 'y', 't']:
        return True
    elif v.lower() in ['off', 'no', 'false', 'n', 'f']:
        return False
    else:
        raise NotImplementedError

# Define default paths to models
DEFAULT_FACE_DETECTION_MODEL = './NhanDangKhuonMat_onnx/model/face_detection_yunet_2023mar.onnx'
DEFAULT_FACE_RECOGNITION_MODEL = './NhanDangKhuonMat_onnx/model/face_recognition_sface_2021dec.onnx'
DEFAULT_SVC_MODEL_PATH = './NhanDangKhuonMat_onnx/model/svc.pkl'

# Load SVC model and names dictionary
svc = joblib.load(DEFAULT_SVC_MODEL_PATH)
mydict = ['BanDinh', 'BanKhoa', 'BanNguyen', 'BanThinh', 'LcYuu']

# Visualize function
def visualize(input, faces, fps, names=None, thickness=2):
    if faces[1] is not None:
        for idx, face in enumerate(faces[1]):
            coords = face[:-1].astype(np.int32)
            cv.rectangle(input, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), thickness)
            cv.circle(input, (coords[4], coords[5]), 2, (255, 0, 0), thickness)
            cv.circle(input, (coords[6], coords[7]), 2, (0, 0, 255), thickness)
            cv.circle(input, (coords[8], coords[9]), 2, (0, 255, 0), thickness)
            cv.circle(input, (coords[10], coords[11]), 2, (255, 0, 255), thickness)
            cv.circle(input, (coords[12], coords[13]), 2, (0, 255, 255), thickness)
            if names is not None:
                name = names[idx]
                cv.putText(input, name, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.putText(input, 'FPS: {:.2f}'.format(fps), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return input

# Streamlit app
def main():
    st.title('Nhận dạng khuôn mặt')
    
    # Sidebar settings
    st.sidebar.title('Settings')
    face_detection_model_path = st.sidebar.text_input('Face Detection Model Path', DEFAULT_FACE_DETECTION_MODEL)
    face_recognition_model_path = st.sidebar.text_input('Face Recognition Model Path', DEFAULT_FACE_RECOGNITION_MODEL)
    svc_model_path = st.sidebar.text_input('SVC Model Path', DEFAULT_SVC_MODEL_PATH)
    
    # Load models
    detector = cv.FaceDetectorYN.create(face_detection_model_path, "", (320, 320), 0.9, 0.3, 5000)
    recognizer = cv.FaceRecognizerSF.create(face_recognition_model_path, "")
    
    # Display
    FRAME_WINDOW = st.image([])
    
    # Start and stop buttons
    col1, col2 = st.columns(2)
    start_button = col1.button('Start')
    stop_button = col2.button('Stop')

    if 'run' not in st.session_state:
        st.session_state.run = False

    if start_button:
        st.session_state.run = True
    
    if stop_button:
        st.session_state.run = False

    if st.session_state.run:
        cap = cv.VideoCapture(0)
        frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        detector.setInputSize([frameWidth, frameHeight])

        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.write('No frames grabbed!')
                break

            # Inference
            faces = detector.detect(frame)
            recognized_names = []

            if faces[1] is not None:
                for face in faces[1]:
                    face_align = recognizer.alignCrop(frame, face)
                    face_feature = recognizer.feature(face_align)
                    test_predict = svc.predict(face_feature)
                    result = mydict[test_predict[0]]
                    recognized_names.append(result)

            # Draw results on the frame
            frame = visualize(frame, faces, 30, names=recognized_names)
            
            # Display the frame in Streamlit
            FRAME_WINDOW.image(frame, channels='BGR')

        cap.release()

if __name__ == '__main__':
    main()
