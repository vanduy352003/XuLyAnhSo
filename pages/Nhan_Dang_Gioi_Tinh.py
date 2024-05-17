import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2 as cv
from PIL import Image

# Streamlit page configuration
st.set_page_config(
    page_title="Nhận dạng giới tính",
    page_icon="😉",
)

st.subheader('Nhận dạng khuôn mặt')
st.image(Image.open('images/recognition_face.jpg'), width=700)
FRAME_WINDOW = st.image([])

# Initialize video capture
cap = cv.VideoCapture(0)

# Session state initialization
if 'running' not in st.session_state:
    st.session_state.running = False

# Start and stop buttons
start_btn, stop_btn = st.columns(2)
with start_btn:
    start_press = st.button('Start')
with stop_btn:
    stop_press = st.button('Stop')

# Handle button presses
if start_press:
    st.session_state.running = True
if stop_press:
    st.session_state.running = False

# Load the models outside the loop
model = load_model('utility/RecognitionGender/gender_detection.h5')
classes = ['man', 'woman']

# Load face detection and recognition models
detector = cv.FaceDetectorYN.create(
    'utility/RecognitionFace/face_detection_yunet_2023mar.onnx',
    "",
    (320, 320),
    0.9,
    0.3,
    5000
)
recognizer = cv.FaceRecognizerSF.create(
    'utility/RecognitionFace/face_recognition_sface_2021dec.onnx',
    ""
)

# Set input size for the detector
frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
detector.setInputSize([frameWidth, frameHeight])

# Function to process frame
def process_frame(frame):
    # Detect faces
    faces = detector.detect(frame)

    if faces[1] is not None:
        for face in faces[1]:
            coords = face[:-1].astype(np.int32)
            cv.rectangle(frame, (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3]), (0, 255, 0), 2)
            
            # Align the detected face for recognition
            face_align = recognizer.alignCrop(frame, face)
            
            # Convert the aligned face to a format suitable for gender detection
            face_resized = cv.resize(face_align, (96, 96))
            face_resized = face_resized.astype("float") / 255.0
            face_resized = img_to_array(face_resized)
            face_resized = np.expand_dims(face_resized, axis=0)
            
            # Perform gender detection
            conf = model.predict(face_resized)[0]
            idx = np.argmax(conf)
            label = "{}: {:.2f}%".format(classes[idx], conf[idx] * 100)
            
            # Write label and confidence above face rectangle
            cv.putText(frame, label, (coords[0], coords[1] - 10), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

# Main loop for processing video frames
while st.session_state.running:
    ret, frame = cap.read()
    if not ret:
        st.session_state.running = False
        break
    frame = process_frame(frame)
    FRAME_WINDOW.image(frame, channels='BGR')

# Release resources when stopping
if not st.session_state.running:
    cap.release()
    cv.destroyAllWindows()
    FRAME_WINDOW.image(Image.open('images/video_notfound.jpg'))
