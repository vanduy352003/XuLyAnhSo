import streamlit as st
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2 as cv
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
from keras.models import Sequential

# Streamlit page configuration
st.set_page_config(page_title="Nhận dạng cảm xúc", page_icon="📱")

# Background and header styling
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("");
    background-size: 100% 100%;
}
[data-testid="stHeader"]{
    background: rgba(0,0,0,0);
}
[data-testid="stToolbar"]{
    right:2rem;
}
[data-testid="stSidebar"] > div:first-child {
    background-image: url("https://i.pinimg.com/736x/1b/e2/91/1be2919a288c48fe59ba448f92898bcc.jpg");
    background-position: center;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

st.title('Nhận dạng cảm xúc')

# Load the model architecture from JSON file
with open("./utility/RecognitionEmotion/facialemotionmodel.json", "r") as json_file:
    model_json = json_file.read()

# Register the 'Sequential' class if it's not found

model = model_from_json(model_json, custom_objects={'Sequential': Sequential})

# Load the weights into the model
model.load_weights("./utility/RecognitionEmotion/facialemotionmodel.h5")

# Load the Haar Cascade for face detection
haar_file = cv.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv.CascadeClassifier(haar_file)

# Function to extract features from the image
def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

# Streamlit image display
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

# Labels for the emotion categories
labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Function to process frame
def process_frame(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        cv.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = cv.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]
        cv.putText(frame, prediction_label, (x, y - 10), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)

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
