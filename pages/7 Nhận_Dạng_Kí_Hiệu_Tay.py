import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(
    page_title="Nhận ký hiệu tay ASL",
    page_icon="",
)

st.title('Nhận dạng ký hiệu tay ASL')

# Load the trained model
model_dict = pickle.load(open('utility/RecognitionHandSignASL/model.p', 'rb'))
model = model_dict['model']

# Initialize Mediapipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'A', 1: 'L', 2: 'B'}


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

FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)

while st.session_state.running:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    if not ret:
        st.session_state.running = False
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])

        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                    cv2.LINE_AA)

    FRAME_WINDOW.image(frame_rgb)

# Release resources when stopping
if not st.session_state.running:
    cap.release()
    cv2.destroyAllWindows()
    FRAME_WINDOW.image(Image.open('images/video_notfound.jpg'))