import streamlit as st
from keras.models import load_model
from imutils.contours import sort_contours
import numpy as np
import imutils
import cv2
from PIL import Image

# Load the handwriting OCR model
model = load_model("utility/RecognitionHandWrite/handwrite.h5")

# Function to process the uploaded image and perform OCR
def process_image(image):
    # Convert the image to grayscale and blur it to reduce noise
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Perform edge detection, find contours in the edge map, and sort the resulting contours from left-to-right
    edged = cv2.Canny(blurred, 30, 150)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # Initialize the list of contour bounding boxes and associated characters that we'll be OCR'ing
    chars = []

    # Loop over the contours
    for c in cnts:
        # Compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)

        # Filter out bounding boxes, ensuring they are neither too small nor too large
        if (w >= 5 and w <= 150) and (h >= 15 and h <= 120):
            # Extract the character and threshold it to make the character appear as *white* (foreground) on a *black* background
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
            (tH, tW) = thresh.shape

            # Resize along the width dimension if the width is greater than the height
            if tW > tH:
                thresh = imutils.resize(thresh, width=32)
            # Otherwise, resize along the height
            else:
                thresh = imutils.resize(thresh, height=32)

            # Re-grab the image dimensions and pad the image to ensure 32x32 dimensions
            (tH, tW) = thresh.shape
            dX = int(max(0, 32 - tW) / 2.0)
            dY = int(max(0, 32 - tH) / 2.0)

            padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY, left=dX, right=dX, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
            padded = cv2.resize(padded, (32, 32))

            # Prepare the padded image for classification via the handwriting OCR model
            padded = padded.astype("float32") / 255.0
            padded = np.expand_dims(padded, axis=-1)

            # Update the list of characters that will be OCR'd
            chars.append((padded, (x, y, w, h)))

    # Extract the bounding box locations and padded characters
    boxes = [b[1] for b in chars]
    chars = np.array([c[0] for c in chars], dtype="float32")

    # OCR the characters using the handwriting recognition model
    preds = model.predict(chars)

    # Define the list of label names
    labelNames = "0123456789"
    labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labelNames = [l for l in labelNames]

    # Loop over the predictions and bounding box locations together
    for (pred, (x, y, w, h)) in zip(preds, boxes):
        # Find the index of the label with the largest corresponding probability, then extract the probability and label
        i = np.argmax(pred)
        prob = pred[i]
        label = labelNames[i]

        # Draw the prediction on the image
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)

    return image

# Streamlit application
st.title("Handwriting OCR with Streamlit")
st.write("Upload an image of handwritten text to perform OCR.")

# File uploader for images
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Process the image and perform OCR
    processed_image = process_image(image)

    # Convert the processed image back to RGB for displaying in Streamlit
    processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

    # Display the original and processed images
    st.image([Image.open(uploaded_file), processed_image], caption=['Original Image', 'Processed Image'], use_column_width=True)