import streamlit as st
from PIL import Image
import cv2 as cv
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Nhận dạng trái cây",
    page_icon="🍓",
)

st.title('Nhận dạng trái cây')

# Load model and classes
model = 'utility/RecognitionFruit/trai_cay.onnx'

classes = ['buoi', 'cam_vang', 'chom_chom', 'dau', 'mang_cut']

# Load a network
net = cv.dnn.readNet(model)
net.setPreferableBackend(0)
net.setPreferableTarget(0)
outNames = net.getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

mywidth = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = net.getLayerNames()
    lastLayerId = net.getLayerId(layerNames[-1])
    lastLayer = net.getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)
            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return

img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "tif", "bmp", "gif", "png"])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    frame = np.array(image)
    frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button('Detect'):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        inpWidth = mywidth if mywidth else frameWidth
        inpHeight = myheight if myheight else frameHeight
        blob = cv.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv.CV_8U)

        net.setInput(blob, scalefactor=scale, mean=mean)
        if net.getLayer(0).outputNameToIndex('im_info') != -1:
            frame = cv.resize(frame, (inpWidth, inpHeight))
            net.setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

        outs = net.forward(outNames)
        postprocess(frame, outs)

        color_converted = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        st.image(pil_image, caption="Processed Image", use_column_width=True)
