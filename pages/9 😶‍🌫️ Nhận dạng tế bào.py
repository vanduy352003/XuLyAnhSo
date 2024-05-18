import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np

# define load model function
def load_model():
    # load model
    model = YOLO(r"./utility/DetectBloodCell/model.pt")
    return model

def main():
    st.title("Blood Cell Detection")
    st.markdown(
        """
        <style>
        .uploader {
            display: block;
            width: 60%;
            padding: 20px;
            margin: auto;
            border: 2px dashed #ddd;
            border-radius: 8px;
            text-align: center;
        }
        .result-container {
            background-color: #f4f4f4;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    uploaded_image = st.file_uploader("Nhấp vào để tải ảnh lên", accept_multiple_files=False, type=['jpg', 'jpeg', 'png'])

    if uploaded_image is not None:
        if uploaded_image.type.startswith('image/'):
            model = load_model()

        # Hiển thị hình ảnh đầu vào
        input_image = Image.open(uploaded_image)
        st.image(input_image, caption='Ảnh đã tải', use_column_width=True)

        # Tạo nút "Predict" để thực hiện dự đoán khi được nhấn
        if st.button("Predict"):
            # Đảm bảo rằng dữ liệu hình ảnh được chuyển thành numpy array
            input_image_np = np.array(input_image)

            # Dự đoán và hiển thị hình ảnh đầu ra
            output_image, wbc, rbc = predict(model, input_image_np)

            # Hiển thị hình ảnh đầu ra
            st.image(output_image, caption='Ảnh sau khi xử lí', use_column_width=True)

            # Hiển thị kết quả trong một container với màu nền
            with st.container():
                st.markdown(f'<p style="color:blue; font-size:20px;">White Blood Cell: {wbc}</p>', unsafe_allow_html=True)
                st.markdown(f'<p style="color:blue; font-size:20px;">Red Blood Cell: {rbc}</p>', unsafe_allow_html=True)

def predict(model, image_data):
    wbc = 0
    rbc = 0
    results = model(image_data)
    output_image = results[0].plot()
    output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
    for box in results[0].boxes.cls:
        if np.array(box) == 0:
            rbc += 1
        else:
            wbc += 1
    return output_image, wbc, rbc



if __name__ == '__main__':
    main()
