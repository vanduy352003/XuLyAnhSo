import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Trang Chủ",
    page_icon="👋",
)

logo = Image.open('images\logo.png')
st.image(logo, width=800)

st.markdown(
    """
    ### Website Xử Lý Ảnh Số
    - Thực hiện bởi: Lương Chin Du và Trần Văn Duy
    - Giảng viên hướng dẫn: ThS. Trần Tiến Đức
    """
)

st.markdown("""### Video giới thiệu về Website""")
st.markdown("""[Video giới thiệu website Xử Lý Ảnh Số]()""")