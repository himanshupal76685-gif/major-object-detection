import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Major Object Detection Project",
    layout="wide"
)

st.title("🔍 AI-Based Object Detection System")
st.write("Major Project | YOLOv8 + Streamlit")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

# ---------------- SIDEBAR ----------------
st.sidebar.title("📌 Detection Options")
option = st.sidebar.radio(
    "Choose Detection Type",
    ("Image Detection", "Video Detection", "Webcam Detection (Local)")
)

# =================================================
# IMAGE DETECTION
# =================================================
if option == "Image Detection":
    st.header("🖼 Image Object Detection")

    uploaded_image = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        img_array = np.array(image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)

        if st.button("Detect Objects"):
            with st.spinner("Detecting..."):
                results = model(img_array)
                detected_img = results[0].plot()

            with col2:
                st.image(
                    detected_img,
                    caption="Detected Objects",
                    use_container_width=True
                )

# =================================================
# VIDEO DETECTION
# =================================================
elif option == "Video Detection":
    st.header("🎥 Video Object Detection")

    uploaded_video = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        st.info("Processing video... please wait")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)
            frame = results[0].plot()
            stframe.image(frame, channels="BGR")

        cap.release()
        st.success("Video processing completed")

# =================================================
# WEBCAM DETECTION (LOCAL ONLY)
# =================================================
elif option == "Webcam Detection (Local)":
    st.header("📷 Live Webcam Object Detection")

    st.warning(
        "⚠ Webcam detection works on LOCAL system only "
        "(not supported on Streamlit Cloud)."
    )

    run = st.checkbox("Start Webcam")

    FRAME_WINDOW = st.image([])
    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Unable to access webcam")
            break

        results = model(frame)
        frame = results[0].plot()
        FRAME_WINDOW.image(frame, channels="BGR")

    camera.release()
