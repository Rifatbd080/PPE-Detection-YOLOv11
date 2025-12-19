import streamlit as st
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import cv2
import tempfile
import numpy as np
import yaml

# --- Updated Path to match your actual folder ---
MODEL_PATH = 'best_ppe_yolo11_model.pt'
DATA_YAML_PATH = '/content/mask-hairnet-gloves-1/data.yaml'

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

MODEL = load_model()
# Get class names directly from the model weights for 100% accuracy
NAMES = MODEL.names 

st.set_page_config(page_title="PPE Detector", layout="wide")
st.title("🛡️ PPE Object Detector")

# --- Sidebar ---
st.sidebar.header("Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.01, 1.0, 0.20)

st.sidebar.markdown("---")
st.sidebar.subheader("Model Class Mapping")
# This will show you exactly what index the model uses for 'glove' or 'hairnet'
st.sidebar.json(NAMES) 

count_display = st.sidebar.empty()

def process_frame(frame, conf_threshold):
    results = MODEL.predict(frame, conf=conf_threshold, verbose=False)[0]
    annotator = Annotator(frame.copy(), line_width=2)
    counts = {name: 0 for name in NAMES.values()}

    if results.boxes:
        for box in results.boxes:
            c = int(box.cls[0])
            label = f"{NAMES[c]} {box.conf[0]:.2f}"
            annotator.box_label(box.xyxy[0], label, color=colors(c, True))
            counts[NAMES[c]] += 1
            
    return annotator.result(), counts

# --- Main Logic ---
source_option = st.sidebar.radio("Select Source", ('Image Upload', 'Video Upload'))

if source_option == 'Image Upload':
    uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file:
        img = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        res_img, counts = process_frame(img, confidence)
        st.image(res_img, channels="BGR", use_container_width=True)
        with count_display.container():
            st.write("### 📊 Detection Counts")
            for k, v in counts.items(): st.write(f"**{k}:** `{v}`")

elif source_option == 'Video Upload':
    uploaded_file = st.file_uploader("Upload Video", type=['mp4', 'avi'])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
        st_frame = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            res_frame, counts = process_frame(frame, confidence)
            st_frame.image(res_frame, channels="BGR", use_container_width=True)
            with count_display.container():
                st.write("### 📊 Detection Counts")
                for k, v in counts.items(): st.write(f"**{k}:** `{v}`")
        cap.release()
