import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Detector", layout="wide")
st.title("Sistem Deteksi Penambat Kereta Api (YOLOv8)")
st.sidebar.title("Pengaturan Deteksi")

# --- LOAD MODEL (TANPA FOLDER) ---
@st.cache_resource
def load_model():
    # Mengambil path absolut agar aman di Streamlit Cloud
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt') # Langsung panggil best.pt
    
    if not os.path.exists(model_path):
        st.error(f"File 'best.pt' tidak ditemukan di root directory!")
        st.stop()
        
    model = YOLO(model_path)
    return model

model = load_model()

# --- SIDEBAR SETTINGS ---
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)
source_option = st.sidebar.selectbox("Pilih Sumber Input", ("Upload Video", "Kamera Real-time"))

# --- LOGIKA DASHBOARD & TRACKING ---
# Inisialisasi state agar tidak reset setiap frame
if 'track_history' not in st.session_state:
    st.session_state.track_history = defaultdict(list)
    st.session_state.counted_ids = set()
    st.session_state.summary_counts = Counter()

def process_frame(frame, model, roi_points, y_ref):
    height, width = frame.shape[:2]
    
    # Overlay Visual ROI
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

    # Garis Hitung
    cv2.line(frame, (int(0.28 * width), y_ref), (int(0.72 * width), y_ref), (255, 0, 0), 3)

    # Inference (Tracking)
    results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                label = model.names[cls]
                st.session_state.track_history[tid].append(label)

                if cy > y_ref and tid not in st.session_state.counted_ids:
                    st.session_state.counted_ids.add(tid)
                    final_label = Counter(st.session_state.track_history[tid]).most_common(1)[0][0]
                    st.session_state.summary_counts[final_label] += 1

                color = (0, 255, 0) if tid in st.session_state.counted_ids else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# --- MAIN DISPLAY ---
col1, col2 = st.columns([3, 1])
with col1:
    st_frame = st.empty()
with col2:
    st.subheader("Hasil Perhitungan")
    rekap_placeholder = st.empty()

# Penanganan Video Input
cap = None
if source_option == "Upload Video":
    video_file = st.file_uploader("Upload video .mp4", type=['mp4', 'avi', 'mov'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
else:
    cap = cv2.VideoCapture(0)

if cap:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_ref = int(0.6 * height)
    roi_points = np.array([[int(0.25*width), int(0.85*height)], [int(0.42*width), int(0.35*height)], 
                           [int(0.58*width), int(0.35*height)], [int(0.75*width), int(0.85*height)]], np.int32)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        processed = process_frame(frame, model, roi_points, y_ref)
        st_frame.image(processed, channels="BGR", use_container_width=True)

        with rekap_placeholder.container():
            for cls_name in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                count = st.session_state.summary_counts[cls_name]
                st.markdown(f"**{cls_name}:** :{'red' if cls_name == 'Hilang' else 'green'}[{count} unit]")

    cap.release()
