import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Detector", layout="wide")
st.title("Sistem Deteksi Penambat Kereta Api (ROI Sinkron)")

# --- INISIALISASI FOLDER CAPTURE ---
CAPTURE_DIR = "captured_missing"
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt')
    if not os.path.exists(model_path):
        st.error("File 'best.pt' tidak ditemukan di root directory!")
        st.stop()
    return YOLO(model_path)

model = load_model()

# --- SESSION STATE ---
if 'summary_counts' not in st.session_state:
    st.session_state.summary_counts = Counter()
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()
if 'track_history' not in st.session_state:
    st.session_state.track_history = defaultdict(list)
if 'captured_files' not in st.session_state:
    st.session_state.captured_files = []

# --- SIDEBAR ---
st.sidebar.title("Kontrol Deteksi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)
source_option = st.sidebar.selectbox("Pilih Sumber Input", ("Upload Video", "Kamera Real-time"))

if st.sidebar.button("Reset Data & Folder"):
    st.session_state.summary_counts = Counter()
    st.session_state.counted_ids = set()
    st.session_state.track_history = defaultdict(list)
    st.session_state.captured_files = []
    for f in os.listdir(CAPTURE_DIR):
        os.remove(os.path.join(CAPTURE_DIR, f))
    st.rerun()

def process_frame(frame, model, roi_points, y_ref, width):
    # Tracking menggunakan YOLOv8
    results = model.track(frame, persist=True, conf=conf_threshold, imgsz=1024, verbose=False)
    
    # Visual ROI Melayang (Sesuai script final.py)
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

    # Garis Hitung (Sesuai script final.py)
    x_kiri_garis = int(0.28 * width)
    x_kanan_garis = int(0.72 * width)
    cv2.line(frame, (x_kiri_garis, y_ref), (x_kanan_garis, y_ref), (255, 0, 0), 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label = model.names[cls]

            # Cek jika titik tengah berada dalam ROI
            if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                st.session_state.track_history[tid].append(label)

                # Logika Crossing Line (y_ref = 0.6 * height)
                if cy > y_ref and tid not in st.session_state.counted_ids:
                    st.session_state.counted_ids.add(tid)
                    final_label = Counter(st.session_state.track_history[tid]).most_common(1)[0][0]
                    st.session_state.summary_counts[final_label] += 1
                    
                    # Auto-Capture jika 'Hilang'
                    if final_label == "Hilang":
                        filepath = os.path.join(CAPTURE_DIR, f"MISSING_{tid}_{int(time.time())}.jpg")
                        cv2.imwrite(filepath, results[0].plot())
                        st.session_state.captured_files.append(filepath)

                # Visualisasi Bounding Box
                color = (0, 255, 0) if tid in st.session_state.counted_ids else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# --- MAIN LOOP ---
col1, col2 = st.columns([3, 1])
st_frame = col1.empty()
rekap_placeholder = col2.empty()

video_file = None
if source_option == "Upload Video":
    video_file = st.file_uploader("Upload video .mp4", type=['mp4', 'avi', 'mov'])

if video_file or source_option == "Kamera Real-time":
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # SINKRONISASI ROI MELAYANG DARI final.py
        y_atas  = int(0.35 * height) 
        y_bawah = int(0.85 * height) 
        y_ref   = int(0.6 * height) 
        
        roi_points = np.array([
            [int(0.25 * width), y_bawah], # Kiri bawah
            [int(0.42 * width), y_atas],  # Kiri atas
            [int(0.58 * width), y_atas],  # Kanan atas
            [int(0.75 * width), y_bawah]  # Kanan bawah
        ], np.int32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed = process_frame(frame, model, roi_points, y_ref, width)
            st_frame.image(processed, channels="BGR", use_container_width=True)

            with rekap_placeholder.container():
                st.write("### Rekapitulasi")
                # Urutan sesuai dashboard Anda
                for c in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                    count = st.session_state.summary_counts[c]
                    color = "red" if c == "Hilang" else "green"
                    st.markdown(f"**{c}:** :{color}[{count} unit]")
                
                if st.session_state.captured_files:
                    st.write("---")
                    st.image(st.session_state.captured_files[-1], caption="Terakhir Dicapture")

        cap.release()
