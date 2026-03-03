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
st.title("Sistem Deteksi & Auto-Capture Penambat Hilang")

# --- INISIALISASI FOLDER CAPTURE ---
# Folder ini akan dibuat di server Streamlit Cloud (temporary)
CAPTURE_DIR = "captured_missing"
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt')
    if not os.path.exists(model_path):
        st.error("File 'best.pt' tidak ditemukan!")
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
st.sidebar.title("Kontrol & Dokumentasi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.20)
source_option = st.sidebar.selectbox("Pilih Sumber Input", ("Upload Video", "Kamera Real-time"))

if st.sidebar.button("Reset Semua Data"):
    st.session_state.summary_counts = Counter()
    st.session_state.counted_ids = set()
    st.session_state.track_history = defaultdict(list)
    st.session_state.captured_files = []
    # Hapus file lama di folder
    for f in os.listdir(CAPTURE_DIR):
        os.remove(os.path.join(CAPTURE_DIR, f))
    st.rerun()

def process_frame(frame, model, roi_points, y_ref):
    results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)
    
    # Overlay Visual
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
    cv2.line(frame, (0, y_ref), (frame.shape[1], y_ref), (255, 0, 0), 2)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label = model.names[cls]

            # Gambar Box Kuning untuk semua deteksi
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # Logika Hitung & Capture dalam ROI
            if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                st.session_state.track_history[tid].append(label)

                if cy > y_ref and tid not in st.session_state.counted_ids:
                    st.session_state.counted_ids.add(tid)
                    final_label = Counter(st.session_state.track_history[tid]).most_common(1)[0][0]
                    st.session_state.summary_counts[final_label] += 1
                    
                    # FITUR AUTO CAPTURE: Jika label adalah 'Hilang'
                    if final_label == "Hilang":
                        timestamp = time.strftime("%H%M%S")
                        filename = f"MISSING_ID_{tid}_{timestamp}.jpg"
                        filepath = os.path.join(CAPTURE_DIR, filename)
                        
                        # Simpan frame asli (annotated)
                        annotated_frame = results[0].plot()
                        cv2.imwrite(filepath, annotated_frame)
                        st.session_state.captured_files.append(filepath)
    
    return frame

# --- TAMPILAN UTAMA ---
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
        y_ref = int(0.6 * height)
        roi_points = np.array([[int(0.1*width), height], [int(0.35*width), int(0.3*height)], 
                               [int(0.65*width), int(0.3*height)], [int(0.9*width), height]], np.int32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed = process_frame(frame, model, roi_points, y_ref)
            st_frame.image(processed, channels="BGR", use_container_width=True)

            with rekap_placeholder.container():
                st.write("### Rekapitulasi")
                for c in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                    count = st.session_state.summary_counts[c]
                    st.metric(label=c, value=f"{count} unit")
                
                if st.session_state.captured_files:
                    st.write("---")
                    st.write(f"📸 **{len(st.session_state.captured_files)} Capture Hilang**")
                    # Tampilkan gambar terakhir yang dicapture
                    last_img = st.session_state.captured_files[-1]
                    st.image(last_img, caption="Deteksi Hilang Terakhir")

        cap.release()
