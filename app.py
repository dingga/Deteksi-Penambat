import streamlit as st
import cv2
import numpy as np
import tempfile
from ultralytics import YOLO
from collections import defaultdict, Counter

st.title("Sistem Deteksi Penambat Rel")

# 1. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt') # Pastikan file ini ada di repo GitHub Anda

model = load_model()

# 2. Upload Video
uploaded_file = st.file_uploader("Unggah Video Penambat (.mp4, .avi)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Simpan file sementara karena OpenCV butuh path file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty() # Placeholder untuk video
    
    # Inisialisasi Data
    summary_counts = Counter()
    counted_ids = set()
    track_history = defaultdict(list)
    
    # Parameter ROI (Bisa dibuat slider jika mau)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y_ref = int(0.6 * height)

    stop_button = st.button("Stop Proses")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or stop_button:
            break

        # Logika Tracking YOLO
        results = model.track(frame, persist=True, conf=0.15, verbose=False)
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, tid, cls in zip(boxes, ids, clss):
                x1, y1, x2, y2 = box
                cy = int((y1 + y2) / 2)

                # Logika Hitung Sederhana (Garis Horizontal)
                if cy > y_ref and tid not in counted_ids:
                    counted_ids.add(tid)
                    label = model.names[cls]
                    summary_counts[label] += 1

                # Gambar Box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Tampilkan ke Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st_frame.image(frame_rgb, channels="RGB", use_column_width=True)

        # Dashboard Statistik di Sidebar
        st.sidebar.title("Hasil Deteksi")
        for name, count in summary_counts.items():
            st.sidebar.write(f"**{name}**: {count}")

    cap.release()
    st.success("Proses Selesai!")
