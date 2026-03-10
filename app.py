import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import Counter, defaultdict
import tempfile
import os

st.set_page_config(page_title="Deteksi Penambat Kereta", layout="wide")

st.title("🛤️ Sistem Deteksi & Monitoring Penambat")
st.sidebar.header("Konfigurasi Model")

# 1. Load Model
MODEL_PATH = 'best.pt' # Pastikan file model ada di direktori yang sama
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# 2. Upload Video
uploaded_file = st.sidebar.file_uploader("Unggah Video Rel", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st.info("Sedang memproses video... Harap tunggu.")
    
    # Progress Bar & Placeholders
    frame_placeholder = st.empty()
    col1, col2 = st.columns([1, 1])
    recap_placeholder = col1.empty()
    gallery_placeholder = col2.empty()
    
    # Inisialisasi Data (Mirip kode Python Anda)
    counted_ids = set()
    summary_counts = Counter()
    missing_images = []
    y_ref = 500 # Sesuaikan dengan resolusi video

    # 3. Loop Pemrosesan
    results = model.track(source=tfile.name, persist=True, stream=True, conf=0.15)
    
    for res in results:
        frame = res.plot() # Ambil frame yang sudah di-annotate
        
        if res.boxes is not None and res.boxes.id is not None:
            ids = res.boxes.id.cpu().numpy().astype(int)
            clss = res.boxes.cls.cpu().numpy().astype(int)
            boxes = res.boxes.xyxy.cpu().numpy()

            for box, tid, cls in zip(boxes, ids, clss):
                label = model.names[cls]
                
                # Logika hitung & capture jika "Hilang"
                if label == "Hilang" and tid not in counted_ids:
                    counted_ids.add(tid)
                    summary_counts[label] += 1
                    
                    # Simpan screenshot ke list
                    crop = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    if crop.size > 0:
                        missing_images.append((tid, frame.copy())) # Simpan full frame atau crop

        # Update Tampilan Web Secara Real-time
        frame_placeholder.image(frame, channels="BGR", use_column_width=True)
        
        with recap_placeholder.container():
            st.subheader("📊 Rekapitulasi")
            st.write(f"Penambat Hilang: {summary_counts['Hilang']}")
            # Tampilkan kelas lainnya sesuai Counter

    st.success("Proses Selesai!")
    
    # 4. Menampilkan Screenshot Penambat Hilang
    st.divider()
    st.subheader("📸 Dokumentasi Penambat Hilang")
    if missing_images:
        cols = st.columns(3)
        for idx, (tid, img) in enumerate(missing_images):
            with cols[idx % 3]:
                st.image(img, caption=f"ID: {tid} - Terdeteksi Hilang", channels="BGR")