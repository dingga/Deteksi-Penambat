import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os

st.set_page_config(page_title="Monitor Penambat Rel ITB", layout="wide")

st.title("🛤️ Sistem Deteksi & Monitoring Penambat Rel")
st.write("Aplikasi ini mendeteksi kerusakan penambat secara real-time dari rekaman video.")

# 1. KONFIGURASI MODEL
# Gunakan nama file langsung, asumsikan best.pt ada di root GitHub Anda
MODEL_PATH = 'best.pt' 

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

model = load_model()

if model is None:
    st.error(f"❌ File model '{MODEL_PATH}' tidak ditemukan di repository GitHub Anda.")
    st.stop()

# 2. SIDEBAR & INPUT
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah Video Rekaman Rel (.mp4, .avi)", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Simpan file sementara untuk diproses OpenCV
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 3. LOGIKA ROI & GARIS (Sesuai kode tesis Anda)
    y_atas  = int(0.10 * height)
    y_bawah = int(0.70 * height)
    y_ref   = int(0.50 * height) # Garis hitung

    roi_points = np.array([
        [int(0.35 * width), y_bawah],
        [int(0.40 * width), y_atas],
        [int(0.60 * width), y_atas],
        [int(0.65 * width), y_bawah]
    ], np.int32)

    # Inisialisasi Data Tracking
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()
    missing_images = [] # Untuk galeri screenshot

    # Placeholder Tampilan
    col_vid, col_stat = st.columns([2, 1])
    frame_placeholder = col_vid.empty()
    stat_placeholder = col_stat.empty()
    gallery_header = st.empty()
    gallery_placeholder = st.container()

    # 4. PROSES DETEKSI
    if st.sidebar.button("Mulai Deteksi"):
        results = model.track(source=tfile.name, persist=True, imgsz=1024, stream=True, conf=0.15)
        
        for frame_idx, res in enumerate(results):
            frame = res.orig_img.copy()
            
            # Gambar ROI dan Garis Ref
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Cek ROI
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        # Logika Hitung saat melewati garis y_ref
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1
                            
                            # Capture jika Hilang
                            if final_label == "Hilang":
                                # Ambil potongan gambar penambat yang hilang (crop)
                                crop = frame[max(0, int(y1)):min(height, int(y2)), 
                                             max(0, int(x1)):min(width, int(x2))]
                                if crop.size > 0:
                                    missing_images.append({"id": tid, "img": crop})

                        # Gambar Box
                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 3)
            
            # Update Tampilan Real-time
            frame_placeholder.image(frame, channels="BGR", use_column_width=True)
            
            with stat_placeholder:
                st.subheader("📊 Statistik")
                for cls_name in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                    st.metric(label=cls_name, value=summary_counts[cls_name])
            
            # Update Galeri Foto Secara Periodik
            if missing_images:
                gallery_header.subheader(f"📸 Dokumentasi Penambat Hilang ({len(missing_images)})")
                with gallery_placeholder:
                    cols = st.columns(4)
                    for i, item in enumerate(missing_images[-4:]): # Tampilkan 4 terbaru
                        cols[i].image(item["img"], caption=f"ID: {item['id']}")

        st.success("✅ Pemrosesan selesai.")

