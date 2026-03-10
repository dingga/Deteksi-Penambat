import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict, Counter
from ultralytics import YOLO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Kereta API", layout="wide")

st.title("🚉 Sistem Deteksi & Counting Penambat Rel")
st.markdown("Aplikasi ini mendeteksi komponen penambat (E-Clip, DE-Clip, KA-Clip) dan mengidentifikasi penambat yang **Hilang** menggunakan YOLOv8.")

# --- SIDEBAR: KONFIGURASI MODEL ---
st.sidebar.header("Konfigurasi")
# Gunakan model default atau upload (di sini diasumsikan file .pt ada di path yang benar)
MODEL_PATH = 'best.pt' # Sesuaikan dengan lokasi model Anda
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

@st.cache_resource
def load_model(path):
    try:
        return YOLO(path)
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_model(MODEL_PATH)

# --- UPLOAD VIDEO ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    # Simpan file sementara
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # ROI Settings (Mengikuti logika script asli Anda)
    y_atas = int(0.35 * height)
    y_bawah = int(0.85 * height)
    roi_points = np.array([
        [int(0.25 * width), y_bawah],
        [int(0.42 * width), y_atas],
        [int(0.58 * width), y_atas],
        [int(0.75 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.6 * height)

    # UI Layout: Video di kiri, Statistik di kanan
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()
    
    # Data Struktur
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()

    if st.sidebar.button("Mulai Deteksi"):
        # Jalankan tracking
        results = model.track(source=tfile.name, persist=True, imgsz=1024, stream=True, conf=conf_threshold)

        for res in results:
            frame = res.orig_img
            
            # Overlay ROI (Visualisasi Area Hijau)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            
            # Garis Hitung Biru
            cv2.line(frame, (int(0.28 * width), y_ref), (int(0.72 * width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                    # Logika ROI & Counting
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                        # Visualisasi Bounding Box
                        color = (0, 255, 0) if tid in counted_ids else (0, 255, 255)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid} {model.names[cls]}", (int(x1), int(y1)-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update Frame di Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            # Update Statistik di Sidebar/Col2 secara dinamis
            with stats_placeholder.container():
                st.subheader("📊 Hasil Perhitungan")
                st.write(f"**Total Aset:** {sum(summary_counts.values())}")
                
                # Buat tabel hasil
                df_stats = pd.DataFrame({
                    "Jenis Aset": summary_counts.keys(),
                    "Jumlah": summary_counts.values()
                })
                st.table(df_stats)
                
                if summary_counts["Hilang"] > 0:
                    st.warning(f"⚠️ Terdeteksi {summary_counts['Hilang']} penambat hilang!")

    cap.release()
else:
    st.info("Silakan upload video melalui sidebar untuk memulai.")
