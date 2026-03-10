import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")

st.title("🚉 Sistem Deteksi & Dokumentasi Penambat Rel")
st.markdown("Hasil deteksi penambat **Hilang** akan otomatis muncul di galeri bawah secara real-time.")

# --- INISIALISASI FOLDER DOKUMENTASI ---
CAPTURE_FOLDER = 'deteksi_hilang'
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.header("Konfigurasi")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    try: return YOLO(path)
    except: return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"❌ File '{MODEL_PATH}' tidak ditemukan.")
    st.stop()

uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4", "mov", "avi"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Hitung
    y_atas, y_bawah = int(0.35 * height), int(0.85 * height)
    roi_points = np.array([
        [int(0.25 * width), y_bawah], [int(0.42 * width), y_atas],
        [int(0.58 * width), y_atas], [int(0.75 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.6 * height)

    # Layout Utama
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    # Area Galeri di Bawah
    st.divider()
    st.subheader("📸 Galeri Bukti Penambat Hilang")
    gallery_placeholder = st.empty() # Tempat untuk menampilkan list foto
    
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()
    captured_images = [] # Menyimpan array gambar untuk ditampilkan

    if st.sidebar.button("Mulai Analisis"):
        # Bersihkan folder lama
        for f in os.listdir(CAPTURE_FOLDER): os.remove(os.path.join(CAPTURE_FOLDER, f))
        
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Overlay Visual
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28 * width), y_ref), (int(0.72 * width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)

                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            # Tampilkan di Galeri jika "Hilang"
                            if "Hilang" in final_label:
                                snapshot = res.plot()
                                snapshot_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                                captured_images.append({
                                    "img": snapshot_rgb,
                                    "title": f"ID: {tid} - Frame: {frame_idx}"
                                })
                                
                                # Update Galeri secara real-time (tampilkan 4 foto terbaru)
                                with gallery_placeholder.container():
                                    cols = st.columns(4)
                                    for i, item in enumerate(reversed(captured_images[-4:])):
                                        cols[i].image(item["img"], caption=item["title"], use_container_width=True)

                        color = (0, 0, 255) if "Hilang" in label else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with stats_placeholder.container():
                st.table(pd.DataFrame({"Kategori": summary_counts.keys(), "Jumlah": summary_counts.values()}))

    cap.release()
    os.unlink(tfile.name)
