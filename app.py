import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- PENETAPAN SISTEM (Mesti di atas sekali) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"

st.set_page_config(page_title="Sistem Deteksi Penambat - ITB", layout="wide")

st.title("🚉 Dashboard Deteksi & Dokumentasi Penambat")
st.markdown("Muat naik video rel untuk mengesan dan merakam lokasi penambat yang **Hilang**.")

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt') # Pastikan nama fail di GitHub ialah best.pt

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"Fail model 'best.pt' tidak dijumpai di direktori utama.")
    st.stop()

# --- SIDEBAR ---
st.sidebar.header("Konfigurasi")
uploaded_video = st.sidebar.file_uploader("Pilih Video Rel (MP4)", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garisan Rujukan (Sesuai parameter tesis anda)
    y_ref = int(0.6 * height)
    roi = np.array([
        [int(0.25 * width), int(0.85 * height)], 
        [int(0.42 * width), int(0.35 * height)],
        [int(0.58 * width), int(0.35 * height)], 
        [int(0.75 * width), int(0.85 * height)]
    ], np.int32)

    # Layout Dashboard
    col_vid, col_data = st.columns([2, 1])
    frame_placeholder = col_vid.empty()
    stats_placeholder = col_data.empty()

    st.divider()
    # Placeholder Galeri
    gallery_header = st.empty()
    gallery_placeholder = st.empty()

    if st.sidebar.button("Mula Analisis"):
        track_history = defaultdict(list)
        counted = set()
        summary_counts = Counter()
        captured_images = [] 

        # YOLO Tracking (Streaming Mode)
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Lukis ROI & Garisan pada Video
            cv2.polylines(frame, [roi], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28 * width), y_ref), (int(0.72 * width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    # Semak jika dalam ROI
                    if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                        track_history[tid].append(label)

                        # Logika Counting melepasi garisan
                        if cy > y_ref and tid not in counted:
                            counted.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            # Capture Screenshot jika "Hilang"
                            if "Hilang" in final_label:
                                annotated_frame = res.plot() # Ambil frame dengan box YOLO
                                snapshot_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                                captured_images.append({"img": snapshot_rgb, "id": tid})
                                
                                # Kemaskini Galeri Real-time (4 terbaru)
                                with gallery_placeholder.container():
                                    gallery_header.subheader(f"📸 Screenshot Penambat Hilang ({len(captured_images)} Kes)")
                                    cols = st.columns(4)
                                    items_to_show = captured_images[-4:]
                                    for i, item in enumerate(reversed(items_to_show)):
                                        with cols[i]:
                                            st.image(item["img"], caption=f"ID: {item['id']}", use_container_width=True)

            # Update Frame Video
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Update Statistik
            with stats_placeholder.container():
                st.subheader("📊 Statistik Deteksi")
                st.metric("Total Unit Diproses", sum(summary_counts.values()))
                st.table(pd.DataFrame({
                    "Kategori": summary_counts.keys(), 
                    "Jumlah": summary_counts.values()
                }))

        # Tampilkan Galeri Penuh Selepas Selesai
        st.success("✅ Analisis Selesai!")
        if captured_images:
            st.divider()
            st.subheader("📁 Semua Dokumentasi Lokasi Hilang")
            full_cols = st.columns(5)
            for idx, item in enumerate(captured_images):
                with full_cols[idx % 5]:
                    st.image(item["img"], caption=f"ID: {item['id']}", use_container_width=True)

    cap.release()
    os.unlink(tfile.name)
else:
    st.info("Sila muat naik video untuk memulakan sistem.")
