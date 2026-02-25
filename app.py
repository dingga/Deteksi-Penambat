import streamlit as st
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os
from PIL import Image

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Monitoring", layout="wide")

# Perbaikan parameter: gunakan unsafe_allow_html=True
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚊 Sistem Monitoring Penambat Rel (Tesis)")
st.sidebar.title("Pengaturan Analisis")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- 3. SIDEBAR: INPUT ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rekaman Rel", type=["mp4", "avi", "mov"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.20)
process_button = st.sidebar.button("🚀 Jalankan Deteksi")

# --- 4. LOGIKA UTAMA ---
if uploaded_video is not None:
    suffix = os.path.splitext(uploaded_video.name)[1]
    
    if process_button:
        # Penanganan file sementara yang lebih aman
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter()
        captured_faults = []
        
        y_ref = int(0.50 * height)
        
        col_vid, col_stat = st.columns([2, 1])
        st_frame = col_vid.empty()
        st_metrics = col_stat.empty()
        
        st.info("Sedang memproses video...")

        # Menjalankan tracking
        results = model.track(source=temp_path, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Gambar garis hitung
            cv2.line(frame, (0, y_ref), (width, y_ref), (255, 0, 0), 2)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    label = model.names[cls]
                    track_history[tid].append(label)

                    # Logika hitung asli Anda
                    if cy > y_ref and tid not in counted_ids:
                        counted_ids.add(tid)
                        final_label = Counter(track_history[tid]).most_common(1)[0][0]
                        summary_counts[final_label] += 1
                        
                        if "hilang" in final_label.lower():
                            annotated_res = res.plot()
                            res_rgb = cv2.cvtColor(annotated_res, cv2.COLOR_BGR2RGB)
                            captured_faults.append((Image.fromarray(res_rgb), f"ID:{tid} | Frame:{frame_idx}"))

            # Update tampilan real-time
            frame_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            with st_metrics:
                st.subheader("📊 Inventaris Aset")
                st.metric("Total Terhitung", len(counted_ids))
                for lbl, count in summary_counts.items():
                    st.write(f"✅ **{lbl}**: {count} unit")

        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.success("Analisis Selesai!")

        # Galeri temuan
        if captured_faults:
            st.divider()
            st.subheader("🚨 Galeri Penambat Hilang")
            cols = st.columns(3)
            for i, (img, caption) in enumerate(captured_faults):
                with cols[i % 3]:
                    st.image(img, caption=caption, use_container_width=True)
