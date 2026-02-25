import streamlit as st
import cv2
import pandas as pd
from ultralytics import YOLO
from collections import defaultdict, Counter
import tempfile
import os
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Monitoring", layout="wide")

st.title("🚊 Sistem Monitoring Penambat Rel (Tesis)")
st.sidebar.title("Pengaturan Aplikasi")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- SIDEBAR ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rel", type=["mp4", "avi", "mov"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.2)
process_button = st.sidebar.button("Mulai Analisis Video")

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    
    if process_button:
        cap = cv2.VideoCapture(tfile.name)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter()
        captured_images = []
        
        y_ref = int(0.50 * height)
        
        col1, col2 = st.columns([2, 1])
        st_frame = col1.empty()
        st_metrics = col2.empty()
        
        st.info("Sedang memproses video...")
        
        # BARIS YANG ERROR TADI (Baris 83):
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            cv2.line(frame, (0, y_ref), (width, y_ref), (255, 0, 0), 2)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    cx, cy = int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)
                    label = model.names[cls]
                    track_history[tid].append(label)

                    if cy > y_ref and tid not in counted_ids:
                        counted_ids.add(tid)
                        final_label = Counter(track_history[tid]).most_common(1)[0][0]
                        summary_counts[final_label] += 1
                        
                        if "hilang" in final_label.lower():
                            annotated_frame = res.plot()
                            img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                            captured_images.append((img_rgb, f"ID:{tid} - Frame:{frame_idx}"))

            # Tampilan Visual
            frame_visual = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            st_frame.image(frame_visual, channels="RGB", use_container_width=True)
            
            with st_metrics:
                st.subheader("📊 Statistik Aset")
                st.metric("Total Terhitung", len(counted_ids))
                for lbl, count in summary_counts.items():
                    st.write(f"**{lbl}**: {count}")

        cap.release()
        st.success("Analisis Selesai!")

        if captured_images:
            st.markdown("---")
            st.subheader("🚨 Temuan Penambat Hilang")
            cols = st.columns(3)
            for i, (img, cap_text) in enumerate(captured_images):
                with cols[i % 3]:
                    st.image(img, caption=cap_text, use_container_width=True)
