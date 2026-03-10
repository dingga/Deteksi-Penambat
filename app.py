import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI SISTEM ---
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")
st.title("🚉 Dashboard Deteksi & Galeri Penambat")

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt') # Sesuaikan nama file di GitHub

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(MODEL_PATH)

if model is None:
    st.error("File model 'best.pt' tidak ditemukan di repositori.")
    st.stop()

# --- SIDEBAR & UPLOAD ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI Trapezoid (Sesuai logika tesis Anda)
    y_atas, y_bawah = int(0.10 * height), int(0.70 * height)
    roi_points = np.array([
        [int(0.35 * width), y_bawah], [int(0.40 * width), y_atas],
        [int(0.60 * width), y_atas], [int(0.65 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.5 * height) # Garis Hitung

    col_vid, col_stat = st.columns([2, 1])
    frame_win = col_vid.empty()
    table_win = col_stat.empty()

    st.divider()
    st.subheader("📸 Dokumentasi Otomatis Penambat Hilang")
    gallery_placeholder = st.empty()

    if st.sidebar.button("Mulai Analisis"):
        track_history = defaultdict(list)
        counted = set()
        summary_counts = Counter()
        captured_images = [] 

        results = model.track(source=tfile.name, persist=True, imgsz=1024, stream=True, conf=conf_threshold)

        for res in results:
            img = res.orig_img
            cv2.polylines(img, [roi_points], True, (0, 255, 0), 2)
            cv2.line(img, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                for box, tid, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int)):
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        if cy > y_ref and tid not in counted:
                            counted.add(tid)
                            final_lbl = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_lbl] += 1
                            
                            if final_lbl == "Hilang":
                                # Capture snapshot rapi dengan bounding box
                                annotated = res.plot()
                                captured_images.append({
                                    "img": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                                    "txt": f"ID:{tid}"
                                })
                                
                                # Update Galeri (4 terbaru)
                                with gallery_placeholder.container():
                                    cols = st.columns(4)
                                    for i, item in enumerate(reversed(captured_images[-4:])):
                                        cols[i].image(item["img"], caption=item["txt"], use_container_width=True)

            frame_win.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_container_width=True)
            table_win.table(pd.DataFrame({"Kategori": summary_counts.keys(), "Unit": summary_counts.values()}))

    cap.release()
    os.unlink(tfile.name)
