import streamlit as st
import os

# --- PROTEKSI SISTEM ---
# Memaksa OpenCV berjalan dalam mode offscreen (mencegah error libGL)
os.environ["QT_QPA_PLATFORM"] = "offscreen" 

try:
    import cv2
    import numpy as np
    import pandas as pd
    import tempfile
    from collections import defaultdict, Counter
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.info("Pastikan requirements.txt Anda HANYA berisi 'opencv-python-headless'.")
    st.stop()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")
st.title("🚉 Dashboard Deteksi & Galeri Penambat")
st.markdown("Hasil deteksi penambat **Hilang** akan otomatis muncul di galeri bawah secara real-time.")

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    try:
        return YOLO(path)
    except:
        return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"File 'best.pt' tidak ditemukan di {MODEL_PATH}")
    st.stop()
else:
    st.sidebar.success("✅ Model Berhasil Dimuat")

# --- SIDEBAR & UPLOAD ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.15)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Hitung (Parameter Tesis)
    y_ref = int(0.6 * height)
    roi = np.array([
        [int(0.25 * width), int(0.85 * height)], 
        [int(0.42 * width), int(0.35 * height)],
        [int(0.58 * width), int(0.35 * height)], 
        [int(0.75 * width), int(0.85 * height)]
    ], np.int32)

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    st.divider()
    st.subheader("📸 Galeri Bukti Penambat Hilang")
    gallery_container = st.container()

    if st.sidebar.button("Mulai Analisis"):
        track_history = defaultdict(list)
        counted = set()
        summary_counts = Counter()
        captured_images = [] 

        # YOLO Tracking
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Visualisasi ROI & Line
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

                    if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                        track_history[tid].append(label)

                        if cy > y_ref and tid not in counted:
                            counted.add(tid)
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            if "Hilang" in final_label:
                                snapshot = res.plot()
                                snapshot_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                                captured_images.append({"img": snapshot_rgb, "txt": f"ID:{tid}"})
                                
                                # Update Galeri (4 foto terbaru)
                                with gallery_container:
                                    cols = st.columns(4)
                                    for i, item in enumerate(reversed(captured_images[-4:])):
                                        cols[i].image(item["img"], caption=item["txt"], use_container_width=True)

            # Update Live Video & Statistik
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            with stats_placeholder.container():
                st.table(pd.DataFrame({"Kategori": summary_counts.keys(), "Unit": summary_counts.values()}))

    cap.release()
    os.unlink(tfile.name)
