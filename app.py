import streamlit as st
import os

# --- PROTEKSI IMPORT UNTUK STREAMLIT CLOUD ---
try:
    import cv2
    import numpy as np
    import pandas as pd
    import tempfile
    from collections import defaultdict, Counter
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.info("Pastikan requirements.txt Anda menggunakan 'opencv-python-headless' dan bukan 'opencv-python'.")
    st.stop()

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")

st.title("🚉 Sistem Deteksi & Galeri Penambat Rel")
st.markdown("Hasil deteksi penambat **Hilang** akan langsung muncul di galeri bawah.")

# --- LOAD MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    try: return YOLO(path)
    except: return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"❌ File '{MODEL_PATH}' tidak ditemukan di GitHub.")
    st.stop()

# --- SIDEBAR & UPLOAD ---
st.sidebar.header("Konfigurasi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)
uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Garis Hitung (Parameter Tesis)
    y_ref = int(0.6 * height)
    roi_points = np.array([
        [int(0.25 * width), int(0.85 * height)], 
        [int(0.42 * width), int(0.35 * height)],
        [int(0.58 * width), int(0.35 * height)], 
        [int(0.75 * width), int(0.85 * height)]
    ], np.int32)

    # Layout: Video di kiri, Statistik di kanan
    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()

    # Area Galeri di bawah
    st.divider()
    st.subheader("📸 Galeri Bukti Penambat Hilang")
    gallery_container = st.container()
    
    if st.sidebar.button("Mulai Analisis"):
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter()
        captured_images = [] # List untuk galeri

        # Loop deteksi
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Draw ROI & Line
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

                            # Tampilkan di galeri jika labelnya "Hilang"
                            if "Hilang" in final_label:
                                snapshot = res.plot() # Ambil gambar dengan box YOLO
                                captured_images.append({
                                    "img": cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB),
                                    "title": f"ID: {tid} | Frame: {frame_idx}"
                                })

                        # Gambar box di layar live
                        color = (0, 0, 255) if "Hilang" in label else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            # Update tampilan video live
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

            # Update statistik di samping
            with stats_placeholder.container():
                st.table(pd.DataFrame({"Kategori": summary_counts.keys(), "Unit": summary_counts.values()}))

            # Update galeri (tampilkan 4 foto terbaru)
            if captured_images:
                with gallery_container:
                    cols = st.columns(4)
                    # Ambil 4 foto terakhir untuk ditampilkan di grid
                    for i, item in enumerate(reversed(captured_images[-4:])):
                        cols[i].image(item["img"], caption=item["title"], use_container_width=True)

    cap.release()
    os.unlink(tfile.name)
