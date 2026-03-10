import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
from collections import defaultdict, Counter
from ultralytics import YOLO

# Dashboard Tesis Master
st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")
st.title("🚉 Dashboard Deteksi & Galeri Penambat")

# Load Model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    return YOLO(path)

model = load_model(MODEL_PATH)

# UI Sidebar
uploaded_video = st.sidebar.file_uploader("Upload Video Rel", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence", 0.0, 1.0, 0.15)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI & Line
    y_ref = int(0.6 * h)
    roi = np.array([[int(0.25*w), int(0.85*h)], [int(0.42*w), int(0.35*h)],
                    [int(0.58*w), int(0.35*h)], [int(0.75*w), int(0.85*h)]], np.int32)

    col1, col2 = st.columns([2, 1])
    frame_win = col1.empty()
    table_win = col2.empty()

    st.divider()
    st.subheader("📸 Galeri Penambat Hilang")
    galeri_container = st.container()

    if st.sidebar.button("Mulai Analisis"):
        track_history = defaultdict(list)
        counted = set()
        counts = Counter()
        imgs_hilang = []

        # YOLO Tracking (Memerlukan lapx di requirements.txt)
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for f_idx, res in enumerate(results):
            img = res.orig_img
            cv2.polylines(img, [roi], True, (0, 255, 0), 2)
            cv2.line(img, (int(0.28*w), y_ref), (int(0.72*w), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                for box, tid, cls in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.cpu().numpy().astype(int), res.boxes.cls.cpu().numpy().astype(int)):
                    cx, cy = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
                    label = model.names[cls]

                    if cv2.pointPolygonTest(roi, (cx, cy), False) >= 0:
                        track_history[tid].append(label)
                        if cy > y_ref and tid not in counted:
                            counted.add(tid)
                            final_lbl = Counter(track_history[tid]).most_common(1)[0][0]
                            counts[final_lbl] += 1
                            
                            if "Hilang" in final_lbl:
                                snapshot = res.plot()
                                snapshot_rgb = cv2.cvtColor(snapshot, cv2.COLOR_BGR2RGB)
                                imgs_hilang.append({"img": snapshot_rgb, "txt": f"ID:{tid}"})
                                
                                # Tampilkan 4 foto terbaru secara real-time
                                with galeri_container:
                                    cols = st.columns(4)
                                    for i, item in enumerate(reversed(imgs_hilang[-4:])):
                                        cols[i].image(item["img"], caption=item["txt"], use_container_width=True)

            frame_win.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            table_win.table(pd.DataFrame({"Kategori": counts.keys(), "Unit": counts.values()}))

    cap.release()
    os.unlink(tfile.name)
