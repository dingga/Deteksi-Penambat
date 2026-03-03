import streamlit as st
import cv2
import numpy as np
import tempfile
from collections import defaultdict, Counter
from ultralytics import YOLO
from PIL import Image

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Detector", layout="wide")
st.title("Sistem Deteksi Penambat Kereta Api (YOLOv8)")
st.sidebar.title("Pengaturan Deteksi")

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di folder 'models' di GitHub Anda
    model = YOLO('models/best.pt') 
    return model

model = load_model()

# --- SIDEBAR SETTINGS ---
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)
source_option = st.sidebar.selectbox("Pilih Sumber Input", ("Upload Video", "Kamera Real-time"))

# --- LOGIKA DASHBOARD & TRACKING ---
track_history = defaultdict(list)
counted_ids = set()
summary_counts = Counter()

def process_frame(frame, model, roi_points, y_ref):
    height, width = frame.shape[:2]
    
    # Overlay Visual ROI
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

    # Garis Hitung
    x_kiri_garis = int(0.28 * width)
    x_kanan_garis = int(0.72 * width)
    cv2.line(frame, (x_kiri_garis, y_ref), (x_kanan_garis, y_ref), (255, 0, 0), 3)

    # Inference
    results = model.track(frame, persist=True, conf=conf_threshold, verbose=False)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Cek ROI
            if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                label = model.names[cls]
                track_history[tid].append(label)

                # Logika Hitung Crossing Line
                if cy > y_ref and tid not in counted_ids:
                    counted_ids.add(tid)
                    final_label = Counter(track_history[tid]).most_common(1)[0][0]
                    summary_counts[final_label] += 1

                # Visualisasi
                color = (0, 255, 0) if tid in counted_ids else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} {model.names[cls]}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return frame

# --- MAIN APP LOGIC ---
col1, col2 = st.columns([3, 1])

with col1:
    st_frame = st.empty()

with col2:
    st.subheader("Hasil Perhitungan")
    rekap_placeholder = st.empty()

# Penanganan Video
cap = None
if source_option == "Upload Video":
    video_file = st.file_uploader("Upload video .mp4", type=['mp4', 'avi', 'mov'])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
else:
    cap = cv2.VideoCapture(0)

if cap:
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Pengaturan ROI (Sesuai script Colab Anda)
    y_atas  = int(0.35 * height)
    y_bawah = int(0.85 * height)
    roi_points = np.array([
        [int(0.25 * width), y_bawah],
        [int(0.42 * width), y_atas],
        [int(0.58 * width), y_atas],
        [int(0.75 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.6 * height)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        processed_frame = process_frame(frame, model, roi_points, y_ref)
        
        # Tampilan Video
        st_frame.image(processed_frame, channels="BGR", use_container_width=True)

        # Update Rekapitulasi di Sidebar/Kolom Kanan
        with rekap_placeholder.container():
            for cls_name in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                count = summary_counts[cls_name]
                color = "red" if cls_name == "Hilang" else "green"
                st.markdown(f"**{cls_name}:** :{color}[{count} unit]")

    cap.release()
