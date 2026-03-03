import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import zipfile
from io import BytesIO
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Railway Fastener Detector", layout="wide")
st.title("Sistem Deteksi & Dokumentasi Penambat Kereta Api")

# --- INISIALISASI FOLDER & SESSION STATE ---
CAPTURE_DIR = "captured_missing"
if not os.path.exists(CAPTURE_DIR):
    os.makedirs(CAPTURE_DIR)

if 'summary_counts' not in st.session_state:
    st.session_state.summary_counts = Counter()
if 'counted_ids' not in st.session_state:
    st.session_state.counted_ids = set()
if 'track_history' not in st.session_state:
    st.session_state.track_history = defaultdict(list)
if 'captured_files' not in st.session_state:
    st.session_state.captured_files = []

# --- LOAD MODEL ---
@st.cache_resource
def load_model():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'best.pt')
    if not os.path.exists(model_path):
        st.error("File 'best.pt' tidak ditemukan di direktori utama!")
        st.stop()
    return YOLO(model_path)

model = load_model()

# --- SIDEBAR & KONTROL ---
st.sidebar.title("Kontrol Deteksi")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)
source_option = st.sidebar.selectbox("Pilih Sumber Input", ("Upload Video", "Kamera Real-time"))

if st.sidebar.button("Reset Semua Data"):
    st.session_state.summary_counts = Counter()
    st.session_state.counted_ids = set()
    st.session_state.track_history = defaultdict(list)
    st.session_state.captured_files = []
    for f in os.listdir(CAPTURE_DIR):
        os.remove(os.path.join(CAPTURE_DIR, f))
    st.rerun()

# --- FUNGSI PEMROSESAN ---
def process_frame(frame, model, roi_points, y_ref, width):
    # Inference Tracking
    results = model.track(frame, persist=True, conf=conf_threshold, imgsz=1024, verbose=False)
    
    # Visual ROI Melayang
    overlay = frame.copy()
    cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
    frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
    cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

    # Garis Hitung
    x_kiri_garis = int(0.28 * width)
    x_kanan_garis = int(0.72 * width)
    cv2.line(frame, (x_kiri_garis, y_ref), (x_kanan_garis, y_ref), (255, 0, 0), 3)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        clss = results[0].boxes.cls.cpu().numpy().astype(int)

        for box, tid, cls in zip(boxes, ids, clss):
            x1, y1, x2, y2 = box
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
            label = model.names[cls]

            # Cek ROI
            if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                st.session_state.track_history[tid].append(label)

                # Logika Hitung Crossing Line
                if cy > y_ref and tid not in st.session_state.counted_ids:
                    st.session_state.counted_ids.add(tid)
                    final_label = Counter(st.session_state.track_history[tid]).most_common(1)[0][0]
                    st.session_state.summary_counts[final_label] += 1
                    
                    # Auto-Capture jika Hilang
                    if final_label == "Hilang":
                        filepath = os.path.join(CAPTURE_DIR, f"ID_{tid}_{int(time.time())}.jpg")
                        # Plot hasil deteksi pada frame yang disimpan
                        cv2.imwrite(filepath, results[0].plot())
                        st.session_state.captured_files.append(filepath)

                # Visualisasi Box
                color = (0, 255, 0) if tid in st.session_state.counted_ids else (0, 255, 255)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                cv2.putText(frame, f"ID:{tid} {label}", (int(x1), int(y1)-5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

# --- TAMPILAN UTAMA ---
col_vid, col_res = st.columns([3, 1])

with col_vid:
    st_frame = st.empty()

with col_res:
    st.subheader("Rekapitulasi")
    rekap_placeholder = st.empty()
    st.write("---")
    st.subheader("Aksi Dokumentasi")
    download_placeholder = st.empty()

# Penanganan Input
video_file = None
if source_option == "Upload Video":
    video_file = st.file_uploader("Upload video .mp4", type=['mp4', 'avi', 'mov'])

if video_file or source_option == "Kamera Real-time":
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(0)

    if cap.isOpened():
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Koordinat ROI Sinkron final.py
        y_atas, y_bawah, y_ref = int(0.35 * height), int(0.85 * height), int(0.6 * height)
        roi_points = np.array([
            [int(0.25 * width), y_bawah], [int(0.42 * width), y_atas],
            [int(0.58 * width), y_atas], [int(0.75 * width), y_bawah]
        ], np.int32)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            processed = process_frame(frame, model, roi_points, y_ref, width)
            st_frame.image(processed, channels="BGR", use_container_width=True)

            # Update Rekap
            with rekap_placeholder.container():
                for c in ["DE CLIP", "E Clip", "Hilang", "KA Clip"]:
                    count = st.session_state.summary_counts[c]
                    st.markdown(f"**{c}:** :{'red' if c == 'Hilang' else 'green'}[{count} unit]")

        cap.release()

# --- BAGIAN BAWAH: GALERI & DOWNLOAD ---
if st.session_state.captured_files:
    st.write("---")
    st.subheader(f"🖼️ Galeri Deteksi Hilang ({len(st.session_state.captured_files)} file)")
    
    # Tombol Unduh Semua (.zip)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "x") as csv_zip:
        for f in st.session_state.captured_files:
            csv_zip.write(f, arcname=os.path.basename(f))
    
    st.download_button(
        label="📥 Unduh Semua Foto (.zip)",
        data=buf.getvalue(),
        file_name="rekap_penambat_hilang.zip",
        mime="application/zip"
    )

    # Tampilkan Galeri Grid
    cols = st.columns(4)
    for idx, img_path in enumerate(st.session_state.captured_files):
        with cols[idx % 4]:
            st.image(img_path, caption=f"ID: {os.path.basename(img_path)}")
