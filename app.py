import streamlit as st
import cv2
import numpy as np
import pandas as pd
import tempfile
import os
import zipfile
from collections import defaultdict, Counter
from ultralytics import YOLO

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Kereta API", layout="wide")

st.title("🚉 Sistem Deteksi & Dokumentasi Penambat Rel")
st.markdown("Aplikasi untuk mendeteksi komponen penambat dan otomatis mengambil **Screenshot** pada objek yang terdeteksi **Hilang**.")

# --- INISIALISASI FOLDER DOKUMENTASI ---
CAPTURE_FOLDER = 'deteksi_hilang'
if not os.path.exists(CAPTURE_FOLDER):
    os.makedirs(CAPTURE_FOLDER)

# --- SIDEBAR: KONFIGURASI ---
st.sidebar.header("Konfigurasi")

# Menggunakan path absolut untuk memastikan file best.pt ditemukan
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        # Load model YOLOv8
        return YOLO(path)
    except Exception:
        return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"❌ File 'best.pt' tidak terbaca di: {MODEL_PATH}")
    st.info("Pastikan file sudah di-push ke GitHub dan coba tekan 'Reboot App' di dashboard Streamlit.")
else:
    st.sidebar.success("✅ Model Load Success")

uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4", "mov", "avi"])

if uploaded_video is not None and model is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    # PERBAIKAN: Menggunakan konstanta yang benar
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # ROI Melayang (Floating ROI) sesuai parameter tesis Anda
    y_atas, y_bawah = int(0.35 * height), int(0.85 * height)
    roi_points = np.array([
        [int(0.25 * width), y_bawah], [int(0.42 * width), y_atas],
        [int(0.58 * width), y_atas], [int(0.75 * width), y_bawah]
    ], np.int32)
    y_ref = int(0.6 * height)

    col1, col2 = st.columns([2, 1])
    frame_placeholder = col1.empty()
    stats_placeholder = col2.empty()
    
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()
    captured_files = []

    if st.sidebar.button("Mulai Deteksi"):
        # Reset folder dokumentasi
        for f in os.listdir(CAPTURE_FOLDER): os.remove(os.path.join(CAPTURE_FOLDER, f))
        
        # Jalankan tracking
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Draw ROI
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

                            # Auto-Screenshot "Hilang"
                            if "Hilang" in final_label:
                                snapshot = res.plot()
                                file_path = os.path.join(CAPTURE_FOLDER, f"Hilang_ID_{tid}.jpg")
                                cv2.imwrite(file_path, snapshot)
                                if file_path not in captured_files:
                                    captured_files.append(file_path)

                        color = (0, 0, 255) if "Hilang" in label else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

            with stats_placeholder.container():
                st.subheader("📊 Statistik")
                st.write(f"**Total Aset:** {sum(summary_counts.values())}")
                st.table(pd.DataFrame({"Kategori": summary_counts.keys(), "Jumlah": summary_counts.values()}))
                if len(captured_files) > 0:
                    st.warning(f"📸 {len(captured_files)} Bukti Foto Tersimpan")

        if captured_files:
            zip_path = "bukti_deteksi.zip"
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for f in captured_files: zipf.write(f, os.path.basename(f))
            st.sidebar.download_button("📥 Download Bukti (ZIP)", open(zip_path, "rb"), file_name=zip_path)

    cap.release()
    os.unlink(tfile.name)
