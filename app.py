import streamlit as st
import os

# --- PROTEKSI IMPORT ---
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Gagal memuat modul: {e}")
    st.info("Pastikan requirements.txt menggunakan 'opencv-python-headless'.")
    st.stop()

import pandas as pd
import tempfile
import zipfile
from collections import defaultdict, Counter

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat ITB", layout="wide")
st.title("🚉 Sistem Deteksi & Dokumentasi Penambat Rel")

# --- PATH MODEL ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

@st.cache_resource
def load_model(path):
    if not os.path.exists(path): return None
    try: return YOLO(path)
    except: return None

model = load_model(MODEL_PATH)

if model is None:
    st.error(f"❌ File 'best.pt' tidak ditemukan di {MODEL_PATH}")
    st.stop()
else:
    st.sidebar.success("✅ Model Siap")

# --- PROSES VIDEO ---
uploaded_video = st.sidebar.file_uploader("Upload Video (MP4)", type=["mp4"])

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

    if st.sidebar.button("Jalankan Analisis"):
        results = model.track(source=tfile.name, persist=True, imgsz=640, stream=True)
        # ... (Logika deteksi & screenshot tetap sama seperti sebelumnya)
