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

# Menambahkan CSS custom untuk tampilan yang lebih profesional
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_name_html=True)

st.title("🚊 Sistem Monitoring Penambat Rel (Tesis)")
st.sidebar.title("Pengaturan Analisis")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Model YOLOv8/v11 yang sudah Anda upload ke GitHub
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Pastikan file 'best.pt' ada di root repository.")

# --- 3. SIDEBAR: INPUT & PARAMETER ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rekaman Rel", type=["mp4", "avi", "mov"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.20)
process_button = st.sidebar.button("🚀 Jalankan Deteksi")

# --- 4. LOGIKA UTAMA ---
if uploaded_video is not None:
    # Menggunakan suffix agar OpenCV mengenali format file
    suffix = os.path.splitext(uploaded_video.name)[1]
    
    if process_button:
        # Membuat file sementara yang aman untuk sistem Linux (Streamlit Cloud)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Inisialisasi data tracking & counting
        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter()
        captured_faults = [] # Untuk galeri penambat hilang
        
        # Garis referensi (Garis Hitung)
        y_ref = int(0.50 * height)
        
        # Layout Dashboard
        col_vid, col_stat = st.columns([2, 1])
        st_frame = col_vid.empty()
        st_metrics = col_stat.empty()
        
        st.info("Sedang memproses video... Harap jangan menutup halaman ini.")

        # Menjalankan tracking dengan parameter yang dioptimalkan
        # Menggunakan imgsz=640 agar tidak crash di server RAM kecil
        results = model.track(source=temp_path, persist=True, imgsz=640, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            
            # Visualisasi Garis Hitung
            cv2.line(frame, (0, y_ref), (width, y_ref), (255, 0, 0), 2)
            cv2.putText(frame, "Garis Hitung", (10, y_ref - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)
                confs = res.boxes.conf.cpu().numpy()

                for box, tid, cls, conf in zip(boxes, ids, clss, confs):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]
                    
                    # Simpan riwayat label untuk voting
                    track_history[tid].append(label)

                    # LOGIKA HITUNG (Sesuai preferensi Anda)
                    if cy > y_ref and tid not in counted_ids:
                        counted_ids.add(tid)
                        # Majority vote untuk label kelas
                        final_label = Counter(track_history[tid]).most_common(1)[0][0]
                        summary_counts[final_label] += 1
                        
                        # Jika terdeteksi 'hilang', ambil screenshot untuk galeri
                        if "hilang" in final_label.lower():
                            annotated_res = res.plot() # Ambil frame dengan box
                            res_rgb = cv2.cvtColor(annotated_res, cv2.COLOR_BGR2RGB)
                            captured_images = Image.fromarray(res_rgb)
                            captured_faults.append((captured_images, f"ID:{tid} | Frame:{frame_idx}"))

            # Tampilkan Video secara Real-time
            frame_rgb = cv2.cvtColor(res.plot(), cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Tampilkan Statistik secara Real-time
            with st_metrics:
                st.subheader("📊 Inventaris Aset")
                st.metric("Total Terhitung", len(counted_ids))
                for lbl, count in summary_counts.items():
                    st.write(f"✅ **{lbl}**: {count} unit")

        cap.release()
        os.remove(temp_path) # Hapus file sementara
        st.success("Analisis Selesai!")

        # --- 5. TAMPILKAN GALERI TEMUAN ---
        if captured_faults:
            st.divider()
            st.subheader("🚨 Galeri Penambat Hilang (Temuan Otomatis)")
            cols = st.columns(3)
            for i, (img, caption) in enumerate(captured_faults):
                with cols[i % 3]:
                    st.image(img, caption=caption, use_container_width=True)
