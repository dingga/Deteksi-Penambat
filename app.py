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

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("🚊 Sistem Monitoring Penambat Rel (Tesis)")
st.sidebar.title("Pengaturan Analisis")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_model():
    # Pastikan file best.pt ada di direktori yang sama dengan app.py di GitHub
    return YOLO("best.pt")

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# --- 3. SIDEBAR: INPUT ---
uploaded_video = st.sidebar.file_uploader("Upload Video Rekaman Rel", type=["mp4", "avi", "mov"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.05, 1.0, 0.15)
process_button = st.sidebar.button("🚀 Jalankan Deteksi")

# --- 4. LOGIKA UTAMA ---
if uploaded_video is not None:
    suffix = os.path.splitext(uploaded_video.name)[1]
    
    if process_button:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        cap = cv2.VideoCapture(temp_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = int(cap.get(cv2.CAP_PROP_FPS))
        
        # --- KONFIGURASI ROI MELAYANG (FLOATING ROI) ---
        y_atas  = int(0.35 * height) 
        y_bawah = int(0.85 * height) 
        roi_points = np.array([
            [int(0.25 * width), y_bawah], [int(0.42 * width), y_atas],
            [int(0.58 * width), y_atas], [int(0.75 * width), y_bawah]
        ], np.int32)
        y_ref = int(0.6 * height) # Garis Hitung

        track_history = defaultdict(list)
        counted_ids = set()
        summary_counts = Counter()
        captured_faults = []
        
        col_vid, col_stat = st.columns([2, 1])
        st_frame = col_vid.empty()
        st_metrics = col_stat.empty()
        
        st.info("Sedang memproses video dengan logika ROI Akurasi Tinggi...")

        # Jalankan tracking dengan stream=True agar hemat RAM
        results = model.track(source=temp_path, persist=True, imgsz=1024, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img
            frame_clean = frame.copy() # Frame untuk snapshot tanpa garis ROI
            
            # Gambar Visual ROI & Line pada frame monitoring
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    
                    # CEK ROI: Hanya proses jika objek di dalam area hijau (rel tengah)
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        label = model.names[cls]
                        track_history[tid].append(label)

                        # LOGIKA HITUNG & SNAPSHOT (Majority Voting)
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)
                            
                            # Tentukan label final berdasarkan histori track ID tersebut
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1
                            
                            # Jika Status "Hilang", ambil snapshot otomatis
                            if "hilang" in final_label.lower():
                                # Gunakan anotasi YOLO pada frame snapshot
                                annotated_res = res.plot() 
                                res_rgb = cv2.cvtColor(annotated_res, cv2.COLOR_BGR2RGB)
                                captured_faults.append((Image.fromarray(res_rgb), f"ID:{tid} | Frame:{frame_idx}"))

                        # Visualisasi box pada monitoring
                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Update Tampilan Monitoring (Real-time)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            st_frame.image(frame_rgb, channels="RGB", use_container_width=True)
            
            with st_metrics:
                st_metrics.empty() # Bersihkan statistik lama
                st.subheader("📊 Inventaris Aset")
                st.metric("Total Aset Lewat", len(counted_ids))
                
                # Tampilkan rincian per kategori
                categories = ["DE CLIP", "E Clip", "Hilang", "KA Clip"]
                for cat in categories:
                    val = summary_counts.get(cat, 0)
                    st.write(f"🔹 **{cat}**: {val} unit")

        cap.release()
        if os.path.exists(temp_path):
            os.remove(temp_path)
        st.success("Analisis Selesai!")

        # Galeri Temuan Spesifik "Hilang"
        if captured_faults:
            st.divider()
            st.subheader(f"🚨 Bukti Temuan Penambat Hilang ({len(captured_faults)} titik)")
            cols = st.columns(3)
            for i, (img, caption) in enumerate(captured_faults):
                with cols[i % 3]:
                    st.image(img, caption=caption, use_container_width=True)
