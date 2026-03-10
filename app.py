import streamlit as st
import os
import cv2
import numpy as np
import pandas as pd
import tempfile
from collections import defaultdict, Counter
from ultralytics import YOLO
import shutil

# --- PENETAPAN SISTEM (Mesti di atas sekali) ---
os.environ["QT_QPA_PLATFORM"] = "offscreen"  # Proteksi agar OpenCV tidak mencari display/layar

# --- LOAD MODEL ---
MODEL_PATH = 'best.pt'  # Pastikan nama file model sudah benar

@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    else:
        st.error(f"Model tidak ditemukan di {path}.")
        st.stop()

model = load_model(MODEL_PATH)

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sistem Deteksi Penambat - ITB", layout="wide")
st.title("🚉 Dashboard Deteksi & Dokumentasi Penambat")
st.markdown("Muat naik video rel untuk mengesan dan merakam lokasi penambat yang **Hilang**.")

# --- SIDEBAR ---
st.sidebar.header("Konfigurasi")
uploaded_video = st.sidebar.file_uploader("Pilih Video Rel (MP4)", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    
    cap = cv2.VideoCapture(tfile.name)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # ROI & Garisan Rujukan (Sesuai parameter tesis anda)
    y_ref = int(0.5 * height)
    roi_points = np.array([
        [int(0.35 * width), int(0.70 * height)],  # Kiri bawah
        [int(0.40 * width), int(0.10 * height)],  # Kiri atas
        [int(0.60 * width), int(0.10 * height)],  # Kanan atas
        [int(0.65 * width), int(0.70 * height)]   # Kanan bawah
    ], np.int32)

    # Penyiapan Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter("hasil_counting_final.mp4", fourcc, fps, (width, height))

    # Struktur Data
    track_history = defaultdict(list)
    counted_ids = set()
    summary_counts = Counter()
    final_results = []

    # --- Mulai Analisis ---
    if st.sidebar.button("Mulai Analisis"):
        results = model.track(source=tfile.name, persist=True, imgsz=1024, stream=True, conf=conf_threshold)

        for frame_idx, res in enumerate(results):
            frame = res.orig_img

            # Visualisasi ROI (Hijau Transparan)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [roi_points], (0, 255, 0))
            frame = cv2.addWeighted(overlay, 0.1, frame, 0.9, 0)
            cv2.polylines(frame, [roi_points], True, (0, 255, 0), 2)

            # Visualisasi Garis Hitung (Biru)
            cv2.line(frame, (int(0.28*width), y_ref), (int(0.72*width), y_ref), (255, 0, 0), 3)

            if res.boxes is not None and res.boxes.id is not None:
                boxes = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int)
                clss = res.boxes.cls.cpu().numpy().astype(int)

                for box, tid, cls in zip(boxes, ids, clss):
                    x1, y1, x2, y2 = box
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    label = model.names[cls]

                    # Cek apakah objek berada di dalam ROI
                    if cv2.pointPolygonTest(roi_points, (cx, cy), False) >= 0:
                        track_history[tid].append(label)

                        # LOGIKA HITUNG: Saat objek melewati garis pertama kali
                        if cy > y_ref and tid not in counted_ids:
                            counted_ids.add(tid)

                            # Tentukan label final menggunakan Majority Vote
                            final_label = Counter(track_history[tid]).most_common(1)[0][0]
                            summary_counts[final_label] += 1

                            # Catat data SNAPSHOT UNIK untuk proses capture
                            final_results.append({
                                'frame': frame_idx,
                                'track_id': tid,
                                'class_name': final_label
                            })

                        # Gambar Box di Video
                        color = (0, 0, 255) if label == "Hilang" else (0, 255, 0)
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        cv2.putText(frame, f"ID:{tid}", (int(x1), int(y1)-5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Dashboard Rekapitulasi di Layar Video
            cv2.rectangle(frame, (20, 20), (350, 210), (0, 0, 0), -1)
            classes_display = ["DE CLIP", "E Clip", "Hilang", "KA Clip"]
            for i, cls_name in enumerate(classes_display):
                count = summary_counts[cls_name]
                text_color = (0, 0, 255) if cls_name == "Hilang" else (0, 255, 0)
                cv2.putText(frame, f"{cls_name}: {count}", (40, 60 + (i*35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

            out_video.write(frame)

        cap.release()
        out_video.release()

        # Simpan ke DataFrame agar bisa dibaca Cell Capture
        df = pd.DataFrame(final_results)

        # Tampilkan Laporan Akhir
        st.success("✅ Analisis Selesai!")
        st.subheader("📊 Statistik Deteksi")
        st.table(df)

        # Simpan Galeri Foto Lokasi Hilang
        capture_folder = 'Hasil_Penambat_Hilang'
        if not os.path.exists(capture_folder):
            os.makedirs(capture_folder)

        df_missing = df[df['class_name'] == 'Hilang']

        cap = cv2.VideoCapture(tfile.name)
        for _, row in df_missing.iterrows():
            frame_no, track_id = int(row['frame']), int(row['track_id'])
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
            ret, frame = cap.read()

            if ret:
                results_single = model.predict(frame, conf=0.15, imgsz=1024, verbose=False)
                annotated_frame = results_single[0].plot()
                file_name = f"{capture_folder}/Hilang_ID_{track_id}_Frame_{frame_no}.jpg"
                cv2.imwrite(file_name, annotated_frame)

        cap.release()

        # Zip dan Download
        shutil.make_archive(f'{capture_folder}_zip', 'zip', capture_folder)
        st.download_button("Unduh Semua Foto Hilang", f"{capture_folder}_zip.zip")

else:
    st.info("Silakan unggah video untuk memulai deteksi.")
