import streamlit as st
import os

import streamlit as st
import os

# WAJIB: Taruh ini di baris paling atas sebelum import cv2
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
except ImportError as e:
    st.error(f"Modul tidak ditemukan: {e}")
    st.stop()
    
# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Penambat Rel - ITB", layout="wide")
st.title("🚉 Dashboard Deteksi & Dokumentasi Penambat")
st.markdown("Sistem deteksi komponen penambat rel secara otomatis dengan dokumentasi temuan **Hilang**.")

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

# --- SIDEBAR ---
st.sidebar.header("Konfigurasi")
uploaded_video = st.sidebar.file_uploader("Upload Video Rel (MP4)", type=["mp4"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.15)

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

    # Layout Utama
    col_main, col_stat = st.columns([2, 1])
    frame_placeholder = col_main.empty()
    stats_placeholder = col_stat.empty()

    st.divider()
    # Placeholder untuk galeri real-time (hanya tampilkan 4 terbaru)
    gallery_placeholder = st.empty() 

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
                                captured_images.append({"img": snapshot_rgb, "id": tid})
                                
                                # REVISI: Update galeri real-time (4 terbaru)
                                with gallery_placeholder.container():
                                    st.subheader(f"📸 Monitoring Temuan ({len(captured_images)} Hilang)")
                                    cols = st.columns(4)
                                    latest_items = captured_images[-4:]
                                    for i, item in enumerate(reversed(latest_items)):
                                        with cols[i]:
                                            st.image(item["img"], caption=f"ID: {item['id']}", use_container_width=True)

            # Update Live Video Feed
            frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            
            # Update Statistik
            with stats_placeholder.container():
                st.subheader("📊 Statistik")
                st.metric("Total Terdeteksi", sum(summary_counts.values()))
                st.table(pd.DataFrame({
                    "Kategori": summary_counts.keys(), 
                    "Jumlah": summary_counts.values()
                }))

        # --- BAGIAN SETELAH ANALISIS SELESAI ---
        st.success(f"✅ Analisis Selesai. Total {len(captured_images)} penambat hilang ditemukan.")
        
        if len(captured_images) > 0:
            st.divider()
            st.subheader("📁 Galeri Dokumentasi Lengkap")
            st.info("Berikut adalah seluruh penambat yang terdeteksi Hilang.")
            
            # Grid Galeri Lengkap (5 kolom)
            full_grid_cols = st.columns(5)
            for idx, item in enumerate(captured_images):
                with full_grid_cols[idx % 5]:
                    st.image(item["img"], caption=f"ID: {item['id']}", use_container_width=True)
            
            # Fitur Unduh Laporan CSV
            st.divider()
            df_report = pd.DataFrame([{"ID_Penambat": x["id"], "Status": "Hilang"} for x in captured_images])
            st.download_button(
                label="📥 Download Laporan Deteksi (CSV)",
                data=df_report.to_csv(index=False),
                file_name="laporan_penambat_hilang.csv",
                mime="text/csv"
            )

    cap.release()
    os.unlink(tfile.name)
else:
    st.info("Silakan upload video rel untuk memulai analisis.")
