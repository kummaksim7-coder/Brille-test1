import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="ColorVision",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ ColorVision")
st.write("Vergleiche das Originalvideo mit der gefilterten Version.")

# Design
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #10172a, #202044);
    color: white;
}
h1, h2, h3, p, label {
    color: white !important;
}
</style>
""", unsafe_allow_html=True)


# Filterfunktion
def apply_filter(frame, strength, mode):
    # BGR -> LAB
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)

    L = lab[:, :, 0]
    A = lab[:, :, 1]
    B = lab[:, :, 2]

    amount = strength / 100.0

    if mode == "Rot-Grün":
        # Experimentelle Veränderung der Rot-Grün-Farbwahrnehmung
        A = 128 + (A - 128) * (1 + amount)
        B = 128 + (B - 128) * (1 + amount * 0.35)

    elif mode == "Farbverschiebung":
        A = A + amount * 12
        B = B - amount * 8

    elif mode == "Starker Filter":
        A = 128 + (A - 128) * (1 + amount * 1.5)
        B = 128 + (B - 128) * (1 + amount)

    lab[:, :, 0] = np.clip(L * 1.8 - 102, 0, 255)
    lab[:, :, 1] = np.clip(A, 0, 255)
    lab[:, :, 2] = np.clip(B, 0, 255)

    lab = lab.astype(np.uint8)

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# Einstellungen
st.subheader("Filtereinstellungen")

mode = st.selectbox(
    "Filter auswählen",
    ["Rot-Grün", "Farbverschiebung", "Starker Filter"]
)

strength = st.slider(
    "Filterstärke",
    min_value=0,
    max_value=500,
    value=150,
    step=10
)

uploaded_video = st.file_uploader(
    "Video hochladen",
    type=["mp4", "mov", "avi", "mkv"]
)

if uploaded_video is not None:
    # Temporäre Videodatei speichern
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_input:
        temp_input.write(uploaded_video.read())
        input_path = temp_input.name

    # Video öffnen
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        st.error("Das Video konnte nicht geöffnet werden.")
        os.remove(input_path)
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 25

    if width <= 0 or height <= 0:
        st.error("Die Videogröße konnte nicht erkannt werden.")
        cap.release()
        os.remove(input_path)
        st.stop()

    st.write(f"**Video:** {width} × {height} Pixel · {fps:.1f} FPS")

    # Fortschrittsanzeige
    progress = st.progress(0)
    status = st.empty()

    # Gefilterte Frames sammeln
    filtered_frames = []

    frame_count = 0

    while True:
        success, frame = cap.read()

        if not success:
            break

        filtered = apply_filter(frame, strength, mode)
        filtered_frames.append(filtered)

        frame_count += 1

        if total_frames > 0:
            progress.progress(
                min(frame_count / total_frames, 1.0)
            )

        status.write(
            f"Verarbeite Video: {frame_count} Frames"
        )

    cap.release()
    os.remove(input_path)

    progress.empty()
    status.empty()

    if not filtered_frames:
        st.error("Es konnten keine Videoframes gelesen werden.")
        st.stop()

    # Gefiltertes Video erstellen
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".mp4"
    ) as temp_output:
        output_path = temp_output.name

    try:
        import imageio_ffmpeg

        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

        import subprocess

        command = [
            ffmpeg,
            "-y",
            "-loglevel", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{width}x{height}",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            output_path
        ]

        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE
        )

        for frame in filtered_frames:
            process.stdin.write(frame.tobytes())

        process.stdin.close()
        error_output = process.stderr.read()
        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(error_output.decode(
                "utf-8", errors="replace"
            ))

        # Videos anzeigen
        st.success("Video erfolgreich verarbeitet!")

        st.subheader("1. Originalvideo")
        st.video(uploaded_video)

        st.subheader("2. Video mit ColorVision-Filter")
        with open(output_path, "rb") as video_file:
            filtered_video = video_file.read()

        st.video(filtered_video)

        st.download_button(
            label="Gefiltertes Video herunterladen",
            data=filtered_video,
            file_name="colorvision_filtered.mp4",
            mime="video/mp4"
        )

    except Exception as e:
        st.error(f"Fehler bei der Videoverarbeitung: {e}")

    finally:
        if os.path.exists(output_path):
            os.remove(output_path)
        
