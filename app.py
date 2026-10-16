import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import subprocess
import imageio_ffmpeg

from pathlib import Path


# ==================================================
# SEITENKONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ColorVision",
    page_icon="🎨",
    layout="wide",
)


# ==================================================
# DESIGN
# ==================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            135deg, #0b1020, #111827, #172033
        );
        color: white;
    }

    h1, h2, h3, h4, p, label, span {
        color: white;
    }

    [data-testid="stSidebar"] {
        background: #111827;
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    .hero {
        padding: 32px;
        border-radius: 22px;
        background: linear-gradient(
            120deg, #4338ca, #7c3aed, #0891b2
        );
        margin-bottom: 25px;
    }

    .hero h1, .hero p {
        color: white !important;
    }

    .card {
        padding: 20px;
        border-radius: 16px;
        border: 1px solid #475569;
        background: #1e293b;
        margin-bottom: 15px;
    }

    .card h3, .card p {
        color: white !important;
    }

    [data-testid="stFileUploader"] * {
        color: white !important;
    }

    [data-testid="stSlider"] * {
        color: white !important;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background: #6d28d9;
        color: white !important;
        border: 1px solid #8b5cf6;
        border-radius: 10px;
        min-height: 42px;
    }

    [data-testid="stAlert"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# VIDEO-FILTER
# ==================================================

def apply_color_filter(frame, strength, mode):
    """
    Experimenteller Farbfilter für einzelne Videoframes.

    frame: OpenCV-Bild im BGR-Format.
    strength: Farbintensität von 0 bis 500.
    mode: ausgewählter Filtermodus.
    """

    # BGR nach RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # RGB nach LAB
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(
        np.float32
    )

    amount = strength / 100.0

    # Kontrast fest auf 1,80
    contrast = 1.80

    if mode == "Standard":
        lab[:, :, 1] = (
            128 + (lab[:, :, 1] - 128) * (1 + amount)
        )

        lab[:, :, 2] = (
            128 + (lab[:, :, 2] - 128) * (1 + amount)
        )

    elif mode == "Rot-Grün":
        lab[:, :, 1] = (
            128 + (lab[:, :, 1] - 128) * (1 + amount)
        )

        lab[:, :, 2] = (
            128 + (lab[:, :, 2] - 128) * (1 + amount * 0.5)
        )

        lab[:, :, 1] += amount * 8

    elif mode == "Farbverschiebung":
        lab[:, :, 1] += amount * 10
        lab[:, :, 2] -= amount * 10

    elif mode == "Starker Filter":
        lab[:, :, 1] = (
            128 + (lab[:, :, 1] - 128) * (1 + amount * 1.5)
        )

        lab[:, :, 2] = (
            128 + (lab[:, :, 2] - 128) * (1 + amount * 1.5)
        )

        lab[:, :, 1] += amount * 12

    # Kontrast anpassen
    lab[:, :, 0] = (
        128 + (lab[:, :, 0] - 128) * contrast
    )

    lab = np.clip(lab, 0, 255).astype(np.uint8)

    # LAB nach RGB und danach zurück nach BGR
    result_rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    result_bgr = cv2.cvtColor(
        result_rgb,
        cv2.COLOR_RGB2BGR
    )

    return result_bgr


# ==================================================
# VIDEO VERARBEITEN
# ==================================================

def process_video(input_path, output_path, strength, mode):
    """
    Verarbeitet das Video frameweise und erzeugt
    anschließend eine MP4-Datei.
    """

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    # Video-Informationen lesen
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise RuntimeError("Das Video konnte nicht geöffnet werden.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Die Videoeigenschaften sind ungültig.")

    # FFmpeg erwartet unkomprimierte Videoframes über stdin.
    command = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "-",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ]

    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_number = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            # Farbfilter anwenden
            filtered_frame = apply_color_filter(
                frame,
                strength,
                mode,
            )

            # Frame an FFmpeg übergeben
            process.stdin.write(
                filtered_frame.tobytes()
            )

            frame_number += 1

            if total_frames > 0:
                progress = min(
                    frame_number / total_frames,
                    1.0,
                )

                progress_bar.progress(progress)

                status_text.write(
                    f"Verarbeite Video: "
                    f"{frame_number} / {total_frames} Frames"
                )

    except Exception:
        if process.poll() is None:
            process.kill()

        raise

    finally:
        cap.release()

    # Eingabestream schließen, damit FFmpeg fertig wird.
    process.stdin.close()

    stderr_output = process.stderr.read().decode(
        "utf-8",
        errors="replace",
    )

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            "FFmpeg konnte das Video nicht erstellen:\n"
            + stderr_output[-3000:]
        )

    if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
        raise RuntimeError("Die Ausgabedatei wurde nicht erstellt.")

    progress_bar.progress(1.0)
    status_text.success("Video erfolgreich verarbeitet!")

    return fps, total_frames


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("# 🎨 ColorVision")
    st.markdown("### Videoeinstellungen")

    mode = st.selectbox(
        "Filtermodus",
        [
            "Standard",
            "Rot-Grün",
            "Farbverschiebung",
            "Starker Filter",
        ],
    )

    strength = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=500,
        value=75,
        step=5,
    )

    st.markdown("---")
    st.markdown("### Kontrast")
    st.markdown("**1,80 – fest eingestellt**")

    st.caption(
        "Die Filter sind experimentell und nicht "
        "medizinisch validiert."
    )


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎨 ColorVision Video</h1>
        <p>
            Lade ein Video hoch und experimentiere mit
            verschiedenen Farbfiltern. Anschließend kannst
            du das bearbeitete Video als MP4 herunterladen.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# VIDEO-UPLOAD
# ==================================================

st.markdown("## 📤 Video hochladen")

uploaded_file = st.file_uploader(
    "Wähle eine Videodatei",
    type=["mp4", "mov", "avi", "mkv"],
    help="Für den Anfang empfehlen wir ein kurzes MP4-Video.",
)


# ==================================================
# VIDEO-VERARBEITUNG
# ==================================================

if uploaded_file is not None:

    st.video(uploaded_file)

    st.info(
        "Die Verarbeitung kann bei langen oder hochauflösenden "
        "Videos einige Zeit dauern. Teste zunächst ein kurzes Video."
    )

    if st.button("🎬 Video mit Filter verarbeiten"):

        input_path = None
        output_path = None

        try:
            # Temporäre Dateien anlegen
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(uploaded_file.name).suffix,
            ) as input_temp:
                input_temp.write(uploaded_file.getvalue())
                input_path = input_temp.name

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".mp4",
            ) as output_temp:
                output_path = output_temp.name

            st.markdown("## ⚙️ Verarbeitung")

            fps, total_frames = process_video(
                input_path,
                output_path,
                strength,
                mode,
            )

            # Ergebnis für die Sitzung speichern
            with open(output_path, "rb") as video_file:
                video_bytes = video_file.read()

            st.session_state["colorvision_video"] = video_bytes
            st.session_state["colorvision_mode"] = mode
            st.session_state["colorvision_strength"] = strength

        except Exception as error:
            st.error("Bei der Videoverarbeitung ist ein Fehler aufgetreten.")
            st.code(str(error))

        finally:
            for path in (input_path, output_path):
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass


# ==================================================
# ERGEBNIS ANZEIGEN
# ==================================================

if "colorvision_video" in st.session_state:

    st.markdown("---")
    st.markdown("## 🎥 Dein bearbeitetes Video")

    st.caption(
        f"Filter: {st.session_state['colorvision_mode']} · "
        f"Farbintensität: "
        f"{st.session_state['colorvision_strength']} % · "
        "Kontrast: 1,80"
    )

    st.video(
        st.session_state["colorvision_video"],
        format="video/mp4",
    )

    st.download_button(
        label="⬇️ Bearbeitetes Video herunterladen",
        data=st.session_state["colorvision_video"],
        file_name="colorvision_filtered.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

    st.info(
        "Der Filter verändert die Farbdarstellung. "
        "Er garantiert nicht, dass Farben für Menschen "
        "mit Rot-Grün-Sehschwäche besser unterscheidbar "
        "werden. Verlasse dich bei sicherheitsrelevanten "
        "Situationen nicht auf diesen Filter."
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div style="text-align:center; padding:30px 0; color:white;">
        COLORVISION VIDEO · EXPERIMENTELLES PROJEKT
    </div>
    """,
    unsafe_allow_html=True,
)
