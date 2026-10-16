import threading

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, webrtc_streamer

st.set_page_config(
    page_title="Live-Farbfilter",
    page_icon="🎨",
    layout="centered",
)

st.title("🎨 Live-Farbfilter")
st.write(
    "Zeige das Kamerabild live an und probiere verschiedene Farbfilter aus. "
    "Die Verarbeitung erfolgt Bild für Bild."
)

st.info(
    "Hinweis: Die Filter verändern Farben, sie korrigieren Farbenblindheit aber nicht "
    "medizinisch und garantieren keine zuverlässige Unterscheidung von Farben."
)

# Einstellungen werden zwischen Streamlit-Oberfläche und Video-Thread geteilt.
settings_lock = threading.Lock()
settings = {
    "filter_mode": "Rot-Grün",
    "intensity": 60,
    "mirror": True,
}


def apply_filter(frame: np.ndarray, mode: str, intensity: int) -> np.ndarray:
    """Apply a selectable colour filter to a BGR OpenCV frame."""
    if mode == "Ohne Filter" or intensity <= 0:
        return frame

    strength = float(intensity) / 100.0
    if mode == "Rot-Grün":
        # LAB's a-channel represents the green-to-red axis.
        # Expanding it can make red/green differences more noticeable.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lab_float = lab.astype(np.float32)
        a = lab_float[:, :, 1]
        contrast = 1.0 + 1.25 * strength
        lab_float[:, :, 1] = np.clip(128.0 + (a - 128.0) * contrast, 0, 255)
        filtered = cv2.cvtColor(
            np.clip(lab_float, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
        )
        return cv2.addWeighted(frame, 1.0 - 0.75 * strength, filtered, 0.75 * strength, 0)

    if mode == "Farbverschiebung":
        # Shift hue gradually while preserving brightness and most saturation.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.int16)
        hue_shift = int(round(18 * strength))  # OpenCV hue range is 0..179
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(frame, 1.0 - 0.85 * strength, filtered, 0.85 * strength, 0)

    if mode == "Starker Filter":
        # Stronger green/red-axis separation plus a modest saturation boost.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 1] = np.clip(128.0 + (lab[:, :, 1] - 128.0) * (1.0 + 2.0 * strength), 0, 255)
        filtered = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.35 * strength), 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(frame, 1.0 - 0.9 * strength, filtered, 0.9 * strength, 0)

    return frame


class LiveFilterProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        with settings_lock:
            mode = settings["filter_mode"]
            intensity = settings["intensity"]
            mirror = settings["mirror"]

        if mirror:
            image = cv2.flip(image, 1)

        image = apply_filter(image, mode, intensity)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


with st.sidebar:
    st.header("Filter einstellen")
    filter_mode = st.selectbox(
        "Filter",
        ["Rot-Grün", "Farbverschiebung", "Starker Filter", "Ohne Filter"],
        index=0,
        help="Wähle aus, wie das Live-Kamerabild verändert werden soll.",
    )
    intensity = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=100,
        value=60,
        step=5,
        help="0 = keine sichtbare Wirkung, 100 = maximale Wirkung des gewählten Filters.",
    )
    mirror = st.toggle("Bild spiegeln", value=True)
    st.caption("Die Einstellungen werden während des laufenden Videos übernommen.")

with settings_lock:
    settings["filter_mode"] = filter_mode
    settings["intensity"] = intensity
    settings["mirror"] = mirror

st.subheader("Live-Kamerabild")
st.write("Tippe auf **START** und erlaube deinem Browser den Kamerazugriff.")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="live-colour-filter",
    video_processor_factory=LiveFilterProcessor,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": {
            "facingMode": {"ideal": "environment"},
            "width": {"ideal": 640},
            "height": {"ideal": 480},
        },
        "audio": False,
    },
    async_processing=True,
)

st.caption(
    "Tipp: Öffne den Streamlit-Link auf deinem Handy in Chrome oder Safari. "
    "Die Kamera funktioniert nur, wenn du den Zugriff erlaubst und die Seite über HTTPS läuft."
)
