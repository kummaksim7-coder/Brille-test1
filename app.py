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

st.title("🎨 Live-Farbfilter fürs Handy")
st.write(
    "Starte die Kamera und wähle einen Filter. Die Änderungen an Filter und "
    "Intensität werden live auf das Kamerabild angewendet."
)

st.info(
    "Wichtig: Das ist ein experimenteller Farbfilter. Er kann Farben sichtbarer "
    "verändern, ist aber keine medizinische Korrektur von Farbenblindheit."
)

# Diese Einstellungen werden vom Streamlit-Teil und vom Video-Thread verwendet.
settings_lock = threading.Lock()
settings = {
    "filter_mode": "Rot-Grün-Hilfe",
    "intensity": 80,
    "mirror": False,
}


def apply_filter(frame: np.ndarray, mode: str, intensity: int) -> np.ndarray:
    """Wendet einen deutlich sichtbaren Farbfilter auf ein BGR-Bild an."""
    if mode == "Ohne Filter" or intensity <= 0:
        return frame

    strength = float(intensity) / 100.0
    original = frame.copy()

    if mode == "Rot-Grün-Hilfe":
        # Verschiebt Grün-Töne deutlich in Richtung Türkis/Blau und Rot-Töne
        # in Richtung Orange. So werden die Farbbereiche sichtbar verändert.
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        # OpenCV-Hue: 0..179. Grünbereich wird Richtung Cyan verschoben.
        green_mask = (hue >= 25) & (hue <= 85) & (sat >= 35)
        hsv[:, :, 0][green_mask] = np.clip(
            hue[green_mask] + 25.0 * strength, 0, 179
        )

        # Rottöne (nahe 0 oder 179) werden Richtung Orange verschoben.
        red_mask = ((hue <= 10) | (hue >= 170)) & (sat >= 35)
        red_hue = hue[red_mask]
        hsv[:, :, 0][red_mask] = np.where(
            red_hue >= 170,
            (red_hue + 180.0 + 7.0 * strength) % 180.0,
            np.clip(red_hue + 7.0 * strength, 0, 179),
        )

        # Etwas mehr Farbsättigung, damit die Änderung klar erkennbar ist.
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.45 * strength), 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(original, 1.0 - strength, filtered, strength, 0)

    if mode == "Farbverschiebung":
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + 35.0 * strength) % 180.0
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.25 * strength), 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(original, 1.0 - strength, filtered, strength, 0)

    if mode == "Kontrast + Farbe":
        # Sichtbar kräftigere Farben und höherer Kontrast.
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Kontrast im Helligkeitskanal erhöhen.
        l_float = l_channel.astype(np.float32)
        l_float = 128.0 + (l_float - 128.0) * (1.0 + 0.8 * strength)
        l_channel = np.clip(l_float, 0, 255).astype(np.uint8)

        # Farbkanäle stärker aus der neutralen Mitte herausziehen.
        a_float = 128.0 + (a_channel.astype(np.float32) - 128.0) * (1.0 + 1.8 * strength)
        b_float = 128.0 + (b_channel.astype(np.float32) - 128.0) * (1.0 + 1.2 * strength)
        lab = cv2.merge((
            l_channel,
            np.clip(a_float, 0, 255).astype(np.uint8),
            np.clip(b_float, 0, 255).astype(np.uint8),
        ))
        filtered = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1.0 + 0.5 * strength), 0, 255)
        filtered = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.addWeighted(original, 1.0 - strength, filtered, strength, 0)

    return frame


class LiveFilterProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")

        # Einstellungen bei jedem Bild neu lesen, damit die Regler live greifen.
        with settings_lock:
            mode = settings["filter_mode"]
            intensity = settings["intensity"]
            mirror = settings["mirror"]

        # Standardmäßig NICHT spiegeln. So bewegen sich Dinge in die erwartete Richtung.
        if mirror:
            image = cv2.flip(image, 1)

        image = apply_filter(image, mode, intensity)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


with st.sidebar:
    st.header("Filter einstellen")
    filter_mode = st.selectbox(
        "Filter",
        [
            "Rot-Grün-Hilfe",
            "Farbverschiebung",
            "Kontrast + Farbe",
            "Ohne Filter",
        ],
        index=0,
        help="Wähle einen Filter. Die Wirkung sollte direkt im Kamerabild sichtbar sein.",
    )
    intensity = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=100,
        value=80,
        step=5,
        help="0 = keine Filterwirkung, 100 = stärkste Filterwirkung.",
    )
    mirror = st.toggle(
        "Bild spiegeln (Selfie-Ansicht)",
        value=False,
        help="Aus: normale Ansicht. Ein: links und rechts werden vertauscht.",
    )
    st.caption("Filter oder Intensität ändern, während die Kamera läuft.")

with settings_lock:
    settings["filter_mode"] = filter_mode
    settings["intensity"] = intensity
    settings["mirror"] = mirror

st.subheader("Live-Kamerabild")
st.write("Tippe auf **START** und erlaube den Kamerazugriff. Falls du schon gestartet hast, ändere links den Filter.")

rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="live-colour-filter-v2",
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
    "Tipp: Öffne die HTTPS-Streamlit-Adresse direkt in Chrome oder Safari. "
    "Wenn du den Code auf GitHub ersetzt, warte auf den Neustart der Streamlit-App "
    "und lade die Seite auf dem Handy neu."
)
