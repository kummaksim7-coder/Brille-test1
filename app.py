import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


# ==================================================
# KONFIGURATION
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
            135deg,
            #0b1020,
            #111827,
            #172033
        );
        color: white;
    }

    html, body, [class*="css"] {
        color: white;
    }

    h1, h2, h3, h4, p, label, span, li {
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
            120deg,
            #4338ca,
            #7c3aed,
            #0891b2
        );
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 40px;
        font-weight: 800;
        color: white !important;
    }

    .hero p {
        color: white !important;
        line-height: 1.8;
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

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: #7c3aed;
        border-color: #a78bfa;
    }

    [data-testid="stAlert"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# FILTERFUNKTIONEN
# ==================================================

def process_image(image, strength, mode):
    """
    Experimentelle Farbfilter für Ishihara-Bilder.

    Die Filter sind keine medizinischen Verfahren und
    garantieren keine bessere Erkennbarkeit.
    """

    rgb = np.array(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(
        np.float32
    )

    # Kontrast ist immer fest auf 1.80 eingestellt.
    contrast = 1.80

    # Stärke 0 bis 500 -> Faktor 0 bis 5.
    amount = strength / 100.0

    # LAB-Farbkanäle:
    # Kanal 0 = Helligkeit
    # Kanal 1 und 2 = Farbinformationen

    if mode == "Standard":
        lab[:, :, 1] = (
            128 + (lab[:, :, 1] - 128) * (1 + amount)
        )

        lab[:, :, 2] = (
            128 + (lab[:, :, 2] - 128) * (1 + amount)
        )

    elif mode == "Rot-Grün":
        # Experimentelle Verschiebung der Farbkanäle.
        # Keine gezielte medizinische Korrektur.
        lab[:, :, 1] = (
            128 + (lab[:, :, 1] - 128) * (1 + amount)
        )

        lab[:, :, 2] = (
            128 + (lab[:, :, 2] - 128) * (1 + amount * 0.5)
        )

        lab[:, :, 1] += amount * 8

    elif mode == "Farbverschiebung":
        # Farbachsen in unterschiedliche Richtungen verschieben.
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

    # Helligkeitskontrast konstant auf 1.80.
    lab[:, :, 0] = (
        128 + (lab[:, :, 0] - 128) * contrast
    )

    lab = np.clip(lab, 0, 255).astype(np.uint8)

    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.markdown("# 🎨 ColorVision")

    st.markdown("### Filtereinstellungen")

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
        "Die Filter sind experimentell. "
        "Probiere unterschiedliche Modi aus."
    )


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎨 ColorVision</h1>
        <p>
            Experimentiere mit verschiedenen Farbfiltern.
            Vergleiche Ishihara-Tafeln und finde heraus,
            welche Farbveränderungen für dich am besten
            funktionieren.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# UPLOAD
# ==================================================

st.markdown("## 📤 Ishihara-Tafel hochladen")

uploaded_file = st.file_uploader(
    "Wähle eine Bilddatei aus",
    type=["png", "jpg", "jpeg", "webp"],
)


# ==================================================
# VERARBEITUNG
# ==================================================

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")

        processed = process_image(
            original,
            strength,
            mode,
        )

        st.markdown("## 🖼️ Bildvergleich")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### Original")
            st.image(
                original,
                use_container_width=True,
            )

        with col2:
            st.markdown(f"### Filter: {mode}")
            st.image(
                processed,
                use_container_width=True,
            )

        buffer = BytesIO()
        processed.save(buffer, format="PNG")

        st.download_button(
            "⬇️ Bearbeitetes Bild herunterladen",
            data=buffer.getvalue(),
            file_name="colorvision_result.png",
            mime="image/png",
            use_container_width=True,
        )

        st.info(
            "Die Filter verändern die Farbdarstellung, "
            "erkennen aber nicht automatisch die Zahl "
            "auf einer Ishihara-Tafel. Ein sichtbares "
            "Ergebnis ist kein medizinischer Farbsehtest."
        )

    except Exception as error:
        st.error("Das Bild konnte nicht verarbeitet werden.")
        st.code(str(error))

else:
    st.markdown(
        """
        <div class="card">
            <h3>✨ Bereit zum Testen?</h3>
            <p>
                Lade eine Ishihara-Tafel hoch und teste
                die verschiedenen Filtermodi.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# FOOTER
# ==================================================

st.markdown(
    """
    <div style="text-align:center; padding:30px 0; color:white;">
        COLORVISION · EXPERIMENTELLES PROJEKT
    </div>
    """,
    unsafe_allow_html=True,
)
