import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


# --------------------------------------------------
# SEITENKONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ColorVision",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 55%,
            #172033 100%
        );
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #273449;
    }

    .hero {
        padding: 35px;
        border-radius: 24px;
        background: linear-gradient(
            120deg,
            #4338ca 0%,
            #7c3aed 55%,
            #0891b2 100%
        );
        margin-bottom: 28px;
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.2);
    }

    .hero h1 {
        color: white;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #e0e7ff;
        font-size: 16px;
        margin-top: 12px;
        line-height: 1.7;
    }

    .section-title {
        font-size: 25px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 18px;
    }

    .info-card {
        padding: 22px;
        border: 1px solid #334155;
        background: rgba(30, 41, 59, 0.75);
        border-radius: 18px;
        margin-bottom: 18px;
    }

    .info-card h3 {
        color: #c4b5fd;
        font-size: 18px;
        margin-top: 0;
    }

    .info-card p {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.7;
    }

    div.stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5
        );
        color: white;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button:hover {
        color: white;
        border: none;
        background: linear-gradient(
            90deg,
            #6d28d9,
            #4338ca
        );
    }

    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px dashed #64748b;
        border-radius: 16px;
        padding: 15px;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        padding: 30px 0 10px;
        font-size: 12px;
        line-height: 1.8;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# BILDBEARBEITUNG
# --------------------------------------------------

def process_image(image, strength, contrast):
    """
    Experimentelle Farb- und Kontrastanpassung.
    Diese Funktion garantiert keine bessere Farberkennung.
    """

    img = np.array(image.convert("RGB"))

    # RGB in den LAB-Farbraum umwandeln
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(
        np.float32
    )

    # Farbintensität anpassen
    factor = strength / 100.0

    lab[:, :, 1] = (
        128 + (lab[:, :, 1] - 128) * (1 + factor)
    )

    lab[:, :, 2] = (
        128 + (lab[:, :, 2] - 128) * (1 + factor)
    )

    # Helligkeitskontrast anpassen
    lab[:, :, 0] = (
        128 + (lab[:, :, 0] - 128) * contrast
    )

    # Werte auf den gültigen Bereich begrenzen
    lab = np.clip(lab, 0, 255).astype(np.uint8)

    # Zurück in RGB umwandeln
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown("# 🎨 ColorVision")

    st.caption("Dein experimenteller Farbfilter")

    st.markdown("---")

    st.markdown("### ⚙️ Einstellungen")

    strength = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=100,
        value=35,
        step=5,
        help="Bestimmt, wie stark die Farbkanäle verändert werden.",
    )

    contrast = st.slider(
        "Kontrast",
        min_value=0.5,
        max_value=1.8,
        value=1.0,
        step=0.1,
        help="Passt den Helligkeitskontrast des Bildes an.",
    )

    st.markdown("---")

    st.markdown("### 💡 Tipp")

    st.caption(
        "Probiere verschiedene Einstellungen aus. "
        "Je nach Bild können die Ergebnisse unterschiedlich wirken."
    )

    st.markdown("---")

    st.caption("ColorVision · Version 1.0")


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🎨 ColorVision</h1>
        <p>
            Entdecke Farben auf eine neue Art.<br>
            Lade ein Foto hoch, experimentiere mit Farbfiltern
            und vergleiche das Ergebnis mit dem Original.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# INFORMATIONSKARTEN
# --------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🖼️ Bild hochladen</h3>
            <p>
                Wähle ein Foto von deinem Computer aus
                und beginne mit dem Farbvergleich.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎛️ Filter einstellen</h3>
            <p>
                Passe Farbintensität und Kontrast
                individuell über die Regler an.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="info-card">
            <h3>🔍 Ergebnisse vergleichen</h3>
            <p>
                Sieh dir das Original und die
                bearbeitete Version nebeneinander an.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# BILDUpload
# --------------------------------------------------

st.markdown(
    '<div class="section-title">📤 Dein Bild</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Wähle ein Bild aus",
    type=["png", "jpg", "jpeg", "webp"],
    help="Unterstützte Dateiformate: PNG, JPG, JPEG und WEBP.",
)


# --------------------------------------------------
# BILD VERARBEITEN
# --------------------------------------------------

if uploaded_file is not None:

    try:
        original = Image.open(uploaded_file).convert("RGB")

        # Verarbeitetes Bild erstellen
        processed = process_image(
            original,
            strength,
            contrast,
        )

        st.success("Dein Bild wurde verarbeitet!")

        st.markdown("### 🖼️ Bildvergleich")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("#### Original")
            st.image(
                original,
                use_container_width=True,
            )

        with col2:
            st.markdown("#### ColorVision-Filter")
            st.image(
                processed,
                use_container_width=True,
            )

        # Download vorbereiten
        buffer = BytesIO()

        processed.save(
            buffer,
            format="PNG",
        )

        st.markdown("---")

        st.markdown("### 📥 Ergebnis speichern")

        st.download_button(
            label="⬇️ Bearbeitetes Bild herunterladen",
            data=buffer.getvalue(),
            file_name="colorvision_result.png",
            mime="image/png",
            use_container_width=True,
        )

        st.markdown(
            """
            <div class="info-card">
                <h3>ℹ️ Hinweis</h3>
                <p>
                    Der Farbfilter verändert die Darstellung der Farben
                    experimentell. Er wurde nicht individuell für eine
                    bestimmte Farbsehschwäche entwickelt und garantiert
                    keine bessere Unterscheidbarkeit von Farben.
                    Er eignet sich nicht zur medizinischen Diagnose.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    except Exception as error:
        st.error(
            "Das Bild konnte nicht verarbeitet werden."
        )

        st.code(str(error))


# --------------------------------------------------
# STARTBEREICH, WENN KEIN BILD HOCHGELADEN WURDE
# --------------------------------------------------

else:
    st.markdown(
        """
        <div class="info-card">
            <h3>✨ Bereit zum Start?</h3>
            <p>
                Lade oben ein Bild hoch. Anschließend kannst du
                das Original mit der bearbeiteten Version vergleichen
                und das Ergebnis herunterladen.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        COLORVISION · EXPERIMENTELLES PROJEKT<br>
        Kein medizinisches Diagnose- oder Behandlungssystem.
    </div>
    """,
    unsafe_allow_html=True,
)
