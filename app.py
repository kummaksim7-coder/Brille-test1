import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO


# ==================================================
# SEITENKONFIGURATION
# ==================================================

st.set_page_config(
    page_title="ColorVision",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ==================================================
# DESIGN: WEISSER TEXT UND DUNKLER HINTERGRUND
# ==================================================

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    /* Allgemeine Schriftfarben */
    html, body, [class*="css"], p, label, span, div {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(
            135deg,
            #0b1020 0%,
            #111827 55%,
            #172033 100%
        );
        color: #ffffff;
    }

    /* Text in der gesamten App */
    .stApp,
    .stApp p,
    .stApp label,
    .stApp span,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp h4,
    .stApp li,
    .stApp strong,
    .stApp small,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] span {
        color: #ffffff;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #334155;
    }

    [data-testid="stSidebar"] * {
        color: #ffffff;
    }

    /* Überschrift */
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
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.25);
    }

    .hero h1 {
        color: #ffffff !important;
        font-size: 42px;
        font-weight: 800;
        margin: 0;
    }

    .hero p {
        color: #ffffff !important;
        font-size: 16px;
        margin-top: 12px;
        line-height: 1.8;
    }

    /* Informationskarten */
    .info-card {
        padding: 22px;
        border: 1px solid #475569;
        background: rgba(30, 41, 59, 0.85);
        border-radius: 18px;
        margin-bottom: 18px;
    }

    .info-card h3 {
        color: #ffffff !important;
        font-size: 18px;
        margin-top: 0;
    }

    .info-card p {
        color: #ffffff !important;
        font-size: 14px;
        line-height: 1.8;
    }

    /* Schaltflächen */
    div.stButton > button,
    div.stDownloadButton > button {
        border-radius: 12px;
        border: 1px solid #8b5cf6;
        background: linear-gradient(
            90deg,
            #7c3aed,
            #4f46e5
        );
        color: #ffffff !important;
        font-weight: 700;
        min-height: 45px;
    }

    div.stButton > button *,
    div.stDownloadButton > button * {
        color: #ffffff !important;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background: linear-gradient(
            90deg,
            #6d28d9,
            #4338ca
        );
        border: 1px solid #a78bfa;
    }

    /* Datei-Upload */
    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.65);
        border: 1px dashed #818cf8;
        border-radius: 16px;
        padding: 15px;
    }

    [data-testid="stFileUploader"] * {
        color: #ffffff !important;
    }

    /* Eingabefelder und Regler */
    [data-testid="stSlider"] *,
    [data-testid="stNumberInput"] *,
    [data-testid="stSelectbox"] * {
        color: #ffffff;
    }

    /* Hinweise */
    [data-testid="stAlert"] * {
        color: #ffffff !important;
    }

    /* Fußzeile */
    .footer {
        color: #ffffff !important;
        text-align: center;
        padding: 30px 0 10px;
        font-size: 12px;
        line-height: 1.8;
    }

    /* Trennlinien */
    hr {
        border-color: #475569;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# BILDBEARBEITUNG
# ==================================================

def process_image(image, strength, contrast):
    """
    Experimentelle Anpassung der Farbkanäle und des Kontrasts.
    Keine Garantie für bessere Farberkennung.
    """

    img = np.array(image.convert("RGB"))

    # RGB in LAB umwandeln
    lab = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2LAB
    ).astype(np.float32)

    # Farbintensität:
    # 0   = keine zusätzliche Verstärkung
    # 100 = Faktor 2
    # 500 = Faktor 6
    factor = strength / 100.0

    lab[:, :, 1] = (
        128 + (lab[:, :, 1] - 128) * (1 + factor)
    )

    lab[:, :, 2] = (
        128 + (lab[:, :, 2] - 128) * (1 + factor)
    )

    # Kontrast konstant auf 1,80
    lab[:, :, 0] = (
        128 + (lab[:, :, 0] - 128) * contrast
    )

    # Gültigen Wertebereich sicherstellen
    lab = np.clip(
        lab,
        0,
        255
    ).astype(np.uint8)

    # Zurück zu RGB
    result = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2RGB
    )

    return Image.fromarray(result)


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("# 🎨 ColorVision")

    st.markdown("Dein experimenteller Farbfilter")

    st.markdown("---")

    st.markdown("### ⚙️ Einstellungen")

    # Farbintensität bis 500 %
    strength = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=500,
        value=75,
        step=5,
        help=(
            "Stelle die zusätzliche Farbverstärkung "
            "zwischen 0 und 500 ein."
        ),
    )

    st.markdown(
        f"**Aktuelle Farbintensität: {strength} %**"
    )

    st.markdown("---")

    # Fester Kontrast
    contrast = 1.80

    st.markdown("### Kontrast")

    st.markdown(
        "**1,80 · Fest eingestellt**"
    )

    st.caption(
        "Der Kontrast bleibt immer auf 1,80."
    )

    st.markdown("---")

    st.markdown("### 💡 Tipp")

    st.markdown(
        "Teste unterschiedliche Farbintensitäten "
        "mit verschiedenen Bildern."
    )

    st.markdown("---")

    st.markdown("ColorVision · Version 1.2")


# ==================================================
# HEADER
# ==================================================

st.markdown(
    """
    <div class="hero">
        <h1>🎨 ColorVision</h1>
        <p>
            Entdecke Farben auf eine neue Art.<br>
            Lade ein Foto hoch, experimentiere mit
            individuellen Farbfiltern und vergleiche
            das Ergebnis mit dem Original.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ==================================================
# INFORMATIONSKARTEN
# ==================================================

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>🖼️ Bild hochladen</h3>
            <p>
                Wähle ein Foto aus und beginne
                mit deinem Farbvergleich.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>🎛️ Farben anpassen</h3>
            <p>
                Stelle die Farbintensität individuell ein.
                Der Kontrast bleibt konstant bei 1,80.
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
                Vergleiche das Original mit dem
                bearbeiteten Bild und speichere das Ergebnis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==================================================
# BILDUPLOAD
# ==================================================

st.markdown("## 📤 Dein Bild")

uploaded_file = st.file_uploader(
    "Wähle ein Bild aus",
    type=["png", "jpg", "jpeg", "webp"],
    help="Unterstützte Formate: PNG, JPG, JPEG und WEBP.",
)


# ==================================================
# BILD VERARBEITEN
# ==================================================

if uploaded_file is not None:

    try:
        original = Image.open(
            uploaded_file
        ).convert("RGB")

        processed = process_image(
            original,
            strength,
            contrast,
        )

        st.success(
            "Dein Bild wurde erfolgreich verarbeitet!"
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
            st.markdown("### ColorVision-Filter")

            st.image(
                processed,
                use_container_width=True,
            )

        st.markdown("---")

        st.markdown("## 📥 Ergebnis speichern")

        buffer = BytesIO()

        processed.save(
            buffer,
            format="PNG",
        )

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
                <h3>ℹ️ Wichtiger Hinweis</h3>
                <p>
                    Dieser Filter verändert die Farben experimentell.
                    Eine höhere Farbintensität bedeutet nicht automatisch,
                    dass Farben bei Rot-Grün-Sehschwäche besser
                    unterschieden werden können. Der Filter ist kein
                    medizinisches Diagnose- oder Behandlungssystem.
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


# ==================================================
# STARTBEREICH OHNE BILD
# ==================================================

else:

    st.markdown(
        """
        <div class="info-card">
            <h3>✨ Bereit zum Start?</h3>
            <p>
                Lade oben ein Bild hoch. Anschließend erscheinen
                hier das Original, die bearbeitete Version und
                die Download-Funktion.
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
    <div class="footer">
        COLORVISION · EXPERIMENTELLES PROJEKT<br>
        Kein medizinisches Diagnose- oder Behandlungssystem.
    </div>
    """,
    unsafe_allow_html=True,
)
