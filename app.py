```python
import streamlit as st
import numpy as np
import cv2
from PIL import Image

# -----------------------------
# SEITENKONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="ColorVision",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# DESIGN
# -----------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0b1020 0%, #111827 55%, #172033 100%);
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid #273449;
    }

    .hero {
        padding: 32px;
        border-radius: 24px;
        background: linear-gradient(120deg, #4338ca 0%, #7c3aed 55%, #0891b2 100%);
        margin-bottom: 28px;
        box-shadow: 0 12px 40px rgba(0,0,0,0.2);
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
        margin-top: 10px;
        margin-bottom: 0;
    }

    .section-title {
        font-size: 23px;
        font-weight: 700;
        margin: 18px 0;
        color: #f1f5f9;
    }

    .info-card {
        padding: 20px;
        border: 1px solid #334155;
        background: rgba(30, 41, 59, 0.75);
        border-radius: 18px;
        margin-bottom: 16px;
    }

    .info-card h3 {
        color: #c4b5fd;
        font-size: 17px;
        margin-top: 0;
    }

    .info-card p {
        color: #cbd5e1;
        font-size: 14px;
        line-height: 1.6;
    }

    div.stButton > button {
        border-radius: 12px;
        border: none;
        background: linear-gradient(90deg, #7c3aed, #4f46e5);
        color: white;
        font-weight: 700;
        padding: 12px 20px;
        transition: 0.2s;
    }

    div.stButton > button:hover {
        border: none;
        color: white;
        background: linear-gradient(90deg, #6d28d9, #4338ca);
        transform: translateY(-1px);
    }

    [data-testid="stFileUploader"] {
        background: rgba(30, 41, 59, 0.5);
        border: 1px dashed #64748b;
        border-radius: 16px;
        padding: 12px;
    }

    .footer {
        color: #94a3b8;
        text-align: center;
        padding: 28px 0 10px;
        font-size: 12px;
    }

    #MainMenu, footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# BILDBEARBEITUNG
# -----------------------------
def process_image(image, strength, contrast):
    img = np.array(image.convert("RGB"))

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)

    # Experimentelle Anpassung der Farbkanäle
    factor = strength / 100.0
    lab[:, :, 1] = 128 + (lab[:, :, 1] - 128) * (1 + factor)
    lab[:, :, 2] = 128 + (lab[:, :, 2] - 128) * (1 + factor)

    lab[:, :, 0] = 128 + (lab[:, :, 0] - 128) * contrast

    lab = np.clip(lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)


# -----------------------------
# SIDEBAR
# -----------------------------
with st.sidebar:
    st.markdown("## 🎨 ColorVision")
    st.caption("Dein persönlicher Farbfilter")

    st.markdown("---")
    st.markdown("### ⚙️ Einstellungen")

    strength = st.slider(
        "Farbintensität",
        min_value=0,
        max_value=100,
        value=35,
        help="Stellt ein, wie stark die Farbkanäle verändert werden."
    )

    contrast = st.slider(
        "Kontrast",
        min_value=0.5,
        max_value=1.8,
        value=1.0,
        step=0.1
    )

    st.markdown("---")
    st.markdown("### 💡 Tipp")
    st.caption(
        "Teste verschiedene Einstellungen mit Fotos, die viele unterschiedliche "
        "Farben enthalten."
    )

    st.markdown("---")
    st.caption("ColorVision · Experimenteller Prototyp")


# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>🎨 ColorVision</h1>
    <p>Entdecke Farben auf eine neue Art. Lade ein Bild hoch und experimentiere
    mit individuellen Farbfiltern.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# INFORMATIONSKARTEN
# -----------------------------
card1, card2, card3 = st.columns(3)

with card1:
    st.markdown("""
    <div class="info-card">
        <h3>🖼️ Bild hochladen</h3>
        <p>Wähle ein eigenes Foto aus und starte deinen Farbvergleich.</p>
    </div>
    """, unsafe_allow_html=True)

with card2:
    st.markdown("""
    <div class="info-card">
        <h3>🎛️ Filter anpassen</h3>
        <p>Verändere Farbintensität und Kontrast über die Einstellungen.</p>
    </div>
    """, unsafe_allow_html=True)

with card3:
    st.markdown("""
    <div class="info-card">
        <h3>🔍 Ergebnisse vergleichen</h3>
        <p>Sieh dir das Original und die bearbeitete Version nebeneinander an.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# UPLOAD
# -----------------------------
st.markdown('<div class="section-title">📤 Dein Bild</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["png", "jpg", "jpeg", "webp"],
    help="Unterstützte Formate: PNG, JPG, JPEG und WEBP."
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")
        processed = process_image(original, strength, contrast)

        st.success("Dein Bild wurde erfolgreich verarbeitet!")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.markdown("### Original")
            st.image(original, use_container_width=True)

        with col2:
            st.markdown("### ColorVision-Filter")
            st.image(processed, use_container_width=True)

        st.markdown("---")

        image_col, settings_col = st.columns([2, 1])

        with image_col:
            st.markdown("### 📥 Ergebnis speichern")
            st.caption("Lade die bearbeitete Version als PNG herunter.")

        with settings_col:
            from io import BytesIO

            buffer = BytesIO()
            processed.save(buffer, format="PNG")

            st.download_button(
                label="⬇️ Bild herunterladen",
                data=buffer.getvalue(),
                file_name="colorvision_result.png",
                mime="image/png",
                use_container_width=True
            )

        st.info(
            "Hinweis: Dieser Filter verändert die Farben eines Bildes experimentell. "
            "Er ist nicht individuell auf eine Farbsehschwäche abgestimmt und "
            "garantiert keine bessere Unterscheidbarkeit von Farben."
        )

    except Exception as e:
        st.error(f"Das Bild konnte nicht verarbeitet werden: {e}")

else:
    st.markdown("""
    <div class="info-card">
        <h3>✨ Bereit zum Start?</h3>
        <p>Lade oben ein Bild hoch. Danach erscheinen hier das Original,
        die bearbeitete Version und die Download-Funktion.</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<div class="footer">
    COLORVISION · EXPERIMENTELLES PROJEKT<br>
    Kein medizinisches Diagnose- oder Behandlungssystem.
</div>
""", unsafe_allow_html=True)
```
