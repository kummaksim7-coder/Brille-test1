```python
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="ColorVision",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ ColorVision")
st.write(
    "Experimentelle Bildverarbeitung für Menschen "
    "mit Rot-Grün-Sehschwäche."
)

st.info(
    "Dieses Projekt verändert die Farben digital. "
    "Es ist kein medizinischer Test und ersetzt keine "
    "professionelle Untersuchung des Farbsehens."
)


def process_image(image, strength, contrast):
    """
    Experimentelle Farbtransformation.

    strength: Stärke der Farbverschiebung (0 bis 100)
    contrast: Kontrastverstärkung (0 bis 100)
    """

    # PIL-Bild in ein RGB-Array umwandeln
    rgb = np.array(image.convert("RGB"))

    # RGB -> Lab:
    # L = Helligkeit, a/b = Farbinformationen
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.float32)

    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    # Stärke normieren
    s = strength / 100.0

    # Experimentelle Farbverschiebung:
    # Die a-Achse beschreibt unter anderem den Übergang
    # zwischen Grün und Rot.
    #
    # Wir verändern diese Achse abhängig von ihrer
    # ursprünglichen Position. Das ist KEINE garantierte
    # Korrektur einer Rot-Grün-Sehschwäche.
    a_centered = a - 128.0

    a_new = (
        128.0
        + a_centered * (1.0 + 1.5 * s)
    )

    # Kontrast der Farbinformationen verstärken
    chroma_scale = 1.0 + contrast / 100.0

    b_centered = b - 128.0
    b_new = 128.0 + b_centered * chroma_scale

    # Werte in den zulässigen Bereich begrenzen
    lab[:, :, 1] = np.clip(a_new, 0, 255)
    lab[:, :, 2] = np.clip(b_new, 0, 255)

    # Zurück in RGB
    lab = lab.astype(np.uint8)
    result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return Image.fromarray(result)


uploaded_file = st.file_uploader(
    "Lade ein Bild hoch",
    type=["png", "jpg", "jpeg", "webp"]
)

if uploaded_file is not None:
    try:
        original = Image.open(uploaded_file).convert("RGB")

        st.subheader("Einstellungen")

        col1, col2 = st.columns(2)

        with col1:
            strength = st.slider(
                "Stärke der Farbverschiebung",
                min_value=0,
                max_value=100,
                value=40,
                help="Stellt ein, wie stark die Rot-Grün-Farbachse "
                     "verändert wird."
            )

        with col2:
            contrast = st.slider(
                "Farbkontrast",
                min_value=0,
                max_value=100,
                value=20,
                help="Verändert den Kontrast entlang der "
                     "zweiten Farbachse."
            )

        processed = process_image(
            original,
            strength,
            contrast
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original")
            st.image(
                original,
                use_container_width=True
            )

        with col2:
            st.subheader("Veränderte Farben")
            st.image(
                processed,
                use_container_width=True
            )

        # Bearbeitetes Bild zum Download bereitstellen
        buffer = BytesIO()
        processed.save(buffer, format="PNG")

        st.download_button(
            label="Bearbeitetes Bild herunterladen",
            data=buffer.getvalue(),
            file_name="colorvision_result.png",
            mime="image/png"
        )

        st.caption(
            "Hinweis: Diese experimentelle Transformation "
            "kann Farben verändern, ohne die Unterscheidbarkeit "
            "für eine bestimmte Person zu verbessern."
        )

    except Exception as error:
        st.error(f"Das Bild konnte nicht verarbeitet werden: {error}")

else:
    st.write(
        "Lade ein Bild hoch, um die Farbtransformation "
        "auszuprobieren."
    )
```
