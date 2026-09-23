from __future__ import annotations

import io
import os
from typing import Any

import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageFont


API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/predict"
)

def draw_detections(image: Image.Image, detections: list[dict[str, Any]]) -> Image.Image:
    """
    Dibuja bounding boxes y etiquetas sobre la imagen.
    """
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    for det in detections:
        bbox = det["bbox"]
        class_name = det["class_name"]
        confidence = det["confidence"]

        x1 = bbox["x1"]
        y1 = bbox["y1"]
        x2 = bbox["x2"]
        y2 = bbox["y2"]

        label = f"{class_name} ({confidence:.2f})"

        # Bounding box
        draw.rectangle(
            [(x1, y1), (x2, y2)],
            outline="red",
            width=3,
        )

        # Fondo para texto
        if font is not None:
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            tx1, ty1, tx2, ty2 = text_bbox
            draw.rectangle(
                [(tx1, ty1 - 2), (tx2 + 4, ty2 + 2)],
                fill="red",
            )
            draw.text((x1 + 2, y1 - 1), label, fill="white", font=font)
        else:
            draw.text((x1, y1), label, fill="white")

    return annotated


def call_api(image_bytes: bytes, filename: str, mime_type: str, conf: float, imgsz: int) -> dict:
    files = {
        "file": (filename, image_bytes, mime_type),
    }

    response = requests.post(
        f"{API_URL}?conf={conf}&imgsz={imgsz}",
        files=files,
        timeout=180,
    )

    if response.status_code != 200:
        raise RuntimeError(f"Error {response.status_code}: {response.text}")

    return response.json()


st.set_page_config(
    page_title="PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 PCB Defect Detection")
st.caption("Frontend para detección de defectos en placas PCB usando YOLO")

with st.sidebar:
    st.header("Configuración")

    api_url_input = st.text_input("API URL", value=API_URL)
    conf = st.slider("Confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.01)
    imgsz = st.selectbox("Image size", options=[320, 416, 512, 640, 768, 960], index=3)

    API_URL = api_url_input

    st.markdown("---")
    st.write("Subí una imagen PCB y presioná **Detect defects**.")

uploaded_file = st.file_uploader(
    "Seleccionar imagen PCB",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Imagen original")
        st.image(image, use_container_width=True)

    if st.button("Detect defects", type="primary"):
        try:
            with st.spinner("Ejecutando inferencia..."):
                result = call_api(
                    image_bytes=image_bytes,
                    filename=uploaded_file.name,
                    mime_type=uploaded_file.type or "image/jpeg",
                    conf=conf,
                    imgsz=imgsz,
                )

            detections = result.get("detections", [])
            annotated = draw_detections(image, detections)

            with col2:
                st.subheader("Imagen con defectos marcados")
                st.image(annotated, use_container_width=True)

            st.markdown("## Resultado")

            m1, m2, m3 = st.columns(3)
            m1.metric("Defectos detectados", result["detection_count"])
            m2.metric("Modelo", result["model_name"])
            m3.metric("Versión", result["model_version"])

            if detections:
                rows = []
                for det in detections:
                    rows.append(
                        {
                            "class_id": det["class_id"],
                            "class_name": det["class_name"],
                            "confidence": round(det["confidence"], 4),
                            "x1": det["bbox"]["x1"],
                            "y1": det["bbox"]["y1"],
                            "x2": det["bbox"]["x2"],
                            "y2": det["bbox"]["y2"],
                        }
                    )

                st.subheader("Detecciones")
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No se detectaron defectos en la imagen.")

        except Exception as exc:
            st.error(f"No se pudo procesar la imagen: {exc}")