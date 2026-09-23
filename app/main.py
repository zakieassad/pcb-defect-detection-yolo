from __future__ import annotations

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
)

from app.model_backend import (
    ModelPrediction,
    build_backend,
)

from app.observability import (
    configure_logging,
    log_event,
    measure_latency,
)

from app.schemas import (
    BoundingBox,
    Detection,
    PredictionResponse,
)


# ============================================================
# API
# ============================================================

app = FastAPI(
    title="PCB Defect Detection API",
    version="0.1.0",
    description=(
        "API para detección automática de defectos "
        "en PCB mediante YOLO11."
    ),
)

configure_logging()

_backend = None


# ============================================================
# BACKEND
# ============================================================

def get_backend():

    global _backend

    if _backend is None:
        _backend = build_backend()

    return _backend


# ============================================================
# CONVERSIÓN RESPUESTA
# ============================================================

def to_response(
    filename: str,
    prediction: ModelPrediction,
) -> PredictionResponse:

    detections = [

        Detection(
            class_id=item.class_id,
            class_name=item.class_name,
            confidence=round(
                item.confidence,
                6,
            ),
            bbox=BoundingBox(
                x1=round(item.x1, 2),
                y1=round(item.y1, 2),
                x2=round(item.x2, 2),
                y2=round(item.y2, 2),
            ),
        )

        for item in prediction.detections
    ]

    return PredictionResponse(
        filename=filename,
        image_width=prediction.image_width,
        image_height=prediction.image_height,
        detection_count=len(detections),
        detections=detections,
        backend=prediction.backend,
        model_name=prediction.model_name,
        model_version=prediction.model_version,
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health() -> dict:

    try:

        backend = get_backend()

        return {
            "status": "ok",
            "backend": backend.backend_name,
            "model_loaded": True,
            "model_name": backend.model_name,
            "model_version": backend.model_version,
        }

    except Exception as exc:

        return {
            "status": "degraded",
            "backend": "unavailable",
            "model_loaded": False,
            "detail": str(exc),
        }


# ============================================================
# PREDICCIÓN INDIVIDUAL
# ============================================================

def score_image(
    filename: str,
    image_bytes: bytes,
    conf: float,
    imgsz: int,
) -> PredictionResponse:

    try:

        prediction = get_backend().predict_image(
            image_bytes=image_bytes,
            confidence=conf,
            imgsz=imgsz,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    return to_response(
        filename,
        prediction,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),

    conf: float = Query(
        default=0.25,
        ge=0.0,
        le=1.0,
    ),

    imgsz: int = Query(
        default=640,
        ge=32,
        le=2048,
    ),
) -> PredictionResponse:

    if (
        file.content_type
        and not file.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=400,
            detail="El archivo debe ser una imagen.",
        )

    image_bytes = await file.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="La imagen está vacía.",
        )

    backend = get_backend()

    # --------------------------------------------------------
    # Guardar imagen de producción en GCS
    # --------------------------------------------------------

    try:
        gcs_uri = backend.save_production_image(
            image_bytes=image_bytes,
            original_filename=file.filename or "image.jpg",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"No se pudo guardar la imagen en GCS: {exc}",
        ) from exc

    # --------------------------------------------------------
    # Predicción
    # --------------------------------------------------------

    with measure_latency() as timer:

        response = score_image(
            filename=file.filename or "image",
            image_bytes=image_bytes,
            conf=conf,
            imgsz=imgsz,
        )

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------

    log_event(
        "prediction",
        latency_ms=timer["latency_ms"],
        filename=file.filename,
        production_image_uri=gcs_uri,
        detection_count=response.detection_count,
        detected_classes=[
            item.class_name
            for item in response.detections
        ],
        model_version=response.model_version,
        backend=response.backend,
    )

    return response