from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from google.cloud import storage
from ultralytics import YOLO

from datetime import datetime, timezone
from uuid import uuid4

# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_BUCKET = "mma-cloudproject-tfi-grupo3"
DEFAULT_MODEL_PREFIX = "models/registry"

# Cloud Run tiene filesystem efímero.
# /tmp es el lugar apropiado para descargar el modelo.
DEFAULT_LOCAL_MODEL_DIR = Path("/tmp/models/registry")


# ============================================================
# ESTRUCTURAS INTERNAS
# ============================================================

@dataclass(frozen=True)
class DetectionResult:
    class_id: int
    class_name: str
    confidence: float

    x1: float
    y1: float
    x2: float
    y2: float


@dataclass(frozen=True)
class ModelPrediction:
    image_width: int
    image_height: int

    detections: list[DetectionResult]

    model_name: str
    model_version: str
    backend: str


# ============================================================
# BACKEND YOLO
# ============================================================

class YOLOGCSBackend:
    """
    Backend de inferencia para YOLO.

    El modelo se obtiene desde Google Cloud Storage y se carga
    localmente en el contenedor de Cloud Run.
    """

    backend_name = "gcs_local"

    def __init__(
        self,
        bucket_name: str | None = None,
        model_prefix: str | None = None,
        model_version: str | None = None,
    ) -> None:

        self.bucket_name = (
            bucket_name
            or os.getenv("MODEL_BUCKET")
            or DEFAULT_BUCKET
        )

        self.model_prefix = (
            model_prefix
            or os.getenv("MODEL_PREFIX")
            or DEFAULT_MODEL_PREFIX
        )

        self.requested_version = (
            model_version
            or os.getenv("MODEL_VERSION")
        )

        self.device = os.getenv("MODEL_DEVICE", "cpu")

        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(self.bucket_name)

        # Lock para evitar que varias requests ejecuten
        # simultáneamente sobre la misma instancia de YOLO.
        self._prediction_lock = threading.Lock()

        # ----------------------------------------------------
        # Determinar versión
        # ----------------------------------------------------

        self.version_name = (
            self.requested_version
            or self._find_latest_version()
        )

        # ----------------------------------------------------
        # Descargar artefactos
        # ----------------------------------------------------

        self.local_version_dir = (
            DEFAULT_LOCAL_MODEL_DIR / self.version_name
        )

        self.local_version_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.weights_path = (
            self.local_version_dir / "model.pt"
        )

        self.metadata_path = (
            self.local_version_dir / "metadata.json"
        )

        self._download_model_files()

        # ----------------------------------------------------
        # Metadata
        # ----------------------------------------------------

        self.metadata = self._load_metadata()

        self.model_version = str(
            self.metadata.get(
                "version",
                self.version_name,
            )
        )

        self.model_name = str(
            self.metadata.get(
                "model_name",
                "yolo11n_deeppcb",
            )
        )

        # ----------------------------------------------------
        # Cargar YOLO
        # ----------------------------------------------------

        print(
            f"Cargando modelo: "
            f"{self.weights_path}"
        )

        self.model = YOLO(
            str(self.weights_path)
        )

        print(
            f"Modelo cargado: "
            f"{self.model_name} "
            f"({self.model_version})"
        )

    # ========================================================
    # VERSIONADO
    # ========================================================

    def _find_latest_version(self) -> str:
        """
        Busca la versión más reciente dentro de:

        gs://bucket/models/registry/
        """

        prefix = self.model_prefix.rstrip("/") + "/"

        blobs = self.storage_client.list_blobs(
            self.bucket_name,
            prefix=prefix,
        )

        versions = set()

        for blob in blobs:

            relative = blob.name[len(prefix):]

            if "/" not in relative:
                continue

            version = relative.split("/", 1)[0]

            if version.startswith("yolo"):
                versions.add(version)

        if not versions:
            raise FileNotFoundError(
                f"No se encontraron modelos en "
                f"gs://{self.bucket_name}/{prefix}"
            )

        latest = sorted(versions)[-1]

        print(
            f"Última versión encontrada: "
            f"{latest}"
        )

        return latest

    # ========================================================
    # DESCARGA
    # ========================================================

    def _download_blob(
        self,
        remote_path: str,
        local_path: Path,
    ) -> None:

        if local_path.exists():
            return

        blob = self.bucket.blob(remote_path)

        if not blob.exists():
            raise FileNotFoundError(
                f"No existe "
                f"gs://{self.bucket_name}/{remote_path}"
            )

        print(
            f"Descargando "
            f"gs://{self.bucket_name}/{remote_path}"
        )

        blob.download_to_filename(
            str(local_path)
        )

    def _download_model_files(self) -> None:

        remote_dir = (
            f"{self.model_prefix.rstrip('/')}/"
            f"{self.version_name}"
        )

        self._download_blob(
            f"{remote_dir}/model.pt",
            self.weights_path,
        )

        # metadata.json es recomendable pero no crítico
        try:
            self._download_blob(
                f"{remote_dir}/metadata.json",
                self.metadata_path,
            )
        except FileNotFoundError:
            print(
                "metadata.json no encontrado. "
                "Se continuará sin metadata."
            )

    # ========================================================
    # METADATA
    # ========================================================

    def _load_metadata(self) -> dict[str, Any]:

        if not self.metadata_path.exists():
            return {}

        with open(
            self.metadata_path,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    # ========================================================
    # PREDICCIÓN
    # ========================================================

    def predict_image(
        self,
        image_bytes: bytes,
        confidence: float = 0.25,
        imgsz: int = 640,
    ) -> ModelPrediction:

        # ----------------------------------------------------
        # Convertir bytes -> imagen OpenCV
        # ----------------------------------------------------

        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                "No se pudo interpretar el archivo como imagen."
            )

        height, width = image.shape[:2]

        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        with self._prediction_lock:

            results = self.model.predict(
                source=image,
                conf=confidence,
                imgsz=imgsz,
                device=self.device,
                verbose=False,
            )

        result = results[0]

        detections: list[DetectionResult] = []

        boxes = result.boxes

        if boxes is not None:

            for box in boxes:

                class_id = int(
                    box.cls[0].item()
                )

                confidence_value = float(
                    box.conf[0].item()
                )

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .tolist()
                )

                class_name = result.names[
                    class_id
                ]

                detections.append(
                    DetectionResult(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence_value,
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2),
                    )
                )

        return ModelPrediction(
            image_width=width,
            image_height=height,
            detections=detections,
            model_name=self.model_name,
            model_version=self.model_version,
            backend=self.backend_name,
            )
            
    def save_production_image(
        self,
        image_bytes: bytes,
        original_filename: str,
    ) -> str:
        """
        Guarda una imagen recibida por la API en GCS.

        Retorna la URI gs://... donde quedó almacenada.
        """

        suffix = Path(original_filename).suffix.lower()

        if not suffix:
            suffix = ".jpg"

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%d_%H%M%S_%f")

        unique_id = uuid4().hex[:8]

        blob_name = (
            f"production/images/"
            f"{timestamp}_{unique_id}{suffix}"
        )

        blob = self.bucket.blob(blob_name)

        blob.upload_from_string(
            image_bytes,
            content_type="application/octet-stream",
        )

        gcs_uri = (
            f"gs://{self.bucket_name}/{blob_name}"
        )

        print(
            f"Imagen de producción guardada: "
            f"{gcs_uri}"
        )

        return gcs_uri

# ============================================================
# FACTORY
# ============================================================

def build_backend() -> YOLOGCSBackend:

    if os.getenv(
        "BREAK_MODEL",
        "",
    ).strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):

        raise RuntimeError(
            "BREAK_MODEL activo: "
            "despliegue roto simulado."
        )

    return YOLOGCSBackend()