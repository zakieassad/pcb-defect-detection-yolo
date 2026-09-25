import argparse
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from google.cloud import storage


# ============================================================
# CONFIGURACIÓN
# ============================================================

DEFAULT_REFERENCE_DIR = Path(
    "data/processed/deep_pcb_yolo/images/train"
)

DEFAULT_BUCKET = "mma-cloudproject-tfi-grupo3"
DEFAULT_PRODUCTION_PREFIX = "production/images"

FEATURES = [
    "brightness",
    "contrast",
    "sharpness",
    "width",
    "height",
]


# ============================================================
# EXTRACCIÓN DE FEATURES
# ============================================================

def extract_image_features(image_path: Path) -> dict | None:
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape

    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "filename": image_path.name,
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "width": width,
        "height": height,
    }


def extract_features_from_folder(folder: Path) -> pd.DataFrame:
    rows = []

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
    }

    files = [
        p
        for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions
    ]

    for image_path in files:
        row = extract_image_features(image_path)

        if row is not None:
            rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# DESCARGA DE IMÁGENES DE PRODUCCIÓN DESDE GCS
# ============================================================

def download_production_images(
    bucket_name: str,
    prefix: str,
    destination: Path,
    limit: int | None = None,
) -> int:

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    blobs = list(
        client.list_blobs(
            bucket_name,
            prefix=prefix,
        )
    )

    valid_blobs = [
        blob
        for blob in blobs
        if Path(blob.name).suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp"}
    ]

    # Usamos las imágenes más recientes
    valid_blobs.sort(
        key=lambda blob: blob.updated or blob.time_created,
        reverse=True,
    )

    if limit is not None:
        valid_blobs = valid_blobs[:limit]

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for blob in valid_blobs:
        local_path = destination / Path(blob.name).name

        blob.download_to_filename(
            str(local_path)
        )

    return len(valid_blobs)


# ============================================================
# PSI
# ============================================================

def calculate_psi(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
) -> float:

    expected = expected.dropna().astype(float)
    actual = actual.dropna().astype(float)

    if expected.empty or actual.empty:
        return math.nan

    # Límites definidos a partir de la distribución
    # de referencia.
    breakpoints = np.quantile(
        expected,
        np.linspace(0, 1, bins + 1),
    )

    # Evitamos límites repetidos.
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 3:
        return 0.0

    expected_counts, _ = np.histogram(
        expected,
        bins=breakpoints,
    )

    actual_counts, _ = np.histogram(
        actual,
        bins=breakpoints,
    )

    expected_pct = (
        expected_counts /
        max(expected_counts.sum(), 1)
    )

    actual_pct = (
        actual_counts /
        max(actual_counts.sum(), 1)
    )

    epsilon = 1e-6

    expected_pct = np.clip(
        expected_pct,
        epsilon,
        None,
    )

    actual_pct = np.clip(
        actual_pct,
        epsilon,
        None,
    )

    psi = np.sum(
        (actual_pct - expected_pct)
        * np.log(
            actual_pct / expected_pct
        )
    )

    return float(psi)


# ============================================================
# INTERPRETACIÓN
# ============================================================

def classify_psi(psi: float) -> str:
    if math.isnan(psi):
        return "SIN DATOS"

    if psi < 0.10:
        return "ESTABLE"

    if psi < 0.25:
        return "ATENCIÓN"

    return "DRIFT"


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Monitoreo de input drift para imágenes de PCB."
        )
    )

    parser.add_argument(
        "--reference",
        default=str(DEFAULT_REFERENCE_DIR),
        help="Carpeta con imágenes de referencia.",
    )

    parser.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help="Bucket de Google Cloud Storage.",
    )

    parser.add_argument(
        "--production-prefix",
        default=DEFAULT_PRODUCTION_PREFIX,
        help="Prefijo de imágenes de producción en GCS.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help=(
            "Cantidad máxima de imágenes de producción "
            "a utilizar."
        ),
    )

    parser.add_argument(
        "--output",
        default="drift_report.csv",
        help="Archivo CSV de salida.",
    )

    args = parser.parse_args()

    reference_dir = Path(args.reference)

    if not reference_dir.exists():
        raise FileNotFoundError(
            f"No existe la carpeta de referencia: "
            f"{reference_dir}"
        )

    print("=" * 70)
    print("PCB INPUT DRIFT CHECK")
    print("=" * 70)

    print()
    print(
        f"Referencia: {reference_dir}"
    )

    reference_df = extract_features_from_folder(
        reference_dir
    )

    print(
        f"Imágenes de referencia: "
        f"{len(reference_df)}"
    )

    if reference_df.empty:
        raise RuntimeError(
            "No se encontraron imágenes "
            "válidas de referencia."
        )

    # Directorio temporal para producción
    with tempfile.TemporaryDirectory() as tmp_dir:
        production_dir = Path(tmp_dir)

        print()
        print(
            "Descargando imágenes de producción "
            "desde GCS..."
        )

        n_downloaded = download_production_images(
            bucket_name=args.bucket,
            prefix=args.production_prefix,
            destination=production_dir,
            limit=args.limit,
        )

        print(
            f"Imágenes descargadas: "
            f"{n_downloaded}"
        )

        if n_downloaded == 0:
            raise RuntimeError(
                "No se encontraron imágenes "
                "de producción en GCS."
            )

        production_df = (
            extract_features_from_folder(
                production_dir
            )
        )

        print(
            f"Imágenes válidas de producción: "
            f"{len(production_df)}"
        )

        results = []

        for feature in FEATURES:
            psi = calculate_psi(
                reference_df[feature],
                production_df[feature],
            )

            status = classify_psi(psi)

            results.append(
                {
                    "feature": feature,
                    "psi": psi,
                    "status": status,
                    "reference_mean": (
                        reference_df[feature].mean()
                    ),
                    "production_mean": (
                        production_df[feature].mean()
                    ),
                }
            )

        report_df = pd.DataFrame(results)

        print()
        print("=" * 70)
        print("RESULTADOS")
        print("=" * 70)
        print()

        print(
            report_df.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )

        report_df.to_csv(
            args.output,
            index=False,
        )

        print()
        print(
            f"Reporte guardado en: "
            f"{args.output}"
        )

        print()

        drift_features = report_df[
            report_df["status"] == "DRIFT"
        ]

        if drift_features.empty:
            print(
                "Resultado general: "
                "no se detectó drift crítico."
            )
        else:
            print(
                "Resultado general: "
                "se detectó drift en:"
            )

            for feature in drift_features[
                "feature"
            ]:
                print(
                    f" - {feature}"
                )


if __name__ == "__main__":
    main()