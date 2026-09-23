"""
Chequeo de drift offline para PCB Defect Detection.

Compara características visuales de las imágenes utilizadas como referencia
con una ventana de imágenes de producción.

Se utiliza Population Stability Index (PSI).

Variables monitoreadas:
    - brightness: brillo medio
    - contrast: contraste
    - sharpness: nitidez
    - width: ancho
    - height: alto

Interpretación orientativa:
    PSI < 0.10      estable
    PSI 0.10-0.25   atención
    PSI > 0.25      drift fuerte

Uso:

    # Referencia vs ventana simulada
    python scripts/check_drift.py

    # Referencia vs imágenes reales de producción
    python scripts/check_drift.py \
        --current data/production/images

    # Elegir otra referencia
    python scripts/check_drift.py \
        --reference data/processed/deep_pcb_yolo/images/train \
        --current data/production/images
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_REFERENCE = (
    ROOT
    / "data"
    / "processed"
    / "deep_pcb_yolo"
    / "images"
    / "train"
)

DEFAULT_FEATURES = [
    "brightness",
    "contrast",
    "sharpness",
    "width",
    "height",
]

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

EPS = 1e-6


# ============================================================
# INTERPRETACIÓN PSI
# ============================================================

def verdict(psi: float) -> str:

    if psi < 0.10:
        return "estable"

    if psi < 0.25:
        return "atencion"

    return "DRIFT"


# ============================================================
# EXTRACCIÓN DE CARACTERÍSTICAS DE IMAGEN
# ============================================================

def extract_image_features(
    image_path: Path,
) -> dict | None:

    image = cv2.imread(str(image_path))

    if image is None:
        print(
            f"Advertencia: no se pudo leer "
            f"{image_path}"
        )
        return None

    height, width = image.shape[:2]

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    brightness = float(
        np.mean(gray)
    )

    contrast = float(
        np.std(gray)
    )

    sharpness = float(
        cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()
    )

    return {
        "image": str(image_path),
        "brightness": brightness,
        "contrast": contrast,
        "sharpness": sharpness,
        "width": width,
        "height": height,
    }


def extract_dataset_features(
    directory: Path,
) -> pd.DataFrame:

    if not directory.exists():
        raise FileNotFoundError(
            f"No existe el directorio: {directory}"
        )

    image_paths = [
        path
        for path in directory.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    if not image_paths:
        raise ValueError(
            f"No se encontraron imágenes en "
            f"{directory}"
        )

    rows = []

    for image_path in image_paths:

        features = extract_image_features(
            image_path
        )

        if features is not None:
            rows.append(features)

    if not rows:
        raise ValueError(
            "No se pudo procesar ninguna imagen."
        )

    return pd.DataFrame(rows)


# ============================================================
# PSI
# ============================================================

def psi_from_props(
    ref_prop: pd.Series,
    cur_prop: pd.Series,
) -> float:

    total = 0.0

    for bin_key in ref_prop.index:

        ref_p = max(
            float(
                ref_prop.get(
                    bin_key,
                    0.0,
                )
            ),
            EPS,
        )

        cur_p = max(
            float(
                cur_prop.get(
                    bin_key,
                    0.0,
                )
            ),
            EPS,
        )

        total += (
            cur_p - ref_p
        ) * math.log(
            cur_p / ref_p
        )

    return total


def psi_numeric(
    ref: pd.Series,
    cur: pd.Series,
    bins: int = 10,
) -> float:

    ref = pd.to_numeric(
        ref,
        errors="coerce",
    ).dropna()

    cur = pd.to_numeric(
        cur,
        errors="coerce",
    ).dropna()

    quantiles = [
        i / bins
        for i in range(bins + 1)
    ]

    edges = sorted(
        set(
            ref.quantile(
                quantiles
            ).tolist()
        )
    )

    # Variable prácticamente constante
    if len(edges) < 2:
        return 0.0

    edges[0] = -math.inf
    edges[-1] = math.inf

    ref_binned = pd.cut(
        ref,
        bins=edges,
        include_lowest=True,
    )

    cur_binned = pd.cut(
        cur,
        bins=edges,
        include_lowest=True,
    )

    ref_prop = (
        ref_binned
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    cur_prop = (
        cur_binned
        .value_counts(
            normalize=True,
            sort=False,
        )
    )

    return psi_from_props(
        ref_prop,
        cur_prop,
    )


# ============================================================
# SIMULACIÓN DE PRODUCCIÓN
# ============================================================

def simulate_production(
    reference: pd.DataFrame,
    seed: int = 7,
) -> pd.DataFrame:
    """
    Genera una ventana sintética con cambios visuales.

    Sirve solamente para demostrar el mecanismo de drift
    cuando todavía no existen imágenes reales de producción.
    """

    sample = reference.sample(
        n=len(reference),
        replace=True,
        random_state=seed,
    ).copy()

    rng = np.random.default_rng(seed)

    # Simulamos imágenes algo más claras
    sample["brightness"] *= 1.15

    # Algo más de contraste
    sample["contrast"] *= 1.10

    # Algo más borrosas
    sample["sharpness"] *= 0.60

    # Pequeña variabilidad
    sample["brightness"] += rng.normal(
        0,
        3,
        len(sample),
    )

    return sample


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Chequeo de drift PSI "
            "para PCB Defect Detection."
        )
    )

    parser.add_argument(
        "--reference",
        type=Path,
        default=DEFAULT_REFERENCE,
        help=(
            "Carpeta de imágenes de referencia. "
            "Por defecto usa images/train."
        ),
    )

    parser.add_argument(
        "--current",
        type=Path,
        default=None,
        help=(
            "Carpeta con imágenes actuales de producción. "
            "Si no se pasa, se simula drift."
        ),
    )

    parser.add_argument(
        "--features",
        nargs="*",
        default=DEFAULT_FEATURES,
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Referencia
    # --------------------------------------------------------

    print(
        "Extrayendo características "
        "de referencia..."
    )

    reference = extract_dataset_features(
        args.reference
    )

    # --------------------------------------------------------
    # Producción
    # --------------------------------------------------------

    if args.current is not None:

        print(
            "Extrayendo características "
            "de producción..."
        )

        current = extract_dataset_features(
            args.current
        )

        origen = str(args.current)

    else:

        current = simulate_production(
            reference
        )

        origen = (
            "ventana simulada "
            "(más clara y menos nítida)"
        )

    # --------------------------------------------------------
    # Resumen
    # --------------------------------------------------------

    print()
    print(
        f"Referencia : "
        f"{args.reference} "
        f"({len(reference)} imágenes)"
    )

    print(
        f"Producción : "
        f"{origen} "
        f"({len(current)} imágenes)"
    )

    print()

    print(
        f"  {'feature':<16} "
        f"{'PSI':>8}   "
        f"veredicto"
    )

    print(
        f"  {'-' * 16} "
        f"{'-' * 8}   "
        f"{'-' * 9}"
    )

    peor = 0.0

    # --------------------------------------------------------
    # PSI por feature
    # --------------------------------------------------------

    for feature in args.features:

        if (
            feature not in reference.columns
            or feature not in current.columns
        ):

            print(
                f"  {feature:<16} "
                f"{'--':>8}   "
                f"(no disponible)"
            )

            continue

        psi = psi_numeric(
            reference[feature],
            current[feature],
        )

        peor = max(
            peor,
            psi,
        )

        print(
            f"  {feature:<16} "
            f"{psi:8.3f}   "
            f"{verdict(psi)}"
        )

    # --------------------------------------------------------
    # Resultado global
    # --------------------------------------------------------

    print()

    if peor >= 0.25:

        print(
            f"  -> PSI máximo {peor:.3f}: "
            "hay drift visual."
        )

        print(
            "     Revisar las imágenes nuevas "
            "y evaluar el desempeño del modelo."
        )

    elif peor >= 0.10:

        print(
            f"  -> PSI máximo {peor:.3f}: "
            "la distribución comenzó a moverse."
        )

        print(
            "     Conviene continuar monitoreando."
        )

    else:

        print(
            f"  -> PSI máximo {peor:.3f}: "
            "distribución estable."
        )


if __name__ == "__main__":
    main()