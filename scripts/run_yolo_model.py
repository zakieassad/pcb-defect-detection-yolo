from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_BUCKET = "mma-cloudproject-tfi-grupo3"
DEFAULT_BUCKET_PREFIX = "models/registry"
DEFAULT_REGISTRY_DIR = ROOT / "models" / "registry"


# ------------------------------------------------------------------
# Resolución de la versión del modelo (local -> bucket como fallback)
# ------------------------------------------------------------------

def find_latest_local_version(registry_dir: Path) -> str | None:
    """Busca la carpeta de versión más reciente dentro de models/registry/."""
    if not registry_dir.exists():
        return None
    versions = sorted(p.name for p in registry_dir.iterdir() if p.is_dir())
    return versions[-1] if versions else None


def list_bucket_versions(bucket: str, bucket_prefix: str) -> list[str]:
    """Lista las carpetas de versión disponibles en gs://<bucket>/<bucket_prefix>/."""
    uri = f"gs://{bucket}/{bucket_prefix.rstrip('/')}/"
    try:
        output = subprocess.check_output(["gcloud", "storage", "ls", uri], stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

    versions = []
    for line in output.decode().splitlines():
        line = line.strip()
        if line.endswith("/"):
            versions.append(line.rstrip("/").rsplit("/", 1)[-1])
    return sorted(versions)


def download_version_from_bucket(bucket: str, bucket_prefix: str, version_name: str, registry_dir: Path) -> None:
    """Descarga models/registry/<version_name>/ desde el bucket a disco local."""
    uri = f"gs://{bucket}/{bucket_prefix.rstrip('/')}/{version_name}"
    registry_dir.mkdir(parents=True, exist_ok=True)
    print(f"Descargando modelo desde el bucket: {uri} -> {registry_dir}")
    subprocess.run(["gcloud", "storage", "cp", "-r", uri, str(registry_dir)], check=True)


def resolve_version_dir(
    version_name: str | None,
    registry_dir: Path,
    bucket: str,
    bucket_prefix: str,
    allow_bucket: bool,
) -> Path:
    """Determina la carpeta de la versión a usar, descargándola del bucket si hace falta."""
    if version_name is None:
        version_name = find_latest_local_version(registry_dir)

    if version_name is not None:
        version_dir = registry_dir / version_name
        if (version_dir / "model.pt").exists():
            return version_dir
        if not allow_bucket:
            raise FileNotFoundError(f"No se encontró model.pt en {version_dir} y --skip-bucket está activo.")
        download_version_from_bucket(bucket, bucket_prefix, version_name, registry_dir)
        return version_dir

    # No hay ninguna versión local: buscamos la más reciente en el bucket.
    if not allow_bucket:
        raise FileNotFoundError(f"No hay versiones locales en {registry_dir} y --skip-bucket está activo.")

    bucket_versions = list_bucket_versions(bucket, bucket_prefix)
    if not bucket_versions:
        raise FileNotFoundError(
            f"No se encontraron versiones ni en {registry_dir} ni en "
            f"gs://{bucket}/{bucket_prefix}/. Pasá --version explícitamente."
        )

    version_name = bucket_versions[-1]
    download_version_from_bucket(bucket, bucket_prefix, version_name, registry_dir)
    return registry_dir / version_name


# ------------------------------------------------------------------
# Metadatos
# ------------------------------------------------------------------

def load_metadata(json_path: Path | None, joblib_path: Path | None) -> dict:
    """Carga los metadatos del modelo. Prioriza el .json; si no existe, cae al .joblib.

    Soporta tanto un .joblib que sea directamente el dict de metadata, como uno que
    envuelva el modelo completo junto con la metadata (formato {"model": ..., "metadata": {...}}).
    """
    if json_path and json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    if joblib_path and joblib_path.exists():
        loaded = joblib.load(joblib_path)
        if isinstance(loaded, dict) and "metadata" in loaded and isinstance(loaded["metadata"], dict):
            return loaded["metadata"]
        if isinstance(loaded, dict):
            return loaded

    return {}


def print_metadata_summary(metadata: dict) -> None:
    if not metadata:
        return
    print("Metadatos del modelo cargado:")
    print(f"  Modelo:  {metadata.get('model_name', 'desconocido')}")
    print(f"  Versión: {metadata.get('version', 'desconocida')}")
    metrics = metadata.get("metrics") or metadata.get("metrics_overall")
    if metrics:
        print(f"  Métricas de entrenamiento: {metrics}")
    print()


# ------------------------------------------------------------------
# Inferencia
# ------------------------------------------------------------------

def run_inference(
    model: YOLO,
    source: Path,
    output_dir: Path,
    conf: float,
    imgsz: int,
    device: str,
) -> None:
    """Corre el modelo sobre una imagen o carpeta de imágenes nuevas."""
    results = model.predict(
        source=str(source),
        conf=conf,
        imgsz=imgsz,
        device=device,
        save=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
    )

    predictions_dir = output_dir / "predictions"
    print(f"\nProcesadas {len(results)} imagen(es). Resultados guardados en: {predictions_dir}\n")

    for result in results:
        image_name = Path(result.path).name
        class_names = result.names
        counts: dict[str, int] = {}
        boxes = result.boxes
        if boxes is not None:
            for cls_id in boxes.cls.tolist():
                class_name = class_names[int(cls_id)]
                counts[class_name] = counts.get(class_name, 0) + 1

        detections = ", ".join(f"{name}: {qty}" for name, qty in counts.items()) if counts else "sin detecciones"
        print(f"- {image_name}: {detections}")


# ------------------------------------------------------------------
# Validación / métricas
# ------------------------------------------------------------------

def extract_val_metrics(results, model_name: str) -> dict:
    box = results.box
    return {
        "model": model_name,
        "precision": float(box.mp),
        "recall": float(box.mr),
        "mAP50": float(box.map50),
        "mAP50_95": float(box.map),
    }


def extract_val_metrics_by_class(results, class_names: dict[int, str], model_name: str) -> list[dict]:
    box = results.box
    rows = []
    for idx, class_name in class_names.items():
        rows.append(
            {
                "model": model_name,
                "class_id": idx,
                "class_name": class_name,
                "precision": float(box.p[idx]),
                "recall": float(box.r[idx]),
                "mAP50": float(box.ap50[idx]),
                "mAP50_95": float(box.ap[idx]),
            }
        )
    return rows


def run_validation(
    model: YOLO,
    data_yaml: Path,
    output_dir: Path,
    metrics_path: Path,
    imgsz: int,
    split: str,
    device: str,
    model_name: str,
) -> None:
    """Evalúa el modelo contra un split con labels (train/val/test) del data.yaml."""
    results = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name="validation",
        exist_ok=True,
    )

    overall = extract_val_metrics(results, model_name)
    by_class = extract_val_metrics_by_class(results, results.names, model_name)

    metrics = {
        "metrics_overall": overall,
        "metrics_by_class": by_class,
    }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("\nMétricas overall:")
    print(json.dumps(overall, indent=2, ensure_ascii=False))
    print(f"\nMétricas guardadas en: {metrics_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Corre el modelo YOLO ya entrenado (inferencia sobre imágenes nuevas o validación con métricas)."
    )
    parser.add_argument(
        "--mode",
        choices=["predict", "val"],
        required=True,
        help="predict: corre inferencia sobre imágenes nuevas. val: evalúa el modelo contra un split con labels (data.yaml).",
    )

    # Selección de versión del modelo (convención models/registry/<version>/).
    parser.add_argument("--version", default=None, help="Nombre de la carpeta de versión (ej: yolo11n_deeppcb_20260912_135338). Si no se pasa, se usa la más reciente local, o si no hay ninguna, la más reciente del bucket.")
    parser.add_argument("--registry-dir", default=DEFAULT_REGISTRY_DIR, type=Path, help="Carpeta local donde viven las versiones (default: models/registry/).")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="Bucket de GCS desde donde bajar el modelo si no está local.")
    parser.add_argument("--bucket-prefix", default=DEFAULT_BUCKET_PREFIX, help="Prefijo dentro del bucket (default: models/registry).")
    parser.add_argument("--skip-bucket", action="store_true", help="No intentar descargar nada del bucket; solo usar archivos locales.")

    # Overrides explícitos por si querés apuntar a rutas puntuales en vez de usar --version.
    parser.add_argument("--weights", default=None, type=Path, help="Ruta al .pt del modelo (si no se pasa, se arma a partir de --version).")
    parser.add_argument("--metadata-json", default=None, type=Path, help="Ruta al .json con metadatos (si no se pasa, se arma a partir de --version).")
    parser.add_argument("--metadata-joblib", default=None, type=Path, help="Ruta al .joblib con metadatos/modelo (si no se pasa, se arma a partir de --version).")

    parser.add_argument("--source", type=Path, help="[modo predict] Imagen o carpeta de imágenes a procesar.")
    parser.add_argument("--data", type=Path, help="[modo val] Ruta al data.yaml del dataset (train/val/test).")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"], help="[modo val] Qué split evaluar.")
    parser.add_argument("--conf", default=0.25, type=float, help="[modo predict] Umbral de confianza mínimo.")
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--device", default="cpu", help="Dispositivo para correr el modelo ('cpu', '0', etc). Default 'cpu' porque Cloud Shell no tiene GPU.")
    parser.add_argument("--output", default=ROOT / "runs" / "inference", type=Path, help="Carpeta donde se guardan resultados/artefactos.")
    parser.add_argument("--metrics-output", default=None, type=Path, help="[modo val] Dónde guardar las métricas resultantes (default: dentro de la carpeta de la versión).")
    args = parser.parse_args()

    version_dir = resolve_version_dir(
        version_name=args.version,
        registry_dir=args.registry_dir,
        bucket=args.bucket,
        bucket_prefix=args.bucket_prefix,
        allow_bucket=not args.skip_bucket,
    )
    print(f"Usando versión: {version_dir.name}  ({version_dir})\n")

    weights = args.weights or (version_dir / "model.pt")
    metadata_json = args.metadata_json or (version_dir / "metadata.json")
    metadata_joblib = args.metadata_joblib or (version_dir / "model.joblib")
    metrics_output = args.metrics_output or (version_dir / "eval_metrics.json")

    if not weights.exists():
        raise FileNotFoundError(f"No se encontró el archivo de pesos: {weights}")

    metadata = load_metadata(metadata_json, metadata_joblib)
    print_metadata_summary(metadata)

    model = YOLO(str(weights))
    model_name = metadata.get("model_name", weights.stem)

    if args.mode == "predict":
        if not args.source:
            parser.error("--source es obligatorio en modo predict.")
        run_inference(model, args.source, args.output, args.conf, args.imgsz, args.device)
    else:
        if not args.data:
            parser.error("--data es obligatorio en modo val.")
        run_validation(model, args.data, args.output, metrics_output, args.imgsz, args.split, args.device, model_name)


if __name__ == "__main__":
    main()
