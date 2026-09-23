# PCB Defect Detection with YOLO11 and Google Cloud

Sistema de detección automática de defectos en placas de circuito impreso (PCB) basado en **YOLO11**, desplegado con una arquitectura MLOps sobre **Google Cloud Platform**.

El proyecto cubre el ciclo completo de un modelo de Machine Learning: entrenamiento, evaluación, versionado, registro de artefactos, almacenamiento en Cloud Storage, serving mediante FastAPI, interfaz web con Streamlit, containerización con Docker, despliegue en Cloud Run, observabilidad y preparación para monitoreo de drift.

---

## 1. Objetivo

El objetivo del proyecto es detectar automáticamente defectos presentes en imágenes de PCB utilizando un modelo de detección de objetos entrenado sobre el dataset **DeepPCB**.

Las clases consideradas son:

- `open`
- `short`
- `mousebite`
- `spur`
- `copper`
- `pin-hole`

El sistema permite cargar una imagen desde una interfaz web, enviarla al modelo desplegado, obtener las detecciones y visualizar los *bounding boxes* junto con la clase y confianza asociadas.

---

## 2. Arquitectura general

```text
                 ENTRENAMIENTO
                      │
                      ▼
              YOLO11 + DeepPCB
                      │
                      ▼
        ┌─────────────────────────┐
        │ Registro de modelo      │
        │                         │
        │ model.pt                │
        │ model.joblib            │
        │ metadata.json           │
        └────────────┬────────────┘
                     │
                     ▼
             GOOGLE CLOUD STORAGE
                     │
         ┌───────────┴────────────┐
         │                        │
 models/registry/          production/images/
         │                        ▲
         ▼                        │
      CLOUD RUN                    │
         │                        │
    ┌────┴─────┐                  │
    │          │                  │
Streamlit   FastAPI ──────────────┘
  :8080       :8000
    │
    ▼
  Usuario
```

La solución desacopla el código de aplicación del artefacto de modelo. La imagen Docker contiene el código y sus dependencias, mientras que los pesos del modelo se recuperan desde Google Cloud Storage en tiempo de ejecución.

---

## 3. Tecnologías utilizadas

- Python 3.12
- Ultralytics YOLO11
- PyTorch
- OpenCV
- FastAPI
- Uvicorn
- Streamlit
- Docker
- Google Cloud Storage
- Google Cloud Build
- Artifact Registry
- Cloud Run
- Cloud Logging
- Pandas / NumPy
- Joblib

---

## 4. Estructura del proyecto

```text
pcb-defect-detection-yolo/
│
├── app/
│   ├── main.py
│   ├── model_backend.py
│   ├── observability.py
│   └── schemas.py
│
├── frontend/
│   ├── app.py
│   └── requirements.txt
│
├── scripts/
│   └── check_drift.py
│
├── data/
│   └── processed/
│       └── deep_pcb_yolo/
│
├── models/
│   └── registry/
│
├── Dockerfile
├── start.sh
├── requirements.txt
├── run_yolo_model.py
└── README.md
```

> Los modelos y datasets de gran tamaño no deberían almacenarse directamente en Git. Se mantienen en Google Cloud Storage.

---

## 5. Dataset

El dataset procesado se almacena en:

```text
gs://mma-cloudproject-tfi-grupo3/processed/deep_pcb_yolo/
```

La estructura utilizada por YOLO contempla:

```text
deep_pcb_yolo/
├── data.yaml
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

El script `run_yolo_model.py` puede trabajar con una copia local del dataset o descargarlo desde Google Cloud Storage cuando sea necesario.

---

## 6. Registro y versionado del modelo

Cada modelo entrenado se registra en una carpeta identificada por versión y timestamp:

```text
models/registry/
└── yolo11n_deeppcb_20260912_135338/
    ├── model.pt
    ├── model.joblib
    └── metadata.json
```

En Google Cloud Storage:

```text
gs://mma-cloudproject-tfi-grupo3/models/registry/
```

### Artefactos

- `model.pt`: artefacto canónico utilizado para inferencia con Ultralytics.
- `model.joblib`: serialización adicional utilizada como respaldo de información.
- `metadata.json`: metadatos del modelo, entrenamiento y métricas.

La versión desplegada puede configurarse mediante la variable de entorno:

```text
MODEL_VERSION
```

Esto permite controlar explícitamente qué modelo utiliza cada revisión de Cloud Run.

---

## 7. Métricas del modelo desplegado

La versión utilizada durante el despliegue obtuvo aproximadamente:

| Métrica | Valor |
|---|---:|
| Precision | 0.7029 |
| Recall | 0.7239 |
| mAP@50 | 0.7892 |
| mAP@50-95 | 0.4840 |

Además de las métricas globales, el proceso de evaluación permite recuperar métricas por clase.

---

## 8. Ejecución del modelo

El script `run_yolo_model.py` centraliza la carga y ejecución del modelo.

### Predicción

```bash
python run_yolo_model.py predict --source ruta/a/imagen.jpg
```

### Validación

```bash
python run_yolo_model.py val
```

La validación vuelve a ejecutar el modelo sobre el dataset y genera métricas de evaluación, incluyendo un archivo:

```text
eval_metrics.json
```

---

## 9. API de inferencia

La API está implementada con **FastAPI**.

Endpoints principales:

```text
GET  /health
POST /predict
```

### Health check

```bash
curl http://127.0.0.1:8000/health
```

### Predicción

El endpoint `/predict` recibe una imagen mediante `multipart/form-data`.

La respuesta incluye:

- nombre del archivo
- ancho y alto de la imagen
- cantidad de detecciones
- clase detectada
- confianza
- coordenadas del bounding box
- backend utilizado
- nombre y versión del modelo

Ejemplo conceptual:

```json
{
  "filename": "pcb.jpg",
  "image_width": 640,
  "image_height": 640,
  "detection_count": 2,
  "detections": [
    {
      "class_id": 1,
      "class_name": "short",
      "confidence": 0.91,
      "bbox": {
        "x1": 120.5,
        "y1": 85.2,
        "x2": 196.4,
        "y2": 154.8
      }
    }
  ],
  "backend": "gcs_local",
  "model_name": "YOLO11n",
  "model_version": "yolo11n_deeppcb_20260912_135338"
}
```

---

## 10. Frontend

El frontend fue desarrollado con **Streamlit**.

Permite:

- cargar una imagen
- visualizar la imagen original
- ejecutar una predicción
- mostrar cantidad de defectos
- visualizar clase y confianza
- dibujar bounding boxes
- mostrar la imagen anotada
- visualizar las detecciones en formato tabular

El frontend consume internamente la API FastAPI mediante:

```text
http://127.0.0.1:8000/predict
```

---

## 11. Ejecución local

### Instalar dependencias

Desde la raíz del proyecto:

```bash
pip install -r requirements.txt
pip install -r frontend/requirements.txt
```

### Iniciar API y frontend

```bash
chmod +x start.sh
./start.sh
```

La arquitectura local utiliza:

```text
FastAPI    → 127.0.0.1:8000
Streamlit  → 0.0.0.0:8080
```

Para verificar la API:

```bash
curl http://127.0.0.1:8000/health
```

Para verificar Streamlit:

```bash
curl http://127.0.0.1:8080/_stcore/health
```

---

## 12. start.sh

El contenedor ejecuta FastAPI y Streamlit simultáneamente.

```bash
#!/bin/sh

uvicorn app.main:app \
  --host 127.0.0.1 \
  --port 8000 &

sleep 2

streamlit run frontend/app.py \
  --server.address 0.0.0.0 \
  --server.port ${PORT:-8080} \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false \
  --browser.gatherUsageStats false
```

FastAPI permanece accesible únicamente dentro del contenedor, mientras que Streamlit escucha en el puerto público proporcionado por Cloud Run.

---

## 13. Docker

La imagen utiliza Python 3.12 slim.

Las dependencias de sistema necesarias para OpenCV incluyen:

```text
libglib2.0-0
libgl1
libgomp1
libxcb1
```

Se utiliza `opencv-python-headless` para evitar dependencias gráficas innecesarias.

Construcción local:

```bash
docker build -t pcb-defect-app .
```

---

## 14. Google Cloud

### Configuración utilizada

```bash
export PROJECT_ID="mma-cloudproject"
export REGION="us-central1"
export BUCKET="mma-cloudproject-tfi-grupo3"
export REPO="pcb-mlops"
export SERVICE="pcb-defect-app"
export MODEL_VERSION="yolo11n_deeppcb_20260912_135338"

export IMAGE="${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO}/${SERVICE}:latest"
```

---

## 15. Artifact Registry

Crear el repositorio si todavía no existe:

```bash
gcloud artifacts repositories create "${REPO}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="PCB defect detection MLOps" \
  --project="${PROJECT_ID}"
```

---

## 16. Build de la imagen

La construcción y publicación se realiza con Cloud Build:

```bash
gcloud builds submit \
  --tag "${IMAGE}" \
  .
```

Cada modificación de archivos incluidos en la imagen Docker, por ejemplo `start.sh`, `app/` o `frontend/`, requiere reconstruir la imagen antes de desplegar.

---

## 17. Despliegue en Cloud Run

```bash
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --platform managed \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300 \
  --set-env-vars MODEL_BUCKET="${BUCKET}",MODEL_PREFIX="models/registry",MODEL_VERSION="${MODEL_VERSION}",MODEL_DEVICE="cpu"
```

Obtener la URL del servicio:

```bash
gcloud run services describe "${SERVICE}" \
  --region="${REGION}" \
  --format='value(status.url)'
```

La URL pública corresponde al frontend Streamlit.

---

## 18. Permisos de Google Cloud Storage

La cuenta de servicio utilizada por Cloud Run requiere acceso de lectura al model registry:

```text
roles/storage.objectViewer
```

Para almacenar imágenes enviadas en producción también requiere:

```text
roles/storage.objectCreator
```

Se recomienda asignar únicamente los permisos mínimos necesarios.

---

## 19. Persistencia de imágenes de producción

Las imágenes utilizadas para inferencia pueden almacenarse automáticamente en:

```text
gs://mma-cloudproject-tfi-grupo3/production/images/
```

Esto permite construir una ventana histórica de inputs reales para monitoreo del comportamiento del sistema.

El flujo de una predicción es:

```text
POST /predict
      ↓
guardar imagen en GCS
      ↓
ejecutar YOLO
      ↓
generar detecciones
      ↓
registrar evento
      ↓
devolver respuesta
```

---

## 20. Observabilidad

La API genera logs estructurados en JSON compatibles con Cloud Logging.

Entre los campos registrados se encuentran:

```text
latency_ms
filename
detection_count
detected_classes
model_version
backend
```

Los datos binarios de las imágenes no se almacenan dentro de los logs.

Esto permite analizar:

- latencia
- volumen de inferencias
- clases detectadas
- versión utilizada
- errores del backend

---

## 21. Monitoreo de drift

El proyecto incorpora una estrategia inicial de **input drift** basada en características visuales de las imágenes.

Variables consideradas:

```text
brightness
contrast
sharpness
width
height
```

Se comparan dos poblaciones:

```text
dataset de entrenamiento
        vs.
imágenes reales de producción
```

El análisis utiliza **Population Stability Index (PSI)** como indicador descriptivo de cambios en las distribuciones.

Adicionalmente, el sistema puede evolucionar hacia monitoreo de **prediction drift**, comparando la distribución temporal de las clases detectadas.

---

## 22. Reproducibilidad

Para reproducir el sistema deben mantenerse controlados:

- código fuente
- versiones de dependencias
- imagen Docker
- variables de entorno
- estructura del bucket
- dataset
- versiones del modelo
- metadatos
- permisos IAM
- configuración de Cloud Run

El modelo no está embebido en la imagen Docker. Esto permite actualizar el artefacto de Machine Learning independientemente del código de serving.

---

## 23. Flujo MLOps implementado

```text
Entrenamiento
      ↓
Evaluación
      ↓
Versionado
      ↓
Registro
      ↓
Cloud Storage
      ↓
Containerización
      ↓
Cloud Build
      ↓
Artifact Registry
      ↓
Cloud Run
      ↓
Inferencia
      ↓
Observabilidad
      ↓
Captura de datos de producción
      ↓
Monitoreo de drift
```

---

## 24. Resultado

El proyecto transforma un experimento de Deep Learning en una solución de Machine Learning desplegable y operable.

El modelo deja de ser únicamente un archivo de pesos para convertirse en un artefacto:

- versionado
- trazable
- reproducible
- desplegable
- observable
- desacoplado del código
- preparado para monitoreo posterior

La interfaz web completa el flujo desde la perspectiva del usuario final, permitiendo cargar imágenes de PCB, ejecutar inferencia y visualizar los defectos identificados por el modelo de forma gráfica.
