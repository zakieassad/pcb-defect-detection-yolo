# Detección automática de defectos en PCB mediante YOLO

Trabajo final de **M1.11 - Deep Learning, Maestría en Management & Analytics, ITBA.**

## 1. Descripción del proyecto

Este proyecto implementa una solución de detección automática de defectos en PCBs utilizando modelos YOLO.

El objetivo es identificar y localizar defectos de fabricación en imágenes de PCB mediante *bounding boxes*. Se comparan dos variantes de modelo:

- YOLO11 Nano
- YOLO11 Small

Ambos modelos fueron entrenados mediante *fine-tuning* a partir de pesos preentrenados y evaluados sobre el dataset público DeepPCB.

## 2. Propuesta de valor

La inspección visual de PCB es una etapa crítica dentro de los procesos de fabricación y control de calidad de productos electrónicos. La presencia de defectos como cortocircuitos, interrupciones de pistas, perforaciones incompletas o excesos de material puede comprometer el funcionamiento del producto final y generar costos asociados a retrabajos o descartes.

La solución propuesta busca automatizar una primera etapa de inspección visual, permitiendo detectar y clasificar defectos de forma más rápida, consistente y escalable que mediante una revisión completamente manual.

Si bien el trabajo utiliza un dataset público, el pipeline desarrollado podría constituir un punto de partida para futuras aplicaciones reales sobre imágenes propias de PCB o PCBA. Esa adaptación requeriría construir y anotar un dataset específico, lo cual queda fuera del alcance del presente trabajo.

## 3. Dataset

Se utilizó el dataset público **DeepPCB**, que contiene imágenes de placas de circuito impreso con anotaciones de defectos mediante *bounding boxes*.

Las clases consideradas son:

- `open`
- `short`
- `mousebite`
- `spur`
- `copper`
- `pin-hole`

El dataset original no se incluye en este repositorio por cuestiones de tamaño. Puede descargarse desde su repositorio público: [github.com/tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB)

La estrcutura del dataset y su adaptación al proyecto se documentan en `docs/dataset_structure.md `

## 4. Metodología

El trabajo se desarrolló siguiendo las siguientes etapas:

1. Exploración del dataset DeepPCB.
2. Validación de estructura, imágenes y anotaciones.
3. Conversión de anotaciones originales al formato requerido por YOLO.
4. Definición de particiones `train`, `val` y `test`.
5. Fine-tuning de YOLO11 Nano y YOLO11 Small.
6. Evaluación sobre conjunto de prueba.
7. Comparación de métricas globales, métricas por clase y costo computacional.
8. Análisis cualitativo de matrices de confusión, curvas Precision-Recall y ejemplos de predicción.

## 5. Estructura del repositorio

```text
pcb-defect-detection-yolo/
│
├── docs/
│   └── dataset_structure.md
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   ├── 02_dataset_preparation.ipynb
│   ├── 03_local_training_debug.ipynb
│   ├── 04_colab_final_training.ipynb
│   └── 05_results_analysis.ipynb
│
├── results/
│   ├── figures/
│   └── metrics/
│
├── src/
│
├── requirements.txt
├── .gitignore
└── README.md
```

## 6. Notebooks

### `01_dataset_exploration.ipynb`

Realiza el análisis exploratorio del dataset:

- estructura de DeepPCB;
- cantidad de imágenes y anotaciones;
- distribución por partición;
- distribución de clases;
- cantidad de defectos por imagen;
- tamaño relativo de las *bounding boxes*;
- visualización de muestras anotadas;
- definición metodológica de particiones.

### `02_dataset_preparation.ipynb`

Convierte DeepPCB al formato YOLO:

- reconstrucción de particiones;
- conversión de bounding boxes;
- generación de estructura compatible con Ultralytics;
- creación de `data.yaml`;
- auditoría automática del dataset procesado.

### `03_local_training_debug.ipynb`

Ejecuta una corrida local mínima con YOLO11 Nano para validar el pipeline de entrenamiento.

Esta corrida no se utiliza como resultado final, ya que el entorno local no cuenta con GPU disponible.

### `04_colab_final_training.ipynb`

Notebook utilizado para el entrenamiento final en Google Colab con GPU.

Incluye:

- descarga de DeepPCB;
- preparación del dataset;
- entrenamiento de YOLO11 Nano;
- entrenamiento de YOLO11 Small;
- evaluación sobre test;
- exportación de métricas y resultados.

### `05_results_analysis.ipynb`

Consolida y analiza los resultados finales:

- métricas globales;
- métricas por clase;
- matrices de confusión;
- curvas Precision-Recall;
- ejemplos cualitativos;
- costo computacional;
- conclusiones.

## 7. Resultados principales

Los modelos fueron evaluados sobre el conjunto de prueba de DeepPCB.

| Modelo  | Precision | Recall | mAP@50 | mAP@50:95 |
| ------- | --------: | -----: | -----: | --------: |
| YOLO11n |    92.25% | 90.68% | 95.38% |    68.56% |
| YOLO11s |    93.90% | 91.43% | 96.19% |    72.33% |

YOLO11 Small obtuvo el mejor desempeño global, especialmente en mAP@50:95, donde superó a YOLO11 Nano por aproximadamente 3,77 puntos porcentuales.

Sin embargo, YOLO11 Small también presenta un mayor costo computacional:

| Modelo  | Parámetros | GFLOPs | Tamaño best.pt | Inferencia |
| ------- | ----------: | -----: | --------------: | ---------: |
| YOLO11n |      2.58 M |    6.3 |          5.5 MB | 2.8 ms/img |
| YOLO11s |      9.42 M |   21.3 |         19.2 MB | 6.0 ms/img |

Por lo tanto, YOLO11 Small resulta preferible si se prioriza desempeño predictivo, mientras que YOLO11 Nano puede ser más conveniente en escenarios con restricciones de hardware o necesidad de inferencia más rápida.

## 8. Reproducción del proyecto

### 8.1 Crear entorno

```bash
conda create -n pcb-yolo python=3.11 -y
conda activate pcb-yolo
pip install -r requirements.txt
```

### 8.2 Descargar dataset

Descargar DeepPCB desde:

```text
https://github.com/tangsanli5201/DeepPCB
```

y ubicarlo localmente en:

```text
data/raw/DeepPCB-master/
```

La carpeta `data/raw/` está excluida del repositorio mediante `.gitignore`.

### 8.3 Preparar dataset YOLO

Ejecutar:

```text
notebooks/02_dataset_preparation.ipynb
```

Esto genera:

```text
data/processed/deep_pcb_yolo/
```

con la estructura compatible con YOLO.

### 8.4 Entrenamiento

El entrenamiento final fue ejecutado en Google Colab con GPU Tesla T4 utilizando:

```text
notebooks/04_colab_final_training.ipynb
```

El entorno local fue utilizado únicamente para validación técnica del pipeline.

## 9. Métricas utilizadas

Las métricas principales fueron:

- **Precision:** proporción de detecciones realizadas que son correctas.
- **Recall:** proporción de defectos reales detectados por el modelo.
- **mAP@50:** promedio de precisión media con umbral IoU 0,50.
- **mAP@50:95:** promedio de precisión media en múltiples umbrales IoU entre 0,50 y 0,95.

En el contexto de inspección de calidad, el recall resulta especialmente relevante porque un falso negativo implica que un defecto real no fue detectado.

## 10. Limitaciones y trabajo futuro

El trabajo fue realizado sobre un dataset público de PCB, no sobre imágenes propias de un entorno productivo real. Por lo tanto, los resultados obtenidos validan experimentalmente el enfoque, pero no garantizan el mismo desempeño en condiciones reales de operación.

Como trabajo futuro se propone:

- evaluar el pipeline sobre imágenes propias de PCB o PCBA;
- incorporar defectos asociados a componentes electrónicos;
- analizar tiempos de inferencia en hardware específico;
- ajustar umbrales de confianza por clase;
- estudiar falsos positivos y falsos negativos en condiciones reales de inspección.
