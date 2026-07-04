
# Estructura del dataset DeepPCB

## 1. Ubicación del dataset

El dataset se almacena localmente dentro del proyecto en:

```text
data/
└── raw/
    └── DeepPCB-master/
```

La carpeta `data/raw/` se encuentra excluida del repositorio mediante `.gitignore`, por lo que el dataset no se sube a GitHub.

---

## 2. Estructura general

Dentro de `DeepPCB-master/` se encuentran distintos recursos del dataset:

```text
DeepPCB-master/
├── evaluation/
├── fig/
├── PCBData/
├── tools/
├── LICENSE
└── README.md
```

La carpeta principal para el proyecto es:

```text
PCBData/
```

---

## 3. Organización de PCBData

La carpeta `PCBData/` contiene diferentes grupos de imágenes:

```text
PCBData/
├── group00041/
├── group12000/
├── group12100/
├── group12300/
├── group13000/
├── group20085/
├── group44000/
├── group50600/
├── group77000/
├── group90100/
├── group92000/
├── test.txt
└── trainval.txt
```

Cada grupo representa un conjunto de imágenes de PCB relacionadas.

Además, el dataset proporciona dos archivos para definir la partición original:

* `trainval.txt`: muestras destinadas a entrenamiento y validación.
* `test.txt`: muestras destinadas a evaluación final.

Por este motivo, no se debe realizar inicialmente una división aleatoria de todo el dataset ignorando estos archivos.

---

## 4. Estructura interna de cada grupo

Por ejemplo:

```text
group00041/
├── 00041/
└── 00041_not/
```

La primera carpeta contiene las imágenes:

```text
00041/
├── 00041000_temp.jpg
├── 00041000_test.jpg
├── 00041001_temp.jpg
├── 00041001_test.jpg
├── 00041002_temp.jpg
├── 00041002_test.jpg
└── ...
```

La segunda carpeta contiene las anotaciones:

```text
00041_not/
├── 00041000.txt
├── 00041001.txt
├── 00041002.txt
└── ...
```

---

## 5. Pares de imágenes

Cada muestra posee dos imágenes relacionadas.

### Imagen template

Ejemplo:

```text
00041000_temp.jpg
```

Representa la imagen de referencia de la PCB.

### Imagen test

Ejemplo:

```text
00041000_test.jpg
```

Representa la imagen de inspección sobre la cual se encuentran los defectos anotados.

Para el enfoque de detección de objetos mediante YOLO planteado en el proyecto, la entrada principal del modelo será la imagen `_test.jpg`.

Las imágenes `_temp.jpg` no se utilizarán inicialmente como entrada del modelo YOLO, aunque podrán considerarse para análisis exploratorio, visualización o discusión de metodologías alternativas.

---

## 6. Relación entre imagen y anotación

Para una muestra determinada:

```text
00041000_temp.jpg
00041000_test.jpg
00041000.txt
```

la relación es:

```text
00041000_temp.jpg
        ↓
Imagen de referencia

00041000_test.jpg
        ↓
Imagen con defectos a detectar

00041000.txt
        ↓
Bounding boxes y clases de los defectos
```

Por lo tanto:

```text
00041000_test.jpg  ↔  00041000.txt
```

es el par principal que se utilizará para entrenar el modelo de detección.

---

## 7. Formato original de las anotaciones

Cada archivo `.txt` puede contener una o más filas.

Ejemplo:

```text
466 441 493 470 3
454 300 493 396 2
331 248 364 283 4
221 314 253 350 4
151 149 182 175 5
492 28 525 55 6
```

Cada fila representa un defecto individual con el formato:

```text
x_min  y_min  x_max  y_max  class_id
```

donde:

* `x_min`: coordenada horizontal izquierda de la bounding box.
* `y_min`: coordenada vertical superior.
* `x_max`: coordenada horizontal derecha.
* `y_max`: coordenada vertical inferior.
* `class_id`: identificador de la clase del defecto.

Ejemplo:

```text
466 441 493 470 3
```

representa una bounding box cuyos extremos son:

```text
Esquina superior izquierda: (466, 441)
Esquina inferior derecha:   (493, 470)
Clase:                       3
```

---

## 8. Clases

El dataset utiliza seis clases de defectos.

Los identificadores originales se encuentran numerados:

```text
1
2
3
4
5
6
```

Antes de implementar la conversión se deberá verificar la correspondencia exacta entre cada identificador y el nombre de clase según la documentación original del dataset.

Este paso es importante para evitar asignar nombres incorrectos a las clases durante el entrenamiento y la evaluación.

---

## 9. Diferencia con el formato YOLO

El formato original de DeepPCB es:

```text
x_min  y_min  x_max  y_max  class_id
```

con coordenadas absolutas expresadas en píxeles.

YOLO requiere:

```text
class_id  x_center  y_center  width  height
```

con coordenadas normalizadas respecto del ancho y alto de la imagen.

Por lo tanto, será necesario implementar una etapa de conversión.

Para una imagen de dimensiones:

```text
W = ancho de imagen
H = alto de imagen
```

las conversiones serán:

```text
x_center = ((x_min + x_max) / 2) / W

y_center = ((y_min + y_max) / 2) / H

width = (x_max - x_min) / W

height = (y_max - y_min) / H
```

Además, dado que las clases originales utilizan identificadores del 1 al 6 y YOLO utiliza normalmente índices desde 0, será necesario convertir:

```text
DeepPCB → YOLO

1 → 0
2 → 1
3 → 2
4 → 3
5 → 4
6 → 5
```

---

## 10. Flujo previsto para el proyecto

La estructura original no será modificada directamente.

Se mantendrá:

```text
data/raw/DeepPCB-master/
```

como copia local del dataset original.

Posteriormente se generará una versión procesada compatible con YOLO dentro de:

```text
data/processed/
```

La estructura esperada será:

```text
data/
├── raw/
│   └── DeepPCB-master/
│
└── processed/
    └── deep_pcb_yolo/
        ├── images/
        │   ├── train/
        │   ├── val/
        │   └── test/
        │
        ├── labels/
        │   ├── train/
        │   ├── val/
        │   └── test/
        │
        └── data.yaml
```

De esta manera se preserva el dataset original y se genera una versión derivada específicamente preparada para los experimentos con YOLO.

---

## 11. Consideraciones metodológicas

Antes de entrenar los modelos se deberá:

1. Verificar la correspondencia exacta entre `class_id` y nombre del defecto.
2. Analizar el contenido de `trainval.txt` y `test.txt`.
3. Definir cómo separar `trainval` en entrenamiento y validación.
4. Verificar que no exista fuga de información entre particiones.
5. Analizar la distribución de clases.
6. Analizar el tamaño y proporción de las bounding boxes.
7. Visualizar imágenes con sus anotaciones originales.
8. Validar visualmente la conversión al formato YOLO antes de iniciar el entrenamiento.

Estas verificaciones formarán parte del análisis exploratorio del dataset.
