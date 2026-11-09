# Clasificación multiclase de lesiones cutáneas mediante transferencia de aprendizaje

## Descripción

Este proyecto tiene como objetivo desarrollar y evaluar un modelo de clasificación multiclase de lesiones cutáneas a partir de imágenes dermatoscópicas, utilizando redes neuronales convolucionales (CNN) y transferencia de aprendizaje (*Transfer Learning*).

Se utiliza el conjunto de datos **HAM10000**, compuesto por 10.015 imágenes correspondientes a siete categorías de lesiones cutáneas.

La arquitectura seleccionada es **EfficientNetB0**, preentrenada con ImageNet. Se busca evaluar su capacidad de clasificación, considerando especialmente el desbalance entre categorías y su efecto sobre el desempeño del modelo.

El proyecto se desarrolla en **Python**, utilizando **TensorFlow/Keras**, y los experimentos se ejecutan en **Kaggle Notebooks** con aceleración por GPU.

> **Estado del proyecto:** En desarrollo. Se han completado la exploración de datos, la partición, el preprocesamiento inicial y la configuración de EfficientNetB0. El entrenamiento y la evaluación están pendientes.

---

## 1. Objetivos

### Objetivo general

Desarrollar y evaluar un modelo de clasificación multiclase de lesiones cutáneas mediante transferencia de aprendizaje utilizando EfficientNetB0 y el conjunto de datos HAM10000.

### Objetivos específicos

1. Explorar la distribución de las imágenes y las categorías del conjunto de datos.
2. Construir conjuntos independientes de entrenamiento, validación y prueba, evitando la presencia de una misma lesión en diferentes particiones.
3. Implementar técnicas de preprocesamiento y aumento de datos.
4. Evaluar estrategias para mitigar el desbalance entre categorías.
5. Entrenar EfficientNetB0 mediante transferencia de aprendizaje y ajuste fino.
6. Evaluar el desempeño utilizando métricas globales y por categoría.
7. Analizar los errores de clasificación y visualizar las regiones relevantes mediante Grad-CAM.

---

## 2. Conjunto de datos

Se utiliza **HAM10000 (Human Against Machine with 10000 training images)**, publicado por Tschandl et al. (2018).

Dataset disponible en Kaggle:

https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

El conjunto contiene:

- **10.015 imágenes dermatoscópicas**.
- **7.470 lesiones únicas**, identificadas mediante `lesion_id`.
- **7 categorías de lesiones cutáneas**.

### Distribución de categorías

| Código | Categoría | Imágenes |
|---|---|---:|
| nv | Nevus melanocítico | 6.705 |
| mel | Melanoma | 1.113 |
| bkl | Lesiones benignas similares a queratosis | 1.099 |
| bcc | Carcinoma basocelular | 514 |
| akiec | Queratosis actínicas y carcinoma intraepitelial | 327 |
| vasc | Lesiones vasculares | 142 |
| df | Dermatofibroma | 115 |
| **Total** | | **10.015** |

Se observa un desbalance importante: la categoría `nv` representa aproximadamente el 67 % de las imágenes, mientras que `df` representa aproximadamente el 1,15 %.

Este desbalance constituye uno de los principales desafíos del proyecto.

---

## 3. Metodología

### 3.1. Exploración y verificación de los datos

Se realizó la lectura del archivo `HAM10000_metadata.csv` y se verificó la disponibilidad de las imágenes originales.

También se analizaron:

- Número de imágenes por categoría.
- Número de lesiones independientes.
- Identificadores de las imágenes.
- Correspondencia entre metadatos y archivos.

Se verificó que las 10.015 imágenes estaban disponibles y que ninguna lesión tenía asignada más de una categoría.

### 3.2. División de los datos

Para evitar fuga de información (*data leakage*), las particiones se realizaron utilizando el identificador `lesion_id`.

Se aplicó una división estratificada por categoría a nivel de lesión, con una proporción aproximada de 70 % para entrenamiento, 15 % para validación y 15 % para prueba.

Se utilizó una semilla aleatoria de `42` para facilitar la reproducibilidad.

| Conjunto | Imágenes | Porcentaje |
|---|---:|---:|
| Entrenamiento | 7.002 | 69,92 % |
| Validación | 1.532 | 15,30 % |
| Prueba | 1.481 | 14,79 % |
| **Total** | **10.015** | **100 %** |

Se comprobó que no existieran imágenes ni lesiones compartidas entre los conjuntos.

Los identificadores y metadatos de cada partición se guardaron en:

- `particiones/train.csv`
- `particiones/val.csv`
- `particiones/test.csv`

Estas particiones se conservarán para permitir comparaciones consistentes entre los experimentos.

### 3.3. Preprocesamiento de imágenes

Las imágenes se prepararon utilizando `tf.data`, con las siguientes características:

- Redimensionamiento a **224 × 224 píxeles**.
- Conservación de los tres canales RGB.
- Conversión a `float32`.
- Procesamiento mediante lotes de 32 imágenes.
- Lectura y preparación paralela de datos.

Los valores de los píxeles se mantienen en el intervalo de 0 a 255, de acuerdo con los requisitos de entrada de la implementación de EfficientNetB0 de Keras.

Se realizó una inspección visual inicial para comprobar la correcta lectura y visualización de las imágenes.

### 3.4. Data Augmentation

Se configuraron transformaciones aleatorias para incrementar la variabilidad de las imágenes durante el entrenamiento:

- Reflexiones horizontales y verticales.
- Rotaciones aleatorias de hasta aproximadamente 36°.
- Variaciones moderadas de zoom, con un factor de 0,10.

Las transformaciones se implementaron mediante capas de aumento de datos de Keras.

No se incluyeron inicialmente modificaciones aleatorias de color, debido a la relevancia potencial de la información cromática en la clasificación de lesiones cutáneas.

Se generaron visualizaciones para inspeccionar el efecto de las transformaciones.

El aumento de datos se aplicará únicamente durante el entrenamiento, no durante la evaluación.

### 3.5. Tratamiento del desbalance de clases

Se calcularon pesos de clase utilizando exclusivamente el conjunto de entrenamiento.

La fórmula empleada fue:

\[
w_c = \frac{N}{K n_c}
\]

Donde:

- \(w_c\): peso de la categoría \(c\).
- \(N\): número total de imágenes de entrenamiento.
- \(K\): número de categorías.
- \(n_c\): número de imágenes de entrenamiento de la categoría \(c\).

Los pesos obtenidos fueron:

| Clase | Imágenes de entrenamiento | Peso |
|---|---:|---:|
| akiec | 230 | 4,3491 |
| bcc | 366 | 2,7330 |
| bkl | 774 | 1,2924 |
| df | 76 | 13,1617 |
| mel | 778 | 1,2857 |
| nv | 4.679 | 0,2138 |
| vasc | 99 | 10,1039 |

El propósito de esta estrategia es asignar mayor importancia a los errores cometidos sobre categorías menos representadas.

Su efecto real sobre el desempeño deberá comprobarse experimentalmente.

---

## 4. Arquitectura propuesta

### EfficientNetB0

Se seleccionó EfficientNetB0 como arquitectura base para la clasificación.

Se utiliza una versión preentrenada con pesos de ImageNet, eliminando el clasificador original mediante `include_top=False`.

Configuración inicial:

| Parámetro | Valor |
|---|---|
| Arquitectura | EfficientNetB0 |
| Pesos iniciales | ImageNet |
| Tamaño de entrada | 224 × 224 × 3 |
| Clasificador original | Eliminado |
| Número de capas de la base | 238 |
| Estado inicial de la base | Congelada |
| Categorías objetivo | 7 |

La carga de la arquitectura y de los pesos preentrenados se completó correctamente.

### Entrenamiento previsto

El entrenamiento se desarrollará en dos fases:

**Fase 1. Extracción de características**

Se mantendrán congelados los pesos de EfficientNetB0 y se entrenará una nueva cabeza clasificadora para las siete categorías.

**Fase 2. Fine-tuning**

Se descongelará una parte de la arquitectura para adaptar progresivamente las características aprendidas al conjunto de imágenes dermatoscópicas.

Los hiperparámetros y resultados se documentarán una vez realizados los experimentos.

---

## 5. Evaluación del modelo

Se utilizarán métricas que permitan analizar tanto el rendimiento global como el desempeño por categoría:

- Accuracy.
- Macro-F1.
- Macro-recall.
- Balanced accuracy.
- Precisión por categoría.
- Recall por categoría.
- Matriz de confusión.

También se contempla el uso de **Grad-CAM** para visualizar las regiones de las imágenes que contribuyen a las predicciones del modelo.

El conjunto de prueba se reservará para la evaluación final.

**Resultados:** Pendientes de entrenamiento y evaluación.

---

## 6. Experimentos previstos

### Modelo A: HAM10000

Clasificación de las siete categorías utilizando EfficientNetB0, Data Augmentation, pesos de clase y fine-tuning.

### Modelo B: ampliación de datos

Se estudiará la incorporación de imágenes adicionales de fuentes externas para aumentar la representación de categorías minoritarias.

Antes de incorporar nuevos datos se deberán verificar las categorías, la procedencia, los duplicados y la ausencia de contaminación de los conjuntos de validación y prueba.

Ambos modelos deberán evaluarse sobre el mismo conjunto de prueba.

### Posible análisis morfológico

Se contempla estudiar operaciones de morfología matemática y segmentación de lesiones como una extensión experimental.

Esta etapa no forma parte todavía de la implementación realizada.

---

## 7. Tecnologías utilizadas

- Python 3.
- TensorFlow / Keras 2.21.0.
- NumPy.
- Pandas.
- Scikit-learn.
- Matplotlib.
- Kaggle Notebooks.
- GPU para aceleración del entrenamiento.

---

## 8. Reproducibilidad

Los experimentos se desarrollan en Kaggle Notebooks.

Para facilitar su reproducción:

1. Descargar el dataset HAM10000.
2. Configurar un entorno con las dependencias utilizadas.
3. Ejecutar el notebook de preparación de datos.
4. Utilizar las particiones guardadas para el entrenamiento y la evaluación.
5. Cargar los pesos preentrenados de ImageNet.
6. Ejecutar los experimentos documentados.

Las instrucciones específicas de ejecución y los archivos definitivos se incorporarán al completar la implementación.

---

## 9. Referencias

Tschandl, P., Rosendahl, C., & Kittler, H. (2018). The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data, 5*, 180161. https://doi.org/10.1038/sdata.2018.161

Gonzalez, R. C., & Woods, R. E. (2018). *Digital Image Processing* (4th ed.). Pearson.

Material de clase y notebooks de la asignatura Visión de Máquina, Universidad Nacional de Colombia.

---

## 10. Limitaciones

El conjunto de datos presenta un desbalance considerable entre categorías.

El desempeño del modelo puede depender de la distribución de los datos, la calidad de las imágenes, las decisiones de preprocesamiento y los hiperparámetros de entrenamiento.

Los resultados obtenidos deberán interpretarse dentro de las condiciones experimentales del proyecto.

**Este proyecto tiene fines académicos y de investigación. No constituye un sistema de diagnóstico médico ni sustituye la evaluación de profesionales de salud.**
