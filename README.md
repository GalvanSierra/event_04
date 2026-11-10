# Evaluación 4 · Equipo 9 — Predicción de fallas en sistemas de camiones industriales

**Asignatura:** Análisis de Datos · **Docente:** Daniel Alexis Nieto Mora · **Periodo:** 2026-2
**Dataset:** [APS Failure at Scania Trucks (UCI)](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks)

> **Estado del avance:** ✅ EDA y preprocesamiento · ✅ Componente supervisado · ✅ Componente no supervisado · ✅ Análisis comparativo · ⏳ Video

---

## 1. Descripción del problema

El sistema **APS** (_Air Pressure System_) de los camiones Scania produce el aire presurizado que usan funciones críticas como el **frenado** y el **cambio de marchas**. Scania publicó lecturas anónimas de **170 sensores** de camiones en operación, etiquetadas así:

- `pos`: el camión tuvo una falla en un componente del **APS**.
- `neg`: la falla corresponde a **otro** componente.

El reto plantea dos preguntas complementarias:

| Componente         | Pregunta                                                                                           | Estado |
| ------------------ | -------------------------------------------------------------------------------------------------- | ------ |
| **Supervisado**    | ¿Podemos predecir qué camiones tendrán una falla en el APS, minimizando el costo de mantenimiento? | ✅     |
| **No supervisado** | ¿Existen perfiles de funcionamiento o registros atípicos, sin usar la etiqueta?                    | ✅     |

**Costo del negocio** (definido por Scania en el reto IDA 2016):

$$\text{Costo} = 10 \cdot FP + 500 \cdot FN$$

Un falso negativo (falla no detectada, avería en ruta) cuesta **50 veces más** que un falso positivo (revisión innecesaria en taller).

## 2. Integrantes y responsabilidades

| Integrante                | Rol                                        | Entregables                                                                              |
| ------------------------- | ------------------------------------------ | ---------------------------------------------------------------------------------------- |
| David Galvan Sierra       | Datos, EDA y preprocesamiento              | `src/data.py`, `src/preprocessing.py`, `notebooks/01_eda_preprocesamiento.ipynb`, README |
| Federico Rodriguez Franco | Componente supervisado                     | `src/metrics.py`, `notebooks/02_supervisado.ipynb`                                       |
| Juan David Osorio Zapata  | Componente no supervisado y conclusiones   | `notebooks/03_no_supervisado.ipynb`, `notebooks/04_comparativo_conclusiones.ipynb`       |
| Todos                     | Análisis comparativo, conclusiones y video | `notebooks/04_comparativo_conclusiones.ipynb`                                            |

## 3. Estructura del proyecto

```
evaluacion 4/
├── README.md
├── requirements.txt
├── data/                      # CSV de UCI (no versionados) + instrucciones de descarga
│   └── README.md
├── src/                       # código compartido por todos los notebooks
│   ├── data.py                # load_aps(), histogram_groups()
│   ├── preprocessing.py       # build_preprocessor(), preprocessor_steps(), transformadores propios
│   └── metrics.py             # aps_cost(), evaluate(), best_threshold_by_cost(), gráficas
├── notebooks/
│   ├── 01_eda_preprocesamiento.ipynb
│   ├── 02_supervisado.ipynb
│   ├── 03_no_supervisado.ipynb
│   └── 04_comparativo_conclusiones.ipynb # todos
├── figures/                   # figuras exportadas por los notebooks (01_*, 02_*)
└── results/                   # métricas en CSV/JSON generadas por los notebooks
```

## 4. Instrucciones de ejecución

Se requiere **Python 3.11 o superior** (probado con Python 3.14).

```bash
python -m venv .venv
```

```bash
.venv\Scripts\activate          # Windows  (Linux/Mac: source .venv/bin/activate)
```

```bash
cd "evaluacion 4"
pip install -r requirements.txt
```

**Datos:** descargar el zip desde [UCI](https://archive.ics.uci.edu/dataset/421/aps+failure+at+scania+trucks) y descomprimir `aps_failure_training_set.csv` y `aps_failure_test_set.csv` en `data/` (ver [data/README.md](evaluacion%204/data/README.md)). La función `load_aps()` salta automáticamente las ~20 líneas de licencia del encabezado y convierte `"na"` en `NaN`.

**Orden de ejecución** (cada notebook se ejecuta desde la carpeta `notebooks/`):

| Notebook                            | Tiempo aprox.* | Genera                                                                        |
| ----------------------------------- | -------------- | ----------------------------------------------------------------------------- |
| `01_eda_preprocesamiento.ipynb`     | ~15 min        | `figures/01_*.png`, `results/01_*.csv`                                        |
| `02_supervisado.ipynb`              | ~40 min        | `figures/02_*.png`, `results/02_*.csv`, `results/02_supervisado_metrics.json` |
| `03_no_supervisado.ipynb`           | variable       | `figures/03_*.png`, `results/03_*.csv`                                        |
| `04_comparativo_conclusiones.ipynb` | lectura        | comparación y conclusiones                                                    |

\* En un portátil de 8 núcleos. La mayor parte del tiempo se va en la validación cruzada de 12 combinaciones modelo×estrategia y en la búsqueda de hiperparámetros.

Uso de los módulos compartidos (por ejemplo, para el notebook 03):

```python
from src.data import load_aps
from src.preprocessing import build_preprocessor

X, y = load_aps("train")              # y solo para interpretar a posteriori en el no supervisado
Z = build_preprocessor().fit_transform(X)   # 170 -> 143 columnas, sin NaN, escaladas
```

## 5. Datos

|                    | Train                                                          | Test         |
| ------------------ | -------------------------------------------------------------- | ------------ |
| Camiones           | 60 000                                                         | 16 000       |
| Fallas APS (`pos`) | 1 000 (**1,67 %**)                                             | 375 (2,34 %) |
| Variables          | 170 numéricas anónimas (100 simples + 7 histogramas × 10 bins) |              |
| Celdas faltantes   | 8,3 % (el 99 % de los camiones tiene al menos un NaN)          |              |

Fuente: Scania CV AB (2016), _APS Failure and Operational Data for Scania Trucks_, UCI Machine Learning Repository, licencia GPL-3.0.

## 6. Principales resultados

### 6.1 EDA y preprocesamiento (notebook 01)

- **Desbalance severo** (1 falla por cada 59 camiones sanos). La accuracy no sirve: predecir siempre `neg` da 98,3 % de accuracy y un costo de 187 500 en test.
- **Los faltantes son informativos y aparecen en bloques.** Cuando falta el bloque `df_000…eb_000`, la tasa de falla sube del 0,9 % al **13,1 %**. Los camiones con falla tienen en mediana 20,5 variables faltantes, frente a 8 en los sanos.
- **Variables muy asimétricas** (157/170 con asimetría > 5) y **redundantes** (172 pares con |r| > 0,95).
- **Histogramas.** En el 97,5 % de los camiones, los 7 histogramas suman lo mismo; probablemente es un contador de uso. Esa suma es unas **15 veces mayor** en los camiones con falla.
- **Imputación.** Mediana, mediana + indicador, KNN, constante e Iterative quedan dentro de una desviación estándar (PR-AUC 0,69–0,72). Se eligió **mediana + indicador de NaN** por simple, barata y porque conserva la señal del faltante. KNN fue unas 16 veces más lento sin mejorar el resultado.
- **Preprocesamiento final:** se eliminan 7 columnas con más del 70 % de NaN, se imputa la mediana con indicadores, se aplica `log1p`, se eliminan las variables constantes y las de |r| > 0,95, y se estandariza. Resultado: **170 → 143 columnas**. Todo se ajusta solo con el train dentro de un `Pipeline` (sin fuga de información).
- **PCA:** hacen falta 53 componentes para el 90 % de la varianza. En t-SNE, las fallas forman **varios grupos compactos**, lo que sugiere más de un perfil de falla.

| Proyección PCA                            | t-SNE                                        |
| ----------------------------------------- | -------------------------------------------- |
| ![PCA](evaluacion%204/figures/01_pca.png) | ![t-SNE](evaluacion%204/figures/01_tsne.png) |

### 6.2 Componente supervisado (notebook 02)

**Modelos comparados:** Regresión Logística, Random Forest y XGBoost. Cada uno con 4 estrategias de desbalance (sin tratamiento, ponderación de clases, SMOTE, undersampling) en CV estratificada de 5 folds, seguido de una búsqueda de hiperparámetros y un **umbral de decisión elegido con predicciones out-of-fold** para minimizar el costo.

**Resultados en el test oficial** (16 000 camiones, usado una sola vez):

| Modelo                 | Umbral    | Recall    | Precision | F1    | PR-AUC    | FP  | FN  | **Costo** |
| ---------------------- | --------- | --------- | --------- | ----- | --------- | --- | --- | --------- |
| Regresión Logística    | 0,029     | 0,944     | 0,465     | 0,623 | 0,864     | 407 | 21  | 14 570    |
| Random Forest          | 0,039     | 0,976     | 0,435     | 0,602 | 0,925     | 475 | 9   | 9 250     |
| **XGBoost** (final)    | **0,005** | **0,973** | 0,465     | 0,629 | **0,936** | 420 | 10  | **9 200** |
| XGBoost con umbral 0,5 | 0,5       | 0,779     | 0,957     | 0,859 | 0,936     | 13  | 83  | 41 630    |

| Referencia                                     | Costo en test |
| ---------------------------------------------- | ------------- |
| No hacer nada (todo `neg`)                     | 187 500       |
| Revisar todos los camiones                     | 156 250       |
| Ganador del reto IDA 2016                      | 9 920         |
| **Nuestro modelo (XGBoost + umbral de costo)** | **9 200**     |

**Hallazgos clave**

1. **XGBoost** es el mejor modelo (PR-AUC 0,936, ROC-AUC 0,996). Detecta **365 de 375 fallas (97,3 %)** enviando a revisión al **4,9 % de la flota**.
2. **El umbral de costo vale más que cualquier técnica de remuestreo.** SMOTE y la ponderación de clases no mejoraron la PR-AUC, y el undersampling la empeoró. En cambio, bajar el umbral de 0,5 a 0,005 redujo el costo un **78 %** (de 41 630 a 9 200).
3. **El F1 y la accuracy engañan aquí.** El F1 baja de 0,86 a 0,63 justo cuando el costo cae un 78 %, porque pondera FP y FN por igual.
4. **Variables más relevantes** (SHAP y permutación): `aa_000`, `ck_000`, `bs_000`, `cb_000`, `ai_000` y la _forma_ de los histogramas `ay`, `az` y `ag`, que reúnen cerca del 36 % de la importancia SHAP. Al estar anonimizadas, no es posible interpretarlas físicamente.

| Costo vs umbral (OOF)                                   | Importancia SHAP                                    |
| ------------------------------------------------------- | --------------------------------------------------- |
| ![Costo](evaluacion%204/figures/02_costo_vs_umbral.png) | ![SHAP](evaluacion%204/figures/02_shap_summary.png) |

**Propuesta de mantenimiento preventivo:** clasificar la flota en niveles de riesgo (_Normal_, _Vigilancia_, _Revisión programada_, _Taller inmediato_) según la probabilidad del modelo, vigilar los sensores clave, acortar los intervalos de inspección del APS en camiones con mucho uso acumulado y recalcular el umbral cuando cambien los costos o la tasa de fallas.

### 6.3 Componente no supervisado (notebook 03)

Para esta parte usamos K-Means e Isolation Forest con una muestra aleatoria de 10 000 registros de entrenamiento. Los métodos se ajustaron sin usar las etiquetas. Después del preprocesamiento quedaron 141 variables y PCA conservó el 90,3 % de la varianza con 53 componentes. Probamos entre 2 y 6 grupos y elegimos 4 por la silueta (0,301); el índice Davies-Bouldin fue 1,780.

El grupo 3 quedó con 1 069 registros y 122 fallas (11,41 %). Isolation Forest marcó 200 anomalías; 52 tenían falla (26 %) y 113 estaban en el grupo 3. Estos resultados describen la muestra, no son un diagnóstico ni una evaluación en test. La etiqueta se usó después de entrenar para entender los resultados.

### 6.4 Análisis comparativo y video

Las ocho preguntas de la guía se responden en `04_comparativo_conclusiones.ipynb`. **Video: pendiente** (máximo 12 minutos, con participación de todos).

## 7. Decisiones técnicas documentadas

| Decisión                         | Justificación                                                                                       | Dónde        |
| -------------------------------- | --------------------------------------------------------------------------------------------------- | ------------ |
| Mediana + indicador de NaN       | Robusta a colas largas. El faltante es informativo. KNN e Iterative no mejoran y son más costosos.  | NB 01 §6     |
| Eliminar columnas con > 70 % NaN | Imputar el 77–82 % de una columna inventa la mayoría de los valores. El efecto en PR-AUC es < 0,01. | NB 01 §7     |
| `log1p`                          | 157/170 variables con asimetría > 5. Lo necesitan la LogReg y los métodos de distancias.            | NB 01 §4     |
| Filtro de correlación 0,95       | 324 → 143 columnas sin perder PR-AUC. Abarata los modelos y el clustering.                          | NB 01 §7     |
| Pipelines (`sklearn`/`imblearn`) | Evita la fuga de información: todo se ajusta solo con los folds de entrenamiento.                   | `src/`       |
| Sin remuestreo + umbral de costo | Mejor PR-AUC, sin datos sintéticos, optimiza directamente la métrica del negocio.                   | NB 02 §2, §4 |
| Umbral elegido con OOF del train | El test no interviene en ninguna decisión.                                                          | NB 02 §4     |
