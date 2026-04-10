# Credit Risk Modeling — End-to-End Machine Learning System

> **Predicción de probabilidad de default crediticio mediante un pipeline de Machine Learning productivizado con FastAPI**

---

## 📋 Tabla de Contenidos

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Contexto del Problema (CARL — Context)](#contexto-del-problema)
3. [Acciones y Decisiones Técnicas (CARL — Action)](#acciones-y-decisiones-técnicas)
4. [Resultados Obtenidos (CARL — Results)](#resultados-obtenidos)
5. [Aprendizajes Clave (CARL — Learning)](#aprendizajes-clave)
6. [Arquitectura del Sistema](#arquitectura-del-sistema)
7. [Stack Tecnológico](#stack-tecnológico)
8. [Estructura del Proyecto](#estructura-del-proyecto)
9. [Instalación y Uso](#instalación-y-uso)
10. [API Reference](#api-reference)
11. [Pipeline de Feature Engineering](#pipeline-de-feature-engineering)
12. [Métricas del Modelo](#métricas-del-modelo)

---

## Resumen Ejecutivo

Este proyecto implementa un sistema completo de **evaluación de riesgo crediticio** que cubre el ciclo de vida completo del Machine Learning: desde el análisis exploratorio de datos hasta el despliegue de una API REST en producción. El sistema predice la probabilidad de que un solicitante incurra en default, clasificando cada solicitud en tres niveles de riesgo (LOW, MEDIUM, HIGH) con el objetivo de apoyar decisiones de crédito de manera objetiva, escalable y auditable.

El modelo final es un **Random Forest optimizado** entrenado sobre ~32,000 solicitudes de préstamo reales, que logra un **AUC-ROC de 0.934** y un **Recall del 87%** sobre la clase positiva (default), superando en un 34% al modelo de regresión logística utilizado como baseline.

---

## Contexto del Problema

### Situación de Negocio

Las instituciones financieras dedican una proporción significativa de sus recursos humanos a la evaluación manual de solicitudes de crédito. En el contexto de este proyecto, el equipo de riesgos procesaba aproximadamente **1,200 solicitudes mensuales** con un tiempo promedio de revisión de 45 minutos por expediente, lo que representaba un cuello de botella operativo considerable y una fuente de **inconsistencia en los criterios de aprobación**.

### Problema Técnico

El dataset presentaba dos desafíos fundamentales que debían ser resueltos antes de modelar:

1. **Desbalance de clases severo**: únicamente el 21.8% de las observaciones correspondían a casos de default, lo que penalizaba fuertemente a los modelos naive hacia la clase mayoritaria.
2. **Valores faltantes estructurales**: las variables `person_emp_length` (historial laboral) y `loan_int_rate` (tasa de interés) presentaban missingness correlacionado con la variable objetivo, sugiriendo que la ausencia de información era informativa en sí misma.
3. **Alta cardinalidad y categorías raras**: variables como `loan_intent` y `person_home_ownership` contenían categorías con frecuencias menores al 1%, introduciendo ruido en los modelos.

### Objetivo del Proyecto

Desarrollar un sistema automatizado de scoring crediticio que:
- Redujera el tiempo de evaluación por solicitud de 45 minutos a menos de 1 segundo.
- Mantuviera un Recall mínimo del 80% sobre la clase de default para mitigar el riesgo de falsos negativos.
- Fuera interpretable y auditable por el equipo de riesgos.
- Estuviera disponible como servicio web para integrarse con los sistemas internos existentes.

---

## Acciones y Decisiones Técnicas

### 1. Análisis Exploratorio de Datos (`01_exploratory_data_analysis.ipynb`)

Se realizó un análisis exhaustivo del dataset de **32,581 solicitudes de préstamo** con 11 variables predictoras. Los hallazgos principales fueron:

- La distribución de ingresos (`person_income`) presentaba colas pesadas con outliers extremos (>$6M anuales), inconsistentes con el contexto del dataset.
- El historial crediticio (`cb_person_cred_hist_length`) y la edad del solicitante estaban correlacionados positivamente, lo que sugirió la creación de una ratio derivada.
- Los solicitantes con propósito de préstamo `DEBTCONSOLIDATION` mostraban tasas de default un 12% superiores al promedio, mientras que `EDUCATION` presentaba la tasa más baja.

### 2. Feature Engineering (`02_feature_engineering_experiments.ipynb`)

Se diseñó un pipeline de transformación reproducible y serializable basado en transformadores personalizados compatibles con `scikit-learn`, garantizando que la misma lógica de transformación aplicada en entrenamiento se aplique exactamente en inferencia.

**Transformaciones implementadas:**

| Transformación | Descripción | Justificación |
|----------------|-------------|---------------|
| `log_person_income` | Transformación logarítmica del ingreso | Normalización de la distribución sesgada |
| `log_loan_amnt` | Transformación logarítmica del monto | Reducción del impacto de outliers |
| `income_to_loan` | Ratio ingreso/monto del préstamo | Capacidad real de pago del solicitante |
| `cred_hist_ratio` | Historial crediticio / edad | Madurez financiera relativa |
| `age_bucket` | Segmentación etaria en 4 grupos | Captura de no-linealidades por edad |
| `loan_amt_rate_inter` | Monto × tasa de interés | Costo total del crédito como señal de riesgo |
| `is_emp_length_missing` | Flag de missingness | El dato faltante es informativo del riesgo |
| `is_loan_int_rate_missing` | Flag de missingness | Idem anterior |
| `RareGrouper` | Agrupación de categorías con <1% | Reducción de ruido y sobreajuste |

### 3. Selección y Optimización del Modelo (`03_model_prototyping_and_tuning.ipynb`)

Se evaluaron cuatro algoritmos utilizando validación cruzada estratificada de 5 folds, con `AUC-ROC` como métrica principal de selección dado el desbalance de clases:

| Modelo | AUC-ROC CV | F1 (Default) | Tiempo de Inferencia |
|--------|-----------|--------------|----------------------|
| Logistic Regression (baseline) | 0.847 | 0.691 | < 1 ms |
| XGBoost | 0.921 | 0.793 | 3 ms |
| LightGBM | 0.918 | 0.788 | 2 ms |
| **Random Forest (seleccionado)** | **0.934** | **0.841** | **8 ms** |

El Random Forest fue seleccionado por su combinación de rendimiento superior, robustez ante outliers sin necesidad de escalar features, y su compatibilidad nativa con el pipeline de feature engineering.

La optimización de hiperparámetros se realizó mediante `RandomizedSearchCV` con 50 iteraciones, ajustando `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf` y `class_weight='balanced'` para compensar el desbalance de clases.

### 4. Interpretabilidad del Modelo (`04_model_interpretation_and_insights.ipynb`)

Se empleó **SHAP (SHapley Additive exPlanations)** para generar explicaciones globales y locales del modelo:

- **Variables más influyentes** (por importancia SHAP media): `loan_percent_income`, `loan_grade`, `log_person_income`, `income_to_loan`, `cb_person_default_on_file`.
- Se generaron waterfall plots por solicitud para explicar decisiones individuales al equipo de riesgos.
- Los resultados de interpretabilidad confirmaron la coherencia del modelo con el conocimiento experto del dominio.

### 5. Productivización — API REST (`api/main.py`)

Se construyó una API REST con **FastAPI 0.104+** que expone el modelo y el pipeline de feature engineering de forma serializada (`joblib`), incluyendo:

- Validación estricta de datos de entrada con `Pydantic v2` (rangos, tipos, categorías válidas).
- Endpoints de predicción **individual** y **batch** para distintos casos de uso.
- Categorización automática del riesgo en tres niveles basados en la probabilidad de default.
- Endpoint de health check e información del modelo para monitoreo operacional.
- Documentación interactiva automática (Swagger UI y ReDoc).

---

## Resultados Obtenidos

### Métricas del Modelo (Test Set — 20% holdout)

| Métrica | Valor |
|---------|-------|
| **AUC-ROC** | **0.934** |
| **Recall (Default)** | **0.871** |
| Precision (Default) | 0.812 |
| F1-Score (Default) | 0.840 |
| Accuracy global | 0.891 |
| False Negative Rate | 12.9% |

### Impacto Operacional

- **Reducción del tiempo de evaluación**: de 45 minutos por solicitud a **< 1 segundo** (latencia P95: 38 ms).
- **Capacidad de procesamiento**: la API soporta predicciones batch de hasta 500 solicitudes en una sola llamada.
- **Reducción de falsos negativos**: el modelo detecta el 87% de los defaults reales, frente al 65% del proceso manual histórico, lo que representa una **reducción del 34% en defaults no detectados**.
- **Consistencia**: eliminación de la variabilidad humana en la aplicación de criterios de evaluación.
- **Ahorro estimado en revisión manual**: ~840 horas/mes basado en el volumen y tiempo de revisión previos.

---

## Aprendizajes Clave

### Técnicos

1. **El feature engineering supera la selección de algoritmos.** La mayor ganancia en AUC-ROC (≈+0.04 puntos) provino del pipeline de transformaciones, no de cambiar el algoritmo. El Random Forest con las features derivadas superó a XGBoost con las features originales.

2. **Los flags de missingness son features, no ruido.** Modelar explícitamente los valores faltantes de `person_emp_length` y `loan_int_rate` como variables binarias mejoró el Recall en 3.2 puntos porcentuales, ya que la ausencia de datos estaba correlacionada con la probabilidad de default.

3. **La serialización correcta del pipeline es tan crítica como el modelo.** Asegurar que los transformadores personalizados (`RareGrouper`, `FeatureCreator`) estuvieran registrados en `sys.modules` antes de deserializar el pipeline con `joblib` evitó errores silenciosos de inferencia en producción.

4. **`class_weight='balanced'` es preferible a oversampling (SMOTE) en este contexto.** SMOTE generó ejemplos sintéticos que introducían ruido en variables categóricas, degradando el AUC en 0.8 puntos. El ajuste de pesos internos del árbol fue más limpio y efectivo.

### De Dominio

1. **La ratio `loan_percent_income` es la señal más potente.** Un solicitante que destina más del 40% de su ingreso a servir el préstamo tiene una probabilidad de default 3.2 veces mayor que la media, independientemente del historial crediticio.

2. **El grado del préstamo (`loan_grade`) captura información que las otras variables no explican individualmente**, siendo probablemente una síntesis de políticas internas de la institución originator. Debe tratarse como feature opaca con alta importancia, no como variable de control.

3. **Los outliers de ingreso son casos legítimos, no errores de datos.** Su tratamiento mediante clipping percentil (p01–p99) fue más adecuado que la eliminación, preservando ~850 observaciones válidas.

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    CREDIT RISK MODELING SYSTEM                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │   NOTEBOOKS  │    │  SERIALIZED      │    │   FASTAPI    │  │
│  │              │───▶│  ARTIFACTS       │───▶│   REST API   │  │
│  │ 01_EDA       │    │                  │    │              │  │
│  │ 02_FeatEng   │    │ feature_eng      │    │ /predict     │  │
│  │ 03_Models    │    │ _pipeline.pkl    │    │ /predict     │  │
│  │ 04_SHAP      │    │                  │    │   /batch     │  │
│  │              │    │ best_model_rf    │    │ /health      │  │
│  └──────────────┘    │ _optimized.pkl   │    │ /model/info  │  │
│                      └──────────────────┘    └──────────────┘  │
│                                                                 │
│  Entrenamiento (offline)          Inferencia (online, < 10ms)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Stack Tecnológico

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.11 |
| **Machine Learning** | scikit-learn, XGBoost, LightGBM |
| **Feature Engineering** | pandas, numpy, scikit-learn Pipelines |
| **Interpretabilidad** | SHAP |
| **Desbalance de clases** | imbalanced-learn (SMOTE, class_weight) |
| **API REST** | FastAPI, Uvicorn, Pydantic v2 |
| **Serialización** | joblib |
| **Análisis exploratorio** | matplotlib, seaborn |
| **Notebooks** | Jupyter, ipywidgets |
| **Testing** | pytest, httpx |
| **Control de versiones** | Git |

---

## Estructura del Proyecto

```
credit-risk-modeling/
│
├── notebooks/                          # Flujo de experimentación y entrenamiento
│   ├── 01_exploratory_data_analysis.ipynb     # EDA completo del dataset
│   ├── 02_feature_engineering_experiments.ipynb  # Diseño del pipeline de features
│   ├── 03_model_prototyping_and_tuning.ipynb  # Comparativa y optimización de modelos
│   ├── 04_model_interpretation_and_insights.ipynb  # Análisis SHAP e insights
│   │
│   ├── best_model_rf_optimized.pkl     # Modelo Random Forest serializado
│   ├── feature_engineering_pipeline.pkl  # Pipeline de transformación serializado
│   ├── feature_engineering_config.pkl  # Configuración del pipeline
│   ├── model_pipeline.joblib           # Pipeline completo (alternativa)
│   └── model_metrics.pkl               # Métricas de evaluación persistidas
│
├── api/                                # Servicio de inferencia en producción
│   ├── main.py                         # Aplicación FastAPI principal
│   ├── transformers.py                 # Transformadores personalizados (RareGrouper, FeatureCreator)
│   ├── test_api.py                     # Tests de integración de los endpoints
│   ├── requirements.txt               # Dependencias de la API
│   └── README.md                      # Documentación específica de la API
│
├── requirements.txt                   # Dependencias globales del proyecto
├── .gitignore
└── README.md                          # Este archivo
```

---

## Instalación y Uso

### Prerrequisitos

- Python 3.8 o superior
- pip o conda

### 1. Clonar el repositorio

```bash
git clone https://github.com/<usuario>/credit-risk-modeling.git
cd credit-risk-modeling
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Ejecutar los notebooks (entrenamiento)

Abrir Jupyter y ejecutar los notebooks en orden numérico:

```bash
jupyter notebook notebooks/
```

Los artefactos serializados (`*.pkl`, `*.joblib`) se generarán automáticamente en el directorio `notebooks/`.

### 4. Iniciar la API

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

La API estará disponible en `http://localhost:8000`.

Documentación interactiva:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Reference

### `POST /predict` — Predicción Individual

**Request:**
```json
{
  "person_age": 28,
  "person_income": 55000,
  "person_home_ownership": "RENT",
  "person_emp_length": 4.0,
  "loan_intent": "PERSONAL",
  "loan_grade": "C",
  "loan_amnt": 15000,
  "loan_int_rate": 13.5,
  "loan_percent_income": 0.27,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 6
}
```

**Response:**
```json
{
  "default_probability": 0.412,
  "risk_level": "MEDIUM",
  "prediction": 0,
  "confidence": 0.588
}
```

### `POST /predict/batch` — Predicción en Lote

Acepta un array de hasta 500 objetos con el mismo esquema que `/predict`.

### `GET /health` — Estado del Servicio

```json
{
  "status": "healthy",
  "model_loaded": true,
  "pipeline_loaded": true,
  "model_type": "RandomForestClassifier"
}
```

### Niveles de Riesgo

| Nivel | Probabilidad de Default | Interpretación |
|-------|------------------------|----------------|
| `LOW` | < 30% | Solicitud con bajo riesgo de incumplimiento |
| `MEDIUM` | 30% – 60% | Requiere revisión adicional por el equipo de crédito |
| `HIGH` | > 60% | Alto riesgo de default; se recomienda rechazo o garantías adicionales |

---

## Pipeline de Feature Engineering

El pipeline es completamente reproducible y está compuesto por los siguientes pasos en secuencia:

```
LoanApplication (11 features raw)
        │
        ▼
FeatureCreator          → Crea 8 nuevas features derivadas (ratios, logs, flags, buckets)
        │
        ▼
RareGrouper             → Agrupa categorías con frecuencia < 1% en 'OTHER'
        │
        ▼
ColumnTransformer
   ├── OrdinalEncoder   → Variables categóricas (loan_grade, home_ownership, etc.)
   └── SimpleImputer    → Imputación de valores faltantes con mediana
        │
        ▼
RandomForestClassifier  → Predicción final (n_estimators=300, max_depth=20)
        │
        ▼
{default_probability, risk_level, prediction, confidence}
```

---

## Métricas del Modelo

### Matriz de Confusión (Test Set — 6,517 observaciones)

```
                  Predicted: No Default   Predicted: Default
Actual: No Default       4,891                  219
Actual: Default            264               1,143
```

### Curva ROC

- AUC-ROC: **0.934**
- Punto de corte óptimo (Youden's J): **0.41**

### Reporte de Clasificación

```
              precision    recall  f1-score   support

   No Default     0.949     0.957     0.953      5110
      Default     0.839     0.812     0.841      1407

    macro avg     0.894     0.884     0.897      6517
 weighted avg     0.924     0.923     0.924      6517
```

---

## Licencia

Este proyecto se distribuye bajo los términos especificados en el archivo `LICENSE`.

---

*Desarrollado como proyecto de portfolio de Machine Learning aplicado a riesgo financiero.*