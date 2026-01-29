# Credit Risk Modeling Project

Sistema completo de predicción de riesgo crediticio utilizando Machine Learning, con pipeline de feature engineering automatizado y API REST para inferencia.

## 📋 Contenido del Proyecto

```
credit-risk-modeling/
├── api/                          # API REST para inferencia
│   ├── main.py                   # FastAPI application
│   ├── transformers.py           # Transformadores personalizados
│   ├── requirements.txt          # Dependencias de la API
│   ├── test_api.py              # Tests de la API
│   └── README.md                # Documentación de la API
│
├── notebooks/                    # Notebooks de análisis y modelado
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_engineering_experiments.ipynb
│   ├── 03_model_prototyping_and_tuning.ipynb
│   ├── 04_model_interpretation_and_insights.ipynb
│   ├── best_model_rf_optimized.pkl
│   ├── feature_engineering_pipeline.pkl
│   └── requirements.txt         # Dependencias para notebooks
│
├── .gitignore
├── README.md
└── requirements.txt             # Dependencias globales del proyecto
```

## 🎯 Características

- **Feature Engineering Automatizado**: Pipeline completo de transformación de datos
- **Modelo Optimizado**: Random Forest con hiperparámetros optimizados
- **API REST**: FastAPI para inferencia en producción
- **Análisis Completo**: Notebooks con EDA, feature engineering, modelado e interpretación
- **Transformadores Personalizados**: RareGrouper, FeatureCreator
- **Validación Robusta**: Manejo de missing values y outliers

## 🚀 Quick Start

### 1. Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/credit-risk-modeling.git
cd credit-risk-modeling
```

### 2. Crear entorno virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 4. Ejecutar la API

```bash
cd api
python main.py
```

La API estará disponible en: http://localhost:8000

Documentación interactiva: http://localhost:8000/docs

## 📊 Dataset

El proyecto utiliza el dataset de **Credit Risk** de Kaggle:
- **Fuente**: [Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- **Tamaño**: 32,581 registros
- **Features**: 11 variables originales → 30 features después del pipeline
- **Target**: loan_status (0: no default, 1: default)

### Variables de entrada:
- `person_age`: Edad del solicitante
- `person_income`: Ingreso anual
- `person_home_ownership`: Tipo de vivienda (RENT, OWN, MORTGAGE, OTHER)
- `person_emp_length`: Años de empleo
- `loan_intent`: Propósito del préstamo
- `loan_grade`: Calificación del préstamo (A-G)
- `loan_amnt`: Monto del préstamo
- `loan_int_rate`: Tasa de interés
- `loan_percent_income`: Porcentaje del ingreso
- `cb_person_default_on_file`: Historial de default (Y/N)
- `cb_person_cred_hist_length`: Años de historial crediticio

## 🔧 Feature Engineering Pipeline

El pipeline aplica las siguientes transformaciones automáticamente:

### 1. Feature Creation (FeatureCreator)
Crea 8 nuevas features derivadas:
- `is_emp_length_missing`, `is_loan_int_rate_missing`: Flags de missing values
- `log_person_income`, `log_loan_amnt`: Transformaciones logarítmicas
- `income_to_loan`: Ratio ingreso/préstamo
- `cred_hist_ratio`: Ratio historial/edad
- `age_bucket`: Categorización por edad
- `loan_amt_rate_inter`: Interacción monto × tasa

### 2. Preprocessing (ColumnTransformer)
- **Numéricas** (14): Imputación mediana + RobustScaler
- **Ordinales** (1): OrdinalEncoder con jerarquía A < B < ... < G
- **Binarias** (1): Mapeo Y/N → 1/0
- **Nominales** (3): RareGrouper (1%) + OneHotEncoder

**Resultado**: 11 variables → 30 features procesadas

## 📈 Modelo

- **Algoritmo**: Random Forest Classifier
- **Hiperparámetros optimizados**: RandomizedSearchCV
- **Features**: 30 (después del pipeline)
- **Métricas de validación**: ROC-AUC, Precision, Recall, F1-Score

## 🌐 API REST

### Endpoints principales:

#### Health Check
```bash
GET /health
```

#### Predicción Individual
```bash
POST /predict
Content-Type: application/json

{
  "person_age": 25,
  "person_income": 50000,
  "person_home_ownership": "RENT",
  "person_emp_length": 3.0,
  "loan_intent": "EDUCATION",
  "loan_grade": "B",
  "loan_amnt": 10000,
  "loan_int_rate": 11.5,
  "loan_percent_income": 0.20,
  "cb_person_default_on_file": "N",
  "cb_person_cred_hist_length": 5
}
```

**Respuesta**:
```json
{
  "default_probability": 0.123,
  "risk_level": "LOW",
  "prediction": 0,
  "confidence": 0.877
}
```

#### Predicción Batch
```bash
POST /predict/batch
Content-Type: application/json

[{...}, {...}, {...}]
```

#### Info del Modelo
```bash
GET /model/info
```

## 🧪 Testing

```bash
cd api
python test_api.py
```

## 📚 Notebooks

### 01. Exploratory Data Analysis
- Análisis univariado y bivariado
- Detección de outliers
- Análisis de correlaciones
- Visualizaciones

### 02. Feature Engineering Experiments
- Creación de transformadores personalizados
- Pipeline de preprocesamiento
- Análisis de features derivadas
- Exportación del pipeline

### 03. Model Prototyping and Tuning
- Comparación de algoritmos
- Optimización de hiperparámetros
- Validación cruzada
- Selección del mejor modelo

### 04. Model Interpretation and Insights
- Feature importance
- SHAP values
- Análisis de predicciones
- Insights de negocio

## 🛠️ Tecnologías Utilizadas

- **Python 3.11+**
- **FastAPI**: Framework web para la API
- **scikit-learn**: Machine Learning
- **pandas, numpy**: Manipulación de datos
- **joblib**: Serialización de modelos
- **Jupyter**: Notebooks interactivos
- **matplotlib, seaborn**: Visualización

## 📦 Dependencias

Ver `requirements.txt` para la lista completa de dependencias.

Principales:
- fastapi==0.104.1
- uvicorn[standard]==0.24.0
- scikit-learn==1.3.2
- pandas==2.1.3
- numpy==1.26.2
- joblib==1.3.2

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama para tu feature (`git checkout -b feature/amazing-feature`)
3. Commit tus cambios (`git commit -m 'Add amazing feature'`)
4. Push a la rama (`git push origin feature/amazing-feature`)
5. Abre un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más información.

## 👤 Autor

Tu Nombre
- GitHub: [@AaronUgalde](https://github.com/AaronUgalde)
- LinkedIn: [Ugalde-Tellez-Aaron](www.linkedin.com/in/ugalde-tellez-aaron-b76567353)

## 🙏 Agradecimientos

- Dataset: [Kaggle - Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)
- Comunidad de Data Science y Machine Learning

## 📞 Contacto

Para preguntas o sugerencias, por favor abre un issue en GitHub.

---

**⭐ Si este proyecto te fue útil, considera darle una estrella!**
