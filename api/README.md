# API de Predicción de Riesgo Crediticio

API REST construida con FastAPI para realizar inferencia con el modelo de Random Forest entrenado para predicción de riesgo crediticio.

## 🚀 Características

- Predicción individual de riesgo de default
- Predicción en lote (batch)
- Validación automática de datos de entrada
- Documentación interactiva automática (Swagger UI)
- Niveles de riesgo categorizados (LOW, MEDIUM, HIGH)
- Health check endpoint

## 📋 Requisitos

- Python 3.8+
- Dependencias listadas en `requirements.txt`

## 🔧 Instalación

1. Crear un entorno virtual:
```bash
python -m venv venv
```

2. Activar el entorno virtual:
```bash
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. Instalar dependencias:
```bash
pip install -r requirements.txt
```

## ▶️ Uso

### Iniciar el servidor

```bash
uvicorn main:app --reload
```

El servidor estará disponible en `http://localhost:8000`

### Documentación interactiva

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 📡 Endpoints

### 1. Root
```http
GET /
```
Información básica de la API.

### 2. Health Check
```http
GET /health
```
Verifica el estado de la API y si el modelo está cargado.

### 3. Información del Modelo
```http
GET /model/info
```
Retorna información sobre el modelo cargado.

### 4. Predicción Individual
```http
POST /predict
```

**Request Body:**
```json
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

**Response:**
```json
{
  "default_probability": 0.23,
  "risk_level": "LOW",
  "prediction": 0,
  "confidence": 0.77
}
```

### 5. Predicción en Lote
```http
POST /predict/batch
```

**Request Body:**
```json
[
  {
    "person_age": 25,
    "person_income": 50000,
    ...
  },
  {
    "person_age": 30,
    "person_income": 65000,
    ...
  }
]
```

## 📊 Campos de Entrada

| Campo | Tipo | Descripción | Valores |
|-------|------|-------------|---------|
| `person_age` | int | Edad del solicitante | 18-100 |
| `person_income` | float | Ingreso anual | > 0 |
| `person_home_ownership` | str | Tipo de vivienda | RENT, OWN, MORTGAGE, OTHER |
| `person_emp_length` | float (opcional) | Años de empleo | >= 0 |
| `loan_intent` | str | Propósito del préstamo | PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION |
| `loan_grade` | str | Grado del préstamo | A, B, C, D, E, F, G |
| `loan_amnt` | float | Monto del préstamo | > 0 |
| `loan_int_rate` | float (opcional) | Tasa de interés | 0-100 |
| `loan_percent_income` | float | % ingreso que representa el préstamo | 0-1 |
| `cb_person_default_on_file` | str | Historial de default | Y, N |
| `cb_person_cred_hist_length` | int | Años de historial crediticio | >= 0 |

## 📈 Respuesta de Predicción

| Campo | Tipo | Descripción |
|-------|------|-------------|
| `default_probability` | float | Probabilidad de default (0-1) |
| `risk_level` | str | Nivel de riesgo: LOW (<0.3), MEDIUM (0.3-0.6), HIGH (>0.6) |
| `prediction` | int | Predicción binaria: 0 (no default), 1 (default) |
| `confidence` | float | Confianza de la predicción (max probability) |

## 🧪 Ejemplo de Uso con Python

```python
import requests

# URL base de la API
BASE_URL = "http://localhost:8000"

# Ejemplo de predicción individual
application = {
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

response = requests.post(f"{BASE_URL}/predict", json=application)
result = response.json()

print(f"Probabilidad de default: {result['default_probability']:.2%}")
print(f"Nivel de riesgo: {result['risk_level']}")
print(f"Predicción: {'Default' if result['prediction'] == 1 else 'No Default'}")
```

## 🧪 Ejemplo con cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```

## 🛠️ Estructura del Proyecto

```
api/
├── main.py              # Aplicación FastAPI principal
├── transformers.py      # Transformadores personalizados
├── requirements.txt     # Dependencias
└── README.md           # Este archivo
```

## 📝 Notas

- El modelo espera exactamente 11 features de entrada (excluyendo `loan_status`)
- Los campos opcionales (`person_emp_length`, `loan_int_rate`) pueden ser `null`
- La API valida automáticamente tipos y rangos de valores
- Las categorías de texto no son case-sensitive (se convierten a uppercase)

## 🐛 Troubleshooting

### Error: "Modelo no cargado"
Asegúrate de que `best_model_rf_optimized.pkl` existe en el directorio `notebooks/`

### Error de validación
Verifica que todos los campos requeridos estén presentes y tengan valores válidos

## 📄 Licencia

Este proyecto es parte del sistema de modelado de riesgo crediticio.
