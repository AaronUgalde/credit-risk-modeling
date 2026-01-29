"""
FastAPI para inferencia del modelo de Credit Risk con Feature Engineering Pipeline
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
import joblib
import numpy as np
import pandas as pd
from typing import Optional, List
import logging
from pathlib import Path
import sys
from sklearn.base import BaseEstimator, TransformerMixin


# ============================================================================
# TRANSFORMADORES PERSONALIZADOS (deben estar antes de cargar el pickle)
# ============================================================================

class RareGrouper(BaseEstimator, TransformerMixin):
    """Agrupa categorías raras en 'OTHER'"""
    def __init__(self, min_pct=0.01):
        self.min_pct = min_pct
        self.mappings = {}
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        for col in X.columns:
            freq = X[col].value_counts(normalize=True)
            rare = freq[freq < self.min_pct].index.tolist()
            self.mappings[col] = set(rare)
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col, rare_set in self.mappings.items():
            X[col] = X[col].where(~X[col].isin(rare_set), 'OTHER')
        return X


class FeatureCreator(BaseEstimator, TransformerMixin):
    """Crea features derivadas para el modelo de riesgo crediticio"""
    def __init__(self, age_bins=[18, 25, 35, 50, 100]):
        self.age_bins = age_bins
        self.income_q01 = None
        self.income_q99 = None
        self.median_loan_rate = None
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.income_q01 = X['person_income'].quantile(0.01)
        self.income_q99 = X['person_income'].quantile(0.99)
        self.median_loan_rate = X['loan_int_rate'].median()
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        # 1. Flags de missing values
        X['is_emp_length_missing'] = X['person_emp_length'].isnull().astype(int)
        X['is_loan_int_rate_missing'] = X['loan_int_rate'].isnull().astype(int)
        
        # 2. Transformaciones logarítmicas
        X['log_person_income'] = np.log1p(X['person_income'].fillna(0))
        X['log_loan_amnt'] = np.log1p(X['loan_amnt'].fillna(0))
        
        # 3. Features de ratio
        X['income_to_loan'] = X['person_income'] / (1 + X['loan_amnt'].fillna(0))
        X['cred_hist_ratio'] = X['cb_person_cred_hist_length'] / (1 + X['person_age'].fillna(0))
        
        # 4. Age buckets
        X['age_bucket'] = pd.cut(
            X['person_age'].fillna(-1),
            bins=self.age_bins,
            labels=False,
            include_lowest=True
        ).astype('Int64').astype(str)
        
        # 5. Feature de interacción
        X['loan_amt_rate_inter'] = X['loan_amnt'] * X['loan_int_rate'].fillna(self.median_loan_rate)
        
        # 6. Clipping de outliers en income
        X['person_income'] = X['person_income'].clip(
            lower=self.income_q01,
            upper=self.income_q99
        )
        
        return X


def binary_mapping(df):
    """Convierte Y/N a 1/0"""
    return pd.DataFrame(df).replace({'Y': 1, 'N': 0}).astype(float)


# ============================================================================
# REGISTRO DE CLASES PARA PICKLE
# ============================================================================
# Esto permite que joblib encuentre las clases al deserializar el pipeline
sys.modules['__main__'].RareGrouper = RareGrouper
sys.modules['__main__'].FeatureCreator = FeatureCreator
sys.modules['__main__'].binary_mapping = binary_mapping


# ============================================================================
# CONFIGURACIÓN DE LA API
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API para predicción de riesgo crediticio con feature engineering automático",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
BASE_PATH = Path(__file__).parent.parent / "notebooks"
MODEL_PATH = BASE_PATH / "best_model_rf_optimized.pkl"
PIPELINE_PATH = BASE_PATH / "feature_engineering_pipeline.pkl"

# Variables globales
model = None
feature_pipeline = None


# ============================================================================
# MODELOS PYDANTIC
# ============================================================================

class LoanApplication(BaseModel):
    """Modelo de entrada para solicitud de préstamo"""
    person_age: int = Field(..., ge=18, le=100, description="Edad del solicitante")
    person_income: float = Field(..., gt=0, description="Ingreso anual")
    person_home_ownership: str = Field(..., description="Tipo de vivienda: RENT, OWN, MORTGAGE, OTHER")
    person_emp_length: Optional[float] = Field(None, ge=0, description="Años de empleo")
    loan_intent: str = Field(..., description="Propósito: PERSONAL, EDUCATION, MEDICAL, VENTURE, HOMEIMPROVEMENT, DEBTCONSOLIDATION")
    loan_grade: str = Field(..., description="Grado: A, B, C, D, E, F, G")
    loan_amnt: float = Field(..., gt=0, description="Monto del préstamo")
    loan_int_rate: Optional[float] = Field(None, ge=0, le=100, description="Tasa de interés")
    loan_percent_income: float = Field(..., ge=0, le=1, description="% del ingreso")
    cb_person_default_on_file: str = Field(..., description="Historial: Y o N")
    cb_person_cred_hist_length: int = Field(..., ge=0, description="Años de historial crediticio")
    
    @field_validator('person_home_ownership')
    @classmethod
    def validate_home_ownership(cls, v):
        valid = ['RENT', 'OWN', 'MORTGAGE', 'OTHER']
        if v.upper() not in valid:
            raise ValueError(f'Debe ser uno de: {valid}')
        return v.upper()
    
    @field_validator('loan_intent')
    @classmethod
    def validate_loan_intent(cls, v):
        valid = ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION']
        if v.upper() not in valid:
            raise ValueError(f'Debe ser uno de: {valid}')
        return v.upper()
    
    @field_validator('loan_grade')
    @classmethod
    def validate_loan_grade(cls, v):
        valid = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
        if v.upper() not in valid:
            raise ValueError(f'Debe ser uno de: {valid}')
        return v.upper()
    
    @field_validator('cb_person_default_on_file')
    @classmethod
    def validate_default_on_file(cls, v):
        if v.upper() not in ['Y', 'N']:
            raise ValueError('Debe ser Y o N')
        return v.upper()

    class Config:
        json_schema_extra = {
            "example": {
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
        }


class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    default_probability: float
    risk_level: str
    prediction: int
    confidence: float


class BatchPredictionResponse(BaseModel):
    """Respuesta batch"""
    predictions: List[PredictionResponse]
    total: int


# ============================================================================
# STARTUP
# ============================================================================

@app.on_event("startup")
async def load_assets():
    """Cargar modelo y pipeline al iniciar"""
    global model, feature_pipeline
    
    try:
        # Cargar pipeline PRIMERO (necesita las clases registradas)
        logger.info(f"📦 Cargando pipeline desde: {PIPELINE_PATH}")
        feature_pipeline = joblib.load(PIPELINE_PATH)
        logger.info(f"✅ Pipeline cargado: {type(feature_pipeline).__name__}")
        
        # Cargar modelo
        logger.info(f"📦 Cargando modelo desde: {MODEL_PATH}")
        model = joblib.load(MODEL_PATH)
        logger.info(f"✅ Modelo cargado: {type(model).__name__}")
        logger.info(f"   Features esperadas: {model.n_features_in_}")
        
    except Exception as e:
        logger.error(f"❌ Error cargando assets: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Credit Risk API with Feature Engineering",
        "version": "2.0.0",
        "status": "active",
        "model_loaded": model is not None,
        "pipeline_loaded": feature_pipeline is not None
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "pipeline_loaded": feature_pipeline is not None,
        "model_type": type(model).__name__ if model else None
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(application: LoanApplication):
    """
    Predice riesgo de default con feature engineering automático
    """
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo o pipeline no cargados")
    
    try:
        # Convertir a DataFrame
        input_data = pd.DataFrame([application.model_dump()])
        logger.info(f"📊 Input shape: {input_data.shape}")
        
        # Aplicar feature engineering
        transformed_data = feature_pipeline.transform(input_data)
        logger.info(f"📊 Transformed shape: {transformed_data.shape}")
        
        # Predicción
        prediction = model.predict(transformed_data)[0]
        probabilities = model.predict_proba(transformed_data)[0]
        
        default_prob = float(probabilities[1])
        confidence = float(max(probabilities))
        
        # Nivel de riesgo
        if default_prob < 0.3:
            risk_level = "LOW"
        elif default_prob < 0.6:
            risk_level = "MEDIUM"
        else:
            risk_level = "HIGH"
        
        logger.info(f"✅ Predicción: prob={default_prob:.3f}, risk={risk_level}")
        
        return PredictionResponse(
            default_probability=default_prob,
            risk_level=risk_level,
            prediction=int(prediction),
            confidence=confidence
        )
    
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(applications: List[LoanApplication]):
    """Predicción en batch"""
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo o pipeline no cargados")
    
    try:
        # Convertir a DataFrame
        input_data = pd.DataFrame([app.model_dump() for app in applications])
        logger.info(f"📊 Batch input: {input_data.shape}")
        
        # Aplicar feature engineering
        transformed_data = feature_pipeline.transform(input_data)
        logger.info(f"📊 Batch transformed: {transformed_data.shape}")
        
        # Predicciones
        predictions = model.predict(transformed_data)
        probabilities = model.predict_proba(transformed_data)
        
        results = []
        for pred, probs in zip(predictions, probabilities):
            default_prob = float(probs[1])
            confidence = float(max(probs))
            
            if default_prob < 0.3:
                risk_level = "LOW"
            elif default_prob < 0.6:
                risk_level = "MEDIUM"
            else:
                risk_level = "HIGH"
            
            results.append(PredictionResponse(
                default_probability=default_prob,
                risk_level=risk_level,
                prediction=int(pred),
                confidence=confidence
            ))
        
        logger.info(f"✅ Batch procesado: {len(results)} predicciones")
        
        return BatchPredictionResponse(
            predictions=results,
            total=len(results)
        )
    
    except Exception as e:
        logger.error(f"❌ Error batch: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model/info")
async def model_info():
    """Información del modelo y pipeline"""
    if model is None or feature_pipeline is None:
        raise HTTPException(status_code=503, detail="Assets no cargados")
    
    return {
        "model": {
            "type": type(model).__name__,
            "n_features": model.n_features_in_,
            "n_estimators": getattr(model, 'n_estimators', None),
        },
        "pipeline": {
            "type": type(feature_pipeline).__name__,
            "steps": [step[0] for step in feature_pipeline.steps] if hasattr(feature_pipeline, 'steps') else None
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
