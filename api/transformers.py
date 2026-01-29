import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class RareGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa categorías raras en 'OTHER'.
    
    Parameters:
    -----------
    min_pct : float
        Umbral de frecuencia mínima (default: 0.01 = 1%)
        Categorías con frecuencia menor a este valor se agrupan.
    
    Example:
    --------
    >>> rg = RareGrouper(min_pct=0.05)  # Agrupar categorías con < 5%
    >>> rg.fit_transform(df[['categorical_col']])
    """
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
    """
    Crea features derivadas para el modelo de riesgo crediticio.
    
    Features creadas:
    ----------------
    1. is_emp_length_missing: Flag binario para missing values en employment length
    2. is_loan_int_rate_missing: Flag binario para missing values en interest rate
    3. log_person_income: Transformación logarítmica del ingreso
    4. log_loan_amnt: Transformación logarítmica del monto del préstamo
    5. income_to_loan: Ratio ingreso/monto del préstamo
    6. cred_hist_ratio: Ratio historial crediticio/edad
    7. age_bucket: Categorización de edad en bins
    8. loan_amt_rate_inter: Interacción monto × tasa de interés
    
    Parameters:
    -----------
    age_bins : list
        Límites de bins para categorizar edad (default: [18, 25, 35, 50, 100])
    """
    def __init__(self, age_bins=[18, 25, 35, 50, 100]):
        self.age_bins = age_bins
        self.income_q01 = None
        self.income_q99 = None
        self.median_loan_rate = None
    
    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        # Guardar estadísticas del training set
        self.income_q01 = X['person_income'].quantile(0.01)
        self.income_q99 = X['person_income'].quantile(0.99)
        self.median_loan_rate = X['loan_int_rate'].median()
        return self
    
    def transform(self, X):
        X = pd.DataFrame(X).copy()
        
        # 1. Flags de missing values
        X['is_emp_length_missing'] = X['person_emp_length'].isnull().astype(int)
        X['is_loan_int_rate_missing'] = X['loan_int_rate'].isnull().astype(int)
        
        # 2. Transformaciones logarítmicas (log1p maneja zeros)
        X['log_person_income'] = np.log1p(X['person_income'].fillna(0))
        X['log_loan_amnt'] = np.log1p(X['loan_amnt'].fillna(0))
        
        # 3. Features de ratio
        X['income_to_loan'] = X['person_income'] / (1 + X['loan_amnt'].fillna(0))
        X['cred_hist_ratio'] = X['cb_person_cred_hist_length'] / (1 + X['person_age'].fillna(0))
        
        # 4. Age buckets (como string para OneHotEncoding posterior)
        X['age_bucket'] = pd.cut(
            X['person_age'].fillna(-1),
            bins=self.age_bins,
            labels=False,
            include_lowest=True
        ).astype('Int64').astype(str)
        
        # 5. Feature de interacción
        X['loan_amt_rate_inter'] = X['loan_amnt'] * X['loan_int_rate'].fillna(self.median_loan_rate)
        
        # 6. Clipping de outliers en income (usando quantiles del train set)
        X['person_income'] = X['person_income'].clip(
            lower=self.income_q01,
            upper=self.income_q99
        )
        
        return X


def binary_mapping(df):
    """Convierte Y/N a 1/0"""
    return pd.DataFrame(df).replace({'Y': 1, 'N': 0}).astype(float)
