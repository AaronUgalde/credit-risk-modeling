"""
Script de ejemplo para testear la API de Credit Risk
"""
import requests
import json

# Configuración
BASE_URL = "http://localhost:8000"

def test_health():
    """Test del endpoint de health check"""
    print("\n" + "="*50)
    print("TEST: Health Check")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test del endpoint de información del modelo"""
    print("\n" + "="*50)
    print("TEST: Model Info")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_single_prediction():
    """Test de predicción individual"""
    print("\n" + "="*50)
    print("TEST: Predicción Individual")
    print("="*50)
    
    # Caso 1: Solicitud de bajo riesgo
    low_risk_app = {
        "person_age": 30,
        "person_income": 75000,
        "person_home_ownership": "OWN",
        "person_emp_length": 8.0,
        "loan_intent": "HOMEIMPROVEMENT",
        "loan_grade": "A",
        "loan_amnt": 10000,
        "loan_int_rate": 7.5,
        "loan_percent_income": 0.13,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 10
    }
    
    print("\nCaso 1: Solicitud de BAJO riesgo")
    print(json.dumps(low_risk_app, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=low_risk_app)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Probabilidad de default: {result['default_probability']:.2%}")
    print(f"Nivel de riesgo: {result['risk_level']}")
    print(f"Predicción: {'Default ❌' if result['prediction'] == 1 else 'No Default ✅'}")
    print(f"Confianza: {result['confidence']:.2%}")
    
    # Caso 2: Solicitud de alto riesgo
    high_risk_app = {
        "person_age": 22,
        "person_income": 25000,
        "person_home_ownership": "RENT",
        "person_emp_length": 0.5,
        "loan_intent": "PERSONAL",
        "loan_grade": "F",
        "loan_amnt": 35000,
        "loan_int_rate": 21.5,
        "loan_percent_income": 0.85,
        "cb_person_default_on_file": "Y",
        "cb_person_cred_hist_length": 1
    }
    
    print("\n" + "-"*50)
    print("\nCaso 2: Solicitud de ALTO riesgo")
    print(json.dumps(high_risk_app, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=high_risk_app)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Probabilidad de default: {result['default_probability']:.2%}")
    print(f"Nivel de riesgo: {result['risk_level']}")
    print(f"Predicción: {'Default ❌' if result['prediction'] == 1 else 'No Default ✅'}")
    print(f"Confianza: {result['confidence']:.2%}")

def test_batch_prediction():
    """Test de predicción en lote"""
    print("\n" + "="*50)
    print("TEST: Predicción en Lote")
    print("="*50)
    
    applications = [
        {
            "person_age": 28,
            "person_income": 55000,
            "person_home_ownership": "RENT",
            "person_emp_length": 4.0,
            "loan_intent": "EDUCATION",
            "loan_grade": "B",
            "loan_amnt": 15000,
            "loan_int_rate": 11.5,
            "loan_percent_income": 0.27,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 6
        },
        {
            "person_age": 35,
            "person_income": 90000,
            "person_home_ownership": "MORTGAGE",
            "person_emp_length": 12.0,
            "loan_intent": "VENTURE",
            "loan_grade": "C",
            "loan_amnt": 25000,
            "loan_int_rate": 13.0,
            "loan_percent_income": 0.28,
            "cb_person_default_on_file": "N",
            "cb_person_cred_hist_length": 8
        },
        {
            "person_age": 26,
            "person_income": 40000,
            "person_home_ownership": "RENT",
            "person_emp_length": 2.0,
            "loan_intent": "DEBTCONSOLIDATION",
            "loan_grade": "D",
            "loan_amnt": 20000,
            "loan_int_rate": 16.5,
            "loan_percent_income": 0.50,
            "cb_person_default_on_file": "Y",
            "cb_person_cred_hist_length": 3
        }
    ]
    
    print(f"\nEnviando {len(applications)} solicitudes...")
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=applications)
    result = response.json()
    
    print(f"\nStatus Code: {response.status_code}")
    print(f"Total procesadas: {result['total']}")
    print("\nResultados:")
    print("-"*80)
    
    for i, pred in enumerate(result['predictions'], 1):
        print(f"\nSolicitud {i}:")
        print(f"  Probabilidad default: {pred['default_probability']:.2%}")
        print(f"  Nivel de riesgo: {pred['risk_level']}")
        print(f"  Predicción: {'Default ❌' if pred['prediction'] == 1 else 'No Default ✅'}")
        print(f"  Confianza: {pred['confidence']:.2%}")

def test_validation_errors():
    """Test de validación de errores"""
    print("\n" + "="*50)
    print("TEST: Validación de Errores")
    print("="*50)
    
    # Caso 1: Edad fuera de rango
    print("\nCaso 1: Edad inválida (menor a 18)")
    invalid_age = {
        "person_age": 17,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "loan_intent": "EDUCATION",
        "loan_grade": "B",
        "loan_amnt": 10000,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_age)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error esperado: {response.json()}")
    
    # Caso 2: Categoría inválida
    print("\nCaso 2: Loan grade inválido")
    invalid_grade = {
        "person_age": 25,
        "person_income": 50000,
        "person_home_ownership": "RENT",
        "loan_intent": "EDUCATION",
        "loan_grade": "Z",  # Inválido
        "loan_amnt": 10000,
        "loan_percent_income": 0.20,
        "cb_person_default_on_file": "N",
        "cb_person_cred_hist_length": 5
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=invalid_grade)
    print(f"Status Code: {response.status_code}")
    if response.status_code != 200:
        print(f"Error esperado: {response.json()}")

if __name__ == "__main__":
    print("\n" + "="*50)
    print("INICIANDO TESTS DE LA API")
    print("="*50)
    
    try:
        test_health()
        test_model_info()
        test_single_prediction()
        test_batch_prediction()
        test_validation_errors()
        
        print("\n" + "="*50)
        print("TESTS COMPLETADOS ✅")
        print("="*50)
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: No se puede conectar a la API")
        print("Asegúrate de que el servidor esté corriendo en http://localhost:8000")
        print("Ejecuta: uvicorn main:app --reload")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
