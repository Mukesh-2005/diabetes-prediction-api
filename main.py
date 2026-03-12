"""
FastAPI Diabetes Prediction API
Uses trained Gradient Boosting model with 42 features
Kaggle Competition Dataset - 700k medical records
"""

from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field, ConfigDict
import joblib
import numpy as np
import pandas as pd
from typing import List
import uvicorn

# ==================== LOAD MODEL & FILES ====================

print("Loading model and preprocessing files...")

try:
    model = joblib.load('diabetes_model.pkl')
    scaler = joblib.load('scaler.pkl')
    encoder = joblib.load('encoder.pkl')
    feature_names = joblib.load('feature_names.pkl')
    numerical_features = joblib.load('numerical_features.pkl')
    model_metadata = joblib.load('model_metadata.pkl')
    print("✅ All files loaded successfully!")
except Exception as e:
    print(f"❌ Error loading files: {e}")
    raise

# ==================== FASTAPI APP ====================

app = FastAPI(
    title="🏥 Diabetes Prediction API",
    description="Predict diabetes risk using Gradient Boosting (66% accuracy)",
    version="1.0.0"
)

# ==================== PYDANTIC MODELS ====================

class DiabetesInput(BaseModel):
    """Patient medical data for prediction"""
    age: int = Field(..., ge=19, le=89, description="Age in years")
    alcohol_consumption_per_week: int = Field(..., ge=0, le=10)
    physical_activity_minutes_per_week: int = Field(..., ge=0, le=1000)
    diet_score: float = Field(..., ge=0, le=10)
    sleep_hours_per_day: float = Field(..., ge=0, le=24)
    screen_time_hours_per_day: float = Field(..., ge=0, le=24)
    bmi: float = Field(..., ge=10, le=60)
    waist_to_hip_ratio: float = Field(..., ge=0.5, le=1.5)
    systolic_bp: int = Field(..., ge=70, le=200)
    diastolic_bp: int = Field(..., ge=40, le=130)
    heart_rate: int = Field(..., ge=40, le=150)
    cholesterol_total: int = Field(..., ge=100, le=300)
    hdl_cholesterol: int = Field(..., ge=20, le=100)
    ldl_cholesterol: int = Field(..., ge=30, le=200)
    triglycerides: int = Field(..., ge=30, le=500)
    family_history_diabetes: int = Field(..., ge=0, le=1)
    hypertension_history: int = Field(..., ge=0, le=1)
    cardiovascular_history: int = Field(..., ge=0, le=1)
    gender: str = Field(..., description="Female, Male, or Other")
    ethnicity: str = Field(..., description="Hispanic, White, Asian, Black, or Other")
    education_level: str = Field(..., description="Highschool, Graduate, Postgraduate, No formal")
    income_level: str = Field(..., description="Low, Lower-Middle, Middle, Upper-Middle, High")
    smoking_status: str = Field(..., description="Current, Never, or Former")
    employment_status: str = Field(..., description="Employed, Retired, Student, or Unemployed")
    
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 55,
            "alcohol_consumption_per_week": 2,
            "physical_activity_minutes_per_week": 150,
            "diet_score": 7.5,
            "sleep_hours_per_day": 7.5,
            "screen_time_hours_per_day": 4.0,
            "bmi": 28.5,
            "waist_to_hip_ratio": 0.9,
            "systolic_bp": 130,
            "diastolic_bp": 85,
            "heart_rate": 75,
            "cholesterol_total": 220,
            "hdl_cholesterol": 45,
            "ldl_cholesterol": 140,
            "triglycerides": 150,
            "family_history_diabetes": 1,
            "hypertension_history": 0,
            "cardiovascular_history": 0,
            "gender": "Male",
            "ethnicity": "White",
            "education_level": "Graduate",
            "income_level": "Middle",
            "smoking_status": "Never",
            "employment_status": "Employed"
        }
    })

class DiabetesPrediction(BaseModel):
    """Prediction output"""
    risk_score: float = Field(..., description="Risk score 0-1")
    risk_level: str = Field(..., description="Low/Medium/High Risk")
    recommendation: str = Field(..., description="Health recommendation")
    confidence_percentage: float = Field(..., description="Model confidence %")

# ==================== HELPER FUNCTIONS ====================

def preprocess_input(patient_data: DiabetesInput) -> np.ndarray:
    """Preprocess input to match training format"""
    df = pd.DataFrame([patient_data.dict()])
    
    numerical_cols = numerical_features
    categorical_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 
                       'smoking_status', 'employment_status']
    
    # Encode categorical
    categorical_data = df[categorical_cols]
    categorical_encoded = encoder.transform(categorical_data)
    categorical_feature_names = encoder.get_feature_names_out(categorical_cols)
    categorical_df = pd.DataFrame(categorical_encoded, columns=categorical_feature_names)
    
    # Get numerical
    numerical_df = df[numerical_cols]
    
    # Combine
    X = pd.concat([numerical_df.reset_index(drop=True), 
                   categorical_df.reset_index(drop=True)], axis=1)
    
    # Scale
    X[numerical_cols] = scaler.transform(X[numerical_cols])
    
    return X.values

def get_risk_level(risk_score: float) -> str:
    """Classify risk"""
    if risk_score < 0.4:
        return "Low Risk ✅"
    elif risk_score < 0.7:
        return "Medium Risk ⚠️"
    else:
        return "High Risk 🚨"

def get_recommendation(risk_score: float) -> str:
    """Get health recommendation"""
    if risk_score < 0.4:
        return "Maintain current healthy lifestyle. Continue regular exercise and balanced diet."
    elif risk_score < 0.7:
        return "Increase physical activity. Monitor diet and weight. Schedule regular checkups."
    else:
        return "Consult healthcare provider immediately. Implement lifestyle changes."

# ==================== ENDPOINTS ====================

@app.get("/", tags=["Root"])
def welcome():
    """Welcome endpoint"""
    return {
        "message": "🏥 Welcome to Diabetes Prediction API!",
        "version": "1.0.0",
        "model": "Gradient Boosting",
        "accuracy": f"{model_metadata['accuracy']*100:.2f}%",
        "documentation": "Visit /docs for interactive documentation"
    }

@app.get("/health", tags=["System"])
def health_check():
    """Health check"""
    return {"status": "healthy ✅", "model": model_metadata['model_name']}

@app.get("/model-info", tags=["Info"])
def get_model_info():
    """Model information"""
    return {
        "model": model_metadata['model_name'],
        "accuracy": round(model_metadata['accuracy'], 4),
        "f1_score": round(model_metadata['f1_score'], 4),
        "roc_auc": round(model_metadata['roc_auc'], 4),
        "training_samples": model_metadata['num_training_samples'],
        "features": model_metadata['num_features']
    }

@app.get("/example", tags=["Info"])
def get_example():
    """Example input"""
    return {
        "example_input": {
            "age": 55,
            "alcohol_consumption_per_week": 2,
            "physical_activity_minutes_per_week": 150,
            "diet_score": 7.5,
            "sleep_hours_per_day": 7.5,
            "screen_time_hours_per_day": 4.0,
            "bmi": 28.5,
            "waist_to_hip_ratio": 0.9,
            "systolic_bp": 130,
            "diastolic_bp": 85,
            "heart_rate": 75,
            "cholesterol_total": 220,
            "hdl_cholesterol": 45,
            "ldl_cholesterol": 140,
            "triglycerides": 150,
            "family_history_diabetes": 1,
            "hypertension_history": 0,
            "cardiovascular_history": 0,
            "gender": "Male",
            "ethnicity": "White",
            "education_level": "Graduate",
            "income_level": "Middle",
            "smoking_status": "Never",
            "employment_status": "Employed"
        }
    }

@app.post("/predict", response_model=DiabetesPrediction, tags=["Predictions"])
def predict_diabetes(patient: DiabetesInput):
    """
    Predict diabetes risk for a single patient
    
    Returns:
    - risk_score: 0-1 (0=healthy, 1=high risk)
    - risk_level: Low/Medium/High Risk
    - recommendation: Health advice
    - confidence_percentage: Confidence %
    """
    try:
        X = preprocess_input(patient)
        risk_score = model.predict_proba(X)[0][1]
        
        return DiabetesPrediction(
            risk_score=round(float(risk_score), 4),
            risk_level=get_risk_level(risk_score),
            recommendation=get_recommendation(risk_score),
            confidence_percentage=round(float(risk_score) * 100, 2)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error: {str(e)}"
        )

# ==================== RUN ====================

if __name__ == "__main__":
    # Use app:app format for proper reload support
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False  # Set to False for production
    )