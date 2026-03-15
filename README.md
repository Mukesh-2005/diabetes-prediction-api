# 🏥 Diabetes Prediction API

**Live Demo:** [diabetes-prediction-api-production-1e6c.up.railway.app](diabetes-prediction-api-production-1e6c.up.railway.app)  
**API Docs:** [diabetes-prediction-api-production-1e6c.up.railway.app/docs](diabetes-prediction-api-production-1e6c.up.railway.app/docs)

---

## 🎯 Overview

A production-ready FastAPI application that predicts diabetes risk using a **Gradient Boosting machine learning model** trained on **700,000 medical records** from the Kaggle diabetes prediction competition.

## ✨ Features

- ✅ **Single & Batch Predictions** - Predict for one or multiple patients
- ✅ **42 Medical Features** - Comprehensive health assessment
- ✅ **66% Accuracy** - High-quality predictions
- ✅ **Auto Documentation** - Interactive Swagger UI & ReDoc
- ✅ **Production Ready** - Fully containerized & deployed
- ✅ **Fast Predictions** - <100ms response time

## 📊 Model Performance
```
Algorithm:      Gradient Boosting Classifier
Accuracy:       66.11%
Precision:      67.47%
Recall:         88.08% (catches most diabetes cases!)
F1-Score:       0.7641
ROC-AUC:        0.6771
Training Data:  700,000 records
Features:       42 (18 numerical + 24 categorical)
```

## 🚀 Quick Start

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run API
python main.py

# Visit Swagger UI
# http://localhost:8000/docs
```

## 📡 API Endpoints

### 1. Health Check
```
GET /health
```
Returns API status

### 2. Model Info
```
GET /model-info
```
Returns model performance metrics

### 3. Example Data
```
GET /example
```
Returns example input for testing

### 4. Single Prediction ⭐
```
POST /predict

Input: 24 medical parameters
Output: Risk score, risk level, recommendation
```

### 5. Batch Prediction
```
POST /predict-batch

Input: List of patient data
Output: List of predictions
```

## 📋 Input Parameters

**Numerical Parameters (18):**
- age, BMI, blood pressure, cholesterol levels
- heart rate, triglycerides, diet score
- physical activity, sleep, screen time
- Medical history indicators

**Categorical Parameters (6):**
- Gender, Ethnicity, Education level
- Income level, Smoking status, Employment status

## 🧪 Example Request
```json
{
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
```

## 📊 Risk Levels

| Score | Level | Recommendation |
|-------|-------|----------------|
| 0.0-0.4 | Low Risk ✅ | Maintain healthy lifestyle |
| 0.4-0.7 | Medium Risk ⚠️ | Increase exercise, monitor diet |
| 0.7-1.0 | High Risk 🚨 | Consult healthcare provider |

## 🏗️ Architecture
```
User Input (24 parameters)
    ↓
Data Preprocessing (Encoding + Scaling)
    ↓
Gradient Boosting Model
    ↓
Risk Prediction (0-1 score)
    ↓
Classification (Low/Medium/High)
    ↓
Health Recommendation
```

## 🐳 Docker
```bash
# Build image
docker build -t diabetes-api .

# Run container
docker run -p 8000:8000 diabetes-api
```

## 📚 Technologies

- **Framework:** FastAPI
- **ML Model:** Gradient Boosting (scikit-learn)
- **Data Processing:** pandas, numpy
- **Deployment:** Docker, Railway.app
- **Documentation:** Swagger UI, ReDoc

## 🎓 Dataset

- **Source:** Kaggle Diabetes Prediction Competition
- **Records:** 700,000 patients
- **Features:** 26 medical parameters
- **Target:** Diabetes diagnosis (binary classification)

## 📈 Performance Metrics

### Confusion Matrix (Test Set)
```
                Predicted No  Predicted Yes
Actual No         3,361        7,940
Actual Yes        2,228       16,471
```

### Key Insights
- **Recall (88%):** Catches most diabetes cases (only misses 12%)
- **Precision (67%):** 67% of positive predictions are correct
- **Medical Focus:** Optimized to minimize missed cases

## 🔗 Important Links

- **GitHub:** [https://github.com/YOUR_USERNAME/diabetes-prediction-api](https://github.com/YOUR_USERNAME/diabetes-prediction-api)
- **Kaggle Dataset:** [https://www.kaggle.com/competitions](https://www.kaggle.com/competitions)
- **FastAPI Docs:** [https://fastapi.tiangolo.com/](https://fastapi.tiangolo.com/)

## 👨‍💻 Author

**Your Name**
- GitHub: [@YOUR_USERNAME](https://github.com/YOUR_USERNAME)
- Portfolio: [your-portfolio.com](https://your-portfolio.com)

## 📝 License

MIT License - feel free to use this project!

## 🙌 Acknowledgments

- Kaggle for the diabetes dataset
- FastAPI community for the framework
- scikit-learn for ML algorithms

---

**⭐ If you found this helpful, please star the repository!**

**Ready to use? Visit the [live demo](https://your-live-url.railway.app/docs)!** 🚀
