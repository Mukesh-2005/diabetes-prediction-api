"""
Train machine learning models on preprocessed Kaggle diabetes data
Optimized for 120k training samples with 42 features
Tests multiple models and selects the best one
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("LOADING PREPROCESSED DATA")
print("="*80)

# Load preprocessed data
X_train = joblib.load('X_train.pkl')
X_test = joblib.load('X_test.pkl')
y_train = joblib.load('y_train.pkl')
y_test = joblib.load('y_test.pkl')
feature_names = joblib.load('feature_names.pkl')

print(f"\n✅ Loaded training set: {X_train.shape}")
print(f"✅ Loaded test set: {X_test.shape}")
print(f"✅ Training labels: {y_train.shape}")
print(f"✅ Test labels: {y_test.shape}")
print(f"✅ Features: {len(feature_names)} features")

# ==================== DEFINE MODELS ====================

print("\n" + "="*80)
print("TRAINING MULTIPLE MODELS (This will take 2-5 minutes)")
print("="*80)

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, 
        random_state=42,
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        n_jobs=-1,
        max_depth=15
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, 
        random_state=42,
        max_depth=5,
        learning_rate=0.1
    )
}

results = {}

for model_name, model in models.items():
    print(f"\n[Training {model_name}...]")
    
    # Train
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  ✅ {model_name} trained!")
    print(f"     Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"     Precision: {precision:.4f}")
    print(f"     Recall:    {recall:.4f}")
    print(f"     F1-Score:  {f1:.4f}")
    print(f"     ROC-AUC:   {roc_auc:.4f}")

# ==================== SELECT BEST MODEL ====================

print("\n" + "="*80)
print("MODEL COMPARISON")
print("="*80)

comparison_data = []
for model_name in results.keys():
    comparison_data.append({
        'Model': model_name,
        'Accuracy': f"{results[model_name]['accuracy']:.4f}",
        'Precision': f"{results[model_name]['precision']:.4f}",
        'Recall': f"{results[model_name]['recall']:.4f}",
        'F1-Score': f"{results[model_name]['f1']:.4f}",
        'ROC-AUC': f"{results[model_name]['roc_auc']:.4f}"
    })

comparison_df = pd.DataFrame(comparison_data)
print("\n", comparison_df.to_string(index=False))

# Select best model by F1-score (best for imbalanced data)
best_model_name = max(results, key=lambda x: results[x]['f1'])
best_model = results[best_model_name]['model']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
print(f"   Accuracy: {results[best_model_name]['accuracy']*100:.2f}%")
print(f"   ROC-AUC:  {results[best_model_name]['roc_auc']:.4f}")

# ==================== DETAILED EVALUATION ====================

print("\n" + "="*80)
print(f"DETAILED EVALUATION - {best_model_name}")
print("="*80)

y_pred = results[best_model_name]['predictions']

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, 
    target_names=['No Diabetes', 'Diabetes']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n   True Negatives (TN):  {cm[0,0]:,} - Correctly predicted no diabetes")
print(f"   False Positives (FP): {cm[0,1]:,} - Wrongly predicted diabetes")
print(f"   False Negatives (FN): {cm[1,0]:,} - Missed diabetes cases ⚠️")
print(f"   True Positives (TP):  {cm[1,1]:,} - Correctly predicted diabetes")

# ==================== FEATURE IMPORTANCE ====================

if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*80)
    print("TOP 15 IMPORTANT FEATURES")
    print("="*80)
    
    importances = best_model.feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    print("\n", feature_importance_df.head(15).to_string(index=False))

# ==================== SAVE FINAL MODEL ====================

print("\n" + "="*80)
print("SAVING FINAL MODEL")
print("="*80)

joblib.dump(best_model, 'diabetes_model.pkl')
print(f"✅ Best model ({best_model_name}) saved to diabetes_model.pkl")

# Save model metadata
model_metadata = {
    'model_name': best_model_name,
    'accuracy': results[best_model_name]['accuracy'],
    'precision': results[best_model_name]['precision'],
    'recall': results[best_model_name]['recall'],
    'f1_score': results[best_model_name]['f1'],
    'roc_auc': results[best_model_name]['roc_auc'],
    'num_features': X_train.shape[1],
    'num_training_samples': X_train.shape[0],
    'num_test_samples': X_test.shape[0]
}

joblib.dump(model_metadata, 'model_metadata.pkl')
print("✅ Model metadata saved to model_metadata.pkl")

print("\n" + "="*80)
print("✅ MODEL TRAINING COMPLETE!")
print("="*80)

print("\nFiles created:")
print("  - diabetes_model.pkl (trained model) ⭐")
print("  - model_metadata.pkl (performance metrics)")

print("\n🎉 Ready for API development!")