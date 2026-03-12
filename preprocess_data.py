"""
Optimized data preprocessing for 700k row Kaggle diabetes dataset
Handles: Imbalanced classes, Outliers, Categorical encoding, Scaling, Sampling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import joblib

print("="*80)
print("OPTIMIZED DATA PREPROCESSING PIPELINE")
print("="*80)

# ==================== STEP 1: LOAD & SAMPLE ====================

print("\n[1/7] Loading and sampling data...")
df = pd.read_csv('train.csv')
print(f"Original: {df.shape[0]} rows")

# Sample for faster processing (keeps class balance)
# Perform stratified sampling manually
sample_size = 150000
stratified_sample = df.groupby('diagnosed_diabetes', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), int(sample_size * len(x) / len(df))), random_state=42)
).reset_index(drop=True)
df = stratified_sample
print(f"Sampled: {df.shape[0]} rows ✅")

# ==================== STEP 2: SEPARATE TARGET ====================

print("\n[2/7] Separating features and target...")
ids = df['id'].copy()
X = df.drop(['id', 'diagnosed_diabetes'], axis=1)
y = df['diagnosed_diabetes'].copy()

print(f"Features shape: {X.shape}")
print(f"Target distribution:")
print(f"  No Diabetes: {(y==0).sum()} ({(y==0).mean()*100:.2f}%)")
print(f"  Diabetes: {(y==1).sum()} ({(y==1).mean()*100:.2f}%)")

# ==================== STEP 3: IDENTIFY FEATURES ====================

print("\n[3/7] Identifying numerical and categorical features...")
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

# ==================== STEP 4: CAP OUTLIERS ====================

print("\n[4/7] Handling outliers with IQR capping...")
outlier_count = 0

for col in numerical_features:
    Q1 = X[col].quantile(0.25)
    Q3 = X[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    before = X[col].copy()
    X[col] = X[col].clip(lower_bound, upper_bound)
    
    changed = (before != X[col]).sum()
    if changed > 0:
        outlier_count += changed

print(f"✅ Total outliers capped: {outlier_count}")

# ==================== STEP 5: ONE-HOT ENCODE CATEGORIES ====================

print("\n[5/7] One-Hot encoding categorical features...")
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
cat_encoded = encoder.fit_transform(X[categorical_features])

# Create DataFrame with new feature names
cat_feature_names = encoder.get_feature_names_out(categorical_features)
cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=X.index)

# Drop original categorical columns and add encoded ones
X = X.drop(categorical_features, axis=1)
X = pd.concat([X.reset_index(drop=True), cat_df.reset_index(drop=True)], axis=1)

print(f"✅ Categorical features encoded")
print(f"   New feature count: {X.shape[1]}")

# ==================== STEP 6: SCALE NUMERICAL FEATURES ====================

print("\n[6/7] Scaling numerical features with StandardScaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[numerical_features])
X[numerical_features] = X_scaled

print(f"✅ Features scaled (mean=0, std=1)")

# Save scaler for API
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(encoder, 'encoder.pkl')
print(f"✅ Scaler and encoder saved")

# ==================== STEP 7: STRATIFIED TRAIN-TEST SPLIT ====================

print("\n[7/7] Stratified train-test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y  # ← MAINTAINS CLASS BALANCE!
)

print(f"Training set: {X_train.shape[0]} samples ({X_train.shape[1]} features)")
print(f"Test set: {X_test.shape[0]} samples")

print(f"\nTraining set class distribution:")
print(f"  No Diabetes: {(y_train==0).sum()} ({(y_train==0).mean()*100:.2f}%)")
print(f"  Diabetes: {(y_train==1).sum()} ({(y_train==1).mean()*100:.2f}%)")

print(f"\nTest set class distribution:")
print(f"  No Diabetes: {(y_test==0).sum()} ({(y_test==0).mean()*100:.2f}%)")
print(f"  Diabetes: {(y_test==1).sum()} ({(y_test==1).mean()*100:.2f}%)")

# ==================== SAVE PREPROCESSED DATA ====================

print("\n" + "="*80)
print("SAVING PREPROCESSED DATA")
print("="*80)

joblib.dump(X_train, 'X_train.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_train, 'y_train.pkl')
joblib.dump(y_test, 'y_test.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(numerical_features, 'numerical_features.pkl')
joblib.dump(categorical_features, 'categorical_features.pkl')

print("✅ X_train.pkl saved")
print("✅ X_test.pkl saved")
print("✅ y_train.pkl saved")
print("✅ y_test.pkl saved")
print("✅ feature_names.pkl saved")
print("✅ scaler.pkl saved")
print("✅ encoder.pkl saved")

print("\n" + "="*80)
print("✅ PREPROCESSING COMPLETE!")
print("="*80)

print("\nFiles ready for model training:")
print("  - X_train.pkl")
print("  - X_test.pkl")
print("  - y_train.pkl")
print("  - y_test.pkl")
print("  - feature_names.pkl")
print("  - scaler.pkl (for API)")
print("  - encoder.pkl (for API)")