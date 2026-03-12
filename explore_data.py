"""
Explore and understand the Kaggle diabetes dataset
Run this FIRST to understand what we're working with
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("LOADING DATA")
print("="*80)

# Load dataset
df = pd.read_csv('train.csv')

print(f"\n✅ Dataset loaded successfully!")
print(f"Shape: {df.shape}")  # (rows, columns)

# ==================== BASIC INFO ====================

print("\n" + "="*80)
print("DATASET OVERVIEW")
print("="*80)

print(f"\nTotal rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"\nColumn names and types:")
print(df.dtypes)

# ==================== FIRST LOOK ====================

print("\n" + "="*80)
print("FIRST 5 ROWS")
print("="*80)
print(df.head())

# ==================== MISSING VALUES ====================

print("\n" + "="*80)
print("MISSING VALUES")
print("="*80)

missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

missing_df = pd.DataFrame({
    'Column': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_percent.values
})

missing_df = missing_df[missing_df['Missing_Count'] > 0]

if len(missing_df) == 0:
    print("✅ No missing values!")
else:
    print(missing_df)

# ==================== STATISTICAL SUMMARY ====================

print("\n" + "="*80)
print("STATISTICAL SUMMARY (Numerical Features)")
print("="*80)

print(df.describe())

# ==================== TARGET DISTRIBUTION ====================

print("\n" + "="*80)
print("TARGET VARIABLE DISTRIBUTION")
print("="*80)

target_counts = df['diagnosed_diabetes'].value_counts()
target_percent = df['diagnosed_diabetes'].value_counts(normalize=True) * 100

print(f"\nDiagnosed Diabetes = 0 (No Diabetes): {target_counts[0]} ({target_percent[0]:.2f}%)")
print(f"Diagnosed Diabetes = 1 (Diabetes): {target_counts[1]} ({target_percent[1]:.2f}%)")

# ==================== DATA TYPES ====================

print("\n" + "="*80)
print("FEATURE TYPES")
print("="*80)

numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Remove id and target from lists
if 'id' in numerical_features:
    numerical_features.remove('id')
if 'diagnosed_diabetes' in numerical_features:
    numerical_features.remove('diagnosed_diabetes')

print(f"\nNumerical Features ({len(numerical_features)}):")
for i, col in enumerate(numerical_features, 1):
    print(f"  {i}. {col}")

print(f"\nCategorical Features ({len(categorical_features)}):")
for i, col in enumerate(categorical_features, 1):
    print(f"  {i}. {col}")

# ==================== CATEGORICAL UNIQUE VALUES ====================

print("\n" + "="*80)
print("CATEGORICAL FEATURES - UNIQUE VALUES")
print("="*80)

for col in categorical_features:
    print(f"\n{col}:")
    print(f"  Unique values: {df[col].nunique()}")
    print(f"  Values: {df[col].unique()}")

# ==================== CORRELATIONS ====================

print("\n" + "="*80)
print("CORRELATION WITH TARGET (Top 10)")
print("="*80)

# Calculate correlation for numerical features only
correlation = df[numerical_features + ['diagnosed_diabetes']].corr()['diagnosed_diabetes'].sort_values(ascending=False)

print("\n", correlation.head(11))

# ==================== OUTLIERS ====================

print("\n" + "="*80)
print("POTENTIAL OUTLIERS (IQR Method)")
print("="*80)

for col in numerical_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    if len(outliers) > 0:
        print(f"\n{col}: {len(outliers)} outliers detected")
        print(f"  Range: {lower_bound:.2f} to {upper_bound:.2f}")

print("\n" + "="*80)
print("✅ DATA EXPLORATION COMPLETE!")
print("="*80)