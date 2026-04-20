import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import joblib
import os
from utils.pipeline import create_full_pipeline

print("=== Starting Improved Customer Churn Training with Pipeline ===\n")

# 1. Load Data
df = pd.read_csv('data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# 2. Basic Cleaning
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df.drop('customerID', axis=1, inplace=True)
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {df['Churn'].mean():.2%}")

# 3. Create Full Pipeline
preprocessor = create_full_pipeline()

# 4. Prepare features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Fit Preprocessor
print("Fitting preprocessing pipeline...")
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# 6. Train XGBoost Model
print("Training XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=400,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='auc',
    scale_pos_weight=len(y_train[y_train==0]) / len(y_train[y_train==1])  # Handle imbalance
)

model.fit(X_train_processed, y_train)

# 7. Evaluate
y_pred = model.predict(X_test_processed)
y_prob = model.predict_proba(X_test_processed)[:, 1]

print("\n=== Model Performance ===")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 8. Save Everything (Pipeline + Model)
os.makedirs('models', exist_ok=True)

joblib.dump(preprocessor, 'models/preprocessor.pkl')
joblib.dump(model, 'models/churn_model.pkl')
joblib.dump(X.columns.tolist(), 'models/feature_names.pkl')

print("\n✅ Training completed successfully!")
print("Saved files:")
print("   • models/preprocessor.pkl")
print("   • models/churn_model.pkl")
print("   • models/feature_names.pkl")